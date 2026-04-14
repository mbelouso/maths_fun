#!/usr/bin/env python3
"""
Mandelbrot Set Explorer
=======================
Interactive PyQt5 application for exploring the Mandelbrot set with
GPU (Numba CUDA) and CPU (Numba @njit parallel) backends.

Features:
  - Smooth iteration colouring with 7 colormaps
  - Left-drag pan, scroll-wheel zoom, debounced re-render
  - Toggleable axes (border / origin)
  - High-resolution export (4K / 8K / 16K) re-computed at full resolution
  - Multi-GPU support, float32/float64 precision
  - CPU multicore via Numba prange

Usage
-----
    conda run -n maths_fun python3 mandelbrot_explorer.py

Author: Matthew Belousoff, Claude 2026
"""

import sys
import os
import time
import threading
import datetime
import math

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.cm as mcm

from numba import njit, prange, set_num_threads

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QDoubleSpinBox, QSpinBox, QGroupBox,
    QPushButton, QComboBox, QCheckBox,
    QScrollArea, QSizePolicy, QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject


# ── CUDA availability probe ────────────────────────────────────────────────
CUDA_AVAILABLE = False
_CUDA_GPUS = []      # list of (id, name) tuples
_CUDA_ERR  = ""

try:
    from numba import cuda
    if cuda.is_available():
        CUDA_AVAILABLE = True
        for i in range(len(cuda.gpus)):
            try:
                dev = cuda.gpus[i]
                _CUDA_GPUS.append((i, dev.name.decode()
                                   if isinstance(dev.name, bytes)
                                   else str(dev.name)))
            except Exception:
                _CUDA_GPUS.append((i, f"GPU {i}"))
    else:
        _CUDA_ERR = "Numba installed but no CUDA-capable GPU detected"
except ImportError:
    _CUDA_ERR = "Numba CUDA not available (pip install numba)"
except Exception as e:
    _CUDA_ERR = f"CUDA init error: {e}"


# ── Palette ──────────────────────────────────────────────────────────────────
DARK_BG = "#07070f"
CTRL_BG = "#12122a"
CTRL_FG = "#ccccee"
ACC_C   = "#7788ff"
GOLD    = "#FFD700"

_QSS = """
QMainWindow, QWidget          { background:#12122a; color:#ccccee; font-size:9pt; }
QGroupBox                     { border:1px solid #334; border-radius:3px; margin-top:10px;
                                color:#FFD700; font-size:8pt; font-weight:bold; }
QGroupBox::title              { subcontrol-origin:margin; left:8px; padding:0 4px; }
QScrollArea                   { border:none; background:#12122a; }
QLabel                        { color:#ccccee; }
QSlider::groove:horizontal    { background:#334; height:4px; border-radius:2px; }
QSlider::handle:horizontal    { background:#7788ff; width:10px; height:10px;
                                border-radius:5px; margin:-3px 0; }
QSlider::sub-page:horizontal  { background:#7788ff; border-radius:2px; }
QDoubleSpinBox, QSpinBox,
QComboBox                     { background:#1a1a2e; color:#ccccee;
                                border:1px solid #334; padding:2px 4px; }
QDoubleSpinBox::up-button,
QDoubleSpinBox::down-button,
QSpinBox::up-button,
QSpinBox::down-button         { width:14px; }
QPushButton                   { background:#1a1a3e; color:#ccccee;
                                border:1px solid #445; padding:5px 8px;
                                border-radius:3px; }
QPushButton:hover             { background:#2a2a5e; }
QPushButton:pressed           { background:#0a0a2e; }
QCheckBox                     { color:#ccccee; font-size:9pt; }
QProgressBar                  { background:#1a1a2e; border:1px solid #334;
                                border-radius:3px; text-align:center;
                                color:#ccccee; font-size:8pt; }
QProgressBar::chunk           { background:#7788ff; border-radius:2px; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA COMPUTE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

if CUDA_AVAILABLE:

    @cuda.jit
    def _mandelbrot_kernel(re_min, re_max, im_min, im_max,
                           width, height, max_iter, out):
        """One thread per pixel.  Computes smooth iteration count."""
        idx = cuda.grid(1)
        if idx >= width * height:
            return

        row = idx // width
        col = idx % width

        # Map pixel to complex plane
        cr = re_min + (re_max - re_min) * col / (width - 1)
        ci = im_min + (im_max - im_min) * row / (height - 1)

        zr = 0.0
        zi = 0.0
        n = 0
        BAILOUT = 256.0  # large bailout for smooth colouring

        while n < max_iter:
            zr2 = zr * zr
            zi2 = zi * zi
            if zr2 + zi2 > BAILOUT:
                break
            zi = 2.0 * zr * zi + ci
            zr = zr2 - zi2 + cr
            n += 1

        if n < max_iter:
            # Smooth iteration count
            log_zn = math.log(zr * zr + zi * zi) * 0.5
            nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
            out[idx] = n + 1.0 - nu
        else:
            out[idx] = -1.0   # inside the set


    def _gpu_worker(gpu_id, re_min, re_max, im_min, im_max,
                    width, height, max_iter, row_start, row_end, use_fp32):
        """Run the Mandelbrot kernel on a single GPU for a block of rows."""
        cuda.select_device(gpu_id)

        fp = np.float32 if use_fp32 else np.float64
        local_height = row_end - row_start
        N_local = width * local_height

        # Compute the imaginary range for this block of rows
        full_im_span = im_max - im_min
        local_im_min = im_min + full_im_span * row_start / height
        local_im_max = im_min + full_im_span * row_end / height

        d_out = cuda.device_array(N_local, dtype=fp)

        threads_per_block = 256
        blocks = (N_local + threads_per_block - 1) // threads_per_block

        _mandelbrot_kernel[blocks, threads_per_block](
            fp(re_min), fp(re_max), fp(local_im_min), fp(local_im_max),
            width, local_height, max_iter, d_out)

        cuda.synchronize()
        return d_out.copy_to_host().astype(np.float64)


    def compute_mandelbrot_cuda(re_min, re_max, im_min, im_max,
                                width, height, max_iter,
                                gpu_ids=None, use_fp32=False,
                                progress_cb=None):
        """Compute Mandelbrot on GPU(s).  Returns (height, width) float64 array."""
        if gpu_ids is None:
            gpu_ids = [gid for gid, _ in _CUDA_GPUS]
        n_gpus = len(gpu_ids)

        row_splits = np.array_split(np.arange(height), n_gpus)

        if progress_cb:
            progress_cb(5)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        result = np.empty(height * width, dtype=np.float64)
        futures = {}

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            for gi, gpu_id in enumerate(gpu_ids):
                rows = row_splits[gi]
                if len(rows) == 0:
                    continue
                r_start = int(rows[0])
                r_end   = int(rows[-1]) + 1
                f = pool.submit(_gpu_worker, gpu_id,
                                re_min, re_max, im_min, im_max,
                                width, height, max_iter,
                                r_start, r_end, use_fp32)
                futures[f] = (r_start, r_end)

            done_count = 0
            for f in as_completed(futures):
                r_start, r_end = futures[f]
                chunk = f.result()
                result[r_start * width: r_end * width] = chunk
                done_count += 1
                if progress_cb:
                    progress_cb(5 + int(90 * done_count / len(futures)))

        if progress_cb:
            progress_cb(95)

        return result.reshape(height, width)


# ═══════════════════════════════════════════════════════════════════════════════
# CPU COMPUTE ENGINE  (Numba @njit parallel)
# ═══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True, cache=True)
def _mandelbrot_cpu(re_min, re_max, im_min, im_max,
                    width, height, max_iter):
    """Compute Mandelbrot with smooth colouring using Numba prange over rows."""
    out = np.empty((height, width), dtype=np.float64)
    BAILOUT = 256.0

    for row in prange(height):
        ci = im_min + (im_max - im_min) * row / (height - 1)
        for col in range(width):
            cr = re_min + (re_max - re_min) * col / (width - 1)

            zr = 0.0
            zi = 0.0
            n = 0

            while n < max_iter:
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > BAILOUT:
                    break
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
                n += 1

            if n < max_iter:
                log_zn = math.log(zr * zr + zi * zi) * 0.5
                nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                out[row, col] = n + 1.0 - nu
            else:
                out[row, col] = -1.0

    return out


def compute_mandelbrot_cpu(re_min, re_max, im_min, im_max,
                           width, height, max_iter,
                           n_workers=None, progress_cb=None):
    """Compute Mandelbrot on CPU.  Returns (height, width) float64 array."""
    if n_workers is not None and n_workers >= 1:
        set_num_threads(n_workers)

    if progress_cb:
        progress_cb(5)

    result = _mandelbrot_cpu(re_min, re_max, im_min, im_max,
                             width, height, max_iter)

    if progress_cb:
        progress_cb(95)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# COLOURING
# ═══════════════════════════════════════════════════════════════════════════════

def apply_colourmap(grid, cmap_name, log_scale=False):
    """Map iteration-count grid to RGBA uint8 (H, W, 4).

    Pixels with value -1 (inside the set) are coloured black.
    """
    mask_inside = grid < 0
    data = grid.copy()
    data[mask_inside] = 0.0

    if log_scale:
        data = np.log1p(data)

    vmin = data[~mask_inside].min() if np.any(~mask_inside) else 0.0
    vmax = data[~mask_inside].max() if np.any(~mask_inside) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    normed = (data - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0.0, 1.0)

    cmap = mcm.get_cmap(cmap_name)
    rgba = (cmap(normed) * 255).astype(np.uint8)

    # Interior → black
    rgba[mask_inside] = [0, 0, 0, 255]

    return rgba


# ═══════════════════════════════════════════════════════════════════════════════
# INTERESTING LOCATIONS (presets)
# ═══════════════════════════════════════════════════════════════════════════════

PRESETS = [
    ("Full Set",       -0.5,               0.0,              1.0),
    ("Seahorse",       -0.745,             0.186,            50.0),
    ("Elephant",        0.2821,            0.0075,          200.0),
    ("Mini Brot",      -1.7685,            0.0014,         1000.0),
    ("Spiral",         -0.0452,            0.9868,          100.0),
    ("Lightning",      -1.315180982,       0.073481725,    5000.0),
]

EXPORT_RESOLUTIONS = [
    ("4K   (3840 x 2160)",   3840,  2160),
    ("8K   (7680 x 4320)",   7680,  4320),
    ("16K  (15360 x 8640)", 15360,  8640),
]


# ═══════════════════════════════════════════════════════════════════════════════
# WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class _ParamSlider(QWidget):
    """Label + QSlider(0-1000) + QDoubleSpinBox, bidirectionally synced."""

    valueChanged = pyqtSignal(float)

    def __init__(self, label, vmin, vmax, vinit, decimals=2, suffix="",
                 parent=None):
        super().__init__(parent)
        self._vmin  = float(vmin)
        self._vmax  = float(vmax)
        self._block = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet("color:#aaaacc; font-size:8pt;")
        lay.addWidget(lbl)

        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(4)
        lay.addWidget(row_w)

        self._sl = QSlider(Qt.Horizontal)
        self._sl.setRange(0, 1000)
        self._sl.setValue(self._to_ticks(vinit))
        row_l.addWidget(self._sl, 3)

        self._sb = QDoubleSpinBox()
        self._sb.setDecimals(decimals)
        self._sb.setRange(self._vmin, self._vmax)
        self._sb.setValue(float(vinit))
        self._sb.setSuffix(suffix)
        self._sb.setFixedWidth(76)
        row_l.addWidget(self._sb)

        self._sl.valueChanged.connect(self._from_slider)
        self._sb.valueChanged.connect(self._from_spin)

    def _to_ticks(self, v):
        span = self._vmax - self._vmin
        return int(round(1000.0 * (float(v) - self._vmin) / span)) if span else 0

    def _from_ticks(self, t):
        return self._vmin + t * (self._vmax - self._vmin) / 1000.0

    def _from_slider(self, ticks):
        if self._block:
            return
        v = self._from_ticks(ticks)
        self._block = True
        self._sb.setValue(v)
        self._block = False
        self.valueChanged.emit(v)

    def _from_spin(self, v):
        if self._block:
            return
        self._block = True
        self._sl.setValue(self._to_ticks(v))
        self._block = False
        self.valueChanged.emit(v)

    def value(self):
        return self._sb.value()

    def set_value(self, v, silent=True):
        v = float(np.clip(v, self._vmin, self._vmax))
        self._block = True
        self._sl.setValue(self._to_ticks(v))
        self._sb.setValue(v)
        self._block = False
        if not silent:
            self.valueChanged.emit(v)


class _LogParamSlider(QWidget):
    """Like _ParamSlider but maps slider linearly in log10 space.

    Good for zoom values spanning many orders of magnitude.
    """

    valueChanged = pyqtSignal(float)

    def __init__(self, label, vmin, vmax, vinit, decimals=2, suffix="",
                 parent=None):
        super().__init__(parent)
        self._vmin_log = math.log10(float(vmin))
        self._vmax_log = math.log10(float(vmax))
        self._block = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet("color:#aaaacc; font-size:8pt;")
        lay.addWidget(lbl)

        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(4)
        lay.addWidget(row_w)

        self._sl = QSlider(Qt.Horizontal)
        self._sl.setRange(0, 1000)
        self._sl.setValue(self._to_ticks(vinit))
        row_l.addWidget(self._sl, 3)

        self._sb = QDoubleSpinBox()
        self._sb.setDecimals(decimals)
        self._sb.setRange(float(vmin), float(vmax))
        self._sb.setValue(float(vinit))
        self._sb.setSuffix(suffix)
        self._sb.setFixedWidth(96)
        row_l.addWidget(self._sb)

        self._sl.valueChanged.connect(self._from_slider)
        self._sb.valueChanged.connect(self._from_spin)

    def _to_ticks(self, v):
        log_v = math.log10(max(float(v), 10**self._vmin_log))
        span = self._vmax_log - self._vmin_log
        return int(round(1000.0 * (log_v - self._vmin_log) / span)) if span else 0

    def _from_ticks(self, t):
        log_v = self._vmin_log + t * (self._vmax_log - self._vmin_log) / 1000.0
        return 10.0 ** log_v

    def _from_slider(self, ticks):
        if self._block:
            return
        v = self._from_ticks(ticks)
        self._block = True
        self._sb.setValue(v)
        self._block = False
        self.valueChanged.emit(v)

    def _from_spin(self, v):
        if self._block:
            return
        self._block = True
        self._sl.setValue(self._to_ticks(v))
        self._block = False
        self.valueChanged.emit(v)

    def value(self):
        return self._sb.value()

    def set_value(self, v, silent=True):
        vmin = 10.0 ** self._vmin_log
        vmax = 10.0 ** self._vmax_log
        v = float(np.clip(v, vmin, vmax))
        self._block = True
        self._sl.setValue(self._to_ticks(v))
        self._sb.setValue(v)
        self._block = False
        if not silent:
            self.valueChanged.emit(v)


# ── Thread-safe signal bridge ────────────────────────────────────────────────

class _StatusBridge(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_RE_CENTER = -0.5
_DEFAULT_IM_CENTER =  0.0
_DEFAULT_ZOOM      =  1.0
_BASE_HALF_WIDTH   =  1.75   # at zoom=1: re in [-2.25, 1.25]
_BASE_HALF_HEIGHT  =  1.25   # at zoom=1: im in [-1.25, 1.25]


class MandelbrotExplorer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mandelbrot Explorer")
        self.resize(1440, 860)

        self._computing  = False
        self._exporting  = False
        self._generation = 0
        self._grid       = None
        self._cmap_name  = "inferno"
        self._log_scale  = False
        self._show_axes  = True
        self._axes_pos   = "Border"
        self._t0         = 0.0

        # Viewport state
        self._re_center = _DEFAULT_RE_CENTER
        self._im_center = _DEFAULT_IM_CENTER
        self._zoom      = _DEFAULT_ZOOM
        self._max_iter  = 256

        self._bridge = _StatusBridge()
        self._bridge.progress.connect(self._on_progress)
        self._bridge.finished.connect(self._on_compute_done)
        self._bridge.error.connect(self._on_compute_error)

        # Separate bridge for exports
        self._export_bridge = _StatusBridge()
        self._export_bridge.progress.connect(self._on_export_progress)
        self._export_bridge.finished.connect(self._on_export_done)
        self._export_bridge.error.connect(self._on_export_error)

        self._build_plot()
        self._build_controls()
        self._wire_central()

        # Debounce timer for re-render after pan/zoom
        self._rerender_timer = QTimer(self)
        self._rerender_timer.setSingleShot(True)
        self._rerender_timer.setInterval(300)
        self._rerender_timer.timeout.connect(self._on_rerender)

        # Auto-compute on startup
        QTimer.singleShot(100, self._start_compute)

    # ── Canvas ───────────────────────────────────────────────────────────────

    def _build_plot(self):
        self.fig = Figure(facecolor=DARK_BG)
        self._canvas = FigureCanvas(self.fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(DARK_BG)
        self.ax.set_xlabel("Re(c)", color=CTRL_FG, fontsize=9)
        self.ax.set_ylabel("Im(c)", color=CTRL_FG, fontsize=9)
        self.ax.set_title("Mandelbrot Set", color=CTRL_FG, fontsize=11, pad=10)
        self.ax.tick_params(colors="#556", labelsize=7)

        self._im = None

        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

        # Pan state
        self._panning   = False
        self._pan_start = None

    # ── Controls ─────────────────────────────────────────────────────────────

    def _build_controls(self):
        self._ctrl = QWidget()
        lay = QVBoxLayout(self._ctrl)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)

        # Title
        title = QLabel("MANDELBROT\nEXPLORER")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color:{GOLD}; font-size:11pt; font-weight:bold;")
        lay.addWidget(title)

        # ── Navigation ──────────────────────────────────────────────────
        grp = QGroupBox("Navigation")
        g_lay = QVBoxLayout(grp)

        self.sl_re = _ParamSlider("Re (center)", -2.5, 1.5, _DEFAULT_RE_CENTER, 10)
        self.sl_im = _ParamSlider("Im (center)", -2.0, 2.0, _DEFAULT_IM_CENTER, 10)
        self.sl_zoom = _LogParamSlider("Zoom", 0.5, 1e15, _DEFAULT_ZOOM, 2, "x")
        self.sl_re.valueChanged.connect(self._on_nav_change)
        self.sl_im.valueChanged.connect(self._on_nav_change)
        self.sl_zoom.valueChanged.connect(self._on_nav_change)
        for w in (self.sl_re, self.sl_im, self.sl_zoom):
            g_lay.addWidget(w)

        self._lbl_coord = QLabel("Move mouse over fractal for coordinates")
        self._lbl_coord.setWordWrap(True)
        self._lbl_coord.setStyleSheet("color:#888; font-size:8pt;")
        g_lay.addWidget(self._lbl_coord)

        self._btn_reset = QPushButton("Reset View")
        self._btn_reset.clicked.connect(self._reset_view)
        g_lay.addWidget(self._btn_reset)
        lay.addWidget(grp)

        # ── Computation ─────────────────────────────────────────────────
        grp = QGroupBox("Computation")
        g_lay = QVBoxLayout(grp)

        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Max iterations:"))
        self._sb_iter = QSpinBox()
        self._sb_iter.setRange(50, 100000)
        self._sb_iter.setValue(256)
        self._sb_iter.setFixedWidth(76)
        r_lay.addWidget(self._sb_iter)
        r_lay.addStretch()
        g_lay.addWidget(row)

        btn_row = QWidget()
        btn_lay = QGridLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(4)
        for i, n in enumerate([100, 256, 500, 1000, 5000]):
            b = QPushButton(str(n))
            b.clicked.connect(lambda _, nn=n: self._sb_iter.setValue(nn))
            btn_lay.addWidget(b, i // 3, i % 3)
        g_lay.addWidget(btn_row)
        lay.addWidget(grp)

        # ── Backend ─────────────────────────────────────────────────────
        grp = QGroupBox("Backend")
        g_lay = QVBoxLayout(grp)

        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Engine:"))
        self._cb_backend = QComboBox()
        self._cb_backend.addItem("CUDA (GPU)")
        self._cb_backend.addItem("CPU (Numba)")
        if not CUDA_AVAILABLE:
            self._cb_backend.setCurrentIndex(1)
            self._cb_backend.model().item(0).setEnabled(False)
        self._cb_backend.currentIndexChanged.connect(self._on_backend_change)
        r_lay.addWidget(self._cb_backend)
        g_lay.addWidget(row)

        # CUDA-specific controls
        self._cuda_controls = QWidget()
        cuda_lay = QVBoxLayout(self._cuda_controls)
        cuda_lay.setContentsMargins(0, 0, 0, 0)
        cuda_lay.setSpacing(4)

        self._gpu_chks = []
        if _CUDA_GPUS:
            for gid, gname in _CUDA_GPUS:
                chk = QCheckBox(f"GPU {gid}: {gname}")
                chk.setChecked(True)
                self._gpu_chks.append(chk)
                cuda_lay.addWidget(chk)
        else:
            lbl = QLabel(_CUDA_ERR if _CUDA_ERR else "No GPUs detected")
            lbl.setStyleSheet("color:#ff6666; font-size:8pt;")
            cuda_lay.addWidget(lbl)

        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Precision:"))
        self._cb_precision = QComboBox()
        self._cb_precision.addItems(["float32 (fast)", "float64"])
        r_lay.addWidget(self._cb_precision)
        cuda_lay.addWidget(row)

        g_lay.addWidget(self._cuda_controls)

        # CPU-specific controls
        self._cpu_controls = QWidget()
        cpu_lay = QVBoxLayout(self._cpu_controls)
        cpu_lay.setContentsMargins(0, 0, 0, 0)
        cpu_lay.setSpacing(4)

        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Workers:"))
        self._sb_workers = QSpinBox()
        n_cpus = os.cpu_count() or 4
        self._sb_workers.setRange(1, n_cpus)
        self._sb_workers.setValue(max(1, n_cpus - 1))
        self._sb_workers.setFixedWidth(56)
        r_lay.addWidget(self._sb_workers)
        cpu_lbl = QLabel(f"  / {n_cpus} cores")
        cpu_lbl.setStyleSheet("color:#888; font-size:8pt;")
        r_lay.addWidget(cpu_lbl)
        r_lay.addStretch()
        cpu_lay.addWidget(row)

        g_lay.addWidget(self._cpu_controls)
        lay.addWidget(grp)

        self._on_backend_change(self._cb_backend.currentIndex())

        # ── Display ─────────────────────────────────────────────────────
        grp = QGroupBox("Display")
        g_lay = QVBoxLayout(grp)

        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Colormap:"))
        self._cb_cmap = QComboBox()
        self._cb_cmap.addItems([
            "inferno", "viridis", "plasma", "magma", "hot", "turbo",
            "twilight",
        ])
        self._cb_cmap.currentTextChanged.connect(self._on_cmap_change)
        r_lay.addWidget(self._cb_cmap)
        g_lay.addWidget(row)

        self._chk_log = QCheckBox("Log scale  (reveals fine detail)")
        self._chk_log.stateChanged.connect(self._on_log_toggle)
        g_lay.addWidget(self._chk_log)

        self._chk_axes = QCheckBox("Show axes")
        self._chk_axes.setChecked(True)
        self._chk_axes.stateChanged.connect(self._on_axes_toggle)
        g_lay.addWidget(self._chk_axes)

        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Axes:"))
        self._cb_axes_pos = QComboBox()
        self._cb_axes_pos.addItems(["Border", "Origin"])
        self._cb_axes_pos.currentTextChanged.connect(self._on_axes_pos_change)
        r_lay.addWidget(self._cb_axes_pos)
        g_lay.addWidget(row)

        lay.addWidget(grp)

        # ── Actions ─────────────────────────────────────────────────────
        grp = QGroupBox("Actions")
        g_lay = QVBoxLayout(grp)

        self._btn_compute = QPushButton("Compute")
        self._btn_compute.setStyleSheet(
            f"color:{GOLD}; font-weight:bold; font-size:10pt; padding:8px;")
        self._btn_compute.clicked.connect(self._start_compute)
        g_lay.addWidget(self._btn_compute)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        g_lay.addWidget(self._progress)

        sep = QLabel("")
        sep.setFixedHeight(4)
        g_lay.addWidget(sep)

        self._btn_export = QPushButton("Export PNG")
        self._btn_export.clicked.connect(self._start_export)
        g_lay.addWidget(self._btn_export)

        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Resolution:"))
        self._cb_export_res = QComboBox()
        for label, w, h in EXPORT_RESOLUTIONS:
            self._cb_export_res.addItem(label, (w, h))
        r_lay.addWidget(self._cb_export_res)
        g_lay.addWidget(row)

        self._export_progress = QProgressBar()
        self._export_progress.setRange(0, 100)
        self._export_progress.setValue(0)
        g_lay.addWidget(self._export_progress)

        lay.addWidget(grp)

        # ── Presets ─────────────────────────────────────────────────────
        grp = QGroupBox("Presets")
        g_lay = QVBoxLayout(grp)

        btn_row = QWidget()
        btn_lay = QGridLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(4)
        for i, (name, re_c, im_c, zoom) in enumerate(PRESETS):
            b = QPushButton(name)
            b.clicked.connect(
                lambda _, r=re_c, im=im_c, z=zoom: self._apply_preset(r, im, z))
            btn_lay.addWidget(b, i // 2, i % 2)
        g_lay.addWidget(btn_row)
        lay.addWidget(grp)

        # ── Status ──────────────────────────────────────────────────────
        status_msg = "Ready.  Set parameters and click Compute."
        if CUDA_AVAILABLE:
            gpu_str = ", ".join(name for _, name in _CUDA_GPUS)
            status_msg += f"\nCUDA: {gpu_str}"
        else:
            status_msg += f"\n{_CUDA_ERR}"
        self._lbl_status = QLabel(status_msg)
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color:#888; font-size:8pt;")
        lay.addWidget(self._lbl_status)

        lay.addStretch()

    def _on_backend_change(self, idx):
        is_cuda = (idx == 0)
        self._cuda_controls.setVisible(is_cuda)
        self._cpu_controls.setVisible(not is_cuda)

    # ── Layout ───────────────────────────────────────────────────────────────

    def _wire_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)
        h.addWidget(self._canvas, 3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(310)
        scroll.setWidget(self._ctrl)
        h.addWidget(scroll)

    # ── Viewport helpers ────────────────────────────────────────────────────

    def _viewport(self):
        """Return (re_min, re_max, im_min, im_max) for current center/zoom."""
        hw = _BASE_HALF_WIDTH  / self._zoom
        hh = _BASE_HALF_HEIGHT / self._zoom
        return (self._re_center - hw, self._re_center + hw,
                self._im_center - hh, self._im_center + hh)

    def _canvas_size(self):
        """Return (width, height) in pixels for the current canvas size."""
        w = self._canvas.width()
        h = self._canvas.height()
        return max(w, 100), max(h, 100)

    def _sync_sliders_from_view(self):
        """Update navigation sliders to match current internal state."""
        self.sl_re.set_value(self._re_center, silent=True)
        self.sl_im.set_value(self._im_center, silent=True)
        self.sl_zoom.set_value(self._zoom, silent=True)

    # ── Compute ──────────────────────────────────────────────────────────────

    def _start_compute(self):
        if self._computing:
            return
        self._computing = True
        self._generation += 1
        gen = self._generation

        self._btn_compute.setEnabled(False)
        self._btn_compute.setText("Computing...")
        self._progress.setValue(0)

        # Read parameters
        self._re_center = self.sl_re.value()
        self._im_center = self.sl_im.value()
        self._zoom      = self.sl_zoom.value()
        self._max_iter  = self._sb_iter.value()

        re_min, re_max, im_min, im_max = self._viewport()
        width, height = self._canvas_size()

        is_cuda = self._cb_backend.currentIndex() == 0

        if is_cuda:
            gpu_ids  = [_CUDA_GPUS[i][0]
                        for i, chk in enumerate(self._gpu_chks)
                        if chk.isChecked()]
            use_fp32 = self._cb_precision.currentText().startswith("float32")
            self._compute_info = (f"{len(gpu_ids)} GPU(s), "
                                  f"{'fp32' if use_fp32 else 'fp64'}")
            if not gpu_ids:
                self._computing = False
                self._btn_compute.setEnabled(True)
                self._btn_compute.setText("Compute")
                self._lbl_status.setText("No GPUs selected.")
                return
        else:
            n_workers = self._sb_workers.value()
            self._compute_info = f"{n_workers} CPU workers"

        self._t0 = time.time()
        bridge = self._bridge

        def worker():
            try:
                def pcb(pct):
                    if gen == self._generation:
                        bridge.progress.emit(pct)

                if is_cuda:
                    result = compute_mandelbrot_cuda(
                        re_min, re_max, im_min, im_max,
                        width, height, self._max_iter,
                        gpu_ids=gpu_ids, use_fp32=use_fp32,
                        progress_cb=pcb)
                else:
                    result = compute_mandelbrot_cpu(
                        re_min, re_max, im_min, im_max,
                        width, height, self._max_iter,
                        n_workers=n_workers, progress_cb=pcb)

                if gen == self._generation:
                    bridge.finished.emit(result)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                bridge.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, pct):
        self._progress.setValue(pct)

    def _on_compute_done(self, result):
        elapsed = time.time() - self._t0
        self._computing = False
        self._btn_compute.setEnabled(True)
        self._btn_compute.setText("Compute")
        self._progress.setValue(100)

        self._grid = result
        h, w = result.shape

        self._lbl_status.setText(
            f"{w} x {h} px  |  {self._max_iter} iters  |  "
            f"{self._compute_info}  |  {elapsed:.2f}s"
        )

        self._update_display()

    def _on_compute_error(self, msg):
        self._computing = False
        self._btn_compute.setEnabled(True)
        self._btn_compute.setText("Compute")
        self._lbl_status.setText(f"Error: {msg}")

    # ── Display ──────────────────────────────────────────────────────────────

    def _update_display(self):
        if self._grid is None:
            return

        grid = self._grid.astype(np.float64)
        mask_inside = grid < 0
        disp = grid.copy()
        disp[mask_inside] = 0.0

        if self._log_scale:
            disp = np.log1p(disp)

        self.ax.clear()

        re_min, re_max, im_min, im_max = self._viewport()

        self._im = self.ax.imshow(
            disp,
            extent=[re_min, re_max, im_min, im_max],
            origin="lower",
            cmap=self._cmap_name,
            aspect="auto",
            interpolation="nearest",
        )
        # Black out interior pixels via set_clim (interior is 0 after mask)
        outside = disp[~mask_inside]
        if outside.size > 0:
            self._im.set_clim(outside.min(), outside.max())

        self._apply_axes_style()

        self.fig.tight_layout()
        self._canvas.draw_idle()

    def _apply_axes_style(self):
        """Apply axis visibility and position settings."""
        if not self._show_axes:
            self.ax.axis("off")
            return

        self.ax.axis("on")
        self.ax.set_title("Mandelbrot Set", color=CTRL_FG, fontsize=11, pad=10)
        self.ax.tick_params(colors="#556", labelsize=7)

        if self._axes_pos == "Origin":
            self.ax.spines["left"].set_position(("data", 0))
            self.ax.spines["bottom"].set_position(("data", 0))
            self.ax.spines["right"].set_visible(False)
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["left"].set_color("#556")
            self.ax.spines["bottom"].set_color("#556")
            self.ax.set_xlabel("Re(c)", color=CTRL_FG, fontsize=9)
            self.ax.set_ylabel("Im(c)", color=CTRL_FG, fontsize=9)
            self.ax.xaxis.set_label_coords(1.0, 0.48)
            self.ax.yaxis.set_label_coords(0.48, 1.0)
        else:
            for spine in self.ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#334")
            self.ax.set_xlabel("Re(c)", color=CTRL_FG, fontsize=9)
            self.ax.set_ylabel("Im(c)", color=CTRL_FG, fontsize=9)

    def _on_cmap_change(self, name):
        self._cmap_name = name
        self._update_display()

    def _on_log_toggle(self, state):
        self._log_scale = bool(state)
        self._update_display()

    def _on_axes_toggle(self, state):
        self._show_axes = bool(state)
        self._update_display()

    def _on_axes_pos_change(self, text):
        self._axes_pos = text
        self._update_display()

    # ── Navigation ───────────────────────────────────────────────────────────

    def _on_nav_change(self, _val):
        """Called when user edits center/zoom sliders manually."""
        self._re_center = self.sl_re.value()
        self._im_center = self.sl_im.value()
        self._zoom      = self.sl_zoom.value()
        self._schedule_rerender()

    def _apply_preset(self, re_c, im_c, zoom):
        self._re_center = re_c
        self._im_center = im_c
        self._zoom      = zoom
        self._sync_sliders_from_view()
        self._start_compute()

    def _reset_view(self):
        self._apply_preset(_DEFAULT_RE_CENTER, _DEFAULT_IM_CENTER, _DEFAULT_ZOOM)

    # ── Mouse: pan, zoom, hover ──────────────────────────────────────────────

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:   # left click → start pan
            self._panning = True
            self._pan_start = (event.xdata, event.ydata)

    def _on_mouse_release(self, event):
        if event.button == 1 and self._panning:
            self._panning = False
            self._pan_start = None
            # Update internal state from current axis limits and re-render
            self._read_viewport_from_axes()
            self._sync_sliders_from_view()
            self._start_compute()

    def _on_mouse_move(self, event):
        if self._panning and self._pan_start and event.inaxes == self.ax:
            if event.xdata is None or event.ydata is None:
                return
            dx = self._pan_start[0] - event.xdata
            dy = self._pan_start[1] - event.ydata
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            self.ax.set_xlim(x0 + dx, x1 + dx)
            self.ax.set_ylim(y0 + dy, y1 + dy)
            self._canvas.draw_idle()
            return

        # Hover tooltip
        self._on_hover(event)

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        factor = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        self.ax.set_xlim(cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
        self.ax.set_ylim(cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)
        self._canvas.draw_idle()

        # Debounced re-render
        self._read_viewport_from_axes()
        self._sync_sliders_from_view()
        self._schedule_rerender()

    def _read_viewport_from_axes(self):
        """Read the current axes limits and update internal center/zoom."""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self._re_center = (x0 + x1) / 2.0
        self._im_center = (y0 + y1) / 2.0
        half_w = (x1 - x0) / 2.0
        half_h = (y1 - y0) / 2.0
        zoom_w = _BASE_HALF_WIDTH / half_w if half_w > 0 else 1.0
        zoom_h = _BASE_HALF_HEIGHT / half_h if half_h > 0 else 1.0
        self._zoom = (zoom_w + zoom_h) / 2.0

    def _schedule_rerender(self):
        """Restart the debounce timer for re-render."""
        self._rerender_timer.start()

    def _on_rerender(self):
        """Debounced re-render after pan/zoom settling."""
        self._start_compute()

    def _on_hover(self, event):
        if self._grid is None or event.inaxes != self.ax:
            return
        re_val, im_val = event.xdata, event.ydata
        if re_val is None or im_val is None:
            return

        # Map cursor to grid pixel
        re_min, re_max, im_min, im_max = self._viewport()
        h, w = self._grid.shape
        col = int(round((re_val - re_min) / (re_max - re_min) * (w - 1)))
        row = int(round((im_val - im_min) / (im_max - im_min) * (h - 1)))
        col = np.clip(col, 0, w - 1)
        row = np.clip(row, 0, h - 1)
        val = self._grid[row, col]

        if val < 0:
            iter_str = "IN SET"
        else:
            iter_str = f"{val:.1f} iters"

        self._lbl_coord.setText(
            f"Re = {re_val:.10f}   Im = {im_val:.10f}\n{iter_str}")

    # ── Export (high-resolution re-render) ──────────────────────────────────

    def _start_export(self):
        if self._exporting:
            return
        self._exporting = True
        self._btn_export.setEnabled(False)
        self._btn_export.setText("Rendering...")
        self._btn_compute.setEnabled(False)
        self._export_progress.setValue(0)

        export_w, export_h = self._cb_export_res.currentData()
        re_min, re_max, im_min, im_max = self._viewport()
        max_iter  = self._max_iter
        cmap_name = self._cmap_name
        log_scale = self._log_scale

        is_cuda = self._cb_backend.currentIndex() == 0

        if is_cuda:
            gpu_ids  = [_CUDA_GPUS[i][0]
                        for i, chk in enumerate(self._gpu_chks)
                        if chk.isChecked()]
            use_fp32 = self._cb_precision.currentText().startswith("float32")
            if not gpu_ids:
                self._exporting = False
                self._btn_export.setEnabled(True)
                self._btn_export.setText("Export PNG")
                self._btn_compute.setEnabled(True)
                self._lbl_status.setText("No GPUs selected for export.")
                return
        else:
            n_workers = self._sb_workers.value()

        bridge = self._export_bridge

        def worker():
            try:
                def pcb(pct):
                    bridge.progress.emit(pct)

                if is_cuda:
                    grid = compute_mandelbrot_cuda(
                        re_min, re_max, im_min, im_max,
                        export_w, export_h, max_iter,
                        gpu_ids=gpu_ids, use_fp32=use_fp32,
                        progress_cb=pcb)
                else:
                    grid = compute_mandelbrot_cpu(
                        re_min, re_max, im_min, im_max,
                        export_w, export_h, max_iter,
                        n_workers=n_workers, progress_cb=pcb)

                bridge.progress.emit(96)

                # Apply colourmap
                rgba = apply_colourmap(grid, cmap_name, log_scale)

                bridge.progress.emit(98)

                # Save as PNG
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"mandelbrot_{export_w}x{export_h}_{cmap_name}_{ts}.png"

                # Use matplotlib.image.imsave (no PIL dependency)
                import matplotlib.image as mimg
                mimg.imsave(fname, rgba)

                bridge.finished.emit(fname)

            except Exception as exc:
                import traceback
                traceback.print_exc()
                bridge.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_export_progress(self, pct):
        self._export_progress.setValue(pct)

    def _on_export_done(self, fname):
        self._exporting = False
        self._btn_export.setText("Saved")
        self._btn_compute.setEnabled(not self._computing)
        self._export_progress.setValue(100)
        self._lbl_status.setText(f"Exported: {fname}")
        QTimer.singleShot(3000, self._restore_export_btn)

    def _restore_export_btn(self):
        self._btn_export.setText("Export PNG")
        self._btn_export.setEnabled(True)

    def _on_export_error(self, msg):
        self._exporting = False
        self._btn_export.setEnabled(True)
        self._btn_export.setText("Export PNG")
        self._btn_compute.setEnabled(True)
        self._lbl_status.setText(f"Export error: {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(_QSS)
    win = MandelbrotExplorer()
    win.show()
    sys.exit(app.exec_())
