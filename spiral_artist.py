#!/usr/bin/env python3
"""
Spiral Artist  (PyQt5 edition)
==============================
Generalized spirograph: r(k) = k^p,  θ(k) = k × angle°,  k = 1 … N.

Controls live in a PyQt5 panel; spiral geometry is computed in a background
thread so the UI stays responsive while dragging sliders.

Export dialog lets you pick background (Dark / Light / White / Transparent),
DPI, and output size independently of what you see on screen.

Usage
-----
    conda run -n maths_fun python3 spiral_artist.py
"""

import sys
import threading

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QDoubleSpinBox, QSpinBox, QGroupBox,
    QPushButton, QComboBox, QCheckBox, QScrollArea, QSizePolicy,
    QDialog, QFormLayout,
)
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal

# ── Colours ─────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d0d1a"
LIGHT_BG = "#f5f5f0"
CTRL_BG  = "#12122a"
CTRL_FG  = "#ccccee"
ACC_C    = "#7788ff"
GOLD     = "#FFD700"

GOLDEN_ANGLE = 180.0 * (3.0 - np.sqrt(5.0))   # ≈ 137.508°
ALT_GOLDEN   = 360.0 - GOLDEN_ANGLE            # ≈ 222.492°

QSS = f"""
QMainWindow, QWidget    {{ background: {CTRL_BG}; color: {CTRL_FG}; }}
QGroupBox {{
    border: 1px solid #334; border-radius: 4px; margin-top: 8px;
    font-size: 8pt; color: {GOLD}; font-weight: bold;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 2px; }}
QSlider::groove:horizontal {{ height: 4px; background: #334; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACC_C}; width: 12px; height: 12px;
    border-radius: 6px; margin: -4px 0;
}}
QSlider::sub-page:horizontal {{ background: {ACC_C}; border-radius: 2px; }}
QDoubleSpinBox, QSpinBox {{
    background: #1a1a2e; color: {CTRL_FG}; border: 1px solid #334;
    border-radius: 3px; padding: 1px 3px; font-size: 8pt;
}}
QPushButton {{
    background: #1a1a3e; color: {CTRL_FG}; border: 1px solid #334;
    border-radius: 3px; padding: 3px 6px; font-size: 8pt;
}}
QPushButton:hover   {{ background: #223355; }}
QPushButton:checked {{ background: #1a3a50; border-color: {ACC_C}; }}
QComboBox {{
    background: #1a1a2e; color: {CTRL_FG}; border: 1px solid #334;
    border-radius: 3px; padding: 2px 4px; font-size: 8pt;
}}
QComboBox QAbstractItemView {{
    background: #1a1a2e; color: {CTRL_FG};
    selection-background-color: #223355;
}}
QCheckBox  {{ color: {CTRL_FG}; font-size: 8pt; }}
QLabel     {{ color: {CTRL_FG}; font-size: 8pt; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical   {{ background: #1a1a2e; width: 8px; }}
QScrollBar::handle:vertical {{ background: #334; border-radius: 4px; }}
QDialog {{ background: {CTRL_BG}; }}
"""


# ── Thread-safe signal bridge ────────────────────────────────────────────────────
class _StatusBridge(QObject):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)


# ── Float param slider  (QLabel + QSlider 0-1000 + QDoubleSpinBox) ───────────────
class _ParamSlider(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label, vmin, vmax, vinit, decimals=2, parent=None):
        super().__init__(parent)
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._block = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(1)
        lay.addWidget(QLabel(label))

        row = QWidget()
        rl  = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(3)
        lay.addWidget(row)

        self._sl = QSlider(Qt.Horizontal)
        self._sl.setRange(0, 1000)
        self._sl.setValue(self._to_ticks(vinit))
        rl.addWidget(self._sl, 3)

        self._sb = QDoubleSpinBox()
        self._sb.setDecimals(decimals)
        self._sb.setRange(self._vmin, self._vmax)
        self._sb.setValue(float(vinit))
        self._sb.setFixedWidth(72)
        rl.addWidget(self._sb)

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


# ── Integer param slider  (QLabel + QSlider 0-1000 + QSpinBox) ──────────────────
class _IntParamSlider(QWidget):
    valueChanged = pyqtSignal(int)

    def __init__(self, label, vmin, vmax, vinit, parent=None):
        super().__init__(parent)
        self._vmin = int(vmin)
        self._vmax = int(vmax)
        self._block = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(1)
        lay.addWidget(QLabel(label))

        row = QWidget()
        rl  = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(3)
        lay.addWidget(row)

        self._sl = QSlider(Qt.Horizontal)
        self._sl.setRange(0, 1000)
        self._sl.setValue(self._to_ticks(vinit))
        rl.addWidget(self._sl, 3)

        self._sb = QSpinBox()
        self._sb.setRange(self._vmin, self._vmax)
        self._sb.setValue(int(vinit))
        self._sb.setFixedWidth(72)
        rl.addWidget(self._sb)

        self._sl.valueChanged.connect(self._from_slider)
        self._sb.valueChanged.connect(self._from_spin)

    def _to_ticks(self, v):
        span = self._vmax - self._vmin
        return int(round(1000.0 * (float(v) - self._vmin) / span)) if span else 0

    def _from_ticks(self, t):
        return int(round(self._vmin + t * (self._vmax - self._vmin) / 1000.0))

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
        v = int(np.clip(v, self._vmin, self._vmax))
        self._block = True
        self._sl.setValue(self._to_ticks(v))
        self._sb.setValue(v)
        self._block = False
        if not silent:
            self.valueChanged.emit(v)


# ── Export dialog ────────────────────────────────────────────────────────────────
class _ExportDialog(QDialog):
    _SIZES = [
        ("Sq 2K   2160×2160",  2160, 2160),
        ("Sq 4K   4096×4096",  4096, 4096),
        ("Sq 6K   6000×6000",  6000, 6000),
        ("1080p   1920×1080",  1920, 1080),
        ("1440p   2560×1440",  2560, 1440),
        ("4K      3840×2160",  3840, 2160),
    ]
    _DPIS = [72, 100, 150, 200, 300, 600, 1200]
    _BGS  = ["Dark", "Light", "White", "Transparent"]

    def __init__(self, artist, parent=None):
        super().__init__(parent)
        self._artist = artist
        self.setWindowTitle("Export PNG")
        self.setModal(True)
        self.setStyleSheet(QSS)
        self.setMinimumWidth(340)

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(6)
        lay.addLayout(form)

        self._cb_bg = QComboBox()
        self._cb_bg.addItems(self._BGS)
        self._cb_bg.setCurrentIndex(0 if artist.plot_bg == DARK_BG else 1)
        form.addRow("Background:", self._cb_bg)

        self._cb_dpi = QComboBox()
        for d in self._DPIS:
            self._cb_dpi.addItem(f"{d} dpi")
        self._cb_dpi.setCurrentIndex(self._DPIS.index(300))
        form.addRow("DPI:", self._cb_dpi)

        self._cb_size = QComboBox()
        for label, _w, _h in self._SIZES:
            self._cb_size.addItem(label)
        form.addRow("Size:", self._cb_size)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        lay.addWidget(self._lbl_status)

        self._btn_export = QPushButton("Export PNG")
        self._btn_export.clicked.connect(self._do_export)
        lay.addWidget(self._btn_export)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        lay.addWidget(btn_close)

    def _do_export(self):
        bg_name = self._cb_bg.currentText()
        dpi     = self._DPIS[self._cb_dpi.currentIndex()]
        _, W, H = self._SIZES[self._cb_size.currentIndex()]

        bg_color    = {"Dark": DARK_BG, "Light": LIGHT_BG,
                       "White": "white", "Transparent": "none"}[bg_name]
        transparent = (bg_name == "Transparent")

        self._btn_export.setEnabled(False)
        self._lbl_status.setText("Exporting…")

        artist = self._artist

        def worker():
            try:
                xs, ys = artist._compute_coords(artist.n, artist.r_exp, artist.angle)
                lw     = max(0.5, artist.line_width * min(W, H) / 1800)

                fig = Figure(figsize=(W / dpi, H / dpi), facecolor=bg_color)
                ax  = fig.add_axes([0, 0, 1, 1], facecolor=bg_color)
                ax.set_aspect("equal")
                ax.axis("off")

                if artist.mode in ("Line", "Both"):
                    colors = artist._palette_colors(len(xs))
                    pts    = np.column_stack([xs, ys]).reshape(-1, 1, 2)
                    segs   = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    lc = LineCollection(segs, colors=colors[:-1],
                                        linewidth=lw, alpha=artist.line_alpha,
                                        capstyle="round")
                    ax.add_collection(lc)

                if artist.mode in ("Dots", "Both"):
                    colors = artist._palette_colors(len(xs))
                    ax.scatter(xs, ys, c=colors, s=artist.dot_size,
                               linewidths=0, rasterized=True)

                ax.autoscale_view()

                pal   = artist.palette[:4]
                rev   = "r" if artist.reverse_pal else ""
                fname = (
                    f"spiral_{artist.angle:.2f}deg"
                    f"_p{artist.r_exp:.2f}"
                    f"_N{artist.n}"
                    f"_{pal}{rev}"
                    f"_{bg_name.lower()}_{W}x{H}@{dpi}dpi.png"
                )
                fig.savefig(fname, dpi=dpi, bbox_inches="tight",
                            pad_inches=0.02, transparent=transparent)
                plt.close(fig)
                artist._export_bridge.finished.emit(fname)
            except Exception as exc:
                artist._export_bridge.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def on_export_done(self, fname):
        self._lbl_status.setText(f"Saved: {fname}")
        self._btn_export.setEnabled(True)

    def on_export_error(self, msg):
        self._lbl_status.setText(f"Error: {msg}")
        self._btn_export.setEnabled(True)


# ── Main window ──────────────────────────────────────────────────────────────────
class SpiralArtist(QMainWindow):
    MAX_N    = 100_000
    PALETTES = [
        "plasma", "viridis", "inferno", "magma", "turbo",
        "rainbow", "cool", "twilight", "hot", "hsv",
    ]
    DEFAULTS = dict(
        n           = 3_000,
        r_exp       = 0.5,
        angle       = GOLDEN_ANGLE,
        line_width  = 1.0,
        line_alpha  = 0.85,
        dot_size    = 3.0,
        mode        = "Line",
        palette     = "plasma",
        reverse_pal = False,
        plot_bg     = DARK_BG,
    )

    def __init__(self):
        super().__init__()
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)

        self._generation  = 0
        self._last_coords = None          # (xs, ys) from last successful compute

        # Bridges
        self._bridge = _StatusBridge()
        self._bridge.finished.connect(self._on_compute_done)
        self._bridge.error.connect(self._on_compute_error)

        self._export_bridge = _StatusBridge()

        # Debounce: physics changes wait 250 ms before spawning thread
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(250)
        self._render_timer.timeout.connect(self._start_compute)

        self.setWindowTitle("Spiral Artist")
        self.setStyleSheet(QSS)
        self.resize(1400, 860)

        self._build_plot()
        self._build_controls()
        self._wire_layout()

        QTimer.singleShot(80, self._start_compute)

    # ── Canvas ────────────────────────────────────────────────────────────────

    def _build_plot(self):
        self.fig     = Figure(facecolor=DARK_BG)
        self._canvas = FigureCanvasQTAgg(self.fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_axes([0.01, 0.01, 0.98, 0.95])
        self.ax.set_facecolor(self.plot_bg)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self._title_text = self.fig.text(
            0.5, 0.985, "", ha="center", va="top",
            color=GOLD, fontsize=11, fontweight="bold",
        )

    # ── Control panel ─────────────────────────────────────────────────────────

    def _build_controls(self):
        self._ctrl = QWidget()
        lay = QVBoxLayout(self._ctrl)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        # ── Spiral formula ─────────────────────────────────────────────────
        grp_f = QGroupBox("Spiral Formula")
        fl    = QVBoxLayout(grp_f)
        self.sl_n     = _IntParamSlider("N points", 10, self.MAX_N, self.n)
        self.sl_rexp  = _ParamSlider("r exponent  p", 0.1, 3.0, self.r_exp)
        self.sl_angle = _ParamSlider("θ step  (°)", 0.5, 360.0, self.angle, decimals=3)
        for w in (self.sl_n, self.sl_rexp, self.sl_angle):
            fl.addWidget(w)
        self.sl_n.valueChanged.connect(    lambda v: self._physics_changed("n", int(v)))
        self.sl_rexp.valueChanged.connect( lambda v: self._physics_changed("r_exp", v))
        self.sl_angle.valueChanged.connect(lambda v: self._physics_changed("angle", v))
        lay.addWidget(grp_f)

        # ── Presets ────────────────────────────────────────────────────────
        grp_p = QGroupBox("Presets")
        pg    = QGridLayout(grp_p)
        pg.setSpacing(4)
        for i, (label, ang, rexp) in enumerate([
            ("Sunflower \u2605", GOLDEN_ANGLE, 0.5),
            ("Alt. golden",      ALT_GOLDEN,   0.5),
            ("Pentagon",         144.0,         0.5),
            ("Galaxy",           GOLDEN_ANGLE,  1.0),
        ]):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, a=ang, r=rexp: self._preset(a, r))
            pg.addWidget(btn, i // 2, i % 2)
        lay.addWidget(grp_p)

        # ── Visual style ───────────────────────────────────────────────────
        grp_v = QGroupBox("Visual Style")
        vl    = QVBoxLayout(grp_v)
        self.sl_lw    = _ParamSlider("Line width",  0.1, 8.0,  self.line_width, decimals=1)
        self.sl_alpha = _ParamSlider("Alpha",        0.0, 1.0,  self.line_alpha, decimals=2)
        self.sl_dot   = _ParamSlider("Dot size",     0.5, 30.0, self.dot_size,   decimals=1)
        for w in (self.sl_lw, self.sl_alpha, self.sl_dot):
            vl.addWidget(w)
        self.sl_lw.valueChanged.connect(   lambda v: self._visual_changed("line_width", v))
        self.sl_alpha.valueChanged.connect(lambda v: self._visual_changed("line_alpha", v))
        self.sl_dot.valueChanged.connect(  lambda v: self._visual_changed("dot_size", v))

        mode_row = QWidget()
        ml = QHBoxLayout(mode_row)
        ml.setContentsMargins(0, 2, 0, 2)
        ml.setSpacing(4)
        self._mode_btns = {}
        for m in ("Dots", "Line", "Both"):
            btn = QPushButton(m)
            btn.setCheckable(True)
            btn.setChecked(m == self.mode)
            btn.clicked.connect(lambda _, lbl=m: self._set_mode(lbl))
            ml.addWidget(btn)
            self._mode_btns[m] = btn
        vl.addWidget(mode_row)
        lay.addWidget(grp_v)

        # ── Palette ────────────────────────────────────────────────────────
        grp_pal = QGroupBox("Palette")
        pl      = QVBoxLayout(grp_pal)
        self._cb_palette = QComboBox()
        self._cb_palette.addItems(self.PALETTES)
        self._cb_palette.setCurrentIndex(self.PALETTES.index(self.palette))
        self._cb_palette.currentTextChanged.connect(
            lambda t: self._visual_changed("palette", t))
        pl.addWidget(self._cb_palette)

        self._chk_rev = QCheckBox("Reverse palette")
        self._chk_rev.setChecked(self.reverse_pal)
        self._chk_rev.toggled.connect(lambda v: self._visual_changed("reverse_pal", bool(v)))
        pl.addWidget(self._chk_rev)
        lay.addWidget(grp_pal)

        # ── Background ─────────────────────────────────────────────────────
        grp_bg = QGroupBox("Background")
        bl     = QHBoxLayout(grp_bg)
        self._btn_dark  = QPushButton("Dark")
        self._btn_light = QPushButton("Light")
        for b in (self._btn_dark, self._btn_light):
            b.setCheckable(True)
        self._btn_dark.setChecked(True)
        self._btn_dark.clicked.connect( lambda: self._set_bg(DARK_BG))
        self._btn_light.clicked.connect(lambda: self._set_bg(LIGHT_BG))
        bl.addWidget(self._btn_dark)
        bl.addWidget(self._btn_light)
        lay.addWidget(grp_bg)

        # ── Export ─────────────────────────────────────────────────────────
        grp_exp = QGroupBox("Export")
        el      = QVBoxLayout(grp_exp)
        self._btn_export = QPushButton("Export PNG…")
        self._btn_export.clicked.connect(self._open_export)
        el.addWidget(self._btn_export)
        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        el.addWidget(self._lbl_status)
        lay.addWidget(grp_exp)

        lay.addStretch()

    def _wire_layout(self):
        central = QWidget()
        self.setCentralWidget(central)
        hl = QHBoxLayout(central)
        hl.setContentsMargins(4, 4, 4, 4)
        hl.setSpacing(4)
        hl.addWidget(self._canvas, 3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setWidget(self._ctrl)
        hl.addWidget(scroll)

    # ── Computation (background thread) ───────────────────────────────────────

    @staticmethod
    def _compute_coords(n, r_exp, angle):
        k     = np.arange(1, n + 1, dtype=float)
        r     = k ** r_exp
        theta = np.radians(k * angle)
        return r * np.cos(theta), r * np.sin(theta)

    def _palette_colors(self, n):
        t = np.linspace(0, 1, n)
        if self.reverse_pal:
            t = 1 - t
        return plt.get_cmap(self.palette)(t)

    def _start_compute(self):
        self._generation += 1
        gen    = self._generation
        n      = self.n
        r_exp  = self.r_exp
        angle  = self.angle
        bridge = self._bridge

        def worker():
            try:
                xs, ys = SpiralArtist._compute_coords(n, r_exp, angle)
                if gen == self._generation:
                    bridge.finished.emit((xs, ys))
            except Exception as exc:
                bridge.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_compute_done(self, result):
        self._last_coords = result
        self._render(*result)

    def _on_compute_error(self, msg):
        self._lbl_status.setText(f"Render error: {msg}")

    # ── Rendering (main thread) ───────────────────────────────────────────────

    def _render(self, xs, ys):
        ax = self.ax
        ax.cla()
        ax.set_facecolor(self.plot_bg)
        ax.set_aspect("equal")
        ax.axis("off")
        self.fig.patch.set_facecolor(self.plot_bg)

        if self.mode in ("Line", "Both"):
            colors = self._palette_colors(len(xs))
            pts    = np.column_stack([xs, ys]).reshape(-1, 1, 2)
            segs   = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, colors=colors[:-1],
                                linewidth=self.line_width, alpha=self.line_alpha,
                                capstyle="round")
            ax.add_collection(lc)

        if self.mode in ("Dots", "Both"):
            colors = self._palette_colors(len(xs))
            ax.scatter(xs, ys, c=colors, s=self.dot_size,
                       linewidths=0, rasterized=True)

        ax.autoscale_view()
        self._title_text.set_text(
            f"r = k^{self.r_exp:.2f}   θ = k × {self.angle:.3f}°"
            f"   ·   {self.n:,} pts   ·   {self.palette}"
        )
        self._canvas.draw_idle()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _physics_changed(self, attr, val):
        setattr(self, attr, val)
        self._render_timer.start()          # debounced thread launch

    def _visual_changed(self, attr, val):
        setattr(self, attr, val)
        if self._last_coords is not None:
            self._render(*self._last_coords)    # instant, no thread
        else:
            self._render_timer.start()

    def _set_mode(self, mode):
        self.mode = mode
        for lbl, btn in self._mode_btns.items():
            btn.setChecked(lbl == mode)
        if self._last_coords is not None:
            self._render(*self._last_coords)
        else:
            self._render_timer.start()

    def _preset(self, angle, r_exp):
        self.angle = angle
        self.r_exp = r_exp
        self.sl_angle.set_value(angle)
        self.sl_rexp.set_value(r_exp)
        self._render_timer.start()

    def _set_bg(self, color):
        self.plot_bg = color
        self._btn_dark.setChecked( color == DARK_BG)
        self._btn_light.setChecked(color == LIGHT_BG)
        if self._last_coords is not None:
            self._render(*self._last_coords)
        else:
            self._render_timer.start()

    def _open_export(self):
        dlg = _ExportDialog(self, parent=self)
        self._export_bridge.finished.connect(dlg.on_export_done)
        self._export_bridge.error.connect(dlg.on_export_error)
        self._export_bridge.finished.connect(self._on_export_status)
        self._export_bridge.error.connect(self._on_export_error)
        dlg.exec_()
        try:
            self._export_bridge.finished.disconnect(dlg.on_export_done)
            self._export_bridge.error.disconnect(dlg.on_export_error)
            self._export_bridge.finished.disconnect(self._on_export_status)
            self._export_bridge.error.disconnect(self._on_export_error)
        except TypeError:
            pass

    def _on_export_status(self, fname):
        self._lbl_status.setText(f"Saved: {fname}")

    def _on_export_error(self, msg):
        self._lbl_status.setText(f"Export error: {msg}")


# ── Entry point ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SpiralArtist()
    win.show()
    sys.exit(app.exec_())
