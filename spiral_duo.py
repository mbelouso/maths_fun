#!/usr/bin/env python3
"""
Spiral Duo
==========
Two independent spirals rendered simultaneously and alpha-blended together.

Formula (per spiral)
---------------------
    r(k) = k ^ p
    θ(k) = k × angle°
    k    = 1 … N

Architecture
------------
  GUI        : PyQt5 (native widgets, instant startup)
  Rendering  : matplotlib Agg backend → numpy RGBA buffer (one Figure per render call)
  Threading  : ThreadPoolExecutor renders both spirals in parallel;
               QRunnable keeps heavy work off the Qt main thread
  Compositing: Porter-Duff 'over' to alpha-blend both spirals onto a background

Usage
-----
    pip install PyQt5          # if not already installed
    python spiral_duo.py
"""

import sys
import math
import numpy as np
from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QSlider, QSpinBox, QDoubleSpinBox, QPushButton, QComboBox,
    QButtonGroup, QAbstractButton, QTabWidget, QGroupBox, QScrollArea,
    QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QRunnable, QThreadPool, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

import matplotlib
matplotlib.use("Agg")                          # non-interactive, no window overhead
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection

# ── Constants ──────────────────────────────────────────────────────────────────

GOLDEN_ANGLE = 180.0 * (3.0 - math.sqrt(5.0))   # ≈ 137.508°
ALT_GOLDEN   = 360.0 - GOLDEN_ANGLE              # ≈ 222.492°

DARK_BG  = (13,  13,  26)    # #0d0d1a
LIGHT_BG = (245, 245, 240)   # #f5f5f0

PALETTES = [
    "plasma", "viridis", "inferno", "magma", "turbo",
    "rainbow", "cool", "twilight", "hot", "hsv",
]

PRESETS = [
    ("Sunflower ★", GOLDEN_ANGLE, 0.5),
    ("Alt. Golden",  ALT_GOLDEN,  0.5),
    ("Pentagon",     144.0,        0.5),
    ("Galaxy",       GOLDEN_ANGLE, 1.0),
]

# (label, pixel_width, pixel_height)
EXPORT_RESOLUTIONS = [
    ("HD   1280×720",   1280,  720),
    ("FHD  1920×1080",  1920, 1080),
    ("QHD  2560×1440",  2560, 1440),
    ("4K   3840×2160",  3840, 2160),
    ("Sq   2000×2000",  2000, 2000),
    ("Sq   4000×4000",  4000, 4000),
]

# ── Stylesheet (dark theme) ────────────────────────────────────────────────────

STYLE = """
QWidget {
    background-color: #0d0d1a;
    color: #ccccee;
    font-size: 11px;
}
QGroupBox {
    border: 1px solid #334466;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    color: #FFD700;
    font-weight: bold;
}
QTabWidget::pane { border: 1px solid #334466; }
QTabBar::tab {
    background: #12122a;
    color: #ccccee;
    padding: 5px 14px;
    border: 1px solid #334;
}
QTabBar::tab:selected { background: #1a3a50; color: #FFD700; }
QSlider::groove:horizontal {
    background: #223355;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #7788ff;
    width: 13px;
    height: 13px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #7788ff; border-radius: 2px; }
QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #1a1a2e;
    border: 1px solid #334466;
    border-radius: 3px;
    padding: 2px 4px;
    color: #ccccee;
    min-width: 68px;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button,       QSpinBox::down-button {
    background-color: #12122a;
    border: none;
}
QComboBox::drop-down   { border: none; }
QComboBox QAbstractItemView {
    background-color: #1a1a2e;
    selection-background-color: #1a3a50;
}
QPushButton {
    background-color: #1a1a3e;
    border: 1px solid #334466;
    border-radius: 3px;
    padding: 4px 8px;
    color: #ccccee;
}
QPushButton:hover    { background-color: #223355; }
QPushButton:checked  { background-color: #1a3a50; color: #FFD700; border-color: #7788ff; }
QLabel#spiralView    { background-color: #0d0d1a; border: 1px solid #334; }
QScrollBar:vertical  { background: #0d0d1a; width: 8px; }
QScrollBar::handle:vertical { background: #334466; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

# ── Spiral parameters ──────────────────────────────────────────────────────────

@dataclass
class SpiralParams:
    n:          int   = 500
    r_exp:      float = 0.2
    angle:      float = GOLDEN_ANGLE
    scale:      float = 1.0      # multiplies r(k) — use to match sizes between spirals
    line_width: float = 1.0
    line_alpha: float = 0.85
    dot_size:   float = 3.0
    angle_offset: float = 0.0    # degrees added to every θ(k)
    mode:         str   = "Line"   # "Dots" | "Line" | "Both"
    palette:      str   = "plasma"
    reverse:      bool  = False

# ── Thread-safe render functions ───────────────────────────────────────────────

def _draw_spiral(ax, params: SpiralParams, global_alpha: float) -> None:
    """
    Draw one spiral onto `ax`. Does NOT call autoscale_view or canvas.draw.
    Both spirals must be drawn onto the same axes so they share one coordinate
    system — this ensures that offset changes rotate around the same origin.
    """
    k     = np.arange(1, params.n + 1, dtype=np.float64)
    r     = params.scale * (k ** params.r_exp)
    theta = np.deg2rad(k * params.angle + params.angle_offset)
    xs    = r * np.cos(theta)
    ys    = r * np.sin(theta)

    t = np.linspace(0.0, 1.0, params.n)
    if params.reverse:
        t = 1.0 - t
    colors = matplotlib.colormaps[params.palette](t)   # (N, 4) float64

    if params.mode in ("Line", "Both") and params.n > 1:
        pts  = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(
            segs, colors=colors[:-1],
            linewidth=params.line_width,
            alpha=params.line_alpha * global_alpha,
            capstyle="round",
        )
        ax.add_collection(lc)

    if params.mode in ("Dots", "Both"):
        dot_c = colors.copy()
        dot_c[:, 3] *= global_alpha
        ax.scatter(xs, ys, c=dot_c, s=params.dot_size, linewidths=0)


def render_combined(p1: SpiralParams, p2: SpiralParams, blend: float,
                    bg_rgb: tuple, width: int, height: int,
                    transparent: bool = False) -> np.ndarray:
    """
    Render both spirals onto one shared axes and return a (H × W × 4) uint8
    RGBA array. Thread-safe: creates its own Figure/Canvas.

    Using a single axes means autoscale_view sees all data from both spirals,
    so the origin (rotation centre for offset) maps to the same pixel for both,
    and both are displayed at the same scale regardless of their parameters.

    Blend cross-fade:
        blend=0   → only spiral 1 visible
        blend=0.5 → both at full opacity
        blend=1   → only spiral 2 visible

    transparent=True → figure and axes background are fully transparent (RGBA
        export); for the live preview the widget background shows through.
    """
    dpi = 96
    bg_f = (0.0, 0.0, 0.0, 0.0) if transparent else (
        bg_rgb[0] / 255.0, bg_rgb[1] / 255.0, bg_rgb[2] / 255.0, 1.0
    )

    fig    = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.set_facecolor(bg_f)
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(bg_f)
    if transparent:
        ax.patch.set_alpha(0.0)
    ax.set_aspect("equal")
    ax.axis("off")

    a1 = float(min(1.0, max(0.0, 2.0 * (1.0 - blend))))
    a2 = float(min(1.0, max(0.0, 2.0 * blend)))

    _draw_spiral(ax, p1, a1)
    _draw_spiral(ax, p2, a2)

    ax.autoscale_view()
    canvas.draw()

    buf = canvas.buffer_rgba()
    return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4).copy()

# ── Qt worker signals (must be top-level QObject subclasses) ──────────────────

class RenderSignals(QObject):
    finished = pyqtSignal(QImage)

class ExportSignals(QObject):
    done = pyqtSignal(str)   # emits saved filename

# ── Render worker ─────────────────────────────────────────────────────────────

class RenderWorker(QRunnable):
    def __init__(self, p1: SpiralParams, p2: SpiralParams,
                 blend: float, bg_rgb: tuple, width: int, height: int,
                 transparent: bool = False):
        super().__init__()
        self.p1, self.p2    = p1, p2
        self.blend          = blend
        self.bg_rgb         = bg_rgb
        self.W, self.H      = width, height
        self.transparent    = transparent
        self.signals        = RenderSignals()
        self.setAutoDelete(True)

    @pyqtSlot()
    def run(self):
        arr  = render_combined(self.p1, self.p2, self.blend, self.bg_rgb,
                               self.W, self.H, self.transparent)
        qimg = QImage(arr.data, self.W, self.H, arr.strides[0], QImage.Format_RGBA8888)
        self.signals.finished.emit(qimg.copy())

# ── Export worker ─────────────────────────────────────────────────────────────

class ExportWorker(QRunnable):
    def __init__(self, p1: SpiralParams, p2: SpiralParams,
                 blend: float, bg_rgb: tuple,
                 width: int, height: int, transparent: bool = False):
        super().__init__()
        self.p1, self.p2    = p1, p2
        self.blend          = blend
        self.bg_rgb         = bg_rgb
        self.W, self.H      = width, height
        self.transparent    = transparent
        self.signals        = ExportSignals()
        self.setAutoDelete(True)

    @pyqtSlot()
    def run(self):
        arr = render_combined(self.p1, self.p2, self.blend, self.bg_rgb,
                              self.W, self.H, self.transparent)
        if self.transparent:
            bg_tag = "transparent"
        else:
            bg_tag = "lite" if self.bg_rgb == LIGHT_BG else "dark"
        fname = (f"duo_{self.p1.angle:.2f}v{self.p2.angle:.2f}deg"
                 f"_blend{self.blend:.2f}_{bg_tag}_{self.W}x{self.H}.png")
        qimg = QImage(arr.data, self.W, self.H, arr.strides[0], QImage.Format_RGBA8888)
        qimg.copy().save(fname)
        self.signals.done.emit(fname)

# ── Reusable parameter widgets ────────────────────────────────────────────────

class ParamWidget(QWidget):
    """QLabel  +  QSlider  +  QDoubleSpinBox  for a float parameter."""
    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, vmin: float, vmax: float, vinit: float,
                 decimals: int = 2, step: float = None, parent=None):
        super().__init__(parent)
        self._scale   = 10 ** decimals
        self._blocked = False

        lbl = QLabel(label)
        lbl.setFixedWidth(52)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(vmin * self._scale), int(vmax * self._scale))
        self.slider.setValue(int(vinit * self._scale))

        self.spin = QDoubleSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step if step else (vmax - vmin) / 100.0)
        self.spin.setValue(vinit)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 1, 0, 1)
        row.addWidget(lbl)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v: float, silent: bool = False):
        self._blocked = True
        self.slider.setValue(int(v * self._scale))
        self.spin.setValue(v)
        self._blocked = False
        if not silent:
            self.valueChanged.emit(v)

    def _from_slider(self, ival: int):
        if self._blocked:
            return
        fval = ival / self._scale
        self._blocked = True
        self.spin.setValue(fval)
        self._blocked = False
        self.valueChanged.emit(fval)

    def _from_spin(self, fval: float):
        if self._blocked:
            return
        self._blocked = True
        self.slider.setValue(int(fval * self._scale))
        self._blocked = False
        self.valueChanged.emit(fval)


class IntParamWidget(QWidget):
    """QLabel  +  QSlider  +  QSpinBox  for an integer parameter."""
    valueChanged = pyqtSignal(int)

    def __init__(self, label: str, vmin: int, vmax: int, vinit: int,
                 step: int = 500, parent=None):
        super().__init__(parent)
        self._blocked = False

        lbl = QLabel(label)
        lbl.setFixedWidth(52)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(vmin, vmax)
        self.slider.setValue(vinit)
        self.slider.setSingleStep(step)
        self.slider.setPageStep(step * 5)

        self.spin = QSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setValue(vinit)
        self.spin.setSingleStep(step)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 1, 0, 1)
        row.addWidget(lbl)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> int:
        return self.spin.value()

    def setValue(self, v: int, silent: bool = False):
        self._blocked = True
        self.slider.setValue(v)
        self.spin.setValue(v)
        self._blocked = False
        if not silent:
            self.valueChanged.emit(v)

    def _from_slider(self, v: int):
        if self._blocked:
            return
        self._blocked = True
        self.spin.setValue(v)
        self._blocked = False
        self.valueChanged.emit(v)

    def _from_spin(self, v: int):
        if self._blocked:
            return
        self._blocked = True
        self.slider.setValue(v)
        self._blocked = False
        self.valueChanged.emit(v)

# ── Spiral configuration panel ────────────────────────────────────────────────

class SpiralPanel(QWidget):
    """Full set of controls for one spiral. Emits `changed` on any update."""
    changed = pyqtSignal()

    def __init__(self, defaults: SpiralParams, parent=None):
        super().__init__(parent)
        self._suspend = False
        self._build_ui(defaults)

    def _build_ui(self, p: SpiralParams):
        vb = QVBoxLayout(self)
        vb.setSpacing(5)
        vb.setContentsMargins(6, 6, 6, 6)

        # ── Formula ──────────────────────────────────────────────────────────
        frm_grp = QGroupBox("Formula")
        frm_vb  = QVBoxLayout(frm_grp)
        frm_vb.setSpacing(3)

        self.wn      = IntParamWidget("N",      10, 100_000, p.n,          step=500)
        self.wrexp   = ParamWidget("p",          0.1, 3.0,   p.r_exp,      decimals=2, step=0.05)
        self.wangle  = ParamWidget("θ  (°)",    0.5, 360.0,  p.angle,      decimals=3, step=0.5)
        self.woffset = ParamWidget("offset (°)", 0.0, 360.0,  p.angle_offset, decimals=1, step=1.0)
        self.wscale  = ParamWidget("scale",      0.1, 5.0,   p.scale,      decimals=2, step=0.05)

        for w in (self.wn, self.wrexp, self.wangle, self.woffset, self.wscale):
            frm_vb.addWidget(w)
            w.valueChanged.connect(self._emit_changed)

        vb.addWidget(frm_grp)

        # ── Presets ───────────────────────────────────────────────────────────
        pre_grp = QGroupBox("Presets")
        pre_grid = QGridLayout(pre_grp)
        pre_grid.setSpacing(4)

        for i, (name, angle, rexp) in enumerate(PRESETS):
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, a=angle, r=rexp: self._apply_preset(a, r))
            pre_grid.addWidget(btn, i // 2, i % 2)

        vb.addWidget(pre_grp)

        # ── Visual style ──────────────────────────────────────────────────────
        vis_grp = QGroupBox("Visual style")
        vis_vb  = QVBoxLayout(vis_grp)
        vis_vb.setSpacing(3)

        self.wlw  = ParamWidget("Line w",  0.1, 8.0,  p.line_width, decimals=1, step=0.1)
        self.wla  = ParamWidget("Alpha",   0.0, 1.0,  p.line_alpha, decimals=2, step=0.05)
        self.wds  = ParamWidget("Dot sz",  0.5, 30.0, p.dot_size,   decimals=1, step=0.5)

        for w in (self.wlw, self.wla, self.wds):
            vis_vb.addWidget(w)
            w.valueChanged.connect(self._emit_changed)

        # Display mode (Dots / Line / Both)
        mode_row = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for lbl in ("Dots", "Line", "Both"):
            btn = QPushButton(lbl)
            btn.setCheckable(True)
            btn.setChecked(lbl == p.mode)
            self._mode_group.addButton(btn)
            mode_row.addWidget(btn)
        self._mode_group.buttonClicked.connect(self._emit_changed)
        vis_vb.addLayout(mode_row)

        vb.addWidget(vis_grp)

        # ── Palette ───────────────────────────────────────────────────────────
        pal_grp = QGroupBox("Palette")
        pal_vb  = QVBoxLayout(pal_grp)
        pal_vb.setSpacing(4)

        pal_row = QHBoxLayout()
        pal_row.addWidget(QLabel("Map"))
        self.palette_box = QComboBox()
        self.palette_box.addItems(PALETTES)
        self.palette_box.setCurrentText(p.palette)
        self.palette_box.currentTextChanged.connect(self._emit_changed)
        pal_row.addWidget(self.palette_box, 1)
        pal_vb.addLayout(pal_row)

        self.rev_btn = QPushButton("Reverse gradient")
        self.rev_btn.setCheckable(True)
        self.rev_btn.setChecked(p.reverse)
        self.rev_btn.toggled.connect(self._emit_changed)
        pal_vb.addWidget(self.rev_btn)

        vb.addWidget(pal_grp)
        vb.addStretch()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _emit_changed(self, *_):
        if not self._suspend:
            self.changed.emit()

    def _apply_preset(self, angle: float, r_exp: float):
        """Apply preset with a single render (suppress intermediate signals)."""
        self._suspend = True
        self.wangle.setValue(angle, silent=True)
        self.wrexp.setValue(r_exp,  silent=True)
        self._suspend = False
        self.changed.emit()

    def params(self) -> SpiralParams:
        """Read all widget values and return current SpiralParams."""
        checked = self._mode_group.checkedButton()
        return SpiralParams(
            n            = self.wn.value(),
            r_exp        = self.wrexp.value(),
            angle        = self.wangle.value(),
            angle_offset = self.woffset.value(),
            scale        = self.wscale.value(),
            line_width = self.wlw.value(),
            line_alpha = self.wla.value(),
            dot_size   = self.wds.value(),
            mode       = checked.text() if checked else "Line",
            palette    = self.palette_box.currentText(),
            reverse    = self.rev_btn.isChecked(),
        )

# ── Main window ───────────────────────────────────────────────────────────────

class SpiralDuo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spiral Duo")
        self.resize(1400, 860)

        self._blend       = 0.5
        self._bg_rgb      = DARK_BG
        self._transparent = False

        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(4)

        self._workers     = set()   # keeps QRunnable Python wrappers alive (GC fix)
        self._render_gen  = 0       # incremented each render; dropped if stale

        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._do_render)

        self._build_ui()

    def _build_ui(self):
        root_w  = QWidget()
        root_hb = QHBoxLayout(root_w)
        root_hb.setContentsMargins(6, 6, 6, 6)
        root_hb.setSpacing(6)
        self.setCentralWidget(root_w)

        # ── Left: spiral canvas ───────────────────────────────────────────────
        self.view = QLabel()
        self.view.setObjectName("spiralView")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setMinimumSize(500, 400)
        root_hb.addWidget(self.view, 3)

        # ── Right: control panel ──────────────────────────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(320)
        ctrl_vb = QVBoxLayout(ctrl)
        ctrl_vb.setContentsMargins(0, 0, 0, 0)
        ctrl_vb.setSpacing(6)

        # Blend slider
        blend_grp = QGroupBox("Blend  ( 0 = Spiral 1  ·  1 = Spiral 2 )")
        blend_vb  = QVBoxLayout(blend_grp)
        self.blend_w = ParamWidget("", 0.0, 1.0, self._blend, decimals=2, step=0.01)
        self.blend_w.valueChanged.connect(self._on_blend)
        blend_vb.addWidget(self.blend_w)
        ctrl_vb.addWidget(blend_grp)

        # Spiral tabs
        p1 = SpiralParams(angle=GOLDEN_ANGLE, palette="plasma")
        p2 = SpiralParams(angle=ALT_GOLDEN,   palette="viridis")

        self.panel1 = SpiralPanel(p1)
        self.panel2 = SpiralPanel(p2)
        self.panel1.changed.connect(self._schedule_render)
        self.panel2.changed.connect(self._schedule_render)

        def _scrolled(panel):
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(panel)
            sa.setFrameShape(QScrollArea.NoFrame)
            return sa

        tabs = QTabWidget()
        tabs.addTab(_scrolled(self.panel1), "Spiral 1")
        tabs.addTab(_scrolled(self.panel2), "Spiral 2")
        ctrl_vb.addWidget(tabs, 1)

        # Background + Export
        bot_grp = QGroupBox("Settings")
        bot_vb  = QVBoxLayout(bot_grp)

        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Background"))
        self._btn_dark   = QPushButton("Dark");        self._btn_dark.setCheckable(True);  self._btn_dark.setChecked(True)
        self._btn_light  = QPushButton("Light");       self._btn_light.setCheckable(True)
        self._btn_transp = QPushButton("Transparent"); self._btn_transp.setCheckable(True)
        bg_grp = QButtonGroup(self); bg_grp.setExclusive(True)
        bg_grp.addButton(self._btn_dark)
        bg_grp.addButton(self._btn_light)
        bg_grp.addButton(self._btn_transp)
        self._btn_dark.clicked.connect(  lambda: self._set_bg(DARK_BG,  False))
        self._btn_light.clicked.connect( lambda: self._set_bg(LIGHT_BG, False))
        self._btn_transp.clicked.connect(lambda: self._set_bg(DARK_BG,  True))
        bg_row.addWidget(self._btn_dark)
        bg_row.addWidget(self._btn_light)
        bg_row.addWidget(self._btn_transp)
        bot_vb.addLayout(bg_row)

        exp_row = QHBoxLayout()
        self.res_box = QComboBox()
        for label, _, _ in EXPORT_RESOLUTIONS:
            self.res_box.addItem(label)
        self.res_box.setCurrentIndex(1)   # default FHD 1920×1080
        self.export_btn = QPushButton("Export PNG")
        self.export_btn.clicked.connect(self._export)
        exp_row.addWidget(self.res_box)
        exp_row.addWidget(self.export_btn, 1)
        bot_vb.addLayout(exp_row)

        ctrl_vb.addWidget(bot_grp)
        root_hb.addWidget(ctrl)

    # ── Render pipeline ───────────────────────────────────────────────────────

    def _schedule_render(self):
        self._timer.start(120)   # debounce: 120 ms after last change

    def _on_blend(self, v: float):
        self._blend = v
        self._schedule_render()

    def _set_bg(self, rgb: tuple, transparent: bool):
        self._bg_rgb      = rgb
        self._transparent = transparent
        self._schedule_render()

    def _do_render(self):
        W = max(self.view.width(),  100)
        H = max(self.view.height(), 100)
        self._render_gen += 1
        gen = self._render_gen
        worker = RenderWorker(
            self.panel1.params(), self.panel2.params(),
            self._blend, self._bg_rgb, W, H,
            transparent=self._transparent,
        )
        self._workers.add(worker)
        worker.signals.finished.connect(
            lambda img, w=worker, g=gen: self._on_render_done(img, w, g)
        )
        self._pool.start(worker)

    @pyqtSlot(QImage)
    def _on_render_done(self, qimg: QImage, worker=None, gen: int = 0):
        self._workers.discard(worker)
        if gen != self._render_gen:
            return   # stale result — a newer render is already in flight
        self.view.setPixmap(QPixmap.fromImage(qimg))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_render()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        _, W, H = EXPORT_RESOLUTIONS[self.res_box.currentIndex()]
        worker = ExportWorker(
            self.panel1.params(), self.panel2.params(),
            self._blend, self._bg_rgb, W, H,
            transparent=self._transparent,
        )
        self._workers.add(worker)
        worker.signals.done.connect(
            lambda fname, w=worker: self._on_export_done(fname, w)
        )
        self._pool.start(worker)
        self.export_btn.setText("Rendering…")
        self.export_btn.setEnabled(False)

    @pyqtSlot(str)
    def _on_export_done(self, fname: str, worker=None):
        self._workers.discard(worker)
        self.export_btn.setText("Saved ✓")
        self.export_btn.setEnabled(True)
        QTimer.singleShot(3000, lambda: self.export_btn.setText("Export PNG"))

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = SpiralDuo()
    win.show()
    QTimer.singleShot(200, win._schedule_render)
    sys.exit(app.exec_())
