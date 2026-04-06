"""3_body_problem.py — Interactive 3D three-body problem visualiser (PyQt5).

Layout
------
  Left   : interactive matplotlib 3D canvas (rotate with mouse)
  Right  : native Qt control panel (scrollable)

Physics units
-------------
  G = 1, distances in AU, masses in M☉
  1 sim-time unit = 1 / (2π) years  (from Kepler: T=2π at a=1 AU, M=1 M☉)

Collision model
---------------
  Bodies merging within Collision Radius: inelastic merge (momentum-conserving).
  Absorbed body's position becomes NaN; absorber gains the mass.

Author: Matthew Belousoff, Claude 2026
"""

import sys
import threading
import datetime

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")                          # must be before pyplot/Figure
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QDoubleSpinBox, QGroupBox,
    QPushButton, QComboBox, QCheckBox,
    QScrollArea, QSizePolicy,
)
from PyQt5.QtCore  import Qt, QTimer, pyqtSignal, QObject

from three_body_physics import (
    Body, body_color, body_size, body_label,
    integrate, integrate_with_collisions, compute_forces,
    PRESETS, COLLISION_RADIUS, _figure8_bodies,
)

# ── Time conversion ──────────────────────────────────────────────────────────
# G=1, AU, M☉  →  1 sim-time = 1/(2π) years
_YR = 1.0 / (2.0 * np.pi)

# ── Palette ───────────────────────────────────────────────────────────────────
DARK_BG = "#07070f"
CTRL_BG = "#12122a"
CTRL_FG = "#ccccee"
ACC_C   = "#7788ff"
GOLD    = "#FFD700"

CAT_STYLE = {
    "stable":      ("background:#163a16;", "background:#1f5a1f;"),
    "semi-stable": ("background:#3a2d08;", "background:#5a4510;"),
    "chaotic":     ("background:#3a1010;", "background:#5a1818;"),
}

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
QDoubleSpinBox::down-button   { width:14px; }
QPushButton                   { background:#1a1a3e; color:#ccccee;
                                border:1px solid #445; padding:5px 8px;
                                border-radius:3px; }
QPushButton:hover             { background:#2a2a5e; }
QPushButton:pressed           { background:#0a0a2e; }
QCheckBox                     { color:#ccccee; font-size:9pt; }
"""


# ── Reusable parameter row: label + slider + spinbox ─────────────────────────
class _ParamSlider(QWidget):
    """Label above, then QSlider + QDoubleSpinBox side-by-side, bidirectionally synced."""

    valueChanged = pyqtSignal(float)

    def __init__(self, label, vmin, vmax, vinit, decimals=2, suffix="", parent=None):
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

    # ── internal helpers ──────────────────────────────────────────────────────
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

    # ── public API ────────────────────────────────────────────────────────────
    def value(self):
        return self._sb.value()

    def set_value(self, v, silent=True):
        """Update both widgets.  Emits valueChanged unless silent=True."""
        v = float(np.clip(v, self._vmin, self._vmax))
        self._block = True
        self._sl.setValue(self._to_ticks(v))
        self._sb.setValue(v)
        self._block = False
        if not silent:
            self.valueChanged.emit(v)


# ── Status signal helper (thread-safe bridge from worker → main thread) ───────
class _StatusBridge(QObject):
    message = pyqtSignal(str)


# ── Main window ───────────────────────────────────────────────────────────────
class ThreeBodyApp(QMainWindow):
    N_TRAIL_SEGS = 6
    TRAIL_ALPHAS = [0.05, 0.12, 0.22, 0.38, 0.56, 0.76]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Three-Body Problem")
        self.resize(1440, 860)

        # ── Simulation state ──────────────────────────────────────────────────
        self._preset_name  = list(PRESETS.keys())[0]
        self._base_bodies  = None
        self._traj         = None        # (ts, pos_arr, vel_arr, merges)
        self._pending_traj = None
        self._integrating  = False
        self._exporting    = False
        self._frame_idx    = 0
        self._merge_flash  = 0
        self._active_mass  = [1.0, 1.0, 1.0]

        self.use_collisions = True
        self.mass           = [1.0, 1.0, 1.0]
        self.sep_scale      = 1.0
        self.tilt_deg       = 0.0
        self.coll_radius    = COLLISION_RADIUS
        self.t_end          = 20.0
        self.rtol_exp       = 9
        self.trail_len      = 60.0
        self.anim_speed     = 3.0
        self.ode_method     = "DOP853"

        # Live camera state
        self._view_radius = 1.0
        self._view_center = np.zeros(3)

        # Matplotlib artist lists
        self._core_scat   = []
        self._glow_scat   = [[], []]
        self._trail_lines = []
        self._label_txt   = []
        self._force_lines = []
        self._force_tips  = []
        self._show_forces = False

        # Thread → main-thread status bridge
        self._bridge = _StatusBridge()
        self._bridge.message.connect(self._on_status)

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_plot()       # creates self.fig, self.ax3d, self._canvas
        self._build_controls()   # creates self._ctrl_widget
        self._wire_central()     # assembles central widget

        # Load first preset (runs integration in background)
        self._apply_preset(self._preset_name)

        # Animation timer — fires on Qt's event loop (main thread)
        self._timer = QTimer(self)
        self._timer.setInterval(40)          # 25 fps
        self._timer.timeout.connect(self._update_frame)
        self._timer.start()

    # ── Central widget layout ─────────────────────────────────────────────────
    def _wire_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        lay = QHBoxLayout(central)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        lay.addWidget(self._canvas, 3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(310)
        scroll.setWidget(self._ctrl_widget)
        lay.addWidget(scroll)

    # ── Matplotlib 3D canvas ──────────────────────────────────────────────────
    def _build_plot(self):
        self.fig = Figure(facecolor=DARK_BG)
        self._canvas = FigureCanvas(self.fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self._style_3d_ax()

        # Overlay texts on the 3D axis
        self._status_txt = self.ax3d.text2D(
            0.5, 0.5, "", ha="center", va="center",
            color=GOLD, fontsize=14, fontweight="bold",
            transform=self.ax3d.transAxes)
        self._merge_txt = self.ax3d.text2D(
            0.5, 0.06, "", ha="center", va="bottom",
            color="#ff9944", fontsize=10, fontweight="bold",
            transform=self.ax3d.transAxes)
        self._time_txt = self.ax3d.text2D(
            0.02, 0.96, "t = 0.000 yr", ha="left", va="top",
            color=CTRL_FG, fontsize=9.5,
            transform=self.ax3d.transAxes)

        self._build_body_artists()

    def _style_3d_ax(self):
        self.ax3d.set_facecolor(DARK_BG)
        # Hide the three pane faces and their edges completely
        for attr in ("xaxis", "yaxis", "zaxis"):
            pane = getattr(self.ax3d, attr).pane
            pane.fill = False
            pane.set_edgecolor("none")
        # No grid planes — they shimmer with dynamic scaling
        self.ax3d.grid(False)
        # Keep tick marks + labels along each axis line for spatial reference
        self.ax3d.tick_params(
            axis="both", colors="#445", labelsize=6,
            length=3, width=0.5, pad=1)
        for attr in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(self.ax3d, attr)
            axis.label.set_color("#556")
            # Subtle axis spine line
            axis.line.set_color("#334")
            axis.line.set_linewidth(0.6)
        self.ax3d.set_xlabel("x  (AU)", fontsize=7.5, labelpad=2)
        self.ax3d.set_ylabel("y  (AU)", fontsize=7.5, labelpad=2)
        self.ax3d.set_zlabel("z  (AU)", fontsize=7.5, labelpad=2)

    # ── Qt control panel ──────────────────────────────────────────────────────
    def _build_controls(self):
        self._ctrl_widget = QWidget()
        lay = QVBoxLayout(self._ctrl_widget)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # Title
        title = QLabel("THREE-BODY PROBLEM")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color:#FFD700; font-size:11pt; font-weight:bold; padding:4px;")
        lay.addWidget(title)

        # ── Presets ──────────────────────────────────────────────────────────
        grp_pre = QGroupBox("Presets")
        grid = QGridLayout(grp_pre)
        grid.setSpacing(3)
        names  = list(PRESETS.keys())
        n_cols = 2
        self._preset_btns = {}
        for idx, name in enumerate(names):
            cat     = PRESETS[name]["category"]
            bg, hbg = CAT_STYLE[cat]
            btn = QPushButton(name)
            btn.setStyleSheet(
                f"QPushButton {{{bg} color:#ccccee; border:1px solid #445;"
                f" border-radius:3px; padding:3px 2px; font-size:8pt;}}"
                f"QPushButton:hover {{{hbg}}}"
            )
            btn.clicked.connect(lambda _, n=name: self._apply_preset(n))
            grid.addWidget(btn, idx // n_cols, idx % n_cols)
            self._preset_btns[name] = btn
        lay.addWidget(grp_pre)

        # ── Bodies ───────────────────────────────────────────────────────────
        grp_bod = QGroupBox("Bodies")
        bod_lay = QVBoxLayout(grp_bod)
        bod_lay.setSpacing(4)
        self._sl_mass = []
        for i in range(3):
            sl = _ParamSlider(f"Mass {i+1}  (M\u2609)", 0.001, 20.0,
                              self.mass[i], decimals=3)
            sl.valueChanged.connect(lambda v, idx=i: self._on_mass(idx, v))
            bod_lay.addWidget(sl)
            self._sl_mass.append(sl)
        lay.addWidget(grp_bod)

        # ── Configuration ──────────────────────────────────────────────────
        grp_cfg = QGroupBox("Configuration")
        cfg_lay = QVBoxLayout(grp_cfg)
        cfg_lay.setSpacing(4)

        self._sl_sep  = _ParamSlider("Separation Scale", 0.10, 3.0,
                                     self.sep_scale, decimals=2)
        self._sl_tilt = _ParamSlider("Tilt  \u00b0", 0.0, 180.0,
                                     self.tilt_deg, decimals=1)
        self._sl_coll = _ParamSlider("Collision Radius  (AU)", 0.01, 0.50,
                                     self.coll_radius, decimals=3)
        for sl, attr in [(self._sl_sep,  "sep_scale"),
                         (self._sl_tilt, "tilt_deg"),
                         (self._sl_coll, "coll_radius")]:
            sl.valueChanged.connect(lambda v, a=attr: setattr(self, a, v))
            cfg_lay.addWidget(sl)

        self._cb_coll = QCheckBox("Collision Detection")
        self._cb_coll.setChecked(self.use_collisions)
        self._cb_coll.stateChanged.connect(
            lambda s: setattr(self, "use_collisions", s == Qt.Checked))
        cfg_lay.addWidget(self._cb_coll)
        lay.addWidget(grp_cfg)

        # ── Integration ───────────────────────────────────────────────────
        grp_int = QGroupBox("Integration")
        int_lay = QVBoxLayout(grp_int)
        int_lay.setSpacing(4)

        self._sl_dur = _ParamSlider("Duration  (sim-yr)", 5.0, 500.0,
                                    self.t_end, decimals=1)
        self._sl_dur.valueChanged.connect(lambda v: setattr(self, "t_end", v))
        int_lay.addWidget(self._sl_dur)

        meth_row = QHBoxLayout()
        meth_lbl = QLabel("ODE Method")
        meth_lbl.setStyleSheet("color:#aaaacc; font-size:8pt;")
        meth_row.addWidget(meth_lbl)
        self._combo_method = QComboBox()
        self._combo_method.addItems(["DOP853", "RK45"])
        self._combo_method.setCurrentText(self.ode_method)
        self._combo_method.currentTextChanged.connect(
            lambda t: setattr(self, "ode_method", t))
        meth_row.addWidget(self._combo_method)
        int_lay.addLayout(meth_row)

        self._sl_tol = _ParamSlider("Tolerance  1e-N", 4, 12,
                                    float(self.rtol_exp), decimals=0)
        self._sl_tol.valueChanged.connect(
            lambda v: setattr(self, "rtol_exp", int(round(v))))
        int_lay.addWidget(self._sl_tol)
        lay.addWidget(grp_int)

        # ── Visual ────────────────────────────────────────────────────────
        grp_vis = QGroupBox("Visual")
        vis_lay = QVBoxLayout(grp_vis)
        vis_lay.setSpacing(4)

        self._sl_trail = _ParamSlider("Trail  %", 5.0, 100.0,
                                      self.trail_len, decimals=0)
        self._sl_trail.valueChanged.connect(
            lambda v: setattr(self, "trail_len", v))
        self._sl_speed = _ParamSlider("Speed", 1.0, 30.0,
                                      self.anim_speed, decimals=0)
        self._sl_speed.valueChanged.connect(
            lambda v: setattr(self, "anim_speed", v))
        vis_lay.addWidget(self._sl_trail)
        vis_lay.addWidget(self._sl_speed)

        self._cb_forces = QCheckBox("Show Force Vectors")
        self._cb_forces.setChecked(False)
        self._cb_forces.stateChanged.connect(
            lambda s: setattr(self, "_show_forces", s == Qt.Checked))
        vis_lay.addWidget(self._cb_forces)
        lay.addWidget(grp_vis)

        # ── Action buttons ────────────────────────────────────────────────
        self._btn_integrate = QPushButton("\u25b6   Integrate")
        self._btn_integrate.setStyleSheet(
            "QPushButton {background:#1a3a5e; color:#FFD700; font-size:10pt;"
            " font-weight:bold; padding:7px; border-radius:4px;}"
            "QPushButton:hover {background:#2a5a8e;}"
        )
        self._btn_integrate.clicked.connect(self._integrate_async)
        lay.addWidget(self._btn_integrate)

        self._btn_export = QPushButton("Export MP4")
        self._btn_export.setStyleSheet(
            "QPushButton {background:#1e2a1e; color:#ccccee; font-size:9pt;"
            " padding:5px; border-radius:4px;}"
            "QPushButton:hover {background:#2a4a2a;}"
        )
        self._btn_export.clicked.connect(self._on_export_click)
        lay.addWidget(self._btn_export)

        # ── Info label ────────────────────────────────────────────────────
        self._lbl_info = QLabel("")
        self._lbl_info.setAlignment(Qt.AlignCenter)
        self._lbl_info.setWordWrap(True)
        self._lbl_info.setStyleSheet(
            "color:#888aaa; font-size:8pt; padding:4px;")
        lay.addWidget(self._lbl_info)
        lay.addStretch()

    # ── Status slot (main thread, connected to _StatusBridge signal) ──────────
    def _on_status(self, msg: str):
        self._status_txt.set_text(msg)
        self._canvas.draw_idle()

    # ── Matplotlib body artists ───────────────────────────────────────────────
    def _build_body_artists(self):
        for lst in ([self._core_scat]
                    + [self._glow_scat[0], self._glow_scat[1]]
                    + self._trail_lines
                    + [self._label_txt]
                    + [self._force_lines, self._force_tips]):
            for art in lst:
                try:
                    art.remove()
                except Exception:
                    pass

        self._core_scat   = []
        self._glow_scat   = [[], []]
        self._trail_lines = []
        self._label_txt   = []
        self._force_lines = []
        self._force_tips  = []

        bodies = self._base_bodies or _figure8_bodies()
        self._active_mass = [b.mass for b in bodies]

        for bi, body in enumerate(bodies):
            col = body_color(body.mass)
            sz  = body_size(body.mass)
            x, y, z = body.pos

            sc  = self.ax3d.scatter([x], [y], [z], s=sz,     c=[col],
                                     alpha=0.94, depthshade=False, zorder=5)
            sg1 = self.ax3d.scatter([x], [y], [z], s=sz * 4,  c=[col],
                                     alpha=0.20, depthshade=False, zorder=4)
            sg2 = self.ax3d.scatter([x], [y], [z], s=sz * 10, c=[col],
                                     alpha=0.07, depthshade=False, zorder=3)
            self._core_scat.append(sc)
            self._glow_scat[0].append(sg1)
            self._glow_scat[1].append(sg2)

            lines = [
                self.ax3d.plot([], [], [], color=col,
                                alpha=float(self.TRAIL_ALPHAS[j]),
                                linewidth=0.9)[0]
                for j in range(self.N_TRAIL_SEGS)
            ]
            self._trail_lines.append(lines)

            txt = self.ax3d.text(
                x, y, z, f" {body_label(body.mass)}",
                fontsize=6.5, color=CTRL_FG,
                ha="left", va="bottom", alpha=0.75, zorder=6)
            self._label_txt.append(txt)

            # Force vector artists (hidden until checkbox toggled)
            fline = self.ax3d.plot([0, 0], [0, 0], [0, 0], color=col,
                                    linewidth=2.0, alpha=0.7)[0]
            ftip  = self.ax3d.scatter([0], [0], [0], s=40, c=[col],
                                       marker="^", alpha=0.85,
                                       depthshade=False, zorder=7)
            fline.set_visible(False)
            ftip.set_visible(False)
            self._force_lines.append(fline)
            self._force_tips.append(ftip)

    # ── Preset ───────────────────────────────────────────────────────────────
    def _apply_preset(self, name: str):
        if name not in PRESETS:
            return
        self._preset_name = name
        cfg  = PRESETS[name]
        base = cfg["bodies"]()
        self._base_bodies = base

        for i, body in enumerate(base[:3]):
            v = float(np.clip(body.mass, 0.001, 20.0))
            self.mass[i] = v
            self._sl_mass[i].set_value(v)

        self.sep_scale = 1.0;  self._sl_sep.set_value(1.0)
        self.tilt_deg  = 0.0;  self._sl_tilt.set_value(0.0)

        t_end = cfg.get("t_end", 20.0)
        self.t_end = t_end
        self._sl_dur.set_value(t_end)

        rtol = cfg.get("rtol", 1e-9)
        self.rtol_exp = int(round(-np.log10(rtol)))
        self._sl_tol.set_value(float(self.rtol_exp))

        method = cfg.get("method", "DOP853")
        self.ode_method = method
        idx = self._combo_method.findText(method)
        if idx >= 0:
            self._combo_method.setCurrentIndex(idx)

        cat = cfg["category"].upper()
        self._lbl_info.setText(f"[{cat}]  {cfg.get('info', '')}")

        self._build_body_artists()
        self._integrate_async()

    # ── Body construction (applies mass / sep / tilt overrides) ──────────────
    def _get_bodies(self):
        if self._base_bodies is None:
            return None
        sep   = self.sep_scale
        theta = np.radians(self.tilt_deg)
        Rx    = np.array([
            [1, 0,             0            ],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)],
        ])
        # Rescale velocities so bound orbits stay bound when mass is changed
        M_orig    = sum(b.mass for b in self._base_bodies)
        M_new     = sum(self.mass[:len(self._base_bodies)])
        vel_scale = np.sqrt(M_new / M_orig) if M_orig > 0 else 1.0

        bodies = []
        for i, base in enumerate(self._base_bodies):
            pos = Rx @ (base.pos * sep)
            vel = Rx @ (base.vel * vel_scale / np.sqrt(max(sep, 1e-6)))
            bodies.append(Body(self.mass[i], pos, vel, base.name))
        return bodies

    # ── Async integration (starts background thread) ──────────────────────────
    def _integrate_async(self):
        if self._integrating:
            return
        self._integrating  = True
        self._pending_traj = None
        self._bridge.message.emit("Computing\u2026")
        threading.Thread(target=self._integrate_worker, daemon=True).start()

    def _integrate_worker(self):
        bodies = self._get_bodies()
        if bodies is None:
            self._integrating = False
            return
        rtol = 10.0 ** (-self.rtol_exp)
        try:
            if self.use_collisions:
                result = integrate_with_collisions(
                    bodies,
                    t_end=self.t_end, n_out=4000,
                    rtol=rtol, atol=rtol * 0.1,
                    method=self.ode_method,
                    collision_radius=self.coll_radius,
                )
            else:
                ts, pos_arr, vel_arr = integrate(
                    bodies,
                    t_end=self.t_end, n_out=4000,
                    rtol=rtol, atol=rtol * 0.1,
                    method=self.ode_method,
                )
                result = (ts, pos_arr, vel_arr, [])   # empty merge list
            self._pending_traj = result               # adopted on next frame
        except Exception as exc:
            print(f"[Integration error] {exc}")
            self._bridge.message.emit(f"Error: {exc}")
            self._integrating = False

    # ── Mass slider callback (live colour/size update) ────────────────────────
    def _on_mass(self, idx, val):
        self.mass[idx] = val
        if idx < len(self._core_scat):
            col = body_color(val)
            sz  = body_size(val)
            self._core_scat[idx].set_facecolor([col])
            self._core_scat[idx].set_sizes([sz])
            self._glow_scat[0][idx].set_facecolor([col])
            self._glow_scat[0][idx].set_sizes([sz * 4])
            self._glow_scat[1][idx].set_facecolor([col])
            self._glow_scat[1][idx].set_sizes([sz * 10])
            for ln in self._trail_lines[idx]:
                ln.set_color(col)
            self._label_txt[idx].set_text(f" {body_label(val)}")
            if idx < len(self._force_lines):
                self._force_lines[idx].set_color(col)
                self._force_tips[idx].set_facecolor([col])

    # ── View seed (called once when a new trajectory is adopted) ─────────────
    def _autoscale(self):
        if self._traj is None:
            return
        ts, pos_arr, vel_arr, merges = self._traj
        pos0  = pos_arr[0]
        valid = ~np.any(np.isnan(pos0), axis=1)
        pts   = pos0[valid]
        if len(pts) == 0:
            return
        center = pts.mean(axis=0)
        radius = (float(np.linalg.norm(pts - center, axis=1).max()) * 2.2
                  if len(pts) > 1 else 0.5)
        radius = max(radius, 0.3)
        self._view_radius = radius
        self._view_center = center.copy()
        c, r = self._view_center, self._view_radius
        self.ax3d.set_xlim3d([c[0] - r, c[0] + r])
        self.ax3d.set_ylim3d([c[1] - r, c[1] + r])
        self.ax3d.set_zlim3d([c[2] - r, c[2] + r])

    # ── Per-frame smooth camera ───────────────────────────────────────────────
    def _dynamic_view(self, pos_arr, fi):
        """Track active bodies with asymmetric EMA; ignore escaped outliers."""
        visible = []
        for bi in range(min(len(self._core_scat), pos_arr.shape[1])):
            x, y, z = pos_arr[fi, bi]
            if not np.isnan(x):
                visible.append([x, y, z])
        if not visible:
            return

        pts      = np.array(visible)
        centroid = pts.mean(axis=0)

        if len(pts) >= 2:
            dists    = np.linalg.norm(pts - centroid, axis=1)
            d_sorted = np.sort(dists)
            # One body escaped if it is ≥3× farther than the next-farthest
            if len(pts) >= 3 and d_sorted[-1] >= 3.0 * max(d_sorted[-2], 1e-9):
                inliers    = pts[dists < d_sorted[-1]]
                center_now = inliers.mean(axis=0)
                span       = np.linalg.norm(inliers - center_now, axis=1).max()
                target_r   = float(max(span * 2.2, 0.3))
            else:
                center_now = centroid
                target_r   = float(max(d_sorted[-1] * 1.9, 0.3))
        else:
            center_now = centroid
            target_r   = self._view_radius      # single body — hold zoom

        # Fast zoom-out, slow zoom-in
        alpha_r = 0.08 if target_r > self._view_radius else 0.02
        self._view_radius += alpha_r * (target_r - self._view_radius)
        self._view_center += 0.04  * (center_now - self._view_center)

        c, r = self._view_center, self._view_radius
        self.ax3d.set_xlim3d([c[0] - r, c[0] + r])
        self.ax3d.set_ylim3d([c[1] - r, c[1] + r])
        self.ax3d.set_zlim3d([c[2] - r, c[2] + r])

    # ── Animation frame (called by QTimer on the main thread) ─────────────────
    def _update_frame(self):
        # ── Adopt pending trajectory ──────────────────────────────────────
        if self._pending_traj is not None:
            self._traj         = self._pending_traj
            self._pending_traj = None
            self._integrating  = False
            self._frame_idx    = 0
            self._merge_flash  = 0
            # Reset all artist visibility
            for bi in range(len(self._core_scat)):
                for art in [self._core_scat[bi],
                             self._glow_scat[0][bi], self._glow_scat[1][bi],
                             self._label_txt[bi]]:
                    art.set_visible(True)
            # Reset colours to base IC
            for bi, body in enumerate(self._base_bodies or []):
                self._active_mass[bi] = body.mass
                col = body_color(body.mass)
                sz  = body_size(body.mass)
                self._core_scat[bi].set_facecolor([col])
                self._core_scat[bi].set_sizes([sz])
                self._glow_scat[0][bi].set_facecolor([col])
                self._glow_scat[0][bi].set_sizes([sz * 4])
                self._glow_scat[1][bi].set_facecolor([col])
                self._glow_scat[1][bi].set_sizes([sz * 10])
                for ln in self._trail_lines[bi]:
                    ln.set_color(col)
                self._label_txt[bi].set_text(f" {body_label(body.mass)}")
            self._status_txt.set_text("")
            self._autoscale()

        if self._integrating or self._traj is None:
            self._canvas.draw_idle()
            return

        ts, pos_arr, vel_arr, merges = self._traj
        n = len(ts)
        if n == 0:
            self._canvas.draw_idle()
            return

        speed           = max(1, int(self.anim_speed))
        prev_fi         = self._frame_idx
        self._frame_idx = (self._frame_idx + speed) % n
        fi              = self._frame_idx

        # ── Time display ──────────────────────────────────────────────────
        self._time_txt.set_text(f"t = {ts[fi] * _YR:.3f} yr")

        # ── Merge flash counter ───────────────────────────────────────────
        if self._merge_flash > 0:
            self._merge_flash -= 1
            if self._merge_flash == 0:
                self._merge_txt.set_text("")

        # ── Detect merge events crossing this frame ───────────────────────
        t_prev = ts[prev_fi]
        t_now  = ts[fi]
        if fi < prev_fi:          # wrap-around
            t_prev = -1e18
        for t_m, absorber_oi, absorbed_oi, new_mass in merges:
            if t_prev < t_m <= t_now and absorber_oi < len(self._core_scat):
                lbl = body_label(new_mass)
                self._merge_txt.set_text(
                    f"\u26a1 COLLISION\n"
                    f"Body {absorber_oi+1} + Body {absorbed_oi+1} \u2192 {lbl}\n"
                    f"m = {new_mass:.3g} M\u2609"
                )
                self._merge_flash = 80
                if self._active_mass[absorber_oi] != new_mass:
                    self._active_mass[absorber_oi] = new_mass
                    col = body_color(new_mass)
                    sz  = body_size(new_mass)
                    self._core_scat[absorber_oi].set_facecolor([col])
                    self._core_scat[absorber_oi].set_sizes([sz])
                    self._glow_scat[0][absorber_oi].set_facecolor([col])
                    self._glow_scat[0][absorber_oi].set_sizes([sz * 4])
                    self._glow_scat[1][absorber_oi].set_facecolor([col])
                    self._glow_scat[1][absorber_oi].set_sizes([sz * 10])
                    for ln in self._trail_lines[absorber_oi]:
                        ln.set_color(col)
                    self._label_txt[absorber_oi].set_text(f" {lbl}")

        # ── Update body artists ───────────────────────────────────────────
        trail_pts = max(2, int(n * self.trail_len / 100.0))
        start_idx = max(0, fi - trail_pts)

        for bi in range(len(self._core_scat)):
            if bi >= pos_arr.shape[1]:
                break
            x, y, z = pos_arr[fi, bi]

            if np.isnan(x):
                for art in [self._core_scat[bi],
                             self._glow_scat[0][bi], self._glow_scat[1][bi],
                             self._label_txt[bi]]:
                    art.set_visible(False)
                for ln in self._trail_lines[bi]:
                    ln.set_data_3d([], [], [])
                continue

            for art in [self._core_scat[bi],
                         self._glow_scat[0][bi], self._glow_scat[1][bi],
                         self._label_txt[bi]]:
                art.set_visible(True)

            xa, ya, za = np.array([x]), np.array([y]), np.array([z])
            self._core_scat[bi]._offsets3d    = (xa, ya, za)
            self._glow_scat[0][bi]._offsets3d = (xa, ya, za)
            self._glow_scat[1][bi]._offsets3d = (xa, ya, za)

            try:
                self._label_txt[bi].set_position((x, y))
                self._label_txt[bi].set_3d_properties(
                    z + 0.05 * abs(z) + 0.1, "z")
            except Exception:
                pass

            trail_pos  = pos_arr[start_idx:fi + 1, bi]
            valid_mask = ~np.any(np.isnan(trail_pos), axis=1)
            trail_pos  = trail_pos[valid_mask]
            K = len(trail_pos)
            if K > 1:
                seg_pts = max(1, K // self.N_TRAIL_SEGS)
                for j, ln in enumerate(self._trail_lines[bi]):
                    s = j * seg_pts
                    e = min((j + 1) * seg_pts + 1, K)
                    if s < K:
                        ln.set_data_3d(trail_pos[s:e, 0],
                                       trail_pos[s:e, 1],
                                       trail_pos[s:e, 2])
                    else:
                        ln.set_data_3d([], [], [])
            else:
                for ln in self._trail_lines[bi]:
                    ln.set_data_3d([], [], [])

        # ── Force vectors ─────────────────────────────────────────────
        if self._show_forces and self._traj is not None and len(self._force_lines) > 0:
            masses_now = np.array(self._active_mass[:pos_arr.shape[1]])
            pos_now    = pos_arr[fi]
            forces     = compute_forces(pos_now, masses_now)
            valid_f    = ~np.isnan(forces[:, 0])
            f_norms    = np.linalg.norm(forces[valid_f], axis=1)
            max_f      = f_norms.max() if len(f_norms) > 0 and f_norms.max() > 1e-15 else 1.0
            arrow_scale = 0.25 * self._view_radius / max_f

            for bi in range(len(self._force_lines)):
                if bi >= pos_arr.shape[1] or np.isnan(pos_now[bi, 0]):
                    self._force_lines[bi].set_visible(False)
                    self._force_tips[bi].set_visible(False)
                    continue
                fv  = forces[bi] * arrow_scale
                p   = pos_now[bi]
                tip = p + fv
                self._force_lines[bi].set_data_3d(
                    [p[0], tip[0]], [p[1], tip[1]], [p[2], tip[2]])
                self._force_tips[bi]._offsets3d = (
                    np.array([tip[0]]), np.array([tip[1]]), np.array([tip[2]]))
                self._force_lines[bi].set_visible(True)
                self._force_tips[bi].set_visible(True)
        else:
            for bi in range(len(self._force_lines)):
                self._force_lines[bi].set_visible(False)
                self._force_tips[bi].set_visible(False)

        # ── Smooth live camera ────────────────────────────────────────────
        self._dynamic_view(pos_arr, fi)
        self._canvas.draw_idle()

    # ── MP4 Export ────────────────────────────────────────────────────────────
    def _on_export_click(self):
        if self._traj is None:
            self._bridge.message.emit("No trajectory \u2014 click Integrate first")
            return
        if self._exporting:
            return
        self._exporting = True
        self._bridge.message.emit("Exporting MP4\u2026")
        threading.Thread(target=self._export_worker, daemon=True).start()

    def _export_worker(self):
        try:
            from matplotlib.animation import FFMpegWriter
            from matplotlib.figure    import Figure as _Fig
        except Exception:
            self._bridge.message.emit("FFMpeg unavailable \u2014 install ffmpeg")
            self._exporting = False
            return
        try:
            import matplotlib.pyplot as _plt
            ts, pos_arr, vel_arr, merges = self._traj
            n, N_orig = len(ts), pos_arr.shape[1]

            # Snapshot view parameters (read-only, safe from this thread)
            elev = float(self.ax3d.elev)
            azim = float(self.ax3d.azim)
            xlim = tuple(self.ax3d.get_xlim3d())
            ylim = tuple(self.ax3d.get_ylim3d())
            zlim = tuple(self.ax3d.get_zlim3d())

            fig_e = _Fig(figsize=(16, 9), dpi=120)
            fig_e.patch.set_facecolor(DARK_BG)
            ax_e  = fig_e.add_subplot(111, projection="3d")
            ax_e.set_facecolor(DARK_BG)
            for attr in ("xaxis", "yaxis", "zaxis"):
                pane = getattr(ax_e, attr).pane
                pane.fill = False
                pane.set_edgecolor("none")
            ax_e.grid(False)
            ax_e.tick_params(axis="both", colors="#445", labelsize=6,
                             length=3, width=0.5, pad=1)
            for attr in ("xaxis", "yaxis", "zaxis"):
                axis = getattr(ax_e, attr)
                axis.line.set_color("#334")
                axis.line.set_linewidth(0.6)
            ax_e.set_xlabel("x  (AU)", fontsize=7.5)
            ax_e.set_ylabel("y  (AU)", fontsize=7.5)
            ax_e.set_zlabel("z  (AU)", fontsize=7.5)
            ax_e.view_init(elev=elev, azim=azim)
            ax_e.set_xlim3d(xlim)
            ax_e.set_ylim3d(ylim)
            ax_e.set_zlim3d(zlim)

            time_txt_e = ax_e.text2D(
                0.02, 0.96, "t = 0.000 yr", ha="left", va="top",
                color=CTRL_FG, fontsize=9.5, transform=ax_e.transAxes)

            bodies_ic  = self._base_bodies or _figure8_bodies()
            scat_e     = []
            lines_e    = []
            active_m_e = [b.mass for b in bodies_ic]

            for bi in range(N_orig):
                m   = bodies_ic[bi].mass if bi < len(bodies_ic) else 1.0
                col = body_color(m)
                sz  = body_size(m)
                sc  = ax_e.scatter([], [], [], s=sz,     c=[col],
                                    alpha=0.94, depthshade=False)
                sg1 = ax_e.scatter([], [], [], s=sz * 4,  c=[col],
                                    alpha=0.20, depthshade=False)
                sg2 = ax_e.scatter([], [], [], s=sz * 10, c=[col],
                                    alpha=0.07, depthshade=False)
                scat_e.append((sc, sg1, sg2))
                lns = [
                    ax_e.plot([], [], [], color=col,
                               alpha=float(self.TRAIL_ALPHAS[j]), lw=0.9)[0]
                    for j in range(self.N_TRAIL_SEGS)
                ]
                lines_e.append(lns)

            fps      = 30
            n_frames = min(n, 1200)
            stride   = max(1, n // n_frames)
            frame_ix = list(range(0, n, stride))

            stamp     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = (self._preset_name
                         .replace("\n", "").replace(" ", "_")
                         .replace("(", "").replace(")", ""))
            fname     = f"three_body_{safe_name}_{stamp}.mp4"

            writer    = FFMpegWriter(fps=fps,
                                      metadata={"title": "Three-Body Problem"},
                                      bitrate=4000)
            trail_pts = max(2, int(n * self.trail_len / 100.0))

            with writer.saving(fig_e, fname, dpi=120):
                for out_i, fi in enumerate(frame_ix):
                    start_idx = max(0, fi - trail_pts)
                    t_now     = ts[fi]
                    time_txt_e.set_text(f"t = {t_now * _YR:.3f} yr")

                    for t_m, absorber_oi, absorbed_oi, new_mass in merges:
                        if t_m <= t_now and active_m_e[absorber_oi] != new_mass:
                            active_m_e[absorber_oi] = new_mass
                            col = body_color(new_mass)
                            sz  = body_size(new_mass)
                            sc, sg1, sg2 = scat_e[absorber_oi]
                            sc.set_facecolor([col]);  sc.set_sizes([sz])
                            sg1.set_facecolor([col]); sg1.set_sizes([sz * 4])
                            sg2.set_facecolor([col]); sg2.set_sizes([sz * 10])
                            for ln in lines_e[absorber_oi]:
                                ln.set_color(col)

                    for bi in range(N_orig):
                        x, y, z      = pos_arr[fi, bi]
                        sc, sg1, sg2 = scat_e[bi]
                        if np.isnan(x):
                            for art in (sc, sg1, sg2):
                                art.set_visible(False)
                            for ln in lines_e[bi]:
                                ln.set_data_3d([], [], [])
                            continue
                        for art in (sc, sg1, sg2):
                            art.set_visible(True)
                        xa = np.array([x]); ya = np.array([y]); za = np.array([z])
                        sc._offsets3d  = (xa, ya, za)
                        sg1._offsets3d = (xa, ya, za)
                        sg2._offsets3d = (xa, ya, za)

                        trail_pos  = pos_arr[start_idx:fi + 1, bi]
                        valid_mask = ~np.any(np.isnan(trail_pos), axis=1)
                        trail_pos  = trail_pos[valid_mask]
                        K = len(trail_pos)
                        if K > 1:
                            seg_pts = max(1, K // self.N_TRAIL_SEGS)
                            for j, ln in enumerate(lines_e[bi]):
                                s = j * seg_pts
                                e = min((j + 1) * seg_pts + 1, K)
                                if s < K:
                                    ln.set_data_3d(trail_pos[s:e, 0],
                                                   trail_pos[s:e, 1],
                                                   trail_pos[s:e, 2])
                                else:
                                    ln.set_data_3d([], [], [])
                        else:
                            for ln in lines_e[bi]:
                                ln.set_data_3d([], [], [])

                    if out_i % max(1, len(frame_ix) // 20) == 0:
                        pct = int(100 * out_i / len(frame_ix))
                        self._bridge.message.emit(f"Exporting\u2026 {pct}%")

                    writer.grab_frame()

            _plt.close(fig_e)
            self._bridge.message.emit(f"\u2713 Saved: {fname}")
        except Exception as exc:
            self._bridge.message.emit(f"Export error: {exc}")
            raise
        finally:
            self._exporting = False


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(_QSS)
    win = ThreeBodyApp()
    win.show()
    sys.exit(app.exec_())
