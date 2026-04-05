#!/usr/bin/env python3
"""
Lissajous Explorer
==================
Physical simulation of the classic double-pendulum Lissajous drawing machine.

The original instrument used two pendulums swinging at right angles — one
controlling horizontal motion, one vertical — with a light or stylus tracing
the combined path onto paper or sand.

Equations
---------
  x(t) = Ax · exp(-γ·t/T) · sin(ωx·t + δ)
  y(t) = Ay · exp(-γ·t/T) · sin(ωy·t)

where T is the total trace duration, so damping is always relative.
Integer ratios ωx:ωy create closed (Lissajous) curves.
Non-integer ratios produce slowly-rotating open paths.

Controls
--------
  Frequencies  : ωx, ωy — integer ratios create closed curves
  Phase        : δ (0–360°) — shifts/rotates the pattern
  Amplitudes   : Ax, Ay — relative reach of each pendulum
  Damping γ    : 0 = ideal frictionless pendulum; >0 = decay spiral
  Cycles       : how many periods to trace
  Visual style : line width, alpha, gradient palette
  Presets      : 8 classic ratio configurations
  Animate trace: watch the pen trace the path in real time; speed adjustable

Usage
-----
    conda run -n maths_fun python3 lissajous_explorer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
from matplotlib.animation import FuncAnimation

# ── Colour scheme ─────────────────────────────────────────────────────────────

CTRL_BG = "#12122a"
CTRL_FG = "#ccccee"
ACC_C   = "#7788ff"
GOLD    = "#FFD700"
BTN_OFF = "#1a1a3e"
BTN_ON  = "#1a3a50"
DARK_BG = "#07070f"

# Export sizes mirroring spirograph.py
EXPORT_SIZES = [
    ("1080p",  1920, 1080),
    ("1440p",  2560, 1440),
    ("4K",     3840, 2160),
    ("Sq 2K",  2160, 2160),
    ("Sq 4K",  4096, 4096),
]


class LissajousExplorer:

    PALETTES = [
        "plasma", "viridis", "inferno", "magma",
        "turbo", "rainbow", "cool", "twilight",
    ]

    # (label, ωx, ωy, δ°, cycles)
    PRESETS = [
        ("1:1",   1, 1, 90, 1),
        ("1:2",   1, 2, 90, 2),
        ("1:3",   1, 3, 90, 3),
        ("2:3",   2, 3, 90, 3),
        ("3:4",   3, 4, 90, 4),
        ("3:5",   3, 5, 90, 5),
        ("4:5",   4, 5, 90, 5),
        ("5:6",   5, 6, 90, 6),
    ]

    DEFAULTS = dict(
        omega_x     = 3,
        omega_y     = 2,
        phase       = 90.0,
        amp_x       = 1.0,
        amp_y       = 1.0,
        damping     = 0.0,
        cycles      = 3,
        n_points    = 3000,
        line_width  = 1.2,
        line_alpha  = 0.90,
        anim_speed  = 30,
        palette     = "plasma",
        reverse_pal = False,
    )

    def __init__(self):
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)

        self._batch     = False
        self._anim      = None
        self._anim_idx  = 0

        # Computation cache — keyed by physics params so visual-only
        # changes (line_width, alpha, palette) skip the numpy work.
        self._cache_key = None
        self._xs = self._ys = self._ts = None
        # Pre-built segment arrays for animation (set at anim start)
        self._segs = self._ct = None

        self._build_figure()
        self._draw()

    # ── Figure layout ──────────────────────────────────────────────────────────

    def _build_figure(self):
        plt.rcParams.update({
            "figure.facecolor": CTRL_BG,
            "axes.facecolor":   CTRL_BG,
            "axes.edgecolor":   "#334",
            "text.color":       CTRL_FG,
            "xtick.color":      "#556",
            "ytick.color":      "#556",
        })

        self.fig = plt.figure(figsize=(15, 9), facecolor=CTRL_BG)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Lissajous Explorer")

        outer = gridspec.GridSpec(
            1, 2, width_ratios=[3, 1],
            left=0.03, right=0.98, top=0.94, bottom=0.04, wspace=0.04,
        )

        # ── Canvas ────────────────────────────────────────────────────────────
        self.ax = self.fig.add_subplot(outer[0], facecolor=DARK_BG)
        self.ax.set_aspect("equal")
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.title = self.fig.text(
            0.38, 0.96, "", ha="center", va="top",
            color=CTRL_FG, fontsize=10,
        )

        # ── Control panel ─────────────────────────────────────────────────────
        heights = [
            0.4,              # "Pendulum Frequencies" header   [0]
            1.3, 1.3,         # ωx, ωy                          [1,2]
            0.4,              # "Phase & Amplitude" header        [3]
            1.3, 1.3, 1.3,   # phase, Ax, Ay                    [4,5,6]
            0.4,              # "Pendulum Physics" header         [7]
            1.3, 1.3, 1.3,   # damping, cycles, n_points         [8,9,10]
            0.4, 1.3, 1.3,   # "Visual Style" header + lw, alpha [11,12,13]
            0.4, 1.6,        # "Presets" header + 4×2 grid       [14,15]
            0.9,             # animate trace button              [16]
            1.3,             # anim speed slider                 [17]
            0.4, 1.6,        # "Export" header + 2 rows buttons  [18,19]
            0.4, 4.8,        # "Palette" header + radio          [20,21]
            0.8,             # reverse button                    [22]
        ]
        ctrl = gridspec.GridSpecFromSubplotSpec(
            len(heights), 1, subplot_spec=outer[1],
            hspace=0.2, height_ratios=heights,
        )

        def hdr(row, text):
            a = self.fig.add_subplot(ctrl[row], facecolor=CTRL_BG)
            a.set_axis_off()
            a.text(0.5, 0.5, text, ha="center", va="center",
                   color=GOLD, fontsize=9, fontweight="bold")

        def plain_btn(ax, text, active=False):
            b = Button(ax, text, color=BTN_ON if active else BTN_OFF,
                       hovercolor="#223355")
            b.label.set_color(CTRL_FG)
            b.label.set_fontsize(8)
            return b

        # ── Frequencies ───────────────────────────────────────────────────────
        hdr(0, "─── Pendulum Frequencies ───")
        self.sl_wx, self.tb_wx = self._slider_tb_row(
            ctrl, 1, "ωx", 1, 12, self.omega_x, "%d", int, "omega_x")
        self.sl_wy, self.tb_wy = self._slider_tb_row(
            ctrl, 2, "ωy", 1, 12, self.omega_y, "%d", int, "omega_y")

        # ── Phase & Amplitudes ────────────────────────────────────────────────
        hdr(3, "─── Phase & Amplitude ───")
        self.sl_phase, self.tb_phase = self._slider_tb_row(
            ctrl, 4, "δ (°)", 0.0, 360.0, self.phase, "%.1f", float, "phase")
        self.sl_ax, self.tb_ax = self._slider_tb_row(
            ctrl, 5, "Ax", 0.1, 2.0, self.amp_x, "%.2f", float, "amp_x")
        self.sl_ay, self.tb_ay = self._slider_tb_row(
            ctrl, 6, "Ay", 0.1, 2.0, self.amp_y, "%.2f", float, "amp_y")

        # ── Pendulum physics ──────────────────────────────────────────────────
        hdr(7, "─── Pendulum Physics ───")
        self.sl_damp, self.tb_damp = self._slider_tb_row(
            ctrl, 8, "Damping γ", 0.0, 8.0, self.damping, "%.2f", float, "damping")
        self.sl_cyc, self.tb_cyc = self._slider_tb_row(
            ctrl, 9, "Cycles", 1, 20, self.cycles, "%d", int, "cycles")
        self.sl_n, self.tb_n = self._slider_tb_row(
            ctrl, 10, "N pts", 500, 20000, self.n_points, "%d", int, "n_points")

        # ── Visual style ──────────────────────────────────────────────────────
        hdr(11, "─── Visual Style ───")
        self.sl_lw, self.tb_lw = self._slider_tb_row(
            ctrl, 12, "Line w", 0.3, 6.0, self.line_width, "%.1f", float, "line_width")
        self.sl_la, self.tb_la = self._slider_tb_row(
            ctrl, 13, "Alpha", 0.1, 1.0, self.line_alpha, "%.2f", float, "line_alpha")

        # ── Presets (4×2 grid) ────────────────────────────────────────────────
        hdr(14, "─── Presets ───")
        ax_pre = self.fig.add_subplot(ctrl[15], facecolor=CTRL_BG)
        ax_pre.set_axis_off()
        self._preset_btns = []
        for i, (lbl, ox, oy, ph, cyc) in enumerate(self.PRESETS):
            col = i % 4
            row = i // 4
            x0 = col / 4 + 0.005
            y0 = (1 - row) / 2 + 0.04
            a = ax_pre.inset_axes([x0, y0, 0.235, 0.44])
            b = plain_btn(a, lbl)
            b.label.set_fontsize(7.5)
            b.on_clicked(lambda _, ox=ox, oy=oy, ph=ph, cyc=cyc:
                         self._apply_preset(ox, oy, ph, cyc))
            self._preset_btns.append(b)

        # ── Animate trace ─────────────────────────────────────────────────────
        ax_ab = self.fig.add_subplot(ctrl[16], facecolor=CTRL_BG)
        ax_ab.set_axis_off()
        ax_anim = ax_ab.inset_axes([0.1, 0.05, 0.80, 0.90])
        self._btn_anim = Button(ax_anim, "▶  Animate Trace",
                                color=BTN_OFF, hovercolor="#223355")
        self._btn_anim.label.set_color(GOLD)
        self._btn_anim.label.set_fontsize(8)
        self._btn_anim.on_clicked(self._toggle_anim)

        self.sl_speed, self.tb_speed = self._slider_tb_row(
            ctrl, 17, "Speed", 1, 200, self.anim_speed, "%d", int, "anim_speed",
            draw=False)   # read live during animation — must not interrupt it

        # ── Export ────────────────────────────────────────────────────────────
        hdr(18, "─── Export PNG ───")
        ax_exp = self.fig.add_subplot(ctrl[19], facecolor=CTRL_BG)
        ax_exp.set_axis_off()
        self._export_btns = []
        # Row 1: first 3 sizes; Row 2: last 2 sizes
        rows = [EXPORT_SIZES[:3], EXPORT_SIZES[3:]]
        for ri, row_sizes in enumerate(rows):
            n = len(row_sizes)
            for ci, (lbl, ew, eh) in enumerate(row_sizes):
                x0 = ci / n + 0.005
                y0 = (1 - ri) / 2 + 0.04
                a = ax_exp.inset_axes([x0, y0, 0.99 / n - 0.01, 0.44])
                b = plain_btn(a, lbl)
                b.label.set_fontsize(7)
                b.on_clicked(lambda _, ew=ew, eh=eh: self._export(ew, eh))
                self._export_btns.append(b)

        # ── Palette ───────────────────────────────────────────────────────────
        hdr(20, "─── Palette ───")
        ax_pal = self.fig.add_subplot(ctrl[21], facecolor=CTRL_BG)
        self.radio_pal = RadioButtons(
            ax_pal, self.PALETTES,
            active=self.PALETTES.index(self.palette),
            activecolor=GOLD,
        )
        for lbl in self.radio_pal.labels:
            lbl.set_color(CTRL_FG)
            lbl.set_fontsize(8.5)
        self.radio_pal.on_clicked(lambda lbl: self._set(palette=lbl))

        ax_rev = self.fig.add_subplot(ctrl[22], facecolor=CTRL_BG)
        ax_rev.set_axis_off()
        ax_rb = ax_rev.inset_axes([0.20, 0.1, 0.60, 0.8])
        self._btn_rev = plain_btn(ax_rb, "Reverse palette")
        self._btn_rev.label.set_fontsize(7.5)
        self._btn_rev.on_clicked(lambda _: self._toggle_reverse())

    # ── Slider + TextBox factory ───────────────────────────────────────────────

    def _slider_tb_row(self, ctrl, row, label, vmin, vmax, vinit, fmt, cast, attr,
                       draw=True):
        """One GridSpec row: 62% slider + 33% TextBox. Returns (slider, textbox).

        draw=False: only update the attribute (used for live-read params like
        anim_speed that do not require a redraw and must not interrupt animation).
        """
        ax_row = self.fig.add_subplot(ctrl[row], facecolor=CTRL_BG)
        ax_row.set_axis_off()
        ax_sl = ax_row.inset_axes([0.00, 0.05, 0.62, 0.90])
        ax_tb = ax_row.inset_axes([0.65, 0.10, 0.33, 0.80])

        sl = Slider(ax_sl, label, vmin, vmax, valinit=vinit,
                    color=ACC_C, track_color="#334")
        sl.label.set_color(CTRL_FG);   sl.label.set_fontsize(8)
        sl.valtext.set_color(CTRL_FG); sl.valtext.set_fontsize(7)

        tb = TextBox(ax_tb, "", initial=fmt % vinit, textalignment="center")
        tb.text_disp.set_color(CTRL_FG)
        tb.ax.set_facecolor("#1a1a2e")

        def on_sl_changed(val):
            setattr(self, attr, cast(val))
            if not self._batch:
                tb.set_val(fmt % cast(val))
                if draw:
                    self._draw()

        def on_tb_submit(text):
            try:
                v = float(text.strip())
                v = max(vmin, min(vmax, v))
                sl.set_val(cast(v))         # triggers on_sl_changed → redraw
            except ValueError:
                tb.set_val(fmt % getattr(self, attr))   # revert bad input

        sl.on_changed(on_sl_changed)
        tb.on_submit(on_tb_submit)
        return sl, tb

    # ── Physics ────────────────────────────────────────────────────────────────

    def _cache_key_now(self):
        return (self.omega_x, self.omega_y, self.phase,
                self.amp_x, self.amp_y, self.damping,
                self.cycles, self.n_points)

    def _compute(self):
        """Return (x, y, t) — cached when physics params are unchanged."""
        key = self._cache_key_now()
        if key != self._cache_key:
            T   = 2 * np.pi * self.cycles
            t   = np.linspace(0, T, int(self.n_points))
            env = np.exp(-self.damping * t / max(T, 1e-12))
            ph  = np.radians(self.phase)
            self._xs = self.amp_x * env * np.sin(self.omega_x * t + ph)
            self._ys = self.amp_y * env * np.sin(self.omega_y * t)
            self._ts = t
            self._cache_key = key
            self._segs = self._ct = None   # invalidate animation buffers
        return self._xs, self._ys, self._ts

    def _ensure_anim_buffers(self):
        """Pre-build segment + colour arrays once per physics config."""
        if self._segs is not None:
            return
        x, y, t = self._compute()
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        self._segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        self._ct   = t[:-1] / t[-1]

    # ── Static draw ────────────────────────────────────────────────────────────

    def _setup_ax(self):
        self.ax.cla()
        self.ax.set_facecolor(DARK_BG)
        self.ax.set_aspect("equal")
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        mx = max(self.amp_x, self.amp_y) * 1.08
        self.ax.set_xlim(-mx, mx)
        self.ax.set_ylim(-mx, mx)
        return mx

    def _make_lc(self, segs, c_t):
        cname = self.palette + ("_r" if self.reverse_pal else "")
        lc = LineCollection(segs, cmap=cname, norm=plt.Normalize(0, 1),
                            linewidth=self.line_width, alpha=self.line_alpha,
                            capstyle="round")
        lc.set_array(c_t)
        return lc

    def _update_title(self):
        ratio = f"ωx={self.omega_x}  ωy={self.omega_y}  δ={self.phase:.1f}°"
        damp  = f"   γ={self.damping:.2f}" if self.damping > 1e-4 else ""
        self.title.set_text(f"Lissajous  ·  {ratio}{damp}")

    def _draw(self):
        if self._batch:
            return
        # Stop any running animation so we return to static view
        if self._anim is not None:
            self._stop_anim()

        x, y, t = self._compute()
        self._setup_ax()

        pts  = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        c_t  = t[:-1] / t[-1]

        self.ax.add_collection(self._make_lc(segs, c_t))
        self._update_title()
        self.fig.canvas.draw_idle()

    # ── Control helpers ────────────────────────────────────────────────────────

    def _set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self._draw()

    def _apply_preset(self, ox, oy, phase, cycles):
        self._batch = True
        self.sl_wx.set_val(ox);          self.omega_x = ox
        self.sl_wy.set_val(oy);          self.omega_y = oy
        self.sl_phase.set_val(phase);    self.phase   = float(phase)
        self.sl_cyc.set_val(cycles);     self.cycles  = cycles
        self.tb_wx.set_val("%d" % ox)
        self.tb_wy.set_val("%d" % oy)
        self.tb_phase.set_val("%.1f" % phase)
        self.tb_cyc.set_val("%d" % cycles)
        self._batch = False
        self._draw()

    def _toggle_reverse(self):
        self.reverse_pal = not self.reverse_pal
        self._draw()

    # ── Trace animation ────────────────────────────────────────────────────────

    def _stop_anim(self):
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None
        self._btn_anim.label.set_text("▶  Animate Trace")
        self._btn_anim.label.set_color(GOLD)

    def _toggle_anim(self, _):
        if self._anim is not None:
            self._stop_anim()
            self._draw()      # restore full static trace
            return

        # ── start animation ───────────────────────────────────────────────────
        self._compute()            # ensure cache is warm
        self._ensure_anim_buffers()
        self._anim_idx = 0

        self._setup_ax()

        cname = self.palette + ("_r" if self.reverse_pal else "")

        # Growing trace — updated in-place each frame
        self._anim_lc = LineCollection(
            [], cmap=cname, norm=plt.Normalize(0, 1),
            linewidth=self.line_width, alpha=self.line_alpha, capstyle="round",
        )
        self.ax.add_collection(self._anim_lc)

        # Pen dot at current position
        self._anim_dot, = self.ax.plot(
            [], [], "o", color="white", markersize=5, zorder=10,
            markeredgewidth=0,
        )

        self._update_title()
        self._btn_anim.label.set_text("◼  Stop")
        self._btn_anim.label.set_color("#ff6644")

        n_segs = len(self._segs)

        def _step(_frame):
            k = min(self._anim_idx + self.anim_speed, n_segs)
            self._anim_idx = k

            self._anim_lc.set_segments(self._segs[:k])
            self._anim_lc.set_array(self._ct[:k])

            if k < n_segs:
                px, py = self._xs[k], self._ys[k]
                self._anim_dot.set_data([px], [py])
            else:
                self._anim_dot.set_data([], [])
                self._anim_idx = 0   # loop

            return self._anim_lc, self._anim_dot

        self._anim = FuncAnimation(
            self.fig, _step, interval=30,
            blit=True, cache_frame_data=False,
        )
        self.fig.canvas.draw_idle()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self, W, H):
        # figsize in inches at 100 dpi → exactly W×H pixels
        fig2 = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=DARK_BG)
        ax2  = fig2.add_subplot(111, facecolor=DARK_BG)
        ax2.set_aspect("equal")
        ax2.set_axis_off()

        x, y, t = self._compute()
        pts  = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        c_t  = t[:-1] / t[-1]
        lw   = max(1.0, self.line_width * min(W, H) / 900)   # scale lw to output size

        cname = self.palette + ("_r" if self.reverse_pal else "")
        lc = LineCollection(segs, cmap=cname, norm=plt.Normalize(0, 1),
                            linewidth=lw, alpha=self.line_alpha,
                            capstyle="round")
        lc.set_array(c_t)
        ax2.add_collection(lc)
        mx = max(self.amp_x, self.amp_y) * 1.05
        ax2.set_xlim(-mx, mx)
        ax2.set_ylim(-mx, mx)

        fig2.tight_layout(pad=0)
        fname = (f"lissajous_{self.omega_x}v{self.omega_y}"
                 f"_d{self.phase:.0f}_g{self.damping:.2f}"
                 f"_{self.palette}_{W}x{H}.png")
        fig2.savefig(fname, dpi=100, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig2)
        print(f"Saved: {fname}")

    def run(self):
        plt.show()


if __name__ == "__main__":
    LissajousExplorer().run()
