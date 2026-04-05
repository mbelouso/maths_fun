#!/usr/bin/env python3
"""
Double Pendulum
===============
Interactive simulation of the double pendulum — a classic chaotic system.
Two rigid rods are connected end-to-end; the motion is governed by the
Lagrangian equations derived from Newtonian mechanics.

Equations of motion (Lagrangian mechanics)
-------------------------------------------
  Δ  = θ1 − θ2
  D  = 2m1 + m2 − m2·cos(2Δ)
  α1 = [−g(2m1+m2)sin θ1 − m2·g·sin(θ1−2θ2) − 2·sin Δ·m2·(ω2²L2 + ω1²L1·cos Δ)] / L1D
  α2 = [2·sin Δ·(ω1²L1(m1+m2) + g(m1+m2)cos θ1 + ω2²L2·m2·cos Δ)] / L2D

Integer frequency ratios → regular motion.  Large angles → chaos.

Controls
--------
  θ1, θ2  : initial angles of each arm (degrees from vertical)
  ω1, ω2  : initial angular velocities (rad/s)
  L1, L2  : arm lengths (m)
  m1, m2  : bob masses (kg)
  g       : gravitational acceleration (m/s²)
  Trail   : number of past positions to show
  Sim speed: time multiplier (drag live during simulation)
  Presets  : 4 stable + 4 chaotic starting configurations
  Pause / Reset: control the simulation
  Export MP4: pre-compute the trajectory from current state and save
              a clean video (requires ffmpeg — install with:
              conda install -c conda-forge ffmpeg)

Usage
-----
    conda run -n maths_fun python3 double_pendulum.py
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ── Colour scheme ─────────────────────────────────────────────────────────────

CTRL_BG  = "#12122a"
CTRL_FG  = "#ccccee"
ACC_C    = "#7788ff"
GOLD     = "#FFD700"
BTN_OFF  = "#1a1a3e"
BTN_ON   = "#1a3a50"
DARK_BG  = "#07070f"

COL_ROD  = "#8899cc"
COL_BOB1 = "#4499ff"
COL_BOB2 = "#ff8833"


class DoublePendulum:

    # (label, θ1°, θ2°, ω1, ω2, L1, L2, m1, m2, g)
    PRESETS = [
        ("Small  ✓",    20,   10, 0, 0, 1.0, 1.0, 1.0, 1.0,  9.81),
        ("Sym    ✓",    45,  -45, 0, 0, 1.0, 1.0, 1.0, 1.0,  9.81),
        ("Chaotic 1",  120,   60, 0, 0, 1.0, 1.0, 1.0, 1.0,  9.81),
        ("Chaotic 2",  150,   30, 0, 0, 1.0, 1.0, 1.0, 1.0,  9.81),
        ("Butterfly",  170,   10, 0, 0, 1.0, 1.0, 1.0, 1.0,  9.81),
        ("Near Top",   179,    0, 0, 0, 1.0, 1.0, 1.0, 1.0,  9.81),
        ("Zero-G",      90,   45, 0, 0, 1.0, 1.0, 1.0, 1.0,  0.5),
        ("Unequal",     90,   45, 0, 0, 1.5, 0.5, 2.0, 0.3,  9.81),
    ]

    DEFAULTS = dict(
        th1=120.0, th2=60.0, w1=0.0, w2=0.0,
        L1=1.0, L2=1.0,
        m1=1.0, m2=1.0,
        g=9.81,
        trail_len=600,
        sim_speed=1.0,
    )

    DT  = 1 / 500   # integration timestep (s) — fixed for stability
    FPS = 30        # live animation and export frame rate

    def __init__(self):
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)

        self._paused  = False
        self._batch   = False
        self._state   = None       # [θ1, ω1, θ2, ω2]  (radians)
        self._t       = 0.0
        self._trail   = deque(maxlen=int(self.trail_len))
        self._trail_cmap = plt.get_cmap("hot")

        self._build_figure()
        self._reset_state()
        self._start_anim()

    # ── Equations of motion ───────────────────────────────────────────────────

    def _derivs(self, s):
        θ1, ω1, θ2, ω2 = s
        L1, L2, m1, m2, g = self.L1, self.L2, self.m1, self.m2, self.g
        Δ = θ1 - θ2
        D = 2*m1 + m2 - m2*np.cos(2*Δ)
        α1 = (- g*(2*m1+m2)*np.sin(θ1)
              - m2*g*np.sin(θ1 - 2*θ2)
              - 2*np.sin(Δ)*m2*(ω2**2*L2 + ω1**2*L1*np.cos(Δ))
              ) / (L1 * D)
        α2 = (2*np.sin(Δ)*(ω1**2*L1*(m1+m2)
                            + g*(m1+m2)*np.cos(θ1)
                            + ω2**2*L2*m2*np.cos(Δ))
              ) / (L2 * D)
        return np.array([ω1, α1, ω2, α2])

    def _rk4(self, s, dt):
        k1 = self._derivs(s)
        k2 = self._derivs(s + 0.5*dt*k1)
        k3 = self._derivs(s + 0.5*dt*k2)
        k4 = self._derivs(s + dt*k3)
        return s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    def _positions(self, s):
        θ1, _ω1, θ2, _ω2 = s
        x1 =  self.L1*np.sin(θ1);  y1 = -self.L1*np.cos(θ1)
        x2 = x1 + self.L2*np.sin(θ2);  y2 = y1 - self.L2*np.cos(θ2)
        return (x1, y1), (x2, y2)

    def _energy(self, s):
        θ1, ω1, θ2, ω2 = s
        L1, L2, m1, m2, g = self.L1, self.L2, self.m1, self.m2, self.g
        KE = (0.5*(m1+m2)*L1**2*ω1**2
              + 0.5*m2*L2**2*ω2**2
              + m2*L1*L2*ω1*ω2*np.cos(θ1-θ2))
        PE = -(m1+m2)*g*L1*np.cos(θ1) - m2*g*L2*np.cos(θ2)
        return KE, PE

    # ── State management ──────────────────────────────────────────────────────

    def _reset_state(self):
        self._state = np.array([np.radians(self.th1), self.w1,
                                np.radians(self.th2), self.w2])
        self._t = 0.0
        self._trail = deque(maxlen=max(2, int(self.trail_len)))

    # ── Figure construction ───────────────────────────────────────────────────

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
            self.fig.canvas.manager.set_window_title("Double Pendulum")

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

        lim = (self.L1 + self.L2) * 1.15
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)

        # Pre-create all animated artists (empty data)
        self._trail_lc = LineCollection(
            [], linewidth=1.5, capstyle="round", zorder=1,
        )
        self.ax.add_collection(self._trail_lc)

        self._rod1, = self.ax.plot([], [], color=COL_ROD, lw=4,
                                   solid_capstyle="round", zorder=2)
        self._rod2, = self.ax.plot([], [], color=COL_ROD, lw=3,
                                   solid_capstyle="round", zorder=2)

        # Fixed pivot — not animated, draw once
        self.ax.plot([0], [0], "o", color="white", ms=8, zorder=5)

        self._bob1, = self.ax.plot([], [], "o", color=COL_BOB1,
                                   ms=12, zorder=4, markeredgewidth=0)
        self._bob2, = self.ax.plot([], [], "o", color=COL_BOB2,
                                   ms=12, zorder=4, markeredgewidth=0)

        self._info = self.ax.text(
            0.02, 0.97, "", transform=self.ax.transAxes,
            va="top", ha="left", color=CTRL_FG, fontsize=8,
            fontfamily="monospace",
        )

        self.title = self.fig.text(
            0.38, 0.96, "Double Pendulum", ha="center", va="top",
            color=CTRL_FG, fontsize=10,
        )

        # ── Control panel ─────────────────────────────────────────────────────
        heights = [
            0.4,              # "Initial Conditions" header   [0]
            1.3, 1.3,         # θ1, θ2                        [1,2]
            1.3, 1.3,         # ω1, ω2                        [3,4]
            0.4,              # "Pendulum" header             [5]
            1.3, 1.3,         # L1, L2                        [6,7]
            1.3, 1.3,         # m1, m2                        [8,9]
            0.4,              # "Environment" header          [10]
            1.3,              # g                             [11]
            1.3, 1.3,         # trail, sim speed              [12,13]
            0.4, 2.2,         # "Presets" header + 4×2 grid   [14,15]
            0.9,              # pause + reset row             [16]
            0.4, 1.5,         # "Export" header + dur buttons [17,18]
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

        # ── Initial conditions ─────────────────────────────────────────────────
        hdr(0, "─── Initial Conditions ───")
        self.sl_th1, self.tb_th1 = self._slider_tb_row(
            ctrl, 1, "θ1 (°)", -180, 180, self.th1, "%.1f", float, "th1")
        self.sl_th2, self.tb_th2 = self._slider_tb_row(
            ctrl, 2, "θ2 (°)", -180, 180, self.th2, "%.1f", float, "th2")
        self.sl_w1, self.tb_w1 = self._slider_tb_row(
            ctrl, 3, "ω1 (r/s)", -10, 10, self.w1, "%.2f", float, "w1")
        self.sl_w2, self.tb_w2 = self._slider_tb_row(
            ctrl, 4, "ω2 (r/s)", -10, 10, self.w2, "%.2f", float, "w2")

        # ── Pendulum geometry & mass ───────────────────────────────────────────
        hdr(5, "─── Pendulum ───")
        self.sl_L1, self.tb_L1 = self._slider_tb_row(
            ctrl, 6, "L1 (m)", 0.1, 2.0, self.L1, "%.2f", float, "L1")
        self.sl_L2, self.tb_L2 = self._slider_tb_row(
            ctrl, 7, "L2 (m)", 0.1, 2.0, self.L2, "%.2f", float, "L2")
        self.sl_m1, self.tb_m1 = self._slider_tb_row(
            ctrl, 8, "m1 (kg)", 0.1, 5.0, self.m1, "%.2f", float, "m1")
        self.sl_m2, self.tb_m2 = self._slider_tb_row(
            ctrl, 9, "m2 (kg)", 0.1, 5.0, self.m2, "%.2f", float, "m2")

        # ── Environment ───────────────────────────────────────────────────────
        hdr(10, "─── Environment ───")
        self.sl_g, self.tb_g = self._slider_tb_row(
            ctrl, 11, "g (m/s²)", 0.0, 25.0, self.g, "%.2f", float, "g")
        # Trail and speed: read live — do not reset simulation
        self.sl_trail, self.tb_trail = self._slider_tb_row(
            ctrl, 12, "Trail pts", 50, 3000, self.trail_len, "%d", int,
            "trail_len", draw=False)
        self.sl_speed, self.tb_speed = self._slider_tb_row(
            ctrl, 13, "Sim speed", 0.1, 8.0, self.sim_speed, "%.1f", float,
            "sim_speed", draw=False)

        # ── Presets (4×2 grid) ────────────────────────────────────────────────
        hdr(14, "─── Presets ───")
        ax_pre = self.fig.add_subplot(ctrl[15], facecolor=CTRL_BG)
        ax_pre.set_axis_off()
        self._preset_btns = []
        cols_p = 4
        for i, preset in enumerate(self.PRESETS):
            col = i % cols_p
            row = i // cols_p
            x0  = col / cols_p + 0.005
            y0  = (1 - row) * 0.5 + 0.04
            a   = ax_pre.inset_axes([x0, y0, 0.99/cols_p - 0.015, 0.44])
            b   = plain_btn(a, preset[0])
            b.label.set_fontsize(6.5)
            b.on_clicked(lambda _, p=preset: self._apply_preset(p))
            self._preset_btns.append(b)

        # ── Pause / Reset ─────────────────────────────────────────────────────
        ax_ctrl = self.fig.add_subplot(ctrl[16], facecolor=CTRL_BG)
        ax_ctrl.set_axis_off()
        ax_pp  = ax_ctrl.inset_axes([0.00, 0.05, 0.49, 0.90])
        ax_rst = ax_ctrl.inset_axes([0.51, 0.05, 0.49, 0.90])

        self._btn_pp = Button(ax_pp, "⏸  Pause",
                              color=BTN_OFF, hovercolor="#223355")
        self._btn_pp.label.set_color(GOLD)
        self._btn_pp.label.set_fontsize(8)
        self._btn_pp.on_clicked(self._toggle_pause)

        self._btn_rst = Button(ax_rst, "↺  Reset",
                               color=BTN_OFF, hovercolor="#223355")
        self._btn_rst.label.set_color(CTRL_FG)
        self._btn_rst.label.set_fontsize(8)
        self._btn_rst.on_clicked(self._do_reset)

        # ── Export MP4 ────────────────────────────────────────────────────────
        hdr(17, "─── Export MP4 ───")
        ax_exp = self.fig.add_subplot(ctrl[18], facecolor=CTRL_BG)
        ax_exp.set_axis_off()
        self._exp_btns = []
        for ci, (dur, lbl) in enumerate([(10, "10 s"), (30, "30 s"), (60, "60 s")]):
            a = ax_exp.inset_axes([ci/3 + 0.01, 0.05, 0.31, 0.90])
            b = plain_btn(a, lbl)
            b.label.set_fontsize(8)
            b.on_clicked(lambda _, d=dur: self._export(d))
            self._exp_btns.append(b)

    # ── Slider + TextBox factory ───────────────────────────────────────────────

    def _slider_tb_row(self, ctrl, row, label, vmin, vmax, vinit, fmt, cast, attr,
                       draw=True):
        """One GridSpec row: 62% slider + 33% TextBox.

        draw=True : changing the value resets and restarts the simulation.
        draw=False: value is read live each frame — slider does not reset.
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
                    self._do_reset(None)

        def on_tb_submit(text):
            try:
                v = float(text.strip())
                v = max(vmin, min(vmax, v))
                sl.set_val(cast(v))
            except ValueError:
                tb.set_val(fmt % getattr(self, attr))

        sl.on_changed(on_sl_changed)
        tb.on_submit(on_tb_submit)
        return sl, tb

    # ── Control helpers ────────────────────────────────────────────────────────

    def _apply_preset(self, preset):
        lbl, th1, th2, w1, w2, L1, L2, m1, m2, g = preset
        self._batch = True
        self.sl_th1.set_val(th1); self.th1 = float(th1); self.tb_th1.set_val("%.1f" % th1)
        self.sl_th2.set_val(th2); self.th2 = float(th2); self.tb_th2.set_val("%.1f" % th2)
        self.sl_w1.set_val(w1);   self.w1  = float(w1);  self.tb_w1.set_val("%.2f" % w1)
        self.sl_w2.set_val(w2);   self.w2  = float(w2);  self.tb_w2.set_val("%.2f" % w2)
        self.sl_L1.set_val(L1);   self.L1  = float(L1);  self.tb_L1.set_val("%.2f" % L1)
        self.sl_L2.set_val(L2);   self.L2  = float(L2);  self.tb_L2.set_val("%.2f" % L2)
        self.sl_m1.set_val(m1);   self.m1  = float(m1);  self.tb_m1.set_val("%.2f" % m1)
        self.sl_m2.set_val(m2);   self.m2  = float(m2);  self.tb_m2.set_val("%.2f" % m2)
        self.sl_g.set_val(g);     self.g   = float(g);   self.tb_g.set_val("%.2f" % g)
        self._batch = False
        self._do_reset(None)

    def _do_reset(self, _):
        self._reset_state()
        lim = (self.L1 + self.L2) * 1.15
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)

    def _toggle_pause(self, _):
        self._paused = not self._paused
        if self._paused:
            self._btn_pp.label.set_text("▶  Resume")
            self._btn_pp.label.set_color("#44ff88")
        else:
            self._btn_pp.label.set_text("⏸  Pause")
            self._btn_pp.label.set_color(GOLD)
        self.fig.canvas.draw_idle()

    # ── Artist update (shared by animation and export renderer) ───────────────

    def _trail_colors(self, n):
        """n×4 RGBA array — fades from dim to bright along the trail."""
        alphas = np.linspace(0.04, 0.92, n)
        rgba   = self._trail_cmap(0.55 + 0.45 * alphas)   # hot: orange→white
        rgba[:, 3] = alphas
        return rgba

    def _update_artists(self, state, trail):
        (x1, y1), (x2, y2) = self._positions(state)

        self._rod1.set_data([0, x1], [0, y1])
        self._rod2.set_data([x1, x2], [y1, y2])

        ms1 = np.clip(self.m1 * 12, 8, 28)
        ms2 = np.clip(self.m2 * 12, 8, 28)
        self._bob1.set_data([x1], [y1]); self._bob1.set_markersize(ms1)
        self._bob2.set_data([x2], [y2]); self._bob2.set_markersize(ms2)

        arr = np.asarray(trail)
        if len(arr) > 1:
            pts  = arr.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            self._trail_lc.set_segments(segs)
            self._trail_lc.set_color(self._trail_colors(len(segs)))
        else:
            self._trail_lc.set_segments([])

        KE, PE = self._energy(state)
        self._info.set_text(
            f"t  = {self._t:7.2f} s\n"
            f"KE = {KE:7.3f} J\n"
            f"PE = {PE:7.3f} J\n"
            f"E  = {KE+PE:7.3f} J"
        )

        return (self._rod1, self._rod2, self._bob1, self._bob2,
                self._trail_lc, self._info)

    # ── Live animation ────────────────────────────────────────────────────────

    def _start_anim(self):
        # Number of integration steps per rendered frame:
        # sim_speed * (1/FPS) seconds of simulation time, split into DT steps.
        def _frame(_i):
            if not self._paused:
                # Resize trail deque if slider changed
                tlen = max(2, int(self.trail_len))
                if self._trail.maxlen != tlen:
                    self._trail = deque(list(self._trail), maxlen=tlen)

                n_steps = max(1, round(self.sim_speed / (self.FPS * self.DT)))
                for _ in range(n_steps):
                    self._state = self._rk4(self._state, self.DT)
                    self._t    += self.DT

                _, (x2, y2) = self._positions(self._state)
                self._trail.append((x2, y2))

            return self._update_artists(self._state, self._trail)

        self._anim = FuncAnimation(
            self.fig, _frame,
            interval=1000 // self.FPS,
            blit=True, cache_frame_data=False,
        )

    # ── Export MP4 ────────────────────────────────────────────────────────────

    def _export(self, duration_s):
        if not FFMpegWriter.isAvailable():
            print("ffmpeg not found.  Install with:")
            print("  conda install -c conda-forge ffmpeg")
            return

        print(f"Computing {duration_s}s trajectory …", flush=True)
        n_frames      = duration_s * self.FPS
        steps_p_frame = max(1, round(1.0 / (self.FPS * self.DT)))

        # Pre-compute all states starting from current live position
        states = np.empty((n_frames + 1, 4))
        states[0] = self._state
        s = self._state.copy()
        for i in range(1, n_frames + 1):
            for _ in range(steps_p_frame):
                s = self._rk4(s, self.DT)
            states[i] = s

        print("Rendering …", flush=True)

        trail_max = max(2, int(self.trail_len))
        trail     = deque(maxlen=trail_max)

        lim  = (self.L1 + self.L2) * 1.15

        fig_e = plt.figure(figsize=(8, 8), facecolor=DARK_BG)
        ax_e  = fig_e.add_subplot(111, facecolor=DARK_BG)
        ax_e.set_aspect("equal")
        ax_e.set_axis_off()
        ax_e.set_xlim(-lim, lim)
        ax_e.set_ylim(-lim, lim)

        e_trail_lc = LineCollection([], linewidth=1.5, capstyle="round", zorder=1)
        ax_e.add_collection(e_trail_lc)
        ax_e.plot([0], [0], "o", color="white", ms=7, zorder=5)
        e_rod1, = ax_e.plot([], [], color=COL_ROD, lw=4, solid_capstyle="round", zorder=2)
        e_rod2, = ax_e.plot([], [], color=COL_ROD, lw=3, solid_capstyle="round", zorder=2)
        ms1 = float(np.clip(self.m1 * 12, 8, 28))
        ms2 = float(np.clip(self.m2 * 12, 8, 28))
        e_bob1, = ax_e.plot([], [], "o", color=COL_BOB1, ms=ms1, zorder=4, markeredgewidth=0)
        e_bob2, = ax_e.plot([], [], "o", color=COL_BOB2, ms=ms2, zorder=4, markeredgewidth=0)
        e_txt   = ax_e.text(0.02, 0.97, "", transform=ax_e.transAxes,
                            va="top", ha="left", color=CTRL_FG,
                            fontsize=9, fontfamily="monospace")

        def render(i):
            state = states[i]
            (x1, y1), (x2, y2) = self._positions(state)
            trail.append((x2, y2))

            e_rod1.set_data([0, x1], [0, y1])
            e_rod2.set_data([x1, x2], [y1, y2])
            e_bob1.set_data([x1], [y1])
            e_bob2.set_data([x2], [y2])

            arr = np.asarray(trail)
            if len(arr) > 1:
                pts  = arr.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                n    = len(segs)
                alphas = np.linspace(0.04, 0.92, n)
                rgba   = self._trail_cmap(0.55 + 0.45*alphas)
                rgba[:, 3] = alphas
                e_trail_lc.set_segments(segs)
                e_trail_lc.set_color(rgba)
            else:
                e_trail_lc.set_segments([])

            e_txt.set_text(f"t = {i / self.FPS:.2f} s")
            return e_rod1, e_rod2, e_bob1, e_bob2, e_trail_lc, e_txt

        anim_e = FuncAnimation(fig_e, render, frames=n_frames + 1,
                               blit=True, cache_frame_data=False)

        fname = (f"double_pendulum"
                 f"_th{self.th1:.0f}v{self.th2:.0f}"
                 f"_L{self.L1:.1f}v{self.L2:.1f}"
                 f"_g{self.g:.1f}"
                 f"_{duration_s}s.mp4")

        writer = FFMpegWriter(fps=self.FPS, bitrate=4000,
                              metadata=dict(title="Double Pendulum"))
        try:
            anim_e.save(fname, writer=writer, dpi=150)
            print(f"Saved: {fname}")
        except Exception as e:
            print(f"Export failed: {e}")
        finally:
            plt.close(fig_e)

    def run(self):
        plt.show()


if __name__ == "__main__":
    DoublePendulum().run()
