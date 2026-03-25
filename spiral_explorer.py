#!/usr/bin/env python3
"""
Spiral Explorer
===============
Interactively explore generalised Vogel-style spirals of the form:

    r(k)  = k^p          (p adjustable)
    θ(k)  = k × angle°   (angle adjustable)

where k = 1, 2, 3, … N.

At p=0.5 and angle≈137.508° you recover the classic Vogel sunflower spiral.

Controls
--------
  Spiral parameters : N points, r exponent (p), θ step (°)
  Presets           : Sunflower (golden angle) | Alt. golden
  Visual style      : dot size, connecting line width / alpha / toggle
  Colour mode       : Primes | Twin primes | Gradient | Mod 6 | Mod 12

Usage
-----
    python spiral_explorer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, RadioButtons, Button
import matplotlib.gridspec as gridspec

# ── Palette ────────────────────────────────────────────────────────────────────

BG      = "#0d0d1a"
PRIME_C = "#FFD700"
TWIN_C  = "#FF4081"
CTRL_BG = "#12122a"
CTRL_FG = "#ccccee"
ACC_C   = "#7788ff"

GOLDEN_ANGLE     = 180.0 * (3.0 - np.sqrt(5.0))   # ≈ 137.508°
ALT_GOLDEN_ANGLE = 360.0 - GOLDEN_ANGLE            # ≈ 222.492°


# ── Sieve ──────────────────────────────────────────────────────────────────────

def sieve(limit: int) -> np.ndarray:
    ip = np.ones(limit + 1, dtype=bool)
    ip[:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if ip[i]:
            ip[i * i :: i] = False
    return ip


# ── Main class ─────────────────────────────────────────────────────────────────

class SpiralExplorer:
    MAX_N = 100_000
    _ip   = sieve(MAX_N + 10)

    COLOR_MODES = ["Primes", "Twin primes", "Gradient", "Mod 6", "Mod 12"]

    DEFAULTS = dict(
        n          = 2_000,
        r_exp      = 0.5,
        angle      = GOLDEN_ANGLE,
        dot_size   = 5.0,
        line_width = 0.8,
        line_alpha = 0.25,
        show_line  = False,
        mode       = "Primes",
    )

    def __init__(self):
        self.n          = self.DEFAULTS["n"]
        self.r_exp      = self.DEFAULTS["r_exp"]
        self.angle      = self.DEFAULTS["angle"]
        self.dot_size   = self.DEFAULTS["dot_size"]
        self.line_width = self.DEFAULTS["line_width"]
        self.line_alpha = self.DEFAULTS["line_alpha"]
        self.show_line  = self.DEFAULTS["show_line"]
        self.mode       = self.DEFAULTS["mode"]

        self._batch   = False   # suppress mid-preset redraws
        self._drawing = False

        self._build_figure()
        self._draw()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_figure(self):
        plt.rcParams.update({
            "figure.facecolor": BG,
            "axes.facecolor":   BG,
            "axes.edgecolor":   "#334",
            "text.color":       CTRL_FG,
            "xtick.color":      "#556",
            "ytick.color":      "#556",
        })

        self.fig = plt.figure(figsize=(15, 9), facecolor=BG)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Spiral Explorer")

        outer = gridspec.GridSpec(
            1, 2, width_ratios=[3, 1],
            left=0.03, right=0.98, top=0.94, bottom=0.04, wspace=0.04,
        )
        self.ax = self.fig.add_subplot(outer[0], facecolor=BG)

        # 13-row control column
        heights = [
            0.4, 1.0, 1.0, 1.0, 0.9,   # spiral parameters
            0.4, 1.0, 1.0, 1.0, 0.9,   # visual style
            0.4, 3.2, 0.9,              # colour mode + reset
        ]
        ctrl = gridspec.GridSpecFromSubplotSpec(
            len(heights), 1, subplot_spec=outer[1],
            hspace=0.30, height_ratios=heights,
        )

        def hdr(row, text):
            a = self.fig.add_subplot(ctrl[row], facecolor=CTRL_BG)
            a.set_axis_off()
            a.text(0.5, 0.5, text, ha="center", va="center",
                   color=PRIME_C, fontsize=9, fontweight="bold")

        def slider(row, label, vmin, vmax, vinit, vstep=None, fmt=None):
            a = self.fig.add_subplot(ctrl[row], facecolor=CTRL_BG)
            kw = dict(label=label, valmin=vmin, valmax=vmax, valinit=vinit,
                      color=ACC_C, track_color="#334")
            if vstep is not None:
                kw["valstep"] = vstep
            if fmt is not None:
                kw["valfmt"] = fmt
            sl = Slider(a, **kw)
            for attr in ("label", "valtext"):
                obj = getattr(sl, attr)
                obj.set_color(CTRL_FG)
                obj.set_fontsize(8)
            return sl

        def btn(ax_row, text, col=CTRL_BG):
            b = Button(ax_row, text, color=col, hovercolor="#222255")
            b.label.set_color(CTRL_FG)
            b.label.set_fontsize(8)
            return b

        # ── Spiral parameters ────────────────────────────────────────────────
        hdr(0, "Spiral parameters")
        self.sl_n     = slider(1, "N points",   10, self.MAX_N, self.n,     vstep=10)
        self.sl_rexp  = slider(2, "r = k ^  p", 0.1,      3.0, self.r_exp, fmt="%.2f")
        self.sl_angle = slider(3, "θ step (°)", 0.5,    360.0, self.angle,  fmt="%.3f")

        # Two preset buttons side by side
        ax_pre = self.fig.add_subplot(ctrl[4], facecolor=CTRL_BG)
        ax_pre.set_axis_off()
        ax_g = ax_pre.inset_axes([0.02, 0.1, 0.46, 0.8])
        ax_a = ax_pre.inset_axes([0.52, 0.1, 0.46, 0.8])
        self.btn_gold = btn(ax_g, "Sunflower ★")
        self.btn_alt  = btn(ax_a, "Alt. golden")

        # ── Visual style ─────────────────────────────────────────────────────
        hdr(5, "Visual style")
        self.sl_dotsize = slider(6, "Dot size",   1.0, 40.0, self.dot_size,   fmt="%.1f")
        self.sl_lw      = slider(7, "Line width", 0.1,  5.0, self.line_width, fmt="%.1f")
        self.sl_la      = slider(8, "Line alpha", 0.0,  1.0, self.line_alpha, fmt="%.2f")

        ax_tog = self.fig.add_subplot(ctrl[9], facecolor=CTRL_BG)
        self.btn_line = Button(ax_tog, "Line: OFF", color="#1a1a3e", hovercolor="#222266")
        self.btn_line.label.set_color(CTRL_FG)
        self.btn_line.label.set_fontsize(8)

        # ── Colour mode ──────────────────────────────────────────────────────
        hdr(10, "Colour mode")
        ax_mode = self.fig.add_subplot(ctrl[11], facecolor=CTRL_BG)
        self.radio_mode = RadioButtons(
            ax_mode, self.COLOR_MODES,
            active=self.COLOR_MODES.index(self.mode),
            activecolor=PRIME_C,
        )
        for lbl in self.radio_mode.labels:
            lbl.set_color(CTRL_FG)
            lbl.set_fontsize(9)

        ax_rst = self.fig.add_subplot(ctrl[12], facecolor=CTRL_BG)
        self.btn_reset = btn(ax_rst, "Reset zoom")

        # ── Title ────────────────────────────────────────────────────────────
        self.title = self.fig.text(
            0.38, 0.975, "", ha="center", va="top",
            color=PRIME_C, fontsize=12, fontweight="bold",
        )

        # ── Wire up callbacks ────────────────────────────────────────────────
        self.sl_n.on_changed(    lambda v: self._set(n=int(v)))
        self.sl_rexp.on_changed( lambda v: self._set(r_exp=float(v)))
        self.sl_angle.on_changed(lambda v: self._set(angle=float(v)))
        self.sl_dotsize.on_changed(lambda v: self._set(dot_size=float(v)))
        self.sl_lw.on_changed(   lambda v: self._set(line_width=float(v)))
        self.sl_la.on_changed(   lambda v: self._set(line_alpha=float(v)))
        self.btn_line.on_clicked(self._toggle_line)
        self.btn_gold.on_clicked(self._preset_sunflower)
        self.btn_alt.on_clicked( self._preset_alt_golden)
        self.radio_mode.on_clicked(lambda lbl: self._set(mode=lbl))
        self.btn_reset.on_clicked(self._on_reset)

    # ── Computation ───────────────────────────────────────────────────────────

    def _coords(self):
        k     = np.arange(1, self.n + 1, dtype=float)
        r     = k ** self.r_exp
        theta = np.radians(k * self.angle)
        return r * np.cos(theta), r * np.sin(theta)

    def _colors(self, nums: np.ndarray) -> np.ndarray:
        ip    = self._ip
        mode  = self.mode
        N     = len(nums)
        valid = nums < len(ip)

        # Helper: bool array of primes in nums
        def is_prime_mask():
            m = np.zeros(N, dtype=bool)
            m[valid] = ip[nums[valid]]
            return m

        if mode == "Primes":
            gold = np.array(mcolors.to_rgba(PRIME_C),    dtype=np.float32)
            dark = np.array(mcolors.to_rgba("#2a2a3e"),  dtype=np.float32)
            dark[3] = 0.25
            ip_m  = is_prime_mask()
            rgba  = np.where(ip_m[:, None], gold, dark).astype(np.float32)

        elif mode == "Twin primes":
            twin = np.zeros(len(ip), dtype=bool)
            ps   = np.where(ip)[0]
            ok   = (ps + 2) < len(ip)
            tp   = ps[ok][ip[(ps + 2)[ok]]]
            twin[tp]     = True
            twin[tp + 2] = True

            gold  = np.array(mcolors.to_rgba(PRIME_C),   dtype=np.float32)
            pink  = np.array(mcolors.to_rgba(TWIN_C),    dtype=np.float32)
            dark  = np.array(mcolors.to_rgba("#2a2a3e"), dtype=np.float32)
            dark[3] = 0.20

            ip_m  = is_prime_mask()
            tw_m  = np.zeros(N, dtype=bool)
            tw_m[valid] = twin[nums[valid]]
            rgba  = np.where(tw_m[:, None], pink,
                    np.where(ip_m[:, None], gold, dark)).astype(np.float32)

        elif mode == "Gradient":
            t    = np.linspace(0, 1, N)
            rgba = plt.cm.plasma(t).astype(np.float32)

        elif mode == "Mod 6":
            rgba  = plt.cm.hsv((nums % 6) / 6.0).astype(np.float32)
            ip_m  = is_prime_mask()
            rgba[~ip_m, 3] = 0.18

        elif mode == "Mod 12":
            rgba  = plt.cm.hsv((nums % 12) / 12.0).astype(np.float32)
            ip_m  = is_prime_mask()
            rgba[~ip_m, 3] = 0.15

        else:
            rgba = np.full((N, 4), 0.5, dtype=np.float32)

        return rgba

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        if self._batch:
            return
        self._drawing = True

        self.ax.cla()
        self.ax.set_facecolor(BG)
        self.ax.set_aspect("equal")
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        nums = np.arange(1, self.n + 1, dtype=np.int32)
        xs, ys = self._coords()

        # Prime mask
        ip   = self._ip
        vld  = nums < len(ip)
        ipm  = np.zeros(len(nums), dtype=bool)
        ipm[vld] = ip[nums[vld]]

        rgba  = self._colors(nums)
        sizes = np.where(ipm, self.dot_size * 2.0, self.dot_size)

        # Connecting line
        if self.show_line:
            self.ax.plot(
                xs, ys,
                color=CTRL_FG, lw=self.line_width, alpha=self.line_alpha,
                zorder=1, solid_capstyle="round",
            )

        # Composites first so primes render on top
        comp = ~ipm
        if comp.any():
            self.ax.scatter(
                xs[comp], ys[comp],
                c=rgba[comp], s=sizes[comp],
                linewidths=0, rasterized=True, zorder=2,
            )
        if ipm.any():
            self.ax.scatter(
                xs[ipm], ys[ipm],
                c=rgba[ipm], s=sizes[ipm],
                linewidths=0, rasterized=True, zorder=3,
            )

        self.title.set_text(
            f"r = k^{self.r_exp:.2f},  θ = k × {self.angle:.3f}°"
            f"   ·   {self.n:,} pts   ·   {self.mode}"
        )

        self._drawing = False
        self.fig.canvas.draw_idle()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._draw()

    def _toggle_line(self, _event):
        self.show_line = not self.show_line
        if self.show_line:
            self.btn_line.label.set_text("Line: ON")
            self.btn_line.ax.set_facecolor("#1a3040")
        else:
            self.btn_line.label.set_text("Line: OFF")
            self.btn_line.ax.set_facecolor("#1a1a3e")
        self._draw()

    def _preset_sunflower(self, _event):
        """Golden angle: uniform angular distribution (Vogel sunflower)."""
        self._batch = True
        self.r_exp  = 0.5
        self.angle  = GOLDEN_ANGLE
        self.sl_rexp.set_val(0.5)
        self.sl_angle.set_val(GOLDEN_ANGLE)
        self._batch = False
        self._draw()

    def _preset_alt_golden(self, _event):
        """Alternate golden angle (360 − φ): same distribution, opposite winding."""
        self._batch = True
        self.r_exp  = 0.5
        self.angle  = ALT_GOLDEN_ANGLE
        self.sl_rexp.set_val(0.5)
        self.sl_angle.set_val(ALT_GOLDEN_ANGLE)
        self._batch = False
        self._draw()

    def _on_reset(self, _event):
        self.ax.autoscale()
        self.fig.canvas.draw_idle()

    # ── Launch ────────────────────────────────────────────────────────────────

    def show(self):
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    explorer = SpiralExplorer()
    explorer.show()
