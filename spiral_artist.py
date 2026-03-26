#!/usr/bin/env python3
"""
Spiral Artist
=============
Generalized spirograph: r(k) = k^p,  θ(k) = k × angle°,  k = 1 … N.

Controls
--------
  Spiral formula : N points, r exponent (p), θ step (°) — sliders with keyboard input
  Presets        : Sunflower | Alt. golden | Pentagon | Galaxy
  Visual style   : line width, line alpha, dot size; display mode (Dots | Line | Both)
  Palette        : 10 gradient colormaps + reverse
  Background     : Dark | Light
  Export         : save high-resolution PNG at 150 / 300 / 600 dpi

Usage
-----
    python spiral_artist.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
import matplotlib.gridspec as gridspec

# ── UI colours (control panel — always dark) ───────────────────────────────────

CTRL_BG = "#12122a"
CTRL_FG = "#ccccee"
ACC_C   = "#7788ff"
GOLD    = "#FFD700"
BTN_OFF = "#1a1a3e"
BTN_ON  = "#1a3a50"

# Plot background options
DARK_BG  = "#0d0d1a"
LIGHT_BG = "#f5f5f0"

GOLDEN_ANGLE = 180.0 * (3.0 - np.sqrt(5.0))   # ≈ 137.508°
ALT_GOLDEN   = 360.0 - GOLDEN_ANGLE            # ≈ 222.492°


# ── Main class ─────────────────────────────────────────────────────────────────

class SpiralArtist:
    MAX_N = 100_000
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
        export_dpi  = 300,
    )

    def __init__(self):
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)

        self._batch   = False
        self._drawing = False

        self._build_figure()
        self._draw()

    # ── Layout ────────────────────────────────────────────────────────────────

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
            self.fig.canvas.manager.set_window_title("Spiral Artist")

        outer = gridspec.GridSpec(
            1, 2, width_ratios=[3, 1],
            left=0.03, right=0.98, top=0.94, bottom=0.04, wspace=0.04,
        )
        self.ax = self.fig.add_subplot(outer[0], facecolor=self.plot_bg)

        heights = [
            0.4, 1.3, 1.3, 1.3,   # "Spiral formula" header + N, p, angle
            1.1,                    # presets 2×2
            0.4, 1.3, 1.1, 1.1,   # "Visual style" header + lw, alpha, dotsize
            1.0,                    # display mode buttons
            0.4, 5.2,              # "Palette" header + 10-item radio
            0.9,                    # reverse button
            0.4, 1.0,              # "Background" header + dark/light buttons
            0.35, 1.5,             # "Export" header + export row
        ]
        ctrl = gridspec.GridSpecFromSubplotSpec(
            len(heights), 1, subplot_spec=outer[1],
            hspace=0.25, height_ratios=heights,
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

        # ── Spiral formula ───────────────────────────────────────────────────
        hdr(0, "Spiral formula")

        self.sl_n,     self.tb_n     = self._slider_tb_row(ctrl, 1,
            "N",      10, self.MAX_N, self.n,     "%d",    int,   "n")
        self.sl_rexp,  self.tb_rexp  = self._slider_tb_row(ctrl, 2,
            "k ^  p", 0.1, 3.0,      self.r_exp, "%.2f",  float, "r_exp")
        self.sl_angle, self.tb_angle = self._slider_tb_row(ctrl, 3,
            "θ (°)",  0.5, 360.0,    self.angle, "%.3f",  float, "angle")

        # ── Presets (2×2 grid) ───────────────────────────────────────────────
        ax_pre = self.fig.add_subplot(ctrl[4], facecolor=CTRL_BG)
        ax_pre.set_axis_off()
        ax_p0 = ax_pre.inset_axes([0.01, 0.52, 0.48, 0.44])
        ax_p1 = ax_pre.inset_axes([0.51, 0.52, 0.48, 0.44])
        ax_p2 = ax_pre.inset_axes([0.01, 0.04, 0.48, 0.44])
        ax_p3 = ax_pre.inset_axes([0.51, 0.04, 0.48, 0.44])

        # Store as instance attrs to prevent garbage collection
        self._btn_sun  = plain_btn(ax_p0, "Sunflower \u2605")
        self._btn_alt  = plain_btn(ax_p1, "Alt. golden")
        self._btn_pent = plain_btn(ax_p2, "Pentagon")
        self._btn_gal  = plain_btn(ax_p3, "Galaxy")

        self._btn_sun.on_clicked( lambda _: self._preset(GOLDEN_ANGLE, 0.5))
        self._btn_alt.on_clicked( lambda _: self._preset(ALT_GOLDEN,   0.5))
        self._btn_pent.on_clicked(lambda _: self._preset(144.0,         0.5))
        self._btn_gal.on_clicked( lambda _: self._preset(GOLDEN_ANGLE,  1.0))

        # ── Visual style ─────────────────────────────────────────────────────
        hdr(5, "Visual style")

        self.sl_lw, self.tb_lw = self._slider_tb_row(ctrl, 6,
            "Line w", 0.1, 8.0, self.line_width, "%.1f", float, "line_width")

        ax_la = self.fig.add_subplot(ctrl[7], facecolor=CTRL_BG)
        self.sl_la = self._plain_slider(ax_la, "Alpha", 0.0, 1.0,
                                        self.line_alpha, "%.2f")
        self.sl_la.on_changed(lambda v: self._set(line_alpha=float(v)))

        ax_ds = self.fig.add_subplot(ctrl[8], facecolor=CTRL_BG)
        self.sl_ds = self._plain_slider(ax_ds, "Dot sz", 0.5, 30.0,
                                        self.dot_size, "%.1f")
        self.sl_ds.on_changed(lambda v: self._set(dot_size=float(v)))

        # Display mode buttons
        ax_mode = self.fig.add_subplot(ctrl[9], facecolor=CTRL_BG)
        ax_mode.set_axis_off()
        ax_md = ax_mode.inset_axes([0.01, 0.1, 0.32, 0.8])
        ax_ml = ax_mode.inset_axes([0.34, 0.1, 0.32, 0.8])
        ax_mb = ax_mode.inset_axes([0.67, 0.1, 0.32, 0.8])

        self._mode_btns = {
            "Dots": plain_btn(ax_md, "Dots", self.mode == "Dots"),
            "Line": plain_btn(ax_ml, "Line", self.mode == "Line"),
            "Both": plain_btn(ax_mb, "Both", self.mode == "Both"),
        }
        for label, btn in self._mode_btns.items():
            btn.on_clicked(lambda _, lbl=label: self._set_mode(lbl))

        # ── Palette ──────────────────────────────────────────────────────────
        hdr(10, "Palette")
        ax_pal = self.fig.add_subplot(ctrl[11], facecolor=CTRL_BG)
        self.radio_pal = RadioButtons(
            ax_pal, self.PALETTES,
            active=self.PALETTES.index(self.palette),
            activecolor=GOLD,
        )
        for lbl in self.radio_pal.labels:
            lbl.set_color(CTRL_FG)
            lbl.set_fontsize(8.5)
        self.radio_pal.on_clicked(lambda lbl: self._set(palette=lbl))

        ax_rev = self.fig.add_subplot(ctrl[12], facecolor=CTRL_BG)
        self.btn_reverse = plain_btn(ax_rev, "Reverse: OFF")
        self.btn_reverse.on_clicked(self._toggle_reverse)

        # ── Background ───────────────────────────────────────────────────────
        hdr(13, "Background")
        ax_bg = self.fig.add_subplot(ctrl[14], facecolor=CTRL_BG)
        ax_bg.set_axis_off()
        ax_bk = ax_bg.inset_axes([0.01, 0.1, 0.48, 0.8])
        ax_bw = ax_bg.inset_axes([0.51, 0.1, 0.48, 0.8])
        self._btn_bg_dark  = plain_btn(ax_bk, "Dark",  active=True)
        self._btn_bg_light = plain_btn(ax_bw, "Light", active=False)
        self._btn_bg_dark.on_clicked( lambda _: self._set_bg(DARK_BG))
        self._btn_bg_light.on_clicked(lambda _: self._set_bg(LIGHT_BG))

        # ── Export ───────────────────────────────────────────────────────────
        hdr(15, "Export")
        ax_exp = self.fig.add_subplot(ctrl[16], facecolor=CTRL_BG)
        ax_exp.set_axis_off()

        ax_d150 = ax_exp.inset_axes([0.01, 0.52, 0.30, 0.44])
        ax_d300 = ax_exp.inset_axes([0.33, 0.52, 0.30, 0.44])
        ax_d600 = ax_exp.inset_axes([0.65, 0.52, 0.34, 0.44])
        ax_save = ax_exp.inset_axes([0.01, 0.04, 0.98, 0.44])

        self._dpi_buttons = {
            150: plain_btn(ax_d150, "150 dpi"),
            300: plain_btn(ax_d300, "300 dpi", active=True),
            600: plain_btn(ax_d600, "600 dpi"),
        }
        for dpi, btn in self._dpi_buttons.items():
            btn.on_clicked(lambda _, d=dpi: self._set_dpi(d))

        self.btn_export = plain_btn(ax_save, "Export PNG")
        self.btn_export.on_clicked(self._export)

        # ── Title ────────────────────────────────────────────────────────────
        self.title = self.fig.text(
            0.38, 0.975, "", ha="center", va="top",
            color=GOLD, fontsize=12, fontweight="bold",
        )

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _slider_tb_row(self, ctrl, row, label, vmin, vmax, vinit, fmt, cast, attr):
        """One GridSpec row: 62% slider + 33% TextBox. Returns (slider, textbox)."""
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

    def _plain_slider(self, ax, label, vmin, vmax, vinit, fmt):
        sl = Slider(ax, label, vmin, vmax, valinit=vinit,
                    color=ACC_C, track_color="#334")
        sl.label.set_color(CTRL_FG);   sl.label.set_fontsize(8)
        sl.valtext.set_color(CTRL_FG); sl.valtext.set_fontsize(8)
        return sl

    # ── Coordinate computation ────────────────────────────────────────────────

    def _coords(self):
        k     = np.arange(1, self.n + 1, dtype=float)
        r     = k ** self.r_exp
        theta = np.radians(k * self.angle)
        return r * np.cos(theta), r * np.sin(theta)

    def _palette_colors(self, n):
        t = np.linspace(0, 1, n)
        if self.reverse_pal:
            t = 1 - t
        return plt.get_cmap(self.palette)(t)

    # ── Draw helpers ──────────────────────────────────────────────────────────

    def _draw_gradient_line(self, ax, xs, ys):
        n        = len(xs)
        points   = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors   = self._palette_colors(n)
        lc = LineCollection(
            segments, colors=colors[:-1],
            linewidth=self.line_width, alpha=self.line_alpha,
            capstyle="round",
        )
        ax.add_collection(lc)

    def _draw_dots(self, ax, xs, ys):
        colors = self._palette_colors(len(xs))
        ax.scatter(xs, ys, c=colors, s=self.dot_size,
                   linewidths=0, rasterized=True)

    def _draw_spiral(self, ax):
        xs, ys = self._coords()
        if self.mode in ("Line", "Both"):
            self._draw_gradient_line(ax, xs, ys)
        if self.mode in ("Dots", "Both"):
            self._draw_dots(ax, xs, ys)
        ax.autoscale_view()

    # ── Main draw ─────────────────────────────────────────────────────────────

    def _draw(self):
        if self._batch:
            return
        self._drawing = True

        self.ax.cla()
        self.ax.set_facecolor(self.plot_bg)
        self.ax.set_aspect("equal")
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self._draw_spiral(self.ax)

        self.title.set_text(
            f"r = k^{self.r_exp:.2f},  θ = k × {self.angle:.3f}°"
            f"   ·   {self.n:,} pts   ·   {self.palette}"
        )

        self._drawing = False
        self.fig.canvas.draw_idle()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._draw()

    def _set_mode(self, mode):
        self.mode = mode
        for lbl, btn in self._mode_btns.items():
            btn.ax.set_facecolor(BTN_ON if lbl == mode else BTN_OFF)
        self._draw()

    def _preset(self, angle, r_exp):
        """Apply a two-parameter preset with a single redraw."""
        self._batch = True
        self.angle  = angle
        self.r_exp  = r_exp
        self.sl_angle.set_val(angle)
        self.sl_rexp.set_val(r_exp)
        self._batch = False
        # Sync textboxes manually (on_changed was suppressed during batch)
        self.tb_angle.set_val("%.3f" % angle)
        self.tb_rexp.set_val("%.2f"  % r_exp)
        self._draw()

    def _toggle_reverse(self, _event):
        self.reverse_pal = not self.reverse_pal
        self.btn_reverse.ax.set_facecolor(BTN_ON if self.reverse_pal else BTN_OFF)
        self.btn_reverse.label.set_text(
            "Reverse: ON" if self.reverse_pal else "Reverse: OFF"
        )
        self._draw()

    def _set_bg(self, color):
        self.plot_bg = color
        is_dark = (color == DARK_BG)
        self._btn_bg_dark.ax.set_facecolor( BTN_ON  if is_dark else BTN_OFF)
        self._btn_bg_light.ax.set_facecolor(BTN_OFF if is_dark else BTN_ON)
        self._draw()

    def _set_dpi(self, dpi):
        self.export_dpi = dpi
        for d, btn in self._dpi_buttons.items():
            btn.ax.set_facecolor(BTN_ON if d == dpi else BTN_OFF)
        self.fig.canvas.draw_idle()

    def _export(self, _event):
        dpi  = self.export_dpi
        size = 12   # inches → 1800 / 3600 / 7200 px at 150 / 300 / 600 dpi

        fig = plt.figure(figsize=(size, size), facecolor=self.plot_bg)
        ax  = fig.add_axes([0, 0, 1, 1], facecolor=self.plot_bg)
        ax.set_aspect("equal")
        ax.axis("off")

        self._draw_spiral(ax)

        pal   = self.palette[:4]
        rev   = "r" if self.reverse_pal else ""
        bg    = "lite" if self.plot_bg == LIGHT_BG else "dark"
        fname = (
            f"spiral_{self.angle:.2f}deg"
            f"_p{self.r_exp:.2f}"
            f"_N{self.n}"
            f"_{pal}{rev}"
            f"_{bg}_{dpi}dpi.png"
        )
        fig.savefig(fname, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        self.btn_export.label.set_text("Saved \u2713")
        self.fig.canvas.draw_idle()

    # ── Launch ────────────────────────────────────────────────────────────────

    def show(self):
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    artist = SpiralArtist()
    artist.show()
