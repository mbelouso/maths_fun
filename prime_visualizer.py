#!/usr/bin/env python3
"""
Prime Spiral Visualizer
=======================
Arrange positive integers in one of three spiral patterns and colour them by
primality (or other residue properties) to reveal hidden geometric structure.

Spirals
-------
  Ulam   – integers laid out in a rectangular outward spiral on a grid
  Sacks  – Robert Sacks' Archimedean spiral: k at (√k · cos(2π√k), √k · sin(2π√k))
  Vogel  – sunflower / golden-angle spiral: k at (√k · cos(kφ), √k · sin(kφ))

Controls
--------
  Spiral type  : Ulam | Sacks | Vogel
  Colour mode  : Primes | Mod 6 | Mod 30 | Twin primes
  Size slider  : adjusts number of integers displayed

Usage
-----
    python prime_visualizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, RadioButtons, Button
import matplotlib.gridspec as gridspec

# ── Colour palette ─────────────────────────────────────────────────────────────

BG         = "#0d0d1a"
PRIME_C    = "#FFD700"   # gold
COMP_C     = "#1a1a2e"   # dark navy
TWIN_C     = "#FF4081"   # hot pink
CTRL_BG    = "#12122a"
CTRL_FG    = "#ccccee"


# ── Sieve ──────────────────────────────────────────────────────────────────────

def sieve(limit: int) -> np.ndarray:
    """Return bool array where [n] is True iff n is prime (0-indexed)."""
    ip = np.ones(limit + 1, dtype=bool)
    ip[:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if ip[i]:
            ip[i * i :: i] = False
    return ip


# ── Spiral generators ──────────────────────────────────────────────────────────

def ulam_grid(side: int) -> np.ndarray:
    """
    Return (side × side) array of ints arranged in the Ulam spiral.
    side is forced to the next odd number ≥ side so the centre is well-defined.
    """
    if side % 2 == 0:
        side += 1
    n = side
    grid = np.zeros((n, n), dtype=np.int32)
    cx = cy = n // 2
    x, y = cx, cy
    num = 1
    grid[y, x] = num
    num += 1
    step = 1
    while num <= n * n:
        for _ in range(step):          # right
            if num > n * n: break
            x += 1
            if 0 <= x < n and 0 <= y < n:
                grid[y, x] = num
            num += 1
        for _ in range(step):          # up
            if num > n * n: break
            y -= 1
            if 0 <= x < n and 0 <= y < n:
                grid[y, x] = num
            num += 1
        step += 1
        for _ in range(step):          # left
            if num > n * n: break
            x -= 1
            if 0 <= x < n and 0 <= y < n:
                grid[y, x] = num
            num += 1
        for _ in range(step):          # down
            if num > n * n: break
            y += 1
            if 0 <= x < n and 0 <= y < n:
                grid[y, x] = num
            num += 1
        step += 1
    return grid


def sacks_coords(n: int):
    """Sacks spiral coordinates for integers 1…n."""
    k = np.arange(1, n + 1, dtype=float)
    r = np.sqrt(k)
    theta = 2.0 * np.pi * r
    return r * np.cos(theta), r * np.sin(theta)


def vogel_coords(n: int):
    """Vogel sunflower spiral coordinates for integers 1…n."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))   # ≈ 137.508°
    k = np.arange(1, n + 1, dtype=float)
    r = np.sqrt(k)
    theta = k * golden_angle
    return r * np.cos(theta), r * np.sin(theta)


# ── Colour value computation ───────────────────────────────────────────────────

def compute_colors(nums: np.ndarray, ip: np.ndarray, mode: str):
    """
    Map integer array `nums` to RGBA colours according to `mode`.

    Returns
    -------
    colors : (N, 4) float32 RGBA array
    """
    N = len(nums)
    rgba = np.zeros((N, 4), dtype=np.float32)

    valid = (nums > 0) & (nums < len(ip))
    v = nums[valid]

    if mode == "Primes":
        gold  = np.array(mcolors.to_rgba(PRIME_C),  dtype=np.float32)
        dark  = np.array(mcolors.to_rgba(COMP_C),   dtype=np.float32)
        dark[3] = 0.25                                # make composites dimmer
        rgba[valid]  = np.where(ip[v, None], gold, dark)
        rgba[~valid] = dark

    elif mode == "Mod 6":
        cmap = plt.cm.hsv
        rgba[valid]  = cmap((v % 6) / 6.0).astype(np.float32)
        rgba[~valid] = [0.1, 0.1, 0.1, 0.3]
        # Dim composites so primes still pop
        is_comp = ~ip[v]
        rgba_valid = rgba[valid]
        rgba_valid[is_comp, 3] = 0.18
        rgba[valid] = rgba_valid

    elif mode == "Mod 30":
        cmap = plt.cm.turbo
        rgba[valid]  = cmap((v % 30) / 30.0).astype(np.float32)
        rgba[~valid] = [0.1, 0.1, 0.1, 0.3]
        is_comp = ~ip[v]
        rgba_valid = rgba[valid]
        rgba_valid[is_comp, 3] = 0.12
        rgba[valid] = rgba_valid

    elif mode == "Twin primes":
        twin = np.zeros(len(ip), dtype=bool)
        primes = np.where(ip)[0]
        mask_p2 = primes + 2
        ok = mask_p2 < len(ip)
        twin_p = primes[ok][ip[mask_p2[ok]]]
        twin[twin_p] = True
        twin[twin_p + 2] = True

        gold  = np.array(mcolors.to_rgba(PRIME_C),  dtype=np.float32)
        pink  = np.array(mcolors.to_rgba(TWIN_C),   dtype=np.float32)
        dark  = np.array(mcolors.to_rgba(COMP_C),   dtype=np.float32)
        dark[3] = 0.20

        colors_v = np.where(
            twin[v, None], pink,
            np.where(ip[v, None], gold, dark)
        ).astype(np.float32)
        rgba[valid]  = colors_v
        rgba[~valid] = dark

    return rgba


# ── Main visualizer class ──────────────────────────────────────────────────────

class PrimeSpiralViz:
    # Default parameters
    DEFAULT_SPIRAL = "Ulam"
    DEFAULT_MODE   = "Primes"
    DEFAULT_SIDE   = 101      # Ulam grid side (101×101 = 10201 numbers)
    DEFAULT_N      = 3000     # scatter point count for Sacks/Vogel
    MAX_SIDE       = 1001
    MAX_N          = 1_000_000

    # Pre-compute primes up to the largest we'll ever need
    _ip = sieve(MAX_SIDE ** 2 + 10)

    def __init__(self):
        self.spiral = self.DEFAULT_SPIRAL
        self.mode   = self.DEFAULT_MODE
        self.side   = self.DEFAULT_SIDE      # current Ulam side
        self.n_pts  = self.DEFAULT_N         # current scatter point count

        # State for dynamic zoom labels
        self._label_artists = []
        self._grid = self._xs = self._ys = self._nums = self._ip_cur = None
        self._drawing  = False   # blocks _on_zoom re-entrance during _draw()
        self._last_lim = None    # last (xlim, ylim) seen by _on_zoom

        self._build_figure()
        self.ax.callbacks.connect('xlim_changed', self._on_zoom)
        self.ax.callbacks.connect('ylim_changed', self._on_zoom)
        self._draw()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_figure(self):
        plt.rcParams.update({
            "figure.facecolor":  BG,
            "axes.facecolor":    BG,
            "axes.edgecolor":    "#334",
            "text.color":        CTRL_FG,
            "xtick.color":       "#556",
            "ytick.color":       "#556",
        })

        self.fig = plt.figure(figsize=(14, 9), facecolor=BG)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Prime Spiral Visualizer")

        # Main grid: plot | controls
        outer = gridspec.GridSpec(1, 2, width_ratios=[3, 1],
                                  left=0.04, right=0.98,
                                  top=0.95, bottom=0.05, wspace=0.04)
        self.ax = self.fig.add_subplot(outer[0], facecolor=BG)

        ctrl = gridspec.GridSpecFromSubplotSpec(
            7, 1, subplot_spec=outer[1],
            hspace=0.5, height_ratios=[0.5, 2, 0.3, 2, 0.3, 1.2, 0.5]
        )

        # ── Spiral type radio ────────────────────────────────────────────────
        ax_sp_lbl = self.fig.add_subplot(ctrl[0], facecolor=CTRL_BG)
        ax_sp_lbl.set_axis_off()
        ax_sp_lbl.text(0.5, 0.5, "Spiral type", ha="center", va="center",
                       color=CTRL_FG, fontsize=9, fontweight="bold")

        ax_spiral = self.fig.add_subplot(ctrl[1], facecolor=CTRL_BG)
        self.radio_spiral = RadioButtons(
            ax_spiral, ["Ulam", "Sacks", "Vogel"],
            active=0,
            activecolor=PRIME_C,
        )
        self._style_radio(self.radio_spiral)
        self.radio_spiral.on_clicked(self._on_spiral)

        # ── Colour mode radio ────────────────────────────────────────────────
        ax_cm_lbl = self.fig.add_subplot(ctrl[2], facecolor=CTRL_BG)
        ax_cm_lbl.set_axis_off()
        ax_cm_lbl.text(0.5, 0.5, "Colour mode", ha="center", va="center",
                       color=CTRL_FG, fontsize=9, fontweight="bold")

        ax_mode = self.fig.add_subplot(ctrl[3], facecolor=CTRL_BG)
        self.radio_mode = RadioButtons(
            ax_mode, ["Primes", "Mod 6", "Mod 30", "Twin primes"],
            active=0,
            activecolor=PRIME_C,
        )
        self._style_radio(self.radio_mode)
        self.radio_mode.on_clicked(self._on_mode)

        # ── Size slider ──────────────────────────────────────────────────────
        ax_sl_lbl = self.fig.add_subplot(ctrl[4], facecolor=CTRL_BG)
        ax_sl_lbl.set_axis_off()
        ax_sl_lbl.text(0.5, 0.5, "Size", ha="center", va="center",
                       color=CTRL_FG, fontsize=9, fontweight="bold")

        ax_slider = self.fig.add_subplot(ctrl[5], facecolor=CTRL_BG)
        self.slider = Slider(
            ax_slider, label="", valmin=21, valmax=self.MAX_SIDE, valinit=self.side,
            valstep=2,
            color=PRIME_C, track_color="#334",
        )
        self.slider.label.set_color(CTRL_FG)
        self.slider.valtext.set_color(CTRL_FG)
        self.slider.on_changed(self._on_size)

        # ── Reset button ─────────────────────────────────────────────────────
        ax_btn = self.fig.add_subplot(ctrl[6], facecolor=CTRL_BG)
        self.btn_reset = Button(ax_btn, "Reset zoom",
                                color=CTRL_BG, hovercolor="#222244")
        self.btn_reset.label.set_color(CTRL_FG)
        self.btn_reset.on_clicked(self._on_reset)

        # ── Title ────────────────────────────────────────────────────────────
        self.title = self.fig.text(
            0.38, 0.975, "", ha="center", va="top",
            color=PRIME_C, fontsize=13, fontweight="bold"
        )

    @staticmethod
    def _style_radio(radio):
        for lbl in radio.labels:
            lbl.set_color(CTRL_FG)
            lbl.set_fontsize(9)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        self._drawing = True          # block _on_zoom during redraw
        self._last_lim = None         # force label refresh after redraw
        self._label_artists.clear()   # artists are removed by ax.cla() below
        self.ax.cla()
        self.ax.set_facecolor(BG)
        self.ax.set_aspect("equal")
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        if self.spiral == "Ulam":
            self._draw_ulam()
        elif self.spiral == "Sacks":
            self._draw_scatter(sacks_coords)
        else:
            self._draw_scatter(vogel_coords)

        self.title.set_text(
            f"{self.spiral} Spiral  ·  {self.mode}  ·  "
            f"{'%d×%d' % (self.side, self.side) if self.spiral == 'Ulam' else '%d pts' % self.n_pts}"
        )
        self._drawing = False         # allow _on_zoom again
        self._update_labels()
        self.fig.canvas.draw_idle()

    def _draw_ulam(self):
        n = self.side
        grid = ulam_grid(n)
        limit = int(grid.max()) + 1
        ip = self._ip[:limit] if limit <= len(self._ip) else sieve(limit)

        self._grid, self._ip_cur = grid, ip

        flat = grid.ravel()
        rgba = compute_colors(flat, ip, self.mode)
        img  = rgba.reshape(n, n, 4)

        # Gamma-brighten the prime channel slightly
        if self.mode == "Primes":
            img[..., :3] = np.clip(img[..., :3] ** 0.8, 0, 1)

        self.ax.imshow(img, interpolation="nearest", origin="upper")

    def _on_zoom(self, _ax):
        """Called on xlim/ylim change (zoom / pan). Refresh number labels."""
        if self._drawing:
            return
        lim = (tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim()))
        if lim == self._last_lim:
            return
        self._last_lim = lim
        self._clear_labels()
        self._update_labels()
        # No draw_idle() here — the zoom/pan handler already schedules its own
        # redraw; calling draw_idle() again triggers another xlim_changed which
        # would clear the labels we just drew.

    def _clear_labels(self):
        for txt in self._label_artists:
            txt.remove()
        self._label_artists.clear()

    def _update_labels(self):
        if self.spiral == "Ulam":
            self._draw_ulam_labels()
        else:
            self._draw_scatter_labels()

    def _draw_ulam_labels(self):
        if self._grid is None:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        visible_w = x1 - x0
        visible_h = abs(y1 - y0)
        if visible_w > 40 or visible_h > 40:
            return
        n = self._grid.shape[0]
        col_lo = max(0, int(np.floor(x0)))
        col_hi = min(n - 1, int(np.ceil(x1)))
        row_lo = max(0, int(np.floor(min(y0, y1))))
        row_hi = min(n - 1, int(np.ceil(max(y0, y1))))
        if (col_hi - col_lo + 1) * (row_hi - row_lo + 1) > 400:
            return
        ip = self._ip_cur
        fs = max(4, min(11, int(120 / max(visible_w, 1))))
        for row in range(row_lo, row_hi + 1):
            for col in range(col_lo, col_hi + 1):
                val = int(self._grid[row, col])
                if val == 0:
                    continue
                color = PRIME_C if (val < len(ip) and ip[val]) else "#ffffff"
                txt = self.ax.text(
                    col, row, str(val),
                    ha="center", va="center",
                    fontsize=fs, color=color,
                    alpha=0.9 if (val < len(ip) and ip[val]) else 0.45,
                    fontfamily="monospace"
                )
                self._label_artists.append(txt)

    def _draw_scatter_labels(self):
        if self._xs is None:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        in_view = (
            (self._xs >= x0) & (self._xs <= x1) &
            (self._ys >= min(y0, y1)) & (self._ys <= max(y0, y1))
        )
        count = int(in_view.sum())
        if count == 0 or count > 100:
            return
        ip  = self._ip_cur
        fs  = max(5, min(10, int(80 / max(1, count ** 0.5))))
        for num, x, y in zip(self._nums[in_view], self._xs[in_view], self._ys[in_view]):
            color = PRIME_C if (int(num) < len(ip) and ip[int(num)]) else "#ffffff"
            txt = self.ax.text(
                x, y, str(int(num)),
                ha="center", va="bottom",
                fontsize=fs, color=color,
                alpha=0.9 if (int(num) < len(ip) and ip[int(num)]) else 0.45,
                fontfamily="monospace"
            )
            self._label_artists.append(txt)

    def _draw_scatter(self, coord_fn):
        n = self.n_pts
        limit = n + 10
        ip = self._ip[:limit] if limit <= len(self._ip) else sieve(limit)

        xs, ys = coord_fn(n)
        nums   = np.arange(1, n + 1, dtype=np.int32)
        rgba   = compute_colors(nums, ip, self.mode)

        self._xs, self._ys, self._nums, self._ip_cur = xs, ys, nums, ip

        # Size: primes slightly larger
        is_prime_mask = ip[nums]
        sizes = np.where(is_prime_mask, 8.0, 2.0)

        # Draw composites first (background layer), then primes on top
        comp_mask  = ~is_prime_mask
        prime_mask =  is_prime_mask

        if comp_mask.any():
            self.ax.scatter(
                xs[comp_mask], ys[comp_mask],
                c=rgba[comp_mask], s=sizes[comp_mask],
                linewidths=0, rasterized=True, zorder=1
            )
        if prime_mask.any():
            self.ax.scatter(
                xs[prime_mask], ys[prime_mask],
                c=rgba[prime_mask], s=sizes[prime_mask],
                linewidths=0, rasterized=True, zorder=2
            )

    # ── Widget callbacks ──────────────────────────────────────────────────────

    def _on_spiral(self, label):
        self.spiral = label
        # Sync slider range and value
        if label == "Ulam":
            self.slider.valmin  = 21
            self.slider.valmax  = self.MAX_SIDE
            self.slider.valstep = 2
            self.slider.set_val(self.side if self.side % 2 == 1 else self.side + 1)
        else:
            self.n_pts = min(self.MAX_N, max(1000, (self.side ** 2 // 1000 + 1) * 1000))
            self.slider.valmin  = 200
            self.slider.valmax  = self.MAX_N
            self.slider.valstep = 1000
            self.slider.set_val(self.n_pts)
        self._draw()

    def _on_mode(self, label):
        self.mode = label
        self._draw()

    def _on_size(self, val):
        v = int(val)
        if self.spiral == "Ulam":
            self.side  = v if v % 2 == 1 else v + 1
        else:
            self.n_pts = v
        self._draw()

    def _on_reset(self, _event):
        self._last_lim = None     # force label refresh on next xlim callback
        self.ax.autoscale()
        self.fig.canvas.draw_idle()

    # ── Launch ────────────────────────────────────────────────────────────────

    def show(self):
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    viz = PrimeSpiralViz()
    viz.show()
