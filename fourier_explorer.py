#!/usr/bin/env python3
"""
Fourier Explorer

Any closed curve can be decomposed into a sum of rotating circles (epicycles).
This tool computes the DFT of preset shapes, animates the epicycle reconstruction,
and shows the Fourier space: amplitude spectrum and the distribution of DFT
coefficients in the complex plane.

Controls:
  Harmonics  — how many frequency components to use (largest amplitude first)
  Speed      — animation rate (cycles per second)
  Preset     — target shape
  Pause/Play — freeze the animation
  Circles    — toggle the rotating-circle overlay
  Reset      — restart the trace
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # explicit backend — avoids Qt/Wayland issues on WSL2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle

# ── Dark theme ────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'figure.facecolor':  '#0c0c1a',
    'axes.facecolor':    '#0c0c1a',
    'axes.edgecolor':    '#252545',
    'axes.labelcolor':   '#9090b8',
    'xtick.color':       '#555575',
    'ytick.color':       '#555575',
    'text.color':        '#c0c0e0',
    'grid.color':        '#141430',
    'grid.linewidth':    0.5,
})

BG       = '#0c0c1a'
C_ORIG   = '#1e1e48'   # faint reference curve
C_TRACE  = '#dde0ff'   # drawn path
C_CIRCLE = '#1e2e70'   # epicycle rings
C_ARM    = '#4060b8'   # radial arms
C_TIP    = '#ff5050'   # pen tip dot
C_SPEC   = '#1e2e60'   # unselected spectrum bars
C_SEL    = '#5070e8'   # selected spectrum bars
C_ACCENT = '#5070e8'

N_SAMPLES = 512
MAX_TERMS = 100        # hard cap on epicycle count


# ── Preset curves ─────────────────────────────────────────────────────────────

def _norm(x, y):
    s = max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-9)
    return x / s, y / s


def c_heart(N=N_SAMPLES):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    return _norm(x, y)


def c_square(N=N_SAMPLES):
    n = N // 4
    s = np.linspace(-1, 1, n, endpoint=False)
    x = np.concatenate([ s,              np.full(n,  1), s[::-1],        np.full(n, -1)])
    y = np.concatenate([ np.full(n, -1), s,              np.full(n, 1),  s[::-1]      ])
    return x[:N], y[:N]


def c_star(N=N_SAMPLES, pts=5):
    n_verts = 2 * pts
    R, r = 1.0, 0.38
    vx = np.array([np.cos(i * np.pi / pts - np.pi/2) * (R if i % 2 == 0 else r)
                   for i in range(n_verts)])
    vy = np.array([np.sin(i * np.pi / pts - np.pi/2) * (R if i % 2 == 0 else r)
                   for i in range(n_verts)])
    t    = np.linspace(0, n_verts, N, endpoint=False)
    idx  = t.astype(int) % n_verts
    frac = t - np.floor(t)
    nxt  = (idx + 1) % n_verts
    return vx[idx] + (vx[nxt] - vx[idx]) * frac, vy[idx] + (vy[nxt] - vy[idx]) * frac


def c_triangle(N=N_SAMPLES):
    vx = np.array([np.cos(np.pi/2 + i * 2*np.pi/3) for i in range(3)])
    vy = np.array([np.sin(np.pi/2 + i * 2*np.pi/3) for i in range(3)])
    t    = np.linspace(0, 3, N, endpoint=False)
    idx  = t.astype(int) % 3
    frac = t - np.floor(t)
    nxt  = (idx + 1) % 3
    return vx[idx] + (vx[nxt] - vx[idx]) * frac, vy[idx] + (vy[nxt] - vy[idx]) * frac


def c_lissajous(N=N_SAMPLES):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.sin(3*t + np.pi/2), np.sin(2*t)


def c_spirograph(N=N_SAMPLES):
    R, r, d = 5, 3, 4
    revs = r // np.gcd(R, r)
    t = np.linspace(0, 2 * np.pi * revs, N, endpoint=False)
    x = (R - r) * np.cos(t) + d * np.cos((R - r) / r * t)
    y = (R - r) * np.sin(t) - d * np.sin((R - r) / r * t)
    return _norm(x, y)


def c_epitrochoid(N=N_SAMPLES):
    R, r, d = 5, 2, 3
    revs = R // np.gcd(R, r)
    t = np.linspace(0, 2 * np.pi * revs, N, endpoint=False)
    x = (R + r) * np.cos(t) - d * np.cos((R + r) / r * t)
    y = (R + r) * np.sin(t) - d * np.sin((R + r) / r * t)
    return _norm(x, y)


def c_astroid(N=N_SAMPLES):
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.cos(t) ** 3, np.sin(t) ** 3


PRESETS = {
    'Heart':       c_heart,
    'Square':      c_square,
    'Star':        c_star,
    'Triangle':    c_triangle,
    'Lissajous':   c_lissajous,
    'Spirograph':  c_spirograph,
    'Epitrochoid': c_epitrochoid,
    'Astroid':     c_astroid,
}
PRESET_NAMES = list(PRESETS.keys())


# ── FFT helpers ───────────────────────────────────────────────────────────────

def fft_sorted(x, y):
    """DFT of z = x+iy, returned sorted largest-amplitude first."""
    z = x + 1j * y
    N = len(z)
    C     = np.fft.fft(z) / N
    freqs = (np.fft.fftfreq(N) * N).astype(int)   # integer: 0,1,…,N/2,-(N/2-1),…,-1
    order = np.argsort(np.abs(C))[::-1]
    return C[order], freqs[order]


def epicycle_chain(C_sel, f_sel, p):
    """(x, y) joint positions at normalised time p ∈ [0, 1)."""
    pos = [(0.0, 0.0)]
    cx = cy = 0.0
    for C, f in zip(C_sel, f_sel):
        ang = 2 * np.pi * f * p + np.angle(C)
        r   = abs(C)
        cx += r * np.cos(ang)
        cy += r * np.sin(ang)
        pos.append((cx, cy))
    return pos


# ── Application ───────────────────────────────────────────────────────────────

class FourierExplorer:
    def __init__(self):
        self.n_terms = 30
        self.speed   = 1.0      # cycles per (N_SAMPLES animation steps)
        self.circles = True
        self.paused  = False
        self.p       = 0.0      # normalised time [0, 1)
        self.trace   = []       # (x, y) history

        self._setup_figure()
        self._load('Heart')
        self._anim = FuncAnimation(self.fig, self._frame, interval=16, blit=False)
        plt.show()

    # ── Figure & axes ─────────────────────────────────────────────────────────

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor(BG)
        self.fig.suptitle('Fourier Explorer', color=C_ACCENT, fontsize=13, y=0.99)

        # Main plot area above the widget strip
        gs = gridspec.GridSpec(
            2, 2,
            left=0.04, right=0.97, bottom=0.23, top=0.97,
            wspace=0.07, hspace=0.38,
            width_ratios=[3, 2],
            height_ratios=[5, 3],
        )
        self.ax_epi   = self.fig.add_subplot(gs[:, 0])    # left col: epicycles
        self.ax_amp   = self.fig.add_subplot(gs[0, 1])    # top-right: amplitude spectrum
        self.ax_plane = self.fig.add_subplot(gs[1, 1])    # bot-right: complex plane

        self._style(self.ax_epi,   'Epicycle reconstruction')
        self._style(self.ax_amp,   'Amplitude spectrum  |Cₖ|',
                    xlabel='Frequency index  k')
        self._style(self.ax_plane, 'DFT coefficients — complex plane')

        self.ax_epi.set_aspect('equal')
        self.ax_plane.set_aspect('equal')
        self.ax_plane.axhline(0, color='#202040', lw=0.6)
        self.ax_plane.axvline(0, color='#202040', lw=0.6)

        self._add_widgets()
        self._init_artists()

    def _style(self, ax, title, xlabel=None):
        ax.set_facecolor(BG)
        ax.set_title(title, color='#8888b8', fontsize=10, pad=4)
        for sp in ax.spines.values():
            sp.set_color('#202040')
        ax.tick_params(colors='#505070', labelsize=8)
        ax.grid(True)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=8, color='#606080')

    def _add_widgets(self):
        wbg = '#101028'
        wfg = C_ACCENT

        # ── Sliders ───────────────────────────────────────────────────────────
        ax_n   = self.fig.add_axes([0.07, 0.155, 0.40, 0.022], facecolor=wbg)
        ax_spd = self.fig.add_axes([0.07, 0.110, 0.40, 0.022], facecolor=wbg)

        self.sl_n   = Slider(ax_n,   'Harmonics', 1, MAX_TERMS,
                             valinit=self.n_terms, valstep=1,   color=wfg)
        self.sl_spd = Slider(ax_spd, 'Speed',     0.1, 8.0,
                             valinit=self.speed,   valstep=0.1, color=wfg)
        for sl in [self.sl_n, self.sl_spd]:
            sl.label.set_color('#b0b0d0')
            sl.valtext.set_color(wfg)

        # ── Buttons ───────────────────────────────────────────────────────────
        bkw = dict(color='#181838', hovercolor='#222255')
        ax_pp  = self.fig.add_axes([0.07,  0.055, 0.09,  0.040])
        ax_cir = self.fig.add_axes([0.175, 0.055, 0.115, 0.040])
        ax_rst = self.fig.add_axes([0.305, 0.055, 0.09,  0.040])

        self.btn_pp  = Button(ax_pp,  'Pause',      **bkw)
        self.btn_cir = Button(ax_cir, 'Circles: ON', **bkw)
        self.btn_rst = Button(ax_rst, 'Reset',       **bkw)

        for b in [self.btn_pp, self.btn_cir, self.btn_rst]:
            b.label.set_color('#a8a8f0')
            b.label.set_fontsize(9)

        # ── Preset radio ──────────────────────────────────────────────────────
        ax_rad = self.fig.add_axes([0.59, 0.005, 0.13, 0.21], facecolor=wbg)
        ax_rad.set_title('Preset', color='#8888c0', fontsize=9, pad=2)
        self.radio = RadioButtons(ax_rad, PRESET_NAMES, active=0)
        for lbl in self.radio.labels:
            lbl.set_fontsize(8)
            lbl.set_color('#b0b0d0')

        # ── Wire up ───────────────────────────────────────────────────────────
        self.sl_n.on_changed(lambda v: self._set_terms(int(v)))
        self.sl_spd.on_changed(lambda v: setattr(self, 'speed', v))
        self.btn_pp.on_clicked(self._toggle_pause)
        self.btn_cir.on_clicked(self._toggle_circles)
        self.btn_rst.on_clicked(lambda _: self._reset())
        self.radio.on_clicked(self._load)

    def _init_artists(self):
        ax = self.ax_epi
        self.ln_orig,  = ax.plot([], [], color=C_ORIG,  lw=1,   zorder=1)
        self.ln_trace, = ax.plot([], [], color=C_TRACE, lw=1.2, zorder=4)
        self.dot_tip,  = ax.plot([], [], 'o', color=C_TIP, ms=5, zorder=6)

        self.circ_patches = [
            Circle((0, 0), 0, fill=False, color=C_CIRCLE, lw=0.7, alpha=0.7, zorder=2)
            for _ in range(MAX_TERMS)
        ]
        for cp in self.circ_patches:
            ax.add_patch(cp)

        self.arm_lines = [
            ax.plot([], [], color=C_ARM, lw=0.9, zorder=3)[0]
            for _ in range(MAX_TERMS)
        ]

        # Spectrum bars and complex-plane scatter built on first load
        self._spec_bars  = None
        self._spec_freqs = None

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self, name):
        self._reset()
        x, y = PRESETS[name](N_SAMPLES)
        self.ox, self.oy = x, y
        self.C_all, self.f_all = fft_sorted(x, y)
        self._set_terms(int(self.sl_n.val))
        self._build_spectrum()
        self._build_complex_plane()

    def _set_terms(self, n):
        self.n_terms = max(1, min(n, len(self.C_all)))
        self.C_sel   = self.C_all[:self.n_terms]
        self.f_sel   = self.f_all[:self.n_terms]
        self._reset()
        self._update_spectrum_colors()
        self._build_complex_plane()

    def _reset(self):
        self.p     = 0.0
        self.trace = []

    # ── Spectrum (amplitude) ──────────────────────────────────────────────────

    def _build_spectrum(self):
        ax = self.ax_amp
        ax.cla()
        self._style(ax, 'Amplitude spectrum  |Cₖ|', xlabel='Frequency index  k')

        N = N_SAMPLES
        # Display ±SHOW_W around zero (where signal energy concentrates)
        SHOW_W = min(80, N // 4)
        C_raw  = np.fft.fft(self.ox + 1j * self.oy) / N
        freqs  = (np.fft.fftfreq(N) * N).astype(int)
        mask   = np.abs(freqs) <= SHOW_W
        f_show = freqs[mask]
        a_show = np.abs(C_raw[mask])

        sel_set = set(self.f_sel.tolist())
        colors  = [C_SEL if int(f) in sel_set else C_SPEC for f in f_show]

        self._spec_bars  = ax.bar(f_show, a_show, color=colors, width=1, linewidth=0)
        self._spec_freqs = f_show
        ax.set_xlim(-SHOW_W - 1, SHOW_W + 1)

        # Energy captured annotation
        e_sel = float(np.sum(np.abs(self.C_sel) ** 2))
        e_all = float(np.sum(np.abs(self.C_all) ** 2))
        pct   = 100 * e_sel / max(e_all, 1e-15)
        ax.text(0.99, 0.97, f'{self.n_terms} terms · {pct:.1f}% energy',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='#8888c0')

        self.fig.canvas.draw_idle()

    def _update_spectrum_colors(self):
        if self._spec_bars is None:
            return
        sel_set = set(self.f_sel.tolist())
        for bar, f in zip(self._spec_bars, self._spec_freqs):
            bar.set_facecolor(C_SEL if int(f) in sel_set else C_SPEC)

        # Update energy annotation: easiest is to rebuild the text
        ax = self.ax_amp
        # Remove old text annotations
        for txt in ax.texts:
            txt.remove()
        e_sel = float(np.sum(np.abs(self.C_sel) ** 2))
        e_all = float(np.sum(np.abs(self.C_all) ** 2))
        pct   = 100 * e_sel / max(e_all, 1e-15)
        ax.text(0.99, 0.97, f'{self.n_terms} terms · {pct:.1f}% energy',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='#8888c0')
        self.fig.canvas.draw_idle()

    # ── Complex plane (Fourier space) ─────────────────────────────────────────

    def _build_complex_plane(self):
        ax = self.ax_plane
        ax.cla()
        self._style(ax, 'DFT coefficients — complex plane')
        ax.set_aspect('equal')
        ax.axhline(0, color='#202040', lw=0.6)
        ax.axvline(0, color='#202040', lw=0.6)
        ax.set_xlabel('Re(Cₖ)', fontsize=8, color='#606080')
        ax.set_ylabel('Im(Cₖ)', fontsize=8, color='#606080')

        # All unselected coefficients — dim small dots
        C_unsel = self.C_all[self.n_terms:]
        if len(C_unsel):
            ax.scatter(C_unsel.real, C_unsel.imag,
                       s=4, color='#1c1c40', alpha=0.8, zorder=1, linewidths=0)

        # Selected coefficients — bright, with spokes to origin
        if len(self.C_sel):
            for C in self.C_sel:
                ax.plot([0, C.real], [0, C.imag],
                        color='#1e2e68', lw=0.6, alpha=0.7, zorder=2)
            ax.scatter(self.C_sel.real, self.C_sel.imag,
                       s=18, color=C_SEL, zorder=3, linewidths=0)

        # Axes limits
        max_r = max(float(np.max(np.abs(self.C_all[:self.n_terms]))), 0.01) * 1.4
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)

        self.fig.canvas.draw_idle()

    # ── Animation frame ───────────────────────────────────────────────────────

    def _frame(self, _):
        if self.paused:
            return

        # Advance: speed=1 → one full cycle per N_SAMPLES frames
        self.p = (self.p + self.speed / N_SAMPLES) % 1.0

        pos = epicycle_chain(self.C_sel, self.f_sel, self.p)
        tip = pos[-1]
        self.trace.append(tip)
        if len(self.trace) > N_SAMPLES:
            self.trace = self.trace[-N_SAMPLES:]

        # Reference curve (closed)
        self.ln_orig.set_data(
            np.append(self.ox, self.ox[0]),
            np.append(self.oy, self.oy[0]),
        )

        # Traced path
        if self.trace:
            tx, ty = zip(*self.trace)
            self.ln_trace.set_data(tx, ty)

        # Epicycle circles and arms
        n_active = len(pos) - 1
        for i in range(MAX_TERMS):
            vis = self.circles and i < n_active
            self.circ_patches[i].set_visible(vis)
            self.arm_lines[i].set_visible(vis)
            if vis:
                cx, cy = pos[i]
                r = abs(self.C_sel[i])
                self.circ_patches[i].set_center((cx, cy))
                self.circ_patches[i].set_radius(r)
                self.arm_lines[i].set_data(
                    [pos[i][0], pos[i + 1][0]],
                    [pos[i][1], pos[i + 1][1]],
                )

        self.dot_tip.set_data([tip[0]], [tip[1]])

        # Fit the epicycle view to the outermost circle
        reach = sum(abs(C) for C in self.C_sel) * 1.12
        lim   = max(reach, 1.05)
        self.ax_epi.set_xlim(-lim, lim)
        self.ax_epi.set_ylim(-lim, lim)

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _toggle_pause(self, _):
        self.paused = not self.paused
        self.btn_pp.label.set_text('Play ' if self.paused else 'Pause')

    def _toggle_circles(self, _):
        self.circles = not self.circles
        self.btn_cir.label.set_text(f'Circles: {"ON" if self.circles else "OFF"}')


if __name__ == '__main__':
    FourierExplorer()
