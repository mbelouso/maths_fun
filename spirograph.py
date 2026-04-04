#!/usr/bin/env python3
"""
Interactive Spirograph using pygame.

Hypotrochoid: inner gear (r) rolling inside outer gear (R), pen at distance d.
  x(t) = (R-r)*cos(t) + d*cos((R-r)/r * t)
  y(t) = (R-r)*sin(t) - d*sin((R-r)/r * t)

Epitrochoid: rolling gear outside the fixed gear.

Controls: sliders for R, r, d, revolutions, line width, resolution.
Keys: R=redraw, C=clear, A=toggle animate, Esc=quit
"""

import math
from math import gcd, pi, cos, sin
import os
import datetime
import pygame
import matplotlib.cm as mcm

# ── Layout ────────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1920, 1200
PANEL_W = 300
FPS = 60

# Colours
BG_CANVAS  = (10, 10, 18)
BG_PANEL   = (24, 24, 38)
BORDER     = (50, 50, 80)
ACCENT     = (100, 140, 240)
TEXT       = (220, 220, 240)
SUBTEXT    = (120, 120, 160)
SLIDER_BG  = (45, 45, 68)
SLIDER_FG  = (100, 140, 240)
KNOB_COL   = (180, 200, 255)


# ── Math ──────────────────────────────────────────────────────────────────────

def _safe_gcd(a, b):
    a, b = max(1, int(round(abs(a)))), max(1, int(round(abs(b))))
    return gcd(a, b)

def spirograph_points(R, r, d, revolutions, n_points=4000, mode='hypo'):
    """Return list of (x, y) for the spirograph curve."""
    r = max(0.5, r)
    g = _safe_gcd(R, r)
    period = 2 * pi * r / g
    t_max  = period * revolutions
    pts = []
    for i in range(n_points + 1):
        t = t_max * i / n_points
        if mode == 'hypo':
            x = (R - r) * cos(t) + d * cos((R - r) / r * t)
            y = (R - r) * sin(t) - d * sin((R - r) / r * t)
        else:
            x = (R + r) * cos(t) - d * cos((R + r) / r * t)
            y = (R + r) * sin(t) - d * sin((R + r) / r * t)
        pts.append((x, y))
    return pts

def to_screen(x, y, cx, cy, scale):
    return (int(cx + x * scale), int(cy - y * scale))


# ── Colour maps ───────────────────────────────────────────────────────────────

CMAPS = ['viridis', 'plasma', 'magma', 'inferno', 'cividis', 'twilight', 'cool', 'rainbow']

def build_lut(cmap_name, n):
    """Pre-compute n RGB tuples from a matplotlib colormap."""
    cmap = mcm.get_cmap(cmap_name)
    lut = []
    for i in range(n):
        r, g, b, _ = cmap(i / max(n - 1, 1))
        lut.append((int(r * 255), int(g * 255), int(b * 255)))
    return lut


# ── Widgets ───────────────────────────────────────────────────────────────────

class Slider:
    TRACK_H = 6
    KNOB_R  = 8

    def __init__(self, x, y, w, label, vmin, vmax, value, step=1, fmt='{:.0f}'):
        self.rect  = pygame.Rect(x, y, w, 24)
        self.label = label
        self.vmin, self.vmax = vmin, vmax
        self.value = float(value)
        self.step  = step
        self.fmt   = fmt
        self.dragging = False
        self.on_change = None

    @property
    def _track(self):
        r = self.rect
        cy = r.y + r.height // 2
        return pygame.Rect(r.x, cy - self.TRACK_H // 2, r.width, self.TRACK_H)

    def _val_to_x(self, v):
        t = (v - self.vmin) / (self.vmax - self.vmin)
        return int(self._track.x + t * self._track.width)

    def _x_to_val(self, x):
        t = max(0.0, min(1.0, (x - self._track.x) / self._track.width))
        v = self.vmin + t * (self.vmax - self.vmin)
        steps = round((v - self.vmin) / self.step)
        return max(self.vmin, min(self.vmax, self.vmin + steps * self.step))

    def set_value(self, v):
        self.value = max(self.vmin, min(self.vmax, v))

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            kx, ky = self._val_to_x(self.value), self._track.centery
            if math.hypot(ev.pos[0] - kx, ev.pos[1] - ky) <= self.KNOB_R + 4:
                self.dragging = True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            old = self.value
            self.value = self._x_to_val(ev.pos[0])
            if self.value != old and self.on_change:
                self.on_change(self.value)

    def draw(self, surf, font):
        tr = self._track
        # Track
        pygame.draw.rect(surf, SLIDER_BG, tr, border_radius=3)
        fill_w = self._val_to_x(self.value) - tr.x
        if fill_w > 0:
            pygame.draw.rect(surf, SLIDER_FG,
                             pygame.Rect(tr.x, tr.y, fill_w, tr.height), border_radius=3)
        # Knob
        kx, ky = self._val_to_x(self.value), tr.centery
        pygame.draw.circle(surf, KNOB_COL, (kx, ky), self.KNOB_R)
        pygame.draw.circle(surf, ACCENT,   (kx, ky), self.KNOB_R, 2)
        # Labels
        lbl  = font.render(self.label, True, TEXT)
        val  = font.render(self.fmt.format(self.value), True, ACCENT)
        top  = self.rect.y - 16
        surf.blit(lbl, (self.rect.x, top))
        surf.blit(val, (self.rect.right - val.get_width(), top))


class Button:
    def __init__(self, rect, label, toggle=False, active=False):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.toggle = toggle
        self.active = active
        self._hover = False
        self.on_click = None

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(ev.pos)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                if self.toggle:
                    self.active = not self.active
                if self.on_click:
                    self.on_click()
                return True
        return False

    def draw(self, surf, font):
        if self.active:
            base = (60, 180, 100)
        else:
            base = (55, 55, 85)
        col = tuple(min(255, c + 25) for c in base) if self._hover else base
        pygame.draw.rect(surf, col, self.rect, border_radius=5)
        txt = font.render(self.label, True, TEXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))


# ── Gear overlay ──────────────────────────────────────────────────────────────

def draw_gear_overlay(surf, cx, cy, R, r, d, t, scale, mode):
    # Fixed ring
    pygame.draw.circle(surf, (55, 55, 90), (cx, cy), int(R * scale), 1)
    # Rolling gear centre
    if mode == 'hypo':
        gx = int(cx + (R - r) * scale * cos(t))
        gy = int(cy - (R - r) * scale * sin(t))
        gear_angle = -(R - r) / r * t
    else:
        gx = int(cx + (R + r) * scale * cos(t))
        gy = int(cy - (R + r) * scale * sin(t))
        gear_angle = -(R + r) / r * t

    gr = max(2, int(r * scale))
    pygame.draw.circle(surf, (80, 110, 170), (gx, gy), gr, 1)
    # Pen arm
    px = int(gx + d * scale * cos(gear_angle))
    py = int(gy + d * scale * sin(gear_angle))
    pygame.draw.line(surf, (110, 110, 170), (gx, gy), (px, py), 1)
    pygame.draw.circle(surf, (255, 90, 90), (px, py), 5)


# ── Main application ──────────────────────────────────────────────────────────

class SpirographApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption("Interactive Spirograph")
        self.clock = pygame.time.Clock()

        self.font  = pygame.font.SysFont('segoeui', 13)
        self.tfont = pygame.font.SysFont('segoeui', 17, bold=True)

        self.canvas = pygame.Surface((WIN_W - PANEL_W, WIN_H))
        self.canvas.fill(BG_CANVAS)

        # State
        self.mode      = 'hypo'
        self.palette   = 'viridis'
        self.animate   = False
        self.show_gear = False
        self.anim_t    = 0.0
        self.anim_idx  = 0
        self.points    = []
        self.scale     = 1.0
        self.cx = self.cy = 0
        self._export_msg   = ''
        self._export_timer = 0

        self._build_ui()
        self._recompute()

    # ── Canvas helpers ────────────────────────────────────────────────────────

    def _canvas_size(self):
        sw, sh = self.screen.get_size()
        return sw - PANEL_W, sh

    def _panel_x(self):
        return self.screen.get_width() - PANEL_W

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        px = self._panel_x()
        x0 = px + 14
        W  = PANEL_W - 28
        y  = 46

        def mk_sl(label, vmin, vmax, val, step=1, fmt='{:.0f}'):
            nonlocal y
            s = Slider(x0, y + 18, W, label, vmin, vmax, val, step, fmt)
            s.on_change = lambda v: self._on_param_change()
            y += 54
            return s

        self.sl_R   = mk_sl('Outer ring radius  R', 40,  300, 160)
        self.sl_r   = mk_sl('Rolling gear radius r', 5,  200,  60)
        self.sl_d   = mk_sl('Pen offset  d',         0,  300,  80)
        self.sl_rev = mk_sl('Revolutions',           1,   40,   1)
        self.sl_lw  = mk_sl('Line width',            1,    6,   2)
        self.sl_pts = mk_sl('Resolution (pts)',    500, 10000, 4000, step=500)

        # Speed slider does not trigger recompute — read live during animation
        self.sl_speed = Slider(x0, y + 18, W, 'Anim speed (pts/frame)', 1, 300, 20, step=1)
        y += 54

        y += 4
        # ── Colourmap ─────────────────────────────────────────────────────────
        bw = (W - 6) // 4          # 4 per row
        self._pal_btns = {}
        first_pal_y = y
        for i, name in enumerate(CMAPS):
            bx = x0 + (i % 4) * (bw + 2)
            by = y + (i // 4) * 32
            b = Button((bx, by, bw, 27), name, toggle=False)
            b.active = (name == self.palette)
            def _mk_pal(n):
                def cb():
                    self.palette = n
                    for nn, bb in self._pal_btns.items():
                        bb.active = (nn == n)
                    self._recompute()
                return cb
            b.on_click = _mk_pal(name)
            self._pal_btns[name] = b
        y += 68
        self._first_pal_y = first_pal_y

        # ── Mode ──────────────────────────────────────────────────────────────
        hw = (W - 4) // 2
        self.btn_hypo = Button((x0,        y, hw, 27), 'Hypotrochoid', toggle=True, active=True)
        self.btn_epi  = Button((x0+hw+4,   y, hw, 27), 'Epitrochoid',  toggle=True, active=False)
        y += 34

        def _set_hypo():
            self.btn_hypo.active = True; self.btn_epi.active = False
            self.mode = 'hypo'; self._recompute()
        def _set_epi():
            self.btn_epi.active = True; self.btn_hypo.active = False
            self.mode = 'epi'; self._recompute()
        self.btn_hypo.on_click = _set_hypo
        self.btn_epi.on_click  = _set_epi

        # ── Show gears / Animate ──────────────────────────────────────────────
        self.btn_gear  = Button((x0,      y, hw, 27), 'Show Gears', toggle=True)
        self.btn_anim  = Button((x0+hw+4, y, hw, 27), 'Animate',    toggle=True)
        self.btn_gear.on_click = lambda: None
        self.btn_anim.on_click = lambda: setattr(self, 'animate', self.btn_anim.active)
        y += 34

        # ── Clear / Redraw ────────────────────────────────────────────────────
        self.btn_clear  = Button((x0,      y, hw, 27), 'Clear')
        self.btn_redraw = Button((x0+hw+4, y, hw, 27), 'Redraw')
        self.btn_clear.on_click  = self._clear
        self.btn_redraw.on_click = self._recompute
        y += 40

        # ── Presets ───────────────────────────────────────────────────────────
        pw = (W - 6) // 2
        presets = [
            ('Classic',  160, 40,  80, 4, 'hypo'),
            ('Star',     175, 25, 100, 7, 'hypo'),
            ('Flower',   140, 20,  90, 3, 'epi'),
            ('Orbit',    160, 80,  40, 5, 'hypo'),
        ]
        self._preset_btns = []
        self._preset_label_y = y
        y += 18
        for i, (name, R, r, d, rev, m) in enumerate(presets):
            bx = x0 + (i % 2) * (pw + 3)
            by = y + (i // 2) * 32
            b = Button((bx, by, pw, 27), name)
            def _mk_preset(R_, r_, d_, rev_, m_):
                def cb():
                    self.sl_R.set_value(R_); self.sl_r.set_value(r_)
                    self.sl_d.set_value(d_); self.sl_rev.set_value(rev_)
                    self.mode = m_
                    self.btn_hypo.active = (m_ == 'hypo')
                    self.btn_epi.active  = (m_ == 'epi')
                    self._on_param_change()
                return cb
            b.on_click = _mk_preset(R, r, d, rev, m)
            self._preset_btns.append(b)

        # ── Export PNG ────────────────────────────────────────────────────────
        y += 72      # skip past the 2 preset rows
        self._export_label_y = y
        y += 18

        EXPORT_SIZES = [
            ('1080p',  1920, 1080),
            ('1440p',  2560, 1440),
            ('4K',     3840, 2160),
            ('Sq 2K',  2160, 2160),
            ('Sq 4K',  4096, 4096),
        ]
        # Layout: 3 buttons row 1, 2 buttons row 2
        row_layouts = [EXPORT_SIZES[:3], EXPORT_SIZES[3:]]
        self._export_btns = []
        for row_i, row in enumerate(row_layouts):
            n_cols = len(row)
            ebw = (W - (n_cols - 1) * 3) // n_cols
            for col_i, (lbl, ew, eh) in enumerate(row):
                bx = x0 + col_i * (ebw + 3)
                by = y + row_i * 32
                b = Button((bx, by, ebw, 27), lbl)
                def _mk_exp(w_, h_):
                    return lambda: self._export_png(w_, h_)
                b.on_click = _mk_exp(ew, eh)
                self._export_btns.append(b)
        y += 68

        self._all_buttons = (
            list(self._pal_btns.values()) +
            [self.btn_hypo, self.btn_epi, self.btn_gear, self.btn_anim,
             self.btn_clear, self.btn_redraw] +
            self._preset_btns + self._export_btns
        )
        self._all_sliders = [self.sl_R, self.sl_r, self.sl_d, self.sl_rev, self.sl_lw, self.sl_pts, self.sl_speed]

    # ── Parameter helpers ─────────────────────────────────────────────────────

    def _read_params(self):
        self.R          = self.sl_R.value
        self.r          = max(0.5, self.sl_r.value)
        self.d          = self.sl_d.value
        self.revolutions = int(self.sl_rev.value)
        self.line_width  = int(self.sl_lw.value)
        self.n_points    = int(self.sl_pts.value)

    def _on_param_change(self):
        self._recompute()

    def _recompute(self):
        self._read_params()
        cw, ch = self._canvas_size()

        pts = spirograph_points(self.R, self.r, self.d,
                                self.revolutions, self.n_points, self.mode)
        self.points = pts

        max_ext = max((abs(v) for p in pts for v in p), default=1.0)
        margin  = 40
        self.scale = min((cw - margin * 2) / 2, (ch - margin * 2) / 2) / max(max_ext, 1e-9)
        self.cx, self.cy = cw // 2, ch // 2

        if len(self.canvas.get_size()) == 0 or self.canvas.get_size() != (cw, ch):
            self.canvas = pygame.Surface((cw, ch))
        self.color_lut = build_lut(self.palette, len(pts))
        self.canvas.fill(BG_CANVAS)
        self._draw_full()
        self.anim_t = 0.0; self.anim_idx = 0

    def _clear(self):
        self.canvas.fill(BG_CANVAS)
        self.anim_t = 0.0; self.anim_idx = 0

    def _draw_full(self):
        pts = self.points
        n   = len(pts)
        if n < 2:
            return
        lw  = self.line_width
        cx, cy, sc = self.cx, self.cy, self.scale
        lut = self.color_lut
        for i in range(n - 1):
            p1 = to_screen(*pts[i],   cx, cy, sc)
            p2 = to_screen(*pts[i+1], cx, cy, sc)
            pygame.draw.line(self.canvas, lut[i], p1, p2, lw)

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_png(self, w, h):
        """Render the spirograph at (w×h) and save as PNG to cwd."""
        pts = self.points
        if len(pts) < 2:
            return

        # Scale to fit with margin
        max_ext = max((abs(v) for p in pts for v in p), default=1.0)
        margin  = max(w, h) * 0.04
        ex_scale = min((w - margin * 2) / 2, (h - margin * 2) / 2) / max(max_ext, 1e-9)
        ecx, ecy = w // 2, h // 2

        # Line width scaled relative to the live canvas
        lw = max(1, round(self.line_width * ex_scale / max(self.scale, 1e-9)))

        surf = pygame.Surface((w, h))
        surf.fill(BG_CANVAS)

        lut = build_lut(self.palette, len(pts))
        n   = len(pts)
        for i in range(n - 1):
            p1 = to_screen(*pts[i],   ecx, ecy, ex_scale)
            p2 = to_screen(*pts[i+1], ecx, ecy, ex_scale)
            pygame.draw.line(surf, lut[i], p1, p2, lw)

        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name = (f"spiro_R{self.R:.0f}_r{self.r:.0f}_d{self.d:.0f}"
                f"_rev{self.revolutions}_{self.mode}_{self.palette}_{w}x{h}_{ts}.png")
        path = os.path.join(os.getcwd(), name)
        pygame.image.save(surf, path)

        self._export_msg   = f"Saved: {name}"
        self._export_timer = FPS * 4   # show for 4 seconds

    # ── Animation ─────────────────────────────────────────────────────────────

    def _anim_step(self):
        pts = self.points
        n   = len(pts)
        if n < 2:
            return
        speed = max(1, int(self.sl_speed.value))
        end   = min(self.anim_idx + speed, n - 1)

        lw  = self.line_width
        cx, cy, sc = self.cx, self.cy, self.scale
        lut = self.color_lut
        for i in range(self.anim_idx, end):
            pygame.draw.line(self.canvas, lut[i],
                             to_screen(*pts[i],   cx, cy, sc),
                             to_screen(*pts[i+1], cx, cy, sc), lw)

        self.anim_idx = end
        if self.anim_idx >= n - 1:
            self.anim_idx = 0
            self.canvas.fill(BG_CANVAS)

        # Gear angle
        g = _safe_gcd(self.R, self.r)
        period = 2 * pi * self.r / g
        t_max  = period * self.revolutions
        self.anim_t = (self.anim_idx / (n - 1)) * t_max

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_panel(self):
        sw, sh = self.screen.get_size()
        px = self._panel_x()
        pygame.draw.rect(self.screen, BG_PANEL, (px, 0, PANEL_W, sh))
        pygame.draw.line(self.screen, BORDER, (px, 0), (px, sh), 1)

        # Title
        title = self.tfont.render("Spirograph", True, ACCENT)
        self.screen.blit(title, (px + (PANEL_W - title.get_width()) // 2, 12))

        # Sliders
        for sl in self._all_sliders:
            sl.draw(self.screen, self.font)

        # Colourmap label
        pal_lbl = self.font.render("Colourmap", True, SUBTEXT)
        self.screen.blit(pal_lbl, (px + 14, self._first_pal_y - 14))

        # Preset label
        pre_lbl = self.font.render("Presets", True, SUBTEXT)
        self.screen.blit(pre_lbl, (px + 14, self._preset_label_y + 2))

        # Export label
        exp_lbl = self.font.render("Export PNG", True, SUBTEXT)
        self.screen.blit(exp_lbl, (px + 14, self._export_label_y + 2))

        for b in self._all_buttons:
            b.draw(self.screen, self.font)

        # Export flash message
        if self._export_msg:
            msg = self.font.render(self._export_msg, True, (120, 220, 120))
            # Word-wrap by splitting at "Saved: " and the filename
            surf_w = PANEL_W - 28
            if msg.get_width() > surf_w:
                # Show just the filename truncated
                short = self._export_msg.replace('Saved: spiro_', 'Saved: ')
                msg = self.font.render(short[:48] + '…', True, (120, 220, 120))
            self.screen.blit(msg, (px + 14, sh - 60))

        # Status bar
        info = self.font.render(
            f"R={self.sl_R.value:.0f}  r={self.sl_r.value:.0f}  "
            f"d={self.sl_d.value:.0f}  ×{int(self.sl_rev.value)}", True, SUBTEXT)
        self.screen.blit(info, (px + 14, sh - 22))

        # Key hints
        hints = self.font.render("R redraw  C clear  A animate  Esc quit", True, (70, 70, 100))
        self.screen.blit(hints, (px + 14, sh - 42))

    # ── Event loop ────────────────────────────────────────────────────────────

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if   ev.key == pygame.K_ESCAPE: return False
                elif ev.key == pygame.K_r:  self._recompute()
                elif ev.key == pygame.K_c:  self._clear()
                elif ev.key == pygame.K_a:
                    self.btn_anim.active = not self.btn_anim.active
                    self.animate = self.btn_anim.active
                    if self.animate: self.anim_idx = 0; self.canvas.fill(BG_CANVAS)

            if ev.type == pygame.VIDEORESIZE:
                cw, ch = ev.w - PANEL_W, ev.h
                self.canvas = pygame.Surface((cw, ch))
                self._recompute()

            for sl in self._all_sliders:
                sl.handle_event(ev)
            for b in self._all_buttons:
                b.handle_event(ev)
        return True

    def run(self):
        running = True
        while running:
            running = self.handle_events()

            if self.animate:
                self._anim_step()

            if self._export_timer > 0:
                self._export_timer -= 1
                if self._export_timer == 0:
                    self._export_msg = ''

            # Blit canvas
            self.screen.blit(self.canvas, (0, 0))

            # Gear overlay
            if self.btn_gear.active:
                t = self.anim_t if self.animate else 0.0
                draw_gear_overlay(self.screen, self.cx, self.cy,
                                  self.sl_R.value, self.sl_r.value, self.sl_d.value,
                                  t, self.scale, self.mode)

            self._draw_panel()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == '__main__':
    SpirographApp().run()
