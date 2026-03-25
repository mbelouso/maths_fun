#!/usr/bin/env python3
"""
Prime Spiral Gallery — integers 1 to 100
=========================================
Static three-panel figure showing Ulam, Sacks, and Vogel spiral
arrangements of integers 1–100, with every number labelled.

  Gold   = prime
  White  = composite  (dim)

Usage
-----
    python prime_gallery_100.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Palette ─────────────────────────────────────────────────────────────────────

N       = 100
BG      = "#0d0d1a"
PRIME_C = "#FFD700"    # gold
COMP_C  = "#ccccdd"    # pale white-blue

# ── Sieve ────────────────────────────────────────────────────────────────────────

def sieve(limit):
    ip = np.ones(limit + 1, dtype=bool)
    ip[:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if ip[i]:
            ip[i * i :: i] = False
    return ip

ip = sieve(N)

# ── Spiral generators ────────────────────────────────────────────────────────────

def ulam_grid(side):
    """Return (side×side) array of ints arranged in the Ulam spiral (side→odd)."""
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
            if 0 <= x < n and 0 <= y < n: grid[y, x] = num
            num += 1
        for _ in range(step):          # up
            if num > n * n: break
            y -= 1
            if 0 <= x < n and 0 <= y < n: grid[y, x] = num
            num += 1
        step += 1
        for _ in range(step):          # left
            if num > n * n: break
            x -= 1
            if 0 <= x < n and 0 <= y < n: grid[y, x] = num
            num += 1
        for _ in range(step):          # down
            if num > n * n: break
            y += 1
            if 0 <= x < n and 0 <= y < n: grid[y, x] = num
            num += 1
        step += 1
    return grid


def sacks_coords(n):
    """Sacks spiral: k at (√k·cos(2π√k), √k·sin(2π√k))."""
    k = np.arange(1, n + 1, dtype=float)
    r = np.sqrt(k)
    theta = 2.0 * np.pi * r
    return r * np.cos(theta), r * np.sin(theta)


def vogel_coords(n):
    """Vogel sunflower spiral using the golden angle."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    k = np.arange(1, n + 1, dtype=float)
    r = np.sqrt(k)
    theta = k * golden_angle
    return r * np.cos(theta), r * np.sin(theta)


# ── Figure ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(19, 7), facecolor=BG)
plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.06)
fig.suptitle(
    "Integers 1–100 in three spiral arrangements    ·    gold = prime    ·    white = composite",
    color=PRIME_C, fontsize=12, y=0.95,
)

for ax in axes:
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# ── Panel 1: Ulam ────────────────────────────────────────────────────────────────

ax = axes[0]
ax.set_title("Ulam Spiral", color=PRIME_C, fontsize=12, pad=8)

grid = ulam_grid(11)   # 11×11 = 121 cells; only label 1…100
n = grid.shape[0]

# Cell background image: gold-tinted for primes, plain dark for composites/empty
img = np.zeros((n, n, 4), dtype=float)
for r in range(n):
    for c in range(n):
        val = int(grid[r, c])
        if val == 0 or val > N:
            img[r, c] = mcolors.to_rgba("#08080f")
        elif ip[val]:
            img[r, c] = (0.13, 0.10, 0.00, 1.0)   # dark gold tint
        else:
            img[r, c] = mcolors.to_rgba("#0d0d22")

ax.imshow(img, interpolation="nearest", origin="upper",
          extent=(-0.5, n - 0.5, n - 0.5, -0.5))

# Subtle grid lines between cells
for i in range(n + 1):
    ax.axhline(i - 0.5, color="#16162a", lw=0.6, zorder=2)
    ax.axvline(i - 0.5, color="#16162a", lw=0.6, zorder=2)

# Connecting line 1 → 2 → … → 100 (traces the rectangular spiral path)
pos = {}
for r in range(n):
    for c in range(n):
        val = int(grid[r, c])
        if 1 <= val <= N:
            pos[val] = (c, r)
line_x = [pos[k][0] for k in range(1, N + 1)]
line_y = [pos[k][1] for k in range(1, N + 1)]
ax.plot(line_x, line_y, color="#4488bb", lw=0.9, alpha=0.55, zorder=2)

# Number labels
for r in range(n):
    for c in range(n):
        val = int(grid[r, c])
        if val == 0 or val > N:
            continue
        is_p = bool(ip[val])
        ax.text(
            c, r, str(val),
            ha="center", va="center",
            color=PRIME_C if is_p else COMP_C,
            alpha=1.0 if is_p else 0.50,
            fontsize=8,
            fontfamily="monospace",
            fontweight="bold" if is_p else "normal",
            zorder=3,
        )

ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(n - 0.5, -0.5)    # invert y: row 0 at top


# ── Panels 2 & 3: Sacks and Vogel ───────────────────────────────────────────────

def draw_scatter_panel(ax, title, xs, ys):
    ax.set_title(title, color=PRIME_C, fontsize=12, pad=8)

    prime_mask = ip[1:]            # boolean array, index 0 → num 1

    # Connecting line 1 → 2 → … → 100
    ax.plot(xs, ys, color="#4488bb", lw=0.9, alpha=0.55, zorder=1)

    # Composite dots (small, dim)
    ax.scatter(xs[~prime_mask], ys[~prime_mask],
               s=18, color="#252540", linewidths=0, zorder=2)

    # Prime dots (larger, gold)
    ax.scatter(xs[prime_mask], ys[prime_mask],
               s=55, color=PRIME_C, linewidths=0, zorder=3)

    # Labels — draw composites first so primes render on top
    for is_p in [False, True]:
        for num in range(1, N + 1):
            if bool(ip[num]) != is_p:
                continue
            x, y = xs[num - 1], ys[num - 1]
            ax.text(
                x, y, str(num),
                ha="center", va="center",
                color=PRIME_C if is_p else COMP_C,
                alpha=1.0 if is_p else 0.50,
                fontsize=6.5,
                fontfamily="monospace",
                fontweight="bold" if is_p else "normal",
                zorder=5 if is_p else 4,
            )


draw_scatter_panel(axes[1], "Sacks Spiral",          *sacks_coords(N))
draw_scatter_panel(axes[2], "Vogel (Sunflower) Spiral", *vogel_coords(N))


# ── Save & show ──────────────────────────────────────────────────────────────────

out = "prime_spirals_1_100.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved {out}")
plt.show()
