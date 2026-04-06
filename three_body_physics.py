"""3_body_physics.py — Physics engine for the three-body problem.

Simulation units
----------------
G = 1    (dimensionless)
Mass : 1 unit ≈ 1 M☉  (solar-mass equivalent for display purposes)
For N bodies this module is fully general.

Author: Matthew Belousoff, Claude 2026
"""

import numpy as np
import mpmath
from dataclasses import dataclass, field
from typing import List, Tuple
from scipy.integrate import solve_ivp

# ── Constants ──────────────────────────────────────────────────────────────────
G                = 1.0    # gravitational constant (simulation units)
SOFTENING        = 1e-4  # softening length; prevents 1/r² singularity at close approach
COLLISION_RADIUS = 0.08  # default merge radius (AU-equivalent); bodies closer than this merge

# ── Body ───────────────────────────────────────────────────────────────────────
@dataclass
class Body:
    mass: float           # solar-mass equivalents
    pos:  np.ndarray      # shape (3,)  — position in AU-equivalents
    vel:  np.ndarray      # shape (3,)  — velocity in AU/yr-equivalents
    name: str = ""

    def copy(self) -> "Body":
        return Body(self.mass, self.pos.copy(), self.vel.copy(), self.name)


# ── State packing/unpacking ─────────────────────────────────────────────────────
def build_state(bodies: List[Body]) -> np.ndarray:
    """Pack bodies into flat state vector [pos×N | vel×N]."""
    N = len(bodies)
    s = np.empty(6 * N)
    for i, b in enumerate(bodies):
        s[3 * i     : 3 * i + 3] = b.pos
        s[3 * N + 3 * i : 3 * N + 3 * i + 3] = b.vel
    return s


def parse_state(s: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack state vector → (pos, vel) each shape (N, 3)."""
    return s[:3 * N].reshape(N, 3), s[3 * N:].reshape(N, 3)


# ── ODE right-hand side (vectorised numpy) ─────────────────────────────────────
def nbody_rhs(t: float, s: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """N-body equations of motion.

    dr[i,j] = pos[j] - pos[i]  →  force on i from j is G*mj*dr/dist³.
    Diagonal (i==i) gives dr=0 so contributes exactly 0; no masking needed.
    """
    N = len(masses)
    pos, vel = parse_state(s, N)
    dr   = pos[None, :, :] - pos[:, None, :]          # (N, N, 3)
    dist2 = np.sum(dr ** 2, axis=-1, keepdims=True) + SOFTENING ** 2  # (N, N, 1)
    acc  = G * np.sum(masses[None, :, None] * dr / dist2 ** 1.5, axis=1)  # (N, 3)
    return np.concatenate([vel.ravel(), acc.ravel()])


# ── Force computation (for visualisation) ─────────────────────────────────────
def compute_forces(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Net gravitational force vector on each body.  NaN-aware.

    Parameters
    ----------
    positions : (N, 3)
    masses    : (N,)

    Returns
    -------
    forces : (N, 3) — NaN rows for NaN-position (absorbed) bodies.
    """
    N = len(masses)
    forces = np.full((N, 3), np.nan)
    valid = ~np.any(np.isnan(positions), axis=1)
    n_valid = valid.sum()
    if n_valid < 2:
        forces[valid] = 0.0
        return forces
    idx   = np.where(valid)[0]
    pos_v = positions[idx]
    m_v   = masses[idx]
    dr    = pos_v[None, :, :] - pos_v[:, None, :]          # (Nv, Nv, 3)
    dist2 = np.sum(dr ** 2, axis=-1, keepdims=True) + SOFTENING ** 2
    # F_i = m_i * a_i = G * m_i * Σ_j m_j * dr_ij / |dr_ij|^3
    acc   = G * np.sum(m_v[None, :, None] * dr / dist2 ** 1.5, axis=1)
    f_net = m_v[:, None] * acc
    for k, oi in enumerate(idx):
        forces[oi] = f_net[k]
    return forces


# ── Integrator ─────────────────────────────────────────────────────────────────
def integrate(
    bodies:  List[Body],
    t_end:   float,
    n_out:   int   = 4000,
    rtol:    float = 1e-9,
    atol:    float = 1e-9,
    method:  str   = "DOP853",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the N-body system.

    Returns
    -------
    ts        : (n_out,) time samples
    pos_arr   : (n_out, N, 3) positions
    vel_arr   : (n_out, N, 3) velocities
    """
    N      = len(bodies)
    masses = np.array([b.mass for b in bodies])
    s0     = build_state(bodies)
    t_eval = np.linspace(0.0, t_end, n_out)

    sol = solve_ivp(
        fun=lambda t, s: nbody_rhs(t, s, masses),
        t_span=(0.0, t_end),
        y0=s0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    ts      = sol.t
    pos_arr = sol.y[:3 * N].T.reshape(-1, N, 3)
    vel_arr = sol.y[3 * N:].T.reshape(-1, N, 3)
    return ts, pos_arr, vel_arr


# ── Collision-aware integrator ──────────────────────────────────────────────────
def integrate_with_collisions(
    bodies:           List[Body],
    t_end:            float,
    n_out:            int   = 4000,
    rtol:             float = 1e-9,
    atol:             float = 1e-9,
    method:           str   = "DOP853",
    collision_radius: float = COLLISION_RADIUS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Integrate with collision detection and inelastic merging.

    When two bodies approach within *collision_radius*, they merge:
    - combined mass = m₁ + m₂
    - new position  = centre-of-mass  (momentum conserving)
    - new velocity  = COM velocity    (momentum conserving)
    Integration continues with N-1 bodies.

    Returns
    -------
    ts      : (n_frames,)
    pos_arr : (n_frames, N_orig, 3)  — NaN where body was absorbed
    vel_arr : (n_frames, N_orig, 3)  — NaN where body was absorbed
    merges  : list of (t_merge, absorber_orig_idx, absorbed_orig_idx, new_mass)
    """
    N_orig  = len(bodies)
    active  = [b.copy() for b in bodies]   # mutable working list
    orig_ix = list(range(N_orig))           # active[i] → original body index

    seg_ts, seg_pos, seg_vel = [], [], []
    merges:  list = []
    t_cur   = 0.0

    while t_cur < t_end:
        N = len(active)
        if N <= 0:
            break

        masses = np.array([b.mass for b in active])
        s0     = build_state(active)
        frac   = min(1.0, (t_end - t_cur) / t_end) if t_end > 0 else 1.0
        n_seg  = max(50, int(n_out * frac))
        t_eval = np.linspace(t_cur, t_end, n_seg)

        # ── Build collision event functions (one per body pair) ───────────────
        pairs  = [(i, j) for i in range(N) for j in range(i + 1, N)]
        events = []
        for (ai, aj) in pairs:
            def _make_ev(ii, jj, _N=N):
                def ev(t, s):
                    pos, _ = parse_state(s, _N)
                    return np.linalg.norm(pos[ii] - pos[jj]) - collision_radius
                ev.terminal  = True
                ev.direction = -1   # only trigger while approaching
                return ev
            events.append(_make_ev(ai, aj))

        sol = solve_ivp(
            fun=lambda t, s: nbody_rhs(t, s, masses),
            t_span=(t_cur, t_end),
            y0=s0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            events=events if pairs else None,
            dense_output=False,
        )

        K       = sol.y.shape[1]
        pos_raw = sol.y[:3 * N].T.reshape(K, N, 3)
        vel_raw = sol.y[3 * N:].T.reshape(K, N, 3)

        # Pad segment to N_orig bodies (NaN where currently inactive)
        pos_s = np.full((K, N_orig, 3), np.nan)
        vel_s = np.full((K, N_orig, 3), np.nan)
        for ai, oi in enumerate(orig_ix):
            pos_s[:, oi, :] = pos_raw[:, ai, :]
            vel_s[:, oi, :] = vel_raw[:, ai, :]

        seg_ts.append(sol.t)
        seg_pos.append(pos_s)
        seg_vel.append(vel_s)

        if sol.status != 1:        # no event — completed normally
            break

        # ── Identify the colliding pair ───────────────────────────────────────
        hit_ev = None
        for ei, (te, (ai, aj)) in enumerate(zip(sol.t_events, pairs)):
            if len(te) > 0:
                hit_ev = (ai, aj, float(te[0]))
                break
        if hit_ev is None:
            break

        ai, aj, t_coll = hit_ev
        t_cur           = t_coll

        # Last known positions/velocities for the active pair
        mi, mj   = active[ai].mass, active[aj].mass
        m_new    = mi + mj
        pi, pj   = pos_raw[-1, ai], pos_raw[-1, aj]
        vi, vj   = vel_raw[-1, ai], vel_raw[-1, aj]
        p_new    = (mi * pi + mj * pj) / m_new
        v_new    = (mi * vi + mj * vj) / m_new

        # Higher-mass body absorbs; tie → ai absorbs aj
        if mi >= mj:
            absorber_ai, absorbed_ai = ai, aj
        else:
            absorber_ai, absorbed_ai = aj, ai
        absorber_oi = orig_ix[absorber_ai]
        absorbed_oi = orig_ix[absorbed_ai]

        merged = Body(m_new, p_new, v_new,
                      f"{active[absorber_ai].name}+{active[absorbed_ai].name}")
        merges.append((t_coll, absorber_oi, absorbed_oi, m_new))

        # Rebuild active list (drop the absorbed body)
        new_active, new_orig = [], []
        for idx, (b, oi) in enumerate(zip(active, orig_ix)):
            if idx == absorber_ai:
                new_active.append(merged)
                new_orig.append(absorber_oi)
            elif idx == absorbed_ai:
                pass           # absorbed
            else:
                new_active.append(b)
                new_orig.append(oi)
        active  = new_active
        orig_ix = new_orig

    # ── Assemble full trajectory ───────────────────────────────────────────────
    if not seg_ts:
        # Nothing was integrated — return static initial frame
        pv = np.stack([np.array([b.pos for b in bodies])] * 2).reshape(2, N_orig, 3)
        vv = np.stack([np.array([b.vel for b in bodies])] * 2).reshape(2, N_orig, 3)
        return np.array([0.0, t_end]), pv, vv, merges

    ts_out  = np.concatenate(seg_ts)
    pos_out = np.concatenate(seg_pos, axis=0)
    vel_out = np.concatenate(seg_vel, axis=0)
    return ts_out, pos_out, vel_out, merges


# ── Visual helpers ──────────────────────────────────────────────────────────────
def body_color(mass: float) -> tuple:
    """Return (R, G, B) in [0,1]³ based on mass (solar-mass equivalents).

    < 1 M☉  → gas-giant / sub-stellar palette   (orange → brown-dwarf red)
    ≥ 1 M☉  → main-sequence HR colours          (G yellow → O blue)
    """
    m = max(float(mass), 1e-9)
    if m < 1.0:
        # log-scale: 1e-4 → orange-tan Jupiter, 0.08 → red-M-dwarf
        frac = np.clip(np.log10(m / 1e-4) / np.log10(1.0 / 1e-4), 0.0, 1.0)
        r = float(np.interp(frac, [0, 0.15, 0.55, 1.0], [0.82, 0.60, 0.88, 0.95]))
        g = float(np.interp(frac, [0, 0.15, 0.55, 1.0], [0.52, 0.28, 0.22, 0.25]))
        b = float(np.interp(frac, [0, 0.15, 0.55, 1.0], [0.22, 0.12, 0.10, 0.10]))
    else:
        # log10(m): 0 at 1 M☉, ~1 at 10 M☉, ~2 at 100 M☉
        log_m = np.log10(m)
        frac  = np.clip(log_m / 2.0, 0.0, 1.0)
        r = float(np.interp(frac, [0, 0.22, 0.55, 0.80, 1.0], [1.00, 1.00, 0.82, 0.50, 0.28]))
        g = float(np.interp(frac, [0, 0.22, 0.55, 0.80, 1.0], [0.92, 0.88, 0.90, 0.68, 0.48]))
        b = float(np.interp(frac, [0, 0.22, 0.55, 0.80, 1.0], [0.28, 0.52, 1.00, 1.00, 1.00]))
    return (r, g, b)


def body_size(mass: float) -> float:
    """Marker area (points²) ∝ mass^(1/3)  clamped to [60, 2500]."""
    return float(np.clip(200.0 * max(float(mass), 1e-9) ** (1.0 / 3.0), 60, 2500))


def body_label(mass: float) -> str:
    m = float(mass)
    if m >= 1.0:
        lm = np.log10(m)
        if lm < 0.12:  return "G-type ☀"
        if lm < 0.35:  return "F-type ★"
        if lm < 0.65:  return "A-type ★"
        if lm < 1.05:  return "B-type ★"
        return "O-type ★"
    if m < 0.002:  return "Gas Giant"
    if m < 0.08:   return "Brown Dwarf"
    if m < 0.35:   return "M-dwarf"
    return "K/M-dwarf"


# ── Preset helpers ──────────────────────────────────────────────────────────────
def _body(m, pos, vel, name="") -> Body:
    return Body(mass=float(m),
                pos=np.array(pos, dtype=float),
                vel=np.array(vel, dtype=float),
                name=name)


def _equilateral_lagrange(m1: float, m2: float, m3: float, sep: float = 2.0) -> List[Body]:
    """Three masses at vertices of equilateral triangle with circular-orbit velocities."""
    masses = np.array([m1, m2, m3])
    M      = masses.sum()
    a      = sep   # triangle side length
    # Vertices (COM already at origin for equal masses; correct COM for unequal)
    verts = np.array([
        [ a / np.sqrt(3),            0.0,  0.0],
        [-a / (2 * np.sqrt(3)),  a / 2.0,  0.0],
        [-a / (2 * np.sqrt(3)), -a / 2.0,  0.0],
    ])
    com = np.sum(masses[:, None] * verts, axis=0) / M
    verts -= com
    # Angular velocity from Newton's law for equilateral triangle (any mass ratio):
    # ω² ≈ G*M/a³  (exact for equal masses; good approx for moderate imbalance)
    omega = np.sqrt(G * M / a ** 3)
    vels  = np.zeros_like(verts)
    for i in range(3):
        r   = verts[i, :2]
        r_n = np.linalg.norm(r)
        if r_n > 1e-12:
            tangent       = np.array([-r[1], r[0]]) / r_n
            vels[i, :2]  = omega * r_n * tangent
    return [Body(masses[i], verts[i], vels[i], f"Body {i+1}") for i in range(3)]


# High-precision Figure-8 IC (Chenciner & Montgomery 2000) via mpmath
mpmath.mp.dps = 50
_F8_POS = [
    np.array([float(mpmath.mpf("-0.97000436")),  float(mpmath.mpf("0.24308753")), 0.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([float(mpmath.mpf( "0.97000436")), float(mpmath.mpf("-0.24308753")), 0.0]),
]
_F8_VEL_HALF = np.array([float(mpmath.mpf("0.93240737")) / 2,
                          float(mpmath.mpf("0.86473146")) / 2, 0.0])
_F8_VEL = [
    _F8_VEL_HALF.copy(),
    np.array([-float(mpmath.mpf("0.93240737")), -float(mpmath.mpf("0.86473146")), 0.0]),
    _F8_VEL_HALF.copy(),
]


def _figure8_bodies(mass: float = 1.0, scale: float = 1.5) -> List[Body]:
    """Figure-8 orbit bodies.  scale > 1 enlarges the orbit."""
    bodies = []
    for i in range(3):
        pos = _F8_POS[i] * scale
        # Velocity scales as 1/√scale to keep Kepler relation for equal-mass case
        vel = _F8_VEL[i] / np.sqrt(scale)
        bodies.append(Body(mass, pos, vel, f"Body {i+1}"))
    return bodies


# ── Presets dictionary ──────────────────────────────────────────────────────────
PRESETS: dict = {
    "Figure-8\n(Stable)": {
        "bodies":   lambda: _figure8_bodies(mass=1.0, scale=1.5),
        "t_end":    20.0,
        "method":   "DOP853",
        "rtol":     1e-11,
        "category": "stable",
        "info":     "Choreographic orbit — all three bodies chase each other along a\nfigure-8 path. Exactly periodic for equal masses.",
    },
    "Lagrange\nEquilateral": {
        "bodies":   lambda: _equilateral_lagrange(1.0, 1.0, 1.0, sep=2.0),
        "t_end":    15.0,
        "method":   "DOP853",
        "rtol":     1e-11,
        "category": "stable",
        "info":     "Three equal masses at the vertices of a rotating equilateral\ntriangle. Lagrange's exact solution.",
    },
    "Hierarchical\nTriple": {
        "bodies":   lambda: [
            _body(2.0, [ 0.20, 0.10,  0.05], [ 0.00, -0.25,  0.05], "Primary"),
            _body(0.8, [-1.20, 0.00,  0.00], [ 0.00, -1.20,  0.00], "Secondary"),
            _body(1.0, [ 1.10, 0.10,  0.00], [ 0.00,  1.40,  0.00], "Companion"),
        ],
        "t_end":    40.0,
        "method":   "DOP853",
        "rtol":     1e-9,
        "category": "semi-stable",
        "info":     "Close binary pair with a wider companion. Hierarchical triples\nare common in real stellar systems.",
    },
    "Sun-Jupiter\n-Saturn": {
        "bodies":   lambda: [
            _body(1.000, [ 0.00,  0.00, 0.00], [ 0.00,  0.00, 0.00], "Star"),
            _body(0.001, [-5.20,  0.00, 0.00], [ 0.00, -0.43, 0.00], "Giant 1"),
            _body(0.0003,[ 0.00,  9.55, 0.00], [ 0.29,  0.00, 0.00], "Giant 2"),
        ],
        "t_end":    200.0,
        "method":   "RK45",
        "rtol":     1e-8,
        "category": "semi-stable",
        "info":     "Realistic star + two gas giants at Jupiter/Saturn separations.\nSemi-stable over the shown epoch.",
    },
    "Pythagorean\n(Chaotic)": {
        "bodies":   lambda: [
            _body(3.0, [-1.0,  3.0, 0.0], [0.0, 0.0, 0.0], "Heavy   3"),
            _body(4.0, [ 1.0, -1.0, 0.0], [0.0, 0.0, 0.0], "Heavier 4"),
            _body(5.0, [-1.0, -1.0, 0.0], [0.0, 0.0, 0.0], "Heaviest5"),
        ],
        "t_end":    100.0,
        "method":   "DOP853",
        "rtol":     1e-10,
        "category": "chaotic",
        "info":     "Pythagorean 3-body: masses 3,4,5 at rest in right-triangle\nformation. Classic chaotic scattering ending in ejection.",
    },
    "Figure-8\nPerturbed": {
        "bodies":   lambda: [
            b if i != 0
            else Body(b.mass + 0.06,
                      b.pos + np.array([0.03, 0.0, 0.02]),
                      b.vel,
                      b.name)
            for i, b in enumerate(_figure8_bodies(mass=1.0, scale=1.5))
        ],
        "t_end":    45.0,
        "method":   "DOP853",
        "rtol":     1e-10,
        "category": "semi-stable",
        "info":     "Figure-8 with a small mass+position perturbation on body 1.\nStable initially, then diverges chaotically.",
    },
    "Random\nChaotic": {
        "bodies":   lambda: [
            _body(1.2, [ 1.6,  0.1,  0.4], [-0.10,  0.55,  0.12], "A"),
            _body(0.9, [-0.6,  1.6, -0.3], [ 0.42, -0.20,  0.30], "B"),
            _body(1.5, [-1.0, -1.5,  0.1], [-0.32, -0.35, -0.22], "C"),
        ],
        "t_end":    60.0,
        "method":   "DOP853",
        "rtol":     1e-9,
        "category": "chaotic",
        "info":     "Unequal masses with arbitrary initial positions and velocities.\nRapid onset of orbital chaos.",
    },
}


# ── Module self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("3_body_physics.py — self-test")
    for name, cfg in PRESETS.items():
        bodies = cfg["bodies"]()
        masses = [b.mass for b in bodies]
        print(f"  {name.replace(chr(10),' '):<30s}  masses={[f'{m:.4g}' for m in masses]}"
              f"  cat={cfg['category']}")
    # Quick integration test on Figure-8 (scale=1 → exact period 6.3259)
    print("\nIntegrating Figure-8 at scale=1, 1 period (DOP853)…", end=" ", flush=True)
    bodies_f8 = _figure8_bodies(mass=1.0, scale=1.0)
    ts, pos, vel = integrate(bodies_f8, t_end=6.3259, n_out=1000,
                              rtol=1e-12, atol=1e-12, method="DOP853")
    drift = np.linalg.norm(pos[-1] - pos[0])
    ok = "OK" if drift < 1e-3 else "WARN"
    print(f"done.  |r(T)-r(0)| = {drift:.2e}  [{ok}]")
