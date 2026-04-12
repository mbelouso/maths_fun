#!/usr/bin/env python3
"""
Double Pendulum Chaos Map — CUDA Multi-GPU Edition
===================================================
Interactive PyQt5 application that visualises the stability landscape of a
double pendulum by sweeping initial angles (θ1, θ2) across a 2D grid.

Each cell represents a pendulum released from rest at those angles.  The
colour encodes a chaos metric (flip count or peak angular velocity).

This version uses Numba CUDA to run one GPU thread per pendulum, with
support for splitting the grid across multiple GPUs.  A CPU (NumPy)
fallback is included for machines without CUDA.

Usage
-----
    conda run -n maths_fun python3 pendulum_chaos_map_cuda.py

Author: Matthew Belousoff, Claude 2026
"""

import sys
import os
import time
import threading
import datetime
import pickle
import math
import multiprocessing

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from collections import deque
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QDoubleSpinBox, QSpinBox, QGroupBox,
    QPushButton, QComboBox, QCheckBox,
    QScrollArea, QSizePolicy, QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject


# ── CUDA availability probe ────────────────────────────────────────────────
CUDA_AVAILABLE = False
_CUDA_GPUS = []      # list of (id, name) tuples
_CUDA_ERR  = ""

try:
    from numba import cuda
    if cuda.is_available():
        CUDA_AVAILABLE = True
        for i in range(len(cuda.gpus)):
            try:
                dev = cuda.gpus[i]
                _CUDA_GPUS.append((i, dev.name.decode()
                                   if isinstance(dev.name, bytes)
                                   else str(dev.name)))
            except Exception:
                _CUDA_GPUS.append((i, f"GPU {i}"))
    else:
        _CUDA_ERR = "Numba installed but no CUDA-capable GPU detected"
except ImportError:
    _CUDA_ERR = "Numba not installed (pip install numba)"
except Exception as e:
    _CUDA_ERR = f"CUDA init error: {e}"


# ── Palette ──────────────────────────────────────────────────────────────────
DARK_BG = "#07070f"
CTRL_BG = "#12122a"
CTRL_FG = "#ccccee"
ACC_C   = "#7788ff"
GOLD    = "#FFD700"

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
QDoubleSpinBox::down-button,
QSpinBox::up-button,
QSpinBox::down-button         { width:14px; }
QPushButton                   { background:#1a1a3e; color:#ccccee;
                                border:1px solid #445; padding:5px 8px;
                                border-radius:3px; }
QPushButton:hover             { background:#2a2a5e; }
QPushButton:pressed           { background:#0a0a2e; }
QCheckBox                     { color:#ccccee; font-size:9pt; }
QProgressBar                  { background:#1a1a2e; border:1px solid #334;
                                border-radius:3px; text-align:center;
                                color:#ccccee; font-size:8pt; }
QProgressBar::chunk           { background:#7788ff; border-radius:2px; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA PHYSICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

if CUDA_AVAILABLE:

    @cuda.jit(device=True)
    def _derivs_device(th1, w1, th2, w2, L, M, M2, m2, g, inv_L):
        """Lagrangian equations of motion for a single double pendulum.

        Returns (d_th1, d_w1, d_th2, d_w2).
        """
        delta = th1 - th2
        sin_d = math.sin(delta)
        cos_d = math.cos(delta)
        inv_D = 1.0 / (M2 - m2 * math.cos(2.0 * delta))

        d_th1 = w1
        d_w1  = (-g * M2 * math.sin(th1)
                 - m2 * g * math.sin(th1 - 2.0 * th2)
                 - 2.0 * sin_d * m2 * (w2 * w2 * L + w1 * w1 * L * cos_d)
                ) * (inv_L * inv_D)
        d_th2 = w2
        d_w2  = (2.0 * sin_d * (w1 * w1 * L * M
                                  + g * M * math.cos(th1)
                                  + w2 * w2 * L * m2 * cos_d)
                ) * (inv_L * inv_D)
        return d_th1, d_w1, d_th2, d_w2


    @cuda.jit
    def _chaos_kernel(th1_vals, th2_vals, n1, n2,
                      L, m1, m2, g, dt, n_steps,
                      flips_th1_out, flips_th2_out, max_w2_out):
        """One thread per pendulum.  Full RK4 integration + chaos metrics."""
        idx = cuda.grid(1)
        if idx >= n1 * n2:
            return

        # Map linear index to grid position
        i2 = idx // n1    # row (th2 index)
        i1 = idx % n1     # col (th1 index)

        # Initialise state: angles from grid, velocities = 0
        th1 = th1_vals[i1]
        w1  = 0.0
        th2 = th2_vals[i2]
        w2  = 0.0

        # Pre-compute physics constants
        M     = m1 + m2
        M2    = 2.0 * m1 + m2
        inv_L = 1.0 / L
        dt2   = 0.5 * dt
        dt6   = dt / 6.0

        # Flip tracking: wrap initial angles to [-pi, pi]
        PI     = 3.141592653589793
        TWO_PI = 6.283185307179586
        prev_a1 = ((th1 + PI) % TWO_PI) - PI
        prev_a2 = ((th2 + PI) % TWO_PI) - PI
        f_th1 = 0
        f_th2 = 0
        mw2   = 0.0

        # ── Main RK4 loop ──────────────────────────────────────────
        for _ in range(n_steps):
            # k1
            k1_0, k1_1, k1_2, k1_3 = _derivs_device(
                th1, w1, th2, w2, L, M, M2, m2, g, inv_L)
            # k2
            k2_0, k2_1, k2_2, k2_3 = _derivs_device(
                th1 + dt2 * k1_0, w1 + dt2 * k1_1,
                th2 + dt2 * k1_2, w2 + dt2 * k1_3,
                L, M, M2, m2, g, inv_L)
            # k3
            k3_0, k3_1, k3_2, k3_3 = _derivs_device(
                th1 + dt2 * k2_0, w1 + dt2 * k2_1,
                th2 + dt2 * k2_2, w2 + dt2 * k2_3,
                L, M, M2, m2, g, inv_L)
            # k4
            k4_0, k4_1, k4_2, k4_3 = _derivs_device(
                th1 + dt * k3_0, w1 + dt * k3_1,
                th2 + dt * k3_2, w2 + dt * k3_3,
                L, M, M2, m2, g, inv_L)

            # RK4 update
            th1 += dt6 * (k1_0 + 2.0 * k2_0 + 2.0 * k3_0 + k4_0)
            w1  += dt6 * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
            th2 += dt6 * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)
            w2  += dt6 * (k1_3 + 2.0 * k2_3 + 2.0 * k3_3 + k4_3)

            # Flip detection: wrapped angle jump > pi means ±pi crossing
            cur_a1 = ((th1 + PI) % TWO_PI) - PI
            cur_a2 = ((th2 + PI) % TWO_PI) - PI
            d1 = cur_a1 - prev_a1
            d2 = cur_a2 - prev_a2
            if d1 > PI or d1 < -PI:
                f_th1 += 1
            if d2 > PI or d2 < -PI:
                f_th2 += 1
            prev_a1 = cur_a1
            prev_a2 = cur_a2

            # Peak angular velocity of outer bob
            abs_w2 = w2 if w2 >= 0.0 else -w2
            if abs_w2 > mw2:
                mw2 = abs_w2

        # Write outputs
        flips_th1_out[idx] = f_th1
        flips_th2_out[idx] = f_th2
        max_w2_out[idx]    = mw2


    def _gpu_worker(gpu_id, th1_vals, th2_slice, n1, n2_local,
                    L, m1, m2, g, dt, n_steps, use_fp32):
        """Run the chaos kernel on a single GPU for a contiguous block of rows.

        Called from a ThreadPoolExecutor thread.
        Returns (flips_th1, flips_th2, max_w2) as host arrays.
        """
        cuda.select_device(gpu_id)

        fp = np.float32 if use_fp32 else np.float64
        N_local = n1 * n2_local

        # Transfer input arrays to device
        d_th1 = cuda.to_device(th1_vals.astype(fp))
        d_th2 = cuda.to_device(th2_slice.astype(fp))

        # Allocate output arrays on device
        d_flips_th1 = cuda.device_array(N_local, dtype=np.int32)
        d_flips_th2 = cuda.device_array(N_local, dtype=np.int32)
        d_max_w2    = cuda.device_array(N_local, dtype=fp)

        # Launch kernel
        threads_per_block = 256
        blocks = (N_local + threads_per_block - 1) // threads_per_block

        _chaos_kernel[blocks, threads_per_block](
            d_th1, d_th2, n1, n2_local,
            fp(L), fp(m1), fp(m2), fp(g), fp(dt), n_steps,
            d_flips_th1, d_flips_th2, d_max_w2)

        cuda.synchronize()

        return (d_flips_th1.copy_to_host(),
                d_flips_th2.copy_to_host(),
                d_max_w2.copy_to_host())


    def compute_chaos_grid_cuda(th1_range, th2_range, n1, n2,
                                L=1.0, m1=1.0, m2=1.0, g=9.81,
                                t_end=20.0, dt=0.002,
                                gpu_ids=None, use_fp32=False,
                                progress_cb=None):
        """Compute chaos metrics on GPU(s).

        Parameters
        ----------
        gpu_ids : list[int] or None
            GPU device IDs to use. None = all available.
        use_fp32 : bool
            float32 for ~32x speedup on consumer GPUs.

        Returns dict with keys: "flips_2", "flips_both", "max_w2".
        """
        if gpu_ids is None:
            gpu_ids = [gid for gid, _ in _CUDA_GPUS]
        n_gpus = len(gpu_ids)

        th1_vals = np.linspace(th1_range[0], th1_range[1], n1)
        th2_vals = np.linspace(th2_range[0], th2_range[1], n2)
        n_steps  = int(t_end / dt)

        # Split rows across GPUs
        row_splits = np.array_split(np.arange(n2), n_gpus)

        if progress_cb:
            progress_cb(0)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        N = n1 * n2
        flips_th1_all = np.zeros(N, dtype=np.int32)
        flips_th2_all = np.zeros(N, dtype=np.int32)
        max_w2_all    = np.zeros(N, dtype=np.float64)

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = {}
            for gpu_id, rows in zip(gpu_ids, row_splits):
                if len(rows) == 0:
                    continue
                r0 = int(rows[0])
                r1 = int(rows[-1]) + 1
                th2_slice = th2_vals[r0:r1].copy()
                n2_local = r1 - r0
                fut = pool.submit(
                    _gpu_worker, gpu_id, th1_vals, th2_slice,
                    n1, n2_local, L, m1, m2, g, dt, n_steps, use_fp32)
                futures[fut] = (r0, r1)

            done_count = 0
            for fut in as_completed(futures):
                r0, r1 = futures[fut]
                f1, f2, mw = fut.result()
                idx0 = r0 * n1
                idx1 = r1 * n1
                flips_th1_all[idx0:idx1] = f1
                flips_th2_all[idx0:idx1] = f2
                max_w2_all[idx0:idx1]    = mw.astype(np.float64)
                done_count += 1
                if progress_cb:
                    progress_cb(int(100 * done_count / len(futures)))

        return {
            "flips_2":    flips_th2_all.reshape(n2, n1),
            "flips_both": (flips_th1_all + flips_th2_all).reshape(n2, n1),
            "max_w2":     max_w2_all.reshape(n2, n1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CPU FALLBACK ENGINE  (vectorised NumPy, from pendulum_chaos_map.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_derivs(L, m1, m2, g):
    """Return a closure that computes derivatives for fixed physics params."""
    M  = m1 + m2
    M2 = 2*m1 + m2
    inv_L = 1.0 / L

    def derivs(s, out):
        """Write derivatives of s into pre-allocated out.  Both are (N, 4)."""
        th1 = s[:, 0];  w1 = s[:, 1]
        th2 = s[:, 2];  w2 = s[:, 3]
        delta = th1 - th2
        sin_d = np.sin(delta)
        cos_d = np.cos(delta)
        inv_D = 1.0 / (M2 - m2*np.cos(2*delta))

        out[:, 0] = w1
        out[:, 1] = (-g*M2*np.sin(th1)
                     - m2*g*np.sin(th1 - 2*th2)
                     - 2*sin_d*m2*(w2*w2*L + w1*w1*L*cos_d)
                    ) * (inv_L * inv_D)
        out[:, 2] = w2
        out[:, 3] = (2*sin_d*(w1*w1*L*M
                               + g*M*np.cos(th1)
                               + w2*w2*L*m2*cos_d)
                    ) * (inv_L * inv_D)
    return derivs


def _compute_chunk(args):
    """Integrate a chunk of pendulums.  Top-level for pickle/multiprocessing."""
    states, L, m1, m2, g, t_end, dt, chunk_id = args
    N = states.shape[0]
    if N == 0:
        return (chunk_id,
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float64))

    flips_th2 = np.zeros(N, dtype=np.int32)
    flips_th1 = np.zeros(N, dtype=np.int32)
    max_w2    = np.zeros(N, dtype=np.float64)

    prev_w1 = ((states[:, 0] + np.pi) % (2*np.pi)) - np.pi
    prev_w2 = ((states[:, 2] + np.pi) % (2*np.pi)) - np.pi

    k1  = np.empty_like(states)
    k2  = np.empty_like(states)
    k3  = np.empty_like(states)
    k4  = np.empty_like(states)
    tmp = np.empty_like(states)

    derivs = _make_derivs(L, m1, m2, g)
    dt2 = 0.5 * dt
    dt6 = dt / 6.0

    n_steps = int(t_end / dt)
    for step in range(n_steps):
        derivs(states, k1)
        np.add(states, dt2 * k1, out=tmp);  derivs(tmp, k2)
        np.add(states, dt2 * k2, out=tmp);  derivs(tmp, k3)
        np.add(states, dt  * k3, out=tmp);  derivs(tmp, k4)
        states += dt6 * (k1 + 2*k2 + 2*k3 + k4)

        cur_w1 = ((states[:, 0] + np.pi) % (2*np.pi)) - np.pi
        cur_w2 = ((states[:, 2] + np.pi) % (2*np.pi)) - np.pi
        flips_th1 += (np.abs(cur_w1 - prev_w1) > np.pi).astype(np.int32)
        flips_th2 += (np.abs(cur_w2 - prev_w2) > np.pi).astype(np.int32)
        prev_w1 = cur_w1
        prev_w2 = cur_w2

        np.maximum(max_w2, np.abs(states[:, 3]), out=max_w2)

    return (chunk_id, flips_th2, flips_th1, max_w2)


def compute_chaos_grid(th1_range, th2_range, n1, n2,
                       L=1.0, m1=1.0, m2=1.0, g=9.81,
                       t_end=20.0, dt=0.002, n_workers=1,
                       progress_cb=None):
    """Compute chaos metrics for an n2×n1 grid of double pendulums (CPU)."""
    th1_vals = np.linspace(th1_range[0], th1_range[1], n1)
    th2_vals = np.linspace(th2_range[0], th2_range[1], n2)
    TH1, TH2 = np.meshgrid(th1_vals, th2_vals)

    N = n1 * n2
    states_all = np.zeros((N, 4), dtype=np.float64)
    states_all[:, 0] = TH1.ravel()
    states_all[:, 2] = TH2.ravel()

    n_workers = max(1, min(n_workers, N))

    # ── Single-process fast path ────────────────────────────────────────
    if n_workers <= 1:
        flips_th2 = np.zeros(N, dtype=np.int32)
        flips_th1 = np.zeros(N, dtype=np.int32)
        max_w2    = np.zeros(N, dtype=np.float64)

        prev_w1 = ((states_all[:, 0] + np.pi) % (2*np.pi)) - np.pi
        prev_w2 = ((states_all[:, 2] + np.pi) % (2*np.pi)) - np.pi

        k1  = np.empty_like(states_all)
        k2  = np.empty_like(states_all)
        k3  = np.empty_like(states_all)
        k4  = np.empty_like(states_all)
        tmp = np.empty_like(states_all)
        derivs = _make_derivs(L, m1, m2, g)
        dt2 = 0.5 * dt; dt6 = dt / 6.0
        n_steps = int(t_end / dt)
        report_every = max(1, n_steps // 100)
        for step in range(n_steps):
            derivs(states_all, k1)
            np.add(states_all, dt2 * k1, out=tmp);  derivs(tmp, k2)
            np.add(states_all, dt2 * k2, out=tmp);  derivs(tmp, k3)
            np.add(states_all, dt  * k3, out=tmp);  derivs(tmp, k4)
            states_all += dt6 * (k1 + 2*k2 + 2*k3 + k4)

            cur_w1 = ((states_all[:, 0] + np.pi) % (2*np.pi)) - np.pi
            cur_w2 = ((states_all[:, 2] + np.pi) % (2*np.pi)) - np.pi
            flips_th1 += (np.abs(cur_w1 - prev_w1) > np.pi).astype(np.int32)
            flips_th2 += (np.abs(cur_w2 - prev_w2) > np.pi).astype(np.int32)
            prev_w1 = cur_w1
            prev_w2 = cur_w2

            np.maximum(max_w2, np.abs(states_all[:, 3]), out=max_w2)

            if progress_cb and step % report_every == 0:
                progress_cb(int(100 * step / n_steps))
        if progress_cb:
            progress_cb(100)
        return {
            "flips_2":    flips_th2.reshape(n2, n1),
            "flips_both": (flips_th1 + flips_th2).reshape(n2, n1),
            "max_w2":     max_w2.reshape(n2, n1),
        }

    # ── Multi-process path ──────────────────────────────────────────────
    row_splits = np.array_split(np.arange(n2), n_workers)
    chunks = []
    for cid, rows in enumerate(row_splits):
        if len(rows) == 0:
            continue
        r0, r1 = rows[0], rows[-1] + 1
        idx0 = r0 * n1
        idx1 = r1 * n1
        chunk_states = states_all[idx0:idx1].copy()
        chunks.append((chunk_states, L, m1, m2, g, t_end, dt, cid))

    if progress_cb:
        progress_cb(0)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    flips_th2_all = np.zeros(N, dtype=np.int32)
    flips_th1_all = np.zeros(N, dtype=np.int32)
    max_w2_all    = np.zeros(N, dtype=np.float64)
    done_count = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_compute_chunk, c): c[7] for c in chunks}
        for fut in as_completed(futures):
            cid = futures[fut]
            _, chunk_f2, chunk_f1, chunk_mw2 = fut.result()
            rows = row_splits[cid]
            r0 = rows[0] * n1
            r1 = (rows[-1] + 1) * n1
            flips_th2_all[r0:r1] = chunk_f2
            flips_th1_all[r0:r1] = chunk_f1
            max_w2_all[r0:r1]    = chunk_mw2
            done_count += 1
            if progress_cb:
                progress_cb(int(100 * done_count / len(chunks)))

    return {
        "flips_2":    flips_th2_all.reshape(n2, n1),
        "flips_both": (flips_th1_all + flips_th2_all).reshape(n2, n1),
        "max_w2":     max_w2_all.reshape(n2, n1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REUSABLE WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class _ParamSlider(QWidget):
    """Label + QSlider(0-1000) + QDoubleSpinBox, bidirectionally synced."""

    valueChanged = pyqtSignal(float)

    def __init__(self, label, vmin, vmax, vinit, decimals=2, suffix="",
                 parent=None):
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

    def value(self):
        return self._sb.value()

    def set_value(self, v, silent=True):
        v = float(np.clip(v, self._vmin, self._vmax))
        self._block = True
        self._sl.setValue(self._to_ticks(v))
        self._sb.setValue(v)
        self._block = False
        if not silent:
            self.valueChanged.emit(v)


# ── Thread-safe signal bridge ────────────────────────────────────────────────

class _StatusBridge(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-PENDULUM PREVIEW WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

_COL_ROD  = "#8899cc"
_COL_BOB1 = "#4499ff"
_COL_BOB2 = "#ff8833"

_DT  = 1.0 / 500          # integration timestep
_FPS = 30                  # animation frame rate


def _scalar_derivs(s, L, m1, m2, g):
    """Lagrangian EoM for a single double pendulum (scalar)."""
    th1, w1, th2, w2 = s
    d  = th1 - th2
    D  = 2*m1 + m2 - m2*np.cos(2*d)
    a1 = (-g*(2*m1+m2)*np.sin(th1)
          - m2*g*np.sin(th1 - 2*th2)
          - 2*np.sin(d)*m2*(w2**2*L + w1**2*L*np.cos(d))
         ) / (L * D)
    a2 = (2*np.sin(d)*(w1**2*L*(m1+m2)
                        + g*(m1+m2)*np.cos(th1)
                        + w2**2*L*m2*np.cos(d))
         ) / (L * D)
    return np.array([w1, a1, w2, a2])


def _scalar_rk4(s, dt, L, m1, m2, g):
    k1 = _scalar_derivs(s, L, m1, m2, g)
    k2 = _scalar_derivs(s + 0.5*dt*k1, L, m1, m2, g)
    k3 = _scalar_derivs(s + 0.5*dt*k2, L, m1, m2, g)
    k4 = _scalar_derivs(s + dt*k3, L, m1, m2, g)
    return s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


class PendulumPreviewWindow(QWidget):
    """Separate window showing an animated double pendulum at given (θ1, θ2)."""

    def __init__(self, th1_deg, th2_deg, L=1.0, m1=1.0, m2=1.0, g=9.81,
                 parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle(
            f"Double Pendulum  θ₁={th1_deg:.1f}°  θ₂={th2_deg:.1f}°")
        self.resize(700, 700)

        self._L  = L
        self._m1 = m1
        self._m2 = m2
        self._g  = g

        self._state = np.array([np.radians(th1_deg), 0.0,
                                np.radians(th2_deg), 0.0])
        self._init_state = self._state.copy()
        self._t     = 0.0
        self._trail = deque(maxlen=600)
        self._trail_cmap = matplotlib.colormaps["hot"]

        self._build_ui(th1_deg, th2_deg)
        self._build_canvas()
        self._paused = False

        self._timer = QTimer(self)
        self._timer.setInterval(1000 // _FPS)
        self._timer.timeout.connect(self._frame)
        self._timer.start()

    def _build_ui(self, th1_deg, th2_deg):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        self._lbl = QLabel(
            f"θ₁ = {th1_deg:.1f}°   θ₂ = {th2_deg:.1f}°   "
            f"m₁ = {self._m1:.2f}   m₂ = {self._m2:.2f}   g = {self._g:.2f}")
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setStyleSheet("color:#aaa; font-size:8pt;")
        lay.addWidget(self._lbl)

        self._fig = Figure(facecolor=DARK_BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._canvas, 1)

        btn_row = QWidget()
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(6)

        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_pause.clicked.connect(self._toggle_pause)
        btn_lay.addWidget(self._btn_pause)

        btn_reset = QPushButton("↺  Reset")
        btn_reset.clicked.connect(self._reset)
        btn_lay.addWidget(btn_reset)

        btn_lay.addStretch()
        lay.addWidget(btn_row)

    def _build_canvas(self):
        self._ax = self._fig.add_subplot(111, facecolor=DARK_BG, aspect="equal")
        for sp in self._ax.spines.values():
            sp.set_visible(False)
        self._ax.set_xticks([])
        self._ax.set_yticks([])

        lim = self._L * 2 * 1.15
        self._ax.set_xlim(-lim, lim)
        self._ax.set_ylim(-lim, lim)

        self._trail_lc = LineCollection(
            [], linewidth=1.5, capstyle="round", zorder=1)
        self._ax.add_collection(self._trail_lc)

        self._rod1, = self._ax.plot([], [], color=_COL_ROD, lw=4,
                                     solid_capstyle="round", zorder=2)
        self._rod2, = self._ax.plot([], [], color=_COL_ROD, lw=3,
                                     solid_capstyle="round", zorder=2)
        self._ax.plot([0], [0], "o", color="white", ms=8, zorder=5)

        ms1 = np.clip(self._m1 * 12, 8, 28)
        ms2 = np.clip(self._m2 * 12, 8, 28)
        self._bob1, = self._ax.plot([], [], "o", color=_COL_BOB1,
                                     ms=ms1, zorder=4, markeredgewidth=0)
        self._bob2, = self._ax.plot([], [], "o", color=_COL_BOB2,
                                     ms=ms2, zorder=4, markeredgewidth=0)

        self._info_txt = self._ax.text(
            0.02, 0.97, "", transform=self._ax.transAxes,
            va="top", ha="left", color=CTRL_FG, fontsize=8,
            fontfamily="monospace")

    def _positions(self, s):
        th1, _, th2, _ = s
        L = self._L
        x1 =  L*np.sin(th1);       y1 = -L*np.cos(th1)
        x2 = x1 + L*np.sin(th2);   y2 = y1 - L*np.cos(th2)
        return (x1, y1), (x2, y2)

    def _energy(self, s):
        th1, w1, th2, w2 = s
        L, m1, m2, g = self._L, self._m1, self._m2, self._g
        KE = (0.5*(m1+m2)*L**2*w1**2
              + 0.5*m2*L**2*w2**2
              + m2*L*L*w1*w2*np.cos(th1 - th2))
        PE = -(m1+m2)*g*L*np.cos(th1) - m2*g*L*np.cos(th2)
        return KE, PE

    def _frame(self):
        if self._paused:
            return
        n_steps = max(1, round(1.0 / (_FPS * _DT)))
        for _ in range(n_steps):
            self._state = _scalar_rk4(
                self._state, _DT, self._L, self._m1, self._m2, self._g)
            self._t += _DT

        (x1, y1), (x2, y2) = self._positions(self._state)
        self._trail.append((x2, y2))

        self._rod1.set_data([0, x1], [0, y1])
        self._rod2.set_data([x1, x2], [y1, y2])
        self._bob1.set_data([x1], [y1])
        self._bob2.set_data([x2], [y2])

        arr = np.asarray(self._trail)
        if len(arr) > 1:
            pts  = arr.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            n = len(segs)
            alphas = np.linspace(0.04, 0.92, n)
            rgba = self._trail_cmap(0.55 + 0.45*alphas)
            rgba[:, 3] = alphas
            self._trail_lc.set_segments(segs)
            self._trail_lc.set_color(rgba)
        else:
            self._trail_lc.set_segments([])

        KE, PE = self._energy(self._state)
        self._info_txt.set_text(
            f"t  = {self._t:7.2f} s\n"
            f"KE = {KE:7.3f} J\n"
            f"PE = {PE:7.3f} J\n"
            f"E  = {KE+PE:7.3f} J")

        self._canvas.draw_idle()

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._btn_pause.setText("▶  Resume")
        else:
            self._btn_pause.setText("⏸  Pause")

    def _reset(self):
        self._state = self._init_state.copy()
        self._t = 0.0
        self._trail.clear()
        self._paused = False
        self._btn_pause.setText("⏸  Pause")

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class PendulumChaosMapCUDA(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double Pendulum Chaos Map — CUDA")
        self.resize(1440, 860)

        self._computing  = False
        self._generation = 0
        self._grids      = None
        self._metric     = "Flips (θ₂)"
        self._cmap_name  = "inferno"
        self._log_scale  = False
        self._t0         = 0.0
        self._previews   = []

        self._th1_min = -180.0
        self._th1_max =  180.0
        self._th2_min = -180.0
        self._th2_max =  180.0

        self._phys_m1 = 1.0
        self._phys_m2 = 1.0
        self._phys_g  = 9.81

        self._bridge = _StatusBridge()
        self._bridge.progress.connect(self._on_progress)
        self._bridge.finished.connect(self._on_compute_done)
        self._bridge.error.connect(self._on_compute_error)

        self._build_plot()
        self._build_controls()
        self._wire_central()

    # ── Canvas ───────────────────────────────────────────────────────────────

    def _build_plot(self):
        self.fig = Figure(facecolor=DARK_BG)
        self._canvas = FigureCanvas(self.fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[1, 0.04], wspace=0.05)
        self.ax    = self.fig.add_subplot(gs[0, 0])
        self._cbar_ax = self.fig.add_subplot(gs[0, 1])

        self.ax.set_facecolor(DARK_BG)
        self.ax.set_xlabel("θ₁  (degrees)", color=CTRL_FG, fontsize=9)
        self.ax.set_ylabel("θ₂  (degrees)", color=CTRL_FG, fontsize=9)
        self.ax.set_title("Double Pendulum Chaos Map", color=CTRL_FG,
                          fontsize=11, pad=10)
        self.ax.tick_params(colors="#556", labelsize=7)
        self._cbar_ax.set_visible(False)

        self._im = None
        self._colorbar = None

        self._canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._canvas.mpl_connect("button_press_event", self._on_click)

    # ── Controls ─────────────────────────────────────────────────────────────

    def _build_controls(self):
        self._ctrl = QWidget()
        lay = QVBoxLayout(self._ctrl)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)

        # title
        title = QLabel("DOUBLE PENDULUM\nCHAOS MAP  (CUDA)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color:{GOLD}; font-size:11pt; font-weight:bold;")
        lay.addWidget(title)

        # ── Metric selector ─────────────────────────────────────────────
        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        lbl = QLabel("Metric:")
        lbl.setStyleSheet("font-weight:bold;")
        r_lay.addWidget(lbl)
        self._cb_metric = QComboBox()
        self._cb_metric.addItems(["Flips (θ₂)", "Flips (θ₁+θ₂)", "Peak |ω₂|"])
        self._cb_metric.currentTextChanged.connect(self._on_metric_change)
        r_lay.addWidget(self._cb_metric)
        lay.addWidget(row)

        # ── Angular Range ───────────────────────────────────────────────
        grp = QGroupBox("Angular Range")
        g_lay = QVBoxLayout(grp)
        self.sl_th1_min = _ParamSlider("θ₁ min", -180, 0, -180, 1, "°")
        self.sl_th1_max = _ParamSlider("θ₁ max", 0, 180, 180, 1, "°")
        self.sl_th2_min = _ParamSlider("θ₂ min", -180, 0, -180, 1, "°")
        self.sl_th2_max = _ParamSlider("θ₂ max", 0, 180, 180, 1, "°")
        for w in (self.sl_th1_min, self.sl_th1_max,
                  self.sl_th2_min, self.sl_th2_max):
            g_lay.addWidget(w)
        lay.addWidget(grp)

        # ── Grid Resolution ─────────────────────────────────────────────
        grp = QGroupBox("Grid Resolution")
        g_lay = QVBoxLayout(grp)

        btn_row = QWidget()
        btn_lay = QGridLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(4)
        for i, (label, n) in enumerate([
            ("Coarse 50", 50), ("Med 100", 100), ("Fine 200", 200),
            ("HD 500", 500), ("Ultra 1K", 1000),
        ]):
            b = QPushButton(label)
            b.clicked.connect(lambda _, nn=n: self._sb_grid.setValue(nn))
            btn_lay.addWidget(b, i // 3, i % 3)
        g_lay.addWidget(btn_row)

        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("N × N :"))
        self._sb_grid = QSpinBox()
        self._sb_grid.setRange(10, 5000)
        self._sb_grid.setValue(100)
        self._sb_grid.setFixedWidth(76)
        r_lay.addWidget(self._sb_grid)
        r_lay.addStretch()
        g_lay.addWidget(row)
        lay.addWidget(grp)

        # ── Simulation ──────────────────────────────────────────────────
        grp = QGroupBox("Simulation")
        g_lay = QVBoxLayout(grp)
        self.sl_t_end = _ParamSlider("Duration (s)", 5, 60, 20, 1)
        g_lay.addWidget(self.sl_t_end)
        lay.addWidget(grp)

        # ── Backend ─────────────────────────────────────────────────────
        grp = QGroupBox("Backend")
        g_lay = QVBoxLayout(grp)

        # Backend selector
        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Engine:"))
        self._cb_backend = QComboBox()
        self._cb_backend.addItem("CUDA (GPU)")
        self._cb_backend.addItem("CPU (NumPy)")
        if not CUDA_AVAILABLE:
            self._cb_backend.setCurrentIndex(1)         # default to CPU
            self._cb_backend.model().item(0).setEnabled(False)
        self._cb_backend.currentIndexChanged.connect(self._on_backend_change)
        r_lay.addWidget(self._cb_backend)
        g_lay.addWidget(row)

        # CUDA-specific controls container
        self._cuda_controls = QWidget()
        cuda_lay = QVBoxLayout(self._cuda_controls)
        cuda_lay.setContentsMargins(0, 0, 0, 0)
        cuda_lay.setSpacing(4)

        # GPU checkboxes
        self._gpu_chks = []
        if _CUDA_GPUS:
            for gid, gname in _CUDA_GPUS:
                chk = QCheckBox(f"GPU {gid}: {gname}")
                chk.setChecked(True)
                self._gpu_chks.append(chk)
                cuda_lay.addWidget(chk)
        else:
            lbl = QLabel(_CUDA_ERR if _CUDA_ERR else "No GPUs detected")
            lbl.setStyleSheet("color:#ff6666; font-size:8pt;")
            cuda_lay.addWidget(lbl)

        # Precision selector
        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Precision:"))
        self._cb_precision = QComboBox()
        self._cb_precision.addItems(["float32 (fast)", "float64"])
        r_lay.addWidget(self._cb_precision)
        cuda_lay.addWidget(row)

        g_lay.addWidget(self._cuda_controls)

        # CPU-specific controls container
        self._cpu_controls = QWidget()
        cpu_lay = QVBoxLayout(self._cpu_controls)
        cpu_lay.setContentsMargins(0, 0, 0, 0)
        cpu_lay.setSpacing(4)

        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Workers:"))
        self._sb_workers = QSpinBox()
        n_cpus = os.cpu_count() or 4
        self._sb_workers.setRange(1, n_cpus)
        self._sb_workers.setValue(max(1, n_cpus - 1))
        self._sb_workers.setFixedWidth(56)
        r_lay.addWidget(self._sb_workers)
        cpu_lbl = QLabel(f"  / {n_cpus} cores")
        cpu_lbl.setStyleSheet("color:#888; font-size:8pt;")
        r_lay.addWidget(cpu_lbl)
        r_lay.addStretch()
        cpu_lay.addWidget(row)

        g_lay.addWidget(self._cpu_controls)
        lay.addWidget(grp)

        # Set initial visibility
        self._on_backend_change(self._cb_backend.currentIndex())

        # ── Physics ─────────────────────────────────────────────────────
        grp = QGroupBox("Physics")
        g_lay = QVBoxLayout(grp)
        lbl = QLabel("L₁ = L₂ = 1.0 m  (fixed, equal arms)")
        lbl.setStyleSheet("color:#888; font-size:8pt; font-style:italic;")
        g_lay.addWidget(lbl)
        self.sl_m1 = _ParamSlider("m₁ (kg)", 0.1, 5.0, 1.0, 2)
        self.sl_m2 = _ParamSlider("m₂ (kg)", 0.1, 5.0, 1.0, 2)
        self.sl_g  = _ParamSlider("g (m/s²)", 0.1, 25.0, 9.81, 2)
        for w in (self.sl_m1, self.sl_m2, self.sl_g):
            g_lay.addWidget(w)
        lay.addWidget(grp)

        # ── Display ─────────────────────────────────────────────────────
        grp = QGroupBox("Display")
        g_lay = QVBoxLayout(grp)

        row = QWidget()
        r_lay = QHBoxLayout(row)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)
        r_lay.addWidget(QLabel("Colormap:"))
        self._cb_cmap = QComboBox()
        self._cb_cmap.addItems([
            "inferno", "viridis", "plasma", "magma", "hot", "turbo",
            "twilight",
        ])
        self._cb_cmap.currentTextChanged.connect(self._on_cmap_change)
        r_lay.addWidget(self._cb_cmap)
        g_lay.addWidget(row)

        self._chk_log = QCheckBox("Log scale  (reveals boundary detail)")
        self._chk_log.stateChanged.connect(self._on_log_toggle)
        g_lay.addWidget(self._chk_log)
        lay.addWidget(grp)

        # ── Actions ─────────────────────────────────────────────────────
        grp = QGroupBox("Actions")
        g_lay = QVBoxLayout(grp)

        self._btn_compute = QPushButton("▶  Compute")
        self._btn_compute.setStyleSheet(
            f"color:{GOLD}; font-weight:bold; font-size:10pt; padding:8px;")
        self._btn_compute.clicked.connect(self._start_compute)
        g_lay.addWidget(self._btn_compute)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        g_lay.addWidget(self._progress)

        self._btn_export = QPushButton("Export PNG")
        self._btn_export.clicked.connect(self._export)
        g_lay.addWidget(self._btn_export)

        self._btn_load = QPushButton("Load .pkl")
        self._btn_load.clicked.connect(self._load_pkl)
        g_lay.addWidget(self._btn_load)
        lay.addWidget(grp)

        # ── Status ──────────────────────────────────────────────────────
        status_msg = "Ready.  Set parameters and click Compute."
        if CUDA_AVAILABLE:
            gpu_str = ", ".join(name for _, name in _CUDA_GPUS)
            status_msg += f"\nCUDA: {gpu_str}"
        else:
            status_msg += f"\n{_CUDA_ERR}"
        self._lbl_status = QLabel(status_msg)
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color:#888; font-size:8pt;")
        lay.addWidget(self._lbl_status)

        lay.addStretch()

    def _on_backend_change(self, idx):
        is_cuda = (idx == 0)
        self._cuda_controls.setVisible(is_cuda)
        self._cpu_controls.setVisible(not is_cuda)

    # ── Layout ───────────────────────────────────────────────────────────────

    def _wire_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)
        h.addWidget(self._canvas, 3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(310)
        scroll.setWidget(self._ctrl)
        h.addWidget(scroll)

    # ── Compute ──────────────────────────────────────────────────────────────

    def _start_compute(self):
        if self._computing:
            return
        self._computing = True
        self._generation += 1
        gen = self._generation

        self._btn_compute.setEnabled(False)
        self._btn_compute.setText("Computing…")
        self._progress.setValue(0)

        # read parameters
        th1_min = np.radians(self.sl_th1_min.value())
        th1_max = np.radians(self.sl_th1_max.value())
        th2_min = np.radians(self.sl_th2_min.value())
        th2_max = np.radians(self.sl_th2_max.value())
        n_grid  = self._sb_grid.value()
        t_end   = self.sl_t_end.value()
        m1      = self.sl_m1.value()
        m2      = self.sl_m2.value()
        g       = self.sl_g.value()

        # store for display axes
        self._th1_min = self.sl_th1_min.value()
        self._th1_max = self.sl_th1_max.value()
        self._th2_min = self.sl_th2_min.value()
        self._th2_max = self.sl_th2_max.value()
        self._n_grid  = n_grid
        self._t_end_v = t_end
        self._phys_m1 = m1
        self._phys_m2 = m2
        self._phys_g  = g

        self._t0 = time.time()

        bridge    = self._bridge
        is_cuda   = self._cb_backend.currentIndex() == 0

        if is_cuda:
            gpu_ids  = [_CUDA_GPUS[i][0]
                        for i, chk in enumerate(self._gpu_chks)
                        if chk.isChecked()]
            use_fp32 = self._cb_precision.currentText().startswith("float32")
            self._compute_info = f"{len(gpu_ids)} GPU(s), " \
                                 f"{'fp32' if use_fp32 else 'fp64'}"
            if not gpu_ids:
                self._computing = False
                self._btn_compute.setEnabled(True)
                self._btn_compute.setText("▶  Compute")
                self._lbl_status.setText("No GPUs selected.")
                return
        else:
            n_workers = self._sb_workers.value()
            self._compute_info = f"{n_workers} CPU workers"

        def worker():
            try:
                def pcb(pct):
                    if gen == self._generation:
                        bridge.progress.emit(pct)

                if is_cuda:
                    result = compute_chaos_grid_cuda(
                        (th1_min, th1_max), (th2_min, th2_max),
                        n_grid, n_grid,
                        L=1.0, m1=m1, m2=m2, g=g, t_end=t_end, dt=0.002,
                        gpu_ids=gpu_ids, use_fp32=use_fp32,
                        progress_cb=pcb)
                else:
                    result = compute_chaos_grid(
                        (th1_min, th1_max), (th2_min, th2_max),
                        n_grid, n_grid,
                        L=1.0, m1=m1, m2=m2, g=g, t_end=t_end, dt=0.002,
                        n_workers=n_workers, progress_cb=pcb)

                if gen == self._generation:
                    bridge.finished.emit(result)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                bridge.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, pct):
        self._progress.setValue(pct)

    def _on_compute_done(self, result):
        elapsed = time.time() - self._t0
        self._computing = False
        self._btn_compute.setEnabled(True)
        self._btn_compute.setText("▶  Compute")
        self._progress.setValue(100)

        self._grids = {
            "Flips (θ₂)":    result["flips_2"],
            "Flips (θ₁+θ₂)": result["flips_both"],
            "Peak |ω₂|":     result["max_w2"],
        }

        grid = self._grids[self._metric]
        mn, mx = grid.min(), grid.max()

        self._save_pkl()

        if self._metric == "Peak |ω₂|":
            stats = f"Peak |ω₂|: min {mn:.1f},  max {mx:.1f} rad/s"
        else:
            stats = f"Flips: min {int(mn)},  max {int(mx)}"

        self._lbl_status.setText(
            f"{self._n_grid}×{self._n_grid} grid  |  "
            f"t = {self._t_end_v:.0f}s  |  "
            f"{self._compute_info}  |  "
            f"{elapsed:.1f}s elapsed\n{stats}"
        )

        self._update_display()

    def _save_pkl(self):
        if self._grids is None:
            return
        data = dict(
            grids=self._grids,
            th1_min=self._th1_min, th1_max=self._th1_max,
            th2_min=self._th2_min, th2_max=self._th2_max,
            n_grid=self._n_grid, t_end=self._t_end_v,
            m1=self._phys_m1, m2=self._phys_m2, g=self._phys_g,
        )
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n = self._n_grid
        fname = f"chaos_map_{n}x{n}_t{self._t_end_v:.0f}s_{ts}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._last_pkl = fname

    def _load_pkl(self):
        from PyQt5.QtWidgets import QFileDialog
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load chaos map", ".", "Pickle files (*.pkl)")
        if not fname:
            return
        try:
            with open(fname, "rb") as f:
                data = pickle.load(f)

            if "grids" in data:
                self._grids = data["grids"]
            elif "grid" in data:
                old_grid = data["grid"]
                self._grids = {
                    "Flips (θ₂)":    old_grid,
                    "Flips (θ₁+θ₂)": old_grid,
                    "Peak |ω₂|":     np.zeros_like(old_grid, dtype=np.float64),
                }

            self._th1_min  = data["th1_min"]
            self._th1_max  = data["th1_max"]
            self._th2_min  = data["th2_min"]
            self._th2_max  = data["th2_max"]
            self._n_grid   = data["n_grid"]
            self._t_end_v  = data["t_end"]
            self._phys_m1  = data.get("m1", 1.0)
            self._phys_m2  = data.get("m2", 1.0)
            self._phys_g   = data.get("g", 9.81)
            n = self._n_grid
            grid = self._grids[self._metric]
            mn, mx = grid.min(), grid.max()
            self._lbl_status.setText(
                f"Loaded {n}×{n} grid from {fname.split('/')[-1]}\n"
                f"{self._metric}: min {mn},  max {mx}")
            self._update_display()
        except Exception as exc:
            self._lbl_status.setText(f"Load error: {exc}")

    def _on_compute_error(self, msg):
        self._computing = False
        self._btn_compute.setEnabled(True)
        self._btn_compute.setText("▶  Compute")
        self._lbl_status.setText(f"Error: {msg}")

    # ── Display ──────────────────────────────────────────────────────────────

    def _update_display(self):
        if self._grids is None:
            return

        grid = self._grids[self._metric].astype(np.float64)
        if self._log_scale:
            grid = np.log1p(grid)

        self.ax.clear()
        self._im = self.ax.imshow(
            grid,
            extent=[self._th1_min, self._th1_max,
                    self._th2_min, self._th2_max],
            origin="lower",
            cmap=self._cmap_name,
            aspect="auto",
            interpolation="nearest",
        )
        self.ax.set_xlabel("θ₁  (degrees)", color=CTRL_FG, fontsize=9)
        self.ax.set_ylabel("θ₂  (degrees)", color=CTRL_FG, fontsize=9)
        self.ax.set_title("Double Pendulum Chaos Map", color=CTRL_FG,
                          fontsize=11, pad=10)
        self.ax.tick_params(colors="#556", labelsize=7)

        self._cbar_ax.clear()
        self._cbar_ax.set_visible(True)
        if self._log_scale:
            label = f"log(1 + {self._metric})"
        else:
            label = self._metric
        self._colorbar = self.fig.colorbar(self._im, cax=self._cbar_ax,
                                           label=label)
        self._colorbar.ax.yaxis.set_tick_params(color="#556", labelcolor="#aaa")
        self._colorbar.set_label(label, color=CTRL_FG, fontsize=8)

        self.fig.tight_layout()
        self._canvas.draw_idle()

    def _on_cmap_change(self, name):
        self._cmap_name = name
        self._update_display()

    def _on_metric_change(self, name):
        self._metric = name
        self._update_display()

    def _on_log_toggle(self, state):
        self._log_scale = bool(state)
        self._update_display()

    # ── Hover ────────────────────────────────────────────────────────────────

    def _on_hover(self, event):
        if self._grids is None or event.inaxes != self.ax:
            return
        th1, th2 = event.xdata, event.ydata
        if th1 is None or th2 is None:
            return
        grid = self._grids[self._metric]
        n2, n1 = grid.shape
        ci = int(round((th1 - self._th1_min) / (self._th1_max - self._th1_min)
                       * (n1 - 1)))
        ri = int(round((th2 - self._th2_min) / (self._th2_max - self._th2_min)
                       * (n2 - 1)))
        ci = np.clip(ci, 0, n1 - 1)
        ri = np.clip(ri, 0, n2 - 1)
        val = grid[ri, ci]
        if self._metric == "Peak |ω₂|":
            val_str = f"{val:.1f} rad/s"
        else:
            val_str = str(int(val))
        self._lbl_status.setText(
            f"θ₁ = {th1:.1f}°   θ₂ = {th2:.1f}°   "
            f"{self._metric} = {val_str}")

    # ── Click → open pendulum preview ────────────────────────────────────────

    def _on_click(self, event):
        if self._grids is None or event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        th1, th2 = event.xdata, event.ydata
        if th1 is None or th2 is None:
            return

        self._previews = [w for w in self._previews if w.isVisible()]

        win = PendulumPreviewWindow(
            th1, th2, L=1.0,
            m1=self._phys_m1, m2=self._phys_m2, g=self._phys_g)
        win.setStyleSheet(_QSS)
        win.show()
        self._previews.append(win)

        self._lbl_status.setText(
            f"Opened preview: θ₁ = {th1:.1f}°  θ₂ = {th2:.1f}°")

    # ── Export ───────────────────────────────────────────────────────────────

    def _export(self):
        if self._grids is None:
            self._lbl_status.setText("Nothing to export — run Compute first.")
            return

        grid = self._grids[self._metric].astype(np.float64)
        if self._log_scale:
            grid = np.log1p(grid)

        fig_e = Figure(figsize=(12, 10), dpi=300, facecolor=DARK_BG)
        ax_e  = fig_e.add_subplot(111)
        ax_e.set_facecolor(DARK_BG)
        im = ax_e.imshow(
            grid,
            extent=[self._th1_min, self._th1_max,
                    self._th2_min, self._th2_max],
            origin="lower", cmap=self._cmap_name,
            aspect="auto", interpolation="nearest",
        )
        ax_e.set_xlabel("θ₁  (degrees)", color=CTRL_FG, fontsize=10)
        ax_e.set_ylabel("θ₂  (degrees)", color=CTRL_FG, fontsize=10)
        ax_e.set_title("Double Pendulum Chaos Map", color=CTRL_FG, fontsize=13)
        ax_e.tick_params(colors="#556", labelsize=8)
        if self._log_scale:
            label = f"log(1 + {self._metric})"
        else:
            label = self._metric
        cb = fig_e.colorbar(im, ax=ax_e, label=label)
        cb.ax.yaxis.set_tick_params(color="#556", labelcolor="#aaa")
        cb.set_label(label, color=CTRL_FG, fontsize=9)
        fig_e.tight_layout()

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n = self._grids[self._metric].shape[0]
        fname = (f"chaos_map_{n}x{n}_t{self._t_end_v:.0f}s"
                 f"_{self._cmap_name}_{ts}.png")
        fig_e.savefig(fname, dpi=300, facecolor=DARK_BG)
        import matplotlib.pyplot as plt
        plt.close(fig_e)

        self._lbl_status.setText(f"Saved: {fname}")
        self._btn_export.setText("✓ Saved")
        QTimer.singleShot(3000, lambda: self._btn_export.setText("Export PNG"))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(_QSS)
    win = PendulumChaosMapCUDA()
    win.show()
    sys.exit(app.exec_())
