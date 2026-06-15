#!/usr/bin/env python3
"""
Double Slit Explorer
====================
Interactive PyQt5 simulator for the optical double-slit experiment.

Features
--------
- Physically based intensity models:
  - Fraunhofer (far-field approximation)
  - Fresnel (near-field paraxial propagation via aperture integration)
- Source selection:
  - Single wavelength
  - Uniformly averaged wavelength band
- 3D detector rendering (matplotlib): X-Y plane with Z = intensity
- User controls for slit separation, slit width, detector distance, and more

Usage
-----
    conda run -n maths_fun python double_slit_explorer.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


NM_TO_M = 1e-9
UM_TO_M = 1e-6
MM_TO_M = 1e-3


@dataclass
class SimulationParams:
    model: str
    source_mode: str
    slit_separation_m: float
    slit_width_m: float
    slit_height_m: float
    detector_distance_m: float
    detector_span_m: float
    detector_resolution: int
    wavelength_m: float
    wavelength_min_m: float
    wavelength_max_m: float
    wavelength_samples: int
    use_log_display: bool


def build_detector_grid(span_m: float, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = 0.5 * span_m
    x = np.linspace(-half, half, resolution)
    y = np.linspace(-half, half, resolution)
    xx, yy = np.meshgrid(x, y)
    return x, y, xx, yy


def wavelength_samples_from_params(p: SimulationParams) -> np.ndarray:
    if p.source_mode == "Single":
        return np.array([p.wavelength_m], dtype=np.float64)
    return np.linspace(p.wavelength_min_m, p.wavelength_max_m, p.wavelength_samples, dtype=np.float64)


def fraunhofer_intensity(xx: np.ndarray, yy: np.ndarray, wavelength_m: float, p: SimulationParams) -> np.ndarray:
    lam = wavelength_m
    L = p.detector_distance_m
    d = p.slit_separation_m
    a = p.slit_width_m
    b = p.slit_height_m

    beta = d * xx / (lam * L)
    envelope_x = np.sinc(a * xx / (lam * L)) ** 2
    envelope_y = np.sinc(b * yy / (lam * L)) ** 2
    interference = np.cos(np.pi * beta) ** 2
    return envelope_x * envelope_y * interference


def _fresnel_axis_integral(points: np.ndarray, intervals: list[tuple[float, float]], alpha: float, n_samples: int) -> np.ndarray:
    out = np.zeros(points.shape[0], dtype=np.complex128)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for left, right in intervals:
        samples = np.linspace(left, right, n_samples)
        phase = alpha * (points[:, None] - samples[None, :]) ** 2
        out += integrate(np.exp(1j * phase), samples, axis=1)
    return out


def fresnel_intensity(x: np.ndarray, y: np.ndarray, wavelength_m: float, p: SimulationParams) -> np.ndarray:
    lam = wavelength_m
    L = p.detector_distance_m
    d = p.slit_separation_m
    a = p.slit_width_m
    b = p.slit_height_m

    k = 2.0 * np.pi / lam
    alpha = k / (2.0 * L)

    x_intervals = [
        (-0.5 * d - 0.5 * a, -0.5 * d + 0.5 * a),
        (0.5 * d - 0.5 * a, 0.5 * d + 0.5 * a),
    ]
    y_intervals = [(-0.5 * b, 0.5 * b)]

    # Keep sample count tied to grid size for smoother interactions.
    n_samples = int(np.clip(p.detector_resolution, 80, 300))

    ix = _fresnel_axis_integral(x, x_intervals, alpha, n_samples)
    iy = _fresnel_axis_integral(y, y_intervals, alpha, n_samples)

    field = np.outer(iy, ix)
    prefactor = 1.0 / (lam * L)
    intensity = np.abs(prefactor * field) ** 2
    return intensity


def compute_intensity(p: SimulationParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, xx, yy = build_detector_grid(p.detector_span_m, p.detector_resolution)
    wavelengths = wavelength_samples_from_params(p)

    total = np.zeros_like(xx, dtype=np.float64)
    for lam in wavelengths:
        if p.model == "Fraunhofer":
            total += fraunhofer_intensity(xx, yy, lam, p)
        else:
            total += fresnel_intensity(x, y, lam, p)

    total /= float(wavelengths.size)
    return xx, yy, total


def normalize_for_display(intensity: np.ndarray, use_log_display: bool) -> np.ndarray:
    data = np.maximum(intensity, 0.0)
    vmax = float(np.max(data))
    if vmax <= 0.0:
        return np.zeros_like(data)

    norm = data / vmax
    if use_log_display:
        norm = np.log1p(120.0 * norm) / np.log1p(120.0)
    return np.clip(norm, 0.0, 1.0)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = int(max(1, round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def _gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return arr
    kernel = _gaussian_kernel_1d(sigma)
    pad = kernel.size // 2

    # Separable convolution: horizontal pass.
    padded_x = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    tmp = np.empty_like(arr, dtype=np.float64)
    for i in range(arr.shape[0]):
        tmp[i, :] = np.convolve(padded_x[i, :], kernel, mode="valid")

    # Vertical pass.
    padded_y = np.pad(tmp, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(tmp, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(padded_y[:, j], kernel, mode="valid")

    return out


def tone_map_display(
    intensity: np.ndarray,
    use_log_display: bool,
    log_strength: float,
    gamma: float,
    smoothing_sigma: float,
) -> np.ndarray:
    data = np.maximum(intensity, 0.0)
    vmax = float(np.max(data))
    if vmax <= 0.0:
        return np.zeros_like(data)

    z = data / vmax

    if use_log_display:
        ls = max(1e-6, float(log_strength))
        z = np.log1p(ls * z) / np.log1p(ls)

    g = max(0.05, float(gamma))
    z = np.power(np.clip(z, 0.0, 1.0), g)

    z = _gaussian_blur_2d(z, max(0.0, float(smoothing_sigma)))
    return np.clip(z, 0.0, 1.0)


class DoubleSlitExplorer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Double Slit Explorer")
        self.resize(1400, 860)

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.recompute_and_plot)

        self._building = False
        self._surface = None
        self._last_export_state = None

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        controls = self._build_controls()
        plot_area = self._build_plot_canvas()

        layout.addWidget(controls, stretch=0)
        layout.addWidget(plot_area, stretch=1)

        self._set_band_widgets_enabled(False)
        self.recompute_and_plot()

    def _build_plot_canvas(self) -> QWidget:
        box = QGroupBox("3D Intensity")
        v = QVBoxLayout(box)

        self.fig = Figure(figsize=(8, 6), facecolor="#0f1117")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlabel("X [mm]")
        self.ax.set_ylabel("Y [mm]")
        self.ax.set_zlabel("Normalized intensity")
        self.ax.set_title("Double-slit diffraction pattern")

        v.addWidget(self.canvas)

        return box

    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)

        panel = QWidget()
        main_v = QVBoxLayout(panel)

        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Fraunhofer", "Fresnel"])

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Single", "Band"])

        model_form.addRow("Propagation", self.model_combo)
        model_form.addRow("Wavelength mode", self.source_combo)

        geom_group = QGroupBox("Geometry")
        geom_form = QFormLayout(geom_group)

        self.slit_sep_um = self._spin_float(40.0, 1000.0, 220.0, 1.0)
        self.slit_width_um = self._spin_float(5.0, 300.0, 45.0, 1.0)
        self.slit_height_um = self._spin_float(10.0, 800.0, 120.0, 1.0)
        self.detector_dist_m = self._spin_float(0.05, 10.0, 1.5, 0.01)
        self.detector_span_mm = self._spin_float(5.0, 200.0, 40.0, 0.5)
        self.resolution = self._spin_int(40, 220, 120, 1)

        geom_form.addRow("Slit separation [um]", self.slit_sep_um)
        geom_form.addRow("Slit width [um]", self.slit_width_um)
        geom_form.addRow("Slit height [um]", self.slit_height_um)
        geom_form.addRow("Detector distance [m]", self.detector_dist_m)
        geom_form.addRow("Detector span [mm]", self.detector_span_mm)
        geom_form.addRow("Grid resolution", self.resolution)

        wave_group = QGroupBox("Source")
        wave_grid = QGridLayout(wave_group)

        self.lambda_nm = self._spin_float(200.0, 2000.0, 532.0, 1.0)
        self.lambda_min_nm = self._spin_float(380.0, 2000.0, 450.0, 1.0)
        self.lambda_max_nm = self._spin_float(380.0, 2000.0, 700.0, 1.0)
        self.lambda_samples = self._spin_int(2, 60, 20, 1)

        wave_grid.addWidget(QLabel("Single wavelength [nm]"), 0, 0)
        wave_grid.addWidget(self.lambda_nm, 0, 1)
        wave_grid.addWidget(QLabel("Band min [nm]"), 1, 0)
        wave_grid.addWidget(self.lambda_min_nm, 1, 1)
        wave_grid.addWidget(QLabel("Band max [nm]"), 2, 0)
        wave_grid.addWidget(self.lambda_max_nm, 2, 1)
        wave_grid.addWidget(QLabel("Band samples"), 3, 0)
        wave_grid.addWidget(self.lambda_samples, 3, 1)

        self.log_check = QCheckBox("Log intensity display")
        self.log_check.setChecked(True)

        tone_group = QGroupBox("Display Tone Mapping")
        tone_form = QFormLayout(tone_group)
        self.log_strength = self._spin_float(1.0, 1000.0, 120.0, 1.0)
        self.gamma_spin = self._spin_float(0.30, 2.50, 0.85, 0.01)
        self.smoothing_sigma = self._spin_float(0.00, 3.00, 0.80, 0.05)
        tone_form.addRow("Log strength", self.log_strength)
        tone_form.addRow("Gamma", self.gamma_spin)
        tone_form.addRow("Gaussian sigma", self.smoothing_sigma)

        render_group = QGroupBox("Plot")
        render_form = QFormLayout(render_group)

        self.grid_check = QCheckBox("Show axis grid")
        self.grid_check.setChecked(False)

        self.axes_check = QCheckBox("Show axes")
        self.axes_check.setChecked(True)

        self.sample_rows = self._spin_int(40, 1200, 320, 10)
        self.sample_cols = self._spin_int(40, 1200, 320, 10)
        self.cmap_levels = self._spin_int(64, 4096, 2048, 64)
        self.render_upsample = self._spin_int(1, 4, 2, 1)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(
            [
                "inferno",
                "viridis",
                "plasma",
                "magma",
                "cividis",
                "turbo",
                "cubehelix",
                "jet",
            ]
        )

        render_form.addRow("Colormap", self.cmap_combo)
        render_form.addRow("Color levels", self.cmap_levels)
        render_form.addRow("Render upsample", self.render_upsample)
        render_form.addRow("Surface rows", self.sample_rows)
        render_form.addRow("Surface cols", self.sample_cols)
        render_form.addRow("Grid", self.grid_check)
        render_form.addRow("Axes", self.axes_check)

        export_group = QGroupBox("Export")
        export_form = QFormLayout(export_group)

        self.export_dpi = QComboBox()
        self.export_dpi.addItems(["150", "300", "600"])
        self.export_transparent_check = QCheckBox("Transparent background")
        self.export_transparent_check.setChecked(False)
        self.export_btn = QPushButton("Export PNG")
        self.export_btn.clicked.connect(self._export_png)
        self.export_heatmap_btn = QPushButton("Export 2D Heatmap")
        self.export_heatmap_btn.clicked.connect(self._export_heatmap)

        export_form.addRow("DPI", self.export_dpi)
        export_form.addRow("Background", self.export_transparent_check)
        export_form.addRow("", self.export_btn)
        export_form.addRow("", self.export_heatmap_btn)

        self.compute_button = QPushButton("Recompute")
        self.compute_button.clicked.connect(self.recompute_and_plot)

        self.status = QLabel("Ready")
        self.status.setWordWrap(True)

        main_v.addWidget(model_group)
        main_v.addWidget(geom_group)
        main_v.addWidget(wave_group)
        main_v.addWidget(self.log_check)
        main_v.addWidget(tone_group)
        main_v.addWidget(render_group)
        main_v.addWidget(export_group)
        main_v.addWidget(self.compute_button)
        main_v.addWidget(self.status)
        main_v.addStretch(1)

        scroll.setWidget(panel)
        self._wire_signals()
        return scroll

    @staticmethod
    def _spin_float(min_v: float, max_v: float, value: float, step: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(min_v, max_v)
        sb.setValue(value)
        sb.setSingleStep(step)
        sb.setDecimals(4 if step < 0.01 else 2 if step < 0.1 else 1)
        sb.setAlignment(Qt.AlignRight)
        return sb

    @staticmethod
    def _spin_int(min_v: int, max_v: int, value: int, step: int) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(min_v, max_v)
        sb.setValue(value)
        sb.setSingleStep(step)
        sb.setAlignment(Qt.AlignRight)
        return sb

    def _wire_signals(self) -> None:
        widgets = [
            self.model_combo,
            self.source_combo,
            self.slit_sep_um,
            self.slit_width_um,
            self.slit_height_um,
            self.detector_dist_m,
            self.detector_span_mm,
            self.resolution,
            self.lambda_nm,
            self.lambda_min_nm,
            self.lambda_max_nm,
            self.lambda_samples,
            self.log_check,
            self.log_strength,
            self.gamma_spin,
            self.smoothing_sigma,
            self.grid_check,
            self.axes_check,
            self.cmap_levels,
            self.render_upsample,
            self.sample_rows,
            self.sample_cols,
            self.cmap_combo,
        ]
        for w in widgets:
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self._on_controls_changed)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self._on_controls_changed)
            else:
                w.valueChanged.connect(self._on_controls_changed)

        self.source_combo.currentIndexChanged.connect(self._on_source_mode_changed)

    def _on_controls_changed(self) -> None:
        if self._building:
            return
        self._debounce_timer.start(140)

    def _on_source_mode_changed(self) -> None:
        is_band = self.source_combo.currentText() == "Band"
        self._set_band_widgets_enabled(is_band)
        self._on_controls_changed()

    def _set_band_widgets_enabled(self, is_band: bool) -> None:
        self.lambda_nm.setEnabled(not is_band)
        self.lambda_min_nm.setEnabled(is_band)
        self.lambda_max_nm.setEnabled(is_band)
        self.lambda_samples.setEnabled(is_band)

    def _params_from_ui(self) -> SimulationParams:
        model = self.model_combo.currentText()
        source_mode = self.source_combo.currentText()

        slit_sep = self.slit_sep_um.value() * UM_TO_M
        slit_width = self.slit_width_um.value() * UM_TO_M
        slit_height = self.slit_height_um.value() * UM_TO_M
        detector_distance = self.detector_dist_m.value()
        detector_span = self.detector_span_mm.value() * MM_TO_M
        resolution = int(self.resolution.value())

        wavelength = self.lambda_nm.value() * NM_TO_M
        wavelength_min = self.lambda_min_nm.value() * NM_TO_M
        wavelength_max = self.lambda_max_nm.value() * NM_TO_M
        wavelength_samples = int(self.lambda_samples.value())

        if slit_width <= 0.0 or slit_sep <= 0.0 or slit_height <= 0.0:
            raise ValueError("Slit dimensions must be positive.")
        if detector_distance <= 0.0:
            raise ValueError("Detector distance must be positive.")
        if detector_span <= 0.0:
            raise ValueError("Detector span must be positive.")
        if source_mode == "Band" and wavelength_min >= wavelength_max:
            raise ValueError("Band min wavelength must be smaller than band max wavelength.")

        return SimulationParams(
            model=model,
            source_mode=source_mode,
            slit_separation_m=slit_sep,
            slit_width_m=slit_width,
            slit_height_m=slit_height,
            detector_distance_m=detector_distance,
            detector_span_m=detector_span,
            detector_resolution=resolution,
            wavelength_m=wavelength,
            wavelength_min_m=wavelength_min,
            wavelength_max_m=wavelength_max,
            wavelength_samples=wavelength_samples,
            use_log_display=self.log_check.isChecked(),
        )

    def recompute_and_plot(self) -> None:
        try:
            params = self._params_from_ui()
        except ValueError as exc:
            self.status.setText(f"Invalid parameters: {exc}")
            return

        t0 = time.perf_counter()
        try:
            xx, yy, intensity = compute_intensity(params)
            z = tone_map_display(
                intensity,
                params.use_log_display,
                self.log_strength.value(),
                self.gamma_spin.value(),
                self.smoothing_sigma.value(),
            )
        except Exception as exc:
            self.status.setText(f"Computation failed: {exc}")
            return

        self.ax.clear()

        x_mm = xx / MM_TO_M
        y_mm = yy / MM_TO_M
        self._last_export_state = {
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_display": z,
            "params": params,
        }

        factor = int(self.render_upsample.value())
        x_plot, y_plot, z_plot = self._upsample_for_render(x_mm, y_mm, z, factor)
        target_rows = min(self.sample_rows.value(), z_plot.shape[0])
        target_cols = min(self.sample_cols.value(), z_plot.shape[1])
        x_surf, y_surf, z_surf = self._resample_surface_grid(x_plot, y_plot, z_plot, target_rows, target_cols)
        facecolors = self._surface_facecolors(z_surf)

        self._surface = self.ax.plot_surface(
            x_surf,
            y_surf,
            z_surf,
            facecolors=facecolors,
            linewidth=0.0,
            edgecolor="none",
            shade=False,
            antialiased=True,
            rcount=z_surf.shape[0],
            ccount=z_surf.shape[1],
        )

        self.ax.grid(self.grid_check.isChecked())

        if self.axes_check.isChecked():
            self.ax.set_axis_on()
            self.ax.set_xlabel("X [mm]")
            self.ax.set_ylabel("Y [mm]")
            self.ax.set_zlabel("Normalized intensity")
        else:
            self.ax.set_axis_off()

        mode_suffix = "(log)" if params.use_log_display else "(linear)"
        self.ax.set_title(f"{params.model} double-slit intensity {mode_suffix}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        wl_label = (
            f"{params.wavelength_m / NM_TO_M:.1f} nm"
            if params.source_mode == "Single"
            else (
                f"{params.wavelength_min_m / NM_TO_M:.1f}-{params.wavelength_max_m / NM_TO_M:.1f} nm "
                f"({params.wavelength_samples} samples)"
            )
        )
        self.status.setText(
            f"Rendered {params.model}, source={params.source_mode}, wavelength={wl_label}, "
            f"grid={params.detector_resolution}x{params.detector_resolution} in {elapsed_ms:.1f} ms"
        )

        self.canvas.draw_idle()

    def _selected_cmap(self):
        name = self.cmap_combo.currentText()
        levels = int(self.cmap_levels.value())
        try:
            base = matplotlib.colormaps[name]
        except Exception:
            base = matplotlib.cm.get_cmap(name)

        if isinstance(base, LinearSegmentedColormap):
            return base

        colors = base(np.linspace(0.0, 1.0, max(16, levels)))
        return LinearSegmentedColormap.from_list(f"{name}_smooth", colors)

    def _surface_facecolors(self, z: np.ndarray) -> np.ndarray:
        cmap = self._selected_cmap()
        z_faces = 0.25 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:])
        return cmap(np.clip(z_faces, 0.0, 1.0))

    @staticmethod
    def _upsample_for_render(
        x_mm: np.ndarray,
        y_mm: np.ndarray,
        z: np.ndarray,
        factor: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if factor <= 1:
            return x_mm, y_mm, z

        x = x_mm[0, :]
        y = y_mm[:, 0]
        nx = x.size
        ny = y.size

        nx2 = (nx - 1) * factor + 1
        ny2 = (ny - 1) * factor + 1
        x2 = np.linspace(float(x[0]), float(x[-1]), nx2)
        y2 = np.linspace(float(y[0]), float(y[-1]), ny2)

        z_x = np.empty((ny, nx2), dtype=np.float64)
        for j in range(ny):
            z_x[j, :] = np.interp(x2, x, z[j, :])

        z2 = np.empty((ny2, nx2), dtype=np.float64)
        for i in range(nx2):
            z2[:, i] = np.interp(y2, y, z_x[:, i])

        xx2, yy2 = np.meshgrid(x2, y2)
        return xx2, yy2, z2

    @staticmethod
    def _resample_surface_grid(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        rows: int,
        cols: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rows >= z.shape[0] and cols >= z.shape[1]:
            return x, y, z

        row_idx = np.linspace(0, z.shape[0] - 1, max(2, rows), dtype=int)
        col_idx = np.linspace(0, z.shape[1] - 1, max(2, cols), dtype=int)
        row_idx = np.unique(row_idx)
        col_idx = np.unique(col_idx)
        return x[np.ix_(row_idx, col_idx)], y[np.ix_(row_idx, col_idx)], z[np.ix_(row_idx, col_idx)]

    def _export_png(self) -> None:
        default_name = f"double_slit_{int(time.time())}.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export plot as PNG",
            default_name,
            "PNG Images (*.png)",
        )
        if not path:
            return

        try:
            dpi = int(self.export_dpi.currentText())
            transparent = self.export_transparent_check.isChecked()
            self.fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
            mode = "transparent" if transparent else "opaque"
            self.status.setText(f"Exported {mode} PNG to {path} at {dpi} DPI")
        except Exception as exc:
            self.status.setText(f"Export failed: {exc}")

    def _export_heatmap(self) -> None:
        if self._last_export_state is None:
            self.status.setText("Nothing to export yet. Render the plot first.")
            return

        default_name = f"double_slit_heatmap_{int(time.time())}.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export 2D heatmap as PNG",
            default_name,
            "PNG Images (*.png)",
        )
        if not path:
            return

        try:
            dpi = int(self.export_dpi.currentText())
            transparent = self.export_transparent_check.isChecked()
            state = self._last_export_state
            x_mm = state["x_mm"]
            y_mm = state["y_mm"]
            z = state["z_display"]
            params = state["params"]

            face = "none" if transparent else "#0f1117"
            text_color = "white" if transparent else "#e8e8f0"

            fig = Figure(figsize=(10, 7), facecolor=face)
            ax = fig.add_subplot(111)
            ax.set_facecolor(face)
            image = ax.imshow(
                z,
                origin="lower",
                extent=[x_mm[0, 0], x_mm[0, -1], y_mm[0, 0], y_mm[-1, 0]],
                cmap=self._selected_cmap(),
                interpolation="bicubic",
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
            )

            if self.axes_check.isChecked():
                ax.set_xlabel("Detector X (mm)", color=text_color)
                ax.set_ylabel("Detector Y (mm)", color=text_color)
                ax.tick_params(colors=text_color)
                for spine in ax.spines.values():
                    spine.set_color(text_color)
            else:
                ax.set_axis_off()

            if self.grid_check.isChecked() and self.axes_check.isChecked():
                ax.grid(True, color=text_color, alpha=0.15)

            mode_suffix = "log" if params.use_log_display else "linear"
            ax.set_title(
                f"{params.model} double-slit heatmap ({mode_suffix})",
                color=text_color,
                pad=10,
            )

            cbar = fig.colorbar(image, ax=ax, pad=0.02)
            cbar.set_label("Display intensity", color=text_color)
            cbar.ax.yaxis.set_tick_params(color=text_color)
            for tick in cbar.ax.get_yticklabels():
                tick.set_color(text_color)
            cbar.outline.set_edgecolor(text_color)

            fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
            mode = "transparent" if transparent else "opaque"
            self.status.setText(f"Exported {mode} 2D heatmap to {path} at {dpi} DPI")
        except Exception as exc:
            self.status.setText(f"Heatmap export failed: {exc}")


def main() -> None:
    app = QApplication(sys.argv)
    win = DoubleSlitExplorer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
