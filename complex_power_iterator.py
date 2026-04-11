#!/usr/bin/env python3
"""
Complex Power Iterator
======================
Input a complex number z₀ = x + iy, choose a power (2, 3, or 4),
then iterate z_{n+1} = z_n^p  N times.  Each step is drawn as a
chained vector on the complex plane originating from the previous result.

Run:  conda run -n maths_fun python3 complex_power_iterator.py
"""

import sys
import re
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QRadioButton, QButtonGroup, QSpinBox,
    QPushButton, QGroupBox, QTextEdit, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm

# ── constants ─────────────────────────────────────────────────────────────────
ESCAPE_RADIUS = 1e6   # stop iterating beyond this magnitude
BG            = '#1a1a2e'
PANEL_BG      = '#0d0d1a'
FG            = '#e0e0e0'
GRID_COL      = '#2a2a4a'

QSS = """
QMainWindow, QWidget {
    background: #1a1a2e;
    color: #e0e0e0;
}
QGroupBox {
    border: 1px solid #444466;
    border-radius: 4px;
    margin-top: 8px;
    font-weight: bold;
    color: #9090cc;
    padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QLineEdit, QSpinBox, QTextEdit {
    background: #0d0d1a;
    border: 1px solid #444466;
    border-radius: 3px;
    color: #e0e0e0;
    padding: 3px;
}
QPushButton {
    background: #2a2a5a;
    border: 1px solid #5555aa;
    border-radius: 4px;
    color: #c0c0ff;
    padding: 6px 10px;
    font-weight: bold;
}
QPushButton:hover  { background: #3a3a7a; }
QPushButton:pressed{ background: #444499; }
QRadioButton { color: #c0c0e0; spacing: 6px; }
QLabel       { color: #a0a0c0; }
QSpinBox::up-button, QSpinBox::down-button { width: 16px; }
"""


# ── complex number parser ──────────────────────────────────────────────────────
def parse_complex(text: str) -> complex:
    """Accept expressions like  3+2i  /  1.5 - 0.5i  /  i  /  -3  /  2j."""
    s = text.strip().replace(' ', '').lower().replace('j', 'i')

    # Group 1: optional real part   Group 2: optional imag part (ends in i)
    pat = re.compile(
        r'^(?P<real>[+-]?(?:\d+\.?\d*|\.\d+))?'
        r'(?P<imag>[+-]?(?:\d*\.?\d*)?i)?$'
    )
    m = pat.fullmatch(s)
    if not m or not (m.group('real') or m.group('imag')):
        raise ValueError(f"Cannot parse '{text}' as a complex number.\n"
                         "Use the form  x + iy  e.g.  0.8 + 0.6i")

    real = float(m.group('real')) if m.group('real') else 0.0

    imag_str = m.group('imag')
    if imag_str:
        coeff = imag_str.rstrip('i')
        if coeff in ('', '+'):
            imag = 1.0
        elif coeff == '-':
            imag = -1.0
        else:
            imag = float(coeff)
    else:
        imag = 0.0

    return complex(real, imag)


def fmt_z(z: complex) -> str:
    sign = '+' if z.imag >= 0 else '-'
    return f"{z.real:+.5f} {sign} {abs(z.imag):.5f}i"


# ── iteration engine ───────────────────────────────────────────────────────────
def iterate_power(z0: complex, power: int, n_iter: int):
    """
    Returns (points, escaped_at).
    points     – list of finite complex values  [z0, z1, …]
    escaped_at – iteration index where |z| > ESCAPE_RADIUS, or None
    """
    points = [z0]
    z = z0
    for i in range(1, n_iter + 1):
        try:
            z = z ** power
        except OverflowError:
            return points, i
        if not (np.isfinite(z.real) and np.isfinite(z.imag)) or abs(z) > ESCAPE_RADIUS:
            return points, i
        points.append(z)
    return points, None


# ── matplotlib canvas ──────────────────────────────────────────────────────────
class ComplexCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, facecolor=PANEL_BG)
        self._decorate()
        self.fig.tight_layout(pad=1.8)

    def _decorate(self):
        ax = self.ax
        ax.tick_params(colors=FG, labelsize=8)
        ax.set_xlabel('Re', color=FG, fontsize=9)
        ax.set_ylabel('Im', color=FG, fontsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444466')
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.8, zorder=0)
        ax.axhline(0, color='#555577', linewidth=0.9, alpha=0.9, zorder=1)
        ax.axvline(0, color='#555577', linewidth=0.9, alpha=0.9, zorder=1)
        ax.set_title('Complex Power Iteration', color=FG, fontsize=10, pad=8)

    # Draw unit circle (faint reference)
    def _unit_circle(self):
        th = np.linspace(0, 2 * np.pi, 360)
        self.ax.plot(np.cos(th), np.sin(th),
                     color='#33336a', linewidth=0.8, linestyle='--',
                     alpha=0.6, zorder=1, label='Unit circle')

    def plot_iteration(self, z0: complex, power: int, n_iter: int) -> str:
        self.ax.clear()
        self._decorate()
        self._unit_circle()

        points, escaped_at = iterate_power(z0, power, n_iter)
        n = len(points)

        # Coordinate arrays: chain starts at origin
        chain_x = [0.0] + [p.real for p in points]
        chain_y = [0.0] + [p.imag for p in points]

        # Arrows: one per segment in the chain
        cmap = cm.plasma
        n_arrows = len(chain_x) - 1
        for i in range(n_arrows):
            color = cmap(i / max(n_arrows - 1, 1))
            x0, y0 = chain_x[i],     chain_y[i]
            x1, y1 = chain_x[i + 1], chain_y[i + 1]
            self.ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color=color,
                    lw=1.8, mutation_scale=14,
                ),
                zorder=3,
            )

        # Dots at each z_k
        for i, p in enumerate(points):
            color = cmap(i / max(n - 1, 1))
            self.ax.plot(p.real, p.imag, 'o', color=color,
                         markersize=6, zorder=5, markeredgewidth=0.5,
                         markeredgecolor='white')
            # Label first six points only to avoid clutter
            if i <= 5:
                sup = '₀₁₂₃₄₅'[i]
                self.ax.annotate(
                    f'z{sup}', xy=(p.real, p.imag),
                    xytext=(5, 5), textcoords='offset points',
                    color=color, fontsize=8, zorder=6,
                )

        # Origin marker
        self.ax.plot(0, 0, 's', color='#7777aa', markersize=5, zorder=4)

        # ── autoscale around all finite points (including origin) ──────────
        all_x = chain_x
        all_y = chain_y
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        span_x = max(x_max - x_min, 0.5)
        span_y = max(y_max - y_min, 0.5)
        # keep aspect close to 1:1 so angles look correct
        span = max(span_x, span_y)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        pad = span * 0.18 + 0.1
        self.ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
        self.ax.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)

        self.fig.tight_layout(pad=1.8)
        self.draw()

        # ── info text ──────────────────────────────────────────────────────
        sup_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        lines = []
        for i, p in enumerate(points):
            sub = str(i).translate(sup_map)
            lines.append(f"z{sub} = {fmt_z(p)}\n    |z{sub}| = {abs(p):.6f}")
        if escaped_at is not None:
            lines.append(
                f"\n⚠  Escaped at iteration {escaped_at}\n"
                f"   (|z| exceeded {ESCAPE_RADIUS:.0e})"
            )
        return '\n'.join(lines)

    def clear_plot(self):
        self.ax.clear()
        self._decorate()
        self._unit_circle()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.fig.tight_layout(pad=1.8)
        self.draw()


# ── main window ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Complex Power Iterator')
        self.resize(1100, 680)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── canvas (left, expands) ─────────────────────────────────────────
        self.canvas = ComplexCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.clear_plot()
        root.addWidget(self.canvas, 3)

        # ── control panel (right, fixed width) ────────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(290)
        col = QVBoxLayout(ctrl)
        col.setContentsMargins(4, 4, 4, 4)
        col.setSpacing(10)

        # Input
        g_input = QGroupBox('Input  z₀ = x + iy')
        lay = QVBoxLayout(g_input)
        hint = QLabel('Enter a complex number, e.g.  0.8 + 0.6i')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: #7777aa; font-size: 8pt;')
        lay.addWidget(hint)
        self.z_input = QLineEdit('0.8 + 0.6i')
        self.z_input.setPlaceholderText('x + iy')
        self.z_input.returnPressed.connect(self._plot)
        lay.addWidget(self.z_input)
        col.addWidget(g_input)

        # Power
        g_power = QGroupBox('Power  p')
        lay = QVBoxLayout(g_power)
        self.power_bg = QButtonGroup()
        for label, val in [('Square   z²', 2), ('Cube     z³', 3), ('4th power  z⁴', 4)]:
            rb = QRadioButton(label)
            rb.setProperty('power_val', val)
            self.power_bg.addButton(rb)
            lay.addWidget(rb)
            if val == 2:
                rb.setChecked(True)
        col.addWidget(g_power)

        # Iterations
        g_iter = QGroupBox('Iterations  n')
        lay = QHBoxLayout(g_iter)
        lay.addWidget(QLabel('Apply power'))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 50)
        self.iter_spin.setValue(8)
        lay.addWidget(self.iter_spin)
        lay.addWidget(QLabel('times'))
        col.addWidget(g_iter)

        # Presets
        g_pre = QGroupBox('Presets')
        pre_lay = QVBoxLayout(g_pre)
        pre_lay.setSpacing(4)
        pre_lay.setContentsMargins(4, 4, 4, 4)

        _preset_data = [
            ('↗  Escape to ∞', '#cc6666', [
                ('Slow Spiral',  '0.9 + 0.44i',     2, 15),
                ('Fast Burst',   '1.5 + 1.0i',       2,  8),
                ('4th Power',    '1.1 + 0.3i',       4, 10),
            ]),
            ('◉  Spiral Patterns', '#6699cc', [
                ('z²  Orbit',    '0.8 + 0.6i',       2, 15),
                ('Heptagon',     '0.6235 + 0.7818i', 2, 10),
                ('17-Star',      '0.9325 + 0.3612i', 2, 12),
            ]),
            ('↙  Converge to 0', '#66cc88', [
                ('Spiral Drain', '0.7 + 0.5i',       2, 20),
                ('Quick Sink',   '0.5 + 0.5i',       2, 15),
                ('Cubic Drop',   '0.8 + 0.3i',       3, 12),
            ]),
        ]

        for group_label, label_color, items in _preset_data:
            lbl = QLabel(group_label)
            lbl.setStyleSheet(
                f'color: {label_color}; font-size: 8pt; font-weight: bold;'
            )
            pre_lay.addWidget(lbl)
            row = QHBoxLayout()
            row.setSpacing(3)
            for name, z_str, pwr, ni in items:
                btn = QPushButton(name)
                btn.setFont(QFont(btn.font().family(), 7))
                btn.setFixedHeight(24)
                btn.clicked.connect(
                    lambda _checked, z=z_str, p=pwr, n=ni:
                        self._apply_preset(z, p, n)
                )
                row.addWidget(btn)
            pre_lay.addLayout(row)

        col.addWidget(g_pre)

        # Buttons
        plot_btn = QPushButton('Plot  ▶')
        plot_btn.clicked.connect(self._plot)
        col.addWidget(plot_btn)

        clear_btn = QPushButton('Clear')
        clear_btn.clicked.connect(self._clear)
        col.addWidget(clear_btn)

        # Values output
        g_vals = QGroupBox('Values')
        lay = QVBoxLayout(g_vals)
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setFont(QFont('Courier New', 8))
        self.info.setMinimumHeight(180)
        lay.addWidget(self.info)
        col.addWidget(g_vals)

        col.addStretch()
        root.addWidget(ctrl)

        self.setStyleSheet(QSS)

    def _get_power(self) -> int:
        for btn in self.power_bg.buttons():
            if btn.isChecked():
                return btn.property('power_val')
        return 2

    def _plot(self):
        try:
            z0 = parse_complex(self.z_input.text())
        except ValueError as exc:
            self.info.setPlainText(f'Parse error:\n{exc}')
            return
        power  = self._get_power()
        n_iter = self.iter_spin.value()
        result = self.canvas.plot_iteration(z0, power, n_iter)
        self.info.setPlainText(result)

    def _apply_preset(self, z_str: str, power: int, n_iter: int):
        self.z_input.setText(z_str)
        for btn in self.power_bg.buttons():
            if btn.property('power_val') == power:
                btn.setChecked(True)
                break
        self.iter_spin.setValue(n_iter)
        self._plot()

    def _clear(self):
        self.canvas.clear_plot()
        self.info.clear()


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
