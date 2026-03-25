# maths_fun

Some fun ways to look at numbers — interactive and static visualizations of prime numbers arranged in spiral patterns.

## Programs

| File | Description |
|------|-------------|
| `prime_visualizer.py` | Interactive visualizer with controls for spiral type, colour mode, and size |
| `spiral_explorer.py` | Interactive Vogel spiral explorer with adjustable r, θ, dot size, and colour |
| `prime_gallery_100.py` | Static three-panel figure showing integers 1–100 across all three spirals |

---

## Setup

### 1. Create the conda environment

```bash
conda create -n maths_fun python=3.11
conda activate maths_fun
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy matplotlib
```

---

## Usage

### Interactive visualizer — `prime_visualizer.py`

```bash
conda activate maths_fun
python prime_visualizer.py
```

A matplotlib window opens with three control panels on the right:

**Spiral type**
- **Ulam** — integers laid out in a rectangular outward spiral on a grid
- **Sacks** — Robert Sacks' Archimedean spiral: integer k placed at angle 2π√k, radius √k
- **Vogel** — sunflower / golden-angle spiral: integer k placed at angle k × 137.5°, radius √k

**Colour mode**
- **Primes** — primes in gold, composites dim
- **Mod 6** — colour by remainder mod 6 (reveals that all primes > 3 fall on two residue classes)
- **Mod 30** — colour by remainder mod 30
- **Twin primes** — twin prime pairs highlighted in pink, other primes in gold

**Size slider** — adjusts the number of integers displayed (up to ~1 million)

**Reset zoom** — returns the view to fit all points

**Zoom and pan** with standard matplotlib toolbar controls. At high zoom on the Ulam spiral, individual numbers appear as labels inside their cells.

---

### Spiral explorer — `spiral_explorer.py`

```bash
conda activate maths_fun
python spiral_explorer.py
```

Explore generalised spirals of the form `r = k^p`, `θ = k × angle°` interactively.
At `p = 0.5` and `angle ≈ 137.508°` you recover the classic Vogel sunflower.

**Spiral parameters**
- **N points** — number of integers to plot (up to 100,000)
- **r = k ^ p** — controls how quickly the spiral expands (0.1 = tight, 3.0 = very open)
- **θ step (°)** — angle between consecutive integers; try values near 137.5° for sunflower patterns, or round numbers (60°, 90°, 120°) for symmetric arms
- **Sunflower ★** — preset to golden angle (≈ 137.508°), `p = 0.5`
- **Alt. golden** — preset to 360° − golden angle (≈ 222.492°), same distribution with opposite winding

**Visual style**
- **Dot size** — size of each plotted point
- **Line width / Line alpha** — thickness and transparency of the connecting sequence line
- **Line: OFF/ON** — toggle the line connecting consecutive integers

**Colour mode**
- **Primes** — primes in gold, composites dim
- **Twin primes** — twin prime pairs in pink, other primes in gold
- **Gradient** — colour by position in the sequence (plasma colormap)
- **Mod 6** — colour by remainder mod 6
- **Mod 12** — colour by remainder mod 12

---

### Static gallery — `prime_gallery_100.py`

```bash
conda activate maths_fun
python prime_gallery_100.py
```

Generates a three-panel figure of integers 1–100 arranged in the Ulam, Sacks, and Vogel spirals:
- Every number is labelled
- Primes are bold gold; composites are dim white
- A blue line traces the sequence 1 → 2 → … → 100

The figure is saved to `prime_spirals_1_100.png` in the current directory and also displayed on screen.

---

## Spiral reference

| Spiral | Formula | Pattern visible |
|--------|---------|-----------------|
| Ulam | Grid walk starting at 1 in the centre | Diagonal and straight lines of primes |
| Sacks | r = √k, θ = 2π√k | Perfect squares align on the positive x-axis; primes cluster on radial arms |
| Vogel | r = √k, θ = k × 137.508° | Sunflower-like arrangement; no strong prime clustering |
