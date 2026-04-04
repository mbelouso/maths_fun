# maths_fun

Some fun ways to look at numbers and shapes — interactive visualizations of prime number spirals, generalised spirals, and geometric spirographs.

## Programs

| File | Description |
|------|-------------|
| `prime_visualizer.py` | Interactive visualizer with controls for spiral type, colour mode, and size |
| `spiral_explorer.py` | Interactive Vogel spiral explorer with adjustable r, θ, dot size, and colour |
| `spiral_artist.py` | Pure spirograph explorer — gradient coloring, keyboard input, high-res PNG export |
| `spiral_duo.py` | Dual-spiral PyQt5 viewer — two independent spirals with live alpha blend, parallel rendering |
| `spirograph.py` | Interactive gear-based spirograph — hypotrochoid/epitrochoid, animated gear overlay, PNG export |
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
pip install numpy matplotlib pygame PyQt5
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

### Spiral artist — `spiral_artist.py`

```bash
conda activate maths_fun
python spiral_artist.py
```

Pure spirograph explorer built on the same formula (`r = k^p`, `θ = k × angle°`) but focused entirely on aesthetics. No prime-number content — every point is coloured by its position in the sequence using a gradient.

**Spiral formula**
- **N** — number of points (up to 100,000)
- **k ^ p** — r exponent; 0.5 = slow outward growth, 1.0 = linear, 2.0 = fast
- **θ (°)** — angle step per integer; golden angle ≈ 137.508° gives even coverage, round numbers create symmetric arms

Each slider has a **text box** on its right — type an exact value and press Enter to jump precisely to it.

**Presets**
- **Sunflower ★** — golden angle, p = 0.5
- **Alt. golden** — 360° − golden angle, p = 0.5 (opposite winding)
- **Pentagon** — 144°, p = 0.5 (5-arm star pattern)
- **Galaxy** — golden angle, p = 1.0 (stretched radially)

**Visual style**
- **Line w / Alpha** — width and transparency of the gradient connecting line
- **Dot sz** — size of individual point markers
- **Dots / Line / Both** — show dots only, the gradient line only, or both

**Palette** — choose the gradient colormap: plasma, viridis, inferno, rainbow, cool, twilight. **Reverse** flips the direction of the gradient along the sequence.

**Export**
- Select output DPI: 150 (screen), 300 (print), 600 (high detail)
- Click **Export PNG** to save a clean `12 × 12 inch` image (no axes, no widgets) to the current directory. The filename encodes the current parameters.

---

### Dual spiral viewer — `spiral_duo.py`

```bash
conda activate maths_fun
python spiral_duo.py
```

Two fully independent spirals rendered simultaneously and alpha-blended together in a PyQt5 window. Both spirals are drawn in parallel using a `ThreadPoolExecutor`, keeping the UI responsive while rendering.

**Blend slider** — cross-fades between the two spirals:
- 0 = only Spiral 1 visible
- 0.5 = both at equal weight
- 1 = only Spiral 2 visible

**Spiral 1 / Spiral 2 tabs** — each spiral has its own independent set of controls:
- **N, p, θ** — formula parameters (slider + spinbox for precise entry)
- **Presets** — Sunflower ★, Alt. golden, Pentagon, Galaxy
- **Line width, Alpha, Dot size** — visual style
- **Dots / Line / Both** — display mode
- **Palette** — 10 gradient colormaps + reverse toggle

**Background** — Dark or Light.

**Export PNG** — renders a clean 12 × 12 inch image at the chosen DPI to the current directory. Rendering runs off the main thread so the UI stays live.

Default startup: Spiral 1 at golden angle (plasma), Spiral 2 at alternate golden angle (viridis), blend at 0.5 — two complementary sunflowers overlaid.

---

### Interactive spirograph — `spirograph.py`

```bash
conda activate maths_fun
python spirograph.py
```

A pygame window with a dark canvas on the left and a control panel on the right. The curve is redrawn live as you adjust any parameter.

**Gear formula**
- **Hypotrochoid** — inner gear (radius r) rolling inside a fixed outer ring (radius R), with the pen at distance d from the inner gear's centre:
  `x = (R−r)cos t + d·cos((R−r)/r · t)`, `y = (R−r)sin t − d·sin((R−r)/r · t)`
- **Epitrochoid** — inner gear rolling *outside* the fixed gear:
  `x = (R+r)cos t − d·cos((R+r)/r · t)`, `y = (R+r)sin t − d·sin((R+r)/r · t)`

**Sliders**
- **Outer ring radius R** — size of the fixed gear (40–300)
- **Rolling gear radius r** — size of the moving gear (5–200)
- **Pen offset d** — distance of the pen from the rolling gear's centre (0–300)
- **Revolutions** — how many times the inner gear completes a full loop (1–40)
- **Line width** — stroke thickness (1–6)
- **Resolution (pts)** — number of segments computed (500–10,000)
- **Anim speed (pts/frame)** — how many segments are drawn per frame in animate mode (1–300); drag left to watch the gear trace slowly

**Colourmap** — the curve is coloured end-to-end through the selected matplotlib colormap: viridis, plasma, magma, inferno, cividis, twilight, cool, rainbow.

**Mode** — toggle between Hypotrochoid and Epitrochoid.

**Show Gears** — overlays the animated rolling gear and pen arm on the canvas.

**Animate** — draws the curve incrementally frame by frame, looping continuously. Combine with Show Gears to watch the pen trace the pattern.

**Presets**
- **Classic** — R=160, r=40, d=80, 4 revolutions
- **Star** — R=175, r=25, d=100, 7 revolutions
- **Flower** — R=140, r=20, d=90, 3 revolutions (epitrochoid)
- **Orbit** — R=160, r=80, d=40, 5 revolutions

**Export PNG** — renders a fresh off-screen image at the chosen resolution (line width scaled proportionally) and saves it to the current directory. The filename encodes all current parameters and a timestamp.

| Button | Resolution |
|--------|-----------|
| 1080p | 1920 × 1080 |
| 1440p | 2560 × 1440 |
| 4K | 3840 × 2160 |
| Sq 2K | 2160 × 2160 |
| Sq 4K | 4096 × 4096 |

**Keyboard shortcuts**: `R` redraw · `C` clear · `A` toggle animate · `Esc` quit

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
