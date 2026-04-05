# maths_fun

Some fun ways to look at numbers and shapes — interactive visualizations of prime number spirals, generalised spirals, geometric spirographs, Fourier epicycles, Lissajous figures, and a chaotic double pendulum.

## Programs

| File | Description |
|------|-------------|
| `prime_visualizer.py` | Interactive visualizer with controls for spiral type, colour mode, and size |
| `spiral_explorer.py` | Interactive Vogel spiral explorer with adjustable r, θ, dot size, and colour |
| `spiral_artist.py` | Pure spirograph explorer — gradient coloring, keyboard input, high-res PNG export |
| `spiral_duo.py` | Dual-spiral PyQt5 viewer — two independent spirals with live alpha blend, parallel rendering |
| `spirograph.py` | Interactive gear-based spirograph — hypotrochoid/epitrochoid, animated gear overlay, PNG export |
| `fourier_explorer.py` | Fourier series visualizer — epicycle animation, amplitude spectrum, DFT complex-plane view |
| `lissajous_explorer.py` | Lissajous figure explorer — double-pendulum physics, animated trace, damping, phase sweep, PNG export |
| `double_pendulum.py` | Chaotic double pendulum — adjustable lengths, masses, gravity; fading trail; MP4 export |
| `prime_gallery_100.py` | Static three-panel figure showing integers 1–100 across all three spirals |

---

## Setup

### 1. Create the conda environment

```bash
conda create -n maths_fun python=3.11
conda activate maths_fun
```

### 2. Install pip dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy matplotlib pygame PyQt5
```

| Package | Used by |
|---------|---------|
| `numpy` | all programs |
| `matplotlib` | all programs except `spirograph.py` |
| `pygame` | `spirograph.py` |
| `PyQt5` | `spiral_duo.py` |

### 3. Install ffmpeg (for MP4 export)

`double_pendulum.py` can export animations as MP4 files. This requires ffmpeg, which must be installed through conda rather than pip:

```bash
conda install -c conda-forge ffmpeg
```

To verify it is available:

```bash
python -c "from matplotlib.animation import FFMpegWriter; print(FFMpegWriter.isAvailable())"
```

### Running programs

Activate the environment before running any program:

```bash
conda activate maths_fun
python double_pendulum.py
```

Or without activating:

```bash
conda run -n maths_fun python double_pendulum.py
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

### Fourier Explorer — `fourier_explorer.py`

```bash
conda activate maths_fun
python fourier_explorer.py
```

A matplotlib window showing how any closed curve can be expressed as a sum of rotating circles — the Discrete Fourier Transform visualised as epicycles. Three panels update together:

**Left — Epicycle reconstruction**
The curve is traced by a chain of rotating circles. Each circle corresponds to one Fourier frequency component: its radius is the amplitude `|Cₖ|` and it rotates at `k` cycles per period. The faint outline is the target shape; the bright path is what the current set of harmonics draws.

**Top right — Amplitude spectrum `|Cₖ|`**
Bar chart of the DFT coefficient magnitudes, centred at frequency 0, showing ±80 frequency bins. Selected harmonics (included in the reconstruction) are highlighted in blue. An annotation shows how much of the total signal energy the selected terms capture.

**Bottom right — DFT coefficients in the complex plane**
Each Fourier coefficient `Cₖ` is plotted as a dot at `(Re Cₖ, Im Cₖ)`. Selected coefficients are bright; unselected are dim. Spokes to the origin show the phase relationship. This is the "Fourier space" view — amplitude is distance from origin, phase is angle.

**Controls**
- **Harmonics** — how many frequency components to include (1 … 100); terms are added in order of decreasing amplitude
- **Speed** — animation rate (0.1 … 8×)
- **Preset** — target shape: Heart, Square, Star, Triangle, Lissajous, Spirograph, Epitrochoid, Astroid
- **Pause / Play** — freeze the animation
- **Circles: ON/OFF** — show or hide the rotating-circle overlay
- **Reset** — restart the trace from the beginning

**Interesting things to try:**
- Load *Spirograph* and reduce harmonics to 1 — it is exactly a 2-term Fourier series, so even 2 terms reconstruct it perfectly
- Load *Square* and watch how adding more harmonics sharpens the corners (Gibbs phenomenon)
- Load *Heart* — only 8 significant frequencies are needed for a perfect reconstruction
- Toggle circles off and increase Speed to see the final drawn shape clearly

---

### Lissajous Explorer — `lissajous_explorer.py`

```bash
conda activate maths_fun
python lissajous_explorer.py
```

A matplotlib window simulating the classic **double-pendulum Lissajous drawing machine** — two pendulums swinging at right angles, one controlling horizontal motion and one vertical, with a pen tracing the combined path.

**Equations**

```
x(t) = Ax · exp(−γt/T) · sin(ωx·t + δ)
y(t) = Ay · exp(−γt/T) · sin(ωy·t)
```

When the frequency ratio ωx:ωy is a ratio of small integers the curve closes on itself. Non-integer ratios produce slowly rotating, never-closing paths.

**Pendulum Frequencies**
- **ωx / ωy** — frequency of each pendulum; integer ratios (1:2, 2:3, 3:4 …) produce closed Lissajous curves

**Phase & Amplitude**
- **δ (°)** — phase offset between the two pendulums; rotates and reshapes the figure
- **Ax / Ay** — swing amplitude of each pendulum

**Pendulum Physics**
- **Damping γ** — 0 = ideal frictionless pendulums (closed curve forever); increase to simulate friction — the trace spirals inward and eventually collapses to the origin
- **Cycles** — how many periods of the longer pendulum to trace
- **N pts** — number of trace points (more = smoother, slower to render)

**Presets** — eight classic ratio configurations in a 4 × 2 grid:

| Preset | ωx:ωy | Shape |
|--------|--------|-------|
| 1:1 | 1:1 | circle / ellipse |
| 1:2 | 1:2 | figure-eight |
| 1:3 | 1:3 | three-lobed curve |
| 2:3 | 2:3 | three-lobed, denser |
| 3:4 | 3:4 | four-lobed |
| 3:5 | 3:5 | five-lobed |
| 4:5 | 4:5 | five-lobed, denser |
| 5:6 | 5:6 | six-lobed |

**Visual style**
- **Line w / Alpha** — stroke weight and transparency of the gradient trace
- **Palette** — 8 gradient colormaps; the colour maps elapsed time from start (t=0) to end of trace

**Animate Trace** — the pen starts at t=0 and draws the path forward in real time, with a white dot marking the current pen position. When the trace is complete it loops back to the start.
- **Speed** — points drawn per frame (1–200); drag this slider while the animation is running to change its pace without interrupting it

**Export PNG** — saves a clean image (no axes or controls) to the current directory. Five resolution options:

| Button | Resolution |
|--------|-----------|
| 1080p | 1920 × 1080 |
| 1440p | 2560 × 1440 |
| 4K | 3840 × 2160 |
| Sq 2K | 2160 × 2160 |
| Sq 4K | 4096 × 4096 |

**Interesting things to try:**
- Set any preset, then drag **δ** slowly from 0° to 90° — watch the figure rotate from a horizontal line through an ellipse to the classic closed curve
- Set **2:3** and enable **Animate Trace** at low speed — count the lobes as they form
- Add a small **Damping** value (0.3–1.0) with many **Cycles** to see how a real pendulum machine would slowly decay inward
- Try a near-integer ratio (e.g. ωx=3, ωy≈2 by setting ωy=2 and ωx=3 with a non-90° phase) and many cycles — the figure slowly precesses

---

### Double Pendulum — `double_pendulum.py`

```bash
conda activate maths_fun
python double_pendulum.py
```

A real-time simulation of the double pendulum — two rigid arms connected end-to-end, governed by the full Lagrangian equations of motion. Small-angle configurations oscillate regularly; larger angles produce sensitive dependence on initial conditions and chaotic motion. The fading trail shows the recent path of the lower bob.

**Equations of motion**

```
Δ  = θ1 − θ2
D  = 2m1 + m2 − m2·cos(2Δ)
α1 = [−g(2m1+m2)sin θ1 − m2·g·sin(θ1−2θ2) − 2·sin Δ·m2·(ω2²L2 + ω1²L1·cos Δ)] / L1D
α2 = [2·sin Δ·(ω1²L1(m1+m2) + g(m1+m2)cos θ1 + ω2²L2·m2·cos Δ)] / L2D
```

Integrated with a fixed-step RK4 solver at 500 Hz.

**Initial conditions**
- **θ1 / θ2** — starting angle of each arm from vertical (−180° to 180°)
- **ω1 / ω2** — starting angular velocity in rad/s

**Pendulum**
- **L1 / L2** — arm lengths in metres; bob size on screen scales with mass
- **m1 / m2** — bob masses in kg

**Environment**
- **g** — gravitational acceleration (0 to 25 m/s²; set to 0 for weightless drift)
- **Trail pts** — number of past positions shown in the fading trail (live, does not reset)
- **Sim speed** — time multiplier (live); 0.1× for slow motion, 8× for fast-forward

**Presets** — 4 × 2 grid:

| Preset | Behaviour |
|--------|-----------|
| Small ✓ | θ1=20°, θ2=10° — small angle, near-harmonic, stable |
| Sym ✓ | θ1=45°, θ2=−45° — symmetric mode, regular |
| Chaotic 1 | θ1=120°, θ2=60° — classic chaotic motion |
| Chaotic 2 | θ1=150°, θ2=30° — complex overlapping loops |
| Butterfly | θ1=170°, θ2=10° — highly sensitive to initial conditions |
| Near Top | θ1=179°, θ2=0° — upper arm nearly inverted, immediately chaotic |
| Zero-G | g=0.5 m/s² — slow, dreamlike, weightless drift |
| Unequal | L1=1.5, L2=0.5, m1=2.0, m2=0.3 — mismatched arm/mass ratio |

**Pause / Reset** — freeze the simulation or restart from the current slider values.

**Export MP4** — pre-computes the full trajectory starting from the *current live state* (let it run to something interesting, then export), renders a clean 8 × 8 inch video, and saves it to the current directory. Three duration options:

| Button | Duration |
|--------|---------|
| 10 s | short clip |
| 30 s | half-minute |
| 60 s | one minute |

Requires ffmpeg — see Setup above.

**Interesting things to try:**
- Apply **Small ✓**, watch the regular oscillation, then nudge **θ1** to 90° — chaos takes over immediately
- Apply **Butterfly** and reset several times with tiny changes to **θ1** — the long-term paths diverge completely (sensitive dependence)
- Set **g = 0** to watch the pendulum drift with no restoring force
- Let any chaotic preset run for 30 seconds to fill the trail, then export the MP4

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
