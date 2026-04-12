# maths_fun

Some fun ways to look at numbers and shapes — interactive visualizations of prime number spirals, generalised spirals, geometric spirographs, Fourier epicycles, Lissajous figures, complex power iteration, a chaotic double pendulum with a chaos map explorer, and an interactive 3D three-body gravitational simulator.

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
| `pendulum_chaos_map.py` | Double pendulum chaos map — 2D heatmap of stability vs chaos across initial angle space; three selectable metrics; click-to-preview; PNG/pkl export |
| `3_body_problem.py` | Interactive 3D three-body gravitational simulator — 7 presets, real-time collision handling, MP4 export |
| `three_body_physics.py` | Physics engine for `3_body_problem.py` — ODE integration, collision detection, body colour/size helpers |
| `complex_power_iterator.py` | Complex power iterator — iterate z→z^p on the complex plane with chained-vector visualisation; 9 presets across escape/spiral/converge behaviours |
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
pip install numpy matplotlib pygame PyQt5 scipy mpmath
```

| Package | Used by |
|---------|---------|
| `numpy` | all programs |
| `matplotlib` | all programs except `spirograph.py` |
| `scipy` | `3_body_problem.py` (ODE integration) |
| `mpmath` | `3_body_problem.py` (high-precision Figure-8 initial conditions) |
| `pygame` | `spirograph.py` |
| `PyQt5` | `spiral_duo.py`, `3_body_problem.py`, `pendulum_chaos_map.py`, `complex_power_iterator.py` |

### 3. Install ffmpeg (for MP4 export)

`double_pendulum.py` and `3_body_problem.py` can export animations as MP4 files. This requires ffmpeg, which must be installed through conda rather than pip:

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

### Double Pendulum Chaos Map — `pendulum_chaos_map.py`

```bash
conda activate maths_fun
python pendulum_chaos_map.py
```

A PyQt5 application that maps out the stability landscape of the double pendulum across all possible starting angles. The window shows a 2D heatmap where the x-axis is the initial angle θ1 (first arm) and the y-axis is θ2 (second arm). Every pendulum is released from rest — the colour of each cell encodes how chaotically it behaves.

The entire grid of pendulums (up to 500 × 500 = 250,000 simultaneous simulations) is integrated at once using a vectorised numpy RK4 solver. Computation runs in a background thread so the UI stays responsive.

**Chaos metrics**

A dropdown at the top of the control panel selects which metric to display. All three are computed simultaneously during a single run — switching between them is instant, no recomputation needed.

| Metric | What it measures |
|--------|-----------------|
| **Flips (θ₂)** | Number of times the outer bob goes "over the top" — i.e. θ₂ crosses ±π. A pendulum that swings gently scores 0; one that whips around the pivot accumulates many flips. This is the default and the most intuitive chaos indicator. |
| **Flips (θ₁+θ₂)** | Same as above but counting full rotations of *both* bobs combined. Reveals chaos in the inner arm that θ₂-only misses. |
| **Peak \|ω₂\|** | Maximum angular velocity (rad/s) reached by the outer bob during the entire simulation. Chaotic trajectories develop very high peak velocities; stable oscillations stay low. This metric produces a smooth, continuous colour gradient rather than discrete integer counts. |

Flip detection uses wrapped-angle analysis: the raw angle is wrapped to [−π, π] each timestep, and a flip is counted when the wrapped angle jumps by more than π — meaning the bob physically crossed the vertical-up position. This correctly gives 0 flips for small-angle oscillations (where earlier methods using `floor(θ/π)` would false-positive on every zero-crossing).

**Angular Range**
- **θ1 min / max** — horizontal extent of the map (default −180° to 180°)
- **θ2 min / max** — vertical extent of the map (default −180° to 180°)

Narrow the range and increase resolution to zoom into interesting regions of the fractal boundary.

**Grid Resolution**
- **Coarse 50** / **Medium 100** / **Fine 200** — preset buttons for common grid sizes
- **N × N** — custom grid size (10 to 500). Higher values show finer fractal structure but take longer to compute

Estimated compute times (t = 20 s, default physics):

| Grid | Pendulums | Time |
|------|-----------|------|
| 50 × 50 | 2,500 | ~1–3 s |
| 100 × 100 | 10,000 | ~30 s |
| 200 × 200 | 40,000 | ~2 min |

**Simulation**
- **Duration (s)** — how long each pendulum is simulated (5–60 s, default 20 s). Longer durations allow more flips to accumulate, revealing finer detail in the chaotic regions

**Physics**
- **m1 / m2** — bob masses in kg (0.1 to 5.0)
- **g** — gravitational acceleration (0.1 to 25.0 m/s²)
- Arm lengths are fixed at L1 = L2 = 1.0 m (equal arms, as required for the standard chaos map)

**Display**
- **Colormap** — choose from inferno, viridis, plasma, magma, hot, turbo, twilight. Changing the colormap is instant — no recomputation needed
- **Log scale** — applies `log(1 + flips)` to the data, compressing the dynamic range and revealing fine structure at the boundary between stable and chaotic regions that is invisible at linear scale

**Actions**
- **▶ Compute** — starts the simulation in a background thread. A progress bar shows completion. Results are auto-saved to a `.pkl` file in the current directory
- **Export PNG** — saves a high-resolution (300 DPI, 12 × 10 inch) image with the current colormap and axis labels. Filename encodes grid size, duration, colormap, and timestamp
- **Load .pkl** — opens a file dialog to reload any previously saved result. The heatmap, axis ranges, and physics parameters are restored instantly

**Click to preview**

Left-click anywhere on the heatmap to open a **live animated pendulum** in a new window at those exact (θ1, θ2) initial conditions. The preview window shows:
- Animated rods, bobs (blue/orange), and pivot
- Fading trail (hot colormap, 600 points) tracing the lower bob's path
- Live energy readout (t, KE, PE, total E)
- Pause / Resume and Reset buttons

Multiple preview windows can be open simultaneously — each runs independently. This lets you visually compare a stable configuration with a chaotic one side by side.

**Hover tooltip**

Move the mouse over the heatmap to see the exact θ1, θ2, and the selected metric's value at the cursor position in the status label.

**Interesting things to try:**
- Start with **Coarse 50** for a quick overview, then switch to **Fine 200** to see the fractal boundary in detail
- Toggle **Log scale** on — the boundary between stable and chaotic regions has remarkably intricate fractal structure that only becomes visible with logarithmic scaling
- Click a point in the dark (stable) region near the centre, then click a point in the bright (chaotic) region near the edges — watch the dramatically different pendulum behaviour in the preview windows
- Narrow the angular range to a small region around the stable/chaotic boundary (e.g. θ1: 80° to 130°, θ2: 20° to 70°) and compute at Fine resolution — the fractal self-similarity becomes apparent
- Reduce **g** to 1–2 m/s² and recompute — the chaotic region shrinks as gravity weakens
- Set **m1 = 5.0, m2 = 0.1** — the mass asymmetry changes the shape of the stable regions

---

### Three-body problem — `3_body_problem.py`

```bash
conda activate maths_fun
python 3_body_problem.py
```

A real-time 3D simulation of three mutually attracting bodies under Newtonian gravity. The window is split into two independent panels:

- **Left — interactive 3D canvas**: rotate the view with the mouse; all axes are labelled in Astronomical Units (AU). A live clock in the top-left corner shows elapsed simulation time in years. The camera tracks the bodies automatically — zooming in when they are close, zooming out smoothly as they diverge.
- **Right — native Qt control panel**: all controls are standard OS widgets (sliders, spinboxes, checkboxes, dropdowns) that operate independently of the matplotlib canvas so the UI never blocks during rendering or integration.

The trajectory is computed by scipy's DOP853 integrator (8th-order Dormand-Prince) in a background thread; the canvas continues to animate and accept input while the solver runs.

**Physics units**

| Quantity | Unit |
|----------|------|
| Distance | AU (Astronomical Unit) |
| Mass | M☉ (Solar mass) |
| Time | sim-yr; 1 sim-yr = 1/(2π) year ≈ 58.1 days |
| Gravity | G = 1 (natural N-body units) |

At these units a circular orbit at 1 AU around 1 M☉ has period 2π sim-yr = 1 calendar year, consistent with Kepler's third law.

**Colour and size by mass**

| Mass range | Appearance | Classification |
|------------|------------|----------------|
| < 0.08 M☉ | Deep orange | Gas giant |
| 0.08 – 0.3 M☉ | Orange-red | Brown/red dwarf |
| 0.3 – 1.0 M☉ | Red-orange | M-dwarf |
| 1.0 – 1.4 M☉ | Yellow | G-type (Sun-like) |
| 1.4 – 2.0 M☉ | Yellow-white | F-type |
| 2.0 – 8.0 M☉ | White | A-type |
| 8.0 – 20 M☉ | Blue-white | B-type |
| > 20 M☉ | Bright blue | O-type |

Marker area scales as mass^(1/3), clamped to a visible range. Each body is drawn with three scatter layers (core, inner glow, outer halo) and six fading trail segments.

**Presets** — colour-coded by stability (green = stable, amber = semi-stable, red = chaotic):

| Preset | Notes |
|--------|-------|
| Figure-8 | Three equal masses chasing each other on a figure-eight; initial conditions computed with mpmath at 50-digit precision |
| Lagrange △ | Three equal masses at the vertices of an equilateral triangle — the Lagrange L4/L5 solution |
| Hierarchical | A tight binary with a distant third body — semi-stable until perturbation builds up |
| Sun-Jupiter-Saturn | Realistic mass ratios from the Solar System; long-term quasi-periodic |
| Pythagorean | Masses 3, 4, 5 starting in a right-triangle arrangement — strongly chaotic with ejection |
| Fig-8 Perturbed | Figure-8 initial conditions with a small velocity kick — slowly breaks apart |
| Random Chaotic | Randomised masses and positions; behaviour varies per run |

**Bodies**
- **Mass 1 / 2 / 3** — override each body's mass in M☉; the body's colour and glow update live. Applied at the next **▶ Integrate**.

**Configuration**
- **Separation Scale** — multiplies all initial body positions by this factor; velocities are adjusted as 1/√scale to keep the system bound (Kepler scaling)
- **Tilt °** — rotates the orbital plane around the x-axis for a better 3D viewing angle
- **Collision Radius (AU)** — bodies merge if their separation falls below this distance
- **Collision Detection** (checkbox) — when ticked, scipy event detection stops integration at each collision, merges bodies conserving momentum, and continues. When unticked, bodies pass through each other

**Integration**
- **Duration** — total simulated time in sim-yr
- **ODE Method** — DOP853 (8th-order, more accurate) or RK45 (faster)
- **Tolerance 1e-N** — relative tolerance for the ODE solver (higher = more accurate, slower)
- **▶ Integrate** — re-runs the simulation in a background thread; "Computing…" is shown in the canvas while it runs

**Visual**
- **Trail %** — fraction of the trajectory shown as a fading trail behind each body
- **Speed** — animation playback rate (frames advance per timer tick)
- **Show Force Vectors** (checkbox) — draws a colour-matched arrow on each body showing the net gravitational force it is experiencing at that instant. Arrow lengths are scaled so the largest force vector is 25% of the current view radius, keeping them visible at any zoom level

**Export MP4** — renders up to 1200 frames at 1920 × 1080 @ 120 DPI, 30 fps, matching the current view angle and axis limits. Runs in a background thread; progress shown in the canvas. Requires ffmpeg — see Setup above.

**Live camera**
The 3D view updates every frame using an asymmetric exponential moving average:
- **Zoom out** (bodies diverging): fast response (α = 0.08) so bodies never leave the screen
- **Zoom in** (bodies re-approaching): slow ease-in (α = 0.02) to avoid jarring snaps
- **Outlier rejection**: if one body has escaped and is 3× farther than the next-farthest, the camera focuses on the remaining cluster; the escaped body slides out of frame without dominating the scale

**Interesting things to try:**
- Load **Figure-8**, watch the perfect choreography, then switch to **Fig-8 Perturbed** to see how quickly it breaks down
- Load **Pythagorean** with **Collision Detection** on — bodies at masses 3, 4, 5 merge in sequence until one massive object remains
- Load **Lagrange △**, increase **Separation Scale** to 2–3, and watch numerical drift slowly destabilise the equilateral arrangement
- Load any preset, tilt the orbit 45–60°, rotate the 3D view with the mouse to find the best angle, then export an MP4
- Set **Sun-Jupiter-Saturn** and run for Duration = 500 sim-yr to see the long-term quasi-periodic structure

---

### Complex Power Iterator — `complex_power_iterator.py`

```bash
conda activate maths_fun
python complex_power_iterator.py
```

A PyQt5 application that visualises the iteration z → z^p on the complex plane. Enter a starting complex number z₀, choose a power (2, 3, or 4), and watch the sequence z₀, z₁ = z₀^p, z₂ = z₁^p, … drawn as a chain of colour-coded vectors originating from the origin.

**How it works**

Each iteration raises the current value to the chosen power. On the complex plane this has a geometric interpretation:
- **Magnitude**: |z_{n+1}| = |z_n|^p — if |z₀| < 1 the magnitude shrinks to zero; if |z₀| > 1 it grows to infinity; if |z₀| = 1 exactly it stays on the unit circle
- **Angle**: arg(z_{n+1}) = p × arg(z_n) — the angle multiplies by p each step, so points on the unit circle trace out orbits determined by number theory (the order of p modulo the denominator of the starting angle as a fraction of 2π)

**Controls**
- **z₀** — input field for the starting complex number (e.g. `0.8 + 0.6i`, `1.5 - i`, `0.3i`)
- **Power p** — radio buttons for z², z³, z⁴
- **Iterations n** — how many times to apply the power (1–50)
- **Plot ▶** — compute and draw the iteration chain
- **Clear** — reset the canvas

**Display**
- A dashed **unit circle** marks |z| = 1 — the boundary between convergence and escape
- Colour-coded **arrows** show each step of the iteration (plasma colormap)
- **Dots** at each z_k with subscript labels for the first six points
- The **Values** panel lists all computed z_k with their magnitudes
- If |z| exceeds 10⁶, iteration stops and an escape warning is shown

**Presets** — nine presets in three groups:

| Group | Preset | z₀ | p | n | Behaviour |
|-------|--------|----|---|---|-----------|
| **↗ Escape** | Slow Spiral | 0.9 + 0.44i | 2 | 15 | \|z₀\| ≈ 1.002 — spirals outward, escapes ~step 12 |
| | Fast Burst | 1.5 + 1.0i | 2 | 8 | Clearly outside unit circle, escapes in ~5 steps |
| | 4th Power | 1.1 + 0.3i | 4 | 10 | Higher power accelerates escape |
| **◉ Spiral** | z² Orbit | 0.8 + 0.6i | 2 | 15 | Exactly \|z\| = 1 — angle-doubling map on the unit circle |
| | Heptagon | 0.6235 + 0.7818i | 2 | 10 | 7th root of unity; 2³ ≡ 1 (mod 7), visits 3 points forming a triangle |
| | 17-Star | 0.9325 + 0.3612i | 2 | 12 | 17th root of unity; 2⁸ ≡ 1 (mod 17), traces an 8-pointed star |
| **↙ Converge** | Spiral Drain | 0.7 + 0.5i | 2 | 20 | \|z₀\| ≈ 0.86 — graceful inward spiral to zero |
| | Quick Sink | 0.5 + 0.5i | 2 | 15 | \|z₀\| ≈ 0.71 — faster collapse |
| | Cubic Drop | 0.8 + 0.3i | 3 | 12 | Cubic power accelerates convergence |

**Interesting things to try:**
- Click **17-Star** — the 8-pointed pattern arises because the order of 2 modulo 17 is 8, so squaring cycles through exactly 8 of the 17th roots of unity
- Click **z² Orbit** — since 0.8 + 0.6i has |z| = 1 exactly (3-4-5 triangle), the iterates stay on the unit circle forever and the angle-doubling map produces a dense, ergodic orbit
- Click **Slow Spiral** — the starting point is barely outside the unit circle (|z| ≈ 1.002), so the iterates hug the circle for many steps before finally escaping
- Try z₀ = `i` with p = 2 — the sequence is i → −1 → 1 → 1 → 1 … (a fixed point)
- Try z₀ = `-1 + 0i` with p = 3 — the sequence is −1 → −1 → −1 … (another fixed point)

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
