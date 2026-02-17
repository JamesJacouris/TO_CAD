# Post-Processing Algorithms Visualization — Usage Guide

Two beautiful diagram versions have been created to illustrate the four sequential post-processing algorithms.

---

## Version 1: Detailed Full-Page Diagram (RECOMMENDED)

**File:** `postprocessing_diagram.tex`

**Best for:** Dedicated section on post-processing in your "Stage 1: Voxel Reconstruction" chapter

**Features:**
- Shows complete transformation from raw graph → final optimized graph
- Visual progression through all 4 algorithms
- Shows both EDT and Uniform radius estimation modes side-by-side
- Includes legend explaining node types and line styles
- Large, detailed, easy to understand
- Color-coded algorithms (orange → blue → green → purple)

**Size:** Approximately 16cm × 12cm (landscape orientation)

**How to use:**
1. Compile independently: `pdflatex postprocessing_diagram.tex`
2. OR copy the entire TikZ code into your document after your text description
3. Make sure preamble includes:
   ```latex
   \usepackage{tikz}
   \usetikzlibrary{shapes, arrows, positioning, calc, decorations.pathreplacing, bending}
   ```

**Caption suggestion:**
```latex
\caption{Four sequential post-processing algorithms applied to the medial-axis
skeleton. (Algorithm~1)~Edge Collapse merges nodes connected by short edges
$d_{\mathrm{collapse}}$, with BC-tagged nodes (purple) acting as anchors.
(Algorithm~2)~Branch Pruning iteratively removes degree-1 leaf branches
shorter than $l_{\mathrm{prune}}$. (Algorithm~3)~Ramer--Douglas--Peucker
simplification reduces intermediate polyline points to tolerance $\varepsilon_{\mathrm{rdp}}$.
(Algorithm~4)~Radius Estimation assigns cross-sectional radii either via
local EDT geometry ($r_v = \mathrm{EDT}(v) \cdot h$, variable thickness) or
uniform volume conservation ($r = \sqrt{V_{\mathrm{target}}/(\pi\sum L_i)}$,
constant thickness).}
\label{fig:postprocessing_full}
```

---

## Version 2: Compact Inline Diagram

**File:** `postprocessing_compact.tex`

**Best for:** Fitting naturally within text flow, smaller page constraints

**Features:**
- Four algorithm boxes in horizontal sequence
- Simplified graph representations
- Shows both radius estimation modes inline
- Smaller footprint
- Easy to read parameter definitions
- Fully self-contained with figure environment

**Size:** Approximately 12cm × 8cm

**How to use:**

**Option A - Direct copy-paste (easiest):**
1. Copy the entire content from `postprocessing_compact.tex`
2. Paste directly into your LaTeX document where you want the figure
3. The figure environment is included

**Option B - Via LaTeX input command:**
```latex
\input{NOTES/postprocessing_compact.tex}
```

**Required preamble:**
```latex
\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning}
```

---

## Choosing Between Versions

| Aspect | Full Version | Compact Version |
|--------|-------------|-----------------|
| **Detail level** | Very detailed, annotated | Clean, concise |
| **Best for** | Dedicated algorithm section | Embedded in text |
| **Page width** | 16cm (landscape) | 12cm (portrait) |
| **Legend** | Included | Minimal (in caption) |
| **Node count** | Many (realistic) | Few (schematic) |
| **Integration** | Standalone page | Fits in column |
| **Time to compile** | 3-5 seconds | 2-3 seconds |

**Recommendation:** Use **Compact Version** in your main document flow, then **Full Version** on a dedicated figure page or appendix if you want detailed technical exposition.

---

## Integration into Your Dissertation

### Section Structure

```latex
\subsection{Post-Processing (Algorithms 4.4-4.7)}

Four sequential algorithms clean and simplify the raw skeletal graph:

\begin{enumerate}
    \item \textbf{Edge Collapse} (Algorithm 4.4): ...
    \item \textbf{Branch Pruning} (Algorithm 4.5): ...
    \item \textbf{RDP Simplification} (Algorithm 4.6): ...
    \item \textbf{Radius Estimation} (Algorithm 4.7): ...
\end{enumerate}

% INSERT FIGURE HERE
\input{NOTES/postprocessing_compact.tex}

% Continue with detailed explanation...
```

### Alternative: Place After Detailed Text

If you prefer detailed text first, then figure:

```latex
\subsubsection{Algorithm 4.4: Edge Collapse}
...detailed explanation...

\subsubsection{Algorithm 4.5: Branch Pruning}
...detailed explanation...

\subsubsection{Algorithm 4.6: RDP Simplification}
...detailed explanation...

\subsubsection{Algorithm 4.7: Radius Estimation}
...detailed explanation...

% THEN insert figure as visual summary
\input{NOTES/postprocessing_compact.tex}
```

---

## Customization Guide

### Change Colors

Locate color definitions in the TikZ preamble:

```latex
\definecolor{algo1}{RGB}{230, 126, 34}     % Orange (Edge Collapse)
\definecolor{algo2}{RGB}{41, 128, 185}     % Blue (Branch Pruning)
\definecolor{algo3}{RGB}{39, 174, 96}      % Green (RDP)
\definecolor{algo4}{RGB}{155, 89, 182}     % Purple (Radius)
```

**Monochrome option (for B&W printing):**
```latex
\definecolor{algo1}{RGB}{80, 80, 80}
\definecolor{algo2}{RGB}{120, 120, 120}
\definecolor{algo3}{RGB}{160, 160, 160}
\definecolor{algo4}{RGB}{200, 200, 200}
```

**Academic blue palette:**
```latex
\definecolor{algo1}{RGB}{0, 51, 102}
\definecolor{algo2}{RGB}{51, 102, 153}
\definecolor{algo3}{RGB}{102, 153, 204}
\definecolor{algo4}{RGB}{153, 204, 255}
```

### Change Figure Scale

In the `\begin{tikzpicture}` line, modify:

```latex
\begin{tikzpicture}[scale=0.95]    % 95% of original size
```

Adjust to fit your page:
- `scale=0.75` — Very compact
- `scale=0.85` — Smaller
- `scale=0.95` — Default
- `scale=1.1` — Larger
- `scale=1.25` — Much larger

### Change Font Size

Replace `\sffamily\small` with:
- `\tiny` — Very small
- `\scriptsize` — Small
- `\small` — Current
- `\normalsize` — Larger
- `\large` — Much larger

---

## Standalone Compilation

To generate a high-quality PDF for inclusion in presentations or separate documents:

**Full version:**
```bash
cd NOTES/
pdflatex postprocessing_diagram.tex
```

**Compact version:**
```bash
cd NOTES/
pdflatex postprocessing_compact.tex
```

Then include in your main document via:
```latex
\includegraphics[width=0.9\textwidth]{NOTES/postprocessing_compact.pdf}
```

---

## Understanding the Visual Elements

### Node Colors

- **Dark blue/gray circle** = Free node (can be moved)
- **Purple circle** = BC-tagged node (locked, acts as anchor)
- **Light/dotted circle** = Node to be removed

### Edge Styles

- **Solid line** = Edge to be kept
- **Dotted line** = Edge to be removed or intermediate point
- **Line thickness** = Cross-section radius (Algorithm 4 only)

### Algorithm Progression

```
Raw Graph (many nodes, short edges, branches)
           ↓ Algorithm 1: Edge Collapse
    Collapsed (fewer nodes)
           ↓ Algorithm 2: Branch Pruning
    Pruned (no weak branches)
           ↓ Algorithm 3: RDP Simplification
    Simplified (fewer polyline points)
           ↓ Algorithm 4: Radius Estimation
    Final (with assigned radii)
```

---

## Related Content in Your Document

These diagrams work well alongside:

1. **Pseudocode boxes** — Show algorithm pseudocode for each step
2. **Parameter tables** — List default values for `d_collapse`, `l_prune`, `rdp_epsilon`
3. **Example graphs** — Show before/after on actual structures (cantilever, MBB beam)
4. **Mathematical formulations** — For radius estimation modes:
   - EDT: $r_v = \mathrm{EDT}(v) \cdot h$
   - Uniform: $r = \sqrt{V_{\mathrm{target}} / (\pi \sum_i L_i)}$

---

## Tips for Best Results

1. **Placement:** Put figures right after algorithm description, not before
2. **Caption:** Make caption detailed (tells story independently of main text)
3. **Cross-reference:** Use `\ref{fig:postprocessing}` in text
4. **Consistency:** Use same color scheme across all pipeline diagrams
5. **PDF quality:** If using external PDF, ensure 300+ DPI for printing

---

## Troubleshooting

### Figure too wide for page

**Solution:** Reduce scale
```latex
\begin{tikzpicture}[scale=0.85]
```

### Text overlaps in graph visualization

**Solution:** Increase node spacing or reduce font size
```latex
node distance=2cm     % Increase from 1.5cm
% OR
font=\sffamily\tiny   % Reduce font
```

### Colors don't print well in B&W

**Solution:** Use monochrome palette (see Customization section)

### Compilation fails with "Undefined control sequence"

**Solution:** Ensure all TikZ libraries are loaded:
```latex
\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning}
```

---

## File Locations

```
/Users/jamesjacouris/Documents/GitHub/TO_CAD/NOTES/
├── postprocessing_diagram.tex          ← Full version (standalone compilable)
├── postprocessing_compact.tex          ← Compact version (inline)
├── POSTPROCESSING_DIAGRAM_USAGE.md    ← This file
└── pipeline_overview.md                ← Main notes document
```

---

## Quick Start (2 minutes)

1. Open `postprocessing_compact.tex`
2. Copy entire content starting from `\begin{figure}`
3. Paste into your LaTeX document where you want the figure
4. Add to preamble if not already present:
   ```latex
   \usepackage{tikz}
   \usetikzlibrary{shapes, arrows, positioning}
   ```
5. Compile with `pdflatex` or `xelatex`
6. Done! ✅

---

**Last Updated:** 2026-02-17
