# Pipeline Flowchart — Usage Guide

Three professional flowchart versions have been created for your dissertation.

---

## Version 1: Simple Inline Flowchart (RECOMMENDED)

**File:** `flowchart_inline.tex`

**Best for:** Clean, readable introduction section

**How to use:**

1. Copy the entire content from `flowchart_inline.tex`
2. Paste it directly into your main LaTeX document after your introduction paragraph
3. Make sure your document preamble includes:
   ```latex
   \usepackage{tikz}
   \usetikzlibrary{shapes, arrows, positioning, shadows}
   ```

**Output:** Clean 4-stage diagram showing:
- Color-coded stages (orange → blue → green → purple)
- Main pipeline flow (stages connected left-to-right)
- Output files (dashed arrows below)
- Algorithm labels and key metrics

**Typical size:** 15cm wide × 8cm high (fits well on page)

---

## Version 2: Detailed Annotated Flowchart

**File:** `flowchart_detailed.tex`

**Best for:** Comprehensive technical description, section with detailed I/O

**Features:**
- Shows inputs (domain, loads, BCs)
- Shows outputs (JSON file names + contents)
- Shows algorithm boxes with key equations
- Shows numerical metrics (ρ, G, x, r)

**How to use:**
1. Copy content from `flowchart_detailed.tex`
2. Paste into your document
3. Same preamble requirements as Version 1

**Output:** Detailed flowchart showing full data flow

**Typical size:** 16cm wide × 10cm high

---

## Version 3: Standalone Compilable Document

**File:** `pipeline_flowchart.tex`

**Best for:** Standalone figure generation, poster, or presentation slides

**Features:**
- Includes complete LaTeX preamble
- Can be compiled independently: `pdflatex pipeline_flowchart.tex`
- Generates high-quality PDF
- Highly customizable color scheme

**How to use:**
```bash
cd NOTES/
pdflatex pipeline_flowchart.tex
```

Output: `pipeline_flowchart.pdf` (ready to import as image)

---

## Integration Tips

### Option A: Inline Integration (Simplest)

Place the figure directly in your Introduction section:

```latex
\section{Introduction}

Topology optimisation (TO) produces optimal material layouts within
a design domain, but the resulting density fields are not directly
usable in CAD or downstream finite element tools. This work presents
a four-stage pipeline...

[INSERT flowchart_inline.tex CONTENT HERE]

\section{Stage 0: Topology Optimisation}
...
```

### Option B: Separate Figure File

Create a separate file `figures/pipeline.tex` and include via:

```latex
\begin{figure}[htbp]
  \centering
  \input{figures/pipeline.tex}
  \caption{...}
  \label{fig:pipeline}
\end{figure}
```

### Option C: External PDF (Highest Quality)

Compile `pipeline_flowchart.tex` to PDF, then include:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures/pipeline_flowchart.pdf}
  \caption{...}
  \label{fig:pipeline}
\end{figure}
```

---

## Customization Guide

### Change Colors

In any `.tex` file, locate the color definitions:

```latex
\definecolor{stage0}{RGB}{230, 126, 34}    % Orange
\definecolor{stage1}{RGB}{41, 128, 185}    % Blue
\definecolor{stage2}{RGB}{39, 174, 96}     % Green
\definecolor{stage3}{RGB}{155, 89, 182}    % Purple
```

**Some alternative palettes:**

**Professional (Grays):**
```latex
\definecolor{stage0}{RGB}{100, 100, 100}
\definecolor{stage1}{RGB}{120, 120, 120}
\definecolor{stage2}{RGB}{140, 140, 140}
\definecolor{stage3}{RGB}{160, 160, 160}
```

**Vibrant (High Contrast):**
```latex
\definecolor{stage0}{RGB}{255, 102, 0}     % Bright orange
\definecolor{stage1}{RGB}{0, 102, 204}     % Bright blue
\definecolor{stage2}{RGB}{0, 153, 76}      % Bright green
\definecolor{stage3}{RGB}{153, 51, 153}    % Bright purple
```

**Academic (Muted):**
```latex
\definecolor{stage0}{RGB}{190, 140, 80}    % Muted orange
\definecolor{stage1}{RGB}{80, 120, 160}    % Muted blue
\definecolor{stage2}{RGB}{100, 150, 100}   % Muted green
\definecolor{stage3}{RGB}{140, 100, 140}   % Muted purple
```

### Change Font Sizes

In the `tikzset` definitions, modify:

```latex
font=\sffamily\small       % ← Change to \tiny, \footnotesize, \normalsize, \large
```

### Adjust Arrow Styles

Change arrow thickness:

```latex
arrow/.style={
    ->, >=stealth,
    line width=2pt,     % ← Increase/decrease thickness
    draw=black!60
}
```

### Adjust Box Sizes

Modify node dimensions:

```latex
minimum width=4cm,   % ← Change width
minimum height=1cm,  % ← Change height
```

---

## Recommended Usage for 10-Page Dissertation

**Section 1.1 Introduction:**
- Use **Version 1 (Simple Inline)** — clean and professional
- Place immediately after the descriptive paragraph
- Size to fit nicely on the page (scale=0.9 or 0.95)

**Full Page:**
- Use **Version 2 (Detailed)** — for a dedicated "Architecture" section
- Include detailed caption explaining each stage
- Cross-reference with algorithm descriptions in following sections

---

## LaTeX Preamble Requirements

Make sure your document includes:

```latex
\documentclass{article}
% ... other packages ...

\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning, shadows}

\begin{document}
% ... your content ...
\end{document}
```

If using `graphicx` for external PDFs:

```latex
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning, shadows}
```

---

## Troubleshooting

### Flowchart doesn't compile

**Problem:** `! Undefined control sequence \tikz...`

**Solution:** Ensure all TikZ libraries are loaded in preamble:
```latex
\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning, shadows}
```

### Text overlaps or misaligned

**Solution:** Adjust `node distance` parameter:
```latex
node distance=2.2cm    % Increase to spread nodes further apart
```

### Flowchart too wide/narrow for page

**Solution:** Use `scale` parameter in `\begin{tikzpicture}`:
```latex
\begin{tikzpicture}[scale=0.85]  % 0.85 = 85% of original size
```

### Colors don't match

**Solution:** Ensure RGB values are 0–255:
```latex
\definecolor{stage0}{RGB}{230, 126, 34}  % ✓ Correct
\definecolor{stage0}{RGB}{0.9, 0.5, 0.1} % ✗ Wrong (use 0-255 range)
```

---

## Quick Start

**Fastest integration (2 minutes):**

1. Open `flowchart_inline.tex`
2. Copy all code after `\begin{figure}` and before `\end{figure}`
3. Paste into your main document after your introduction text
4. Adjust caption as needed
5. Compile with `pdflatex` or `xelatex`

Done! 🎉

---

## File Locations

```
/Users/jamesjacouris/Documents/GitHub/TO_CAD/NOTES/
├── flowchart_inline.tex         ← RECOMMENDED for main document
├── flowchart_detailed.tex       ← For detailed architecture section
├── pipeline_flowchart.tex       ← Standalone compilable version
└── FLOWCHART_USAGE.md          ← This file
```

---

**Last Updated:** 2026-02-17
