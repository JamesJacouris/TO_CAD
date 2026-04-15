# TO_CAD

Convert 3-D topology optimisation density fields into parametric CAD models
(beams, plates, and fused solid bodies) ready for FreeCAD and STEP export.

## Pipeline Overview

TO_CAD chains four stages to go from a design domain to editable CAD geometry:

```
Stage 0          Stage 1              Stage 2           Stage 3
Top3D SIMP  -->  Yin Medial-Axis  --> OC Cross-Section --> L-BFGS-B Layout
(density .npz)   Thinning (.json)     Sizing              Optimisation
                 + Plate extraction
```

**Stage 0** runs 3-D SIMP topology optimisation with density filtering and
boundary-condition tagging.
**Stage 1** thins the density field to a medial-axis skeleton, extracts a beam
graph and plate regions, assigns EDT-based radii, and classifies
beam vs plate zones using a two-signal topological test.
**Stage 2** sizes beam cross-sections with Optimality Criteria to minimise
frame compliance at a target volume fraction.
**Stage 3** moves node positions with L-BFGS-B to further reduce compliance
while preserving connectivity.

## Key Features

- **Python Top3D** — 3-D SIMP topology optimisation with passive voids, BC tag
  generation, and configurable load/support positions
- **Hybrid beam + plate extraction** — automatic zone classification, separate
  thinning modes, joint creation at beam-plate interfaces
- **Two-signal zone classification** — topological skeleton difference (mode 3
  vs mode 0) combined with octant plane pattern detection
- **Curved Bezier beams** — optional smooth geometry with IGA Timoshenko
  elements for curved members
- **Shell FEM** — MITC3 flat-triangle shell elements for plate regions
- **Mirror-half symmetry** — `--symmetry x|y|z` splits the skeleton at the
  symmetry plane and reflects after optimisation
- **External mesh input** — start from an STL/OBJ file instead of Top3D
- **FreeCAD export** — swept tubes, polygon plates, B-spline surfaces, fused
  solid body, radius heatmap, and pipeline timeline playback
- **Convergence reporting** — multi-section PDF/SVG figures with combined
  SIMP + frame compliance plots

## Installation

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Optional: install [FreeCAD](https://www.freecad.org/) (0.21+) for CAD export,
and [Graphviz](https://graphviz.org/) for Sphinx documentation diagrams.

## Quick Start

### 1. Cantilever beam (full pipeline)

```bash
python run_pipeline.py \
    --nelx 60 --nely 20 --nelz 4 \
    --volfrac 0.3 --problem cantilever \
    --opt_loops 2 --output cantilever.json
```

### 2. Reuse an existing density field

```bash
python run_pipeline.py --skip_top3d \
    --top3d_npz output/hybrid_v2/my_top3d.npz \
    --output reused.json
```

### 3. External mesh (STL/OBJ)

```bash
python run_pipeline.py \
    --mesh_input models/rocker_arm.stl \
    --mesh_pitch 0.5 --output rocker_arm.json
```

### 4. Import into FreeCAD

Open FreeCAD, then **Macro > Macros...** > browse to
`src/export/freecad_reconstruct.py` > **Execute**. Select the output
`*_history.json` file when prompted.

## Pipeline Stages

### Stage 0 — Topology Optimisation (`run_top3d.py`)

SIMP density optimisation with OC updates. Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--nelx/nely/nelz` | 60/20/4 | Domain dimensions |
| `--volfrac` | 0.3 | Target volume fraction |
| `--problem` | cantilever | Problem type (cantilever, mbb, roof, quadcopter) |
| `--max_loop` | 80 | Maximum iterations |

### Stage 1 — Skeleton Reconstruction (`reconstruct.py`)

Yin medial-axis thinning, graph extraction, and plate detection. Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--prune_len` | 2.0 | Remove branches shorter than X mm |
| `--collapse_thresh` | 2.0 | Merge nodes closer than X mm |
| `--rdp` | 1.0 | RDP simplification tolerance (0 = off) |
| `--radius_mode` | edt | Radius assignment (edt or uniform) |
| `--hybrid` | off | Enable beam + plate hybrid pipeline |

### Stage 2 — Size Optimisation (`size_opt.py`)

OC cross-section sizing minimises compliance subject to volume constraint.

### Stage 3 — Layout Optimisation (`layout_opt.py`)

L-BFGS-B node-position optimisation with optional symmetry constraints.

| Flag | Default | Description |
|------|---------|-------------|
| `--opt_loops` | 0 | Number of size+layout iterations |
| `--symmetry` | none | Mirror symmetry axis (x, y, or z) |
| `--limit` | 5.0 | Max node displacement per iteration |

## Example Use Cases

```bash
# Hybrid beam+plate roof structure
python run_pipeline.py --nelx 60 --nely 60 --nelz 30 \
    --volfrac 0.1 --problem roof --hybrid --output roof.json

# Quadcopter frame with passive motor voids
python run_pipeline.py --nelx 40 --nely 40 --nelz 4 \
    --volfrac 0.10 --problem quadcopter \
    --motor_radius 2 --opt_loops 2 --output quad.json

# Curved Bezier beams (geometry only)
python run_pipeline.py --skip_top3d \
    --top3d_npz output/hybrid_v2/roof_top3d.npz \
    --beam_mode curved --opt_loops 2 --output roof_curved.json

# Mirror symmetry about the Y axis
python run_pipeline.py --skip_top3d \
    --top3d_npz output/hybrid_v2/cantilever_top3d.npz \
    --symmetry y --opt_loops 2 --output sym_cantilever.json

# External mesh with visualisation
python run_pipeline.py --mesh_input models/bracket.stl \
    --mesh_pitch 0.5 --visualize --output bracket.json

# Run only Top3D (no reconstruction)
python run_top3d.py --nelx 80 --nely 30 --nelz 6 \
    --volfrac 0.2 --problem cantilever \
    --output output/hybrid_v2/cantilever_top3d.npz
```

## FreeCAD Export

The FreeCAD macro (`src/export/freecad_reconstruct.py`) creates:

- **Beams** — swept circular tubes with hemispherical end caps, coloured by
  radius (blue = thin, red = thick)
- **Plates** — polygon faces or B-spline surfaces with offset shells for
  curved plates
- **Fused body** — a single solid via batched `multiFuse` + `removeSplitter`,
  ready for STEP export
- **Timeline groups** — one group per pipeline stage for before/after comparison
- **Density legend** — 10 coloured reference cubes (red-yellow-green gradient)

## Documentation

Full Sphinx documentation is available in `docs/`. To build locally:

```bash
cd docs && make html
open _build/html/index.html
```

The docs cover architecture, algorithms, API reference, CLI reference, and
step-by-step guides including a pipeline walkthrough and FreeCAD export guide.

## Project Structure

```
TO_CAD/
├── run_pipeline.py              # Main entry point (Stages 0-3)
├── run_top3d.py                 # Standalone Stage 0
├── tune_parameters.py           # Optuna hyperparameter tuning
├── requirements.txt
│
├── src/
│   ├── optimization/            # FEM solvers and optimisers
│   │   ├── fem.py               #   Frame + shell + IGA elements
│   │   ├── size_opt.py          #   OC cross-section sizing
│   │   ├── layout_opt.py        #   L-BFGS-B node layout
│   │   ├── symmetry.py          #   Mirror-half enforcement
│   │   └── top3d.py             #   3-D SIMP solver
│   │
│   ├── pipelines/baseline_yin/  # Skeleton reconstruction
│   │   ├── reconstruct.py       #   Pipeline orchestrator
│   │   ├── thinning.py          #   Yin medial-axis thinning
│   │   ├── topology.py          #   Simple-point test
│   │   ├── graph.py             #   Graph extraction + classification
│   │   ├── plate_extraction.py  #   Plate region detection
│   │   ├── joint_creation.py    #   Beam-plate joints
│   │   └── visualization.py     #   3-D plotting
│   │
│   ├── curves/spline.py         # Bezier fitting + sanitisation
│   ├── export/                  # FreeCAD macro + STL/VTK export
│   ├── mesh_import/             # External STL/OBJ voxelisation
│   ├── problems/                # BC configurations (cantilever, roof, etc.)
│   ├── reporting/               # Convergence figure generation
│   └── tuning/                  # Optuna pipeline runner
│
├── docs/                        # Sphinx RST documentation
├── tests/                       # Test suite
└── scripts/                     # Development/analysis scripts
```

## Output Files

| File | Description |
|------|-------------|
| `*_top3d.npz` | Raw density array + BC tags from Top3D |
| `*_1_reconstructed.json` | Stage 1 skeleton graph |
| `*_3_sized_loop1.json` | After size optimisation |
| `*_2_layout_loop1.json` | After layout optimisation |
| `*.json` | Final output (copy of last stage) |
| `*_history.json` | Full history for FreeCAD timeline |

## License

This project was developed as part of a dissertation at the University of Bath.
