# Top3D-Yin Pipeline (Final)

A robust voxel-to-CAD pipeline fusing topology optimization (Top3D) with geometric reconstruction (Yin's Algorithm). This pipeline automates the generation of lightweight lattice structures with precise boundary condition (BC) propagation.

## Features
- **Integrated Top3D**: Pure Python implementation of the 88-line code.
- **Yin Reconstruction**: Medial axis thinning (skeletonization) with BC tag protection.
- **BC Consolidation**: Ensures supports/loads are consolidated into single, connected graph nodes.
- **Size Optimization**: Resizes struts to meet the target volume fraction.

## Installation

Ensure you have Python 3.8+ and the following dependencies:

```bash
pip install numpy scipy numba matplotlib
```

## Running Test Cases

### 1. Short Cantilever (Fast Validation)
A standard benchmark case.
```bash
python run_pipeline.py \
    --nelx 60 --nely 20 --nelz 2 \
    --volfrac 0.2 --penal 3.0 --rmin 1.5 --max_loop 80 \
    --load_x 60 --load_y 10 --load_z 1 \
    --load_fy -1.0 \
    --prune_len 2.0 --collapse_thresh 3.0 --rdp 2.0 --radius_mode uniform \
    --limit 5.0 --snap 5.0 --visualize \
    --output "short_cantilever.json"
```

### 2. MBB Beam (Simply Supported)
Produces a classic Warren truss pattern.
```bash
python run_pipeline.py \
    --nelx 120 --nely 20 --nelz 2 \
    --volfrac 0.15 --penal 3.0 --rmin 2.0 --max_loop 100 \
    --load_x 60 --load_y 0 --load_z 1 \
    --load_fy -1.0 \
    --prune_len 2.0 --collapse_thresh 3.0 --rdp 2.0 --radius_mode uniform \
    --limit 5.0 --snap 5.0 --visualize \
    --output "mbb_beam.json"
```

### 3. Long Cantilever (Slender Structure)
Tests stability and connectivity in long spans.
```bash
python run_pipeline.py \
    --nelx 100 --nely 20 --nelz 2 \
    --volfrac 0.3 --penal 3.0 --rmin 2.0 --max_loop 100 \
    --load_x 100 --load_y 10 --load_z 1 \
    --load_fy -1.0 \
    --prune_len 2.0 --collapse_thresh 3.0 --rdp 2.0 --radius_mode uniform \
    --limit 5.0 --snap 5.0 --visualize \
    --output "long_cantilever.json"
```

### 4. Bridge (Two-Point Support)
Tests multi-point support handling.
```bash
python run_pipeline.py \
    --nelx 80 --nely 20 --nelz 2 \
    --volfrac 0.15 --penal 3.0 --rmin 1.5 --max_loop 80 \
    --load_x 40 --load_y 20 --load_z 1 \
    --load_fy -1.0 \
    --prune_len 2.0 --collapse_thresh 3.0 --rdp 2.0 --radius_mode uniform \
    --limit 5.0 --snap 5.0 --visualize \
    --output "bridge.json"
```
