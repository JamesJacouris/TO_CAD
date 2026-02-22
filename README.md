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

## Running Test Cases (Command Cheat Sheet)

### 1. Quick Start / Standard Cantilever
A standard bench testing profile.
```bash
python run_pipeline.py --problem cantilever --nelx 60 --nely 20 --nelz 4 --volfrac 0.3 --visualize
```

### 2. Rocker Arm (Hybrid Plate + Optimization)
Runs the Rocker Arm domain, extracts the structural skeleton with plate preservation, and optimizes layout and beam radii.
```bash
python run_pipeline.py --problem rocker_arm --nelx 40 --nely 20 --nelz 10 \
  --hybrid --optimize --visualize
```

### 3. Roof Structure (Skip Top3D, from NPZ)
Reconstructs and optimizes an existing design extracted from a Top3D `.npz` file.
```bash
python run_pipeline.py --problem tagged --skip_top3d --npz_path output/hybrid_v2/Roof_Structure_Test.npz \
  --hybrid --optimize --visualize
```

## Parameter Documentation
Use `--help` on any script (e.g., `python run_pipeline.py --help`) to view the complete list of simplified parameters, such as `--top3d_iters`, `--skel_iters`, `--opt_iters`, and `--move_limit`.
