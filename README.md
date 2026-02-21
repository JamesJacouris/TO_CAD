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

### 150x50x4 Cantilever 

    python run_pipeline.py \
    --nelx 150 --nely 40 --nelz 4 \
    --volfrac 0.3 --penal 3.0 --rmin 3.0 --max_loop 100 \
    --load_x 150 --load_y 20 --load_z 2 \
    --load_fx 0.0 --load_fy -100.0 --load_fz 0.0 \
    --prune_len 2.0 --collapse_thresh 3.0 --rdp 2.5 --radius_mode uniform \
    --limit 5.0 --snap 5.0 --visualize \
    --skip_top3d \
    --output "matlab_replicated.json"


### Best Params for 150x50x4 Cantilever
    python run_pipeline.py \
    --nelx 150 --nely 40 --nelz 4 \
    --volfrac 0.3 --penal 3.0 --rmin 3.0 --max_loop 100 \
    --load_x 150 --load_y 20 --load_z 2 \
    --load_fx 0.0 --load_fy -100.0 --load_fz 0.0 \
    --prune_len 4.15 --collapse_thresh 3.84 --rdp 0.78 --radius_mode uniform \
    --limit 5.0 --snap 5.0 --visualize \
    --skip_top3d \
    --output "matlab_replicated.json"



# Run 100 trials (~3-5 minutes)
python tune_parameters.py output/hybrid_v2/matlab_replicated_top3d.npz --trials 100

python tune_parameters.py output/hybrid_v2/short_cantilever_top3d.npz --trials 100

# Best Params for Short Cantilever
python run_pipeline.py \
    --nelx 60 --nely 20 --nelz 2 \
    --volfrac 0.2 --penal 3.0 --rmin 1.5 --max_loop 80 \
    --load_x 60 --load_y 10 --load_z 1 \
    --load_fy -1.0 \
    --prune_len 2.0 --collapse_thresh 3.0 --rdp 2.0 --radius_mode uniform \
    --limit 5.0 --snap 5.0 --visualize \
    --output "short_cantilever.json"





python run_pipeline.py \
  --nelx 150 --nely 40 --nelz 4 \
  --volfrac 0.3 --penal 3.0 --rmin 3.0 --max_loop 100 \
  --load_x 150 --load_y 20 --load_z 2 \
  --load_fx 0.0 --load_fy -100.0 --load_fz 0.0 \
  --pitch 1.0 \
  --max_iters 50 \
  --prune_len 4.15 \
  --collapse_thresh 3.84 \
  --rdp 0.78 \
  --radius_mode uniform \
  --vol_thresh 0.3 \
  --limit 5.0 \
  --snap 5.0 \
  --iters 50 \
  --opt_loops 2 \
  --problem tagged \
  --output_dir output/hybrid_v2 \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/matlab_replicated_top3d.npz \
  --visualize \
  --output matlab_replicated.json



  run_top3d_rocker_arm.py — new dedicated script
Domain defaults match Yin's paper: 75 × 60 × 12, volfrac=0.085, penal=3, rmin=3, 100 iterations.

BC / Load	Position	What
Rear clamped joints	x=0, full Y range, all Z	All 3 DOFs fixed
Front Y-restricted	x=75, full Y range, all Z	Y DOF only fixed
200 N load (symmetric)	x=37, y=60 (top), z=0 & z=12	−100 N each
100 N load (symmetric)	x=75 (front), y=30, z=0 & z=12	−50 N each
Passive void	x=[20..50), y=[5..28), all Z	Motor/pump opening
All positions are configurable via fractional arguments (--void_x0 0.27, --load200_y 1.0, etc.). Run it with:


# Quick start (all defaults)
python run_top3d_rocker_arm.py

# Then reconstruct the skeleton
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/rocker_arm_top3d.npz \
  --nelx 75 --nely 60 --nelz 12 \
  --vol_thresh 0.28 \
  --visualize \
  --output rocker_arm_final.json
