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
    --output "matlab_replicated.json"

    --skip_top3d \
### Best Params for 150x50x4 Cantilever
    python run_pipeline.py \
    --nelx 150 --nely 40 --nelz 4 \
    --skip_top3d \
    --top3d_npz output/hybrid_v2/matlab_replicated_top3d.npz \
    --volfrac 0.3 --penal 3.0 --rmin 3.0 --max_loop 100 \
    --load_x 150 --load_y 20 --load_z 2 \
    --load_fx 0.0 --load_fy -100.0 --load_fz 0.0 \
    --prune_len 4.15 --collapse_thresh 3.84 --rdp 0.78 --radius_mode uniform \
    --limit 5.0 --snap 5.0 --visualize \
    --render_upsample 4 \
    --output "matlab_replicated_23_march.json"

    --skip_top3d \


  python run_pipeline.py \
    --nelx 150 --nely 40 --nelz 4 \
    --load_x 150 --load_y 20 --load_z 2 \
    --load_fx 0.0 --load_fy -100.0 --load_fz 0.0 \
    --prune_len 4.15 --collapse_thresh 3.84 --rdp 0.78 --radius_mode uniform \
    --limit 5.0 --snap 5.0 --visualize \
    --export_stl \
    --output "matlab_replicated_New_E.json"

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





## HYBRID TEST CASES

BEST PARAMS FOR TEST CASE

python run_pipeline.py
--skip_top3d
--top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz
--hybrid
--output Roof_Structure_strict.json
--min_plate_size 8
--flatness_ratio 7
--min_avg_neighbors 3
--visualize






Full Input

python run_pipeline.py
--top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz
--hybrid
--output Roof_Structure_strict.json
--visualize
--pitch 1.0
--max_iters 50
--vol_thresh 0.3
--plate_thickness_ratio 0.15
--detect_plates auto
--min_plate_size 8
--flatness_ratio 7
--junction_thresh 4
--min_avg_neighbors 3
--prune_len 2.0
--collapse_thresh 2.0
--rdp 2.0
--snap 5.0
--skip_top3d
--radius_mode uniform

python run_pipeline.py
--skip_top3d
--top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz
--hybrid --output Roof_Structure.json
--min_plate_size 10 --flatness_ratio 4 --junction_thresh 4 --visualize

Parameters:

--min_plate_size (default: 4) — Minimum voxels for plate classification --flatness_ratio (default: 3.0) — PCA eigenvalue ratio threshold --junction_thresh (default: 4) — Neighbor count for junction detection





python run_pipeline.py \
  --top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz \
  --skip_top3d \
  --visualize \
  --hybrid \
  --output Roof_Structure_strict.json \
  --pitch 1.0 \
  --max_iters 50 \
  --vol_thresh 0.3 \
  --plate_thickness_ratio 0.15 \
  --detect_plates auto \
  --min_plate_size 8 \
  --flatness_ratio 7 \
  --junction_thresh 4 \
  --min_avg_neighbors 3 \
  --prune_len 2.0 \
  --collapse_thresh 2.0 \
  --rdp 2.0 \
  --snap 5.0 \
  --radius_mode uniform







  # ROOF STRUCTURE TEST CASE FULL Top3D Input - NOT CORRECT OR WORKING

  python run_pipeline.py \
  --nelx 40 --nely 40 --nelz 20 \
  --volfrac 0.05 --penal 3.0 --rmin 1.5 --max_loop 100 \
  --load_fx 0.0 --load_fy 0.0 --load_fz -100.0 \
  --problem roof \
  --pitch 1.0 \
  --max_iters 50 \
  --prune_len 4.15 \
  --collapse_thresh 3.84 \
  --rdp 0.78 \
  --radius_mode uniform \
  --vol_thresh 0.3 \
  --plate_thickness_ratio 0.15 \
  --detect_plates auto \
  --min_plate_size 8 \
  --flatness_ratio 7 \
  --junction_thresh 4 \
  --min_avg_neighbors 3 \
  --hybrid \
  --limit 5.0 \ 
  --snap 5.0 \
  --iters 50 \
  --skip_top3d \
  --output_dir output/hybrid_v2 \
  --visualize \
  --output Roof_Structure_Test.json





###roof_slab test case for hybrid beam-plate testing. Here are the recommended commands:

Quick Top3D-only test (20 iterations):

Medium-sized (better geometry):


python run_pipeline.py \
  --problem roof_slab \
  --nelx 60 --nely 60 --nelz 6 \
  --volfrac 0.12 --penal 3.0 --rmin 2.0 \
  --top3d_iters 80 \
  --load_fy -150.0 \
  --hybrid --opt_loops 2 \
  --output roof_slab_60x60.json \
  --visualize
The roof_slab structure should naturally decompose into:

Plate zone: the thin roof slab (high plate score from planarity + uniform EDT)
Beam zones: the 9 support columns (high beam score from linearity + BC load/support density)





# QUADCOPTER TEST CASE

## Stage 0 — ~5-10 minutes
python run_top3d.py \
  --problem quadcopter \
  --nelx 60 --nely 60 --nelz 20 \
  --volfrac 0.05 --penal 3.0 --rmin 2.0 --max_loop 100 \
  --load_fy -100.0 \
  --motor_arm_frac 0.3 --load_patch_frac 0.15 \
  --output output/hybrid_v2/quadcopter_top3d.npz

## Pipeline (run after Top3D finishes)
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/quadcopter_top3d.npz \
  --nelx 60 --nely 60 --nelz 20 \
  --prune_len 2.0 --collapse_thresh 2.0 --rdp 1.0 \
  --radius_mode uniform --hybrid --visualize \
  --output quadcopter.json



python run_pipeline.py \
    --problem quadcopter \
  --nelx 60 --nely 60 --nelz 6 \
  --volfrac 0.10 --penal 3.0 --rmin 2.0 --max_loop 100 \
  --load_fy -100.0 \
  --motor_arm_frac 0.1 --load_patch_frac 0.1 \
  --motor_radius 4 \
  --visualize \
  --output output/hybrid_v2/quadcopter_v2_top3d.npz



  python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/quadcopter_v2_top3d_top3d.npz \
  --nelx 60 --nely 60 --nelz 6 \
  --prune_len 2.0 --collapse_thresh 2.0 --rdp 1.0 \
  --radius_mode uniform \
  --hybrid --optimize \
  --visualize \
  --output quadcopter_v2.json



  # Wider bolt spacing + lower volfrac → clearly separated chord members
python run_top3d.py --problem quadcopter \
  --nelx 80 --nely 80 --nelz 6 \
  --volfrac 0.12 --penal 3.0 --rmin 2.5 --max_loop 150 \
  --load_fy -100.0 \
  --motor_arm_frac 0.12 --load_patch_frac 0.15 \
  --motor_bolt_spacing 8 --arm_load_n 3 --arm_load_frac 0.4 \
  --output output/hybrid_v2/quad_complex_v3_top3d.npz


python run_pipeline.py \
  --problem quadcopter \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/quad_complex_v3_top3d.npz \
  --nelx 80 --nely 80 --nelz 6 \
  --volfrac 0.12 --penal 3.0 --rmin 2.5 --max_loop 150 \
  --load_fy -100.0 \
  --motor_arm_frac 0.12 --load_patch_frac 0.15 \
  --motor_bolt_spacing 8 --arm_load_n 3 --arm_load_frac 0.4 \
  --prune_len 2.0 --collapse_thresh 2.0 --rdp 1.0 \
  --visualize \
  --output quad_complex_v3.json





  python run_pipeline.py \
  --problem quadcopter \
  --nelx 60 --nely 60 --nelz 6 \
  --volfrac 0.1 --penal 3.0 --rmin 2.0 --max_loop 100 \
  --load_fy -100.0 \
  --motor_arm_frac 0.12 --load_patch_frac 0.12 \
  --motor_bolt_spacing 6 --arm_load_n 4 --arm_load_frac 0.35 \
  --arm_void_width 6 \
  --prune_len 2.0 --collapse_thresh 2.0 --rdp 0.8 \
  --radius_mode uniform \
  --visualize \
  --output quad_branching.json  



  # CURVED OPTIMISATION

  ## Full run from scratch with curved IGA optimisation
python run_pipeline.py \
  --nelx 40 --nely 20 --nelz 4 --volfrac 0.2 \
  --problem tagged \
  --curved \
  --opt_loops 2 \
  --output curved_result.json

## Skip Top3D (reuse existing NPZ) — fastest for testing
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz \
  --nelx 40 --nely 40 --nelz 20 --volfrac 0.05 \
  --curved \
  --opt_loops 2 \
  --visualize \
  --output roof_iga.json

## Hybrid beam+plate with curved IGA beams
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz \
  --hybrid \
  --beam_mode curved \
  --min_plate_size 8 \
  --flatness_ratio 7 \
  --junction_thresh 4 \
  --min_avg_neighbors 3 \
  --prune_len 2.0 \
  --collapse_thresh 2.0 \
  --rdp 2.0 \
  --snap 5.0 \
  --radius_mode uniform \
  --optimize \
  --opt_loops 2 \
  --visualize \
  --output hybrid_iga.json


  python run_pipeline.py --skip_top3d \
  --top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz \
  --nelx 40 --nely 40 --nelz 20 --volfrac 0.05 \
  --hybrid --curved --optimize --opt_loops 2 \
  --r_min 0.5 --limit 5.0 --ctrl_limit 1.5 \
  --output roof_hybrid_curved.json



  # Wall Bracket Test Case

python run_pipeline.py \
--problem wall_bracket \
--nelx 60 --nely 30 --nelz 30 \
--volfrac 0.1 \
--load_fy -10.0 \
--output Wall_Bracket_V2.json \
--visualize \
--pitch 1.0 \
--penal 3.0 \
--rmin 1.5 \
--max_loop 100 \
--prune_len 3.0 \
--collapse_thresh 2.5 \
--skip_top3d \
--top3d_npz output/hybrid_v2/Wall_Bracket_top3d.npz \
--output_dir output/hybrid_v2




# Curved IGA Test Case

# Stage 0: Topology Optimisation (~30s)
python run_top3d.py \
  --nelx 60 --nely 10 --nelz 20 \
  --volfrac 0.10 --penal 3.0 --rmin 1.5 \
  --problem simply_supported \
  --max_loop 100 \
  --output output/simply_supported_top3d.npz

# Stage 1–3: Curved Pipeline (~2min)
python run_pipeline.py \
  --top3d_npz output/simply_supported_top3d.npz \
  --nelx 60 --nely 10 --nelz 20 \
  --curved --optimize --opt_loops 2 \
  --r_min 0.5 --limit 5.0 --ctrl_limit 2.0 \
  --visualize \
  --skip_top3d \
  --output output/simply_supported_curved.json

  ## Trial 2
  # Stage 0: Topology Optimisation
python run_top3d.py \
  --nelx 40 --nely 10 --nelz 40 \
  --volfrac 0.30 --penal 3.0 --rmin 1.5 \
  --problem l_bracket \
  --max_loop 100 \
  --output output/l_bracket_top3d.npz

# Stage 1–3: Curved Pipeline
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/l_bracket_top3d.npz \
  --nelx 40 --nely 10 --nelz 40 \
  --beam_mode curved --optimize --opt_loops 2 \
  --r_min 0.5 --limit 5.0 --ctrl_limit 2.0 \
  --visualize \
  --output output/l_bracket_curved.json


  ## 3
  # Stage 0: Top3D with staggered obstacles (~2min for 80×10×40)
python run_top3d.py \
  --nelx 80 --nely 10 --nelz 40 \
  --volfrac 0.25 --penal 3.0 --rmin 2.0 \
  --problem obstacle_course \
  --max_loop 150 \
  --output output/obstacle_course_top3d.npz

# Stage 1-3: Curved Pipeline
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/obstacle_course_top3d.npz \
  --nelx 80 --nely 10 --nelz 40 \
  --curved --optimize --opt_loops 2 \
  --r_min 0.5 --limit 5.0 --ctrl_limit 2.0 \
  --visualize \
  --output output/obstacle_course_curved.json




  The vault problem is added to run_top3d.py. To run it:
# Stage 0: Topology Optimisation
python run_top3d.py \
  --nelx 40 --nely 10 --nelz 30 \
  --volfrac 0.12 --penal 3.0 --rmin 1.5 \
  --problem vault \
  --max_loop 100 \
  --output output/vault_top3d.npz

# Stage 1-3: Curved Pipeline
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/vault_top3d.npz \
  --nelx 40 --nely 10 --nelz 30 \
  --curved --opt_loops 2 \
  --r_min 0.5 --limit 5.0 --ctrl_limit 2.0 \
  --visualize \
  --output output/vault_curved.json



# Beam Mode CHANGE - NEW FLAG

Here's the summary:

--beam_mode replaces the old --curved flag with three clear options:


## All straight (default) — standard Euler-Bernoulli, no Bézier
python run_pipeline.py --beam_mode straight --output result.json

## All curved — every edge gets Bézier ctrl_pts (previous --curved behaviour)
python run_pipeline.py --beam_mode curved --output result.json

## Mixed — per-edge classification based on skeleton curvature
python run_pipeline.py --beam_mode mixed --output result.json

## Mixed with custom threshold
python run_pipeline.py --beam_mode mixed --curve_threshold 1.0 --output result.json
The old --curved flag still works as a backwards-compatible alias for --beam_mode curved.


python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz \
  --hybrid \
  --beam_mode mixed \
  --geo_reg 1.5 \
  --min_plate_size 8 \
  --flatness_ratio 7 \
  --junction_thresh 4 \
  --min_avg_neighbors 3 \
  --prune_len 2.0 \
  --collapse_thresh 2.0 \
  --rdp 2.0 \
  --snap 5.0 \
  --radius_mode uniform \
  --optimize \
  --opt_loops 2 \
  --visualize \
  --output hybrid_iga.json








python run_pipeline.py \
  --top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz \
  --skip_top3d \
  --visualize \
  --beam_mode mixed \
  --output Roof_Structure_strict_Mixed_Beams.json \
  --pitch 1.0 \
  --max_iters 50 \
  --vol_thresh 0.3 \
  --plate_thickness_ratio 0.15 \
  --detect_plates \
  --min_plate_size 8 \
  --flatness_ratio 7 \
  --junction_thresh 4 \
  --min_avg_neighbors 3 \
  --prune_len 2.0 \
  --collapse_thresh 2.0 \
  --rdp 2.0 \
  --snap 5.0 \
  --optimize \
  --opt_loops 2 \
  --radius_mode uniform





python run_pipeline.py \
  --top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz \
  --skip_top3d \
  --visualize \
  --beam_mode straight \
  --hybrid \
  --output Roof_Structure_Hybrid_Beams_SURFACES_V1.json \
  --pitch 1.0 \
  --max_iters 50 \
  --vol_thresh 0.3 \
  --prune_len 2.0 \
  --collapse_thresh 2.0 \
  --rdp 2.0 \
  --snap 5.0 \
  --radius_mode uniform



  # Curved Cases


# Generate topology - NOT WORKING BAD
python run_top3d.py --problem curved_shell --nelx 40 --nely 10 --nelz 20 \
  --volfrac 0.10 --rmin 2.0 --max_loop 80 \
  --output output/hybrid_v2/curved_shell_top3d.npz

# Reconstruct with hybrid pipeline
python run_pipeline.py --skip_top3d \
  --top3d_npz output/hybrid_v2/curved_shell_top3d.npz \
  --hybrid --output curved_shell_hybrid.json --visualize







# Pipe Bracket
  The pipe bracket problem is implemented. Here's a summary of what was added to run_top3d.py:

Problem: pipe_bracket (Yin's paper, Section 5.3)

Parameter	Value	Scaling
Domain	120 x 60 x 40 (nelx x nely x nelz)	User-configurable
Cylinder R	18 (0.3 * nely)	Scales with height
Pipe 1 centre	(36, 30) = (0.3*nelx, nely/2)	Scales with domain
Pipe 2 centre	(84, 30) = (0.7*nelx, nely/2)	Scales with domain
Fixed	4 vertical corner edges (all Y)	244 nodes at full res
Loads	4 cardinal pts per pipe, F=100 each (-Y)	8 support points total
Passive voids	Two through-Z cylinders	~28% of domain
To run at paper resolution:


python run_top3d.py \
  --problem pipe_bracket \
  --nelx 120 --nely 60 --nelz 40 \
  --volfrac 0.10 --rmin 3.0 --penal 3.0 \
  --max_loop 100 \
  --output output/hybrid_v2/pipe_bracket_top3d.npz


For faster testing (8x fewer elements):


python run_top3d.py \
  --problem pipe_bracket \
  --nelx 60 --nely 30 --nelz 20 \
  --volfrac 0.10 --rmin 3.0 \
  --max_loop 100 \
  --output output/hybrid_v2/pipe_bracket_test_top3d.npz

RUN IT:

python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/pipe_bracket_test_top3d.npz \
  --nelx 60 --nely 30 --nelz 20 \
  --load_fy -800 \
  --volfrac 0.10 \
  --beam_mode mixed \
  --rdp 2.0 \
  --snap 5.0 \
  --radius_mode uniform \
  --prune_len 2.0 \
  --collapse_thresh 2.0 \
  --optimize \
  --opt_loops 2 \
  --output pipe_bracket_test.json \
  --visualize


python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/pipe_bracket_test_top3d.npz \
  --nelx 60 --nely 30 --nelz 20 \
  --load_fy -800 \
  --volfrac 0.10 \
  --rdp 2.0 \
  --snap 5.0 \
  --radius_mode uniform \
  --prune_len 5.0 \
  --collapse_thresh 2.0 \
  --optimize \
  --opt_loops 2 \
  --output pipe_bracket_test.json \
  --visualize

The cylinder parameters, support positions, and load points all scale proportionally, so reduced resolution will produce the same topology at lower fidelity.



WORKING BIG PIPE BRACKET

python run_top3d.py \
  --problem pipe_bracket \
  --nelx 120 --nely 60 --nelz 40 \
  --volfrac 0.10 --rmin 3.0 --penal 3.0 \
  --max_loop 100 \
  --output output/hybrid_v2/pipe_bracket_full_top3d.npz
This will take a while (~288K elements), but it should produce a topology where:

The pipes have R=18 in height=60 — leaving 12 elements above/below each pipe
The upper arches are 3-4 voxels thick — proper structural members, not fragile 1-voxel chains
The skeleton will naturally connect without needing bridge edges
Once it finishes, the pipeline will auto-read the load vector from the NPZ:


python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/pipe_bracket_full_top3d.npz \
  --nelx 120 --nely 60 --nelz 40 \
  --volfrac 0.10 \
  --hybrid \
  --beam_mode straight \
  --rdp 1.0 --snap 5.0 \
  --radius_mode uniform \
  --prune_len 1.0 --collapse_thresh 1.0 \
  --output pipe_bracket_full.json \
  --visualize




python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/pipe_bracket_full_top3d.npz \
  --snap 10.0 --collapse_thresh 10.0 --prune_len 10.0 --rdp 2.0 \
  --r_min 0.5 --r_max 8.0 \
  --opt_loops 2 --iters 50 \
  --beam_mode mixed \
  --output pipe_bracket_optimal_mixed_beams.json

  ### Hybrid Pipe Bracket

  python run_pipeline.py --skip_top3d \
  --top3d_npz output/hybrid_v2/pipe_bracket_full_top3d.npz \
  --hybrid --optimize --opt_loops 2 \
  --snap 2.0 --r_min 0.5 --r_max 8.0 --rdp 2.0 \
  --min_plate_size 2 --flatness_ratio 20.0 --min_avg_neighbors 2.0 \
  --visualize \
  --output pipe_bracket_hybrid_optimal.json





python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/Clear_Beam_Plate_Test_top3d_TEST_20_MARCH.npz\
  --hybrid \
  --output Clear_Test_strict_V2.json \
  --min_plate_size 8 \
  --flatness_ratio 7 \
  --min_avg_neighbors 3 \
  --visualize





python run_pipeline.py \
  --problem roof \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/Clear_Beam_Plate_Test_top3d_TEST_10am.npz \
  --nelx 40 --nely 40 --nelz 20 \
  --volfrac 0.08 --penal 3.0 --rmin 1.5 \
  --max_loop 100 \
  --load_fz -1.0 \
  --visualize \
  --hybrid \
  --min_plate_size 8 \
  --flatness_ratio 7 \
  --min_avg_neighbors 3 \
  --optimize \
  --opt_loops 2 \
  --output output/hybrid_v2/Clear_Beam_Plate_Test_top3d_TEST_10am.npz





















# Generate NPZ:
python run_top3d.py \
  --problem roof \
  --nelx 40 --nely 40 --nelz 20 \
  --volfrac 0.05 --penal 3.0 --rmin 1.5 \
  --max_loop 100 \
  --output output/hybrid_v2/Roof_Structure_Test_top3d.npz

# Reconstruct (your existing command):
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/Roof_Structure_Test_top3d.npz \
  --hybrid \
  --output Roof_Structure_strict.json \
  --min_plate_size 8 --flatness_ratio 7 --min_avg_neighbors 3 \
  --optimize \
  --opt_loops 2 \
  --visualize




  FOR THE BEAM SUPPORTED PLATE TOP3D:

  python tests/test_hybrid_clear.py

  python run_pipeline.py --skip_top3d --top3d_npz output/hybrid_v2/Clear_Beam_Plate_Test_top3d.npz --hybrid --output Clear_Test.json --visualize




# CORRECT BEAM SUPPORTED PLATE TO CLI: 

python run_top3d.py \
  --problem roof \
  --nelx 40 --nely 40 --nelz 20 \
  --volfrac 0.05 --penal 3.0 --rmin 1.5 \
  --max_loop 200 \
  --load_fx 0.0 --load_fy 0.0 --load_fz -5.0 \
  --output output/hybrid_v2/Clear_Beam_Plate_Test_top3d_0.05_VF.npz


python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/Clear_Beam_Plate_Test_top3d_0.05_VF.npz \
  --beam_mode mixed \
  --output Clear_Beam_Plate_Test_top3d_0.05_VF.json \
  --min_plate_size 8 \
  --flatness_ratio 7 \
  --min_avg_neighbors 3 \
  --snap 1.5 \
  --prune_len 1.5 \
  --collapse_thresh 2.0 \
  --rdp 2.0 \
  --optimize \
  --symmetry xz,yz --sym_weight 0.1 \
  --opt_loops 2



python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/Clear_Beam_Plate_Test_top3d_0.05_VF.npz \
  --beam_mode straight \
  --output Clear_Beam_Plate_Test_top3d_0.05_VF_NOT_HYBRID.json \
  --snap 1.5 \
  --prune_len 1.5 \
  --collapse_thresh 0.0 \
  --rdp 1.0 





python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/Clear_Beam_Plate_Test_top3d_LOWER_VF.npz \
  --hybrid \
  --beam_mode straight \
  --output Clear_Test_LOWER_VF.json \
  --min_plate_size 8 \
  --flatness_ratio 7 \
  --min_avg_neighbors 3 \
  --snap 1.5 \
  --prune_len 1.5 \
  --collapse_thresh 2.0 \
  --rdp 2.0 \
  --optimize \
  --opt_loops 2 
  --visualize







python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/Clear_Beam_Plate_Test_top3d.npz \
  --beam_mode straight \
  --hybrid \
  --output Frame_Supported_Plate_March_26.json \
  --min_plate_size 8 \
  --flatness_ratio 7 \
  --min_avg_neighbors 3 \
  --prune_len 2.0 \
  --collapse_thresh 2.0 \
  --rdp 2.0 \
  --snap 5.0 \
  --optimize \
  --opt_loops 2



# ELEVATED SLAB -  Top3D (already done — NPZ exists)
python run_top3d.py --problem elevated_slab \
    --nelx 40 --nely 40 --nelz 30 \
    --volfrac 0.15 --rmin 1.5 --max_loop 80 \
    --output output/hybrid_v2/elevated_slab_v3_top3d.npz

# Hybrid pipeline
python run_pipeline.py --skip_top3d \
    --top3d_npz output/hybrid_v2/elevated_slab_v3_top3d.npz \
    --nelx 40 --nely 40 --nelz 30 --volfrac 0.15 \
    --hybrid --output Elevated_Slab_Final.json \
    --optimize --opt_loops 2 


