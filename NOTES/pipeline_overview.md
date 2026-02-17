# TO_CAD Pipeline Overview

**Branch:** `Top3D_Yin_Pipeline_V2`
**Date:** February 17, 2026
**Purpose:** Dissertation planning and technical documentation

---

## Table of Contents

1. [Dissertation Structure](#dissertation-structure)
2. [Full Pipeline Architecture](#full-pipeline-architecture)
3. [Stage-by-Stage Breakdown](#stage-by-stage-breakdown)
4. [Key Mathematical Formulations](#key-mathematical-formulations)
5. [LaTeX Technical Document](#latex-technical-document)
6. [References](#references)

---

## Dissertation Structure

### Suggested 10-Page Outline

| Section | Pages | Content |
|---------|-------|---------|
| 1. Introduction & Motivation | 1.0 | Problem statement, gap in literature |
| 2. Background & Literature Review | 1.5 | SIMP, thinning algorithms, frame optimization |
| 3. Pipeline Architecture | 0.5 | System overview diagram |
| 4. Stage 0 – Topology Optimization | 1.5 | SIMP, FEA, OC method |
| 5. Stage 1 – Voxel Reconstruction | 2.0 | Yin's algorithm, graph extraction, BC tags |
| 6. Stages 2 & 3 – Layout & Size Opt | 1.5 | Frame FEA, compliance minimization |
| 7. Results & Validation | 1.5 | Cantilever, MBB, bridge test cases |
| 8. Conclusions & Future Work | 0.5 | Limitations, next steps |

---

## Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 0: Topology Optimization (run_top3d.py)                        │
│  Input:  Domain (nelx × nely × nelz), VolFrac, Load, BCs            │
│  Output: top3d.npz (rho densities, bc_tags grid)                    │
│  Process: SIMP-based FEA + OC update                               │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Reconstruction (reconstruct.py)                             │
│  Input:  top3d.npz                                                  │
│  Output: *_1_reconstructed.json (graph + curves + history)          │
│  Sub-stages:                                                         │
│    1a. Voxelization & Thresholding (rho > vol_thresh)              │
│    1b. Thinning: Yin's Algorithm 3.1 (medial axis extraction)      │
│    1c. Graph Extraction: Alg 4.1 + BC tag consolidation (Alg 4.2)  │
│    1d. Post-Processing:                                            │
│        - 4A: Collapse short edges (merge nearby nodes)              │
│        - 4B: Prune branches (remove weak extremities)               │
│        - 4C: RDP polyline simplification                            │
│        - 4D: Radius estimation (EDT or uniform volume matching)    │
│        - 4E: Ensure nodes at bounding extrema                       │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Layout Optimization (layout_opt.py)                         │
│  Input:  *_1_reconstructed.json                                     │
│  Output: *_2_layout.json (optimized node positions)                 │
│  Process:                                                            │
│    - Load problem config (BCs from node_tags via TaggedProblem)    │
│    - FEA solve for initial compliance                               │
│    - Constrained optimization (scipy.optimize.minimize)             │
│    - Snap nearby free nodes to merge redundancy                     │
│    - Enforce move limits + design domain bounds                     │
│    - Report: compliance reduction, volume conservation, movement    │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Size Optimization (size_opt.py)                             │
│  Input:  *_2_layout.json                                            │
│  Output: *_3_sized.json (optimized radii)                           │
│  Process:                                                            │
│    - Optimality Criteria (OC) update with Lagrange multiplier       │
│    - Volume constraint enforcement (target = input solid volume)    │
│    - Move limits on radius per iteration                            │
│    - Convergence in 20-50 iterations                                │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
      FINAL OUTPUT
      {base_name}_3_sized.json → copied to user's --output filename
      {base_name}_history.json (4-stage snapshots for FreeCAD)
```

---

## Stage-by-Stage Breakdown

### Stage 0: Topology Optimization

**Algorithm:** SIMP (Solid Isotropic Material with Penalization)
**Input:** Domain dimensions (nelx × nely × nelz), volfrac, load definition, boundary conditions
**Output:** `top3d.npz` — density grid `rho` + BC tag grid `bc_tags`

**Key Parameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `nelx`, `nely`, `nelz` | 60, 20, 4 (default) | Design domain mesh |
| `volfrac` | 0.3 | Target volume fraction (30% of design space) |
| `penal` | 3.0 | SIMP penalization exponent |
| `rmin` | 1.5 | Filter radius (voxels) |
| `max_loop` | 50-100 | Maximum iterations |
| `E0` | 1.0 | Solid material stiffness |
| `Emin` | 1e-9 | Void material stiffness |
| `ν` (Poisson) | 0.3 | Material property |

**Element Type:** 8-node hexahedral (H8) brick with 24 DOFs per element

**Material Law (SIMP):**
```
E(ρ) = Emin + ρ^p × (E0 - Emin)
```

**Objective Function:**
```
Minimize: C(ρ) = u^T × K(ρ) × u  (compliance)
Subject to: Σ(ρ) / nele ≤ volfrac
            0 < ρ ≤ 1
```

**Sensitivity Analysis:**
```
dC/dρ = -p(E0-Emin)ρ^(p-1) × u_e^T × KE × u_e
```

**Filter (Density-Based):**
```
H[e,f] = max(0, rmin - distance(e,f))
Filtered sensitivity: s̃ = (Σ H×s) / (Σ H)
```

**Optimality Criteria Update:**
```
ρ_new = ρ × √((-dC/dρ) / (λ × dV/dρ))
λ = Lagrange multiplier (bisection to enforce volume)
move_limit = ±0.2 per iteration
Convergence: change < 0.01
```

**BC Tags:**
- `t = 0`: Free element
- `t = 1`: Fixed support region
- `t = 2`: Applied load region

---

### Stage 1: Voxel Reconstruction

**Algorithm:** Yin's Parallel Directional 3D Thinning + Graph Extraction

#### Sub-stage 1a: Voxelization & Thresholding

```python
solid = (rho > vol_thresh)  # Default: 0.3
mesh_bounds = [nelx*pitch, nely*pitch, nelz*pitch]
V_target = np.sum(solid) * pitch^3
```

#### Sub-stage 1b: Medial Axis Thinning

**Algorithm:** Yin's parallel 6-directional sweeping

```
for iteration in [1..max_iters]:
    for direction in [+X, -X, +Y, -Y, +Z, -Z]:
        candidates = find_border_voxels(solid, direction)

        for voxel in candidates:
            if NOT end_voxel AND is_simple_point(voxel):
                if bc_tags[voxel] == 0:  # Not BC-protected
                    DELETE voxel
```

**Key Constraints:**
- **End Voxel:** 1 neighbour in 26-connectivity (skeleton tips)
- **Simple Point:** Removal preserves 26-FG and 6-BG connectivity
- **BC Protection:** Voxels with `bc_tags > 0` are never deleted

**Convergence:** Usually ~10 iterations for domains <100³ voxels

#### Sub-stage 1c: Graph Extraction

**Voxel Classification by Neighbour Count (26-connectivity):**
```
Type 1 (End):    n_neighbors = 1    → skeleton tips
Type 2 (Regular): n_neighbors = 2   → interior segments
Type 3 (Joint):   n_neighbors > 2   → structural nodes
```

**BC Consolidation (Algorithm 4.2):**
```
For each tag value (1, 2):
    - Find connected clusters of tagged voxels
    - Replace with single centroid node
    - Inherit tag to the centroid
    Result: One BC node per tagged cluster
```

**Output:**
```python
nodes = {id: [x,y,z], ...}
edges = [[u, v, length, [intermediate_pts]], ...]
node_tags = {id: 1 or 2}  # 1=fixed, 2=loaded
```

#### Sub-stage 1d: Post-Processing (5 Algorithms)

**1. Edge Collapse:**
- Merge nodes connected by edges shorter than `collapse_thresh` (default 2.0 mm)
- BC tags protect nodes; never merge across different tag values

**2. Branch Pruning:**
- Iteratively remove degree-1 branches (leaf extremities) with cumulative length < `prune_len` (default 2.0 mm)
- Protected: All BC nodes

**3. RDP Polyline Simplification:**
- Ramer-Douglas-Peucker simplification with tolerance `rdp` (default 1.0 mm)
- Reduces intermediate points while preserving shape

**4. Radius Estimation (Two Modes):**

**EDT Mode:**
```
edt = distance_transform_edt(solid, sampling=[pitch]*3)
For each node: radius = edt_value × pitch
```

**Uniform Mode:**
```
radius = sqrt(V_target / (π × Σ edge_lengths))
All edges get same radius (preserves total volume)
```

**5. Extrema Anchoring:**
- Ensures nodes exist at bounding box extremes (min/max on each axis)
- Improves optimization stability

---

### Stage 2: Layout Optimization

**Algorithm:** L-BFGS-B (quasi-Newton, bound-constrained)
**Solver:** SciPy `optimize.minimize`

**Design Variables:** Free node coordinates `x`, `y`, `z`

**Objective:**
```
Minimize: C(x) = u(x)^T × K(x) × u(x)  (compliance)
Subject to: x_0 - limit ≤ x_free ≤ x_0 + limit
            x_fixed = x_0 (BC nodes locked)
```

**FEA Model:** 3D Euler-Bernoulli beam elements

**Section Properties (Circular Cross-Section):**
```
A = π × r²           (area)
I = π × r⁴ / 4       (second moment of inertia)
J = π × r⁴ / 2       (polar moment)
G = E / (2×(1+ν))    (shear modulus)
E = 1000 MPa (default)
ν = 0.3 (Poisson's ratio)
```

**Element Assembly:**
- Local stiffness: 12×12 per beam element (6 DOFs per node)
- Global assembly via rotation matrix transformation
- Sparse matrix solution via `spsolve`

**Parameters:**
```
move_limit = 5.0 mm      (per iteration)
snap_dist = 5.0 mm       (post-optimization node merging)
max_iterations = 50 (L-BFGS-B)
```

**Post-Step:** Union-find node snapping reduces over-connectivity

**Typical Results (150×40×4 cantilever):**
- Compliance: 141,836 → 107,472 (24.2% reduction)
- Volume change: +9.19%
- Max node displacement: 5.80 mm

---

### Stage 3: Size Optimization

**Algorithm:** Optimality Criteria (OC) with Lagrange multiplier bisection

**Design Variables:** Edge radii `r_i`

**Objective:**
```
Minimize: C(r) = u(r)^T × K(r) × u(r)
Subject to: Σ(π × r_i² × L_i) = V_target
            r_min ≤ r_i ≤ r_max
```

**Sensitivities:**
```
dC/dr_i = -u_e^T × (dK_e/dr_i) × u_e
dV/dr_i = 2π × r_i × L_i

dA/dr = 2πr
dI/dr = πr³
dJ/dr = 2πr³
```

**OC Update Rule:**
```
r_new = r_old × [(-dC/dr) / (λ × dV/dr)]^η

where:
  η = 0.5 (exponent, controls update magnitude)
  λ = Lagrange multiplier (bisection: l1=0, l2=1e9)
  move = 0.2 (move limit: 0.8 ≤ r_new/r_old ≤ 1.2)
```

**Parameters:**
```
max_iterations = 50
r_min = 0.1 mm
r_max = 5.0 mm
move_limit = 0.2 (20% per iteration)
convergence = change < 0.001
```

**Convergence:** Typically 20–50 iterations

**Typical Results (150×40×4 cantilever):**
- Compliance: 104,350 → 75,078 (28.1% reduction)
- Volume: 10,639 → 9,985 mm³ (constraint satisfied within 1%)
- Max radius change: 1.38 mm
- Mean radius change: 0.70 mm

---

## Key Mathematical Formulations

### SIMP Material Interpolation

**Power Law:**
```
E(ρ) = Emin + ρ^p × (E0 - Emin)
```

**Physical Interpretation:**
- At ρ=0: E=Emin (void, very soft)
- At ρ=1: E=E0 (solid)
- Intermediate densities penalized by power `p` (typically 3)

### Compliance Minimization

**Strain Energy Formulation:**
```
C = 1/2 × u^T × K × u
```

**Objective:** Minimize C to maximize stiffness (inverse of compliance)

### Optimality Criteria

**General Form:**
```
x_new = x × (∂f/∂x)^(-β) / λ  (for minimize f with volume constraint)
```

**For Topology Opt:**
```
ρ_new = ρ × [(-∂C/∂ρ) / (λ × ∂V/∂ρ)]^(1/2)
```

**For Size Opt:**
```
r_new = r × [(-∂C/∂r) / (λ × ∂V/∂r)]^(1/2)
```

### Sensitivity Filter (SIMP)

**Purpose:** Suppress checkerboard patterns, enforce minimum feature size

**Density Filter:**
```
H_ef = max(0, r_min - ||e - f||)
s̃_e = (Σ_f∈N_e H_ef × s_f) / (Σ_f∈N_e H_ef)
```

---

## LaTeX Technical Document

Complete LaTeX technical writeup is available in the dissertation planning document. Key sections:

1. **Introduction** - Problem motivation and pipeline overview
2. **Stage 0: Topology Optimization** - SIMP formulation, material law, FEA, sensitivities, OC method, BC tags
3. **Stage 1: Voxel Reconstruction** - Voxelization, medial axis thinning (Yin's algorithm), graph extraction, post-processing (5 algorithms)
4. **Stages 2 & 3: Frame Optimization** - Frame FEA model, layout optimization formulation, size optimization formulation
5. **Parameter Summary** - Table of all key parameters with defaults
6. **Results** - Example compliance reduction table
7. **Conclusions** - Key contributions and future work

---

## Critical Files Reference

| File | Purpose |
|------|---------|
| [run_pipeline.py](../run_pipeline.py) | Main entry point, parameter parsing, stage orchestration |
| [src/optimization/top3d.py](../src/optimization/top3d.py) | SIMP topology optimization implementation |
| [src/optimization/fem.py](../src/optimization/fem.py) | 3D frame FEA solver (H8 elements for topology opt, beam elements for frame) |
| [src/pipelines/baseline_yin/thinning.py](../src/pipelines/baseline_yin/thinning.py) | Yin's parallel 3D thinning algorithm |
| [src/pipelines/baseline_yin/graph.py](../src/pipelines/baseline_yin/graph.py) | Graph extraction + BC tag consolidation |
| [src/pipelines/baseline_yin/postprocessing.py](../src/pipelines/baseline_yin/postprocessing.py) | All 5 post-processing algorithms |
| [src/optimization/layout_opt.py](../src/optimization/layout_opt.py) | Layout optimization (L-BFGS-B) |
| [src/optimization/size_opt.py](../src/optimization/size_opt.py) | Size optimization (OC method) |
| [src/problems/tagged_problem.py](../src/problems/tagged_problem.py) | BC handling from node tags |
| [optimization_report.txt](../optimization_report.txt) | Layout opt results |
| [size_opt_report.txt](../size_opt_report.txt) | Size opt results |

---

## Test Cases for Results Section

| Test Case | Domain | volfrac | Purpose |
|-----------|--------|---------|---------|
| Short Cantilever | 60×20×2 | 0.3 | Fast validation, ~3 min |
| **Long Cantilever** | **150×40×4** | **0.3** | **Primary result case** |
| MBB Beam | 120×20×2 | 0.3 | Classic Warren truss pattern |
| Bridge | 80×20×2 | 0.2 | Multi-support handling |

---

## Key Contributions to Highlight

1. **End-to-end Integration:** Unbroken data flow from SIMP density field → parametric CAD-ready JSON

2. **BC Tag Propagation:** Topology-preserving mechanism carries load/support locations through all stages without geometric heuristics

3. **Protected Thinning:** BC-tagged voxels survive medial axis extraction, ensuring load introduction points are preserved

4. **Two-Stage Frame Optimization:** Decoupled layout (geometry) and size (cross-section) optimization leverages different solvers optimally

5. **Parametric Output:** Curves with radii at each point → direct import to FreeCAD or FEA tools

---

## References to Locate

| Citation | Paper |
|----------|-------|
| Bendsøe 1988 | Bendsøe & Kikuchi, "Generating optimal topologies in structural design" |
| Sigmund 2001 | Sigmund, "A 99 line topology optimization code written in Matlab" |
| Yin 1996 | Yin, "A new parallel thinning algorithm for 3D binary images" |
| Zienkiewicz 2000 | Zienkiewicz & Taylor, "The Finite Element Method" (Vol. 1) |
| Douglas & Peucker 1973 | "Algorithms for the reduction of the number of points required to represent a digitized line" |
| Bendsøe 1995 | "Optimization of Structural Topology, Shape, and Material", Springer |

---

## Quick Parameter Tuning Guide

### For Coarse/Smooth Structures:
```
penal = 4.0      (sharper, more binary)
rmin = 2.0-3.0   (minimum feature size)
volfrac = 0.2    (sparser design)
```

### For Fine/Complex Structures:
```
penal = 2.5      (softer, more intermediate densities)
rmin = 1.0-1.5   (smaller features allowed)
volfrac = 0.3-0.5 (more material)
```

### For Reconstruction Cleaning:
```
collapse_thresh = 2-4 mm     (depends on voxel pitch)
prune_len = 2-4 mm           (small branch removal)
rdp = 0.5-1.5 mm             (polyline simplification aggressiveness)
```

### For Optimization Stability:
```
layout move_limit = 5-10 mm  (larger for high-compliance designs)
snap_dist = 5 mm             (merge nodes within this distance)
size move = 0.2              (typically fixed, conservative)
```

---

**Last Updated:** 2026-02-17
**Status:** Dissertation planning phase
