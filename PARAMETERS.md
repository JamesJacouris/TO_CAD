# Pipeline Parameters Reference

Complete reference for all CLI parameters across `run_top3d.py` (Stage 0 standalone) and `run_pipeline.py` (full pipeline).

---

## Quick Start Examples

```bash
# Cantilever beam (150×40×4, full pipeline)
python run_pipeline.py \
  --nelx 150 --nely 40 --nelz 4 \
  --volfrac 0.3 --penal 3.0 --rmin 3.0 --max_loop 100 \
  --load_x 150 --load_y 20 --load_z 2 \
  --load_fy -100.0 \
  --prune_len 4.15 --collapse_thresh 3.84 --rdp 0.78 \
  --radius_mode uniform --limit 5.0 --snap 5.0 \
  --visualize --output matlab_replicated.json

# Roof slab with interior columns (hybrid beam+plate)
python run_pipeline.py \
  --problem roof_slab \
  --nelx 60 --nely 60 --nelz 40 \
  --volfrac 0.12 --penal 3.0 --rmin 2.0 --max_loop 80 \
  --load_fy -150.0 \
  --hybrid --opt_loops 2 \
  --visualize --output roof_slab.json

# Skip Top3D (reuse existing NPZ)
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/matlab_replicated_top3d.npz \
  --prune_len 4.15 --collapse_thresh 3.84 --rdp 0.78 \
  --visualize --output rerun.json
```

---

## run_top3d.py — Standalone Top3D

Used when you want to run topology optimisation only, producing a `.npz` file for later reconstruction.

```bash
python run_top3d.py [OPTIONS]
```

### Domain

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--nelx` | int | 60 | Number of elements along X (length) |
| `--nely` | int | 20 | Number of elements along Y (height) |
| `--nelz` | int | 4 | Number of elements along Z (depth) |
| `--volfrac` | float | 0.3 | Target volume fraction — fraction of domain filled with material (0.0–1.0) |
| `--penal` | float | 3.0 | SIMP penalty exponent. Higher = sharper 0/1 density distribution. Typical: 3.0 |
| `--rmin` | float | 1.5 | Density filter radius in element units. Controls minimum feature size. Typical: 1.5–3.0 |
| `--max_loop` | int | 50 | Maximum OC optimisation iterations. More iterations → finer convergence |

### Problem Type

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--problem` | str | `cantilever` | Boundary condition preset. See [Problem Types](#problem-types) below |

### Load Definition

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--load_x` | int | `nelx` | X index of the load point node |
| `--load_y` | int | `nely` | Y index of the load point node |
| `--load_z` | int | `nelz/2` | Z index of the load point node |
| `--load_fx` | float | 0.0 | Force magnitude in X direction |
| `--load_fy` | float | -1.0 | Force magnitude in Y direction (negative = downward) |
| `--load_fz` | float | 0.0 | Force magnitude in Z direction |
| `--load_dist` | str | `point` | Load distribution mode: `point`, `surface_top`, `surface_bottom` |

**Load distribution modes:**
- `point` — Single node load at `(load_x, load_y, load_z)`
- `surface_top` — Total force divided evenly across all nodes on the top face (`z = nelz`)
- `surface_bottom` — Total force divided evenly across all nodes on the bottom face (`z = 0`)

### Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output` | str | `python_top3d_result.npz` | Output `.npz` file path. Contains `rho`, `bc_tags`, `pitch`, `origin` |

---

## run_pipeline.py — Full Pipeline

Runs all stages: Top3D → Reconstruction → Size Optimisation → Layout Optimisation.

```bash
python run_pipeline.py [OPTIONS]
```

### Top3D Design Domain

Same parameters as `run_top3d.py`. All are passed through to the Stage 0 subprocess.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--nelx` | int | 60 | Elements in X |
| `--nely` | int | 20 | Elements in Y |
| `--nelz` | int | 4 | Elements in Z |
| `--volfrac` | float | 0.3 | Target volume fraction |
| `--penal` | float | 3.0 | SIMP penalty exponent |
| `--rmin` | float | 1.5 | Filter radius (element units) |
| `--max_loop` | int | 50 | Max Top3D iterations |

### Load Definition

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--load_x` | int | `nelx` | Load node X index |
| `--load_y` | int | `nely` | Load node Y index |
| `--load_z` | int | `nelz/2` | Load node Z index |
| `--load_fx` | float | — | Force X component |
| `--load_fy` | float | — | Force Y component (negative = downward) |
| `--load_fz` | float | — | Force Z component |
| `--load_dist` | str | `point` | Load distribution: `point`, `surface_top`, `surface_bottom` |

### Skeletonisation

Controls how the density field from Top3D is converted into a beam skeleton.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pitch` | float | 1.0 | Voxel physical size in mm. Scales all geometry. |
| `--max_iters` | int | 50 | Maximum thinning iterations for Yin's algorithm |
| `--vol_thresh` | float | 0.3 | Density threshold — voxels above this value are treated as solid material |
| `--prune_len` | float | 2.0 | Remove skeleton branches shorter than this value (mm). Reduces noise. Typical: 2.0–5.0 |
| `--collapse_thresh` | float | 2.0 | Collapse graph edges shorter than this value (mm). Merges nearby nodes. Typical: 2.0–4.0 |
| `--rdp` | float | 1.0 | RDP (Ramer-Douglas-Peucker) simplification epsilon (mm). Reduces waypoints on beam edges. 0 = disabled. Typical: 0.5–2.5 |
| `--radius_mode` | str | `uniform` | Beam radius estimation: `uniform` (mean EDT) or `edt` (per-edge EDT samples) |

**Tuning guidance:**
- Increase `--prune_len` and `--collapse_thresh` to get a cleaner, simpler graph
- Decrease them to preserve finer structural detail
- `--rdp 0` keeps all skeleton waypoints; `--rdp 2.0` aggressively simplifies curves

### Hybrid Beam+Plate Mode

Only active when `--hybrid` flag is set.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--hybrid` | flag | off | Enable hybrid beam+plate pipeline |
| `--detect_plates` | str | `auto` | Plate detection mode: `auto`, `off`, `force` |
| `--plate_mode` | str | `bspline` | Plate surface representation: `bspline`, `voxel`, `mesh` |
| `--plate_thickness_ratio` | float | 0.15 | Plate thickness as fraction of domain height |
| `--min_plate_size` | int | 4 | Minimum voxel count for a region to be classified as a plate |
| `--flatness_ratio` | float | 3.0 | PCA eigenvalue ratio threshold. Higher = stricter plate classification |
| `--junction_thresh` | int | 4 | Neighbour count threshold for junction detection |
| `--min_avg_neighbors` | float | 3.0 | Minimum average neighbours for plate classification |

**Plate classification:** The pipeline uses multi-signal scoring (PCA shape, aspect ratio, BC tag density, EDT uniformity) to decide whether each region is a plate or beam. `--flatness_ratio` and `--min_plate_size` are the most impactful parameters to tune.

### Curved Beams (Geometry Only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--curved` | flag | off | Fit cubic Bézier curves to skeleton edges for smooth visualisation. **Does not affect FEM — geometry only.** |

### Optimisation Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--optimize` | flag | off | Enable beam layout and size optimisation (default: on for beam-only, off for hybrid) |
| `--opt_loops` | int | 2 | Number of alternating size+layout optimisation loops |
| `--iters` | int | 50 | Max iterations per optimisation stage |
| `--limit` | float | 5.0 | Node move limit per layout optimisation step (mm) |
| `--snap` | float | 5.0 | Node merge snap distance (mm) — nodes closer than this are merged |
| `--prune_opt_thresh` | float | 0.0 | Post-optimisation edge pruning: remove edges with radius < this fraction of max radius (0 = disabled) |
| `--problem` | str | `tagged` | FEM problem config for optimisation. `tagged` = auto from BC tags. Others: `cantilever`, `roof_slab`, `bridge`, `deck` |

### Output & Visualisation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output` | str | `full_control_beam.json` | Final output JSON filename |
| `--output_dir` | str | `output/hybrid_v2` | Directory where all stage outputs are written |
| `--visualize` | flag | off | Show 3D matplotlib windows at each stage |

### Advanced / Skip Stages

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--skip_top3d` | flag | off | Skip Stage 0 entirely and use an existing `.npz` |
| `--top3d_npz` | str | — | Path to existing `.npz` file when using `--skip_top3d` |

---

## Problem Types

Problems define the **boundary conditions** (fixed supports and loads) for the topology optimisation domain. They are set with `--problem` in both scripts.

### Node Coordinate System

The domain has `(nelx+1) × (nely+1) × (nelz+1)` nodes. Coordinates are indexed from 0:
- **X**: 0 → nelx (left → right / length)
- **Y**: 0 → nely (bottom → top / height)
- **Z**: 0 → nelz (front → back / depth)

All DOFs (degrees of freedom) are fixed in 3 directions (X, Y, Z) at each support node unless stated otherwise.

---

### `cantilever` (default)

A fixed-wall cantilever. The entire left face is clamped, and a point load is applied at the right.

```
Fixed: all nodes where x = 0  (left wall)
Load:  point load at (load_x, load_y, load_z), default (nelx, nely, nelz/2)
```

```
  ████ ─────────────── ←F
  ████
  ████
  (x=0 fixed)                (x=nelx, load point)
```

**Typical use:**
```bash
python run_top3d.py --problem cantilever \
  --nelx 150 --nely 40 --nelz 4 \
  --load_x 150 --load_y 20 --load_z 2 \
  --load_fy -100.0
```

---

### `roof`

A roof slab supported at 4 corner pillars. The 4 corner nodes at z=0 (inset by 1 element from the boundary) are fixed. Load defaults to distributed across the entire top surface.

```
Fixed: 4 corner nodes at (1,1,0), (nx-1,1,0), (1,ny-1,0), (nx-1,ny-1,0)
Load:  surface_top by default (distributed across z=nelz)
```

```
  ↓ ↓ ↓ ↓ ↓ ↓  (top surface, distributed load)
  ┌──────────┐
  │          │
  │          │
  └──────────┘
  *          *   (4 corner supports, z=0)
```

**Typical use:**
```bash
python run_top3d.py --problem roof \
  --nelx 60 --nely 60 --nelz 4 \
  --volfrac 0.15 --load_fy -100.0
```

---

### `roof_slab`

A thin flat slab supported by **9 interior point columns** arranged in a 3×3 grid. Produces a natural plate+beam hybrid decomposition — the slab becomes a plate zone, the columns become beam zones.

```
Fixed: 9 interior nodes at 25%/50%/75% of nelx and nely, at z=0
Load:  surface_top by default (distributed across z=nelz)
```

```
  ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓  (top surface, distributed load)
  ┌─────────────────┐
  │                 │
  │  *    *    *   │  ← 3x3 support columns
  │  *    *    *   │    at 25%, 50%, 75% of X and Y
  │  *    *    *   │
  └─────────────────┘
```

**Typical use:**
```bash
python run_top3d.py --problem roof_slab \
  --nelx 60 --nely 60 --nelz 6 \
  --volfrac 0.12 --load_fy -150.0

# Full hybrid pipeline
python run_pipeline.py --problem roof_slab \
  --nelx 60 --nely 60 --nelz 6 \
  --volfrac 0.12 --load_fy -150.0 \
  --hybrid --output roof_slab.json
```

---

### `bridge`

Both ends of the bottom face are fixed (entire bottom surface, z=0). Suitable for simply-supported structures spanning in X.

```
Fixed: all nodes where z = 0  (entire bottom face)
Load:  point load at (load_x, load_y, load_z)
```

```
  ↓F
  ┌────────────────┐
  │                │
  ████████████████  ← z=0 fully fixed
```

**Typical use:**
```bash
python run_top3d.py --problem bridge \
  --nelx 80 --nely 20 --nelz 2 \
  --load_x 40 --load_y 20 --load_z 1 \
  --load_fy -1.0
```

---

### `deck`

Same fixed supports as `bridge` (entire bottom face). If no load position is specified, the load is automatically split equally across the **4 top corners** at z=nelz — simulating a deck receiving loads at its corners.

```
Fixed: all nodes where z = 0  (entire bottom face)
Load:  4 top corners at (0,0,nz), (nx,0,nz), (0,ny,nz), (nx,ny,nz)  [if no load specified]
```

**Typical use:**
```bash
python run_top3d.py --problem deck \
  --nelx 60 --nely 60 --nelz 10 \
  --volfrac 0.1 --load_fy -100.0
```

---

## Adding a New Problem

To define a custom problem in `run_top3d.py`:

1. Add its name to the `choices` list in the `--problem` argument (line 26).
2. Add an `elif args.problem == "my_problem":` block in the BC definition section.
3. Use the node coordinate arrays `il_flat`, `jl_flat`, `kl_flat` to select nodes:

```python
elif args.problem == "my_problem":
    # Example: fix all nodes on the right face (x = nelx)
    fixed_node_indices = np.where(il_flat == args.nelx)[0]
    fixed_dofs_list = []
    for n in fixed_node_indices:
        fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])  # fix all 3 DOFs
    solver.set_fixed_dofs(np.array(fixed_dofs_list))

    # Set default load distribution if none specified
    if args.load_x is None and args.load_dist == "point":
        args.load_dist = "surface_top"
```

**Node index formula** (F-order):
```
node_id = j + i*(nely+1) + k*(nely+1)*(nelx+1)
```
where `j` = Y index, `i` = X index, `k` = Z index.

To fix only specific DOFs (e.g. Y-direction only):
```python
fixed_dofs_list.append(3*n + 1)   # 0=X, 1=Y, 2=Z
```

---

## Output Files

| File | Description |
|------|-------------|
| `*_top3d.npz` | Raw density field from Top3D. Arrays: `rho` (nely×nelx×nelz), `bc_tags`, `pitch`, `origin` |
| `*_1_reconstructed.json` | Skeleton graph after Stage 1. Contains nodes, edges, radii, bc_tags |
| `*_2_layout.json` | After layout optimisation (node positions updated) |
| `*_3_sized.json` | After size optimisation (radii updated) |
| `*_history.json` | Compliance history across all optimisation loops |
| `*.json` (final) | Final output — use this for FreeCAD reconstruction |
