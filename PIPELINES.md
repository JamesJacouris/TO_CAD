# TO_CAD Pipelines: Complete Command-Line Reference

## Overview

Three main branches, each with different pipelines:

| Branch | Purpose | Primary Features |
|--------|---------|-----------------|
| `Top3D_V2_Iterative_Layout_and_Size_Optimisation` | **Current** — Iterative layout & size optimization | Multi-loop size→layout cycles, full FEM analysis |
| `Hybrid_With_Curved_Beams_V2` | Hybrid beam-plate reconstruction | Zone classification, dual thinning, curved beams, Bézier geometry |
| `feature/freecad-surface-reconstruction` | Hybrid beam-plate surfaces | Same as above, plus mid-surface extraction (in progress) |

---

## 1. Iterative Layout & Size Optimization Pipeline
**Current branch:** `Top3D_V2_Iterative_Layout_and_Size_Optimisation`

This pipeline performs **2 optimization stages** in alternating loops: **Size → Layout → Size → Layout → ...**

### Input Arguments

#### Top3D Domain (Stage 0)
```
--nelx, --nely, --nelz          Domain grid resolution (default: 60, 20, 4)
--volfrac                       Target material fraction (default: 0.3)
--penal                         Penalization exponent (default: 3.0)
--rmin                          Filter radius in voxels (default: 1.5)
--max_loop                      Max Top3D iterations (default: 50)
```

#### Load & Boundary Conditions
```
--load_x, --load_y, --load_z    Load application node (default: nelx, nely, nelz/2)
--load_fx, --load_fy, --load_fz Load force vector (default: 0, -1, 0)
--problem                       BC problem type: 'tagged' (auto), 'cantilever', 'rocker_arm' (default: tagged)
```

#### Skeletonization (Stage 1: Reconstruction)
```
--pitch                         Voxel size in mm (default: 1.0)
--max_iters                     Thinning iterations (default: 50)
--collapse_thresh               Collapse short edges < X mm (default: 2.0)
--prune_len                     Prune branches < X mm (default: 2.0)
--rdp                           RDP simplification epsilon, 0=off (default: 1.0)
--radius_mode                   Radius strategy: 'edt' or 'uniform' (default: uniform)
--vol_thresh                    Density threshold for binary conversion (default: 0.3)
```

#### Optimization (Stages 2 & 3: Layout + Size Loops)
```
--limit                         Layout move limit in mm (default: 5.0)
--snap                          Node merge distance in mm (default: 5.0)
--iters                         Size optimization iterations (default: 50)
--opt_loops                     Number of alternating Size/Layout cycles (default: 1)
```

#### Output & Visualization
```
--output                        Final JSON filename (default: full_control_beam.json)
--output_dir                    Output directory (default: output/hybrid_v2)
--visualize                     Show 3D debug windows (default: false)
--skip_top3d                    Skip Stage 0, reuse existing .npz (default: false)
--top3d_npz                     Path to existing .npz file (if --skip_top3d used)
```

### Example Commands

**Basic run** (single optimization loop):
```bash
python run_pipeline.py \
  --nelx 60 --nely 20 --nelz 4 \
  --volfrac 0.3 \
  --output cantilever_opt.json
```

**Cantilever with custom BC locations and forces**:
```bash
python run_pipeline.py \
  --nelx 60 --nely 20 --nelz 4 \
  --volfrac 0.3 \
  --load_x 60 --load_y 10 --load_z 2 \
  --load_fx 0.0 --load_fy -1.0 --load_fz 0.0 \
  --problem cantilever \
  --output my_cantilever.json
```

**Fine-tuned reconstruction + 2 optimization loops**:
```bash
python run_pipeline.py \
  --nelx 80 --nely 30 --nelz 6 \
  --volfrac 0.2 \
  --pitch 0.5 \
  --collapse_thresh 1.5 \
  --prune_len 1.5 \
  --rdp 0.5 \
  --radius_mode edt \
  --limit 3.0 --snap 3.0 \
  --opt_loops 2 \
  --output fine_tuned_2loop.json \
  --visualize
```

**Reuse existing Top3D result (fast iteration)**:
```bash
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/my_design_top3d.npz \
  --collapse_thresh 1.0 \
  --prune_len 1.0 \
  --opt_loops 3 \
  --output my_design_opt_loop3.json
```

**Rocker arm problem with full FEM output**:
```bash
python run_pipeline.py \
  --nelx 40 --nely 20 --nelz 4 \
  --volfrac 0.25 \
  --problem rocker_arm \
  --iters 100 \
  --opt_loops 5 \
  --limit 5.0 --snap 5.0 \
  --output rocker_arm_final.json \
  --visualize
```

---

## 2. Hybrid Beam-Plate with Curved Beams Pipeline
**Branch:** `Hybrid_With_Curved_Beams_V2`

This pipeline produces **hybrid structures** with **plates** (extruded surfaces) and **curved beams** (Bézier solids). No optimization by default; option to add with `--optimize`.

### Input Arguments

#### Design Domain
```
--nelx, --nely, --nelz          Domain resolution (default: 60, 20, 4)
--volfrac                       Material fraction (default: 0.3)
--penal, --rmin, --max_loop     Top3D settings (defaults: 3.0, 1.5, 50)
```

#### Load & Boundary Conditions
```
--load_x, --load_y, --load_z    Load node position
--load_fx, --load_fy, --load_fz Load force vector (default: 0, -1, 0)
--problem                       BC type: 'tagged', 'cantilever', 'rocker_arm' (default: tagged)
```

#### Skeletonization (Stage 1)
```
--pitch                         Voxel size (default: 1.0)
--collapse_thresh               Short edge collapse < X mm (default: 2.0)
--prune_len                     Branch prune threshold < X mm (default: 2.0)
--rdp                           RDP epsilon (0=off, default: 1.0)
--radius_mode                   'edt' or 'uniform' (default: uniform)
--vol_thresh                    Density threshold (default: 0.3)
--max_iters                     Thinning iterations (default: 50)
```

#### Hybrid Specific (NEW)
```
--hybrid                        Enable zone classification + dual thinning (default: false)
  [Creates both plates AND beams; adds 'plates' key to JSON]

--curved                        Fit cubic Bézier curves to beam paths (default: false)
  [Enables smooth curved beam visualization; creates 'ctrl_pts' in JSON curves]
```

#### Optimization (Stages 2 & 3) — OPTIONAL
```
--optimize                      Enable layout + size optimization stages (default: off for --hybrid)
  [Required to run Stages 2 & 3 on hybrid designs]

--opt_loops                     Number of optimization cycles (default: 1)
--limit                         Layout move limit in mm (default: 5.0)
--snap                          Node merge distance (default: 5.0)
--iters                         Size opt iterations (default: 50)
```

#### Output & Visualization
```
--output                        Output JSON filename (default: output.json)
--output_dir                    Output directory (default: output/hybrid_v2)
--visualize                     Show 3D debug windows (default: false)
--skip_top3d                    Reuse existing .npz (default: false)
--top3d_npz                     Path to .npz file
```

### Example Commands

**Hybrid structure** (plates + straight beams, NO optimization):
```bash
python run_pipeline.py \
  --nelx 60 --nely 30 --nelz 10 \
  --volfrac 0.1 \
  --problem roof \
  --hybrid \
  --output hybrid_roof.json
```

**Hybrid with curved beams** (smooth visualization, NO optimization):
```bash
python run_pipeline.py \
  --nelx 60 --nely 30 --nelz 10 \
  --volfrac 0.1 \
  --hybrid --curved \
  --output hybrid_roof_curved.json \
  --visualize
```

**Hybrid with curved beams + optimization** (2 layout/size loops):
```bash
python run_pipeline.py \
  --nelx 60 --nely 30 --nelz 10 \
  --volfrac 0.1 \
  --hybrid --curved --optimize \
  --opt_loops 2 \
  --limit 3.0 --snap 3.0 \
  --output hybrid_roof_curved_opt.json
```

**Reuse previous Top3D + new reconstruction settings**:
```bash
python run_pipeline.py \
  --skip_top3d \
  --top3d_npz output/hybrid_v2/roof_design_top3d.npz \
  --hybrid --curved \
  --collapse_thresh 1.0 \
  --prune_len 1.0 \
  --output roof_refined.json
```

**Pure curved beams** (no plates, with optimization):
```bash
python run_pipeline.py \
  --nelx 40 --nely 40 --nelz 20 \
  --volfrac 0.05 \
  --curved \
  --optimize --opt_loops 2 \
  --output pure_curved_beams_opt.json
```

---

## 3. FreeCAD Macro Usage

### Running the Simple Beam-Only Macro (Current Branch)

This branch uses `freecad_import_simple.py` — an ultra-minimal, rock-solid macro optimized for iterative optimization output.

1. **Open FreeCAD** and load the macro:
   - **Macro → Load Macro → `src/export/freecad_import_simple.py`**

2. **Run the macro** — it will prompt you to:
   - Select the JSON file output from the pipeline
   - Creates red spheres for nodes, green lines for edges

3. **What it renders**:
   - **Node spheres** (red) — sized by radius
   - **Edge lines** (green) — connecting edges
   - Updates every 20 nodes and 50 edges (no freezing)
   - Auto-fits view to geometry

### Macro Features
- ✅ Ultra-simple geometry (no complex CSG)
- ✅ Zero crash risk — uses only basic FreeCAD objects
- ✅ Fast rendering even with 1000+ nodes
- ✅ Clean, readable output (Nodes group + Edges group)
- ✅ Perfect for iterative optimization output

### Other Branches (Hybrid Macros)

If you switch to `Hybrid_With_Curved_Beams_V2` or `feature/freecad-surface-reconstruction`:
- Use `src/export/freecad_reconstruct_stable_WORKING.py` (if present on that branch)
- Supports plates, curved beams, CSG operations, history playback
- More complex but full-featured for hybrid structures

---

## 4. Quick Comparison

| Feature | Iterative Opt | Hybrid (Straight) | Hybrid (Curved) |
|---------|---|---|---|
| **Stages** | 0→1→2→3 (size loop→layout loop) | 0→1 (no opt by default) | 0→1 (no opt by default) |
| **Geometry** | Pure frame (beams only) | Plates + beams | Plates + curved beams |
| **FEM** | Full 3D frame analysis | N/A (Stage 1 reconstruction only) | N/A |
| **Optimization** | Full (layout + size loops) | Optional (`--optimize`) | Optional (`--optimize`) |
| **FreeCAD render** | Ball-and-stick | Extruded surfaces + B&S | Extruded surfaces + Bézier solids |
| **Default run** | With optimization | Without optimization | Without optimization |

---

## 5. Troubleshooting

### "FreeCAD macro not found"
- **Current branch** (`Top3D_V2_Iterative_Layout_and_Size_Optimisation`): Uses `src/export/freecad_import_simple.py` (beam-only, ultra-reliable)
- **Other branches** (`Hybrid_With_Curved_Beams_V2`, `feature/freecad-surface-reconstruction`): Use `src/export/freecad_reconstruct_stable_WORKING.py` (hybrid-capable)
- If missing: Switch to the branch first, then macro will be available

### Hybrid pipeline output is missing `plates` key
- Ensure `--hybrid` flag is set
- Zone classification must pass (automatic; outputs cyan/red voxel regions)
- Check `output/hybrid_v2/[name]_zones.png` for visual verification

### Curved beams not appearing in FreeCAD
- Verify `--curved` flag was used
- Check JSON file contains `"ctrl_pts"` in curves array
- FreeCAD macro reads `ctrl_pts` and switches to `create_bezier_beam_sweep()` automatically

### Optimization produces NaN compliance
- Domain too small relative to --limit and --snap
- Try: `--limit 2.0 --snap 2.0` for smaller domains
- Increase --collapse_thresh to simplify topology first

---

## 6. File Outputs

All pipelines save to `--output_dir` (default: `output/hybrid_v2/`):

- `[name].json` — Final graph structure (main output)
- `[name]_1_reconstructed.json` — Stage 1 (skeletonization)
- `[name]_2_layout_loop*.json` — Stage 2 iterations
- `[name]_3_sized_loop*.json` — Stage 3 iterations
- `[name]_zones.png` — Zone classification (hybrid only)
- `[name]_history.json` — Full history snapshots

---

## 7. Performance Tips

1. **First run:** Use `--skip_top3d --top3d_npz [existing]` to skip Stage 0 (takes 2–5 min)
2. **Large domains:** Reduce `--collapse_thresh` and `--prune_len` to simplify early
3. **Many optimization loops:** Use `--limit 10.0` for larger moves; reduces iterations
4. **Memory:** Hybrid plate extraction can use 2–4 GB on 100×100×50 domains; reduce `--nelx/y/z` if needed
5. **FreeCAD rendering:** Curved Bézier sweeps are slower than ball-and-stick; use if visual quality matters

---

## Questions?

Each branch has its own git history. To switch:
```bash
git checkout Hybrid_With_Curved_Beams_V2
git checkout Top3D_V2_Iterative_Layout_and_Size_Optimisation
git checkout feature/freecad-surface-reconstruction
```

All three branches share the same `run_pipeline.py` interface but with different Stage 0–3 implementations.
