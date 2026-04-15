# Hybrid Straight/Curved Beam Pipeline

**Date:** March 13, 2026
**Branch:** `Main_V3_Curved/Straight_HYBRID_Optimisation_V1`
**Status:** Implemented, pending testing

---

## Problem

- `--curved` flag was all-or-nothing — every beam got Bezier ctrl_pts regardless of actual curvature
- Short truss members and diagonal braces that are essentially straight were being fitted with curves
- Each curved beam adds 6 extra DOFs to the L-BFGS-B design vector in layout opt — wasted computation
- Curved FEM (24-DOF IGA + static condensation) is ~1.5-2x more expensive than straight (12-DOF Euler-Bernoulli) per element

## Key Finding: FEM + Optimisation Already Support Mixed

- `fem.py:solve_curved_frame()` — checks `ctrl_pts[i] is not None` per edge, straight if None, IGA if present
- `layout_opt.py` — only includes ctrl_pt DOFs for edges where `ctrl_pts[idx] is not None`
- `size_opt.py` — ctrl_pts read-only, handles None entries
- `_extract_ctrl_pts()` in run_pipeline.py — returns None for edges without `"ctrl_pts"` in JSON

No changes needed in any of these files.

## Decision: Where to Classify

Considered 4 insertion points:

| Point | Verdict |
|-------|---------|
| Before thinning | Too early — no edges exist yet |
| After thinning, before graph | No edge/node structure |
| **After graph extraction (Stage 1)** | **Winner** — has skeleton polyline intermediate pts per edge |
| After optimisation | Too late — ctrl_pts already in design vector |

## Metric: Max Perpendicular Deviation

- For each skeleton edge, compute max perpendicular distance of any intermediate waypoint from the straight chord line
- `point_to_line_distance()` already existed in `postprocessing.py` (used by `clean_edge_polylines()`)
- Default threshold: `0.3 * pitch` — below one-third of a voxel = noise, not real curvature
- Also compute arc_ratio (`polyline_length / chord_length`) as secondary metric

## Changes Made

### 1. `postprocessing.py`
- Extracted `_point_to_line_distance()` to module level (was nested inside `clean_edge_polylines()`)
- Added `classify_edge_curvature(p_start, p_end, intermediate_pts, pitch, deviation_thresh)`
- Returns dict: `is_curved`, `max_deviation`, `arc_ratio`, `n_waypoints`

### 2. `reconstruct.py`
- `export_to_json()` now takes `curve_threshold` param
- Per-edge loop: classifies each edge individually instead of global `if curved`
- Curved edges get Bezier fitting + `"ctrl_pts"` in JSON
- Straight edges get simple 2-point representation, no `"ctrl_pts"` key
- Logs summary: `"[Export] Edge classification: 12 curved, 16 straight (threshold=0.30mm)"`
- Wired through `reconstruct_npz()` defaults and CLI argparse

### 3. `run_pipeline.py`
- Added `--curve_threshold` CLI flag (default: None = auto 0.3*pitch)
- Passed to `reconstruct_npz()`
- Fixed `_refit_curves()` — was re-fitting Bezier to straight edges when `ctrl_pts_list[idx]` was None
  - Now: if ctrl_pts_list provided and entry is None, output straight representation

## Usage

```bash
# Default: auto-classify edges (threshold = 0.3 * pitch)
python run_pipeline.py --curved --output result.json

# Stricter — only very curved beams get Bezier
python run_pipeline.py --curved --curve_threshold 1.0 --output result.json

# Legacy behaviour — force all edges curved
python run_pipeline.py --curved --curve_threshold 0 --output result.json
```

## Data Flow

```
Stage 1:
  extract_graph() -> edges with polyline intermediate points
  export_to_json():
    PER EDGE:
      classify_edge_curvature() -> is_curved?
      curved:   fit_cubic_bezier() -> JSON with "ctrl_pts"
      straight: 2-point line       -> JSON without "ctrl_pts"

Stage 2 (Size Opt):
  _extract_ctrl_pts() -> [array or None per edge]  (already worked)
  optimize_size(ctrl_pts=mixed_list)                (already worked)

Stage 3 (Layout Opt):
  _extract_ctrl_pts() -> [array or None per edge]  (already worked)
  optimize_layout(ctrl_pts=mixed_list)              (already worked)
    design vector: [nodes | ctrl_pts for CURVED edges only]
```

## Bug Fixed

`_refit_curves()` previously had a logic error:
- When `ctrl_pts_list[idx]` was None, it fell through to `fit_cubic_bezier()` with empty points
- This would re-curve edges that Stage 1 had classified as straight
- Fixed: explicit check for None -> output straight representation

## Still TODO

- Run on roof test case to verify classification percentages
- Compare compliance: hybrid vs all-curved should be very close
- Benchmark: measure wall-clock speedup from fewer ctrl_pt DOFs in layout opt
- Consider post-fit quality check (Bezier bulge ratio) as secondary filter
