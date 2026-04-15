# Post-Thinning Beam-Plate Classification

## Overview

After topology optimisation produces a density field, the hybrid pipeline must separate **plate** regions (2D surface structures) from **beam** regions (1D skeletal members). This document describes the two-signal post-thinning classification method implemented in `classify_skeleton_post_thinning()`.

The classifier operates on the **skeleton** (output of 3D thinning), not the raw density field. It combines two independent signals to identify plate voxels, then applies region-level filtering and a boundary growth pass.

---

## 1. Two-Pass Thinning

The solid volume is thinned twice using Yin's parallel thinning algorithm with different preservation modes:

| Pass | Mode | Preserves | Result |
|------|------|-----------|--------|
| 1 | `mode=3` | Surface points AND curve endpoints | Medial surfaces (1-voxel sheets) + medial curves |
| 2 | `mode=0` | Curve endpoints only | Medial curves only (surfaces collapsed) |

**Key insight**: Voxels present in the mode=3 skeleton but absent from the mode=0 skeleton are **plate interior voxels** -- they are the surface sheets that curve-preserving thinning collapsed but surface-preserving thinning retained.

```
reconstruct.py (lines 339-381):

  skeleton       = thin_grid_yin(solid, mode=3)   # surfaces + curves
  skeleton_curve = thin_grid_yin(solid, mode=0)   # curves only

  Signal A = (skeleton > 0) AND (skeleton_curve == 0)
```

### 1.1 Limitation: Thin Structures

For structures where members are only 1-2 voxels thick (common at low volume fractions, < 5%), both thinning modes converge to the same skeleton because the members are already at skeletal thickness. In these cases `Signal A` produces zero or very few candidates, and `Signal B` (below) compensates.

---

## 2. Octant Plane Pattern Analysis

### 2.1 Background: Yin Definition 3.14

A voxel is a **surface point** if every 2x2x2 octant of its 3x3x3 neighbourhood satisfies one of two conditions:

1. The filled voxels form one of the **12 plane configurations**, or
2. Fewer than 3 voxels are filled (`n3 < 3`)

The 8 voxels in a 2x2x2 octant are encoded as an 8-bit configuration byte where bit `i = dz*4 + dy*2 + dx`:

```
Position    Bit index
(0,0,0)  ->  0        (1,0,0)  ->  4
(0,0,1)  ->  1        (1,0,1)  ->  5
(0,1,0)  ->  2        (1,1,0)  ->  6
(0,1,1)  ->  3        (1,1,1)  ->  7
```

### 2.2 The 12 Plane Patterns

There are 6 planes through a 2x2x2 cube (3 axis-aligned + 3 diagonal), each with 2 orientations (a pattern and its bitwise complement), giving 12 total:

| Plane | Pattern | Complement | Description |
|-------|---------|------------|-------------|
| z=0   | `0x0F` | `0xF0` | Horizontal bottom / top face |
| y=0   | `0x33` | `0xCC` | Front / back face |
| x=0   | `0x55` | `0xAA` | Left / right face |
| dz==dy | `0xC3` | `0x3C` | Diagonal plane |
| dy==dx | `0x99` | `0x66` | Diagonal plane |
| dz==dx | `0xA5` | `0x5A` | Diagonal plane |

**Bug fix (Mar 2026)**: The original implementation only included the 6 primary patterns, omitting the 6 complements. This caused `is_surface_point()` to fail for surface sheets where the filled half of an octant fell on the complement side (e.g., config `0xF0` for a sheet viewed from below). The fix doubled the pattern set to all 12, enabling mode=3 thinning to correctly preserve medial surfaces.

### 2.3 `count_plane_octants()` vs `is_surface_point()`

In the **solid** volume, `is_surface_point()` works correctly because octants are densely filled -- the `n3 < 3` fallback rarely triggers, and the plane pattern check genuinely discriminates surface from non-surface voxels.

In the **skeleton** (post-thinning), the volume is sparse. Most octants have `n3 < 3` simply because there are few voxels nearby, not because the voxel is on a surface. This makes `is_surface_point()` trivially true for ~79% of skeleton voxels -- unusable as a classifier.

`count_plane_octants()` solves this by counting only octants with **genuine plane pattern matches**, ignoring the `n3 < 3` fallback entirely:

```python
# is_surface_point():  passes if ALL octants satisfy (plane_pattern OR n3 < 3)
#   -> 79% of skeleton voxels pass (too permissive)

# count_plane_octants(): counts octants matching plane_pattern ONLY
#   -> bimodal distribution separates beams from surfaces
```

### 2.4 Bimodal Separation

Empirical analysis on the Roof test case (513 skeleton voxels) shows a clean bimodal distribution:

| n_plane | nc range | Count | Identity |
|---------|----------|-------|----------|
| 0       | 1-4      | 285   | Pure beam voxels (all octants pass only via n3 < 3) |
| 1-3     | 3-6      | 52    | Beam junctions / surface edges |
| **4-8** | **5-9**  | **176** | **Surface interior** (genuine plane patterns in in-plane octants) |

The threshold `n_plane >= 4` cleanly separates surface interior voxels from beam voxels with zero overlap in the `(nc, n_plane)` space.

---

## 3. Combined Classification

### 3.1 Signal Combination

```
Signal A:  Two-pass difference (mode=3 AND NOT mode=0)
           -> Reliable for thick structures (>= 3 voxel member thickness)
           -> Empty/small for thin structures

Signal B:  count_plane_octants(skeleton_neighbourhood) >= 4  AND  nc >= 3
           -> Reliable for both thick and thin structures
           -> Catches surfaces that both thinning modes preserve identically

plate_candidates = Signal_A  OR  Signal_B
```

### 3.2 Region Filtering

Connected components of `plate_candidates` (26-connectivity) are filtered by:

1. **Minimum size**: Regions with fewer than `min_plate_size` voxels (default 3) are discarded.

2. **Global linearity check**: PCA eigenvalues of the region coordinates are computed. If the linearity ratio `lambda_max / lambda_mid > flatness_ratio * 3` (default 9.0), the region is an elongated chain (beam cross-section remnant), not a surface, and is rejected.

### 3.3 Edge Growth Pass

Surface interior voxels (identified by `n_plane >= 4`) miss boundary voxels of the plate, which have fewer in-plane neighbours and thus fewer plane octants (typically `n_plane = 2-3`).

A single-pass growth step captures these edge voxels:

```
For each unclassified skeleton voxel:
    IF  nc >= 2                          (not an isolated endpoint)
    AND count_plane_octants >= 2         (has some surface character)
    AND has at least one plate neighbour (26-connectivity)
    THEN assign to the adjacent plate region
```

The `n_plane >= 2` requirement prevents pure beam voxels (`n_plane = 0`) at plate-beam junctions from being pulled into plate regions. The single-pass (non-iterative) design limits growth to exactly 1 voxel from the confirmed plate boundary.

---

## 4. Pipeline Integration

```
[1] Topology Optimisation (Top3D)
    -> density field rho

[2] Thresholding (rho > vol_thresh)
    -> binary solid volume

[3] Two-pass thinning
    Pass 1: mode=3 -> skeleton (surfaces + curves)
    Pass 2: mode=0 -> skeleton_curve (curves only)

[4] Post-thinning classification
    Signal A: set difference (mode=3 minus mode=0)
    Signal B: count_plane_octants >= 4 on mode=3 skeleton
    Combined -> connected components -> size/linearity filter -> edge growth
    -> zone_mask: 1=plate, 2=beam

[5a] Plate extraction (zone=1 regions)
    -> solid mesh + mid-surface + per-vertex thickness

[5b] Beam graph extraction (zone=2 voxels)
    -> nodes, edges, cross-sections

[6] Joint creation
    -> beam endpoints snapped to plate boundaries
```

---

## 5. Implementation Reference

| Component | File | Function |
|-----------|------|----------|
| Octant config encoding | `topology.py` | `_get_octant_config()` |
| Surface point test (thinning) | `topology.py` | `is_surface_point()` |
| Strict plane octant count | `topology.py` | `count_plane_octants()` |
| Two-pass classification | `graph.py` | `classify_skeleton_post_thinning()` |
| PCA fallback (legacy) | `graph.py` | `_classify_pca_fallback()` |
| Pipeline orchestration | `reconstruct.py` | `reconstruct()` lines 339-381 |

---

## 6. Validation Results

| Test Case | Solid | mode=3 | mode=0 | Signal A | Signal B | Plates | Beams |
|-----------|-------|--------|--------|----------|----------|--------|-------|
| Roof (40x40x20, 4.7% fill) | 1,500 | 513 | 507 | 28 | 176 | 200 (5 regions) | 313 |
| Wall Bracket (30x60x30, 12% fill) | 6,388 | 513 | 424 | 155 | 19 | 54 (7 regions) | 459 |
| Cantilever (150x40x4, 43% fill) | 10,252 | 575 | 527 | 131 | -- | 78 (7 regions) | 497 |

**Observations**:
- Signal A dominates for denser structures (Wall Bracket: 155 diff voxels, only 19 from topo).
- Signal B dominates for sparser structures (Roof: 28 diff voxels, 176 from topo).
- Together they cover the full range of volume fractions encountered in topology optimisation output.
