"""Skeleton-to-graph extraction and post-thinning zone classification.

This module converts a medial-axis skeleton (output of :mod:`thinning`) into
a graph of beam nodes and edges, and classifies skeleton voxels into beam
vs. plate zones for the hybrid pipeline.

Main entry points
-----------------
- :func:`extract_graph` — skeleton → ``(nodes, edges, v_types, node_tags)``
- :func:`classify_skeleton_post_thinning` — zone classification for hybrid mode
"""
import numpy as np
from numba import njit, prange
from scipy.ndimage import label, center_of_mass, binary_dilation
from scipy.spatial import KDTree
from src.pipelines.baseline_yin.topology import (
    get_neighbor_offsets, count_neighbors, is_surface_point, is_surface_point_relaxed, is_surface_boundary,
    get_neighborhood_window, count_plane_octants
)

def draw_line_3d(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)
    dist = np.linalg.norm(p2 - p1)
    num_points = int(np.ceil(dist * 1.5)) + 1
    if num_points < 2: return [tuple(map(int, p1))]
    t = np.linspace(0, 1, num_points)
    points = []
    for i in range(num_points):
        pt = p1 + t[i] * (p2 - p1)
        points.append(tuple(np.round(pt).astype(int)))
    return points

def _thin_ring_cluster(cluster_mask, neighbor_coords, full_shape):
    """Thin a ring-shaped tagged cluster to its 1-voxel medial *curve*.

    Uses Yin mode=0 (curve-preserving) thinning on a sub-volume that
    contains the cluster voxels and the immediate skeleton neighbours
    (tagged, hence protected from deletion).  Mode=0 thins to 1-D
    curves rather than 2-D medial surfaces, which avoids nested-square
    artefacts for flat rings.

    After thinning, bridge lines are drawn to any neighbour skeleton
    voxels that lost 26-adjacency due to the ring shrinking inward.

    Returns a list of ``(z, y, x)`` tuples, or ``None`` on failure.
    """
    from src.pipelines.baseline_yin.thinning import thin_grid_yin

    coords = np.argwhere(cluster_mask)
    if len(coords) < 4:
        return None  # too small for a meaningful ring

    # Extract bounding box with padding for 3×3×3 neighbourhood context
    pad = 2
    mins = np.maximum(coords.min(axis=0) - pad, 0)
    maxs = np.minimum(coords.max(axis=0) + pad + 1, np.array(full_shape))

    # Build sub-volume: cluster voxels + tagged neighbour context
    sub_cluster = cluster_mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    sub_vol = sub_cluster.astype(np.uint8).copy()
    sub_tags = np.zeros_like(sub_vol, dtype=np.int32)

    for n_idx in neighbor_coords:
        lz = int(n_idx[0]) - mins[0]
        ly = int(n_idx[1]) - mins[1]
        lx = int(n_idx[2]) - mins[2]
        if (0 <= lz < sub_vol.shape[0] and 0 <= ly < sub_vol.shape[1]
                and 0 <= lx < sub_vol.shape[2]):
            sub_vol[lz, ly, lx] = 1
            sub_tags[lz, ly, lx] = 1  # protect from deletion

    # Curve-preserving thinning (mode=0) — reduces to 1-D skeleton
    thin_grid_yin(sub_vol, tags=sub_tags, max_iters=50, mode=0)

    # Collect surviving cluster voxels (exclude tagged neighbours)
    survived = np.argwhere((sub_vol > 0) & (sub_tags == 0))
    if len(survived) == 0:
        return None

    result = [(int(z + mins[0]), int(y + mins[1]), int(x + mins[2]))
              for z, y, x in survived]
    ring_set = set(result)

    # Bridge to any neighbour that lost 26-adjacency after thinning
    for n_idx in neighbor_coords:
        nz, ny, nx = int(n_idx[0]), int(n_idx[1]), int(n_idx[2])
        if any((nz + dz, ny + dy, nx + dx) in ring_set
               for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) != (0, 0, 0)):
            continue  # already adjacent
        # Draw a short bridge from the nearest surviving voxel
        dists = [abs(rz - nz) + abs(ry - ny) + abs(rx - nx)
                 for rz, ry, rx in result]
        nearest = result[int(np.argmin(dists))]
        for pt in draw_line_3d(nearest, (nz, ny, nx)):
            if pt not in ring_set and pt != (nz, ny, nx):
                pz, py, px = pt
                if (0 <= pz < full_shape[0] and 0 <= py < full_shape[1]
                        and 0 <= px < full_shape[2]):
                    result.append(pt)
                    ring_set.add(pt)

    return result


def consolidate_tagged_voxels(skeleton, tags, consolidate_values=None):
    """Collapse tagged voxel clusters to single centroid nodes.

    Parameters
    ----------
    consolidate_values : list of int or None
        Which tag values to consolidate.  ``None`` (default) consolidates
        all positive tags.  Pass e.g. ``[1]`` to only consolidate support
        clusters while leaving load-tagged voxels as regular skeleton.
    """
    skel, new_tags, centroids = skeleton.copy(), np.zeros_like(tags), []
    unique_tags = np.unique(tags)
    unique_tags = unique_tags[unique_tags > 0]
    # Copy through tags that won't be consolidated (so they appear in new_tags
    # for node BC assignment during graph extraction)
    if consolidate_values is not None:
        for t_val in unique_tags:
            if t_val not in consolidate_values:
                new_tags[(tags == t_val) & (skel > 0)] = t_val
        unique_tags = np.array([t for t in unique_tags if t in consolidate_values])
    s26 = np.ones((3,3,3), dtype=np.int32)
    for t_val in unique_tags:
        mask = (tags == t_val) & (skel > 0)
        if not np.any(mask): continue
        labeled, num_features = label(mask, structure=s26)
        centers = center_of_mass(mask, labeled, range(1, num_features+1))
        for i, center in enumerate(centers):
            cluster_mask = (labeled == (i + 1))
            dilated = binary_dilation(cluster_mask, structure=s26)
            neighbor_indices = np.argwhere(dilated & (skel > 0) & (~cluster_mask))
            cz, cy, cx = map(int, np.round(center))
            cz, cy, cx = np.clip([cz, cy, cx], [0,0,0], [skel.shape[0]-1, skel.shape[1]-1, skel.shape[2]-1])

            # Ring detection: centroid outside cluster → hollow shape
            if not cluster_mask[cz, cy, cx]:
                n_cluster = int(np.sum(cluster_mask))
                thinned = _thin_ring_cluster(cluster_mask, neighbor_indices,
                                             skel.shape)
                if thinned is not None:
                    skel[cluster_mask] = 0
                    for vz, vy, vx in thinned:
                        skel[vz, vy, vx] = 1
                        new_tags[vz, vy, vx] = t_val
                    print(f"  Ring cluster (tag={t_val}): {n_cluster} → "
                          f"{len(thinned)} voxels (skeletonized)")
                    continue  # skip centroid creation

            centroid_coord = (cz, cy, cx)
            skel[cluster_mask], skel[cz, cy, cx], new_tags[cz, cy, cx] = 0, 1, t_val
            centroids.append(centroid_coord)
            for n_idx in neighbor_indices:
                for lpt in draw_line_3d(centroid_coord, tuple(n_idx)):
                    lz, ly, lx = lpt
                    if 0 <= lz < skel.shape[0] and 0 <= ly < skel.shape[1] and 0 <= lx < skel.shape[2]:
                        skel[lz, ly, lx] = 1
    return skel, new_tags, centroids

@njit(parallel=True)
def classify_voxels(skeleton, tags=None):
    D, H, W = skeleton.shape
    types, offsets = np.zeros_like(skeleton, dtype=np.int8), get_neighbor_offsets()
    for z in prange(D):
        for y in range(H):
            for x in range(W):
                if skeleton[z, y, x] == 0: continue
                count = 0
                for off in offsets:
                    nz, ny, nx = z+off[0], y+off[1], x+off[2]
                    if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                        if skeleton[nz, ny, nx] > 0: count += 1
                types[z, y, x] = 1 if count in [1, 0] else (2 if count == 2 else 3)
    return types

@njit(parallel=True)
def _compute_surface_mask(skeleton):
    D, H, W = skeleton.shape
    surf = np.zeros((D, H, W), dtype=np.int8)
    for z in prange(D):
        for y in range(H):
            for x in range(W):
                if skeleton[z, y, x] > 0 and is_surface_point(get_neighborhood_window(skeleton, z, y, x)):
                    surf[z, y, x] = 1
    return surf

@njit(parallel=True)
def _classify_surface_boundaries(types, skeleton, surf_mask):
    D, H, W = skeleton.shape
    for z in prange(D):
        for y in range(H):
            for x in range(W):
                if surf_mask[z, y, x] > 0:
                    types[z, y, x] = 5 if is_surface_boundary(z, y, x, skeleton, surf_mask) else 4

def classify_voxels_hybrid(skeleton, tags=None):
    types = classify_voxels(skeleton, tags)
    surf_mask = _compute_surface_mask(skeleton)
    _classify_surface_boundaries(types, skeleton, surf_mask)
    return types

@njit(parallel=True)
def _classify_skeleton_surface_vs_curve(skeleton):
    """Classify each skeleton voxel as surface (plate=1) or curve (beam=2).
    
    NOTE: This per-voxel octant approach is kept for reference but is NO LONGER
    the primary classification method. See classify_skeleton_post_thinning()
    which uses per-cluster spatial extent analysis instead.
    """
    D, H, W = skeleton.shape
    zone = np.zeros((D, H, W), dtype=np.int8)
    for z in prange(D):
        for y in range(H):
            for x in range(W):
                if skeleton[z, y, x] == 0:
                    continue
                hood = get_neighborhood_window(skeleton, z, y, x)
                if is_surface_point(hood):
                    zone[z, y, x] = 1  # plate
                else:
                    zone[z, y, x] = 2  # beam
    return zone


def _classify_cluster_by_extent(coords, min_plate_size=8, flatness_ratio=3.0):
    """
    Classify a single skeleton cluster as plate or beam using PCA extent.

    A cluster is plate-like if it spans 2+ dimensions much more than the 3rd
    (the through-thickness direction). A cluster is beam-like if it is 
    elongated mainly in 1 dimension.

    Parameters
    ----------
    coords : ndarray (N, 3)
        Voxel coordinates of the cluster.
    min_plate_size : int
        Minimum voxels for a cluster to be considered plate.
    flatness_ratio : float
        Ratio of 2nd-largest to smallest eigenvalue needed for "plate".

    Returns
    -------
    'plate' or 'beam'
    """
    n = len(coords)
    if n < min_plate_size:
        return 'beam'  # too small for plate

    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)

    # Guard against degenerate covariance 
    if cov.ndim != 2 or cov.shape != (3, 3):
        return 'beam'

    eigenvalues = np.sort(np.linalg.eigvalsh(cov))  # ascending: λ0 ≤ λ1 ≤ λ2
    # Add small epsilon to prevent division by zero
    eps = 1e-6
    lam0, lam1, lam2 = eigenvalues[0] + eps, eigenvalues[1] + eps, eigenvalues[2] + eps

    # Flatness: plate-like if thickness dimension (λ0) is much smaller than
    # both planar dimensions (λ1, λ2)
    # Linearity: beam-like if one dimension (λ2) dominates the other (λ1)
    flatness = lam1 / lam0   # high → flat
    linearity = lam2 / lam1  # high → elongated

    if flatness >= flatness_ratio and linearity < flatness_ratio:
        return 'plate'
    elif linearity >= flatness_ratio:
        return 'beam'
    else:
        # Ambiguous — use size heuristic: larger clusters are more likely plates
        return 'plate' if n >= min_plate_size * 4 else 'beam'


@njit(parallel=True)
def _count_neighbors_layered(skeleton):
    """
    Count neighbors for each voxel, separated by layer.
    
    Returns
    -------
    in_plane_counts : ndarray
        Number of neighbors in the same Z-layer (max 8)
    cross_layer_counts : ndarray
        Number of neighbors in adjacent Z-layers (max 18)
    total_counts : ndarray
        Total 26-neighbors (for reference)
    """
    D, H, W = skeleton.shape
    in_plane = np.zeros((D, H, W), dtype=np.int32)
    cross_layer = np.zeros((D, H, W), dtype=np.int32)
    total = np.zeros((D, H, W), dtype=np.int32)
    
    for z in prange(D):
        for y in range(H):
            for x in range(W):
                if skeleton[z, y, x] == 0:
                    continue
                
                in_plane_c = 0
                cross_layer_c = 0
                
                for dz in range(-1, 2):
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dz == 0 and dy == 0 and dx == 0:
                                continue
                            
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                                if skeleton[nz, ny, nx] > 0:
                                    if dz == 0:
                                        in_plane_c += 1
                                    else:
                                        cross_layer_c += 1
                
                in_plane[z, y, x] = in_plane_c
                cross_layer[z, y, x] = cross_layer_c
                total[z, y, x] = in_plane_c + cross_layer_c
    
    return in_plane, cross_layer, total


@njit(parallel=True)
def _count_neighbors_volume(skeleton):
    """Count 26-neighbors for each voxel in parallel.
    
    NOTE: This function is kept for backward compatibility but is no longer
    used in classify_skeleton_post_thinning. Use _count_neighbors_layered instead.
    """
    D, H, W = skeleton.shape
    counts = np.zeros((D, H, W), dtype=np.int32)
    for z in prange(D):
        for y in range(H):
            for x in range(W):
                if skeleton[z, y, x] == 0:
                    continue
                c = 0
                for dz in range(-1, 2):
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dz == 0 and dy == 0 and dx == 0:
                                continue
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                                if skeleton[nz, ny, nx] > 0:
                                    c += 1
                counts[z, y, x] = c
    return counts


def classify_skeleton_post_thinning(skeleton, min_plate_size=3, flatness_ratio=3.0,
                                     junction_thresh=4, min_avg_neighbors=3.0,
                                     linearity_max=3.5, solid=None,
                                     skeleton_curve=None, signal_mode='both'):
    """
    Classify skeleton voxels as surface (plate) or curve (beam) using
    two-pass thinning comparison.

    Compares mode=3 skeleton (surfaces + curves preserved) with mode=0
    skeleton (curves only).  Voxels present in mode=3 but absent from
    mode=0 are **plate interior** voxels — they are the surface sheets
    that mode=0 collapsed but mode=3 preserved.

    This approach requires **no PCA, no eigenvalue thresholds, and no
    radius parameters**.  The topological test directly identifies plate
    surfaces.

    When ``skeleton_curve`` is not provided, falls back to PCA-based
    classification.

    Parameters
    ----------
    skeleton : ndarray
        Binary skeleton from mode=3 thinning (surfaces + curves).
    min_plate_size : int
        Minimum voxels for a connected surface region to be plate.
    flatness_ratio : float
        Used only for PCA fallback and global region linearity check.
    junction_thresh : int
        Kept for CLI compatibility.
    min_avg_neighbors : float
        Kept for CLI compatibility (not used in two-pass mode).
    linearity_max : float
        Used only for PCA fallback.
    solid : ndarray or None
        Pre-thinning solid volume (used for PCA fallback only).
    skeleton_curve : ndarray or None
        Binary skeleton from mode=0 thinning (curves only).
        When provided, enables the two-pass topological classification.
    signal_mode : str
        Which signals to use: ``'both'`` (default), ``'a_only'``
        (set-difference only), or ``'b_only'`` (plane-octant only).
        Used for ablation studies.

    Returns
    -------
    zone_mask : ndarray
        0=empty, 1=plate, 2=beam
    plate_labels : ndarray
        Connected component labels for plate regions (0=non-plate)
    zone_stats : dict
        Statistics about the classification
    """
    eps = 1e-6

    skel_both = (skeleton > 0).astype(np.int32)
    n_skel_voxels = int(np.sum(skel_both))

    if n_skel_voxels == 0:
        zone_mask = np.zeros_like(skeleton, dtype=np.int8)
        plate_labels = np.zeros_like(skeleton, dtype=np.int32)
        return zone_mask, plate_labels, {
            'n_plate_regions': 0, 'n_plate_voxels': 0,
            'n_beam_voxels': 0, 'n_skeleton_voxels': 0,
            'n_components': 0, 'n_junctions': 0,
        }

    if skeleton_curve is None:
        # Fallback: PCA-based classification (legacy path)
        return _classify_pca_fallback(
            skeleton, min_plate_size, flatness_ratio,
            junction_thresh, min_avg_neighbors, linearity_max, solid
        )

    # ===== TWO-PASS TOPOLOGICAL CLASSIFICATION =====
    skel_curve = (skeleton_curve > 0).astype(np.int32)
    n_curve_voxels = int(np.sum(skel_curve))

    # Signal A: Present in mode=3 but NOT in mode=0
    # These are surface sheets that curve-preserving thinning collapsed.
    diff_candidates = (skel_both > 0) & (skel_curve == 0)
    n_diff = int(np.sum(diff_candidates))

    # Signal B: Strict topology-based surface detection on the mode=3 skeleton.
    # Uses count_plane_octants() which only counts octants with genuine plane
    # pattern matches (NOT the n3 < 3 fallback that makes is_surface_point()
    # trivially true in sparse skeletons).
    # Real surface voxels: 4-8 plane octants.  Beam voxels: 0-2.
    from scipy.ndimage import convolve
    neighbor_kernel = np.ones((3, 3, 3), dtype=np.int32)
    neighbor_kernel[1, 1, 1] = 0
    nc_vol = convolve(skel_both, neighbor_kernel, mode='constant')

    D, H, W = skel_both.shape
    topo_candidates = np.zeros_like(skel_both, dtype=bool)
    skel_coords = np.argwhere(skel_both > 0)
    min_plane_octants = 4  # require at least 4 of 8 octants to have real plane patterns
    for coord in skel_coords:
        z, y, x = coord
        if nc_vol[z, y, x] < 3:
            continue
        hood = get_neighborhood_window(skel_both, z, y, x)
        n_plane = count_plane_octants(hood)
        if n_plane >= min_plane_octants:
            topo_candidates[z, y, x] = True
    n_topo = int(np.sum(topo_candidates))

    # Combine: plate candidate = Signal A OR Signal B (or single signal for ablation)
    if signal_mode == 'a_only':
        plate_candidates = diff_candidates
    elif signal_mode == 'b_only':
        plate_candidates = topo_candidates
    else:
        plate_candidates = diff_candidates | topo_candidates

    print(f"    [PostThinningClassify] Two-pass: mode=3 has {n_skel_voxels}, "
          f"mode=0 has {n_curve_voxels}, diff={n_diff}, topo_strict={n_topo}, "
          f"combined={int(np.sum(plate_candidates))}")

    # Label connected components
    s26 = np.ones((3, 3, 3), dtype=np.int32)
    flat_labels, n_regions = label(plate_candidates.astype(np.int32), structure=s26)

    # Filter regions by size and global shape
    zone_mask = np.zeros_like(skeleton, dtype=np.int8)
    plate_labels_out = np.zeros_like(skeleton, dtype=np.int32)
    plate_id = 0
    n_plate_voxels = 0

    for rid in range(1, n_regions + 1):
        region_mask = (flat_labels == rid)
        region_size = int(np.sum(region_mask))

        if region_size < min_plate_size:
            continue

        # Global linearity check — reject elongated chains that may appear
        # in the difference (e.g., beam cross-section surface remnants)
        region_coords = np.argwhere(region_mask).astype(np.float64)
        if len(region_coords) >= 3:
            centered = region_coords - region_coords.mean(axis=0)
            cov = np.cov(centered, rowvar=False)
            if cov.ndim == 2 and cov.shape == (3, 3):
                ev = np.sort(np.linalg.eigvalsh(cov))
                linearity = (ev[2] + eps) / (ev[1] + eps)
                if linearity > flatness_ratio * 3:
                    z_range = f"{region_coords[:,2].min():.0f}-{region_coords[:,2].max():.0f}"
                    print(f"    [PostThinningClassify] Region {rid}: size={region_size}, "
                          f"z=[{z_range}] -> REJECTED (global_linearity={linearity:.1f})")
                    continue

        plate_id += 1
        zone_mask[region_mask] = 1
        plate_labels_out[region_mask] = plate_id
        n_plate_voxels += region_size

    # Growth pass: capture surface edge voxels adjacent to plate regions.
    # Edge voxels have fewer neighbors (nc >= 2) and fewer plane octants
    # (n_plane >= 2) than interior surface voxels.  Require n_plane >= 2
    # to avoid pulling in pure beam voxels (n_plane = 0) at plate-beam junctions.
    growth_count = 0
    for coord in skel_coords:
        z, y, x = coord
        if zone_mask[z, y, x] != 0:
            continue  # already classified
        if nc_vol[z, y, x] < 2:
            continue  # isolated endpoint
        # Must have some surface character (at least 2 plane octants)
        hood = get_neighborhood_window(skel_both, z, y, x)
        if count_plane_octants(hood) < 2:
            continue
        # Check if any 26-neighbor is a plate voxel
        has_plate_neighbor = False
        for dz in range(-1, 2):
            if has_plate_neighbor:
                break
            for dy in range(-1, 2):
                if has_plate_neighbor:
                    break
                for dx in range(-1, 2):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                        if zone_mask[nz, ny, nx] == 1:
                            has_plate_neighbor = True
                            break
        if has_plate_neighbor:
            # Find which plate region this borders and assign same label
            assigned = False
            for dz in range(-1, 2):
                if assigned:
                    break
                for dy in range(-1, 2):
                    if assigned:
                        break
                    for dx in range(-1, 2):
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                            if plate_labels_out[nz, ny, nx] > 0:
                                zone_mask[z, y, x] = 1
                                plate_labels_out[z, y, x] = plate_labels_out[nz, ny, nx]
                                growth_count += 1
                                assigned = True
                                break
    n_plate_voxels += growth_count
    if growth_count > 0:
        print(f"    [PostThinningClassify] Growth pass: +{growth_count} edge voxels added to plates")

    # All remaining skeleton voxels are beams
    beam_mask = (skel_both > 0) & (zone_mask == 0)
    zone_mask[beam_mask] = 2
    n_beam_voxels = int(np.sum(beam_mask))

    zone_stats = {
        'n_plate_regions': plate_id,
        'n_plate_voxels': n_plate_voxels,
        'n_beam_voxels': n_beam_voxels,
        'n_skeleton_voxels': n_skel_voxels,
        'n_components': n_regions,
        'n_junctions': n_diff,
        'n_signal_a': n_diff,
        'n_signal_b': n_topo,
        'n_overlap': int(np.sum(diff_candidates & topo_candidates)),
        'n_combined': int(np.sum(plate_candidates)),
        'n_growth': growth_count,
        'signal_mode': signal_mode,
    }

    print(f"    [PostThinningClassify] {n_skel_voxels} skeleton voxels "
          f"({n_diff} surface diff, {n_regions} regions) -> "
          f"{n_plate_voxels} plate ({plate_id} regions) + {n_beam_voxels} beam")

    return zone_mask, plate_labels_out, zone_stats


def _classify_pca_fallback(skeleton, min_plate_size, flatness_ratio,
                            junction_thresh, min_avg_neighbors, linearity_max, solid):
    """Legacy PCA-based classification (fallback when skeleton_curve not provided)."""
    from scipy.ndimage import convolve

    eps = 1e-6
    skel_binary = (skeleton > 0).astype(np.int32)
    skel_coords = np.argwhere(skel_binary > 0).astype(np.float64)
    n_skel_voxels = len(skel_coords)

    s26_neighbors = np.ones((3, 3, 3), dtype=np.int32)
    s26_neighbors[1, 1, 1] = 0
    neighbor_counts_vol = convolve(skel_binary, s26_neighbors, mode='constant', cval=0) * skel_binary

    skel_radius = 4.0
    tree = KDTree(skel_coords)
    local_flat = np.zeros(n_skel_voxels)

    for i in range(n_skel_voxels):
        nbrs = tree.query_ball_point(skel_coords[i], skel_radius)
        if len(nbrs) < 4:
            continue
        local_coords = skel_coords[nbrs]
        centered = local_coords - local_coords.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        if cov.ndim != 2 or cov.shape != (3, 3):
            continue
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))
        lam0, lam1 = eigenvalues[0] + eps, eigenvalues[1] + eps
        local_flat[i] = lam1 / lam0

    flat_volume = np.zeros_like(skel_binary)
    for i, coord in enumerate(skel_coords.astype(int)):
        y, x, z = coord
        nc = neighbor_counts_vol[y, x, z]
        if local_flat[i] >= flatness_ratio and nc >= 3:
            flat_volume[y, x, z] = 1

    s26 = np.ones((3, 3, 3), dtype=np.int32)
    flat_labels, n_flat_regions = label(flat_volume, structure=s26)

    zone_mask = np.zeros_like(skeleton, dtype=np.int8)
    plate_labels = np.zeros_like(skeleton, dtype=np.int32)
    plate_id = 0
    n_plate_voxels = 0

    for rid in range(1, n_flat_regions + 1):
        region_mask = (flat_labels == rid)
        region_size = int(np.sum(region_mask))
        if region_size >= min_plate_size:
            plate_id += 1
            zone_mask[region_mask] = 1
            plate_labels[region_mask] = plate_id
            n_plate_voxels += region_size

    beam_mask = (skel_binary > 0) & (zone_mask == 0)
    zone_mask[beam_mask] = 2
    n_beam_voxels = int(np.sum(beam_mask))

    print(f"    [PostThinningClassify] PCA fallback: {n_skel_voxels} skeleton -> "
          f"{n_plate_voxels} plate ({plate_id} regions) + {n_beam_voxels} beam")

    return zone_mask, plate_labels, {
        'n_plate_regions': plate_id, 'n_plate_voxels': n_plate_voxels,
        'n_beam_voxels': n_beam_voxels, 'n_skeleton_voxels': n_skel_voxels,
        'n_components': n_flat_regions, 'n_junctions': 0,
    }


def smooth_polyline(points, iterations=3):
    if len(points) < 3: return points 
    pts = np.array(points)
    new_pts = pts.copy()
    for _ in range(iterations):
        new_pts[1:-1] = 0.5 * pts[1:-1] + 0.25 * (pts[:-2] + pts[2:])
        pts = new_pts.copy()
    return pts.tolist()

def extract_graph(skeleton, pitch, origin, tags=None, hybrid_mode=False,
                  consolidate_tags=None):
    """Convert a thinned skeleton to a beam graph.

    The function classifies skeleton voxels by local connectivity, collapses
    tagged BC regions to single centroid nodes, then traces edges between
    feature voxels (junctions, endpoints) via BFS through degree-2 voxels.

    Args:
        skeleton (numpy.ndarray): Binary skeleton array, shape ``(D, H, W)``,
            ``uint8``.  Voxels with value ``> 0`` are skeleton.
        pitch (float): Voxel size in mm.  Multiplied by voxel indices to
            produce world-space node positions.
        origin (numpy.ndarray): World-space origin ``[ox, oy, oz]`` (mm).
        tags (numpy.ndarray or None): BC tag array, same shape as
            ``skeleton``.  Tag ``1`` = fixed; tag ``2`` = loaded.
        hybrid_mode (bool): If ``True``, use the hybrid voxel classifier
            that distinguishes junction, end, surface, and body voxels.
        consolidate_tags (list of int or None): Which tag values to
            consolidate into centroid nodes.  ``None`` consolidates all
            (legacy behaviour).  Pass ``[1]`` to only consolidate support
            clusters while tracing load-tagged voxels as normal edges.

    Returns:
        tuple:
            - **nodes_arr** (``numpy.ndarray``, shape ``(N, 3)``): Node
              positions in world space (mm).
            - **edges_list** (``list``): Edge tuples
              ``[u, v, weight, waypoints]`` where *u*, *v* are node indices,
              *weight* is edge length (mm), and *waypoints* are intermediate
              skeleton positions.
            - **v_types** (``numpy.ndarray``, shape ``(D, H, W)``): Voxel
              classification array (``int8``): 0=background, 1=endpoint,
              2=body, 3=junction, 4=surface, 5=tagged.
            - **node_tags** (``dict``): Maps node index → BC tag value.
    """
    if tags is not None:
        skeleton, tags, fixed_centroids = consolidate_tagged_voxels(
            skeleton, tags, consolidate_values=consolidate_tags)
    v_types = classify_voxels_hybrid(skeleton) if hybrid_mode else classify_voxels(skeleton)
    if tags is not None:
        for (cz, cy, cx) in fixed_centroids:
            if v_types[cz, cy, cx] > 0: v_types[cz, cy, cx] = 3 
    feature_indices = np.argwhere((v_types == 1) | (v_types == 3) | (v_types == 5))
    voxel_to_node, nodes, node_tags = {}, [], {}
    # skeleton.shape = (nely, nelx, nelz) so argwhere gives (nely_idx, nelx_idx, nelz_idx)
    # Reorder to (nelx, nely, nelz) world coords so world-X=length, world-Y=height, world-Z=depth
    for idx, (z, y, x) in enumerate(feature_indices):
        voxel_to_node[(z, y, x)] = idx
        nodes.append(origin + (np.array([y, z, x]) * pitch) + (pitch * 0.5))  # [nelx, nely, nelz]
        if tags is not None and tags[z, y, x] > 0: node_tags[idx] = int(tags[z, y, x])
    edges, offsets, (D, H, W) = [], get_neighbor_offsets(), skeleton.shape
    for z_start, y_start, x_start in feature_indices:
        u_id = voxel_to_node[(z_start, y_start, x_start)]
        for doz, doy, dox in offsets:
            nz, ny, nx = z_start+doz, y_start+doy, x_start+dox
            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W and skeleton[nz, ny, nx] > 0:
                path, curr_v, prev_v = [(z_start, y_start, x_start)], (nz, ny, nx), (z_start, y_start, x_start)
                while True:
                    path.append(curr_v)
                    cz, cy, cx = curr_v
                    if (cz, cy, cx) in voxel_to_node:
                        if (cz, cy, cx) != (z_start, y_start, x_start) and u_id < voxel_to_node[(cz, cy, cx)]:
                            v_id, inter_coords = voxel_to_node[(cz, cy, cx)], []
                            if len(path) > 2:
                                inter_indices = np.array(path[1:-1])  # cols: [nely, nelx, nelz]
                                inter_indices_reordered = inter_indices[:, [1, 0, 2]]  # → [nelx, nely, nelz]
                                inter_coords = smooth_polyline(origin + (inter_indices_reordered * pitch) + (pitch * 0.5))
                            # Euclidean polyline length (not hop count) for correct collapse thresholds
                            path_arr = np.array(path)  # (N, 3) cols: [nely, nelx, nelz]
                            path_world = origin + path_arr[:, [1, 0, 2]] * pitch + pitch * 0.5
                            edge_weight = float(np.sum(np.linalg.norm(np.diff(path_world, axis=0), axis=1)))
                            edges.append([u_id, v_id, edge_weight, inter_coords])
                        break
                    found_next = False
                    for dnz, dny, dnx in offsets:
                        nz, ny, nx = cz+dnz, cy+dny, cx+dnx
                        if (nz, ny, nx) != prev_v and 0 <= nz < D and 0 <= ny < H and 0 <= nx < W and skeleton[nz, ny, nx] > 0:
                            if not (hybrid_mode and v_types[nz, ny, nx] == 4):
                                prev_v, curr_v, found_next = curr_v, (nz, ny, nx), True
                                break
                    if not found_next: break
    return np.array(nodes), edges, v_types, node_tags
