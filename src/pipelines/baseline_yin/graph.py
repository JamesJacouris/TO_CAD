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
    get_neighborhood_window
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

def consolidate_tagged_voxels(skeleton, tags):
    skel, new_tags, centroids = skeleton.copy(), np.zeros_like(tags), []
    unique_tags = np.unique(tags)
    unique_tags = unique_tags[unique_tags > 0]
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


def classify_skeleton_post_thinning(skeleton, min_plate_size=4, flatness_ratio=3.0,
                                     junction_thresh=4, min_avg_neighbors=3.0):
    """
    Classify skeleton voxels as surface (plate) or curve (beam) using
    local PCA planarity detection.

    After mode=3 thinning, this function:
      1. For each skeleton voxel, computes PCA on its local neighborhood
         (within a radius of 4 voxels).
      2. Voxels where the local eigenvalue ratio λ1/λ0 >= flatness_ratio
         are marked as "locally flat" (part of a surface).
      3. Connected components of locally-flat voxels are grouped into
         surface regions by 26-connectivity.
      4. Regions with >= min_plate_size voxels are classified as plates;
         all other skeleton voxels are beams.

    This approach correctly detects surfaces even in dense, heavily-
    connected skeleton regions where junction-splitting fails.

    Parameters
    ----------
    skeleton : ndarray
        Binary skeleton volume from mode=3 thinning.
    min_plate_size : int
        Minimum voxels for a connected flat region to be plate (default=4).
    flatness_ratio : float
        Local PCA eigenvalue ratio threshold for flatness detection.
    junction_thresh : int
        Not used in local PCA mode (kept for CLI compatibility).
    min_avg_neighbors : float
        Minimum average neighbor count for a region to be classified as a plate (default=3.0).
        Used to filter out 1-voxel wide chains (avg neighbors ~2.5).

    Returns
    -------
    zone_mask : ndarray
        0=empty, 1=plate, 2=beam
    plate_labels : ndarray
        Connected component labels for plate regions (0=non-plate)
    zone_stats : dict
        Statistics about the classification
    """
    s26 = np.ones((3, 3, 3), dtype=np.int32)
    eps = 1e-6
    radius = 4.0

    skel_binary = (skeleton > 0).astype(np.int32)
    skel_coords = np.argwhere(skel_binary > 0).astype(np.float64)
    n_skel_voxels = len(skel_coords)

    if n_skel_voxels == 0:
        zone_mask = np.zeros_like(skeleton, dtype=np.int8)
        plate_labels = np.zeros_like(skeleton, dtype=np.int32)
        return zone_mask, plate_labels, {
            'n_plate_regions': 0, 'n_plate_voxels': 0,
            'n_beam_voxels': 0, 'n_skeleton_voxels': 0,
            'n_components': 0, 'n_junctions': 0,
        }

    # Step 1: Local PCA — compute per-voxel flatness
    tree = KDTree(skel_coords)
    local_flat = np.zeros(n_skel_voxels)

    for i in range(n_skel_voxels):
        nbrs = tree.query_ball_point(skel_coords[i], radius)
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

    # Step 2: Mark locally-flat voxels in the volume
    flat_volume = np.zeros_like(skel_binary)
    for i, coord in enumerate(skel_coords.astype(int)):
        if local_flat[i] >= flatness_ratio:
            flat_volume[coord[0], coord[1], coord[2]] = 1
            
    # [NEW] Step 2.5: Heal Plate Edges (Iterative with Geometry Check)
    # PCA can fail at the exact boundary where a plate meets a beam (mixed neighborhood).
    # If a skeleton voxel is 0 (beam) but has >= 3 neighbors that are 1 (plate), 
    # it's likely part of the plate edge.
    
    # CONSTRAINT: Only heal if the voxel has some "flatness" (ratio > 1.2).
    # True beams have ratio ~1.0. Eroded edges have intermediate values (e.g. 2.0).
    # This stops the healing from eating into the beam.
    
    flat_ratio_volume = np.zeros_like(skeleton, dtype=np.float32)
    for i, coord in enumerate(skel_coords.astype(int)):
        flat_ratio_volume[coord[0], coord[1], coord[2]] = local_flat[i]
        
    from scipy.ndimage import convolve
    s26 = np.ones((3, 3, 3), dtype=np.int32)
    s26[1, 1, 1] = 0
    
    total_healed = 0
    for _ in range(2): # Reduced to 2 iterations for safety
        plate_neighbor_counts = convolve(flat_volume, s26, mode='constant', cval=0)
        
        # Candidates:
        # 1. Not currently plate
        # 2. Has >= 3 plate neighbors
        # 3. Has mild flatness (> 1.2) -> prevents eating beams
        candidates = (
            (skel_binary > 0) & 
            (flat_volume == 0) & 
            (plate_neighbor_counts >= 3) &
            (flat_ratio_volume > 1.2)
        )
        
        n_current_healed = np.sum(candidates)
        if n_current_healed == 0:
            break
            
        flat_volume[candidates] = 1
        total_healed += n_current_healed
        
    if total_healed > 0:
        print(f"    [PostThinningClassify] Healed {total_healed} plate edge voxels (constrained).")

    n_locally_flat = int(np.sum(flat_volume))

    # Step 3: Connected components of flat voxels -> surface regions
    flat_labels, n_flat_regions = label(flat_volume, structure=s26)

    # Step 4: Build zone_mask — large flat regions are plates, rest are beams
    zone_mask = np.zeros_like(skeleton, dtype=np.int8)
    plate_labels = np.zeros_like(skeleton, dtype=np.int32)
    plate_id = 0
    n_plate_voxels = 0

    # Optimization: Pre-calculate neighbor counts for all skeleton voxels
    # A 1-voxel wide chain has ~2 neighbors per voxel.
    # A 2D surface (plate) has >4 neighbors per voxel (interior > 6).
    from scipy.ndimage import convolve
    s26_neighbors = np.ones((3, 3, 3), dtype=np.int32)
    s26_neighbors[1, 1, 1] = 0
    skel_binary = (skeleton > 0).astype(np.int32)
    neighbor_counts_vol = convolve(skel_binary, s26_neighbors, mode='constant', cval=0) * skel_binary

    for rid in range(1, n_flat_regions + 1):
        region_mask = (flat_labels == rid)
        region_size = int(np.sum(region_mask))

        if region_size >= min_plate_size:
            # Local PCA verified each voxel is flat. Do global checks:
            
            # Check 1: Global Linearity (reject beam-like cross-sections)
            region_coords = np.argwhere(region_mask).astype(np.float64)
            if len(region_coords) >= 3:
                centered = region_coords - region_coords.mean(axis=0)
                cov = np.cov(centered, rowvar=False)
                if cov.ndim == 2 and cov.shape == (3, 3):
                    ev = np.sort(np.linalg.eigvalsh(cov))
                    linearity = (ev[2] + eps) / (ev[1] + eps)
                    if linearity > flatness_ratio * 3:
                        continue  # too elongated globally -> beam

            # Check 2: Average Neighbor Count (reject 1-voxel-wide chains)
            # Chains have avg neighbors ~2.0-2.5. Plates have > 4.0.
            region_nc = neighbor_counts_vol[region_mask]
            avg_neighbors = np.mean(region_nc)
            
            if avg_neighbors < min_avg_neighbors:
                 continue # Region is likely a 1-voxel wide chain

            plate_id += 1
            zone_mask[region_mask] = 1
            plate_labels[region_mask] = plate_id
            n_plate_voxels += region_size

    # Step 5: All remaining skeleton voxels are beams
    beam_mask = (skel_binary > 0) & (zone_mask == 0)
    zone_mask[beam_mask] = 2
    n_beam_voxels = int(np.sum(beam_mask))

    zone_stats = {
        'n_plate_regions': plate_id,
        'n_plate_voxels': n_plate_voxels,
        'n_beam_voxels': n_beam_voxels,
        'n_skeleton_voxels': n_skel_voxels,
        'n_components': n_flat_regions,
        'n_junctions': n_locally_flat,
    }

    print(f"    [PostThinningClassify] {n_skel_voxels} skeleton voxels "
          f"({n_locally_flat} locally-flat, {n_flat_regions} regions) -> "
          f"{n_plate_voxels} plate ({plate_id} regions) + {n_beam_voxels} beam")

    return zone_mask, plate_labels, zone_stats


def smooth_polyline(points, iterations=3):
    if len(points) < 3: return points 
    pts = np.array(points)
    new_pts = pts.copy()
    for _ in range(iterations):
        new_pts[1:-1] = 0.5 * pts[1:-1] + 0.25 * (pts[:-2] + pts[2:])
        pts = new_pts.copy()
    return pts.tolist()

def extract_graph(skeleton, pitch, origin, tags=None, hybrid_mode=False):
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
        skeleton, tags, fixed_centroids = consolidate_tagged_voxels(skeleton, tags)
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
                            edges.append([u_id, v_id, float(len(path)-1), inter_coords])
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
