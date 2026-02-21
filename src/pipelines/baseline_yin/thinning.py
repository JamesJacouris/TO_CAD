import numpy as np
from numba import njit, prange
from src.pipelines.baseline_yin.topology import (
    is_simple_point, 
    is_end_voxel, 
    is_surface_point,
    is_surface_point_relaxed,
    get_neighbor_offsets,
    get_neighborhood_window
)

@njit
def get_border_direction_offsets(direction_idx):
    """
    Returns the offset to check for 'border connectivity' against a deletion direction.
    Directions: 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
    """
    if direction_idx == 0: return (-1, 0, 0) # +x sweep (border on -x)
    if direction_idx == 1: return ( 1, 0, 0) # -x sweep (border on +x)
    if direction_idx == 2: return ( 0,-1, 0) # +y
    if direction_idx == 3: return ( 0, 1, 0) # -y
    if direction_idx == 4: return ( 0, 0,-1) # +z
    if direction_idx == 5: return ( 0, 0, 1) # -z
    return (0,0,0)

@njit(parallel=True)
def find_candidates(volume, direction_idx, tags, mode=0, surface_mask=None, edt=None, plate_threshold=3.0, boundary_mask=None):
    """Algorithm 3.1 Loop 1: Find Deletable Candidates."""
    D, H, W = volume.shape
    check_off = get_border_direction_offsets(direction_idx)
    cdz, cdy, cdx = check_off

    counts_per_z = np.zeros(D, dtype=np.int32)
    for z in prange(D):
        c = 0
        for y in range(H):
            for x in range(W):
                if volume[z, y, x] == 0: continue
                if tags is not None and tags[z, y, x] > 0: continue
                if boundary_mask is not None and boundary_mask[z, y, x] > 0: continue

                nz, ny, nx = z + cdz, y + cdy, x + cdx
                is_border = (not (0 <= nz < D and 0 <= ny < H and 0 <= nx < W)) or (volume[nz, ny, nx] == 0)
                if not is_border: continue

                hood = get_neighborhood_window(volume, z, y, x)
                keep = False
                if mode == 0: # Pure Curve
                    if is_end_voxel(hood): keep = True
                elif mode == 1: # Pure Surface
                    if is_surface_point(hood): keep = True
                elif mode == 2: # Hybrid (Yin-style)
                    if edt is not None and edt[z, y, x] <= plate_threshold:
                        if is_surface_point_relaxed(hood): keep = True
                    if not keep and is_end_voxel(hood): keep = True
                elif mode == 3: # Post-thinning hybrid: preserve BOTH
                    if is_surface_point(hood): keep = True
                    if not keep and is_end_voxel(hood): keep = True

                if not keep and is_simple_point(hood): c += 1
        counts_per_z[z] = c

    offsets = np.zeros(D + 1, dtype=np.int32)
    for z in range(D): offsets[z+1] = offsets[z] + counts_per_z[z]
    candidates = np.zeros((offsets[D], 3), dtype=np.int32)

    for z in prange(D):
        ptr = offsets[z]
        for y in range(H):
            for x in range(W):
                if volume[z, y, x] == 0: continue
                if tags is not None and tags[z, y, x] > 0: continue
                if boundary_mask is not None and boundary_mask[z, y, x] > 0: continue
                nz, ny, nx = z + cdz, y + cdy, x + cdx
                is_border = (not (0 <= nz < D and 0 <= ny < H and 0 <= nx < W)) or (volume[nz, ny, nx] == 0)
                if not is_border: continue

                hood = get_neighborhood_window(volume, z, y, x)
                keep = False
                if mode == 0:
                    if is_end_voxel(hood): keep = True
                elif mode == 1:
                    if is_surface_point(hood): keep = True
                elif mode == 2:
                    if edt is not None and edt[z, y, x] <= plate_threshold:
                        if is_surface_point_relaxed(hood): keep = True
                    if not keep and is_end_voxel(hood): keep = True
                elif mode == 3:
                    if is_surface_point(hood): keep = True
                    if not keep and is_end_voxel(hood): keep = True

                if not keep and is_simple_point(hood):
                    candidates[ptr, 0], candidates[ptr, 1], candidates[ptr, 2] = z, y, x
                    ptr += 1
    return candidates

@njit
def sequential_delete(volume, candidates, iteration_map, current_iter, mode=0, surface_mask=None, edt=None, plate_threshold=3.0, boundary_mask=None):
    """Algorithm 3.1 Loop 2: Double Check & Delete."""
    deleted_count = 0
    for i in range(len(candidates)):
        z, y, x = candidates[i]
        if volume[z, y, x] == 0: continue
        if boundary_mask is not None and boundary_mask[z, y, x] > 0: continue
        hood = get_neighborhood_window(volume, z, y, x)
        keep = False
        if mode == 0:
            if is_end_voxel(hood): keep = True
        elif mode == 1:
            if is_surface_point(hood): keep = True
        elif mode == 2:
            if edt is not None and edt[z, y, x] <= plate_threshold:
                if is_surface_point_relaxed(hood): keep = True
            if not keep and is_end_voxel(hood): keep = True
        elif mode == 3:
            if is_surface_point(hood): keep = True
            if not keep and is_end_voxel(hood): keep = True

        if not keep and is_simple_point(hood):
            volume[z, y, x] = 0
            if iteration_map is not None: iteration_map[z, y, x] = current_iter
            deleted_count += 1
    return deleted_count

def thin_grid_yin(volume, tags=None, max_iters=100, record_iterations=False, mode=0, surface_mask=None, edt=None, plate_threshold=3.0, boundary_mask=None):
    """Main loop for Algorithm 3.1.

    Parameters
    ----------
    boundary_mask : ndarray or None
        If provided, voxels where boundary_mask > 0 are protected from deletion.
        Used to preserve beam-plate interface voxels during separate thinning passes.
    """
    directions = [0, 1, 2, 3, 4, 5]
    iteration_map = np.zeros_like(volume, dtype=np.int32) if record_iterations else None

    for it in range(max_iters):
        changed_any = False
        iter_removed = 0
        for d in directions:
            cands = find_candidates(volume, d, tags, mode=mode, surface_mask=surface_mask, edt=edt, plate_threshold=plate_threshold, boundary_mask=boundary_mask)
            if len(cands) == 0: continue
            n = sequential_delete(volume, cands, iteration_map, it + 1, mode=mode, surface_mask=surface_mask, edt=edt, plate_threshold=plate_threshold, boundary_mask=boundary_mask)
            if n > 0:
                iter_removed += n
                changed_any = True
        if not changed_any: break
    return (volume, iteration_map) if record_iterations else volume
