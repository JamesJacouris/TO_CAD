import numpy as np
from numba import njit
from src.pipelines.baseline_yin.topology import (
    is_simple_point, 
    is_end_voxel, 
    get_neighbor_offsets
)

@njit
def get_border_direction_offsets(direction_idx):
    """
    Returns the offset to check for 'border connectivity' against a deletion direction.
    Directions: 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
    Border voxel in +x direction means: it has NO neighbor in +x direction (which is usually vacuum).
    Wait, definition from paper:
    "Border voxel in +x direction... are the ones with no neighboured voxels on -x direction?"
    Let's check Fig 3.12.
    If deletion direction is +x (removing from right to left?), border voxels are on the right?
    Paper says: "for deletion direction +x... border voxels are ones with no neighbors on -x direction."
    This is slightly confusing.
    Usually:
    Delete direction +X = Moving generally towards +X? Or sweep plane +X?
    Let's assume "Deletion Direction +X" means we are "erasing from the +X side".
    So the border voxels are the ones exposed on the +X face.
    If exposed on +X face, they have NO neighbor on +X (+1,0,0).
    
    Let's re-read carefully: "Deletion direction +x ... border voxels correspond to deletion direction +x ... are the ones with no neighbor on -x direction".
    If I have no neighbor on -X, I am exposed on the -X side. 
    So "Deletion Direction +X" might mean "We move in the +X direction, removing voxels".
    So we start from -X side and move to +X side?
    If so, valid candidates are those exposed on -X side.
    
    Let's implement explicit mapping:
    Code 0 (+x): Check neighbor at (-1, 0, 0). If 0, then we are border facing -X.
    Code 1 (-x): Check neighbor at (+1, 0, 0).
    Code 2 (+y): Check (-1 y).
    Code 3 (-y): Check (+1 y).
    Code 4 (+z): Check (-1 z).
    Code 5 (-z): Check (+1 z).
    """
    if direction_idx == 0: return (-1, 0, 0) # +x sweep (border on -x)
    if direction_idx == 1: return ( 1, 0, 0) # -x sweep (border on +x)
    if direction_idx == 2: return ( 0,-1, 0) # +y
    if direction_idx == 3: return ( 0, 1, 0) # -y
    if direction_idx == 4: return ( 0, 0,-1) # +z
    if direction_idx == 5: return ( 0, 0, 1) # -z
    return (0,0,0)

@njit
def get_neighborhood_window(volume, z, y, x):
    """Extract 3x3x3 neighborhood around z,y,x safely."""
    D, H, W = volume.shape
    neighborhood = np.zeros((3, 3, 3), dtype=volume.dtype)
    
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                    neighborhood[dz+1, dy+1, dx+1] = volume[nz, ny, nx]
    return neighborhood

@njit
def find_candidates(volume, direction_idx, tags):
    """
    Algorithm 3.1 Loop 1: Find Deletable Candidates D.
    Returns array of (z, y, x) coordinates.
    """
    D, H, W = volume.shape
    candidates = []
    
    check_off = get_border_direction_offsets(direction_idx)
    cdz, cdy, cdx = check_off
    
    # We iterate over all voxels (optimize by bounding box ideally, but dense scan for now)
    # Numba handles loops well.
    for z in range(D):
        for y in range(H):
            for x in range(W):
                if volume[z, y, x] == 0:
                    continue
                
                # Check Tag (if tags provided)
                if tags is not None and tags[z, y, x] > 0:
                    continue
                    
                # Check Border Condition
                # Neighbor on 'check side' must be 0
                nz, ny, nx = z + cdz, y + cdy, x + cdx
                is_border = False
                if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                    if volume[nz, ny, nx] == 0:
                        is_border = True
                else:
                    is_border = True # Boundary of image is border
                    
                if not is_border:
                    continue
                
                # Extract Neighborhood
                hood = get_neighborhood_window(volume, z, y, x)
                
                # Check End Voxel (Don't delete tips)
                if is_end_voxel(hood):
                    continue
                    
                # Check Simple Point
                if is_simple_point(hood):
                    candidates.append((z, y, x))
                    
    return candidates

@njit
def sequential_delete(volume, candidates, iteration_map, current_iter):
    """
    Algorithm 3.1 Loop 2: Double Check & Delete.
    Sequentially re-checks simplicity and deletes immediately if simple.
    """
    deleted_count = 0
    
    for i in range(len(candidates)):
        z, y, x = candidates[i]
        
        # Verify it's still 1 (maybe deleted? No, list is unique)
        if volume[z, y, x] == 0:
            continue
            
        # Re-extract neighborhood (context changed by previous deletions)
        hood = get_neighborhood_window(volume, z, y, x)
        
        # Re-check Simple Point condition
        if is_simple_point(hood):
            # Check End Voxel again for safety
            if not is_end_voxel(hood):
                volume[z, y, x] = 0
                if iteration_map is not None:
                    iteration_map[z, y, x] = current_iter
                deleted_count += 1
                
    return deleted_count

def thin_grid_yin(volume, tags=None, max_iters=100, record_iterations=False):
    """
    Main loop for Algorithm 3.1.
    volume: 3D int array (0/1) - Will be modified in-place!
    tags: 3D int array (optional)
    record_iterations: Bool. If True, returns (volume, iteration_map).
    """
    # Order: +x, -x, +y, -y, +z, -z
    # Our indices: 0, 1, 2, 3, 4, 5
    directions = [0, 1, 2, 3, 4, 5]
    
    iteration_map = None
    if record_iterations:
        # 0 = Background/Kept Skeleton
        # 1..N = Removed at Iteration I
        iteration_map = np.zeros_like(volume, dtype=np.int32)
    
    total_removed = 0
    for it in range(max_iters):
        changed_any = False
        print(f"  [Thinning] Iteration {it+1}...")
        
        iter_removed = 0
        for d in directions:
            # 1. Find candidates
            cands = find_candidates(volume, d, tags)
            
            if len(cands) == 0:
                continue
                
            cands_arr = np.array(cands, dtype=np.int32)
            
            # 2. Sequential Delete
            # Numba requires compatible types. iteration_map might be None.
            # Passing None to njit functions can be tricky if types mismatch.
            # Let's handle it by passing specific array or dummy.
            # Actually, simpler: Use global iteration_map in python? No, loop is python.
            # Only sequential_delete is njit.
            # We can pass iteration_map if exists, else None.
            # Does numba handle Optional arrays? Yes usually.
            
            if iteration_map is not None:
                n = sequential_delete(volume, cands_arr, iteration_map, it + 1)
            else:
                 # Helper wrapper for None case to appease numba typing if needed
                 # But we can just pass a dummy array or None if supported.
                 # Let's rely on None support.
                 n = sequential_delete(volume, cands_arr, None, 0)
                 
            if n > 0:
                iter_removed += n
                changed_any = True
                
        print(f"    Removed {iter_removed} voxels.")
        total_removed += iter_removed
        
        if not changed_any:
            print("  [Thinning] Converged.")
            if record_iterations:
                return volume, iteration_map
            return volume
            
    print("  [Thinning] Max iterations reached.")
    if record_iterations:
        return volume, iteration_map
    return volume
