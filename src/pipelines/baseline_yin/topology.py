"""Low-level topological predicates for 3-D binary thinning (Numba-accelerated).

All functions operate on a ``3×3×3`` neighbourhood window extracted around a
candidate voxel.  They are compiled by Numba ``@njit`` and cannot be called
from non-Numba contexts with keyword arguments.

Key predicates
--------------
- :func:`is_simple_point` — topology-preserving deletion test (Yin Def 3.14)
- :func:`is_end_voxel` — curve endpoint test (≤ 1 26-neighbour)
- :func:`is_surface_point` — plate surface voxel test
"""
import numpy as np
from numba import njit

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
def get_neighbor_offsets():
    """Returns (26, 3) offsets for 26-neighbors"""
    offsets = []
    for z in range(-1, 2):
        for y in range(-1, 2):
            for x in range(-1, 2):
                if x == 0 and y == 0 and z == 0: continue
                offsets.append((z, y, x))
    return np.array(offsets)

@njit
def count_neighbors(neighborhood):
    """Counts active 26-neighbors in a 3x3x3 neighborhood."""
    s = 0
    for z in range(3):
        for y in range(3):
            for x in range(3):
                if not (z==1 and y==1 and x==1) and neighborhood[z, y, x] > 0: s += 1
    return s

@njit
def is_end_voxel(neighborhood):
    """End voxel has exactly 1 neighbor."""
    return count_neighbors(neighborhood) == 1

@njit 
def is_joint_voxel(neighborhood):
    """Joint voxel has > 2 neighbors."""
    return count_neighbors(neighborhood) > 2

@njit
def _get_octant_config(neighborhood, cz, cy, cx):
    """Returns (config_byte, n3_count) for the 2x2x2 sub-block."""
    config, n3 = 0, 0
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                if neighborhood[cz+dz, cy+dy, cx+dx] > 0:
                    config |= (1 << (dz*4 + dy*2 + dx))
                    n3 += 1
    return config, n3

@njit
def is_surface_point(neighborhood):
    """Definition 3.14. Returns True if voxel is on a medial surface."""
    # Plane patterns: {0x0F, 0x33, 0x55, 0xC3, 0x99, 0xA5}
    for cz in range(2):
        for cy in range(2):
            for cx in range(2):
                config, n3 = _get_octant_config(neighborhood, cz, cy, cx)
                if not ((config in [0x0F, 0x33, 0x55, 0xC3, 0x99, 0xA5]) or (n3 < 3)): return False
    return True

@njit
def is_surface_point_relaxed(neighborhood, threshold=4):
    """
    Relaxed version of Definition 3.14. 
    Returns True if at least 'threshold' (out of 8) corner vertices satisfy 
    the plane pattern or surface edge condition.
    Default threshold 4 is more robust to discretization noise.
    """
    satisfied = 0
    for cz in range(2):
        for cy in range(2):
            for cx in range(2):
                config, n3 = _get_octant_config(neighborhood, cz, cy, cx)
                if (config in [0x0F, 0x33, 0x55, 0xC3, 0x99, 0xA5]) or (n3 < 3):
                    satisfied += 1
    return satisfied >= threshold

@njit
def is_surface_boundary(z, y, x, skeleton, surface_mask):
    """Checks if surface voxel is on boundary of medial surface."""
    D, H, W = skeleton.shape
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dz == 0 and dy == 0 and dx == 0: continue
                nz, ny, nx = z+dz, y+dy, x+dx
                if not (0 <= nz < D and 0 <= ny < H and 0 <= nx < W) or (skeleton[nz, ny, nx] > 0 and surface_mask[nz, ny, nx] == 0):
                    return True
    return False

@njit
def get_components_26(neighborhood):
    """Counts connected components of FG in N26*."""
    temp, dims = neighborhood.copy(), (3, 3, 3)
    temp[1, 1, 1], visited, components = 0, np.zeros(dims, dtype=np.int8), 0
    offsets = np.array([(-1,-1,-1), (-1,-1, 0), (-1,-1, 1), (-1, 0,-1), (-1, 0, 0), (-1, 0, 1), (-1, 1,-1), (-1, 1, 0), (-1, 1, 1), (0,-1,-1), (0,-1, 0), (0,-1, 1), (0, 0,-1), (0, 0, 1), (0, 1,-1), (0, 1, 0), (0, 1, 1), (1,-1,-1), (1,-1, 0), (1,-1, 1), (1, 0,-1), (1, 0, 0), (1, 0, 1), (1, 1,-1), (1, 1, 0), (1, 1, 1)])
    for z in range(3):
        for y in range(3):
            for x in range(3):
                if temp[z, y, x] == 1 and visited[z, y, x] == 0:
                    components, stack = components + 1, [(z, y, x)]
                    visited[z, y, x] = 1
                    while stack:
                        cz, cy, cx = stack.pop()
                        for i in range(26):
                            nz, ny, nx = cz+offsets[i,0], cy+offsets[i,1], cx+offsets[i,2]
                            if 0 <= nz < 3 and 0 <= ny < 3 and 0 <= nx < 3 and temp[nz, ny, nx] == 1 and visited[nz, ny, nx] == 0:
                                visited[nz, ny, nx], _ = 1, stack.append((nz, ny, nx))
    return components

@njit
def get_components_6_bg(neighborhood):
    """Counts connected components of BG (0s) 6-connected to center."""
    temp, comp_map, c_id = neighborhood.copy(), np.zeros((3,3,3), dtype=np.int8), 0
    temp[1, 1, 1], face_neighbors = 1, [(0,1,1), (2,1,1), (1,0,1), (1,2,1), (1,1,0), (1,1,2)]
    offsets_6 = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)])
    for sz, sy, sx in face_neighbors:
        if temp[sz, sy, sx] == 0 and comp_map[sz, sy, sx] == 0:
            c_id, stack = c_id + 1, [(sz, sy, sx)]
            comp_map[sz, sy, sx] = c_id
            while stack:
                cz, cy, cx = stack.pop()
                for i in range(6):
                    nz, ny, nx = cz+offsets_6[i,0], cy+offsets_6[i,1], cx+offsets_6[i,2]
                    if 0 <= nz < 3 and 0 <= ny < 3 and 0 <= nx < 3 and temp[nz, ny, nx] == 0 and comp_map[nz, ny, nx] == 0:
                        comp_map[nz, ny, nx], _ = c_id, stack.append((nz, ny, nx))
    return c_id

@njit
def is_simple_point(neighborhood):
    """Checks if center voxel is a simple point."""
    return get_components_26(neighborhood) == 1 and get_components_6_bg(neighborhood) == 1
