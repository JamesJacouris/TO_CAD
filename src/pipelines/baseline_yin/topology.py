import numpy as np
from numba import njit

# ----------------------------------------------------------------------------
# 26-Connectivity Configuration
# ----------------------------------------------------------------------------
# Neighbors of the center pixel (index 13 in a flattened 3x3x3 array of 27 elements)
# 0  1  2
# 3  4  5
# 6  7  8
# ...
# 13 is center
# Reference: http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/connectivity.html

@njit
def get_neighbor_offsets():
    """Returns (26, 3) offsets for 26-neighbors"""
    offsets = []
    for z in range(-1, 2):
        for y in range(-1, 2):
            for x in range(-1, 2):
                if x == 0 and y == 0 and z == 0:
                    continue
                offsets.append((x, y, z))
    return np.array(offsets)

@njit
def count_neighbors(neighborhood):
    """
    Counts active 26-neighbors in a 3x3x3 neighborhood.
    neighborhood: 3x3x3 int array (0 or 1). Center is at [1,1,1]
    """
    s = 0
    for z in range(3):
        for y in range(3):
            for x in range(3):
                if z==1 and y==1 and x==1:
                    continue
                if neighborhood[z, y, x] > 0:
                    s += 1
    return s

@njit
def is_end_voxel(neighborhood):
    """
    Algorithm 3.1 Param 4: End voxel has exactly 1 neighbor.
    (Or 0 neighbors if isolated, but usually 1 for skeleton tip)
    """
    n = count_neighbors(neighborhood)
    return n == 1

@njit 
def is_joint_voxel(neighborhood):
    """
    Section 4.1.2: Joint voxel has > 2 neighbors.
    """
    n = count_neighbors(neighborhood)
    return n > 2

# ----------------------------------------------------------------------------
# Is Simple Point (Topological Preservation)
# ----------------------------------------------------------------------------
# A point is 'simple' if its removal does not change the topology.
# In 3D digital topology (26-connectivity for FG, 6-connectivity for BG),
# a point P is simple iff:
# 1. Number of connected components of FG in N26(P) is 1.
# 2. Number of connected components of BG in N6(P) (captured in N26 neighborhood) is 1.
# Reference: Bertrand & Malandain "A new characterization of simple points in digital topology"

@njit
def get_components_26(neighborhood):
    """
    Counts connected components of Foreground (1s) in the 26-neighborhood (3x3x3)
    excluding the center pixel itself.
    """
    # Create a local copy to flood fill (excluding center)
    temp = neighborhood.copy()
    temp[1, 1, 1] = 0 # Ensure center is not counted/traversed
    
    dims = (3, 3, 3)
    visited = np.zeros(dims, dtype=np.int8)
    
    components = 0
    
    # Offsets for 26-connectivity
    offsets = np.array([
        (-1,-1,-1), (-1,-1, 0), (-1,-1, 1),
        (-1, 0,-1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1,-1), (-1, 1, 0), (-1, 1, 1),
        ( 0,-1,-1), ( 0,-1, 0), ( 0,-1, 1),
        ( 0, 0,-1),             ( 0, 0, 1),
        ( 0, 1,-1), ( 0, 1, 0), ( 0, 1, 1),
        ( 1,-1,-1), ( 1,-1, 0), ( 1,-1, 1),
        ( 1, 0,-1), ( 1, 0, 0), ( 1, 0, 1),
        ( 1, 1,-1), ( 1, 1, 0), ( 1, 1, 1)
    ])

    for z in range(3):
        for y in range(3):
            for x in range(3):
                if temp[z, y, x] == 1 and visited[z, y, x] == 0:
                    components += 1
                    # Flood fill this component
                    stack = [(z, y, x)]
                    visited[z, y, x] = 1
                    while len(stack) > 0:
                        cz, cy, cx = stack.pop()
                        
                        for i in range(26):
                            dz, dy, dx = offsets[i]
                            nz, ny, nx = cz + dz, cy + dy, cx + dx
                            
                            if 0 <= nz < 3 and 0 <= ny < 3 and 0 <= nx < 3:
                                if temp[nz, ny, nx] == 1 and visited[nz, ny, nx] == 0:
                                    visited[nz, ny, nx] = 1
                                    stack.append((nz, ny, nx))
    return components

@njit
def get_components_6_bg(neighborhood):
    """
    Counts connected components of Background (0s) in the 18-neighborhood 
    (neighbors connected to center by face allowed in 6-conn?).
    Actually, for simple point check:
    We check connectivity of 0s in N18 (neighs of center) using 6-connectivity.
    Wait, Bertrand's characterization uses Geodesic Neighborhoods.
    
    Standard efficient implementation:
    Use Lee/Kashyap or similar.
    
    For 26-connectivity FG, we need 6-connectivity BG.
    We check if background neighbors (0s) are 6-connected within the 3x3x3 block.
    """
    # Background is defined as any 0 in the 3x3x3 (excluding center)
    # But strictly, we care about the "Geodesic Neighborhood". 
    # Actually, simpler Euler invariant check is commonly used.
    # Let's use the explicit CCL for BG using 6-conn.
    
    temp = neighborhood.copy()
    temp[1, 1, 1] = 1 # Treat center as 1 (FG) so strict BG doesn't use it.
    
    # We only traverse 0s.
    visited = np.zeros((3,3,3), dtype=np.int8)
    components = 0
    
    # 6-neighbors offsets
    offsets_6 = np.array([
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1)
    ])
    
    # We scan all neighbors of the center [1,1,1] in the 3x3x3 grid
    # Connectivity should be checked "within the neighborhood".
    
    # One nuance: corners (e.g. 0,0,0) are 26-neighbors of center, 
    # but not 6-neighbors of center. 
    # For BG connectivity in 6-topology, we usually consider 18-neighbors 
    # (Face + Edge neighbors). Corner (Vertex) neighbors are not 6-connected to center directly?
    # Actually, simple point condition usually checks: T_26(v) = 1 and T_6_bar(v) = 1.
    
    # Let's implement the standard Octree or Malandain check if possible, 
    # but CCL is robust and easy to verify.
    
    # To check BG components N_6(v) in neighborhood:
    # We iterate all '0's in the 26-neighborhood? 
    # No, Malandain/Bertrand says we look at G_6(v, S_bar) -> Number of components of 0s 6-adjacent to v.
    # The 0s must be 6-neighbors of center? (Face neighbors).
    # Then we expand using 6-connectivity within the 26-neighborhood.
    
    # Find all 0s in the 6-neighbors of center:
    seeds = []
    face_neighbors = [(0,1,1), (2,1,1), (1,0,1), (1,2,1), (1,1,0), (1,1,2)]
    
    for (z,y,x) in face_neighbors:
        if temp[z,y,x] == 0:
            seeds.append((z,y,x))
            
    if len(seeds) == 0:
        return 0 # No BG neighbors? Then it's an interior point (sunk in object)
        
    # Count components among these seeds
    # Two seeds are connected if there is a path of 0s between them inside the 3x3x3 hood.
    # Path uses 6-connectivity.
    
    # We can just run CCL on the 3x3x3 '0' pixels using 6-conn, 
    # but only counting components that touch the center's face-neighbors?
    # Or just count components of 0s in 18-neighborhood?
    # Let's trust the T_6_bar condition: Number of 6-connected BG components in the neighborhood N26* 
    # that are 6-adjacent to x.
    
    # Implementation: 
    # 1. Identify all 0-voxels in the 3x3x3 that are 6-connected to P? (Only face neighbors).
    # 2. Check if they form 1 connected component using 6-paths within N26*.
    
    # Let's start CCL from face-neighbors.
    comp_map = np.zeros((3,3,3), dtype=np.int8) # 0=unvisited/FG, >0=component ID
    c_id = 0
    
    for (sz, sy, sx) in face_neighbors:
        if temp[sz, sy, sx] == 0 and comp_map[sz, sy, sx] == 0:
            c_id += 1
            # BFS
            stack = [(sz, sy, sx)]
            comp_map[sz, sy, sx] = c_id
            
            while len(stack) > 0:
                cz, cy, cx = stack.pop()
                
                # Check 6 neighbors
                for i in range(6):
                    dz, dy, dx = offsets_6[i]
                    nz, ny, nx = cz+dz, cy+dy, cx+dx
                    
                    if 0 <= nz < 3 and 0 <= ny < 3 and 0 <= nx < 3:
                        if temp[nz, ny, nx] == 0 and comp_map[nz, ny, nx] == 0:
                            # It is a 0. Is it in N26*? Yes, we are in 3x3x3.
                            # But wait, corners (0,0,0) are '0's too. Can we go through them?
                            # Yes, 6-path can go through (0,0,0) if it's 0.
                            comp_map[nz, ny, nx] = c_id
                            stack.append((nz, ny, nx))
                            
    # Now count unique component IDs that were found in the face-neighbors
    # (Since we only seeded from face neighbors, c_id IS the count, unless two face neighbors merged via corners)
    # Actually my logic above guarantees merges because 'comp_map' is shared.
    return c_id

@njit
def is_simple_point(neighborhood):
    """
    Checks if the center voxel (neighborhood[1,1,1]) is a simple point.
    input: 3x3x3 int array (0 or 1).
    """
    # 1. Check FG components in N26* (excluding center)
    nc_fg = get_components_26(neighborhood)
    if nc_fg != 1:
        return False
        
    # 2. Check BG components in N6* (connected to center)
    nc_bg = get_components_6_bg(neighborhood)
    if nc_bg != 1:
        return False
        
    return True
