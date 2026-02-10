import numpy as np
from numba import njit
from scipy.ndimage import label, center_of_mass, binary_dilation
from src.pipelines.baseline_yin.topology import get_neighbor_offsets, count_neighbors

# ----------------------------------------------------------------------------
# Algorithm 4.2: Identify Representative Tagged Voxels
# ----------------------------------------------------------------------------
# Helper: Simple 3D Line Drawing (Bresenham-like)
def draw_line_3d(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist = np.linalg.norm(p2 - p1)
    num_points = int(np.ceil(dist * 1.5)) + 1 # 1.5x sampling to be safe
    if num_points < 2: return [tuple(map(int, p1))]
    
    t = np.linspace(0, 1, num_points)
    points = []
    for i in range(num_points):
        pt = p1 + t[i] * (p2 - p1)
        points.append(tuple(np.round(pt).astype(int)))
    return points

def consolidate_tagged_voxels(skeleton, tags):
    """
    Yin Algorithm 4.2: Consolidates connected components of tagged voxels.
    Replaces each cluster with a single centroid node.
    Reconnects centroid to original cluster's neighbors in the skeleton.
    Returns:
        new_skeleton (copy), new_tags (copy), centroids (list of z,y,x)
    """
    skel = skeleton.copy()
    new_tags = np.zeros_like(tags)
    centroids = []
    
    unique_tags = np.unique(tags)
    unique_tags = unique_tags[unique_tags > 0] # Skip 0
    
    # Structure for 26-connectivity
    s26 = np.ones((3,3,3), dtype=np.int32)
    
    for t_val in unique_tags:
        # Mask for this tag (e.g. Fixed=1) INTERSECTED with Skeleton
        mask = (tags == t_val) & (skel > 0)
        
        if not np.any(mask):
            continue
            
        # Find connected components (Use 26-connectivity)
        labeled_array, num_features = label(mask, structure=s26)
        
        if num_features > 0:
            # Calculate centers
            centers = center_of_mass(mask, labeled_array, range(1, num_features+1))
            
            for i, center in enumerate(centers):
                # 1. Identify Cluster
                cluster_mask = (labeled_array == (i + 1))
                
                # 2. Find Neighbors of this cluster in the skeleton
                # Dilate cluster by 1 voxel
                dilated_mask = binary_dilation(cluster_mask, structure=s26)
                # Neighbors are: In Dilated AND In Skeleton AND NOT In Cluster
                neighbor_mask = dilated_mask & (skel > 0) & (~cluster_mask)
                neighbor_indices = np.argwhere(neighbor_mask) # List of [z,y,x]
                
                # 3. Calculate Centroid
                cz, cy, cx = int(round(center[0])), int(round(center[1])), int(round(center[2]))
                D, H, W = skel.shape
                cz, cy, cx = np.clip([cz, cy, cx], [0,0,0], [D-1, H-1, W-1])
                centroid_coord = (cz, cy, cx)
                
                # 4. Clear the entire cluster from skeleton
                skel[cluster_mask] = 0
                
                # 5. Add the Centroid back
                skel[cz, cy, cx] = 1
                new_tags[cz, cy, cx] = t_val
                centroids.append(centroid_coord)
                
                # 6. Reconnect Centroid to Neighbors
                for n_idx in neighbor_indices:
                    neigh_coord = tuple(n_idx)
                    line_pts = draw_line_3d(centroid_coord, neigh_coord)
                    for lpt in line_pts:
                        lz, ly, lx = lpt
                        if 0 <= lz < D and 0 <= ly < H and 0 <= lx < W:
                            skel[lz, ly, lx] = 1
                            # NOTE: We do NOT tag the connecting line, only the centroid remains tagged.
                            # This ensures only 1 BC node exists.
                
    return skel, new_tags, centroids

# ----------------------------------------------------------------------------
# 1. Type Classification (Section 4.1.2)
# ----------------------------------------------------------------------------
# Regular: 2 neighbors
# End: 1 neighbor
# Joint: >2 neighbors

@njit
def classify_voxels(skeleton, tags=None):
    """
    Returns an array of node types:
    0: None (Not in skeleton)
    1: End Voxel
    2: Regular Voxel
    3: Joint Voxel
    """
    D, H, W = skeleton.shape
    types = np.zeros_like(skeleton, dtype=np.int8)
    
    offsets = get_neighbor_offsets()
    indices = np.argwhere(skeleton > 0)
    
    for i in range(len(indices)):
        z, y, x = indices[i]
        
        # Count neighbors
        count = 0
        for off in offsets:
            dz, dy, dx = off
            nz, ny, nx = z+dz, y+dy, x+dx
            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                if skeleton[nz, ny, nx] > 0:
                    count += 1
                    
        if count == 1:
            types[z, y, x] = 1 # End
        elif count == 2:
            types[z, y, x] = 2 # Regular
        elif count > 2:
            types[z, y, x] = 3 # Joint
        else:
            types[z, y, x] = 1  # Isolated -> End
            
    return types

def smooth_polyline(points, iterations=3):
    """
    Simple Laplacian smoothing for a list of 3D points.
    Keeps first and last points fixed.
    """
    if len(points) < 3:
        return points 
        
    pts = np.array(points)
    new_pts = pts.copy()
    
    for _ in range(iterations):
        # Laplacian: P_i = 0.5 * P_i + 0.25 * (P_i-1 + P_i+1)
        new_pts[1:-1] = 0.5 * pts[1:-1] + 0.25 * (pts[:-2] + pts[2:])
        pts = new_pts.copy()
        
    return pts

def extract_graph(skeleton, pitch, origin, tags=None):
    """
    Extracts Graph G=(V, E) from Skeleton.
    Returns:
        nodes: (N, 3) coordinates
        edges: list of (u, v, weight, intermediates)
        v_types: 3D array of voxel types
        node_tags: dict {node_id: tag_value} (1=Fixed, 2=Loaded)
    """
    # 0. Pre-process: Consolidate Tagged Clusters (Yin's Way)
    if tags is not None:
        print("    [Graph] Consolidating BC Clusters (Algorithm 4.2)...")
        skeleton, tags, fixed_centroids = consolidate_tagged_voxels(skeleton, tags)
        
    # 1. Classify (Standard)
    v_types = classify_voxels(skeleton)
    
    # Force Consolidated Centroids to be Features (Joints) so they are extracted
    if tags is not None:
        for (cz, cy, cx) in fixed_centroids:
            # If it's a Regular node (2), promote to Joint (3) or something special
            # v_types[cz,cy,cx] = 3 ensures it's in feature_indices
             if v_types[cz, cy, cx] > 0: # Ensure it is part of skeleton
                v_types[cz, cy, cx] = 3 
    
    # 2. Identify Feature Voxels (End or Joint)
    feature_indices = np.argwhere((v_types == 1) | (v_types == 3))
    
    # Map (z,y,x) -> Node ID
    voxel_to_node = {}
    nodes = []
    node_tags = {}  # {node_id: tag_value}
    
    # Since we consolidated, we don't need proximity search! 
    # Just read the tag directly from the centroid voxel.
    
    for idx, (z, y, x) in enumerate(feature_indices):
        voxel_to_node[(z, y, x)] = idx
        coord = origin + (np.array([z, y, x]) * pitch) + (pitch * 0.5)
        nodes.append(coord)
        
        # Direct Tag Assignment
        if tags is not None:
            t_val = tags[z, y, x]
            if t_val > 0:
                node_tags[idx] = int(t_val)
                    
    if node_tags:
        n_fixed = sum(1 for v in node_tags.values() if v == 1)
        n_loaded = sum(1 for v in node_tags.values() if v == 2)
        print(f"    [Tags] Consolidated: Found {n_fixed} Fixed, {n_loaded} Loaded nodes")
        
    edges = []
        
    edges = []
    
    # 3. Marching
    offsets = get_neighbor_offsets()
    D, H, W = skeleton.shape
    
    for z_start, y_start, x_start in feature_indices:
        u_id = voxel_to_node[(z_start, y_start, x_start)]
        
        # Check all 26 neighbors
        current_neighbors = []
        for doz, doy, dox in offsets:
            nz, ny, nx = z_start+doz, y_start+doy, x_start+dox
            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                if skeleton[nz, ny, nx] > 0:
                    current_neighbors.append((nz, ny, nx))
                    
        for next_v in current_neighbors:
            path = [(z_start, y_start, x_start)]
            curr_v = next_v
            prev_v = (z_start, y_start, x_start)
            weight = 1.0 
            
            while True:
                path.append(curr_v)
                cz, cy, cx = curr_v
                
                if (cz, cy, cx) in voxel_to_node:
                    if (cz, cy, cx) != (z_start, y_start, x_start):
                        v_id = voxel_to_node[(cz, cy, cx)]
                        
                        if u_id < v_id:
                            intermediates = []
                            if len(path) > 2:
                                inter_indices = path[1:-1]
                                inter_indices_arr = np.array(inter_indices)
                                inter_coords = origin + (inter_indices_arr * pitch) + (pitch * 0.5)
                                # Smooth intermediates
                                intermediates = smooth_polyline(inter_coords.tolist())
                                if isinstance(intermediates, np.ndarray):
                                    intermediates = intermediates.tolist()
                                
                            edges.append([u_id, v_id, weight, intermediates])
                    break
                
                # Find Next Step
                found_next = False
                for doz, doy, dox in offsets:
                    nz, ny, nx = cz+doz, cy+doy, cx+dox
                    if (nz, ny, nx) == prev_v:
                        continue
                    if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                        if skeleton[nz, ny, nx] > 0:
                            prev_v = curr_v
                            curr_v = (nz, ny, nx)
                            weight += 1.0
                            found_next = True
                            break
                            
                if not found_next:
                    break
            
    return np.array(nodes), edges, v_types, node_tags
