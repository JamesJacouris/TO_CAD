"""Graph refinement: pruning, edge collapse, polyline simplification, radius computation.

Applied after skeleton-to-graph extraction to produce a clean, well-connected
beam graph suitable for FEA and CAD export.

Key functions
-------------
- :func:`clean_edge_polylines` — remove spurious intermediate waypoints
- :func:`prune_branches` — delete short dead-end branch tips
- :func:`collapse_short_edges` — merge nodes connected by very short edges
- :func:`compute_edge_radii` — per-edge radius from EDT
- :func:`compute_uniform_radii` — volume-matching uniform radius
- :func:`ensure_nodes_at_bounding_extrema` — guarantee nodes at domain corners
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

def clean_edge_polylines(nodes_dict, edges):
    """
    Removes intermediate points that deviate significantly from the straight
    line between edge endpoints. Fixes spurious skeleton loops.

    nodes_dict: {id: [x, y, z]}
    edges: list of [u, v, w, pts, rad...]
    """
    def point_to_line_distance(point, line_start, line_end):
        """Compute perpendicular distance from point to line segment."""
        v = line_end - line_start
        w = point - line_start
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(w)
        c2 = np.dot(v, v)
        if c1 >= c2:
            return np.linalg.norm(point - line_end)
        b = c1 / c2
        proj = line_start + b * v
        return np.linalg.norm(point - proj)

    cleaned_edges = []
    max_perpendicular_error = 3.0  # Remove intermediate points >3mm from edge line

    for e in edges:
        u, v, w = e[0], e[1], e[2]
        pts = e[3] if len(e) > 3 else []
        rad = e[4] if len(e) > 4 else None

        if len(pts) == 0:
            # No intermediate points, keep edge as-is
            cleaned_edges.append(e)
            continue

        # Filter intermediate points
        p_start = np.array(nodes_dict[u], dtype=float)
        p_end = np.array(nodes_dict[v], dtype=float)

        filtered_pts = []
        for pt in pts:
            pt_arr = np.array(pt, dtype=float)
            perp_error = point_to_line_distance(pt_arr, p_start, p_end)

            # Keep only points that are close to the edge line
            if perp_error <= max_perpendicular_error:
                filtered_pts.append(pt)

        # Rebuild edge with cleaned intermediate points
        new_e = [u, v, w, filtered_pts]
        if rad is not None:
            new_e.append(rad)
        cleaned_edges.append(new_e)

    return nodes_dict, cleaned_edges


def remove_isolated_nodes(nodes_dict, edges, node_tags=None):
    """
    Removes nodes that are statistical outliers (isolated far from main cluster).
    Uses interquartile range (IQR) to identify and remove outlier nodes.
    Preserves tagged nodes (BC points).
    """
    if node_tags is None:
        node_tags = {}

    if len(nodes_dict) < 4:
        return nodes_dict, edges  # Not enough points to detect outliers

    node_ids = list(nodes_dict.keys())
    coords = np.array([nodes_dict[nid] for nid in node_ids])

    # For each axis, detect outliers using IQR method
    outlier_nodes = set()

    for axis in range(3):
        vals = coords[:, axis]
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1

        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            for i, nid in enumerate(node_ids):
                # Don't remove tagged nodes (BC points)
                if nid in node_tags:
                    continue

                if vals[i] < lower_bound or vals[i] > upper_bound:
                    outlier_nodes.add(nid)

    if not outlier_nodes:
        return nodes_dict, edges

    # Remove outlier nodes and their edges
    new_nodes_dict = {nid: coord for nid, coord in nodes_dict.items() if nid not in outlier_nodes}
    new_edges = [e for e in edges if e[0] not in outlier_nodes and e[1] not in outlier_nodes]

    print(f"  [Post] Removed {len(outlier_nodes)} isolated nodes: {outlier_nodes}")

    return new_nodes_dict, new_edges

# ----------------------------------------------------------------------------
# Helper: Graph Representation
# ----------------------------------------------------------------------------
# We use:
# nodes: dict {id: np.array([x,y,z])}
# edges: list of [u, v, weight] (Mutable)
# adjacency: dict {u: {v: w}} (Derived for fast lookup)

def build_adjacency(nodes, edges):
    """Builds adjacency dict from edge list."""
    adj = {uid: {} for uid in nodes}
    for e in edges:
        u, v, w = e[0], e[1], e[2]
        # Undirected
        if u in adj: adj[u][v] = w
        if v in adj: adj[v][u] = w
    return adj

def graph_to_arrays(nodes, edges):
    """Converts dict/list format back to Arrays for Visualization."""
    # Remap IDs to 0..N-1
    old_to_new = {uid: i for i, uid in enumerate(nodes.keys())}
    
    node_coords = np.zeros((len(nodes), 3))
    for uid, coord in nodes.items():
        node_coords[old_to_new[uid]] = coord
    
    edge_list = []
    for e in edges:
        u, v = e[0], e[1]
        if u in old_to_new and v in old_to_new:
            # Reconstruct edge with new Node IDs
            # Preserve weight and intermediates if present
            new_edge = [old_to_new[u], old_to_new[v]] + list(e[2:])
            edge_list.append(new_edge)
            
    return node_coords, edge_list

# ----------------------------------------------------------------------------
# Algorithm 4.3: Recheck Graph (Remove Duplicates & Regular Nodes)
# ----------------------------------------------------------------------------
def recheck_graph(nodes, edges, node_tags=None):
    """
    Cleans up graph:
    1. Removes duplicate edges (keeping min weight).
    2. Removes Regular Nodes (Degree 2) by merging edges.
    Edges format: [u, v, w, intermediates]
    """
    if node_tags is None: node_tags = {}

    # 1. Deduplicate Edges
    # Map (u,v) -> (min_weight, best_intermediates)
    unique_edges = {}
    
    for e in edges:
        u, v, w = e[0], e[1], e[2]
        pts = e[3] if len(e) > 3 else []
        
        if u > v: 
            u, v = v, u
            # If we flip edge direction, we should technically flip the intermediates list
            # P_u -> P_v. If flipped, intermediates should be reversed.
            pts = pts[::-1] 
            
        key = (u, v)
        if key not in unique_edges:
            unique_edges[key] = (w, pts)
        else:
            # Keep minimum weight
            if w < unique_edges[key][0]:
                unique_edges[key] = (w, pts)
            
    # Rebuild edges list
    new_edges = []
    for (u, v), (w, pts) in unique_edges.items():
        new_edges.append([u, v, w, pts])
    edges = new_edges
    
    # 2. Remove Regular Nodes (Degree 2)
    # Loop until stable
    changed = True
    while changed:
        changed = False
        adj = build_adjacency(nodes, edges) # Rebuild adj
        
        # Build map of u -> list of edges connected to u
        # adj structure: {u: {v: w}} - Doesn't give us the edge index or polyline easily.
        # Let's verify build_adjacency logic first. But we need access to 'pts'.
        
        # Helper map: u -> list of (neighbor, weight, pts)
        node_connections = {uid: [] for uid in nodes}
        for u, v, w, pts in edges:
            if u not in node_connections: node_connections[u] = []
            if v not in node_connections: node_connections[v] = []
            # Direction u->v
            node_connections[u].append((v, w, pts))
            # Direction v->u (reversed pts)
            node_connections[v].append((u, w, pts[::-1]))

        target_node = -1
        
        for u, conns in node_connections.items():
            if len(conns) == 2:
                # CRITICAL FIX: Don't remove if Tagged!
                if u in node_tags:
                    continue
                    
                target_node = u
                break
                
        if target_node != -1:
            u = target_node
            conns = node_connections[u]
            
            # Merge: A --(w1, pts1)--> U --(w2, pts2)--> B
            # Becomes A --(w1+w2, pts1 + [U_coord] + pts2)--> B
            
            n1 = conns[0] 
            n2 = conns[1]
            
            A, w1, pts_UA = n1
            B, w2, pts_UB = n2
            
            # Construct new polyline
            # pts_AU = pts_UA reversed (pts_UA is U->A, we want A->U)
            # wait, n1 is (v, w, pts_from_u_to_v).
            # So pts_UA is path U->A.
            # We want path A -> B.
            # path A -> B = (A -> U) + U + (U -> B)
            # A->U is reverse(pts_UA)
            # U->B is pts_UB
            
            U_coord = nodes[u].tolist()
            new_pts = pts_UA[::-1] + [U_coord] + pts_UB
            
            new_w = w1 + w2
            
            # Update Graph
            # Remove node u
            del nodes[u]
            
            # Remove edges connected to u
            # Filter global edge list
            edges = [e for e in edges if e[0] != u and e[1] != u]
            edges.append([A, B, new_w, new_pts])
            
            changed = True
            
    return nodes, edges

# ----------------------------------------------------------------------------
# Algorithm 4.4: Edge Collapse (Topology Cleanup)
# ----------------------------------------------------------------------------
def collapse_short_edges(nodes, edges, threshold, node_tags=None):
    """
    Merges nodes connected by edges with weight < threshold.
    Tagged nodes (in node_tags) are kept as anchors.
    """
    if node_tags is None:
        node_tags = {}
        
    changed = True
    while changed:
        changed = False
        
        for i, (u, v, w, pts) in enumerate(edges):
            if w < threshold:
                # Determine which node to keep
                u_tagged = u in node_tags
                v_tagged = v in node_tags
                
                if u_tagged and v_tagged:
                    if node_tags[u] != node_tags[v]:
                        # Different tags (e.g. Fixed vs Loaded) — don't collapse
                        continue
                    else:
                        # Same tags — okay to collapse
                        keep, remove = u, v
                elif v_tagged:
                    # Keep v, remove u
                    keep, remove = v, u
                elif u_tagged:
                    # Keep u, remove v
                    keep, remove = u, v
                else:
                    # Neither tagged — keep u, merge to centroid
                    keep, remove = u, v
                    p1 = nodes[u]
                    p2 = nodes[v]
                    nodes[keep] = (p1 + p2) * 0.5
                
                # Remap all edges connected to 'remove' -> 'keep'
                new_edges = []
                for j, e in enumerate(edges):
                    eu, ev, ew, epts = e[0], e[1], e[2], e[3] if len(e)>3 else []
                    
                    if eu == remove: eu = keep
                    if ev == remove: ev = keep
                    
                    if eu == ev: continue  # Remove Self Loop
                    
                    new_edges.append([eu, ev, ew, epts])
                
                edges = new_edges
                del nodes[remove]
                
                # Transfer tag if 'remove' was tagged (shouldn't happen due to guard above)
                if remove in node_tags:
                    node_tags[keep] = node_tags.pop(remove)
                
                nodes, edges = recheck_graph(nodes, edges, node_tags)
                
                changed = True
                break
                
    return nodes, edges

# ----------------------------------------------------------------------------
# Algorithm 4.5: Pruning (Remove Zero-Stress Branches)
# ----------------------------------------------------------------------------
def prune_branches(nodes, edges, min_len=0.0, node_tags=None):
    """
    Iteratively removes branches ending in degree-1 nodes.
    Tagged nodes (in node_tags) are never pruned.
    """
    if node_tags is None:
        node_tags = {}
        
    changed = True
    while changed:
        changed = False
        
        adj = build_adjacency(nodes, edges)
        
        to_remove = set()
        
        for u, neighbors in adj.items():
            if len(neighbors) == 1:
                # Degree 1: End Node
                # Don't prune if tagged (Fixed or Loaded)
                if u in node_tags:
                    continue
                    
                v = list(neighbors.keys())[0]
                w = neighbors[v]
                
                if w < min_len:
                    to_remove.add(u)
                    changed = True
                    
        if changed:
            nodes = {k:v for k,v in nodes.items() if k not in to_remove}
            edges = [e for e in edges if e[0] not in to_remove and e[1] not in to_remove]
            nodes, edges = recheck_graph(nodes, edges, node_tags)
            
    return nodes, edges

# ----------------------------------------------------------------------------
# Geometric Simplification (Ramer-Douglas-Peucker)
# ----------------------------------------------------------------------------
def perpendicular_distance(point, start, end):
    """Calculates perpendicular distance of a point from a line segment."""
    if np.all(start == end):
        return np.linalg.norm(point - start)
    
    return np.linalg.norm(np.cross(end - start, start - point)) / np.linalg.norm(end - start)

def rdp(points, epsilon):
    """
    Recursive Ramer-Douglas-Peucker algorithm.
    points: (N, 3) array
    """
    dmax = 0.0
    index = 0
    end = len(points) - 1
    
    if end < 1:
        return points
        
    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d
            
    if dmax > epsilon:
        # Recursive call
        rec_results1 = rdp(points[:index+1], epsilon)
        rec_results2 = rdp(points[index:], epsilon)
        
        # Concatenate results (avoid duplicating the split point)
        return np.vstack((rec_results1[:-1], rec_results2))
    else:
        return np.vstack((points[0], points[end]))

def simplify_graph_geometry(nodes, edges, epsilon=1.0):
    """
    Applies RDP simplification to the polyline geometry of every edge.
    Does NOT change graph topology (nodes remain fixed).
    Just reduces the 'intermediates' count.
    """
    count_before = 0
    count_after = 0
    
    for i in range(len(edges)):
        u, v, w, pts = edges[i]
        
        if len(pts) > 0:
            p1 = nodes[u]
            p2 = nodes[v]
            
            # Construct full chain: Start -> Intermediates -> End
            full_chain = np.vstack([p1, np.array(pts), p2])
            count_before += len(full_chain)
            
            # Simplify
            # Epsilon = max deviation allowed (in voxels/mm)
            simplified_chain = rdp(full_chain, epsilon)
            count_after += len(simplified_chain)
            
            # Extract intermediates back
            # remove start and end
            if len(simplified_chain) > 2:
                new_intermediates = simplified_chain[1:-1].tolist()
            else:
                new_intermediates = []
                
            edges[i] = [u, v, w, new_intermediates]
            
    return nodes, edges

# ----------------------------------------------------------------------------
# Radius Estimation
# ----------------------------------------------------------------------------
def compute_edge_radii(nodes, edges, edt_volume, pitch, origin):
    """
    Assigns radius to each edge based on the Euclidean Distance Transform (EDT).
    Radius = Median EDT value along the edge's polyline path.
    Updates edges in-place to include radius.
    """
    updated_edges = []
    
    for e in edges:
        # Unpack existing
        u, v, w = e[0], e[1], e[2]
        pts = e[3] if len(e) > 3 else []
        
        # Construct full chain for sampling
        p1 = nodes[u]
        p2 = nodes[v]
        full_chain = np.vstack([p1, np.array(pts) if len(pts)>0 else np.empty((0,3)), p2])
        
        # Convert World Coords back to Voxel Indices for lookup
        # World coords are [nelx, nely, nelz] but edt_volume.shape = (nely, nelx, nelz)
        # Index = (Coord - Origin - pitch/2) / pitch
        raw_indices = (full_chain - origin - (pitch * 0.5)) / pitch
        raw_indices = np.round(raw_indices).astype(int)

        # Clip to volume bounds to be safe
        D, H, W = edt_volume.shape  # D=nely, H=nelx, W=nelz
        # raw_indices[:, 0] = nelx index (world X), raw_indices[:, 1] = nely index (world Y)
        nelx_idx = np.clip(raw_indices[:, 0], 0, H-1)
        nely_idx = np.clip(raw_indices[:, 1], 0, D-1)
        nelz_idx = np.clip(raw_indices[:, 2], 0, W-1)

        # Lookup EDT using (nely, nelx, nelz) array indexing
        radii_samples = edt_volume[nely_idx, nelx_idx, nelz_idx]
        
        # Compute Representative Radius
        # Median is robust to thin joints or thick centers
        avg_radius = np.median(radii_samples) * pitch
        
        # Ensure minimum radius (at least half pitch?)
        avg_radius = max(avg_radius, pitch * 0.5)
        
        # Append radius to edge tuple
        # New format: [u, v, w, intermediates, radius]
        if len(e) >= 5:
            e[4] = avg_radius
            updated_edges.append(e)
        else:
            updated_edges.append([u, v, w, pts, avg_radius])
            
    return nodes, updated_edges

def compute_uniform_radii(nodes, edges, total_voxel_volume, pitch):
    """Compute a uniform radius matching the total voxel volume (Yin volume-matching).

    Solves ``r0 = sqrt(Volume_Voxels / (π × Total_Length))`` so that the
    total frame volume equals the input solid voxel volume.
    """
    # 1. Calculate Total Length
    total_length = 0.0
    
    for e in edges:
        # e: [u, v, w, pts, (radius?)]
        u, v = e[0], e[1]
        pts = e[3] if len(e) > 3 else []
        
        # Calculate length of the polyline
        # If pts exist, sum segments. Else just Straight Line distance.
        p1 = nodes[u]
        p2 = nodes[v]
        
        if len(pts) > 0:
            full_chain = np.vstack([p1, np.array(pts), p2])
            # Sum distances between consecutive points
            diffs = np.diff(full_chain, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            edge_len = np.sum(seg_lengths)
        else:
            edge_len = np.linalg.norm(p1 - p2)
            
        total_length += edge_len
        
    if total_length < 1e-6:
        return nodes, edges # Avoid div/0
        
    # 2. Solve for r0
    # V = A * L => A = V / L
    uniform_area = total_voxel_volume / total_length
    uniform_r = np.sqrt(uniform_area / np.pi)
    
    print(f"     [Volume Match] Total Len: {total_length:.2f}, Vol: {total_voxel_volume:.2f}")
    print(f"     [Volume Match] Uniform Radius: {uniform_r:.4f}")
    
    # 3. Apply to edges
    updated_edges = []
    for e in edges:
        # Preserve u,v,w,pts. Overwrite/Append radius.
        new_e = list(e[:4])
        new_e.append(uniform_r)
        updated_edges.append(new_e)
        
    return nodes, updated_edges

def ensure_nodes_at_bounding_extrema(nodes_dict, edges, node_tags=None):
    """
    Checks if global min/max coordinates (X/Y/Z) are covered by Nodes.
    If an extremity lies on an edge (intermediate point), splits the edge there.
    Ensures tips (Min Y, Max Y, etc.) exist as physical Nodes for BC/Loading.

    nodes_dict: {id: [x, y, z]}
    edges: list of [u, v, w, pts, rad...]
    node_tags: dict {node_id: tag} for BC nodes (fixed/loaded) to preserve
    """
    if node_tags is None:
        node_tags = {}
    def point_to_line_distance(point, line_start, line_end):
        """Compute perpendicular distance from point to line segment."""
        v = line_end - line_start
        w = point - line_start
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(w)
        c2 = np.dot(v, v)
        if c1 >= c2:
            return np.linalg.norm(point - line_end)
        b = c1 / c2
        proj = line_start + b * v
        return np.linalg.norm(point - proj)

    print("  [Post] Removing isolated outlier nodes...")
    nodes_dict, edges = remove_isolated_nodes(nodes_dict, edges, node_tags=node_tags)

    print("  [Post] Checking Extremities for Node Coverage...")
    
    # 1. Collect all geometry points from edges (pts)
    edge_points = []
    edge_map = {} # (edge_idx, pt_rel_idx) -> coord
    
    for i, e in enumerate(edges):
        if len(e) >= 4:
            pts = e[3]
            for j, p in enumerate(pts):
                edge_points.append(p)
                edge_map[(i, j)] = np.array(p)
                
    node_coords = list(nodes_dict.values())
    if not node_coords and not edge_points:
        return nodes_dict, edges

    all_pts = np.array(node_coords + edge_points)
    if len(all_pts) == 0:
        return nodes_dict, edges

    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    
    targets = [
        (0, mins[0], "Min X"), (0, maxs[0], "Max X"),
        (1, mins[1], "Min Y"), (1, maxs[1], "Max Y"),
        (2, mins[2], "Min Z"), (2, maxs[2], "Max Z")
    ]
    
    splits_todo = []
    splits_todo = []
    tol = 1.5 
    snap_tol = 5.0 # Max distance to just MOVE an existing node instead of splitting 
    
    new_nodes_dict = nodes_dict.copy()
    next_node_id = max(new_nodes_dict.keys()) + 1 if new_nodes_dict else 0
    new_edges_list = []
    
    # Identify splits
    marked_edges = set()
    
    for axis, target_val, name in targets:
        # Check if existing node covers it
        covered = False
        for nid, coord in new_nodes_dict.items():
            if abs(coord[axis] - target_val) < tol:
                covered = True
                break
        
        if not covered:
            print(f"    Target {name} ({target_val:.1f}) needs a Node.")

            # NOTE: Disabled node snapping to prevent creating long edges by moving
            # distant nodes to boundary planes. The edge-splitting fallback is safer.
                
        if not covered:
            # 2. Find best candidate point in edges (Split)
            # VALIDATION: Only consider intermediate points close to the edge line
            best_dist = 9999.9
            best_cand = None # (e_idx, pt_idx)
            max_perpendicular_error = 3.0  # Points must be within 3mm of edge line

            for (e_idx, pt_idx), val in edge_map.items():
                # First check: is this point near the target extremity?
                axis_dist = abs(val[axis] - target_val)
                if axis_dist >= tol:
                    continue

                # Second check: is this intermediate point geometrically reasonable?
                # Check if it's close to the straight line between edge endpoints
                u, v = edges[e_idx][0], edges[e_idx][1]
                line_start = np.array(nodes_dict[u], dtype=float)
                line_end = np.array(nodes_dict[v], dtype=float)
                point = np.array(val, dtype=float)
                perp_error = point_to_line_distance(point, line_start, line_end)

                if perp_error > max_perpendicular_error:
                    # This intermediate point is an outlier far from the edge line
                    continue

                if axis_dist < best_dist:
                    best_dist = axis_dist
                    best_cand = (e_idx, pt_idx)

            if best_cand:
                e_idx, p_idx = best_cand
                if e_idx not in marked_edges:
                    splits_todo.append((e_idx, p_idx, axis, target_val))
                    marked_edges.add(e_idx)
                    print(f"      Splitting Edge {e_idx} at point {p_idx} (Dist={best_dist:.2f})")
    
    if not splits_todo:
        return nodes_dict, edges
        
    # Apply splits
    split_map = {s[0]: s[1] for s in splits_todo}
    
    for i, e in enumerate(edges):
        if i in split_map:
            # Split this edge
            p_idx = split_map[i]
            pts = e[3]
            u, v = e[0], e[1]
            rad = e[4] if len(e) > 4 else 1.0
            
            # New Node
            split_coord = pts[p_idx]
            new_id = next_node_id
            new_nodes_dict[new_id] = split_coord
            next_node_id += 1
            
            # Segment 1: u -> new
            seg1_pts = pts[:p_idx]
            new_edges_list.append([u, new_id, 1.0, seg1_pts, rad])
            
            # Segment 2: new -> v
            seg2_pts = pts[p_idx+1:] 
            new_edges_list.append([new_id, v, 1.0, seg2_pts, rad])
        else:
            new_edges_list.append(e)
            
    return new_nodes_dict, new_edges_list
