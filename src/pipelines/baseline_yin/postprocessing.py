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
- :func:`classify_edge_curvature` — per-edge straight/curved classification
- :func:`ensure_nodes_at_bounding_extrema` — guarantee nodes at domain corners
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform


def _point_to_line_distance(point, line_start, line_end):
    """Compute perpendicular distance from a point to a line segment."""
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


def classify_edge_curvature(p_start, p_end, intermediate_pts, pitch=1.0,
                             deviation_thresh=None):
    """Decide whether a skeleton edge warrants curved Bézier representation.

    Computes max perpendicular deviation of intermediate skeleton waypoints
    from the straight chord line.  Edges whose deviation is below the
    threshold are classified as straight.

    Parameters
    ----------
    p_start, p_end : array-like, shape (3,)
        Edge endpoint coordinates (mm).
    intermediate_pts : list of array-like
        Skeleton polyline waypoints between the endpoints.
    pitch : float
        Voxel pitch (mm).  Used to set default threshold.
    deviation_thresh : float or None
        Max perpendicular deviation threshold (mm).  Edges below this are
        straight.  If None, defaults to ``0.3 * pitch``.

    Returns
    -------
    dict with keys:
        'is_curved'      : bool
        'max_deviation'   : float  (mm)
        'arc_ratio'       : float  (polyline_length / chord_length)
        'n_waypoints'     : int
    """
    if deviation_thresh is None:
        deviation_thresh = 0.3 * pitch

    p0 = np.asarray(p_start, dtype=float)
    p1 = np.asarray(p_end, dtype=float)
    chord_len = float(np.linalg.norm(p1 - p0))

    if len(intermediate_pts) == 0 or chord_len < 1e-12:
        return {
            'is_curved': False,
            'max_deviation': 0.0,
            'arc_ratio': 1.0,
            'n_waypoints': 0,
        }

    pts = [np.asarray(p, dtype=float) for p in intermediate_pts]

    # Max perpendicular deviation from chord
    max_dev = max(_point_to_line_distance(p, p0, p1) for p in pts)

    # Arc-length ratio
    full_path = [p0] + pts + [p1]
    arc_len = sum(np.linalg.norm(full_path[i + 1] - full_path[i])
                  for i in range(len(full_path) - 1))
    arc_ratio = arc_len / chord_len

    return {
        'is_curved': max_dev >= deviation_thresh,
        'max_deviation': float(max_dev),
        'arc_ratio': float(arc_ratio),
        'n_waypoints': len(pts),
    }


def clean_edge_polylines(nodes_dict, edges):
    """
    Removes intermediate points that deviate significantly from the straight
    line between edge endpoints. Fixes spurious skeleton loops.

    nodes_dict: {id: [x, y, z]}
    edges: list of [u, v, w, pts, rad...]
    """
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
            perp_error = _point_to_line_distance(pt_arr, p_start, p_end)

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

    # 0. Filter stale edges referencing deleted nodes
    edges = [e for e in edges if e[0] in nodes and e[1] in nodes]

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
                # Skip nodes not in nodes dict (stale edge reference)
                if u not in nodes:
                    continue
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
                    # Never collapse an edge between two tagged nodes.
                    # This prevents cascade-collapse of tag=2 load rings
                    # and preserves spatial separation between BC nodes.
                    continue
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
                # Recalculate weights using actual node positions to prevent
                # stale weights from causing cascade collapses after centroid merge
                new_edges = []
                for j, e in enumerate(edges):
                    eu, ev, ew, epts = e[0], e[1], e[2], e[3] if len(e)>3 else []

                    if eu == remove: eu = keep
                    if ev == remove: ev = keep

                    if eu == ev: continue  # Remove Self Loop

                    # Recalculate weight as Euclidean distance between current positions
                    ew = float(np.linalg.norm(np.array(nodes[eu]) - np.array(nodes[ev])))
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
# Algorithm 4.6: Reconnect Disconnected Components (Structural Cleanup)
# ----------------------------------------------------------------------------
def remove_disconnected_components(nodes, edges, node_tags=None):
    """
    Reconnects disconnected components to the main graph (the component
    containing fixed nodes, tag=1).  For each orphan component, finds the
    closest node-pair to the main component and adds a bridge edge.
    Tiny components (≤2 nodes, no tags) are removed instead of bridged.
    """
    from collections import defaultdict, deque

    if node_tags is None:
        node_tags = {}

    adj = defaultdict(set)
    for e in edges:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])

    visited = set()
    components = []
    for nid in sorted(nodes):
        if nid not in visited:
            comp = set()
            q = deque([nid])
            while q:
                n = q.popleft()
                if n in visited:
                    continue
                visited.add(n)
                comp.add(n)
                for nb in adj[n]:
                    if nb not in visited:
                        q.append(nb)
            components.append(comp)

    if len(components) <= 1:
        return nodes, edges

    # Identify main component (the one with fixed nodes)
    main_comp = None
    orphan_comps = []
    for comp in components:
        has_fixed = any(node_tags.get(n) == 1 for n in comp)
        if has_fixed:
            if main_comp is None or len(comp) > len(main_comp):
                if main_comp is not None:
                    orphan_comps.append(main_comp)
                main_comp = comp
            else:
                orphan_comps.append(comp)
        else:
            orphan_comps.append(comp)

    if main_comp is None or not orphan_comps:
        return nodes, edges

    print(f"  [Cleanup] {len(components)} components: main={len(main_comp)} nodes, "
          f"{len(orphan_comps)} orphan(s)")

    # For each orphan component: bridge to main graph
    remove_nodes = set()
    for comp in orphan_comps:
        has_edges = any(e for e in edges if e[0] in comp or e[1] in comp)

        # Isolated nodes with no edges and no tags: remove
        if len(comp) == 1 and not has_edges and not any(n in node_tags for n in comp):
            remove_nodes.update(comp)
            continue

        # Find closest node pair between orphan and main component
        best_dist = float('inf')
        best_orphan = None
        best_main = None
        orphan_coords = np.array([nodes[n] for n in comp])
        orphan_ids = list(comp)
        for mn in main_comp:
            mc = np.array(nodes[mn])
            dists = np.linalg.norm(orphan_coords - mc, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < best_dist:
                best_dist = dists[idx]
                best_orphan = orphan_ids[idx]
                best_main = mn

        # If nodes are co-located (dist < 0.5mm), MERGE instead of bridging
        # to avoid zero-length edges that cause singular FEM matrices.
        MERGE_THRESH = 0.5
        if best_dist < MERGE_THRESH:
            # Redirect all edges from best_orphan to best_main
            for e in edges:
                if e[0] == best_orphan:
                    e[0] = best_main
                if e[1] == best_orphan:
                    e[1] = best_main
            # Redirect all edges from other orphan nodes to their closest main
            # equivalent if needed (for multi-node orphans, merge all close nodes)
            for on in comp:
                if on == best_orphan:
                    continue
                oc = np.array(nodes[on])
                # Find closest main node for this orphan node
                d_to_main = float('inf')
                closest_main = best_main
                for mn in main_comp:
                    d = np.linalg.norm(oc - np.array(nodes[mn]))
                    if d < d_to_main:
                        d_to_main = d
                        closest_main = mn
                if d_to_main < MERGE_THRESH:
                    for e in edges:
                        if e[0] == on:
                            e[0] = closest_main
                        if e[1] == on:
                            e[1] = closest_main
                    remove_nodes.add(on)
            # Transfer tags and remove the merged node
            if best_orphan in node_tags and best_orphan not in [1, 2]:
                pass  # Don't transfer bridge tags
            remove_nodes.add(best_orphan)
            main_comp.update(comp)
            print(f"  [Cleanup] Merged {len(comp)}-node component: "
                  f"node {best_orphan} -> node {best_main} (dist={best_dist:.1f})")
        else:
            # Add bridge edge and tag the orphan endpoint to prevent pruning
            edges.append([best_orphan, best_main, best_dist, []])
            # Tag=9 (bridge) makes this node immune to pruning but ignored by TaggedProblem
            if best_orphan not in node_tags:
                node_tags[best_orphan] = 9
            main_comp.update(comp)  # Merge into main for subsequent orphans
            print(f"  [Cleanup] Bridged {len(comp)}-node component: "
                  f"node {best_orphan} -> node {best_main} (dist={best_dist:.1f})")

    if remove_nodes:
        nodes = {k: v for k, v in nodes.items() if k not in remove_nodes}
        edges = [e for e in edges if e[0] not in remove_nodes and e[1] not in remove_nodes]
        for n in remove_nodes:
            node_tags.pop(n, None)

    # Remove self-loops and duplicate edges created by merging
    seen = set()
    clean_edges = []
    for e in edges:
        if e[0] == e[1]:
            continue  # Self-loop
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key in seen:
            continue  # Duplicate
        seen.add(key)
        clean_edges.append(e)
    if len(clean_edges) < len(edges):
        n_removed = len(edges) - len(clean_edges)
        print(f"  [Cleanup] Removed {n_removed} self-loop/duplicate edges after merging")
    edges.clear()
    edges.extend(clean_edges)

    return nodes, edges


# ----------------------------------------------------------------------------
# Algorithm 4.7: Merge Co-located Nodes (Zero-Length Edge Elimination)
# ----------------------------------------------------------------------------
def merge_colocated_nodes(nodes, edges, node_tags=None, tol=0.1):
    """
    Merges nodes closer than ``tol`` mm to eliminate zero-length edges
    that cause singular FEM stiffness matrices.  Preserves tagged nodes
    (tag 1/2) as merge targets when possible.
    """
    if node_tags is None:
        node_tags = {}

    node_ids = sorted(nodes.keys())
    if len(node_ids) < 2:
        return nodes, edges

    coords = np.array([nodes[n] for n in node_ids])
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    pairs = tree.query_pairs(tol)

    if not pairs:
        return nodes, edges

    # Build merge map: for each pair, keep the node with a higher-priority tag
    # Priority: tag 1 (fixed) > tag 2 (loaded) > tag 9 (bridge) > no tag
    TAG_PRIORITY = {1: 3, 2: 2, 9: 1}
    merge_map = {}  # old_id -> new_id

    # Union-Find to handle transitive merges
    parent = {nid: nid for nid in node_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Keep the one with better tag
        pa = TAG_PRIORITY.get(node_tags.get(ra, 0), 0)
        pb = TAG_PRIORITY.get(node_tags.get(rb, 0), 0)
        if pa >= pb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for i, j in pairs:
        union(node_ids[i], node_ids[j])

    # Build final merge map
    removed = set()
    for nid in node_ids:
        root = find(nid)
        if root != nid:
            merge_map[nid] = root
            removed.add(nid)

    if not removed:
        return nodes, edges

    # Remap edges
    for e in edges:
        if e[0] in merge_map:
            e[0] = merge_map[e[0]]
        if e[1] in merge_map:
            e[1] = merge_map[e[1]]

    # Transfer tags from removed nodes to their merge targets
    for old_id, new_id in merge_map.items():
        if old_id in node_tags and new_id not in node_tags:
            node_tags[new_id] = node_tags[old_id]
        node_tags.pop(old_id, None)

    # Remove merged nodes
    nodes = {k: v for k, v in nodes.items() if k not in removed}

    # Remove self-loops and duplicates
    seen = set()
    clean_edges = []
    for e in edges:
        if e[0] == e[1]:
            continue
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key in seen:
            continue
        seen.add(key)
        clean_edges.append(e)

    n_merged = len(removed)
    n_edges_removed = len(edges) - len(clean_edges)
    print(f"  [Post] Merged {n_merged} co-located nodes (tol={tol}mm), "
          f"removed {n_edges_removed} degenerate edges")

    edges.clear()
    edges.extend(clean_edges)

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


def smooth_graph_curves(nodes, edges, iterations=5, decimate_stride=1):
    """Smooth edge waypoints using Laplacian smoothing for curved beam mode.

    Applies iterative Laplacian smoothing to each edge's intermediate waypoints
    to reduce voxel-grid noise while preserving overall curve shape.  Endpoints
    (graph nodes) are never moved.

    Optionally decimates waypoints after smoothing by keeping every Nth point
    to reduce point count without significant shape loss.

    Args:
        nodes: Node dictionary ``{id: [x, y, z]}``.
        edges: Edge list ``[[u, v, w, waypoints, ...], ...]``.
        iterations: Number of Laplacian smoothing passes (default 5).
        decimate_stride: Keep every Nth waypoint after smoothing (1=all).

    Returns:
        (nodes, edges) with smoothed (and optionally decimated) waypoints.
    """
    from src.pipelines.baseline_yin.graph import smooth_polyline

    count_before = 0
    count_after = 0

    for i in range(len(edges)):
        u, v, w = edges[i][0], edges[i][1], edges[i][2]
        pts = edges[i][3] if len(edges[i]) > 3 else []

        if len(pts) < 2:
            continue

        count_before += len(pts)

        # Build full polyline with endpoints for context
        p_start = list(nodes[u])
        p_end = list(nodes[v])
        full_chain = [p_start] + [list(p) for p in pts] + [p_end]
        smoothed = smooth_polyline(full_chain, iterations=iterations)
        # Strip endpoints back off
        new_pts = smoothed[1:-1]

        # Decimate: keep every Nth point
        if decimate_stride > 1 and len(new_pts) > 2:
            decimated = [new_pts[j] for j in range(0, len(new_pts), decimate_stride)]
            # Always include the last waypoint if dropped
            if (len(new_pts) - 1) % decimate_stride != 0:
                decimated.append(new_pts[-1])
            new_pts = decimated

        count_after += len(new_pts)

        # Rebuild edge preserving any fields beyond [3]
        edges[i] = [u, v, w, new_pts] + list(edges[i][4:])

    print(f"  [Smooth] {count_before} waypoints -> {count_after} waypoints "
          f"({iterations} iters, stride={decimate_stride})")

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
