"""Two-plane symmetry enforcement for beam frame optimization.

Detects symmetric node/edge pairs by reflecting positions across user-specified
planes, then enforces symmetry via radius averaging (size opt) and a soft penalty
+ exact projection (layout opt).

Usage from run_pipeline.py::

    from src.optimization.symmetry import (
        parse_symmetry_planes, find_symmetric_node_pairs,
        find_symmetric_edge_pairs, average_symmetric_radii,
        enforce_exact_node_symmetry,
    )
"""

import numpy as np
from scipy.spatial import KDTree


def parse_symmetry_planes(symmetry_str, design_bounds):
    """Parse CLI symmetry string and compute plane definitions.

    Args:
        symmetry_str: Comma-separated plane names, e.g. ``"xz,yz"``.
            ``'xz'`` = mirror about XZ plane (flips Y),
            ``'yz'`` = mirror about YZ plane (flips X),
            ``'xy'`` = mirror about XY plane (flips Z).
        design_bounds: ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]``.

    Returns:
        list of dict: Each with ``name``, ``axis`` (int, flipped coordinate),
        ``center`` (float, midpoint on that axis).
    """
    bounds_min = np.array(design_bounds[0], dtype=float)
    bounds_max = np.array(design_bounds[1], dtype=float)
    center = (bounds_min + bounds_max) / 2.0

    plane_map = {
        'xz': {'axis': 1, 'center': center[1]},  # flip Y
        'yz': {'axis': 0, 'center': center[0]},  # flip X
        'xy': {'axis': 2, 'center': center[2]},  # flip Z
    }

    planes = []
    for name in symmetry_str.lower().replace(' ', '').split(','):
        name = name.strip()
        if name in plane_map:
            planes.append({
                'name': name,
                'axis': plane_map[name]['axis'],
                'center': plane_map[name]['center'],
            })
        else:
            print(f"[Symmetry] Warning: unknown plane '{name}', skipping")
    return planes


def find_symmetric_node_pairs(nodes, planes, tol=0.5, locked_nodes=None):
    """Find node pairs that are reflections across symmetry planes.

    Args:
        nodes: ``(N, 3)`` array of node positions.
        planes: List of plane dicts from :func:`parse_symmetry_planes`.
        tol: Max distance (mm) for two nodes to be considered partners.
        locked_nodes: Set of BC node indices (still paired, but flagged).

    Returns:
        dict: Keyed by plane name. Each value is a dict with:
            - ``node_pairs``: list of ``(i, j)`` with ``i < j``
            - ``on_plane_nodes``: list of node indices on the plane
            - ``unmatched_nodes``: list of nodes with no partner
            - ``node_map``: bidirectional ``{i: j, j: i}`` dict
            - ``axis``: int (flipped coordinate)
            - ``center``: float (plane midpoint)
    """
    if locked_nodes is None:
        locked_nodes = set()

    tree = KDTree(nodes)
    result = {}

    for plane in planes:
        ax = plane['axis']
        ctr = plane['center']
        pname = plane['name']

        # Reflect all nodes across plane
        reflected = nodes.copy()
        reflected[:, ax] = 2 * ctr - reflected[:, ax]

        # Query nearest neighbor for each reflected position
        dists, indices = tree.query(reflected)

        node_pairs = []
        on_plane = []
        unmatched = []
        node_map = {}
        paired = set()

        for i in range(len(nodes)):
            # On-plane check
            if abs(nodes[i, ax] - ctr) < tol:
                if i not in paired:
                    on_plane.append(i)
                    paired.add(i)
                continue

            j = indices[i]
            if i == j:
                # Self-match (on-plane node)
                if i not in paired:
                    on_plane.append(i)
                    paired.add(i)
                continue

            if dists[i] > tol:
                if i not in paired:
                    unmatched.append(i)
                continue

            if i in paired or j in paired:
                continue

            # Valid pair: ensure i < j for canonical ordering
            a, b = (min(i, j), max(i, j))
            node_pairs.append((a, b))
            node_map[a] = b
            node_map[b] = a
            paired.add(a)
            paired.add(b)

        # Any remaining unmatched
        for i in range(len(nodes)):
            if i not in paired:
                unmatched.append(i)

        result[pname] = {
            'node_pairs': node_pairs,
            'on_plane_nodes': on_plane,
            'unmatched_nodes': unmatched,
            'node_map': node_map,
            'axis': ax,
            'center': ctr,
        }

    return result


def find_symmetric_edge_pairs(edges, node_pair_info):
    """Find edge pairs where both endpoints are symmetric partners.

    Args:
        edges: ``(M, 2)`` array of edge connectivity.
        node_pair_info: Dict from :func:`find_symmetric_node_pairs`.

    Returns:
        dict: Keyed by plane name. Values are lists of ``(edge_i, edge_j)``
        tuples. Self-symmetric edges (both nodes on-plane) use
        ``(edge_i, edge_i)``.
    """
    # Build lookup: sorted (u,v) → edge index
    edge_lookup = {}
    for idx, (u, v) in enumerate(edges):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        edge_lookup[key] = idx

    result = {}
    for pname, info in node_pair_info.items():
        nmap = info['node_map']
        on_plane = set(info['on_plane_nodes'])
        pairs = []
        seen = set()

        for idx, (u, v) in enumerate(edges):
            if idx in seen:
                continue
            u, v = int(u), int(v)

            # Map endpoints to their partners
            u2 = nmap.get(u, u if u in on_plane else None)
            v2 = nmap.get(v, v if v in on_plane else None)

            if u2 is None or v2 is None:
                continue  # No symmetric partner for this edge

            partner_key = (min(u2, v2), max(u2, v2))
            partner_idx = edge_lookup.get(partner_key)

            if partner_idx is None:
                continue  # Partner edge doesn't exist

            if partner_idx in seen:
                continue

            pairs.append((idx, partner_idx))
            seen.add(idx)
            seen.add(partner_idx)

        result[pname] = pairs

    return result


def average_symmetric_radii(radii, edge_pairs_all_planes):
    """Average radii of symmetric edge groups (exact enforcement).

    Uses Union-Find to handle multi-plane transitivity: with XZ+YZ,
    up to 4 edges are grouped and get a single averaged radius.

    Args:
        radii: ``(M,)`` array of beam radii.
        edge_pairs_all_planes: Dict from :func:`find_symmetric_edge_pairs`.

    Returns:
        ndarray: ``(M,)`` symmetrized radii.
    """
    n = len(radii)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Union all symmetric edge pairs across all planes
    for pairs in edge_pairs_all_planes.values():
        for ei, ej in pairs:
            union(ei, ej)

    # Group edges by root and average
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    r_new = radii.copy()
    for members in groups.values():
        if len(members) > 1:
            avg = np.mean([radii[m] for m in members])
            for m in members:
                r_new[m] = avg

    return r_new


def enforce_exact_node_symmetry(nodes, node_pair_info, planes, locked_nodes=None):
    """Project nodes onto exact symmetric positions by averaging pairs.

    For each pair ``(i, j)`` across a plane:
        - If both free: average and mirror
        - If one locked: free node mirrors the locked one
        - If both locked: skip

    On-plane nodes have their across-plane coordinate set to the center.

    With multiple planes, sequential enforcement can break earlier planes.
    This function iterates until convergence (typically 2-3 passes).

    Args:
        nodes: ``(N, 3)`` array (copied, not modified in-place).
        node_pair_info: Dict from :func:`find_symmetric_node_pairs`.
        planes: List of plane dicts.
        locked_nodes: Set of BC node indices that must not move.

    Returns:
        ndarray: ``(N, 3)`` symmetrized positions.
    """
    if locked_nodes is None:
        locked_nodes = set()

    nodes = nodes.copy()

    # Iterate to handle multi-plane interference
    for _iteration in range(5):
        max_change = 0.0
        nodes_prev = nodes.copy()

        for plane in planes:
            pname = plane['name']
            ax = plane['axis']
            ctr = plane['center']
            info = node_pair_info.get(pname, {})

            # Enforce on-plane nodes
            for i in info.get('on_plane_nodes', []):
                if i not in locked_nodes and i < len(nodes):
                    nodes[i, ax] = ctr

            # Enforce paired nodes — only adjust the flipped axis
            for (i, j) in info.get('node_pairs', []):
                if i >= len(nodes) or j >= len(nodes):
                    continue

                i_locked = i in locked_nodes
                j_locked = j in locked_nodes

                if i_locked and j_locked:
                    continue
                elif i_locked:
                    # Mirror i's position to j (only the flipped axis)
                    nodes[j, ax] = 2 * ctr - nodes[i, ax]
                    # Copy non-flipped axes from i to j
                    for a in range(3):
                        if a != ax:
                            nodes[j, a] = nodes[i, a]
                elif j_locked:
                    nodes[i, ax] = 2 * ctr - nodes[j, ax]
                    for a in range(3):
                        if a != ax:
                            nodes[i, a] = nodes[j, a]
                else:
                    # Average the flipped axis, copy non-flipped axes
                    avg_ax = (nodes[i, ax] + (2 * ctr - nodes[j, ax])) / 2.0
                    nodes[i, ax] = avg_ax
                    nodes[j, ax] = 2 * ctr - avg_ax
                    # Non-flipped axes: average both nodes
                    for a in range(3):
                        if a != ax:
                            avg = (nodes[i, a] + nodes[j, a]) / 2.0
                            nodes[i, a] = avg
                            nodes[j, a] = avg

        max_change = np.max(np.abs(nodes - nodes_prev))
        if max_change < 1e-10:
            break

    return nodes


def mirror_half_skeleton(nodes_dict, edges_list_raw, plane, node_tags=None, tol=0.5):
    """Enforce exact symmetry by keeping one half and mirroring it.

    Keeps all nodes on the 'negative' side of the plane (coord < center)
    plus on-plane nodes, discards the rest, then mirrors nodes and edges
    to create a perfectly symmetric skeleton.

    Args:
        nodes_dict: ``{id: [x,y,z], ...}`` node positions.
        edges_list_raw: list of edge tuples ``[u, v, ...]`` (may include
            polyline waypoints and radius at index 4).
        plane: dict with ``axis`` (int) and ``center`` (float).
        node_tags: ``{node_id: tag_value}`` for BC nodes (preserved).
        tol: distance from plane center to consider a node "on-plane".

    Returns:
        (nodes_dict, edges_list_raw, node_tags) — the symmetric skeleton.
    """
    if node_tags is None:
        node_tags = {}

    ax = plane['axis']
    ctr = plane['center']

    # Classify nodes: keep (coord <= center+tol), on-plane, or discard
    keep_ids = set()
    on_plane_ids = set()
    for nid, pos in nodes_dict.items():
        if abs(pos[ax] - ctr) < tol:
            keep_ids.add(nid)
            on_plane_ids.add(nid)
        elif pos[ax] < ctr:
            keep_ids.add(nid)

    # Pick the side with more nodes (in case structure is biased to one side)
    discard_ids = set(nodes_dict.keys()) - keep_ids
    other_side = set()
    for nid, pos in nodes_dict.items():
        if pos[ax] > ctr + tol:
            other_side.add(nid)
    neg_side = keep_ids - on_plane_ids

    if len(other_side) > len(neg_side):
        # More structure on the positive side — keep that instead
        keep_ids = other_side | on_plane_ids
        neg_side = other_side  # the side we'll mirror FROM

    # Split cross-plane edges: if one endpoint is kept and the other discarded,
    # create a new on-plane node at the intersection and keep the half-edge.
    next_id = max(nodes_dict.keys()) + 1 if nodes_dict else 0
    n_split = 0
    new_edges_from_split = []
    edges_to_remove = set()
    for idx, e in enumerate(edges_list_raw):
        u_in = e[0] in keep_ids
        v_in = e[1] in keep_ids
        if u_in == v_in:
            continue  # both kept or both discarded — no split needed
        # One in, one out — split at plane
        kept_node = e[0] if u_in else e[1]
        disc_node = e[1] if u_in else e[0]
        p_kept = np.array(nodes_dict[kept_node], dtype=float)
        p_disc = np.array(nodes_dict[disc_node], dtype=float)
        denom = p_disc[ax] - p_kept[ax]
        if abs(denom) < 1e-12:
            continue  # parallel to plane, skip
        t = (ctr - p_kept[ax]) / denom
        t = np.clip(t, 0.0, 1.0)
        p_new = p_kept + t * (p_disc - p_kept)
        p_new[ax] = ctr  # exact
        # Create on-plane node
        nodes_dict[next_id] = p_new.tolist()
        on_plane_ids.add(next_id)
        keep_ids.add(next_id)
        # Create half-edge from kept_node to new on-plane node
        new_e = list(e)
        new_e[0] = kept_node
        new_e[1] = next_id
        # Clear intermediate waypoints (they may be on the wrong side)
        if len(new_e) > 3:
            new_e[3] = []
        new_edges_from_split.append(new_e)
        edges_to_remove.add(idx)
        next_id += 1
        n_split += 1

    # Remove original cross-plane edges and add the split halves
    edges_list_raw = [e for idx, e in enumerate(edges_list_raw) if idx not in edges_to_remove]
    edges_list_raw.extend(new_edges_from_split)

    if n_split > 0:
        _axis_name = 'XYZ'[ax]
        print(f"  [Symmetry] Split {n_split} cross-plane edge(s) at {_axis_name}={ctr:.1f}")

    # Keep only edges where both endpoints are in keep_ids
    kept_edges = []
    for e in edges_list_raw:
        if e[0] in keep_ids and e[1] in keep_ids:
            kept_edges.append(list(e))

    # Snap on-plane nodes exactly to center
    for nid in on_plane_ids:
        nodes_dict[nid][ax] = ctr

    # Create mirrored nodes
    next_id = max(nodes_dict.keys()) + 1 if nodes_dict else 0
    mirror_map = {}  # old_id -> new_mirrored_id
    for nid in keep_ids:
        if nid in on_plane_ids:
            mirror_map[nid] = nid  # on-plane nodes map to themselves
            continue
        pos = list(nodes_dict[nid])
        pos[ax] = 2 * ctr - pos[ax]  # reflect
        nodes_dict[next_id] = pos
        mirror_map[nid] = next_id
        # Propagate BC tags to mirrored nodes
        if nid in node_tags:
            node_tags[next_id] = node_tags[nid]
        next_id += 1

    # Create mirrored edges
    mirrored_edges = []
    for e in kept_edges:
        u_mir = mirror_map.get(e[0])
        v_mir = mirror_map.get(e[1])
        if u_mir is None or v_mir is None:
            continue
        if u_mir == e[0] and v_mir == e[1]:
            continue  # both on-plane, edge already exists
        new_e = list(e)
        new_e[0] = u_mir
        new_e[1] = v_mir
        # Mirror polyline waypoints if present (indices 2 and 3)
        if len(new_e) > 2 and isinstance(new_e[2], list):
            new_e[2] = [_mirror_point(p, ax, ctr) for p in new_e[2]]
        if len(new_e) > 3 and isinstance(new_e[3], list):
            new_e[3] = [_mirror_point(p, ax, ctr) for p in new_e[3]]
        mirrored_edges.append(new_e)

    # Remove discarded nodes
    all_keep = set(keep_ids)
    for nid in mirror_map.values():
        all_keep.add(nid)
    for nid in list(nodes_dict.keys()):
        if nid not in all_keep:
            del nodes_dict[nid]
            node_tags.pop(nid, None)

    edges_list_raw = kept_edges + mirrored_edges

    n_kept = len(neg_side)
    n_on_plane = len(on_plane_ids)
    n_mirrored = sum(1 for v in mirror_map.values() if v != mirror_map.get(v, v) or v not in on_plane_ids)
    n_mirrored = len([nid for nid, mid in mirror_map.items() if mid != nid])
    print(f"  [Symmetry] Mirror-half: kept {n_kept} nodes + {n_on_plane} on-plane, "
          f"created {n_mirrored} mirrored nodes, "
          f"{len(kept_edges)} kept + {len(mirrored_edges)} mirrored edges")

    return nodes_dict, edges_list_raw, node_tags


def _mirror_point(point, axis, center):
    """Reflect a 3D point across a plane."""
    p = list(point)
    p[axis] = 2 * center - p[axis]
    return p


def get_on_plane_bound_overrides(nodes, node_pair_info, planes, tol=0.5):
    """Compute bound overrides for on-plane nodes in layout opt.

    Args:
        nodes: ``(N, 3)`` positions.
        node_pair_info: From :func:`find_symmetric_node_pairs`.
        planes: List of plane dicts.
        tol: Distance tolerance.

    Returns:
        dict: ``{node_idx: {axis: center_value}}``.
    """
    overrides = {}
    for plane in planes:
        pname = plane['name']
        ax = plane['axis']
        ctr = plane['center']
        info = node_pair_info.get(pname, {})
        for i in info.get('on_plane_nodes', []):
            overrides.setdefault(i, {})[ax] = ctr
    return overrides
