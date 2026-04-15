"""
Joint Creation: Creates shared nodes at beam-plate interfaces.

Snaps beam graph endpoints to the nearest plate boundary vertex to create
geometrically precise shared-node joints. Uses KD-tree for efficient
nearest-neighbor lookup.
"""

import numpy as np
from scipy.spatial import cKDTree


def create_beam_plate_joints(nodes_dict, edges, node_tags, plates_data,
                              zone_mask, pitch, origin, snap_distance=None):
    """
    Creates shared nodes at beam-plate interfaces by snapping beam endpoints
    to the nearest plate boundary vertex.

    Parameters
    ----------
    nodes_dict : dict {int: ndarray(3,)}
        Beam graph nodes (id -> world coordinate).
    edges : list of lists
        Beam graph edges [u, v, weight, intermediates, ...].
    node_tags : dict {int: int}
        BC tags for nodes (1=fixed, 2=loaded).
    plates_data : list of dict
        Plate data from extract_plates_v2. Each has 'vertices', 'connection_node_ids'.
    zone_mask : ndarray (D, H, W), int32
        Zone classification (1=beam, 2=plate).
    pitch : float
        Voxel size in mm.
    origin : ndarray (3,)
        World-space origin.
    snap_distance : float or None
        Max distance (mm) for snapping. Defaults to 2.0 * pitch.

    Returns
    -------
    nodes_dict, edges, node_tags, plates_data : updated versions
    """
    if snap_distance is None:
        snap_distance = 2.0 * pitch

    if not plates_data or not nodes_dict:
        return nodes_dict, edges, node_tags, plates_data

    # Build a combined KD-tree of all plate MID-SURFACE points (skeleton voxels),
    # not hull vertices. This ensures joints lie on the B-Spline mid-surface.
    all_plate_verts = []
    plate_id_per_vert = []
    plate_local_idx = []
    tag_per_vert = []
    
    tagged_plate_verts = []
    tagged_plate_tags = []

    for p_idx, plate in enumerate(plates_data):
        # Prefer skeleton voxel centers (mid-surface) over hull vertices
        if "mid_surface" in plate:
            verts = plate["mid_surface"].get("vertices", [])
            tags = plate["mid_surface"].get("node_tags", {})
        else:
            verts = np.array(plate.get("voxels", plate.get("vertices", [])))
            tags = {}
            
        if len(verts) == 0:
            continue
        for v_idx in range(len(verts)):
            vert = verts[v_idx]
            all_plate_verts.append(vert)
            plate_id_per_vert.append(p_idx)
            plate_local_idx.append(v_idx)
            
            # str keys in JSON, int keys otherwise
            tag_val = tags.get(v_idx, tags.get(str(v_idx), 0))
            tag_per_vert.append(tag_val)
            
            if tag_val > 0:
                tagged_plate_verts.append(vert)
                tagged_plate_tags.append(tag_val)

    if len(all_plate_verts) == 0:
        return nodes_dict, edges, node_tags, plates_data

    all_plate_verts = np.array(all_plate_verts)
    tree = cKDTree(all_plate_verts)

    tagged_tree = None
    if len(tagged_plate_verts) > 0:
        tagged_tree = cKDTree(np.array(tagged_plate_verts))

    # Per-plate role: does this plate carry loads (tag=2)?
    # If yes, untagged joints must default to tag=2 so loads reach the beams.
    # If no, untagged joints default to tag=1 (rigid support).
    plate_has_loads = {}
    plate_has_fixed = {}
    for i, t in enumerate(tag_per_vert):
        p = plate_id_per_vert[i]
        if t == 2:
            plate_has_loads[p] = True
        elif t == 1:
            plate_has_fixed[p] = True

    # Compute node degrees for endpoint detection
    degree = {}
    for e in edges:
        u, v = e[0], e[1]
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    n_snapped = 0
    n_new_joints = 0
    snapped_nodes = set()
    
    def get_tag_for_coord(coord):
        if tagged_tree is not None:
            # Search for a tagged vertex within a generous radius to ensure loads aren't missed
            dist, idx = tagged_tree.query(coord, distance_upper_bound=pitch * 5.0)
            if dist != np.inf:
                return tagged_plate_tags[idx]
        return 0

    # Track which plate each joint belongs to (for post-snap load transfer)
    joint_plate = {}  # nid -> plate index

    # Pass 1: Snap existing beam nodes within snap_distance
    for nid, coord in list(nodes_dict.items()):
        dist, idx = tree.query(coord)
        if dist <= snap_distance:
            plate_vertex = all_plate_verts[idx]
            nodes_dict[nid] = np.array(plate_vertex)
            p_idx = plate_id_per_vert[idx]
            joint_plate[nid] = p_idx

            # Inherit tag from nearby tagged plate vertices
            if nid not in node_tags or node_tags[nid] == 0:
                plate_tag = get_tag_for_coord(plate_vertex)
                if plate_tag > 0:
                    node_tags[nid] = plate_tag
                # else: resolved in post-snap pass below

            # Record connection
            if "connection_node_ids" not in plates_data[p_idx]:
                plates_data[p_idx]["connection_node_ids"] = []
            if nid not in plates_data[p_idx]["connection_node_ids"]:
                plates_data[p_idx]["connection_node_ids"].append(nid)

            snapped_nodes.add(nid)
            n_snapped += 1

    # Pass 2: For degree-1 beam endpoints NOT already snapped,
    # check extended range and create new junction nodes
    extended_snap = snap_distance * 3.0

    for nid, coord in list(nodes_dict.items()):
        if nid in snapped_nodes:
            continue
        if degree.get(nid, 0) != 1:
            continue

        dist, idx = tree.query(coord)
        if dist <= extended_snap:
            plate_vertex = all_plate_verts[idx]
            p_idx = plate_id_per_vert[idx]

            # Create a new junction node at the plate vertex
            new_id = max(nodes_dict.keys()) + 1
            nodes_dict[new_id] = np.array(plate_vertex)
            
            joint_plate[new_id] = p_idx

            # Inherit tag from nearby tagged plate vertices
            plate_tag = get_tag_for_coord(plate_vertex)
            if plate_tag > 0:
                node_tags[new_id] = plate_tag
            # else: resolved in post-snap pass below

            # Add a short connecting edge
            edge_len = float(dist)
            edges.append([nid, new_id, edge_len, [], edge_len * 0.5])

            # Update degree
            degree[nid] = degree.get(nid, 0) + 1
            degree[new_id] = 1

            # Record connection
            if "connection_node_ids" not in plates_data[p_idx]:
                plates_data[p_idx]["connection_node_ids"] = []
            if new_id not in plates_data[p_idx]["connection_node_ids"]:
                plates_data[p_idx]["connection_node_ids"].append(new_id)

            snapped_nodes.add(new_id)
            n_new_joints += 1

    # ── Post-snap tag assignment based on plate role ──────────────────
    # For each plate, check whether its loads are already transferred to
    # at least one joint.  If not, untagged joints on that plate must
    # become tag=2 so the beam FEM sees the plate's applied forces.
    # Plates that carry no load default untagged joints to tag=1 (support).
    for p_idx in set(joint_plate.values()):
        p_joints = [nid for nid, pi in joint_plate.items() if pi == p_idx]
        has_loaded_joint = any(node_tags.get(nid, 0) == 2 for nid in p_joints)
        untagged = [nid for nid in p_joints if node_tags.get(nid, 0) == 0]

        if not untagged:
            continue

        if plate_has_loads.get(p_idx, False) and not has_loaded_joint:
            # Plate carries loads but no joint inherited tag=2 →
            # transfer load through untagged joints
            for nid in untagged:
                node_tags[nid] = 2
        else:
            # Plate is support-only, or loads already transferred →
            # remaining untagged joints act as rigid support
            for nid in untagged:
                node_tags[nid] = 1

    # Count how many joint nodes are now fixed (tag=1)
    n_fixed_joints = sum(1 for nid in snapped_nodes if node_tags.get(nid, 0) == 1)
    n_loaded_joints = sum(1 for nid in snapped_nodes if node_tags.get(nid, 0) == 2)
    print(f"    [Joints] Snapped {n_snapped} existing nodes, "
          f"created {n_new_joints} new junction nodes.")
    print(f"    [Joints] Tags: {n_fixed_joints} fixed (plate support), "
          f"{n_loaded_joints} loaded (plate load transfer), "
          f"{len(snapped_nodes) - n_fixed_joints - n_loaded_joints} other")

    return nodes_dict, edges, node_tags, plates_data
