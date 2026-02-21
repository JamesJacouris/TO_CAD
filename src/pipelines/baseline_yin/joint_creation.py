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

    for p_idx, plate in enumerate(plates_data):
        # Prefer skeleton voxel centers (mid-surface) over hull vertices
        verts = np.array(plate.get("voxels", plate.get("vertices", [])))
        if len(verts) == 0:
            continue
        for v_idx in range(len(verts)):
            all_plate_verts.append(verts[v_idx])
            plate_id_per_vert.append(p_idx)
            plate_local_idx.append(v_idx)

    if len(all_plate_verts) == 0:
        return nodes_dict, edges, node_tags, plates_data

    all_plate_verts = np.array(all_plate_verts)
    tree = cKDTree(all_plate_verts)

    # Compute node degrees for endpoint detection
    degree = {}
    for e in edges:
        u, v = e[0], e[1]
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    n_snapped = 0
    n_new_joints = 0
    snapped_nodes = set()

    # Pass 1: Snap existing beam nodes within snap_distance
    for nid, coord in list(nodes_dict.items()):
        dist, idx = tree.query(coord)
        if dist <= snap_distance:
            plate_vertex = all_plate_verts[idx]
            nodes_dict[nid] = np.array(plate_vertex)
            p_idx = plate_id_per_vert[idx]

            # Tag as fixed support (don't override load tags)
            # Junction nodes should only be fixed if they inherit a tag from original voxels
            # The original logic here was forcing fixed=1, which we now avoid.
            pass

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
            # node_tags[new_id] = 1  # Fixed support -- REMOVED artificial fixing

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

    print(f"    [Joints] Snapped {n_snapped} existing nodes, "
          f"created {n_new_joints} new junction nodes.")

    return nodes_dict, edges, node_tags, plates_data
