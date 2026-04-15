"""Stage 1 — Skeleton Reconstruction.

Converts a voxel density field (Top3D ``.npz`` output) into a beam/plate
graph suitable for FEM optimisation.

Entry points
------------
``reconstruct_npz(npz_path, output_json, **kwargs)``
    Direct Python API — call from ``run_pipeline.py`` or tests without
    spawning a subprocess.

``export_to_json(nodes, edges, output_path, ...)``
    Serialise a skeleton graph to the canonical ``[u, v, r]`` JSON format.

CLI
---
.. code-block:: bash

    python -m src.pipelines.baseline_yin.reconstruct \\
        --input top3d_result.npz --output stage1.json --hybrid

Pipeline position
-----------------
``run_pipeline.py`` → *Stage 1 — this module* → size_opt → layout_opt
"""
import sys
import os
import argparse
import numpy as np
import json
import open3d as o3d

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(REPO_ROOT)

try:
    from src.curves.spline import fit_cubic_bezier, sample_curve_points
    _CURVES_AVAILABLE = True
except ImportError:
    _CURVES_AVAILABLE = False

from src.pipelines.baseline_yin.thinning import thin_grid_yin
from src.pipelines.baseline_yin.graph import extract_graph
from src.pipelines.baseline_yin.postprocessing import (
    graph_to_arrays,
    collapse_short_edges,
    prune_branches,
    recheck_graph,
    remove_disconnected_components,
    simplify_graph_geometry,
    smooth_graph_curves,
    compute_edge_radii,
    compute_uniform_radii,
    classify_edge_curvature,
)
from scipy.ndimage import distance_transform_edt
from src.pipelines.baseline_yin.visualization import (
    viz_voxels, viz_voxels_density, show_density_colorbar,
    viz_graph, show_step,
    viz_iterative_thinning, viz_skeleton_classification,
    viz_graph_comparison, viz_graph_radii,
    viz_zone_classification, save_zone_visualization
)


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def export_to_json(nodes_dict, edges, output_path, pitch, history=None, target_volume=None, design_bounds=None, node_tags=None, plates=None, plate_mode="bspline", curved=False, load_force=None, vol_thresh=0.3, curve_threshold=None, graph_stages=None):
    """
    Exports clean graph to JSON for FreeCAD Macro.
    """
    if history is None: history = []
    print(f"[Export] Nodes: {len(nodes_dict)}, Edges: {len(edges)}")
    sorted_ids = sorted(nodes_dict.keys())
    id_map = {old: new for new, old in enumerate(sorted_ids)}
    nodes_list_out = [list(nodes_dict[i]) for i in sorted_ids]
    edges_list_out, curves, default_radius = [], [], 0.5 * pitch
    n_curved, n_straight = 0, 0
    _edge_seen = set()  # deduplicate (u, v) pairs

    for edge in edges:
        u_old, v_old = edge[0], edge[1]
        if u_old not in id_map or v_old not in id_map: continue
        u_new, v_new, radius = id_map[u_old], id_map[v_old], edge[4] if len(edge) >= 5 else default_radius
        _ekey = (min(u_new, v_new), max(u_new, v_new))
        if _ekey in _edge_seen:
            continue
        _edge_seen.add(_ekey)
        edges_list_out.append([u_new, v_new, radius])
        pts = edge[3] if len(edge) >= 4 else []
        p_start = np.array(nodes_dict[u_old])
        p_end   = np.array(nodes_dict[v_old])
        if curved and _CURVES_AVAILABLE:
            # Per-edge classification: only fit Bézier for genuinely curved edges
            dev_thresh = curve_threshold if curve_threshold is not None else 0.3 * pitch
            # threshold=0 means force ALL edges curved (legacy behaviour)
            if dev_thresh > 0:
                metrics = classify_edge_curvature(p_start, p_end, pts, pitch,
                                                   deviation_thresh=dev_thresh)
                edge_is_curved = metrics['is_curved']
            else:
                edge_is_curved = True

            if edge_is_curved:
                ctrl_pts = fit_cubic_bezier(p_start, p_end, pts)  # (2, 3)
                p1, p2 = ctrl_pts[0], ctrl_pts[1]
                curve_pts = sample_curve_points(p_start, p1, p2, p_end, radius, N=20)
                curves.append({
                    "ctrl_pts": ctrl_pts.tolist(),
                    "points":   curve_pts,
                    "radius":   float(radius)
                })
                n_curved += 1
            else:
                curve_pts = [list(p_start) + [radius], list(p_end) + [radius]]
                curves.append({"points": curve_pts})
                n_straight += 1
        else:
            # Legacy straight-line / polyline representation
            if len(pts) == 0:
                curve_pts = [list(p_start) + [radius], list(p_end) + [radius]]
            else:
                full_pts = [p_start] + [np.array(p) for p in pts] + [p_end]
                curve_pts = [list(p) + [radius] for p in full_pts]
            curves.append({"points": curve_pts})

    if curved and (n_curved + n_straight) > 0:
        print(f"[Export] Edge classification: {n_curved} curved, {n_straight} straight "
              f"(threshold={curve_threshold if curve_threshold is not None else 0.3 * pitch:.2f}mm)")

    node_tags_out = {str(id_map[old_id]): tag_val for old_id, tag_val in node_tags.items() if old_id in id_map} if node_tags else {}

    # Process plates with surface fitting
    plates_out = plates if plates is not None else []
    
    # Import the new surface fitting module (lazy import to avoid circular dep issues if any)
    try:
        from src.pipelines.baseline_yin.surface_fitting import fit_bspline_surface
    except ImportError:
        print("[Warning] Could not import surface_fitting. B-Spline surfaces will be skipped.")
        def fit_bspline_surface(*args, **kwargs): return None

    for plate in plates_out:
        # Remap connection nodes
        if "connection_node_ids" in plate:
            plate["connection_node_ids"] = [
                id_map[old_id] for old_id in plate["connection_node_ids"]
                if old_id in id_map
            ]
            
        # Fit B-Spline Surface to SKELETON VOXEL CENTERS (mid-surface)
        # Only if plate_mode is 'bspline' to save time
        if plate_mode == "bspline":
            mid_pts = np.array(plate.get("voxels", []))
            if len(mid_pts) > 16:
                try:
                    bspline_data = fit_bspline_surface(mid_pts)
                    if bspline_data:
                        plate["bspline_surface"] = bspline_data
                        print(f"  [Plate {plate.get('id', '?')}] Fitted B-Spline mid-surface ({len(mid_pts)} skeleton pts).")
                except Exception as e:
                    print(f"  [Plate {plate.get('id', '?')}] B-Spline fit failed: {e}")

    # Extract Explicit Joints
    # A joint is where a beam meets a plate.
    # We look at 'connection_node_ids' in plates.
    joints = []
    
    # helper: adjacency list for the final graph
    adj = {}
    for edge in edges_list_out:
        u, v, r = edge[0], edge[1], edge[2]
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append((v, r))
        adj[v].append((u, r))
        
    for p_idx, plate in enumerate(plates_out):
        conn_ids = plate.get("connection_node_ids", [])
        for nid in conn_ids:
            # nid is the new ID in id_map
            # Find connected beams
            if nid in adj:
                neighbors = adj[nid]
                # For each neighbor, if it's NOT in the plate (checking valid beam connection)
                # But connection_node_ids are part of the graph?
                # Usually connection nodes are graph nodes on the boundary.
                # We assume any edge connected to this node is a beam leaving the plate.
                
                pos = nodes_list_out[nid]
                
                for nbr_id, radius in neighbors:
                    # Check if nbr_id is also in the plate?
                    # If nbr_id is also a connection node or inside the plate, it might be a "surface graph" edge (if any).
                    # But in our pipeline, beams attached to plates are what we care about.
                    # Simple heuristic: treat all incident edges as joints for now.
                    
                    nbr_pos = nodes_list_out[nbr_id]
                    
                    # Calculate direction (Beam Normal)
                    # vec = nbr - node
                    vec = np.array(nbr_pos) - np.array(pos)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        direction = (vec / norm).tolist()
                    else:
                        direction = [0, 0, 1]
                        
                    joints.append({
                        "plate_id": plate.get("id", p_idx),
                        "node_id": nid,
                        "location": pos,
                        "direction": direction, # Vector pointing INTO the beam
                        "radius": radius,
                        "type": "fillet"
                    })

    meta = {"method": "Baseline Yin", "pitch": pitch, "units": "mm", "target_volume": target_volume, "design_bounds": design_bounds, "plate_mode": plate_mode, "load_force": load_force if load_force is not None else [0.0, -1.0, 0.0], "vol_thresh": vol_thresh}
    if graph_stages:
        meta["graph_stages"] = graph_stages
    data = {
        "metadata": meta,
        "graph": {"nodes": nodes_list_out, "edges": edges_list_out, "node_tags": node_tags_out},
        "curves": curves, "history": history, "plates": plates_out, "joints": joints
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"[Export] Saved {len(curves)} segments and {len(plates) if plates else 0} plates to {output_path}")

def _rescale_radii_to_volume(nodes_dict, edges_list_raw, target_volume):
    """Scale EDT-derived radii so total beam volume matches the input solid volume.

    Computes per-edge length, sums π·r²·L for current radii, then applies a
    uniform scale factor ``s = sqrt(V_target / V_current)`` to all radii.
    This preserves the EDT-based relative distribution while matching volume.
    """
    total_vol = 0.0
    edge_lengths = []
    for e in edges_list_raw:
        u, v = e[0], e[1]
        pts = e[3] if len(e) > 3 else []
        r = e[4] if len(e) >= 5 else 1.0
        p1 = np.array(nodes_dict[u])
        p2 = np.array(nodes_dict[v])
        if len(pts) > 0:
            chain = np.vstack([p1, np.array(pts), p2])
            L = np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1))
        else:
            L = np.linalg.norm(p1 - p2)
        edge_lengths.append(L)
        total_vol += np.pi * r * r * L

    if total_vol < 1e-12 or target_volume < 1e-12:
        return nodes_dict, edges_list_raw

    scale = np.sqrt(target_volume / total_vol)
    for e in edges_list_raw:
        if len(e) >= 5:
            e[4] = e[4] * scale

    new_vol = total_vol * scale * scale  # π·(sr)²·L = s²·π·r²·L
    print(f"     [Volume Match] EDT radii rescaled: factor={scale:.3f}, "
          f"volume {total_vol:.1f} -> {new_vol:.1f} mm³ (target={target_volume:.1f})")
    return nodes_dict, edges_list_raw


def _run(args):

    # Derive output directory and base name
    output_dir = os.path.dirname(args.output_json) or "."
    output_basename = os.path.splitext(os.path.basename(args.output_json))[0]
    args.out_dir = output_dir  # Add to args for later use

    bc_tags = None
    if args.input_mesh.endswith(".npz"):
        print(f"[1] Loading Top3D Result from {args.input_mesh}...")
        npz_data = np.load(args.input_mesh)
        rho = npz_data['rho']
        if 'bc_tags' in npz_data:
            bc_tags = npz_data['bc_tags'].astype(np.int32)
            print(f"    BC Tags loaded: {int(np.sum(bc_tags == 1))} fixed, {int(np.sum(bc_tags == 2))} loaded")
        solid = (rho > args.vol_thresh).astype(bool)
        dims = solid.shape
        origin = np.array([0.0, 0.0, 0.0])
        # dims = (nely, nelx, nelz); reorder to [nelx, nely, nelz] world coords
        mesh_bounds = np.array([[0.0, 0.0, 0.0], [dims[1]*args.pitch, dims[0]*args.pitch, dims[2]*args.pitch]])
    graph_stages = []
    def record_stage(name, nodes_d, edges_l):
        graph_stages.append({'name': name, 'nodes': len(nodes_d), 'edges': len(edges_l)})

    history_snapshots = []
    def capture_snapshot(name, nodes_d, edges_l, plates=None, curved=False):
        sorted_ids = sorted(nodes_d.keys())
        id_map = {old: new for new, old in enumerate(sorted_ids)}
        n_list = [list(nodes_d[i]) for i in sorted_ids]
        e_list = []
        for e in edges_l:
            if e[0] not in id_map or e[1] not in id_map:
                continue
            u, v = id_map[e[0]], id_map[e[1]]
            radius = e[4] if len(e) >= 5 else 1.0
            waypoints = e[3] if len(e) >= 4 else []
            entry = [u, v, radius, waypoints]
            # Fit Bézier ctrl_pts from waypoints when curved
            if curved and _CURVES_AVAILABLE and len(waypoints) > 0:
                p_start = np.array(nodes_d[e[0]])
                p_end = np.array(nodes_d[e[1]])
                try:
                    ctrl = fit_cubic_bezier(p_start, p_end, waypoints)
                    entry.append(ctrl.tolist())  # [5] = ctrl_pts
                except Exception:
                    pass
            e_list.append(entry)
        history_snapshots.append({"type": "graph", "step": name, "nodes": n_list, "edges": e_list, "plates": plates if plates is not None else []})

    def capture_voxel_snapshot(name, mask, colors=None):
        # mask.shape = (nely, nelx, nelz); reorder to [nelx, nely, nelz] world coords
        raw = np.argwhere(mask)
        reordered = raw[:, [1, 0, 2]]
        pts = origin + (reordered * args.pitch) + (args.pitch * 0.5)
        if len(pts) > 50000:
            pts = pts[::2]
            if colors is not None:
                colors = colors[::2]
        snapshot = {"type": "voxels", "step": name, "points": pts.tolist()}
        if colors is not None:
            snapshot["colors"] = colors.tolist() if isinstance(colors, np.ndarray) else colors
        history_snapshots.append(snapshot)

    def _bc_tag_colors(mask):
        """Colour skeleton voxels by bc_tags: grey default, blue=fixed, orange=loaded, magenta=passive."""
        coords = np.argwhere(mask)  # (N, 3) in (nely, nelx, nelz) order
        colors = np.full((len(coords), 3), 0.7)  # default grey
        for i, (y, x, z) in enumerate(coords):
            tag = bc_tags[y, x, z] if bc_tags is not None else 0
            if tag == 1:    # fixed/support
                colors[i] = [0.2, 0.4, 1.0]   # blue
            elif tag == 2:  # loaded
                colors[i] = [1.0, 0.6, 0.0]   # orange
            elif tag == 3:  # passive void
                colors[i] = [0.8, 0.2, 0.8]   # magenta
        return colors

    # Calculate Target Volume (Input Volume)
    voxel_vol = np.sum(solid) * (args.pitch**3)
    print(f"[Volume] Initial Solid Volume: {voxel_vol:.2f} mm^3")
    
    # Calculate EDT early (needed for thinning protection)
    edt_v = distance_transform_edt(solid)
    
    # Compute density-gradient colours (quantised to N_BINS for FreeCAD compatibility)
    N_BINS = 10
    _lo, _hi = float(args.vol_thresh), 1.0
    _idx = np.argwhere(solid)
    _density_vals = rho[_idx[:, 0], _idx[:, 1], _idx[:, 2]]
    _t_raw = np.clip((_density_vals - _lo) / (_hi - _lo), 0.0, 1.0)
    _t_bin = np.floor(_t_raw * N_BINS) / N_BINS   # at most N_BINS unique values
    try:
        import matplotlib.pyplot as _plt
        _voxel_colors = _plt.get_cmap("RdYlGn")(_t_bin)[:, :3]
    except ImportError:
        # Manual two-segment R→Y→G fallback
        _voxel_colors = np.zeros((len(_t_bin), 3))
        _m_lo = _t_bin <= 0.5
        _s = _t_bin[_m_lo] * 2.0
        _voxel_colors[_m_lo] = np.c_[np.ones_like(_s), _s, np.zeros_like(_s)]
        _m_hi = ~_m_lo
        _s = (_t_bin[_m_hi] - 0.5) * 2.0
        _voxel_colors[_m_hi] = np.c_[1.0 - _s, np.ones_like(_s), np.zeros_like(_s)]
    capture_voxel_snapshot("1_Initial_Voxels", solid, colors=_voxel_colors)

    if args.visualize:
        try:
            show_density_colorbar(vol_thresh=args.vol_thresh, title="Voxel Density")
            show_step(
                "1. Initial Voxel Volume (red=low density, green=high density)",
                [viz_voxels_density(solid, rho, args.pitch, origin, vol_thresh=args.vol_thresh)]
            )
        except Exception as e:
            print(f"  [Viz Warning] Initial voxel visualization failed: {e}")
        
    # ===================================================================
    # HYBRID vs PURE BEAM PATH
    # ===================================================================
    plates_data = []

    if args.hybrid:
        # ---------------------------------------------------------------
        # HYBRID PATH: Single thinning (mode=3) → post-thinning classify
        # ---------------------------------------------------------------
        from src.pipelines.baseline_yin.graph import classify_skeleton_post_thinning
        from src.pipelines.baseline_yin.plate_extraction import (
            extract_plates_v2, recover_plate_regions_from_skeleton
        )
        from src.pipelines.baseline_yin.joint_creation import create_beam_plate_joints

        # [2] Two-pass thinning: mode=3 (surfaces+curves) and mode=0 (curves only)
        n_solid = int(np.sum(solid))
        print(f"[2] Hybrid thinning (mode=3 + mode=0, {n_solid} solid voxels)...")

        # Pass 1: mode=3 — preserves both surface points AND curve endpoints
        if args.visualize:
            skeleton, iter_map = thin_grid_yin(
                solid.copy(), tags=bc_tags, max_iters=args.skel_iters,
                record_iterations=True, mode=3, edt=edt_v
            )
            try:
                show_step("2. Hybrid Thinning (Surface + Curve)", [viz_iterative_thinning(iter_map, args.pitch, origin)])
            except Exception as e:
                print(f"  [Viz Warning] Thinning visualization failed: {e}")
        else:
            skeleton = thin_grid_yin(
                solid.copy(), tags=bc_tags, max_iters=args.skel_iters,
                mode=3, edt=edt_v
            )

        # Pass 2: mode=0 — curves only (collapses surface sheets)
        skeleton_curve = thin_grid_yin(
            solid.copy(), tags=bc_tags, max_iters=args.skel_iters,
            mode=0, edt=edt_v
        )
        n_mode3 = int(np.sum(skeleton > 0))
        n_mode0 = int(np.sum(skeleton_curve > 0))
        print(f"  mode=3: {n_mode3} voxels, mode=0: {n_mode0} voxels, "
              f"difference: {n_mode3 - n_mode0} surface voxels")

        capture_voxel_snapshot("2_Hybrid_Skeleton", skeleton, colors=_bc_tag_colors(skeleton))

        # [2.5] Post-thinning classification via two-pass topological difference
        print(f"[2.5] Post-thinning classification (two-pass thinning comparison)...")
        zone_mask, plate_labels, zone_stats = classify_skeleton_post_thinning(
            skeleton,
            min_plate_size=args.min_plate_size,
            flatness_ratio=args.flatness,
            junction_thresh=args.junction_thresh,
            min_avg_neighbors=args.min_neighbors,
            solid=solid,
            skeleton_curve=skeleton_curve
        )

        # Re-capture skeleton with zone classification colors for debugging
        # Red (1,0,0) for beams (zone=2), Cyan (0,1,1) for plates (zone=1)
        skeleton_coords = np.argwhere(skeleton > 0)
        colors = np.zeros((len(skeleton_coords), 3))
        for i, (z, y, x) in enumerate(skeleton_coords):
            if zone_mask[z, y, x] == 1:  # Plate
                colors[i] = [0, 1, 1]  # Cyan
            elif zone_mask[z, y, x] == 2:  # Beam
                colors[i] = [1, 0, 0]  # Red
        # Replace the plain skeleton snapshot with color-coded version
        for i, snap in enumerate(history_snapshots):
            if snap.get("step") == "2_Hybrid_Skeleton":
                history_snapshots[i] = {
                    "type": "voxels",
                    "step": "2_Hybrid_Skeleton",
                    "points": (origin + skeleton_coords[:, [1, 0, 2]] * args.pitch + args.pitch * 0.5).tolist(),
                    "colors": colors.tolist()
                }
                break

        # Save zone classification visualization
        capture_voxel_snapshot("2.5_Zone_Classification", zone_mask > 0)
        viz_path = os.path.join(args.out_dir, f"{output_basename}_2_5_zones.png")
        if save_zone_visualization(zone_mask, args.pitch, origin, viz_path):
            print(f"  -> Zone visualization saved: {viz_path}")

        if args.visualize:
            try:
                zone_geoms = viz_zone_classification(zone_mask, args.pitch, origin)
                show_step("2.5. Post-Thinning Classification (Red=Beams, Cyan=Plates)", zone_geoms)
            except Exception as e:
                print(f"  [Viz Warning] Zone classification visualization failed: {e}")

            # Skeleton classification viz skipped — transparent point cloud causes hang

        # [3a] Recover full-thickness plate regions from skeleton surface voxels
        print(f"[3a] Recovering plate regions from skeleton...")
        recovered_zone, recovered_labels = recover_plate_regions_from_skeleton(
            zone_mask, plate_labels, solid, edt_v
        )

        # [3b] Extract beam graph from beam skeleton voxels
        beam_skeleton = (skeleton > 0) & (zone_mask == 2)
        beam_tags = bc_tags.copy() if bc_tags is not None else None
        if beam_tags is not None:
            beam_tags[~(beam_skeleton)] = 0

        print(f"[3b] Extracting beam graph ({int(np.sum(beam_skeleton))} beam voxels)...")
        nodes_arr, edges_list_raw, v_types, node_tags = extract_graph(
            beam_skeleton.astype(np.uint8), args.pitch, origin,
            tags=beam_tags, hybrid_mode=False
        )
        nodes_dict = {i: nodes_arr[i] for i in range(len(nodes_arr))}
        record_stage("Raw graph", nodes_dict, edges_list_raw)

        # [3c] Extract plate geometry (solid mesh + mid-surface + thickness)
        plate_skeleton = (skeleton > 0) & (zone_mask == 1)
        print(f"[3c] Extracting plate geometry ({zone_stats['n_plate_regions']} regions)...")
        plates_data = extract_plates_v2(
            plate_skeleton.astype(np.uint8), plate_labels, solid, edt_v,
            args.pitch, origin, zone_mask=recovered_zone, bc_tags=bc_tags,
            recovered_labels=recovered_labels
        )

        # [3d] Create beam-plate joints (snap to nearest plate boundary vertex)
        print(f"[3d] Creating beam-plate joints...")
        nodes_dict, edges_list_raw, node_tags, plates_data = create_beam_plate_joints(
            nodes_dict, edges_list_raw, node_tags, plates_data,
            recovered_zone, args.pitch, origin, snap_distance=4.0
        )
        record_stage("Beam-plate joints", nodes_dict, edges_list_raw)

    else:
        # ---------------------------------------------------------------
        # PURE BEAM PATH: Standard curve-preserving thinning
        # ---------------------------------------------------------------
        # Strip tag=2 (loaded) from thinning tags so only supports (tag=1)
        # are protected.  Loaded voxels form multi-voxel bands that block
        # mode=0 from collapsing surfaces to curves if left protected.
        beam_thin_tags = None
        if bc_tags is not None:
            beam_thin_tags = bc_tags.copy()
            beam_thin_tags[beam_thin_tags == 2] = 0

        print(f"[2] Thinning (Max Iters={args.skel_iters}, Mode=0)...")
        if args.visualize:
            skeleton, iter_map = thin_grid_yin(
                solid.copy(), tags=beam_thin_tags, max_iters=args.skel_iters,
                record_iterations=True, mode=0, edt=edt_v
            )
            show_step("2a. Iterative Removal", [viz_iterative_thinning(iter_map, args.pitch, origin)])
            # 2b skeleton classification viz skipped — transparent point cloud causes hang
        else:
            skeleton = thin_grid_yin(
                solid.copy(), tags=beam_thin_tags, max_iters=args.skel_iters,
                mode=0, edt=edt_v
            )

        capture_voxel_snapshot("2_Skeleton_Voxels", skeleton, colors=_bc_tag_colors(skeleton))

        print(f"[3] Extracting Graph...")
        nodes_arr, edges_list_raw, v_types, node_tags = extract_graph(
            skeleton, args.pitch, origin, tags=bc_tags, hybrid_mode=False
        )
        nodes_dict = {i: nodes_arr[i] for i in range(len(nodes_arr))}
        record_stage("Raw graph", nodes_dict, edges_list_raw)

    capture_snapshot("3_Raw_Graph", nodes_dict, edges_list_raw, plates=plates_data,
                     curved=getattr(args, 'curved', False))
    if args.visualize:
        geoms = viz_graph(nodes_arr, edges_list_raw)
        geoms.append(viz_voxels(skeleton, args.pitch, origin, [0.9, 0.8, 0.8], 0.3))
        show_step("3. Raw Graph + Skeleton Overlay", geoms)

    # Clean up spurious intermediate points that create loops
    print(f"[3.5] Cleaning up edge polylines...")
    from src.pipelines.baseline_yin.postprocessing import clean_edge_polylines
    nodes_dict, edges_list_raw = clean_edge_polylines(nodes_dict, edges_list_raw)
    record_stage("Clean polylines", nodes_dict, edges_list_raw)

    if args.collapse_thresh > 0:
        nodes_dict, edges_list_raw = collapse_short_edges(nodes_dict, edges_list_raw, args.collapse_thresh, node_tags=node_tags)
        # Bridge any components disconnected by collapse
        nodes_dict, edges_list_raw = remove_disconnected_components(nodes_dict, edges_list_raw, node_tags=node_tags)
        record_stage("Collapse short edges", nodes_dict, edges_list_raw)
        capture_snapshot("4A_Collapsed", nodes_dict, edges_list_raw, plates=plates_data)
        if args.visualize:
            n_arr, e_arr = graph_to_arrays(nodes_dict, edges_list_raw)
            show_step("4A. Edge Collapse", viz_graph(n_arr, e_arr))

    if args.prune_len > 0:
        nodes_dict, edges_list_raw = prune_branches(nodes_dict, edges_list_raw, args.prune_len, node_tags=node_tags)
        # Bridge any components disconnected by pruning
        nodes_dict, edges_list_raw = remove_disconnected_components(nodes_dict, edges_list_raw, node_tags=node_tags)
        record_stage("Prune branches", nodes_dict, edges_list_raw)
        capture_snapshot("4B_Pruned", nodes_dict, edges_list_raw, plates=plates_data)

    if getattr(args, 'curved', False):
        # Curved mode: smooth waypoints instead of RDP (preserves curve shape)
        smooth_iters = getattr(args, 'smooth_iters', 5)
        smooth_stride = getattr(args, 'smooth_decimate', 1)
        nodes_dict, edges_list_raw = smooth_graph_curves(
            nodes_dict, edges_list_raw,
            iterations=smooth_iters, decimate_stride=smooth_stride)
        record_stage("Smooth curves", nodes_dict, edges_list_raw)
        capture_snapshot("4C_Smoothed_Curves", nodes_dict, edges_list_raw,
                         plates=plates_data, curved=True)
    elif args.rdp > 0:
        # Straight mode: RDP simplification (existing path)
        nodes_dict, edges_list_raw = simplify_graph_geometry(nodes_dict, edges_list_raw, args.rdp)
        record_stage("RDP simplification", nodes_dict, edges_list_raw)
        capture_snapshot("4C_Simplified_RDP", nodes_dict, edges_list_raw, plates=plates_data)

    if args.radius_mode == 'edt':
        nodes_dict, edges_list_raw = compute_edge_radii(nodes_dict, edges_list_raw, distance_transform_edt(solid, sampling=[args.pitch]*3), args.pitch, origin)
        # Scale EDT radii so total beam volume matches input solid volume
        nodes_dict, edges_list_raw = _rescale_radii_to_volume(nodes_dict, edges_list_raw, voxel_vol)
    elif args.radius_mode == 'uniform':
        nodes_dict, edges_list_raw = compute_uniform_radii(nodes_dict, edges_list_raw, voxel_vol, args.pitch)

    from src.pipelines.baseline_yin.postprocessing import ensure_nodes_at_bounding_extrema, merge_colocated_nodes
    if not getattr(args, 'skip_extrema', False):
        nodes_dict, edges_list_raw = ensure_nodes_at_bounding_extrema(nodes_dict, edges_list_raw, node_tags=node_tags)
        # Merge co-located nodes to eliminate zero-length edges that cause singular FEM
        nodes_dict, edges_list_raw = merge_colocated_nodes(nodes_dict, edges_list_raw, node_tags=node_tags, tol=0.1)
        record_stage("Merge colocated nodes", nodes_dict, edges_list_raw)

    # ── Skeleton-level symmetry enforcement (mirror-half) ──────────────
    _sym_str = getattr(args, 'symmetry', None)
    if _sym_str and mesh_bounds is not None:
        from src.optimization.symmetry import parse_symmetry_planes, mirror_half_skeleton
        _sym_tol = getattr(args, 'sym_tol', None) or 1.5 * args.pitch
        sym_planes = parse_symmetry_planes(_sym_str, mesh_bounds.tolist())
        _pre_n = len(nodes_dict)
        _pre_e = len(edges_list_raw)
        for plane in sym_planes:
            nodes_dict, edges_list_raw, node_tags = mirror_half_skeleton(
                nodes_dict, edges_list_raw, plane,
                node_tags=node_tags, tol=_sym_tol)
        _post_n = len(nodes_dict)
        _post_e = len(edges_list_raw)
        if _pre_e != (_post_e + 1) // 2 * 2 - 1:  # rough check
            print(f"  [Symmetry] Graph: {_pre_n} nodes/{_pre_e} edges -> {_post_n} nodes/{_post_e} edges")
        record_stage("Skeleton symmetry", nodes_dict, edges_list_raw)

    capture_snapshot("5_Extrema_Fixed", nodes_dict, edges_list_raw, plates=plates_data)

    if args.visualize:
        n_arr, e_arr = graph_to_arrays(nodes_dict, edges_list_raw)
        radii_viz = [e[4] if len(e)>=5 else args.pitch for e in edges_list_raw]
        show_step("5. Final Clean Graph (Color=Radius, Extrema Fixed)", viz_graph_radii(n_arr, e_arr, np.array(radii_viz)))
        
    if args.load_fx is not None or args.load_fy is not None or args.load_fz is not None:
        load_force = [
            args.load_fx if args.load_fx is not None else 0.0,
            args.load_fy if args.load_fy is not None else 0.0,
            args.load_fz if args.load_fz is not None else 0.0
        ]
    else:
        load_force = None

    export_to_json(nodes_dict, edges_list_raw, args.output_json, args.pitch, history=history_snapshots, target_volume=voxel_vol, design_bounds=mesh_bounds.tolist(), node_tags=node_tags, plates=plates_data, plate_mode=args.plate_mode, curved=args.curved, load_force=load_force, vol_thresh=args.vol_thresh, curve_threshold=getattr(args, 'curve_threshold', None), graph_stages=graph_stages)
    print("Done.")

def reconstruct_npz(npz_path, output_json, **kwargs):
    """Direct Python API — mirrors all CLI flags of reconstruct.py.

    Args:
        npz_path:    Path to Top3D .npz output file.
        output_json: Destination JSON path.
        **kwargs:    Any CLI flag as a keyword argument (pitch, max_iters,
                     collapse_thresh, prune_len, rdp_epsilon, radius_mode,
                     vol_thresh, hybrid, detect_plates, plate_thickness_ratio,
                     min_plate_size, flatness_ratio, junction_thresh,
                     min_avg_neighbors, load_fx, load_fy, load_fz,
                     plate_mode, curved, visualize).
    """
    import types
    # Normalise public kwarg aliases → internal _run() attribute names.
    # Needed because run_pipeline.py uses the public CLI names while the
    # defaults dict (and _run()) use the reconstruct.py argparse dest names.
    _alias = {
        'flatness_ratio':        'flatness',
        'rdp_epsilon':           'rdp',
        'min_avg_neighbors':     'min_neighbors',
        'max_iters':             'skel_iters',
        'plate_thickness_ratio': 'plate_thickness',
    }
    for pub, priv in _alias.items():
        if pub in kwargs:
            kwargs.setdefault(priv, kwargs.pop(pub))
    defaults = dict(
        pitch=1.0, skel_iters=50, collapse_thresh=2.0, prune_len=5.0,
        rdp=0.0, radius_mode='edt', vol_thresh=0.3,
        hybrid=False, detect_plates='auto', plate_thickness=0.15,
        min_plate_size=4, flatness=3.0, junction_thresh=4,
        min_neighbors=3.0, load_fx=None, load_fy=None, load_fz=None,
        plate_mode='bspline', curved=False, visualize=False,
        curve_threshold=None, smooth_iters=5, smooth_decimate=1,
        symmetry=None, sym_tol=None, skip_extrema=False,
    )
    defaults.update(kwargs)
    _run(types.SimpleNamespace(input_mesh=npz_path, output_json=output_json, **defaults))


def main():
    parser = argparse.ArgumentParser(description="Baseline Yin 3D Skeletonization Pipeline")
    parser.add_argument("input_mesh", help="Input NPZ or STL")
    parser.add_argument("output_json", help="Output JSON")
    parser.add_argument("--pitch", type=float, default=1.0)
    parser.add_argument("--skel_iters", "--max_iters", type=int, default=50)
    parser.add_argument("--collapse_thresh", type=float, default=2.0)
    parser.add_argument("--prune_len", type=float, default=5.0)
    parser.add_argument("--rdp", "--rdp_epsilon", type=float, default=0.0)
    parser.add_argument("--radius_mode", type=str, default="edt", choices=["edt", "uniform"])
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vol_thresh", type=float, default=0.3)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--detect_plates", type=str, default="auto", choices=["auto", "off", "force"])
    parser.add_argument("--plate_thickness", "--plate_thickness_ratio", type=float, default=0.15)
    parser.add_argument("--min_plate_size", type=int, default=4)
    parser.add_argument("--flatness", "--flatness_ratio", type=float, default=3.0)
    parser.add_argument("--junction_thresh", type=int, default=4)
    parser.add_argument("--min_neighbors", "--min_avg_neighbors", type=float, default=3.0)
    parser.add_argument("--load_fx", type=float, default=None)
    parser.add_argument("--load_fy", type=float, default=None)
    parser.add_argument("--load_fz", type=float, default=None)
    parser.add_argument("--plate_mode", type=str, default="bspline", choices=["bspline", "voxel", "mesh"])
    parser.add_argument("--curved", action="store_true")
    parser.add_argument("--curve_threshold", type=float, default=None,
                        help="Max perpendicular deviation (mm) for classifying edge as curved. "
                             "Below this → straight beam. Default: 0.3×pitch. Set to 0 for all curved.")
    parser.add_argument("--smooth_iters", type=int, default=5,
                        help="[curved] Laplacian smoothing iterations for edge waypoints (default: 5)")
    parser.add_argument("--smooth_decimate", type=int, default=1,
                        help="[curved] Decimation stride after smoothing (1=none, 2=half)")
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
