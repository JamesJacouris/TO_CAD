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
    from src.voxelization.voxelize import voxelize_mesh
except ImportError:
    voxelize_mesh = None  # Only needed for STL input, not NPZ
from src.pipelines.baseline_yin.thinning import thin_grid_yin
from src.pipelines.baseline_yin.graph import extract_graph
from src.pipelines.baseline_yin.postprocessing import (
    graph_to_arrays, 
    collapse_short_edges, 
    prune_branches,
    recheck_graph,
    simplify_graph_geometry,
    compute_edge_radii,
    compute_uniform_radii
)
from scipy.ndimage import distance_transform_edt
from src.pipelines.baseline_yin.visualization import (
    viz_voxels, viz_graph, show_step,
    viz_iterative_thinning, viz_skeleton_classification, 
    viz_graph_comparison, viz_graph_radii
)

def export_to_json(nodes, edges, output_path, pitch, history=None):
    """
    Exports clean graph to JSON compatible with FreeCAD macro.
    Nodes: dict {id: [x,y,z]}
    Edges: list [u, v, w]
    """
    # Convert to Ball-and-Stick Chains
    # The macro expects "curves" list.
    # Each curve can be a single segment [P1, P2] for now.
    
    curves = []
    
    # Pre-calculate radii? 
    # Pre-calculate radii? 
    # For now, constant radius or weight-based?
    # Macro expects [x,y,z, r].
    default_radius = pitch * 1.5 
    
    for edge in edges:
        # Check if radius is present (len 5)
        if len(edge) >= 5:
            u, v, w, pts, radius = edge[:5]
        else:
            u, v, w, pts = edge[:4]
            radius = default_radius
            
        p1 = nodes[u]
        p2 = nodes[v]
        
        # Build full curve points: Start -> Intermediates -> End
        # Format: [x, y, z, radius]
        curve_points = []
        
        # Start Node
        curve_points.append([p1[0], p1[1], p1[2], radius])
        
        # Intermediates
        for pt in pts:
            curve_points.append([pt[0], pt[1], pt[2], radius])
def export_to_json(nodes_dict, edges, output_path, pitch, history=None, target_volume=None, design_bounds=None, node_tags=None):
    """
    Exports clean graph to JSON for FreeCAD Macro.
    nodes_dict: {id: [x,y,z], ...}
    edges: list of [u, v, len, polyline, radius]
    """
    if history is None:
        history = []
        
    print(f"[Export] Nodes: {len(nodes_dict)}, Edges: {len(edges)}")
    
    # 0. Build Clean Node List (0-indexed)
    sorted_ids = sorted(nodes_dict.keys())
    id_map = {old: new for new, old in enumerate(sorted_ids)}
    
    nodes_list_out = [list(nodes_dict[i]) for i in sorted_ids]
    
    edges_list_out = []
    
    # Also save detailed curves for reconstruction
    curves = []
    
    default_radius = 0.5 * pitch # fallback
    
    for edge in edges:
        u_old, v_old = edge[0], edge[1]
        
        # Remap to new 0-indexed values
        if u_old not in id_map or v_old not in id_map:
            # Should not happen if graph is clean
            print(f"[Export Warning] Edge references missing nodes: {u_old}-{v_old}")
            continue
            
        u_new = id_map[u_old]
        v_new = id_map[v_old]
        
        radius = edge[4] if len(edge) >= 5 else default_radius
            
        edges_list_out.append([u_new, v_new, radius])
        
        # Save Curve Geometry (Points + Radius)
        pts = edge[3] if len(edge) >= 4 else []
        if len(pts) == 0:
            # Simple line
            p_start = nodes_dict[u_old]
            p_end = nodes_dict[v_old]
            curve_pts = [list(p_start) + [radius], list(p_end) + [radius]]
        else:
            # Polyline: Start -> pts -> End
            p_start = nodes_dict[u_old]
            p_end = nodes_dict[v_old]
            
            full_pts = [p_start] + pts + [p_end]
            curve_pts = [list(p) + [radius] for p in full_pts]
            
        curves.append({"points": curve_pts})

    # Build node_tags output (remapped to new 0-indexed IDs)
    node_tags_out = {}
    if node_tags:
        for old_id, tag_val in node_tags.items():
            if old_id in id_map:
                node_tags_out[str(id_map[old_id])] = tag_val
        print(f"[Export] Node Tags: {len(node_tags_out)} tagged nodes (1=Fixed, 2=Loaded)")
    
    data = {
        "metadata": {
            "method": "Baseline Yin", 
            "pitch": pitch,
            "units": "mm",
            "target_volume": target_volume,
            "design_bounds": design_bounds
        },
        "graph": {
            "nodes": nodes_list_out,
            "edges": edges_list_out,
            "node_tags": node_tags_out
        },
        "curves": curves,
        "history": history,
        "plates": [] 
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Export] Saved {len(curves)} segments to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Baseline Yin 3D Skeletonization Pipeline")
    parser.add_argument("input_mesh", help="Input STL")
    parser.add_argument("output_json", help="Output JSON")
    parser.add_argument("--pitch", type=float, default=1.0, help="Voxel Size")
    
    parser.add_argument("--max_iters", type=int, default=50, help="Thinning iterations")
    
    parser.add_argument("--collapse_thresh", type=float, default=2.0, help="Collapse edges < X")
    parser.add_argument("--prune_len", type=float, default=5.0, help="Prune branches < X")
    parser.add_argument("--rdp_epsilon", type=float, default=0.0, help="RDP Geometric Simplification Epsilon (0 = off)")
    
    parser.add_argument("--radius_mode", type=str, default="edt", choices=["edt", "uniform"], 
                        help="Radius Strategy: 'edt' (Geometric) or 'uniform' (Volume Matching)")
    
    parser.add_argument("--visualize", action="store_true", help="Show debug windows")
    parser.add_argument("--vol_thresh", type=float, default=0.3, help="Volume Threshold for NPZ (default: 0.3)")
    
    args = parser.parse_args()
    
    # 1. Load Data (Mesh or NPZ)
    bc_tags = None  # Will be set if NPZ has bc_tags
    
    if args.input_mesh.endswith(".npz"):
        print(f"[1] Loading Top3D Result from {args.input_mesh}...")
        npz_data = np.load(args.input_mesh)
        if 'rho' not in npz_data:
            raise ValueError("NPZ file must contain 'rho' array!")
        rho = npz_data['rho']
        
        # Load BC Tags if present
        if 'bc_tags' in npz_data:
            bc_tags = npz_data['bc_tags'].astype(np.int32)
            n_fixed = int(np.sum(bc_tags == 1))
            n_loaded = int(np.sum(bc_tags == 2))
            print(f"    BC Tags loaded: {n_fixed} fixed elements, {n_loaded} loaded elements")
        else:
            print("    [Warning] No bc_tags in NPZ. Tags will not propagate.")
        
        # Threshold to Binary Solid
        solid = (rho > args.vol_thresh).astype(bool)
        
        # Note: We no longer force-include tagged voxels in the solid.
        # Proximity-based tagging in extract_graph handles BC assignment.
        
        dims = solid.shape
        origin = np.array([0.0, 0.0, 0.0])
        mesh_bounds = np.array([
            [0.0, 0.0, 0.0],
            [dims[0]*args.pitch, dims[1]*args.pitch, dims[2]*args.pitch]
        ])
    else:
        if voxelize_mesh is None:
            raise ImportError("voxelize_mesh not available. Install trimesh or use .npz input.")
        print(f"[1] Voxelizing {args.input_mesh} (Pitch={args.pitch})...")
        solid, dims, origin, mesh_bounds = voxelize_mesh(args.input_mesh, args.pitch)
    
    # HISTORY TRACKING
    history_snapshots = []
    
    def capture_snapshot(name, nodes_d, edges_l):
        sorted_ids = sorted(nodes_d.keys())
        id_map = {old: new for new, old in enumerate(sorted_ids)}
        
        n_list = [list(nodes_d[i]) for i in sorted_ids]
        e_list = []
        for e in edges_l:
            u, v = e[0], e[1]
            if u in id_map and v in id_map:
                rad = e[4] if len(e) >= 5 else 1.0
                pts = e[3] if len(e) >= 4 else []
                e_list.append([id_map[u], id_map[v], rad, pts])
                
        history_snapshots.append({
            "type": "graph",
            "step": name,
            "nodes": n_list,
            "edges": e_list
        })

    def capture_voxel_snapshot(name, mask):
        indices = np.argwhere(mask)
        pts = origin + (indices * args.pitch) + (args.pitch * 0.5)
        if len(pts) > 50000:
            print(f"Warning: Snapshot {name} has {len(pts)} voxels. Subsampling...")
            pts = pts[::2] 
            
        history_snapshots.append({
            "type": "voxels",
            "step": name,
            "points": pts.tolist()
        })

    # Calculate Target Volume (Input Volume)
    voxel_vol = np.sum(solid) * (args.pitch**3)
    print(f"[Volume] Initial Solid Volume: {voxel_vol:.2f} mm^3")
    
    capture_voxel_snapshot("1_Initial_Voxels", solid) ### SNAPSHOT 1 (Voxels)
    
    if args.visualize:
        pcd = viz_voxels(solid, args.pitch, origin, [0.8, 0.8, 0.8])
        show_step("1. Initial Voxel Volume", [pcd])
        
    # 2. Thinning (Algorithm 3.1)
    # Yin's Method: Pass bc_tags to protect them from deletion!
    print(f"[2] Thinning (Max Iters={args.max_iters})...")
    
    if args.visualize:
        skeleton, iter_map = thin_grid_yin(solid.copy(), tags=bc_tags, max_iters=args.max_iters, record_iterations=True)
        pcd_iter = viz_iterative_thinning(iter_map, args.pitch, origin)
        if pcd_iter: show_step("2a. Iterative Removal", [pcd_iter])
        pcd_class = viz_skeleton_classification(skeleton, args.pitch, origin)
        if pcd_class: show_step("2b. Skeleton Classification", [pcd_class])
    else:
        skeleton = thin_grid_yin(solid.copy(), tags=bc_tags, max_iters=args.max_iters)  # tags=None here!
    
    # We use bc_tags later in extract_graph for Proximity Check
    # Build a full tag grid to pass
    graph_tags = None
    if bc_tags is not None:
        graph_tags = bc_tags  # Pass the full 3D grid
        
    capture_voxel_snapshot("2_Skeleton_Voxels", skeleton) ### SNAPSHOT 2 (Voxels)
    
    # 3. Graph Extraction (Algorithm 4.1) — tagged voxels become graph nodes
    print(f"[3] Extracting Graph...")
    # Pass graph_tags for proximity-based BC assignment
    nodes_arr, edges_list_raw, v_types, node_tags = extract_graph(skeleton, args.pitch, origin, tags=graph_tags)
    
    print(f"    Raw Graph: {len(nodes_arr)} nodes, {len(edges_list_raw)} edges")
    nodes_dict = {i: nodes_arr[i] for i in range(len(nodes_arr))}
    
    capture_snapshot("3_Raw_Graph", nodes_dict, edges_list_raw) # SNAPSHOT 3 (Graph)
    
    if args.visualize:
        geoms = viz_graph(nodes_arr, edges_list_raw)
        pcd_faint = viz_voxels(skeleton, args.pitch, origin, [0.9, 0.8, 0.8], 0.3)
        geoms.append(pcd_faint)
        show_step("3. Raw Graph + Skeleton Overlay", geoms)
        
    # 4. Post-Processing
    
    # A. Collapse Short Edges (Alg 4.4)
    if args.collapse_thresh > 0:
        print(f"[4A] Collapsing Edges < {args.collapse_thresh}...")
        nodes_dict, edges_list_raw = collapse_short_edges(nodes_dict, edges_list_raw, args.collapse_thresh, node_tags=node_tags)
        print(f"     Nodes: {len(nodes_dict)}, Edges: {len(edges_list_raw)}")
        
        capture_snapshot("4A_Collapsed", nodes_dict, edges_list_raw) # SNAPSHOT 4 (Graph)
        
        if args.visualize:
            n_arr, e_arr = graph_to_arrays(nodes_dict, edges_list_raw)
            show_step("4A. Edge Collapse", viz_graph(n_arr, e_arr))
            
    # B. Prune Branches (Alg 4.5)
    if args.prune_len > 0:
        print(f"[4B] Pruning Branches < {args.prune_len}...")
        nodes_dict, edges_list_raw = prune_branches(nodes_dict, edges_list_raw, args.prune_len, node_tags=node_tags)
        print(f"     Nodes: {len(nodes_dict)}, Edges: {len(edges_list_raw)}")
        
        capture_snapshot("4B_Pruned", nodes_dict, edges_list_raw) # SNAPSHOT 3
        
    # C. Geometric Simplification (RDP)
    if args.rdp_epsilon > 0:
        print(f"[4C] Simplifying Polylines (RDP epsilon={args.rdp_epsilon})...")
        nodes_dict, edges_list_raw = simplify_graph_geometry(nodes_dict, edges_list_raw, args.rdp_epsilon)
        print(f"     Nodes: {len(nodes_dict)}, Edges: {len(edges_list_raw)}")
        
        capture_snapshot("4C_Simplified_RDP", nodes_dict, edges_list_raw) # SNAPSHOT 4

    # D. Radius Estimation
    if args.radius_mode == 'edt':
        print("[4D] Estimating Parametric Radii (Euclidean Distance Transform)...")
        edt = distance_transform_edt(solid, sampling=[args.pitch]*3)
        nodes_dict, edges_list_raw = compute_edge_radii(nodes_dict, edges_list_raw, edt, args.pitch, origin)
    
    elif args.radius_mode == 'uniform':
        print("[4D] Estimating Parametric Radii (Volume Matching)...")
        total_vox_vol = np.sum(solid) * (args.pitch**3)
        nodes_dict, edges_list_raw = compute_uniform_radii(nodes_dict, edges_list_raw, total_vox_vol, args.pitch)

    # E. Fix Extrema (Ensure Nodes at Tips)
    from src.pipelines.baseline_yin.postprocessing import ensure_nodes_at_bounding_extrema
    nodes_dict, edges_list_raw = ensure_nodes_at_bounding_extrema(nodes_dict, edges_list_raw)
    
    capture_snapshot("5_Extrema_Fixed", nodes_dict, edges_list_raw)
    
    if args.visualize:
        n_arr, e_arr = graph_to_arrays(nodes_dict, edges_list_raw)
        radii_viz = [e[4] if len(e)>=5 else args.pitch for e in edges_list_raw]
        geoms = viz_graph_radii(n_arr, e_arr, np.array(radii_viz))
        show_step("5. Final Clean Graph (Color=Radius, Extrema Fixed)", geoms)
        
    # 5. Export (with node tags)
    export_to_json(nodes_dict, edges_list_raw, args.output_json, args.pitch, 
                   history=history_snapshots, target_volume=voxel_vol,
                   design_bounds=mesh_bounds.tolist(), node_tags=node_tags)
    print("Done.")

if __name__ == "__main__":
    main()
