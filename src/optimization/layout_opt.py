import numpy as np
import argparse
import json
import sys
import os
from scipy.optimize import minimize

# Adjust path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.optimization.fem import solve_frame
from src.problems import load_problem_config
from src.pipelines.baseline_yin.visualization import viz_graph_radii, show_step, viz_loads
# Removed collapse_short_edges import as we implementing local logic
# from src.pipelines.baseline_yin.postprocessing import collapse_short_edges, recheck_graph, graph_to_arrays

def generate_report(c_init, c_final, nodes_init, nodes_final, edges, radii, iterations, message, filename="optimization_report.txt", target_volume_abs=None):
    """
    Prints and saves a summary of the Layout Optimization.
    """
    # 1. Compliance Stats
    reduction = c_init - c_final
    pct = (reduction / c_init) * 100.0 if c_init > 0 else 0.0
    
    # 2. Movement Stats
    displacements = np.linalg.norm(nodes_final - nodes_init, axis=1)
    max_d = np.max(displacements)
    idx_max = np.argmax(displacements)
    mean_d = np.mean(displacements)
    
    # 3. Volume Stats (Check if conserved)
    def calc_volume(ns):
        vol = 0.0
        for i, (u, v) in enumerate(edges):
            length = np.linalg.norm(ns[u] - ns[v])
            area = np.pi * (radii[i]**2)
            vol += area * length
        return vol
        
    v_init = calc_volume(nodes_init)
    v_final = calc_volume(nodes_final)
    v_change = v_final - v_init
    v_pct = (v_change / v_init) * 100.0 if v_init > 0 else 0.0
    
    v_target_row = ""
    if target_volume_abs:
        v_diff = v_final - target_volume_abs
        v_diff_pct = (v_diff / target_volume_abs) * 100.0
        v_target_row = f"\n  Target (Solid):   {target_volume_abs:.4f} mm^3 (Diff: {v_diff_pct:+.2f}%)"
    
    # 4. Format Report
    lines = [
        "="*50,
        "          LAYOUT OPTIMIZATION REPORT",
        "="*50,
        f"Status:             {message}",
        f"Iterations:         {iterations}",
        "-"*50,
        "COMPLIANCE (Stiffness Inverse):",
        f"  Initial:          {c_init:.4f}",
        f"  Final:            {c_final:.4f}",
        f"  Reduction:        {reduction:.4f} ({pct:.2f}%)",
        "-"*50,
        "VOLUME CHECK:",
        f"  Initial Volume:   {v_init:.4f} mm^3",
        f"  Final Volume:     {v_final:.4f} mm^3",
        f"  Change:           {v_change:.4f} mm^3 ({v_pct:.2f}%)" + v_target_row,
        "-"*50,
        "NODE MOVEMENT (Geometry Change):",
        f"  Max Displacement: {max_d:.4f} mm (Node {idx_max})",
        f"  Mean Displacement:{mean_d:.4f} mm",
        "="*50
    ]
    
    report_text = "\n".join(lines)
    
    # Print to Console
    print("\n" + report_text + "\n")
    
    # Save to File
    try:
        with open(filename, "w") as f:
            f.write(report_text)
        print(f"[Report] Saved detailed report to {filename}")
    except Exception as e:
        print(f"[Report] Failed to save file: {e}")



def obj_compliance(x_flat, nodes_shape, edges, radii, problem, E):
    """
    Objective function for Layout Optimization.
    x_flat: Flattened node coordinates.
    """
    # 1. Reconstruct Nodes
    nodes = x_flat.reshape(nodes_shape)
    
    # 2. Setup Loads/BCs (Re-evaluate on new geometry? 
    # Usually BCs are geometric. If nodes move, BCs might change?
    # For now, we assume standard geometric mapping holds, or we fix the INDICES.
    # If we move a fixed node, it's bad. 
    # We will constrain fixed nodes, so their position won't change much.
    # But strictly, the Problem class finds nodes by boxes.
    # Let's assume Problem definition relies on initial geometry or we pass 'bcs' directly?
    # Better: Recalculate Loads/BCs to be safe, BUT optimization solvers dislike discontinuous changes.
    # If a node moves out of a "Fixed Box", it suddenly becomes free -> Discontinuity.
    # STRATEGY: Calculate BCs ONCE on start, and lock those NODE INDICES. 
    # Loads: If geometric (e.g. surface load), might need update. 
    # But for rocker arm, loads are on specific tips.
    # Let's simple-pass: We calculate BCs/Loads once outside and pass them in?
    # No, 'solve_frame' needs them.
    # Let's cache them.
    pass

    # Actually, we'll use a cached wrapper.
    return 0.0

class LayoutOptimizer:
    def __init__(self, nodes_init, edges, radii, problem, E=1000.0):
        self.nodes_init = nodes_init.copy()
        self.nodes_shape = nodes_init.shape
        self.edges = edges
        self.radii = radii
        self.problem = problem
        self.E = E
        
        # Pre-calc BCs/Loads to lock topology
        # We don't want BCs jumping nodes during optimization
        self.loads_init, self.bcs_init = problem.apply(self.nodes_init)
        
        # Identify Fixed Nodes & Loaded Nodes (for constraints)
        self.fixed_node_indices = set(self.bcs_init.keys())
        self.loaded_node_indices = set(self.loads_init.keys())
        self.locked_node_indices = self.fixed_node_indices.union(self.loaded_node_indices)
        
    def objective(self, x_flat):
        nodes = x_flat.reshape(self.nodes_shape)
        
        # Use initial BC mapping (Topology is fixed, Geometry changes)
        # We assume the Load/BC is attached to the NODE ID, not the coordinate.
        
        u, compliance, _ = solve_frame(nodes, self.edges, self.radii, self.E, 
                                      loads=self.loads_init, bcs=self.bcs_init)
        return compliance

def optimize_layout(nodes, edges, radii, problem, E=1000.0, move_limit=5.0, visualize=False, 
                    target_volume_abs=None, snap_dist=2.0, design_bounds=None, node_tags=None):
    """
    Optimizes Node Positions (x,y,z).
    
    Args:
        design_bounds: [[min_x, min_y, min_z], [max_x, max_y, max_z]] - design domain limits
    """
    optimizer = LayoutOptimizer(nodes, edges, radii, problem, E)
    x0 = nodes.flatten()
    
    # Pre-compute max radius per node (for geometry envelope constraint)
    node_max_radii = np.zeros(len(nodes))
    for i, (u, v) in enumerate(edges):
        node_max_radii[u] = max(node_max_radii[u], radii[i])
        node_max_radii[v] = max(node_max_radii[v], radii[i])
    
    # Bounds
    # Fixed nodes: Exact position (or very tight bounds)
    # Free nodes: +/- move_limit, AND within design_bounds (accounting for radius)
    bounds = []
    
    # Pre-compute locked set
    locked_nodes = optimizer.locked_node_indices
    
    # Convert design bounds to numpy for easier use
    if design_bounds is not None:
        domain_min = np.array(design_bounds[0])
        domain_max = np.array(design_bounds[1])
        print(f"[Layout] Design Domain: [{domain_min[0]:.1f}, {domain_max[0]:.1f}] x [{domain_min[1]:.1f}, {domain_max[1]:.1f}] x [{domain_min[2]:.1f}, {domain_max[2]:.1f}]")
    else:
        domain_min = None
        domain_max = None
    
    for i in range(len(nodes)):
        if i in locked_nodes:
            # Fixed or Loaded Node - constrain to initial position
            c = nodes[i]
            bounds.append((c[0], c[0]))
            bounds.append((c[1], c[1]))
            bounds.append((c[2], c[2]))
        else:
            # Free Node
            c = nodes[i]
            r = node_max_radii[i]  # Max radius of connected edges
            
            # Start with move_limit bounds
            b_min = [c[0] - move_limit, c[1] - move_limit, c[2] - move_limit]
            b_max = [c[0] + move_limit, c[1] + move_limit, c[2] + move_limit]
            
            # Apply design domain bounds (accounting for beam radius)
            if domain_min is not None:
                for axis in range(3):
                    # Node center must be at least r away from domain boundary
                    # so the cylinder doesn't exceed the design envelope
                    domain_width = domain_max[axis] - domain_min[axis]
                    
                    if domain_width < 2 * r:
                        # Domain too thin for this beam - clamp to center
                        center = (domain_min[axis] + domain_max[axis]) / 2.0
                        b_min[axis] = max(b_min[axis], center)
                        b_max[axis] = min(b_max[axis], center)
                    else:
                        b_min[axis] = max(b_min[axis], domain_min[axis] + r)
                        b_max[axis] = min(b_max[axis], domain_max[axis] - r)
                    
                    # Final safety: ensure min <= max
                    if b_min[axis] > b_max[axis]:
                        mid = (b_min[axis] + b_max[axis]) / 2.0
                        b_min[axis] = mid
                        b_max[axis] = mid
            
            bounds.append((b_min[0], b_max[0]))
            bounds.append((b_min[1], b_max[1]))
            bounds.append((b_min[2], b_max[2]))
            
    print(f"[Layout] Starting L-BFGS-B Optimization on {len(x0)} variables...")
    print(f"[Layout] Move Limit: +/- {move_limit} mm")
    
    if visualize:
         # Show Initial
         # Viz 1: Loads & Initial State
         load_geoms = viz_loads(nodes, optimizer.loads_init, optimizer.bcs_init, scale=10.0)
         graph_geoms = viz_graph_radii(nodes, edges, radii)
         show_step("Initial Layout", load_geoms + graph_geoms)
         
    # Callback
    def callback(xk):
        c = optimizer.objective(xk)
        print(f"   Iter: Compliance = {c:.4f}")
        
    # Initial Compliance
    c_init = optimizer.objective(x0)
    print(f"[Layout] Initial Compliance: {c_init:.4f}")
    
    if np.isnan(c_init):
        print("[ERROR] Beam Graph is singular or disconnected (NaN Compliance). Skipping Layout Optimization.")
        return nodes, edges, radii, node_tags, c_init, c_init
        
    # Optimization
    res = minimize(optimizer.objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'disp': True, 'maxiter': 50, 'eps': 1e-4}, # epsilon step for FD
                   callback=None) 
                   
    print(f"[Layout] Done. Final Compliance: {res.fun:.4f}")
    
    nodes_new = res.x.reshape(nodes.shape)
    
    # --- Final Report ---
    generate_report(c_init, res.fun, nodes, nodes_new, edges, radii, res.nit, res.message, target_volume_abs=target_volume_abs)
    # --------------------
    
    if visualize:
        final_geoms = viz_graph_radii(nodes_new, edges, radii)
        
        # Comparison (Red=Old, Green=New)?
        # Or just show result
        show_step("Optimized Layout", final_geoms)
        
    # --- Final Report ---
    generate_report(c_init, res.fun, nodes, nodes_new, edges, radii, res.nit, res.message, target_volume_abs=target_volume_abs)
    # --------------------
    
    # NEW: Snap Nodes (Collapse Short Edges)
    print(f"[Layout] Snapping Clean-up (Limit={snap_dist}mm)...")
    
    # We pass the locked_nodes set to ensure they act as anchors.
    # The clean-up step will prioritize snapping free nodes TO locked nodes.
    nodes_clean, edges_clean, radii_clean, tags_clean = snap_nodes(nodes_new, edges, radii, snap_dist, locked_nodes=optimizer.locked_node_indices, node_tags=node_tags)
    
    if len(nodes_clean) < len(nodes_new):
        print(f"[Layout] Snapped! Nodes reduced from {len(nodes_new)} to {len(nodes_clean)}")
        return nodes_clean, edges_clean, radii_clean, tags_clean, c_init, res.fun
    else:
        return nodes_new, edges, radii, node_tags if node_tags else {}, c_init, res.fun

def snap_nodes(nodes, edges, radii, tol, locked_nodes=None, node_tags=None):
    """
    Merges nodes closer than tol connected by an edge.
    Updates edges, radii, AND node_tags.
    Prioritizes keeping the position of locked_nodes (anchors).
    """
    if locked_nodes is None:
        locked_nodes = set()
    else:
        locked_nodes = set(locked_nodes)
        
    if node_tags is None:
        node_tags = {}
        
    # PROTECT EXTREMA: Add nodes at min/max of each axis to locked set
    # ... (rest of extrema logic stays same)
    mins = np.min(nodes, axis=0)
    maxs = np.max(nodes, axis=0)
    dims = maxs - mins
    long_axis = np.argmax(dims)
    
    tol_extrema = 3.0
    min_val_long = mins[long_axis]
    tip_indices = [i for i, p in enumerate(nodes) if p[long_axis] <= min_val_long + tol_extrema]
    for idx in tip_indices: locked_nodes.add(idx)
    
    max_val_long = maxs[long_axis]
    base_indices = [i for i, p in enumerate(nodes) if p[long_axis] >= max_val_long - tol_extrema]
    for idx in base_indices: locked_nodes.add(idx)
    
    # 1. Identify short edges to collapse (using UFD)
    N = len(nodes)
    parent = np.arange(N)
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    for i, (u, v) in enumerate(edges):
        p1, p2 = nodes[u], nodes[v]
        if np.linalg.norm(p1 - p2) < tol:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                # Merge if tags are compatible
                tag_u, tag_v = node_tags.get(u), node_tags.get(v)
                if tag_u and tag_v and tag_u != tag_v:
                    continue # Block merge of different BCs
                parent[root_v] = root_u

    # 2. Compute Centroids for merged groups
    group_nodes = {}
    for i in range(N):
        root = find(i)
        if root not in group_nodes: group_nodes[root] = []
        group_nodes[root].append(i)
        
    new_coords = {}
    new_node_tags = {}
    
    unique_roots = sorted(group_nodes.keys())
    for new_id, root in enumerate(unique_roots):
        members = group_nodes[root]
        locked_members = [m for m in members if m in locked_nodes]
        
        # Coordinate
        if len(locked_members) > 0:
            new_coords[new_id] = np.mean([nodes[m] for m in locked_members], axis=0)
        else:
            new_coords[new_id] = np.mean([nodes[m] for m in members], axis=0)
            
        # Tag (collect tags from any member)
        for m in members:
            if m in node_tags:
                new_node_tags[new_id] = node_tags[m]
                break
                
    # 3. Build Outputs
    nodes_out = np.array([new_coords[i] for i in range(len(unique_roots))])
    
    root_to_new_id = {root: i for i, root in enumerate(unique_roots)}
    edges_out = []
    radii_out = []
    for i, (u, v) in enumerate(edges):
        ru, rv = find(u), find(v)
        if ru != rv:
            edges_out.append([root_to_new_id[ru], root_to_new_id[rv]])
            radii_out.append(radii[i])
            
    return nodes_out, np.array(edges_out), np.array(radii_out), new_node_tags

def main():
    parser = argparse.ArgumentParser(description="Run Layout Optimization.")
    parser.add_argument("input_json", help="Path to input.json (with radii)")
    parser.add_argument("output_json", help="Path to save optimized.json")
    parser.add_argument("--problem", type=str, required=True, help="Problem Config")
    parser.add_argument("--limit", type=float, default=5.0, help="Move limit (mm)")
    parser.add_argument("--snap", type=float, default=2.0, help="Snap distance for merging close nodes (mm), set to 0 to disable")
    parser.add_argument("--visualise", action="store_true", help="Visualise Result")
    args = parser.parse_args()
    
    # Init Problem
    problem = load_problem_config(args.problem)
    if not problem: return

    # Load JSON
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    # If tagged problem, load node_tags from JSON
    if hasattr(problem, 'load_tags_from_json'):
        problem.load_tags_from_json(args.input_json)
        
    nodes_data = data['graph']['nodes']
    edges_data = data['graph']['edges']
    
    # Manually parse edges to handle mixed types (lists inside rows)
    nodes = np.array(nodes_data)
    edges_list = []
    radii_list = []
    
    for e in edges_data:
        # e is [u, v, len, pts, r] or [u, v, r] or [u, v]
        u, v = int(e[0]), int(e[1])
        edges_list.append([u, v])
        
        # Radius extraction - Be Flexible
        r = 1.0
        if len(e) >= 5: r = float(e[4]) # Index 4 is Radius in Clean Graph
        elif len(e) >= 3: r = float(e[2]) # Index 2 might be Radius in some formats
        radii_list.append(r)
        
    edges = np.array(edges_list)
    radii = np.array(radii_list)
    
    # If tagged problem, load node_tags from JSON
    if hasattr(problem, 'load_tags_from_json'):
        problem.load_tags_from_json(args.input_json)
    
    node_tags_input = problem._node_tags if hasattr(problem, '_node_tags') else {}
        
    nodes_data = data['graph']['nodes']
    edges_data = data['graph']['edges']
    
    # Manually parse edges to handle mixed types (lists inside rows)
    nodes = np.array(nodes_data)
    edges_list = []
    radii_list = []
    
    for e in edges_data:
        # e is [u, v, len, pts, r] or [u, v, r] or [u, v]
        u, v = int(e[0]), int(e[1])
        edges_list.append([u, v])
        if len(e) >= 5:
            radii_list.append(float(e[4]))
        elif len(e) >= 3:
            radii_list.append(float(e[2]))
        else:
            radii_list.append(1.0)
            
    edges = np.array(edges_list)
    radii = np.array(radii_list)
    
    # Retrieve Target Volume if exists
    target_vol_meta = None
    if 'metadata' in data and 'target_volume' in data['metadata']:
        target_vol_meta = data['metadata']['target_volume']
        print(f"[Layout] Found Target Volume in Metadata: {target_vol_meta:.2f}")

    # Retrieve Design Bounds if exists
    design_bounds = None
    if 'metadata' in data and 'design_bounds' in data['metadata']:
        design_bounds = data['metadata']['design_bounds']
        print(f"[Layout] Found Design Bounds in Metadata")

    # Run Layout Opt
    res = optimize_layout(nodes, edges, radii, problem, move_limit=args.limit, 
                          visualize=args.visualise, target_volume_abs=target_vol_meta, 
                          snap_dist=args.snap, design_bounds=design_bounds,
                          node_tags=node_tags_input)
    
    if isinstance(res, tuple) and len(res) == 4:
        nodes_new, edges_new, radii_new, node_tags_new = res
    elif isinstance(res, tuple) and len(res) == 3:
        nodes_new, edges_new, radii_new = res
        node_tags_new = node_tags_input
    else:
        nodes_new = res
        edges_new = edges
        radii_new = radii
        node_tags_new = node_tags_input
    
    # Save
    edges_out_list = []
    for i, (u, v) in enumerate(edges_new):
        r = radii_new[i]
        edges_out_list.append([int(u), int(v), 1.0, [], float(r)])

    data['graph']['nodes'] = nodes_new.tolist()
    data['graph']['edges'] = edges_out_list
    data['graph']['node_tags'] = {str(k): v for k, v in node_tags_new.items()}
    # Remove old 'radii' key if exists to avoid confusion
    if 'radii' in data['graph']:
        del data['graph']['radii']
    
    # Update Curves (Points)
    # The curves correspond to edges. If we moved nodes, the curves (polylines) need to move.
    # However, our edge collapses/pruning might have made curves complex.
    # Linear interpolation for now: Just move endpoints? 
    # The intermediate points in 'curves' are explicit coordinates.
    # If we only optimize Nodes, the intermediates are "stuck".
    # This might look weird. 
    # Correct approach: Re-interpolate intermediates or Optimize Intermediates too? 
    # Optimizing intermediates explodes variables (16 edges * ~10 pts/edge).
    # Simple approach: Re-linearize edges (straight lines) for output to avoid broken polylines.
    # Or map intermediates proportional to node movement.
    # Let's just Linearize curves (remove intermediates) for the Optimized result. 
    # Optimization tends to straighten members anyway.
    
    new_curves = []
    # Use edges_new and radii_new, as nodes_new is re-indexed
    for i, e in enumerate(edges_new):
        u, v = int(e[0]), int(e[1])
        p1 = nodes_new[u]
        p2 = nodes_new[v]
        r = radii_new[i]
        
        # Create 2-point curve (Straight line)
        # Format: [ [x,y,z,r], [x,y,z,r] ]
        pts = [
            list(p1) + [r],
            list(p2) + [r]
        ]
        new_curves.append({'points': pts})
        
    data['curves'] = new_curves
    
    with open(args.output_json, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Layout] Saved to {args.output_json}")
    
    if args.visualise:
        print("[Viz] Showing Final Snapped Layout...")
        final_geoms_snapped = viz_graph_radii(nodes_new, edges_new, radii_new)
        show_step("Final Snapped Layout", final_geoms_snapped)

if __name__ == "__main__":
    main()
