import numpy as np
import json
import argparse
import sys
import os

# Adjust path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.pipelines.baseline_yin.visualization import viz_graph_radii, viz_loads, show_step
from src.optimization.fem import solve_frame, compute_frame_gradients as compute_sensitivities

def optimality_criteria_update(radii, sensitivities, lengths, vol_frac, target_vol, r_min=0.1, r_max=5.0, move=0.2, eta=0.5):
    """
    Standard Optimality Criteria (OC) update for Size Optimization.
    Min C s.t. Vol <= V_target
    
    Update rule:
    r_new = r_old * ( -dC/dr / (lambda * dV/dr) ) ^ eta
    
    where dV/dr = 2 * pi * r * L
    """
    l1, l2 = 0, 1e9
    
    # Pre-calculate dV/dr
    # V = pi * r^2 * L
    # dV/dr = 2 * pi * r * L
    dV_dr = 2 * np.pi * radii * lengths
    
    # Avoid div by zero
    dV_dr = np.maximum(dV_dr, 1e-6)
    
    # B_e = (-dC/dr) / (dV/dr)
    # Note: sensitivities are dC/dr (negative value for compliance reduction)
    # Actually, we usually work with -dC/dr (positive sensitivity) for the ratio
    # Our function returns dC/dr which is negative.
    # So numerator is -sensitivities.
    numerator = -sensitivities
    # Ensure positive for OC (if gradient is positive, it means thickening INCREASES compliance, which shouldn't happen for stiffness)
    numerator = np.maximum(numerator, 1e-10)
    
    B = numerator / dV_dr
    
    r_new = np.zeros_like(radii)
    
    # Bisection for Lambda
    while (l2 - l1) > (l1 + l2) * 1e-4:
        l_mid = 0.5 * (l2 + l1)
        
        # Proposed update based on Lagrange implementation
        term = B / l_mid
        factor = term ** eta
        
        # Damping (Move Limit)
        # We clamp the factor change to avoid oscillations
        # r * (1-move) <= r_new <= r * (1+move)
        
        # Simple implementation of OC update with move limits
        r_trial = radii * factor 
        
        # Clamp to move limit
        r_trial = np.clip(r_trial, radii * (1 - move), radii * (1 + move))
        
        # Clamp to min/max radius
        r_trial = np.clip(r_trial, r_min, r_max)
        
        # Calculate Volume
        vol_trial = np.sum(np.pi * r_trial**2 * lengths)
        
        if vol_trial > target_vol:
            l1 = l_mid # Lambda too small (Volume too big) -> Increase cost
        else:
            l2 = l_mid
            
        r_new = r_trial
        
    return r_new

def optimize_size(nodes, edges, initial_radii, problem, E=1000.0, vol_fraction=1.0, max_iter=50, visualize=False, target_volume_abs=None):
    """
    Main Size Optimization Loop.
    """
    radii = initial_radii.copy()
    
    # Pre-calc lengths
    lengths = []
    for u, v in edges:
        dist = np.linalg.norm(nodes[u] - nodes[v])
        lengths.append(dist)
    lengths = np.array(lengths)
    
    # Target Volume
    current_vol = np.sum(np.pi * radii**2 * lengths)
    vol_init = current_vol
    
    if target_volume_abs is not None:
         target_vol = target_volume_abs
         print(f"[Opt] Using Absolute Target Volume from Metadata/Args")
    else:
         target_vol = current_vol * vol_fraction
    
    print(f"[Opt] Initial Volume: {current_vol:.2f} -> Target: {target_vol:.2f}")
    
    compliance_hist = []
    radii_init = initial_radii.copy()
    
    # Setup Loads/BCs from Problem Config
    loads, bcs = problem.apply(nodes)
            
    print(f"[Opt] Config '{problem.name}': {len(bcs)} fixed nodes, {len(loads)} loaded nodes.")
    
    if len(loads) == 0:
        print("[Error] No loads defined! Optimization cannot proceed.")
        return radii, 0.0, 0.0  # Return tuple format: (radii, c_init, c_final)
        
    if visualize:
        # Viz 1: Loads & Initial State
        load_geoms = viz_loads(nodes, loads, bcs, scale=10.0)
        graph_geoms = viz_graph_radii(nodes, edges, radii)
        show_step("Initial Setup (Yellow=Load, Cyan=Fixed)", load_geoms + graph_geoms)
    
    # Optimization Loop
    for it in range(max_iter):
        # 1. FEM
        u, compliance, _ = solve_frame(nodes, edges, radii, E=E, loads=loads, bcs=bcs)
        compliance_hist.append(compliance)
        
        # 2. Gradient
        gradients = compute_sensitivities(nodes, edges.astype(int), radii, u, E=E)
        
        # 3. Update (OC)
        radii_new = optimality_criteria_update(radii, gradients, lengths, vol_fraction, target_vol)
        
        # 4. Check Change
        change = np.linalg.norm(radii_new - radii) / np.linalg.norm(radii)
        vol_meas = np.sum(np.pi * radii_new**2 * lengths)
        
        print(f"   Iter {it}: Compliance={compliance:.4e}, Vol={vol_meas:.2f}, Change={change:.4f}")
        
        radii = radii_new
        
        if change < 1e-3:
            print("[Opt] Converged.")
            break
            
    if not compliance_hist: # Handle edge case
        compliance_hist = [0.0] * 2
        vol_meas = vol_init
            
            
    if visualize:
         # Viz 2: Final State
         final_geoms = viz_graph_radii(nodes, edges, radii)
         show_step("Optimized Frame (Red=Thick, Blue=Thin)", final_geoms)
         
    # --- Final Report ---
    generate_report(compliance_hist[0], compliance_hist[-1], vol_init, vol_meas, radii_init, radii, it, "Converged" if change < 1e-3 else "Max Iters Reached")
    # --------------------

    # Return radii + initial and final compliance for iteration tracking
    return radii, compliance_hist[0], compliance_hist[-1]

def generate_report(c_init, c_final, v_init, v_final, r_init, r_final, iterations, message, filename="size_opt_report.txt"):
    """
    Prints and saves a summary of the Size Optimization.
    """
    # 1. Compliance Stats
    reduction = c_init - c_final
    pct = (reduction / c_init) * 100.0 if c_init > 0 else 0.0
    
    # 2. Volume Stats
    v_change = v_final - v_init
    v_pct = (v_change / v_init) * 100.0 if v_init > 0 else 0.0
    
    # 3. Radius Stats
    r_diff = r_final - r_init
    max_d = np.max(np.abs(r_diff))
    mean_d = np.mean(np.abs(r_diff))
    
    # 4. Format Report
    lines = [
        "="*50,
        "          SIZE OPTIMIZATION REPORT (OC Method)",
        "="*50,
        f"Status:             {message}",
        f"Iterations:         {iterations}",
        "-"*50,
        "COMPLIANCE (Stiffness Inverse):",
        f"  Initial:          {c_init:.4e}",
        f"  Final:            {c_final:.4e}",
        f"  Reduction:        {reduction:.4e} ({pct:.2f}%)",
        "-"*50,
        "VOLUME CHECK (Constraint):",
        f"  Initial Volume:   {v_init:.4f} mm^3",
        f"  Final Volume:     {v_final:.4f} mm^3",
        f"  Change:           {v_change:.4f} mm^3 ({v_pct:.2f}%)",
        "-"*50,
        "RADIUS STATISTICS (Design Variables):",
        f"  Max Change:       {max_d:.4f} mm",
        f"  Mean Change:      {mean_d:.4f} mm",
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

def load_problem_config(problem_name):
    """
    Dynamically loads a problem class from src/problems/
    Example: 'rocker_arm' -> src.problems.rocker_arm.RockerArmSetup
    """
    try:
        if problem_name == 'rocker_arm':
            from src.problems.rocker_arm import RockerArmSetup
            return RockerArmSetup()
        elif problem_name in ['cantilever', 'top3d_result_YIN_Canteliver_Beam', 'Cantilever_Beam_3D']:
            from src.problems.cantilever import CantileverSetup
            return CantileverSetup()
        elif problem_name == 'generic':
            from src.problems.generic import GenericProblem
            return GenericProblem()
        elif problem_name == 'tagged':
            from src.problems.tagged_problem import TaggedProblem
            return TaggedProblem()
        else:
            raise ValueError(f"Unknown problem: {problem_name}")
    except ImportError as e:
        print(f"[Error] Could not import problem '{problem_name}': {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run Size Optimization on reconstructed JSON.")
    parser.add_argument("input_json", help="Path to baseline_output.json")
    parser.add_argument("output_json", help="Path to save optimized.json")
    parser.add_argument("--problem", type=str, required=True, help="Problem Config Name (e.g. 'rocker_arm')")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--visualize", action="store_true", help="Show 3D Viz")
    args = parser.parse_args()
    
    # Init Problem
    problem = load_problem_config(args.problem)
    if not problem:
        return
    
    # Load JSON
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    # If tagged problem, load node_tags from JSON
    if hasattr(problem, 'load_tags_from_json'):
        problem.load_tags_from_json(args.input_json)
        
    if 'graph' not in data:
        print("[Error] Input JSON does not contain 'graph' block.")
        print("Please rerun baseline_reconstruction.py to generate updated JSON.")
        return
        
    nodes_data = data['graph']['nodes']
    edges_data = data['graph']['edges']
    
    print(f"[Opt] Loaded {len(nodes_data)} nodes, {len(edges_data)} edges.")
    
    nodes = np.array(nodes_data) # (N,3)
    nodes = np.array(nodes_data) # (N,3)
    
    # Manually parse edges to handle mixed types (lists inside rows)
    edges_list = []
    radii_list = []
    
    for e in edges_data:
        # e is [u, v, len, pts, r] or [u, v, r] or [u, v]
        u, v = int(e[0]), int(e[1])
        edges_list.append([u, v])
        
        # Radius extraction
        r = 1.0
        if len(e) >= 5: r = float(e[4])
        elif len(e) >= 3: r = float(e[2])
        radii_list.append(r)
        
    edges = np.array(edges_list)
    radii_init = np.array(radii_list)
        
    # Check for Target Volume in Metadata
    target_vol_meta = None
    if 'metadata' in data and 'target_volume' in data['metadata']:
        target_vol_meta = data['metadata']['target_volume']
        print(f"[Setup] Found Target Volume in Metadata: {target_vol_meta:.2f}")
        
    # Run Optimization
    # P_load is now handled by problem.apply()
    optimized_radii = optimize_size(nodes, edges, radii_init, problem=problem, visualize=args.visualize, max_iter=args.iters, target_volume_abs=target_vol_meta)
    
    # Save Output
    # We update the original data with new radii
    # 1. Update Graph Edges
    for i, r in enumerate(optimized_radii):
        edges_data[i][2] = r
        
    # 2. Update Curves (Visuals)
    curves = data['curves']
    # Assumption: curves are in same order as edges?
    # In baseline_reconstruction, yes they are generated in same loop.
    if len(curves) == len(optimized_radii):
         for i, curve in enumerate(curves):
             pts = curve['points']
             r = optimized_radii[i]
             # Update radius in all points [x,y,z, r]
             for p in pts:
                 p[3] = r
    
    # 3. Update node_tags (Just to be sure they persist)
    if hasattr(problem, '_node_tags'):
        data['graph']['node_tags'] = {str(k): v for k, v in problem._node_tags.items()}
    
    with open(args.output_json, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"[Opt] Saved optimized geometry to {args.output_json}")

if __name__ == "__main__":
    main()
