"""Optimality-Criteria beam cross-section (radius) optimisation.

Iteratively resizes beam radii to minimise structural compliance subject to a
total volume constraint, using the classic OC update rule with bisection for
the Lagrange multiplier and ±20% move limits.

Main entry point
----------------
:func:`optimize_size`
"""
import numpy as np
import json
import argparse
import sys
import os

# Adjust path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.pipelines.baseline_yin.visualization import viz_graph_radii, viz_loads, show_step
from src.optimization.fem import (solve_frame,
                                  compute_frame_gradients as compute_sensitivities,
                                  compute_shell_thickness_gradients,
                                  solve_curved_frame,
                                  compute_curved_size_gradients)
from src.curves.spline import bezier_arc_length

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

def _compute_plate_areas(plates):
    """Compute total mid-surface area per plate for volume = area * thickness."""
    from src.optimization.fem import _get_plate_mid_surface
    areas = []
    for plate in plates:
        ms = _get_plate_mid_surface(plate)
        if ms is None:
            areas.append(0.0)
            continue
        verts = np.array(ms["vertices"])
        tris = ms["triangles"]
        total_area = 0.0
        for tri in tris:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            if max(i0, i1, i2) >= len(verts):
                continue
            e1 = verts[i1] - verts[i0]
            e2 = verts[i2] - verts[i0]
            total_area += 0.5 * np.linalg.norm(np.cross(e1, e2))
        areas.append(total_area)
    return np.array(areas)


def optimize_size(nodes, edges, initial_radii, problem, E=1000.0, vol_fraction=1.0,
                  max_iter=50, visualize=False, target_volume_abs=None,
                  ctrl_pts=None, r_min=0.1, r_max=5.0,
                  plates=None, h_min=0.1, h_max=10.0,
                  sym_data=None):
    """Optimise beam radii (and optionally plate thicknesses) to minimise compliance.

    When ``plates`` is provided, shell elements are assembled alongside beams
    and plate thicknesses are co-optimised using the OC update rule.

    Parameters
    ----------
    nodes : ndarray
        Node positions, shape (N, 3).
    edges : ndarray
        Edge connectivity, shape (M, 2).
    initial_radii : ndarray
        Starting beam radii, shape (M,).
    problem : object
        BC problem config (provides loads and fixed DOFs).
    E : float
        Young's modulus (default 1000.0).
    vol_fraction : float
        Target volume as fraction of initial volume.
    max_iter : int
        Maximum OC iterations (default 50).
    visualize : bool
        Open interactive visualisation windows.
    target_volume_abs : float or None
        Absolute target volume; overrides ``vol_fraction`` if given.
    ctrl_pts : list or None
        Bézier control points per edge (curved mode).
    r_min : float
        Minimum beam radius (mm).
    r_max : float
        Maximum beam radius (mm).
    plates : list of dict or None
        Plate data from reconstruction.
    h_min : float
        Minimum plate thickness (mm).
    h_max : float
        Maximum plate thickness (mm).
    sym_data : dict or None
        Symmetry data for mirror-half enforcement.

    Returns
    -------
    radii : ndarray
        Optimised beam radii.
    c_initial : float
        Compliance before optimisation.
    c_final : float
        Compliance after optimisation.
    history : list
        Per-iteration compliance values.
    plate_thicknesses : list of float or None
        Optimised plate thicknesses (None if no plates).
    """
    radii = initial_radii.copy()
    use_curved = ctrl_pts is not None
    has_plates = plates is not None and len(plates) > 0

    # Pre-calc beam lengths
    lengths = []
    for idx, (u, v) in enumerate(edges):
        if use_curved and ctrl_pts[idx] is not None:
            p0, p3 = nodes[int(u)], nodes[int(v)]
            p1c, p2c = ctrl_pts[idx][0], ctrl_pts[idx][1]
            lengths.append(bezier_arc_length(p0, p1c, p2c, p3))
        else:
            lengths.append(np.linalg.norm(nodes[int(u)] - nodes[int(v)]))
    lengths = np.array(lengths)

    # Plate setup
    plate_thicknesses = None
    plate_areas = None
    if has_plates:
        plate_thicknesses = []
        for p in plates:
            ms = p.get("mid_surface")
            h = p.get("thickness", 1.0)
            if ms is not None:
                h = ms.get("mean_thickness", h)
            plate_thicknesses.append(max(h, h_min))
        plate_thicknesses = np.array(plate_thicknesses)
        plate_areas = _compute_plate_areas(plates)
        plate_thicknesses_init = plate_thicknesses.copy()
        plate_vol_init = np.sum(plate_areas * plate_thicknesses)
        print(f"[Opt] Plates: {len(plates)}, total area={np.sum(plate_areas):.2f} mm², "
              f"initial plate volume={plate_vol_init:.2f} mm³")

    # Target Volume (beams + plates combined)
    beam_vol = np.sum(np.pi * radii**2 * lengths)
    plate_vol = np.sum(plate_areas * plate_thicknesses) if has_plates else 0.0
    vol_init = beam_vol + plate_vol

    if target_volume_abs is not None:
        target_vol = target_volume_abs
        if has_plates:
            # Target includes plate volume now
            target_vol = target_volume_abs + plate_vol
            print(f"[Opt] Target volume adjusted: beam={target_volume_abs:.2f} + "
                  f"plate={plate_vol:.2f} = {target_vol:.2f} mm³")
        print(f"[Opt] Using Absolute Target Volume from Metadata/Args")
    else:
        target_vol = vol_init * vol_fraction

    print(f"[Opt] Initial Volume: {vol_init:.2f} (beam={beam_vol:.2f}, "
          f"plate={plate_vol:.2f}) -> Target: {target_vol:.2f}")

    compliance_hist = []
    radii_init = initial_radii.copy()

    # Setup Loads/BCs from Problem Config
    loads, bcs = problem.apply(nodes)

    print(f"[Opt] Config '{problem.name}': {len(bcs)} fixed nodes, {len(loads)} loaded nodes.")

    if len(loads) == 0:
        print("[Error] No loads defined! Optimization cannot proceed.")
        return radii, 0.0, 0.0, [], plate_thicknesses

    # ── Symmetry pre-computation ─────────────────────────────────
    sym_edge_pairs = None
    if sym_data is not None:
        from src.optimization.symmetry import (
            find_symmetric_node_pairs, find_symmetric_edge_pairs,
            average_symmetric_radii,
        )
        _sym_node_info = find_symmetric_node_pairs(
            nodes, sym_data['planes'], tol=sym_data['tol'])
        sym_edge_pairs = find_symmetric_edge_pairs(edges, _sym_node_info)
        _total_pairs = sum(len(v) for v in sym_edge_pairs.values())
        print(f"[Opt] Symmetry: {_total_pairs} edge pair(s) for radius averaging")

    if visualize:
        load_geoms = viz_loads(nodes, loads, bcs, scale=10.0)
        graph_geoms = viz_graph_radii(nodes, edges, radii, ctrl_pts=ctrl_pts)
        show_step("Initial Setup (Yellow=Load, Cyan=Fixed)", load_geoms + graph_geoms)

    # Optimization Loop
    for it in range(max_iter):
        # 1. FEM Solve (with plates if available)
        if use_curved:
            u, compliance, _ = solve_curved_frame(
                nodes, edges, radii, ctrl_pts, E=E, loads=loads, bcs=bcs)
            all_nodes = nodes
            shell_elements = []
        elif has_plates:
            result = solve_frame(
                nodes, edges, radii, E=E, loads=loads, bcs=bcs,
                plates=plates, plate_thicknesses=plate_thicknesses.tolist())
            u, compliance, beam_elements, shell_elements = result
            # Get expanded node array for gradient computation
            from src.optimization.fem import _build_plate_node_map
            _, all_nodes, _, _ = _build_plate_node_map(plates, len(nodes), nodes, bcs, loads)
        else:
            u, compliance, _ = solve_frame(
                nodes, edges, radii, E=E, loads=loads, bcs=bcs)
            all_nodes = nodes
            shell_elements = []

        compliance_hist.append(compliance)

        if np.isnan(compliance):
            print("[ERROR] System is singular or disconnected (NaN Compliance). "
                  "Skipping Size Optimization.")
            break

        # 2. Beam gradients
        if use_curved:
            gradients = compute_curved_size_gradients(
                nodes, edges.astype(int), radii, ctrl_pts, u, E=E)
        else:
            gradients = compute_sensitivities(
                all_nodes, edges.astype(int), radii, u, E=E)

        # 3. Plate thickness gradients
        if has_plates and len(shell_elements) > 0:
            h_grads = compute_shell_thickness_gradients(
                plates, plate_thicknesses, shell_elements,
                all_nodes, u, E=E, nu=0.3)
        else:
            h_grads = None

        # 4. Joint OC update (beams + plates share one Lagrange multiplier)
        if has_plates and h_grads is not None:
            radii_new, h_new = _joint_oc_update(
                radii, gradients, lengths,
                plate_thicknesses, h_grads, plate_areas,
                target_vol, r_min, r_max, h_min, h_max)
        else:
            radii_new = optimality_criteria_update(
                radii, gradients, lengths, vol_fraction, target_vol,
                r_min=r_min, r_max=r_max)
            h_new = plate_thicknesses

        # 4b. Enforce symmetric radii (exact averaging)
        if sym_edge_pairs is not None:
            radii_new = average_symmetric_radii(radii_new, sym_edge_pairs)

        # 5. Check convergence
        r_change = np.linalg.norm(radii_new - radii) / max(np.linalg.norm(radii), 1e-10)
        h_change = 0.0
        if has_plates and h_new is not None:
            h_change = np.linalg.norm(h_new - plate_thicknesses) / max(np.linalg.norm(plate_thicknesses), 1e-10)

        change = max(r_change, h_change)

        beam_vol_new = np.sum(np.pi * radii_new**2 * lengths)
        plate_vol_new = np.sum(plate_areas * h_new) if h_new is not None else 0.0
        vol_meas = beam_vol_new + plate_vol_new

        if has_plates:
            print(f"   Iter {it}: C={compliance:.4e}, Vol={vol_meas:.2f} "
                  f"(beam={beam_vol_new:.1f}, plate={plate_vol_new:.1f}), "
                  f"Δr={r_change:.4f}, Δh={h_change:.4f}")
        else:
            print(f"   Iter {it}: Compliance={compliance:.4e}, Vol={vol_meas:.2f}, Change={change:.4f}")

        radii = radii_new
        if h_new is not None:
            plate_thicknesses = h_new

        if change < 1e-3:
            print("[Opt] Converged.")
            break

    if not compliance_hist:
        compliance_hist = [0.0] * 2

    if 'vol_meas' not in locals():
        vol_meas = vol_init
    if 'change' not in locals():
        change = 0.0

    if visualize:
        final_geoms = viz_graph_radii(nodes, edges, radii, ctrl_pts=ctrl_pts)
        show_step("Optimized Frame (Red=Thick, Blue=Thin)", final_geoms)

    # --- Final Report ---
    generate_report(compliance_hist[0], compliance_hist[-1], vol_init, vol_meas,
                    radii_init, radii, it,
                    "Converged" if change < 1e-3 else "Max Iters Reached")

    return radii, compliance_hist[0], compliance_hist[-1], list(compliance_hist), plate_thicknesses


def _joint_oc_update(radii, r_grads, lengths,
                     thicknesses, h_grads, plate_areas,
                     target_vol, r_min, r_max, h_min, h_max,
                     move=0.2, eta=0.5):
    """OC update for beams + plates with a shared Lagrange multiplier.

    Volume constraint: Σ π r² L + Σ A_p h_p ≤ V_target
    """
    # dV/dr = 2πrL,  dV/dh = A_p
    dV_dr = np.maximum(2 * np.pi * radii * lengths, 1e-6)
    dV_dh = np.maximum(plate_areas, 1e-6)

    # Numerators (positive sensitivity ratios)
    B_r = np.maximum(-r_grads, 1e-10) / dV_dr
    B_h = np.maximum(-h_grads, 1e-10) / dV_dh

    l1, l2 = 0.0, 1e9

    r_new = radii.copy()
    h_new = thicknesses.copy()

    while (l2 - l1) > (l1 + l2) * 1e-4:
        l_mid = 0.5 * (l2 + l1)

        # Beam update
        r_trial = radii * (B_r / l_mid) ** eta
        r_trial = np.clip(r_trial, radii * (1 - move), radii * (1 + move))
        r_trial = np.clip(r_trial, r_min, r_max)

        # Plate update
        h_trial = thicknesses * (B_h / l_mid) ** eta
        h_trial = np.clip(h_trial, thicknesses * (1 - move), thicknesses * (1 + move))
        h_trial = np.clip(h_trial, h_min, h_max)

        vol_trial = np.sum(np.pi * r_trial**2 * lengths) + np.sum(plate_areas * h_trial)

        if vol_trial > target_vol:
            l1 = l_mid
        else:
            l2 = l_mid

        r_new = r_trial
        h_new = h_trial

    return r_new, h_new

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

from src.problems import load_problem_config

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
