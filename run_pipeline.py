#!/usr/bin/env python3
"""
Baseline Yin + Top3D Unified Pipeline (Main_V2)
================================================

Full pipeline with Yin-style BC tag propagation:
  Stage 0: Python Top3D → .npz (with bc_tags)
  Stage 1: Reconstruction → JSON (skeleton, graph, node_tags, plates)
  Stage 2: Size Optimization → JSON (optimized radii)
  Stage 3: Layout Optimization → JSON (optimized positions)

Modes:
  [Default]  Beam-only: threshold → thin → graph → optimize
  --hybrid   Beam+Plate: zone classify → separate thin → plates + beams → optimize beams
  --curved   Bézier curves: fit cubic Béziers to skeleton edges (geometry only)

Usage:
  # Beam-only
  python run_pipeline.py --nelx 50 --nely 10 --nelz 2 --volfrac 0.3 --output beam.json

  # Hybrid beam+plate
  python run_pipeline.py --skip_top3d --top3d_npz roof.npz --hybrid --output hybrid.json

  # Curved beams (geometry only)
  python run_pipeline.py --skip_top3d --top3d_npz test.npz --curved --opt_loops 2 --output curved.json
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from scipy.spatial import KDTree
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import optimization modules
from src.optimization.layout_opt import optimize_layout
from src.optimization.size_opt import optimize_size
from src.optimization.fem import solve_frame
from src.problems.tagged_problem import TaggedProblem

# Optional: curves for Bézier re-fitting after optimization
try:
    from src.curves.spline import fit_cubic_bezier, sample_curve_points
    _CURVES_AVAILABLE = True
except ImportError:
    _CURVES_AVAILABLE = False


def _compute_compliance(json_data, problem_config, E=1000.0):
    """One-shot FEA solve to get compliance from a JSON graph."""
    nodes = np.array(json_data['graph']['nodes'])
    edges_raw = json_data['graph']['edges']
    edges = np.array([[e[0], e[1]] for e in edges_raw], dtype=int)
    radii = np.array([e[4] if len(e) >= 5 else e[2] for e in edges_raw])

    loads, bcs = problem_config.apply(nodes)
    _, compliance, _ = solve_frame(nodes, edges, radii, E=E, loads=loads, bcs=bcs)
    return compliance


def _frame_volume(nodes, edges, radii):
    """Compute total frame volume: Σ π r² L."""
    if len(edges) == 0:
        return 0.0
    lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    return float(np.sum(np.pi * radii**2 * lengths))


def _geometric_likeness(nodes_ref, nodes_current, domain_diagonal):
    """
    Symmetric Chamfer distance between two node sets, normalized to a
    similarity score in [0, 1] (1 = identical positions).
    """
    if len(nodes_current) == 0 or len(nodes_ref) == 0:
        return 1.0, 0.0

    tree_ref = KDTree(nodes_ref)
    tree_cur = KDTree(nodes_current)
    d_fwd = tree_ref.query(nodes_current)[0]
    d_bwd = tree_cur.query(nodes_ref)[0]
    mean_chamfer = float((d_fwd.mean() + d_bwd.mean()) / 2.0)
    score = float(np.exp(-2.0 * mean_chamfer / domain_diagonal))
    return score, mean_chamfer


def _print_comparison_table(metrics_list, baseline_volume, target_volume):
    """Print formatted comparison table of iterative metrics."""
    print("\n" + "="*130)
    print(" " * 30 + "ITERATIVE LAYOUT + SIZE OPTIMISATION — COMPARISON SUMMARY")
    print("="*130)
    print(f" {'Iter':<5} | {'Stage':<8} | {'Compliance':<14} | {'Δ vs Prev':<10} | {'Volume (mm³)':<13} | {'Vol Err%':<9} | {'Geo. Score':<10} | {'Chamfer (mm)':<12} | {'Nodes':<6} | {'Edges':<6}")
    print("-"*130)

    print(f" {'—':<5} | {'Baseline':<8} | {baseline_volume:>13.2f} | {'—':<10} | {target_volume:>12.2f} | {0.00:>8.2f}% | {1.000:<10.3f} | {0.00:<12.2f} | {'—':<6} | {'—':<6}")

    prev_compliance = baseline_volume
    for row in metrics_list:
        compliance = row['c_layout'] if row['stage'] == 'Layout' else row['c_size']
        delta_pct = ((compliance - prev_compliance) / prev_compliance * 100.0) if prev_compliance > 0 else 0.0
        volume = row['v_layout'] if row['stage'] == 'Layout' else row['v_size']
        vol_err = ((volume - target_volume) / target_volume * 100.0) if target_volume > 0 else 0.0

        print(f" {row['iter']:<5} | {row['stage']:<8} | {compliance:>13.2f} | {delta_pct:>9.2f}% | {volume:>12.2f} | {vol_err:>8.2f}% | {row['geo_score']:<10.3f} | {row['mean_chamfer']:<12.2f} | {row['n_nodes']:<6} | {row['n_edges']:<6}")
        prev_compliance = compliance

    print("="*130)

    if len(metrics_list) > 0:
        last_stage = metrics_list[-1]['stage']
        final_compliance = metrics_list[-1]['c_layout'] if last_stage == 'Layout' else metrics_list[-1]['c_size']
        total_reduction = ((baseline_volume - final_compliance) / baseline_volume * 100.0) if baseline_volume > 0 else 0.0
        print(f"Total compliance reduction: {total_reduction:>6.2f}%  ({baseline_volume:.2f} → {final_compliance:.2f})")

        last_size_idx = len(metrics_list) - 1
        while last_size_idx >= 0 and metrics_list[last_size_idx]['stage'] != 'Size':
            last_size_idx -= 1

        if last_size_idx >= 0:
            final_volume = metrics_list[last_size_idx]['v_size']
            vol_err = abs((final_volume - target_volume) / target_volume * 100.0)
            vol_satisfied = "YES" if vol_err < 0.5 else "PARTIAL"
            print(f"Volume constraint satisfied: {vol_satisfied}  (error = {vol_err:.2f}%)")
        else:
            print(f"Volume constraint satisfied: N/A")
    print("="*130 + "\n")


def _refit_curves(nodes, edges, radii, curved):
    """Build curves list from optimised nodes+edges, with optional Bézier fitting."""
    curves = []
    for (u, v), r in zip(edges, radii):
        p_start = list(nodes[u])
        p_end = list(nodes[v])
        if curved and _CURVES_AVAILABLE:
            ctrl_pts = fit_cubic_bezier(np.array(p_start), np.array(p_end), [])
            pts = sample_curve_points(np.array(p_start), ctrl_pts[0], ctrl_pts[1], np.array(p_end), float(r), N=20)
            curves.append({"ctrl_pts": ctrl_pts.tolist(), "points": pts, "radius": float(r)})
        else:
            curves.append({"points": [p_start + [float(r)], p_end + [float(r)]]})
    return curves


def run_stage(cmd, desc):
    """Run a subprocess command with logging."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FATAL] {desc} failed (exit code {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Yin + Top3D Unified Pipeline (Main_V2)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # === Top3D (Design Domain) ===
    g_top3d = parser.add_argument_group("Top3D Design Domain")
    g_top3d.add_argument("--nelx", type=int, default=60, help="Elements in X (domain length)")
    g_top3d.add_argument("--nely", type=int, default=20, help="Elements in Y (domain height)")
    g_top3d.add_argument("--nelz", type=int, default=4,  help="Elements in Z (domain depth)")
    g_top3d.add_argument("--volfrac", type=float, default=0.3, help="Volume fraction")
    g_top3d.add_argument("--penal", type=float, default=3.0, help="Penalization factor")
    g_top3d.add_argument("--rmin", type=float, default=1.5, help="Filter radius (R)")
    g_top3d.add_argument("--max_loop", type=int, default=50, help="Max Top3D iterations")

    # === Load Definition ===
    g_load = parser.add_argument_group("Load Definition")
    g_load.add_argument("--load_x", type=int, default=None, help="Load node X index (default=nelx)")
    g_load.add_argument("--load_y", type=int, default=None, help="Load node Y index (default=nely)")
    g_load.add_argument("--load_z", type=int, default=None, help="Load node Z index (default=nelz/2)")
    g_load.add_argument("--load_fx", type=float, default=0.0, help="Load force X component")
    g_load.add_argument("--load_fy", type=float, default=-1.0, help="Load force Y component")
    g_load.add_argument("--load_fz", type=float, default=0.0, help="Load force Z component")

    # === Skeletonisation ===
    g_skel = parser.add_argument_group("Skeletonisation")
    g_skel.add_argument("--pitch", type=float, default=1.0, help="Voxel size (mm)")
    g_skel.add_argument("--max_iters", type=int, default=50, help="Max thinning iterations")
    g_skel.add_argument("--prune_len", type=float, default=2.0, help="Prune branches < X mm")
    g_skel.add_argument("--collapse_thresh", type=float, default=2.0, help="Collapse edges < X mm")
    g_skel.add_argument("--rdp", type=float, default=1.0, help="RDP simplification epsilon (0=off)")
    g_skel.add_argument("--radius_mode", type=str, default="uniform", choices=["edt", "uniform"],
                        help="Radius strategy: 'edt' or 'uniform' (Volume Matching)")
    g_skel.add_argument("--vol_thresh", type=float, default=0.3, help="Density threshold for NPZ")

    # === Hybrid (Beam+Plate) Mode ===
    g_hybrid = parser.add_argument_group("Hybrid Beam+Plate Mode")
    g_hybrid.add_argument("--hybrid", action="store_true", help="Enable beam+plate hybrid pipeline")
    g_hybrid.add_argument("--detect_plates", type=str, default="auto", choices=["auto", "off", "force"],
                          help="Plate detection: auto, off, force")
    g_hybrid.add_argument("--plate_mode", type=str, default="bspline", choices=["bspline", "voxel", "mesh"],
                          help="Plate reconstruction mode")
    g_hybrid.add_argument("--plate_thickness_ratio", type=float, default=0.15,
                          help="Max plate half-thickness as fraction of domain diagonal")
    g_hybrid.add_argument("--min_plate_size", type=int, default=4,
                          help="Min voxels for plate classification")
    g_hybrid.add_argument("--flatness_ratio", type=float, default=3.0,
                          help="PCA eigenvalue ratio for flatness detection")
    g_hybrid.add_argument("--junction_thresh", type=int, default=4,
                          help="Neighbor count threshold for junction detection")
    g_hybrid.add_argument("--min_avg_neighbors", type=float, default=3.0,
                          help="Min average neighbor count for plate classification")

    # === Curved Beams ===
    g_curved = parser.add_argument_group("Curved Beams (Geometry Only)")
    g_curved.add_argument("--curved", action="store_true",
                          help="Fit cubic Bézier curves to beam skeleton edges (geometry/visualisation only)")

    # === Layout & Size Optimisation ===
    g_opt = parser.add_argument_group("Layout & Size Optimisation")
    g_opt.add_argument("--limit", type=float, default=5.0, help="Layout move limit (mm)")
    g_opt.add_argument("--snap", type=float, default=5.0, help="Snap distance for node merging (mm)")
    g_opt.add_argument("--iters", type=int, default=50, help="Size opt iterations")
    g_opt.add_argument("--opt_loops", type=int, default=1, help="Number of Layout+Size optimisation loops")
    g_opt.add_argument("--problem", type=str, default="tagged",
                       help="Problem config: 'tagged' (auto from BC tags), 'cantilever', 'rocker_arm'")

    # === Output & Visualisation ===
    g_out = parser.add_argument_group("Output & Visualisation")
    g_out.add_argument("--output", type=str, default="full_control_beam.json", help="Final output JSON filename")
    g_out.add_argument("--output_dir", type=str, default="output/hybrid_v2", help="Output directory")
    g_out.add_argument("--visualize", action="store_true", help="Show 3D debug windows")

    # === Skip Stages ===
    g_skip = parser.add_argument_group("Advanced")
    g_skip.add_argument("--skip_top3d", action="store_true", help="Skip Stage 0 (use existing .npz)")
    g_skip.add_argument("--top3d_npz", type=str, default=None, help="Path to existing .npz")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.output))[0]

    mode_str = "HYBRID BEAM+PLATE" if args.hybrid else "BEAM-ONLY"
    if args.curved:
        mode_str += " + CURVED"

    print("=" * 60)
    print("   BASELINE YIN + TOP3D UNIFIED PIPELINE (Main_V2)")
    print(f"   Mode: {mode_str}")
    print("=" * 60)
    print(f"  Domain:    {args.nelx} x {args.nely} x {args.nelz}")
    print(f"  VolFrac:   {args.volfrac}")
    print(f"  Load:      [{args.load_fx}, {args.load_fy}, {args.load_fz}]")
    print(f"  Problem:   {args.problem}")
    print(f"  Output:    {args.output_dir}/{args.output}")
    print("=" * 60)

    # ========================================
    # STAGE 0: Python Top3D
    # ========================================
    npz_path = os.path.join(args.output_dir, f"{base_name}_top3d.npz")

    if args.skip_top3d:
        if args.top3d_npz:
            npz_path = args.top3d_npz
        if not os.path.exists(npz_path):
            print(f"[FATAL] --skip_top3d specified but NPZ not found: {npz_path}")
            return 1
        print(f"\n[Stage 0] SKIPPED. Using existing: {npz_path}")
    else:
        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "run_top3d.py"),
            "--nelx", str(args.nelx), "--nely", str(args.nely), "--nelz", str(args.nelz),
            "--volfrac", str(args.volfrac), "--penal", str(args.penal),
            "--rmin", str(args.rmin), "--max_loop", str(args.max_loop),
            "--load_fx", str(args.load_fx), "--load_fy", str(args.load_fy),
            "--load_fz", str(args.load_fz),
            "--output", npz_path,
        ]
        if args.load_x is not None: cmd += ["--load_x", str(args.load_x)]
        if args.load_y is not None: cmd += ["--load_y", str(args.load_y)]
        if args.load_z is not None: cmd += ["--load_z", str(args.load_z)]

        if not run_stage(cmd, "STAGE 0: Python Top3D Topology Optimisation"):
            return 1

    # ========================================
    # STAGE 1: Baseline Yin Reconstruction
    # ========================================
    stage1_out = os.path.join(args.output_dir, f"{base_name}_1_reconstructed.json")

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "src/pipelines/baseline_yin/reconstruct.py"),
        npz_path, stage1_out,
        "--pitch", str(args.pitch), "--max_iters", str(args.max_iters),
        "--collapse_thresh", str(args.collapse_thresh),
        "--prune_len", str(args.prune_len),
        "--rdp_epsilon", str(args.rdp),
        "--radius_mode", args.radius_mode,
        "--vol_thresh", str(args.vol_thresh),
    ]
    # Hybrid-specific args
    if args.hybrid:
        cmd.append("--hybrid")
        cmd += [
            "--plate_thickness_ratio", str(args.plate_thickness_ratio),
            "--min_plate_size", str(args.min_plate_size),
            "--flatness_ratio", str(args.flatness_ratio),
            "--junction_thresh", str(args.junction_thresh),
            "--min_avg_neighbors", str(args.min_avg_neighbors),
            "--plate_mode", args.plate_mode,
            "--detect_plates", args.detect_plates,
        ]
    if args.curved:
        cmd.append("--curved")
    if args.visualize:
        cmd.append("--visualize")

    if not run_stage(cmd, "STAGE 1: Skeleton Reconstruction"):
        return 1

    # ========================================
    # Check Stage 1 output for beam edges
    # ========================================
    with open(stage1_out, 'r') as f:
        baseline_data = json.load(f)

    n_beam_edges = len(baseline_data.get('graph', {}).get('edges', []))
    has_plates = len(baseline_data.get('plates', [])) > 0
    plate_only = (n_beam_edges == 0 and has_plates)

    if plate_only:
        print(f"\n[Pipeline] Plate-only structure (0 beam edges). Skipping optimization stages.")
        final_path = os.path.join(args.output_dir, args.output)
        shutil.copy2(stage1_out, final_path)

        # Still create history
        try:
            history = {
                "metadata": baseline_data.get("metadata", {}),
                "history": baseline_data.get("history", []),
                "stages": [{"name": "1. Reconstructed", "curves": baseline_data.get("curves", []), "plates": baseline_data.get("plates", [])}],
                "curves": baseline_data.get("curves", []),
                "plates": baseline_data.get("plates", []),
                "joints": baseline_data.get("joints", []),
                "graph": baseline_data.get("graph", {})
            }
            hist_path = os.path.join(args.output_dir, f"{base_name}_history.json")
            with open(hist_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"[Export] Pipeline history: {hist_path}")
        except Exception as e:
            print(f"[Warning] Could not create history: {e}")

        print(f"\n{'='*60}")
        print("              PIPELINE COMPLETE (Plate-Only)")
        print(f"{'='*60}")
        print(f"  Final Output:  {final_path}")
        print(f"{'='*60}")
        return 0

    # ========================================
    # STAGES 2 & 3: Iterative Layout + Size Optimisation
    # ========================================
    print(f"\n[Opt Loops] Running {args.opt_loops} iteration(s) of Size + Layout Optimisation")

    baseline_nodes = np.array(baseline_data['graph']['nodes'])
    target_volume = baseline_data['metadata'].get('target_volume', None)
    design_bounds = baseline_data['metadata'].get('design_bounds', None)
    domain_diagonal = np.linalg.norm(
        np.array(design_bounds[1]) - np.array(design_bounds[0])
    ) if design_bounds else 100.0

    # Preserve plates from Stage 1 (plates are stable, only beams are optimized)
    plates_data = baseline_data.get('plates', [])
    joints_data = baseline_data.get('joints', [])

    # Compute baseline compliance
    problem_config = TaggedProblem()
    problem_config.load_tags_from_json(stage1_out)
    c_baseline = _compute_compliance(baseline_data, problem_config, E=1000.0)

    metrics_list = []
    current_json = stage1_out

    # Accumulate every stage in execution order for FreeCAD history export
    all_stages = [{"name": "1. Reconstructed", "curves": baseline_data.get("curves", []), "plates": plates_data}]
    final_curves = baseline_data.get("curves", [])

    for loop_idx in range(args.opt_loops):
        loop_num = loop_idx + 1
        suffix = f"_loop{loop_num}"

        print(f"\n{'='*60}")
        print(f"  ITERATION {loop_num}: Size + Layout Optimisation")
        print(f"{'='*60}")

        # --- STAGE 2: Size Optimisation (FIRST) ---
        print(f"\n[Iter {loop_num}] STAGE 2: Size Optimisation")
        sized_json = os.path.join(args.output_dir, f"{base_name}_3_sized{suffix}.json")

        try:
            with open(current_json, 'r') as f:
                size_input = json.load(f)

            nodes_size = np.array(size_input['graph']['nodes'])
            edges_raw = size_input['graph']['edges']
            edges_size = np.array([[e[0], e[1]] for e in edges_raw], dtype=int)
            radii_size = np.array([e[4] if len(e) >= 5 else e[2] for e in edges_raw])

            size_problem = TaggedProblem()
            size_problem.load_tags_from_json(current_json)

            radii_sized, c_size_init, c_size_final = optimize_size(
                nodes_size, edges_size, radii_size, size_problem,
                E=1000.0, vol_fraction=1.0, max_iter=args.iters,
                visualize=args.visualize, target_volume_abs=target_volume
            )

            curves_sized = _refit_curves(nodes_size, edges_size, radii_sized, args.curved)

            sized_data = {
                "metadata": size_input.get("metadata", {}),
                "graph": {
                    "nodes": nodes_size.tolist(),
                    "edges": [[int(u), int(v), float(r)] for u, v, r in zip(edges_size[:, 0], edges_size[:, 1], radii_sized)],
                    "node_tags": size_input['graph'].get('node_tags', {})
                },
                "curves": curves_sized,
                "history": size_input.get("history", []),
                "plates": plates_data,
                "joints": joints_data
            }
            with open(sized_json, 'w') as f:
                json.dump(sized_data, f, indent=2)

            all_stages.append({"name": f"Size Loop {loop_num}", "curves": curves_sized, "plates": plates_data})

            v_sized = _frame_volume(nodes_size, edges_size, radii_sized)
            geo_score_s, mean_chamfer_s = _geometric_likeness(baseline_nodes, nodes_size, domain_diagonal)

            metrics_list.append({
                'iter': loop_num, 'stage': 'Size',
                'c_layout': 0.0, 'c_size': c_size_final,
                'v_layout': v_sized, 'v_size': v_sized,
                'geo_score': geo_score_s, 'mean_chamfer': mean_chamfer_s,
                'n_nodes': len(nodes_size), 'n_edges': len(edges_size),
            })

        except Exception as e:
            print(f"[ERROR] Size Optimisation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # --- STAGE 3: Layout Optimisation (SECOND) ---
        print(f"\n[Iter {loop_num}] STAGE 3: Layout Optimisation")
        layout_json = os.path.join(args.output_dir, f"{base_name}_2_layout{suffix}.json")

        try:
            with open(sized_json, 'r') as f:
                layout_input = json.load(f)

            nodes_layout = np.array(layout_input['graph']['nodes'])
            edges_raw = layout_input['graph']['edges']
            edges_layout = np.array([[e[0], e[1]] for e in edges_raw], dtype=int)
            radii_layout = np.array([e[2] for e in edges_raw])
            node_tags = {int(k): v for k, v in layout_input['graph'].get('node_tags', {}).items()}

            layout_problem = TaggedProblem()
            layout_problem.load_tags_from_json(sized_json)

            nodes_opt, edges_opt, radii_opt, tags_opt, c_layout_init, c_layout_final = optimize_layout(
                nodes_layout, edges_layout, radii_layout, layout_problem,
                E=1000.0, move_limit=args.limit, visualize=args.visualize,
                target_volume_abs=target_volume, snap_dist=args.snap,
                design_bounds=design_bounds, node_tags=node_tags
            )

            curves_layout = _refit_curves(nodes_opt, edges_opt, radii_opt, args.curved)

            layout_data = {
                "metadata": layout_input.get("metadata", {}),
                "graph": {
                    "nodes": nodes_opt.tolist(),
                    "edges": [[int(u), int(v), 1.0, [], float(r)] for u, v, r in zip(edges_opt[:, 0], edges_opt[:, 1], radii_opt)],
                    "node_tags": {str(k): v for k, v in tags_opt.items()}
                },
                "curves": curves_layout,
                "history": layout_input.get("history", []),
                "plates": plates_data,
                "joints": joints_data
            }
            with open(layout_json, 'w') as f:
                json.dump(layout_data, f, indent=2)

            all_stages.append({"name": f"Layout Loop {loop_num}", "curves": curves_layout, "plates": plates_data})
            final_curves = curves_layout

            v_layout = _frame_volume(nodes_opt, edges_opt, radii_opt)
            geo_score_l, mean_chamfer_l = _geometric_likeness(baseline_nodes, nodes_opt, domain_diagonal)

            metrics_list.append({
                'iter': loop_num, 'stage': 'Layout',
                'c_layout': c_layout_final, 'c_size': 0.0,
                'v_layout': v_layout, 'v_size': v_layout,
                'geo_score': geo_score_l, 'mean_chamfer': mean_chamfer_l,
                'n_nodes': len(nodes_opt), 'n_edges': len(edges_opt),
            })

            current_json = layout_json

        except Exception as e:
            print(f"[ERROR] Layout Optimisation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    stage2_out = os.path.join(args.output_dir, f"{base_name}_2_layout_loop{args.opt_loops}.json")
    stage3_out = os.path.join(args.output_dir, f"{base_name}_3_sized_loop{args.opt_loops}.json")

    # ========================================
    # Print Comparison Table
    # ========================================
    if args.opt_loops > 1 or len(metrics_list) > 2:
        _print_comparison_table(metrics_list, c_baseline, target_volume)

    # ========================================
    # FINAL: Copy final output to requested name
    # ========================================
    final_path = os.path.join(args.output_dir, args.output)
    shutil.copy2(stage2_out, final_path)

    # ========================================
    # Pipeline History (for FreeCAD)
    # ========================================
    try:
        history = {
            "metadata": baseline_data.get("metadata", {}),
            "history": baseline_data.get("history", []),
            "stages": all_stages,
            "curves": final_curves,
            "plates": plates_data,
            "joints": joints_data,
            "graph": baseline_data.get("graph", {})
        }

        hist_path = os.path.join(args.output_dir, f"{base_name}_history.json")
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)
        stage_names = [s["name"] for s in all_stages]
        print(f"\n[Export] Pipeline history ({len(all_stages)} stages): {stage_names}")
        print(f"         Saved to: {hist_path}")
    except Exception as e:
        print(f"[Warning] Could not create history: {e}")

    # === Summary ===
    print(f"\n{'='*60}")
    print("              PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Mode:          {mode_str}")
    print(f"  Top3D:         {npz_path}")
    print(f"  Reconstructed: {stage1_out}")
    if args.opt_loops == 1:
        print(f"  Size Opt:      {stage3_out}")
        print(f"  Layout Opt:    {stage2_out}")
    else:
        print(f"  Size Opt:      {stage3_out} (loop {args.opt_loops})")
        print(f"  Layout Opt:    {stage2_out} (loop {args.opt_loops})")
    print(f"  Final Output:  {final_path}")
    if args.opt_loops > 1:
        print(f"  Opt Loops:     {args.opt_loops} iterations completed")
    if has_plates:
        print(f"  Plates:        {len(plates_data)} plate regions")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
