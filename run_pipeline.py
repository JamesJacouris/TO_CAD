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
from src.pipelines.baseline_yin.reconstruct import reconstruct_npz
from src.reporting.convergence import (
    plot_top3d_convergence,
    plot_size_layout_convergence,
    plot_combined_convergence,
    generate_pipeline_report,
)

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
    radii = np.array([e[2] for e in edges_raw])

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


def _build_report_data(args, baseline_data, metrics_list, convergence_stages,
                       top3d_hist, target_volume, c_baseline,
                       nodes_opt=None, edges_opt=None, radii_opt=None,
                       baseline_nodes=None):
    """Assemble the report_data dict consumed by generate_pipeline_report()."""
    meta = baseline_data.get('metadata', {})

    # ── Top3D section ──────────────────────────────────────────────────────
    top3d_sec = {}
    if top3d_hist:
        top3d_sec = {
            'mesh_size':  (args.nelx, args.nely, args.nelz),
            'volfrac':    args.volfrac,
            'penal':      args.penal,
            'rmin':       args.rmin,
            'iterations': len(top3d_hist),
            'max_loop':   args.max_loop,
            'converged':  len(top3d_hist) < args.max_loop,
            'c_initial':  top3d_hist[0],
            'c_final':    top3d_hist[-1],
        }

    # ── Reconstruction section ─────────────────────────────────────────────
    recon_sec = {
        'solid_voxels':    meta.get('solid_voxels'),
        'skeleton_voxels': meta.get('skeleton_voxels'),
        'nodes':  len(baseline_data.get('graph', {}).get('nodes', [])),
        'edges':  len(baseline_data.get('graph', {}).get('edges', [])),
        'plates': len(baseline_data.get('plates', [])),
        'target_volume': target_volume,
        'zone_stats': meta.get('zone_stats', {}),
    }

    # ── Per-loop optimisation section ──────────────────────────────────────
    # Group convergence_stages by loop number
    loops_dict = {}
    for stage in convergence_stages:
        lp = stage['loop']
        loops_dict.setdefault(lp, {})
        stype = stage['type']
        hist  = stage['history']
        ci    = stage.get('c_init', hist[0] if hist else 0.0)
        cf    = stage.get('c_final', hist[-1] if hist else 0.0)

        entry = {
            'iterations': len(hist),
            'c_initial':  ci,
            'c_final':    cf,
        }

        if stype == 'size' and stage.get('radii') is not None:
            r = stage['radii']
            entry.update({
                'radius_min':  float(np.min(r)),
                'radius_max':  float(np.max(r)),
                'radius_mean': float(np.mean(r)),
                'radius_std':  float(np.std(r)),
            })

        if stype == 'layout':
            nb = stage.get('nodes_before')
            na = stage.get('nodes_after')
            if nb is not None and na is not None and len(nb) == len(na):
                disp = np.linalg.norm(np.array(na) - np.array(nb), axis=1)
                entry['max_node_disp']  = float(np.max(disp))
                entry['mean_node_disp'] = float(np.mean(disp))

        loops_dict[lp][stype] = entry

    opt_loops = [{'loop': lp, **stages} for lp, stages in sorted(loops_dict.items())]

    # ── Overall summary ────────────────────────────────────────────────────
    final_compliance = None
    if metrics_list:
        last = metrics_list[-1]
        final_compliance = last['c_layout'] if last['stage'] == 'Layout' else last['c_size']

    final_volume = None
    if radii_opt is not None and nodes_opt is not None and edges_opt is not None:
        final_volume = float(np.sum(
            np.pi * radii_opt**2 *
            np.linalg.norm(nodes_opt[edges_opt[:, 0]] - nodes_opt[edges_opt[:, 1]], axis=1)
        ))

    vol_err = None
    if final_volume is not None and target_volume and target_volume > 0:
        vol_err = abs((final_volume - target_volume) / target_volume * 100.0)

    geo_sim = None
    if metrics_list:
        geo_sim = metrics_list[-1].get('geo_score')

    overall_sec = {
        'baseline_compliance': c_baseline,
        'final_compliance':    final_compliance,
        'volume_target':       target_volume,
        'volume_final':        final_volume,
        'volume_error_pct':    vol_err,
        'geometric_similarity': geo_sim,
    }

    return {
        'problem_name':      args.problem,
        'top3d':             top3d_sec,
        'reconstruction':    recon_sec,
        'optimization_loops': opt_loops,
        'overall':           overall_sec,
    }


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
    g_load.add_argument("--load_fx", type=float, default=None, help="Load force X component")
    g_load.add_argument("--load_fy", type=float, default=None, help="Load force Y component")
    g_load.add_argument("--load_fz", type=float, default=None, help="Load force Z component")
    g_load.add_argument("--load_dist", type=str, default="point", choices=["point", "surface_top", "surface_bottom"],
                        help="Force distribution mode")

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
    g_opt = parser.add_argument_group("Optimization Configuration")
    g_opt.add_argument("--optimize", action="store_true", help="Enable Beam Layout & Size Optimisation (default: False for hybrid mode)")
    g_opt.add_argument("--iters", type=int, default=50, help="Max iterations for internal Top3D or Layout optimization")
    g_opt.add_argument("--opt_loops", type=int, default=2, help="Number of Size + Layout iteration loops")
    g_opt.add_argument("--limit", type=float, default=5.0, help="Move limit for layout optimization (mm)")
    g_opt.add_argument("--prune_opt_thresh", type=float, default=0.0, help="Percentage (0.0-1.0) of max radius to prune dead-weight edges post-optimization")
    g_opt.add_argument("--snap", type=float, default=5.0, help="Snap distance for node merging (mm)")
    g_opt.add_argument("--problem", type=str, default="tagged",
                       help="Problem config: 'tagged' (auto from BC tags), 'cantilever', 'roof_slab', 'bridge', 'deck', 'rocker_arm', 'quadcopter'")
    # Quadcopter-specific (passed through to run_top3d.py)
    g_opt.add_argument("--motor_arm_frac", type=float, default=0.1,
        help="[Quadcopter] Motor mount position as fraction of nelx/nely inset from each corner (default: 0.1)")
    g_opt.add_argument("--load_patch_frac", type=float, default=0.1,
        help="[Quadcopter] Half-width of centre payload patch as fraction of nelx/nely (default: 0.1)")
    g_opt.add_argument("--motor_radius", type=int, default=0,
        help="[Quadcopter] Radius (elements) of circular passive void at each motor mount. 0 = disabled.")
    g_opt.add_argument("--motor_bolt_spacing", type=int, default=0,
        help="[Quadcopter] Split each motor into 2 bolt columns separated perpendicularly to the arm axis "
             "(elements). Creates bending moment → parallel arm branches. 0 = single column (default).")
    g_opt.add_argument("--arm_load_n", type=int, default=0,
        help="[Quadcopter] Distributed load columns per arm between hub and motor. Creates arm bending → X-bracing. 0 = hub only (default).")
    g_opt.add_argument("--arm_load_frac", type=float, default=0.3,
        help="[Quadcopter] Fraction of total load applied to arm load columns. Default: 0.3.")
    g_opt.add_argument("--arm_void_width", type=int, default=0,
        help="[Quadcopter] Width (elements) of passive void strip along each arm centreline. "
             "Forces two distinct skeleton branches per arm. Recommended: 3-5. 0 = disabled (default).")

    # === Output & Visualisation ===
    g_out = parser.add_argument_group("Output & Visualisation")
    g_out.add_argument("--output", type=str, default="full_control_beam.json", help="Final output JSON filename")
    g_out.add_argument("--output_dir", type=str, default="output/hybrid_v2", help="Output directory")
    g_out.add_argument("--visualize", action="store_true", help="Show 3D debug windows")

    # === Skip Stages ===
    g_skip = parser.add_argument_group("Advanced")
    g_skip.add_argument("--skip_top3d", action="store_true", help="Skip Stage 0 (use existing .npz)")
    g_skip.add_argument("--top3d_npz", type=str, default=None, help="Path to existing .npz")

    g_mesh = parser.add_argument_group(
        "External Mesh Input",
        "Feed an STL/OBJ/PLY from any external TO solver directly into the pipeline. "
        "Skips Top3D. Geometry-only by default; add --optimize to run FEM stages.")
    g_mesh.add_argument("--mesh_input", type=str, default=None, metavar="PATH",
        help="Path to solid mesh file (STL / OBJ / PLY). "
             "Replaces --skip_top3d + --top3d_npz.")
    g_mesh.add_argument("--mesh_pitch", type=float, default=None,
        help="Voxel size for mesh voxelization in mm (defaults to --pitch).")

    args = parser.parse_args()

    # ── External mesh input: voxelise → write NPZ → treat as skip_top3d ────────
    if args.mesh_input is not None:
        from src.mesh_import.mesh_voxelizer import save_mesh_as_npz
        os.makedirs(args.output_dir, exist_ok=True)
        _vox_pitch = args.mesh_pitch if args.mesh_pitch is not None else args.pitch
        _mesh_stem = os.path.splitext(os.path.basename(args.mesh_input))[0]
        _vox_npz   = os.path.join(args.output_dir, f"{_mesh_stem}_voxelized.npz")
        save_mesh_as_npz(args.mesh_input, _vox_npz, pitch=_vox_pitch)
        args.skip_top3d = True
        args.top3d_npz  = _vox_npz
        if not args.optimize:
            print("[MeshInput] Geometry-only mode (add --optimize to run FEM stages).")

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.output))[0]

    # Resolve load_vec for TaggedProblem
    if args.load_fx is not None or args.load_fy is not None or args.load_fz is not None:
        load_vec = [
            args.load_fx if args.load_fx is not None else 0.0,
            args.load_fy if args.load_fy is not None else 0.0,
            args.load_fz if args.load_fz is not None else 0.0
        ]
    else:
        load_vec = None

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
            "--load_dist", args.load_dist,
            "--problem", args.problem if args.problem != "tagged" else "cantilever",
            "--output", npz_path,
        ]
        if args.load_fx is not None: cmd += ["--load_fx", str(args.load_fx)]
        if args.load_fy is not None: cmd += ["--load_fy", str(args.load_fy)]
        if args.load_fz is not None: cmd += ["--load_fz", str(args.load_fz)]
        if args.load_x is not None: cmd += ["--load_x", str(args.load_x)]
        if args.load_y is not None: cmd += ["--load_y", str(args.load_y)]
        if args.load_z is not None: cmd += ["--load_z", str(args.load_z)]
        if args.problem == "quadcopter":
            cmd += ["--motor_arm_frac", str(args.motor_arm_frac),
                    "--load_patch_frac", str(args.load_patch_frac),
                    "--motor_radius", str(args.motor_radius),
                    "--motor_bolt_spacing", str(args.motor_bolt_spacing),
                    "--arm_load_n", str(args.arm_load_n),
                    "--arm_load_frac", str(args.arm_load_frac),
                    "--arm_void_width", str(args.arm_void_width)]

        if not run_stage(cmd, "STAGE 0: Python Top3D Topology Optimisation"):
            return 1

    # ── Load Top3D compliance history (saved to NPZ by run_top3d.py) ──────
    top3d_hist = []
    try:
        _npz = np.load(npz_path, allow_pickle=True)
        if 'compliance_history' in _npz:
            top3d_hist = [float(v) for v in _npz['compliance_history']]
    except Exception:
        pass

    # ========================================
    # STAGE 1: Baseline Yin Reconstruction
    # ========================================
    stage1_out = os.path.join(args.output_dir, f"{base_name}_1_reconstructed.json")

    print(f"\n{'='*60}")
    print("  STAGE 1: Skeleton Reconstruction")
    print(f"{'='*60}")
    try:
        reconstruct_npz(
            npz_path, stage1_out,
            pitch=args.pitch, max_iters=args.max_iters,
            collapse_thresh=args.collapse_thresh, prune_len=args.prune_len,
            rdp_epsilon=args.rdp, radius_mode=args.radius_mode,
            vol_thresh=args.vol_thresh,
            load_fx=args.load_fx, load_fy=args.load_fy, load_fz=args.load_fz,
            hybrid=args.hybrid,
            plate_thickness_ratio=args.plate_thickness_ratio,
            min_plate_size=args.min_plate_size,
            flatness_ratio=args.flatness_ratio,
            junction_thresh=args.junction_thresh,
            min_avg_neighbors=args.min_avg_neighbors,
            plate_mode=args.plate_mode,
            detect_plates=args.detect_plates,
            curved=args.curved,
            visualize=args.visualize,
        )
    except Exception as e:
        print(f"[FATAL] STAGE 1: Skeleton Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================
    # Check Stage 1 output for beam edges
    # ========================================
    with open(stage1_out, 'r') as f:
        baseline_data = json.load(f)

    n_beam_edges = len(baseline_data.get('graph', {}).get('edges', []))
    has_plates = len(baseline_data.get('plates', [])) > 0
    plate_only = (n_beam_edges == 0 and has_plates)

    # Beam-only (not hybrid) optimises by default, UNLESS the input came from
    # an external mesh without an explicit --optimize flag (geometry-only mode).
    _mesh_geo_only = (args.mesh_input is not None and not args.optimize)
    run_opt = (not plate_only) and (args.optimize or (not args.hybrid and not _mesh_geo_only))

    if not run_opt:
        if plate_only:
            print(f"\n[Pipeline] Plate-only structure (0 beam edges). Skipping optimization stages.")
        elif _mesh_geo_only:
            print("\n[Pipeline] Mesh input (geometry-only). Skipping Stages 2 & 3. Add --optimize to run FEM.")
        else:
            print("\n[Pipeline] Hybrid structure without --optimize. Skipping Stages 2 & 3.")
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
        if plate_only:
            print("              PIPELINE COMPLETE (Plate-Only)")
        else:
            print("              PIPELINE COMPLETE (Unoptimized Hybrid)")
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
    problem_config = TaggedProblem(load_vector=load_vec)
    problem_config.load_tags_from_json(stage1_out)
    c_baseline = _compute_compliance(baseline_data, problem_config, E=1000.0)

    metrics_list = []
    convergence_stages = []   # {'type', 'loop', 'history'} per size/layout stage
    current_json = stage1_out
    # Final opt outputs (set inside loop; used for report after loop)
    nodes_opt = None
    edges_opt = None
    radii_opt = None

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
            radii_size = np.array([e[2] for e in edges_raw])

            size_problem = TaggedProblem(load_vector=load_vec)
            size_problem.load_tags_from_json(current_json)

            radii_sized, c_size_init, c_size_final, size_hist = optimize_size(
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
            convergence_stages.append({
                'type': 'size', 'loop': loop_num, 'history': size_hist,
                'c_init': c_size_init, 'c_final': c_size_final,
                'radii': radii_sized,
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

            layout_problem = TaggedProblem(load_vector=load_vec)
            layout_problem.load_tags_from_json(sized_json)

            nodes_opt, edges_opt, radii_opt, tags_opt, c_layout_init, c_layout_final, layout_hist = optimize_layout(
                nodes_layout, edges_layout, radii_layout, layout_problem,
                E=1000.0, move_limit=args.limit, visualize=args.visualize,
                target_volume_abs=target_volume, snap_dist=args.snap,
                design_bounds=design_bounds, node_tags=node_tags
            )

            # --- Post-Optimization Dead-Weight Pruning ---
            if args.prune_opt_thresh > 0.0 and len(radii_opt) > 0:
                max_r = np.max(radii_opt)
                cutoff = max_r * args.prune_opt_thresh
                
                keep_edges = []
                keep_radii = []
                keep_node_indices = set(tags_opt.keys())  # Always keep tagged nodes
                
                # 1. Filter edges
                for idx, r in enumerate(radii_opt):
                    if r > cutoff:
                        keep_edges.append(edges_opt[idx])
                        keep_radii.append(r)
                        keep_node_indices.add(edges_opt[idx][0])
                        keep_node_indices.add(edges_opt[idx][1])
                        
                n_removed = len(radii_opt) - len(keep_edges)
                if n_removed > 0:
                    print(f"\n[Prune] Removing {n_removed} dead-weight edges (radius <= {cutoff:.3f} mm, {args.prune_opt_thresh*100:.1f}% of max {max_r:.2f} mm)")
                    
                    # 2. Filter nodes
                    keep_node_indices = sorted(list(keep_node_indices))
                    new_nodes = [nodes_opt[i] for i in keep_node_indices]
                    
                    # 3. Create mapping for edges
                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_node_indices)}
                    
                    # 4. Rebuild edges & tags
                    new_edges = np.array([[old_to_new[e[0]], old_to_new[e[1]]] for e in keep_edges], dtype=int)
                    new_tags = {old_to_new[k]: v for k, v in tags_opt.items() if k in old_to_new}
                    
                    nodes_opt = np.array(new_nodes)
                    edges_opt = new_edges
                    radii_opt = np.array(keep_radii)
                    tags_opt = new_tags
                    print(f"[Prune] Graph reduced to {len(nodes_opt)} nodes, {len(edges_opt)} edges.")

            curves_layout = _refit_curves(nodes_opt, edges_opt, radii_opt, args.curved)

            layout_data = {
                "metadata": layout_input.get("metadata", {}),
                "graph": {
                    "nodes": nodes_opt.tolist(),
                    "edges": [[int(u), int(v), float(r)] for u, v, r in zip(edges_opt[:, 0], edges_opt[:, 1], radii_opt)],
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
            convergence_stages.append({
                'type': 'layout', 'loop': loop_num, 'history': layout_hist,
                'c_init': c_layout_init, 'c_final': c_layout_final,
                'nodes_before': nodes_layout, 'nodes_after': nodes_opt,
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

    # ========================================
    # Convergence Plots + Pipeline Report
    # ========================================
    try:
        fig_base = os.path.join(args.output_dir, base_name)

        # Figure (a) — Top3D SIMP convergence
        if len(top3d_hist) > 1:
            plot_top3d_convergence(
                top3d_hist,
                f"{fig_base}_fig_top3d",
                mesh_size=(args.nelx, args.nely, args.nelz),
                volfrac=args.volfrac,
            )

        # Figure (b) — Size + Layout optimisation trajectory
        if convergence_stages:
            plot_size_layout_convergence(
                convergence_stages,
                f"{fig_base}_fig_opt",
            )

        # Figure (c) — Combined full-pipeline (SIMP + frame on one axis)
        if convergence_stages and np.isfinite(c_baseline):
            plot_combined_convergence(
                top3d_hist, convergence_stages, c_baseline,
                f"{fig_base}_fig_combined",
                mesh_size=(args.nelx, args.nely, args.nelz),
                volfrac=args.volfrac,
            )

        # Text + JSON pipeline report
        report_data = _build_report_data(
            args, baseline_data, metrics_list, convergence_stages,
            top3d_hist, target_volume, c_baseline,
            nodes_opt=nodes_opt,
            edges_opt=edges_opt,
            radii_opt=radii_opt,
            baseline_nodes=baseline_nodes,
        )
        generate_pipeline_report(report_data, f"{fig_base}_report.txt")

    except Exception as e:
        print(f"[Warning] Could not generate convergence plots/report: {e}")
        import traceback
        traceback.print_exc()

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
