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
from src.optimization.fem import solve_frame, compute_beam_strain_energy
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


def _compute_compliance(json_data, problem_config, E=2.1e5, return_beam_se=False,
                        plates=None):
    """One-shot FEA solve to get compliance from a JSON graph.

    When *plates* is provided, shell elements are included in the FEM assembly
    (beam + plate combined stiffness), matching what size/layout optimisation sees.
    """
    nodes = np.array(json_data['graph']['nodes'])
    edges_raw = json_data['graph']['edges']
    edges = np.array([[e[0], e[1]] for e in edges_raw], dtype=int)
    radii = np.array([e[2] for e in edges_raw])

    loads, bcs = problem_config.apply(nodes)

    has_plates = plates is not None and len(plates) > 0
    if has_plates:
        plate_thicknesses = []
        for p in plates:
            ms = p.get("mid_surface")
            h = p.get("thickness", 1.0)
            if ms is not None:
                h = ms.get("mean_thickness", h)
            plate_thicknesses.append(h)
        result = solve_frame(nodes, edges, radii, E=E, loads=loads, bcs=bcs,
                             plates=plates, plate_thicknesses=plate_thicknesses)
        u, compliance, elements = result[0], result[1], result[2]
    else:
        u, compliance, elements = solve_frame(nodes, edges, radii, E=E, loads=loads, bcs=bcs)

    if return_beam_se:
        beam_se = compute_beam_strain_energy(u, elements)
        return compliance, beam_se
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


def _fmt_c(v, width=13):
    """Adaptive compliance formatter: scientific notation when |v| < 0.01."""
    if v == 0:
        return f"{'0.0000':>{width}}"
    if abs(v) < 0.01:
        return f"{v:>{width}.5e}"
    return f"{v:>{width},.4f}"


def _print_comparison_table(metrics_list, baseline_volume, target_volume,
                            top3d_hist=None, continuum_results=None):
    """Print formatted comparison table of iterative metrics."""
    print("\n" + "="*130)
    print(" " * 30 + "ITERATIVE LAYOUT + SIZE OPTIMISATION — COMPARISON SUMMARY")
    print("="*130)
    print(f" {'Iter':<5} | {'Stage':<8} | {'Compliance':<14} | {'Δ vs Prev':<10} | {'Volume (mm³)':<13} | {'Vol Err%':<9} | {'Geo. Score':<10} | {'Chamfer (mm)':<12} | {'Nodes':<6} | {'Edges':<6}")
    print("-"*130)

    if top3d_hist and len(top3d_hist) > 0:
        c_simp = top3d_hist[-1]
        c_simp_initial = top3d_hist[0]
        simp_reduction = ((c_simp_initial - c_simp) / c_simp_initial * 100.0) if c_simp_initial > 0 else 0.0
        print(f" {'—':<5} | {'SIMP*':<8} | {_fmt_c(c_simp)} | {-simp_reduction:>9.2f}% | {target_volume:>12.2f} | {0.00:>8.2f}% | {'—':<10} | {'—':<12} | {'—':<6} | {'—':<6}")

    print(f" {'—':<5} | {'Baseline':<8} | {_fmt_c(baseline_volume)} | {'—':<10} | {target_volume:>12.2f} | {0.00:>8.2f}% | {1.000:<10.3f} | {0.00:<12.2f} | {'—':<6} | {'—':<6}")

    prev_compliance = baseline_volume
    for row in metrics_list:
        compliance = row['c_layout'] if row['stage'] == 'Layout' else row['c_size']
        delta_pct = ((compliance - prev_compliance) / prev_compliance * 100.0) if prev_compliance > 0 else 0.0
        volume = row['v_layout'] if row['stage'] == 'Layout' else row['v_size']
        vol_err = ((volume - target_volume) / target_volume * 100.0) if target_volume > 0 else 0.0

        print(f" {row['iter']:<5} | {row['stage']:<8} | {_fmt_c(compliance)} | {delta_pct:>9.2f}% | {volume:>12.2f} | {vol_err:>8.2f}% | {row['geo_score']:<10.3f} | {row['mean_chamfer']:<12.2f} | {row['n_nodes']:<6} | {row['n_edges']:<6}")
        prev_compliance = compliance

    print("="*130)

    if top3d_hist and len(top3d_hist) > 0:
        print(f"* SIMP uses continuum FEM (E-rescaled to 210 GPa). Not directly comparable to frame FEM — different model fidelity.")

    if len(metrics_list) > 0:
        last_stage = metrics_list[-1]['stage']
        final_compliance = metrics_list[-1]['c_layout'] if last_stage == 'Layout' else metrics_list[-1]['c_size']
        total_reduction = ((baseline_volume - final_compliance) / baseline_volume * 100.0) if baseline_volume > 0 else 0.0
        print(f"Total compliance reduction: {total_reduction:>6.2f}%  ({_fmt_c(baseline_volume, 0).strip()} → {_fmt_c(final_compliance, 0).strip()})")

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
    # ── Yin-style direct compliance comparison (J = f·u) ────────────
    if continuum_results and 'simp_p1_rescaled' in continuum_results:
        c_simp_210k = continuum_results['simp_p1_rescaled']
        print(f"\n{'─'*78}")
        print(f"  DIRECT COMPLIANCE COMPARISON  (Yin et al. 2020 methodology)")
        print(f"  All values: J = f·u,  E = 210 GPa,  same loads/BCs")
        print(f"{'─'*78}")
        print(f"  {'Stage':<35}  {'C (f·u)':>14}  {'Δ vs SIMP':>10}")
        print(f"  {'-'*62}")
        print(f"  {'SIMP binary (hex FEM, p=1)':<35}  {_fmt_c(c_simp_210k, 14)}  {'—':>10}")

        for stage in continuum_results.get('yin_stages', []):
            c_st = stage['compliance']
            pct = ((c_st - c_simp_210k) / c_simp_210k * 100.0) if c_simp_210k > 0 else 0.0
            print(f"  {stage['label']:<35}  {_fmt_c(c_st, 14)}  {pct:>+9.1f}%")

        print(f"  {'-'*62}")
        print(f"  SIMP: continuum hex FEM (p=1 binary).  Frame: beam FEM.")
        print(f"  Both compute J = f·u (work done by loads) at E = 210 GPa.")
        print(f"{'─'*78}")

    print("="*130 + "\n")


def _refit_curves(nodes, edges, radii, curved, ctrl_pts_list=None,
                   prev_curves=None):
    """Build curves list from optimised nodes+edges, with optional Bézier fitting.

    If *ctrl_pts_list* is provided (from IGA optimisation), uses those control
    points directly instead of re-fitting from scratch.

    If *prev_curves* is provided (from a previous pipeline stage), preserves
    original ctrl_pts / polyline waypoints — deforming them to match any
    endpoint movement and updating the radius column.
    """
    curves = []
    for idx, ((u, v), r) in enumerate(zip(edges, radii)):
        p_start = np.array(nodes[u], dtype=float)
        p_end = np.array(nodes[v], dtype=float)

        # Retrieve previous curve entry for this edge (if available)
        prev = prev_curves[idx] if prev_curves and idx < len(prev_curves) else None

        if curved and _CURVES_AVAILABLE:
            # Determine control points: explicit > previous stage > fit fresh
            cp = None
            if ctrl_pts_list is not None and ctrl_pts_list[idx] is not None:
                cp = ctrl_pts_list[idx]  # (2, 3) from IGA optimisation
            elif prev is not None and "ctrl_pts" in prev and prev["ctrl_pts"]:
                # Deform previous ctrl_pts to match updated endpoints
                prev_cp = prev["ctrl_pts"]
                prev_pts = prev.get("points", [])
                if len(prev_pts) >= 2:
                    old_start = np.array(prev_pts[0][:3], dtype=float)
                    old_end = np.array(prev_pts[-1][:3], dtype=float)
                    delta_start = p_start - old_start
                    delta_end = p_end - old_end
                    cp = []
                    for ci, t_approx in enumerate([1/3, 2/3]):
                        p = np.array(prev_cp[ci], dtype=float)
                        p += (1 - t_approx) * delta_start + t_approx * delta_end
                        cp.append(p)
                else:
                    cp = [np.array(c, dtype=float) for c in prev_cp]
            elif prev is not None and prev.get("points") and len(prev["points"]) > 2:
                # Previous stage had waypoints but no ctrl_pts — fit from waypoints
                waypoints = [np.array(pt[:3], dtype=float) for pt in prev["points"][1:-1]]
                try:
                    cp_pair = fit_cubic_bezier(p_start, p_end, waypoints)
                    cp = [cp_pair[0], cp_pair[1]]
                except Exception:
                    cp = None

            if cp is not None:
                try:
                    pts = sample_curve_points(p_start, cp[0], cp[1], p_end, float(r), N=20)
                    curves.append({
                        "ctrl_pts": [cp[0].tolist() if hasattr(cp[0], 'tolist') else list(cp[0]),
                                     cp[1].tolist() if hasattr(cp[1], 'tolist') else list(cp[1])],
                        "points": pts, "radius": float(r)})
                except Exception:
                    # Fallback to straight if sampling fails
                    curves.append({"points": [list(p_start) + [float(r)], list(p_end) + [float(r)]]})
            else:
                # No curvature info — straight segment
                curves.append({"points": [list(p_start) + [float(r)], list(p_end) + [float(r)]]})
        else:
            # Straight mode: preserve original polyline waypoints from previous stage
            prev_pts = prev.get("points", []) if prev else []

            if len(prev_pts) > 2:
                # Deform intermediate waypoints to follow endpoint movement.
                old_start = np.array(prev_pts[0][:3], dtype=float)
                old_end = np.array(prev_pts[-1][:3], dtype=float)
                delta_start = p_start - old_start
                delta_end = p_end - old_end

                new_pts = []
                n = len(prev_pts) - 1
                for j, pt in enumerate(prev_pts):
                    t = j / n  # 0 at start, 1 at end
                    xyz = np.array(pt[:3], dtype=float) + (1 - t) * delta_start + t * delta_end
                    new_pts.append(list(xyz) + [float(r)])
                curves.append({"points": new_pts})
            else:
                curves.append({"points": [list(p_start) + [float(r)], list(p_end) + [float(r)]]})
    return curves


def _extract_ctrl_pts(curves_data, n_edges):
    """Extract ctrl_pts list from Stage 1 JSON curves for IGA optimisation.

    Returns a list of (2, 3) arrays (one per edge), or None entries for
    edges without control point data.
    """
    ctrl_pts_list = []
    for i in range(n_edges):
        if i < len(curves_data) and 'ctrl_pts' in curves_data[i]:
            cp_raw = curves_data[i]['ctrl_pts']
            ctrl_pts_list.append(np.array(cp_raw, dtype=float))
        else:
            ctrl_pts_list.append(None)
    return ctrl_pts_list


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
                       baseline_nodes=None, continuum_results=None):
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
        'graph_stages': meta.get('graph_stages', []),
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
            if nb is not None and na is not None:
                nb_arr = np.array(nb)
                na_arr = np.array(na)
                if len(nb_arr) == len(na_arr):
                    disp = np.linalg.norm(na_arr - nb_arr, axis=1)
                else:
                    # Snap merged nodes — match via nearest-neighbour
                    from scipy.spatial import cKDTree
                    tree = cKDTree(na_arr)
                    dists, _ = tree.query(nb_arr, k=1)
                    disp = dists
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
        'continuum':         continuum_results or {},
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

    # === Beam Representation ===
    g_curved = parser.add_argument_group("Beam Representation")
    g_curved.add_argument("--beam_mode", type=str, default="straight",
                          choices=["straight", "curved", "mixed"],
                          help="Beam representation mode: "
                               "'straight' = all Euler-Bernoulli (default), "
                               "'curved' = all Bézier IGA, "
                               "'mixed' = per-edge classification (straight below threshold, curved above)")
    g_curved.add_argument("--curve_threshold", type=float, default=None,
                          help="[mixed mode] Max perpendicular deviation (mm) for classifying edge as curved. "
                               "Below this → straight beam. Default: 0.3×pitch.")
    g_curved.add_argument("--smooth_iters", type=int, default=5,
                          help="[curved] Laplacian smoothing iterations for edge waypoints (default: 5)")
    g_curved.add_argument("--smooth_decimate", type=int, default=1,
                          help="[curved] Decimation stride after smoothing (1=none, 2=half)")

    # === Layout & Size Optimisation ===
    g_opt = parser.add_argument_group("Optimization Configuration")
    g_opt.add_argument("--optimize", action="store_true", help="Enable Beam Layout & Size Optimisation (default: False for hybrid mode)")
    g_opt.add_argument("--iters", type=int, default=50, help="Max iterations for internal Top3D or Layout optimization")
    g_opt.add_argument("--opt_loops", type=int, default=2, help="Number of Size + Layout iteration loops")
    g_opt.add_argument("--limit", type=float, default=5.0, help="Move limit for layout optimization (mm)")
    g_opt.add_argument("--prune_opt_thresh", type=float, default=0.0, help="Percentage (0.0-1.0) of max radius to prune dead-weight edges post-optimization")
    g_opt.add_argument("--snap", type=float, default=5.0, help="Snap distance for node merging (mm)")
    g_opt.add_argument("--r_min", type=float, default=0.1, help="Minimum beam radius in size optimisation (mm). Raise to prevent low-sensitivity beams from collapsing (e.g. 0.5)")
    g_opt.add_argument("--r_max", type=float, default=5.0, help="Maximum beam radius in size optimisation (mm)")
    g_opt.add_argument("--ctrl_limit", type=float, default=None,
                       help="Max displacement of Bézier interior control points in layout opt (mm). "
                            "Defaults to 0.3 × --limit. Reduce (e.g. 0.5) to prevent extreme arch curvature in --curved mode.")
    g_opt.add_argument("--geo_reg", type=float, default=0.0,
                       help="Geometric regularization weight for layout opt. "
                            "Penalizes node drift from original skeleton topology. "
                            "0 = off (default). Try 0.1–1.0 to keep structure close to TO result.")
    g_opt.add_argument("--vol_weight", type=float, default=10.0,
                       help="Volume penalty weight for layout optimization. "
                            "Penalizes volume deviation from target to prevent layout "
                            "inflating structure. 0 = off. Default 10.0.")
    g_opt.add_argument("--symmetry", type=str, default=None,
                       help="Enforce symmetry about specified planes through domain center. "
                            "Comma-separated: 'xz' (Y-mirror), 'yz' (X-mirror), 'xy' (Z-mirror). "
                            "Example: --symmetry xz,yz for two-plane symmetry.")
    g_opt.add_argument("--sym_weight", type=float, default=0.1,
                       help="Weight for symmetry penalty in layout optimization. "
                            "Higher = tighter symmetry vs. compliance trade-off. "
                            "Default: 0.1. Typical range: 0.01–1.0.")
    g_opt.add_argument("--sym_tol", type=float, default=None,
                       help="Distance tolerance for symmetric node matching (mm). "
                            "Default: 1.5 * pitch.")
    g_opt.add_argument("--problem", type=str, default="tagged",
                       help="Problem config: 'tagged' (auto from BC tags), 'cantilever', 'roof_slab', 'elevated_slab', 'bridge', 'deck', 'rocker_arm', 'quadcopter'")
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
    g_out.add_argument("--export_stl", action="store_true",
                       help="Export Top3D result as watertight STL (marching cubes isosurface) for CAD/FEA import")
    g_out.add_argument("--fig_size", type=str, default=None,
                       help="Figure size as 'WxH' in inches (e.g. '3.5x2.6' for IEEE single-column). "
                            "Default: 9x4 (wide format).")
    g_out.add_argument("--export_vtk", action="store_true",
                       help="Export .vtr files for ParaView (requires pyevtk)")
    g_out.add_argument("--no_render_3d", dest="render_3d", action="store_false", default=True,
                       help="Disable 3D FEA rendering (faster pipeline runs)")
    g_out.add_argument("--render_upsample", type=int, default=1, metavar="N",
                       help="Upsample voxel renders by Nx (re-solves FEM on finer grid). "
                            "E.g. --render_upsample 2 doubles resolution. Default: 1 (no upsampling)")

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

    # ── Translate --beam_mode into internal flags ──────────────────────────────
    # Backwards compat: --curved (if still present) maps to beam_mode=curved
    if getattr(args, 'curved', False) and args.beam_mode == 'straight':
        args.beam_mode = 'curved'
    if args.beam_mode == 'straight':
        args.curved = False
        args.curve_threshold = None
    elif args.beam_mode == 'curved':
        args.curved = True
        args.curve_threshold = 0.0  # force all edges curved
    elif args.beam_mode == 'mixed':
        args.curved = True
        # curve_threshold stays as user-specified or None (auto = 0.3*pitch)

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

    # Auto-read load vector from NPZ if not specified on CLI
    _npz_for_load = args.top3d_npz if args.skip_top3d and args.top3d_npz else None
    if load_vec is None and _npz_for_load and os.path.exists(_npz_for_load):
        try:
            _npz_data = np.load(_npz_for_load, allow_pickle=True)
            if 'load_vector' in _npz_data:
                lv = _npz_data['load_vector']
                load_vec = lv.tolist()
                print(f"[AutoLoad] Read load vector from NPZ: {load_vec}")
        except Exception:
            pass

    mode_str = "HYBRID BEAM+PLATE" if args.hybrid else "BEAM-ONLY"
    if args.beam_mode == 'curved':
        mode_str += " + ALL CURVED"
    elif args.beam_mode == 'mixed':
        thresh_str = f"{args.curve_threshold:.2f}mm" if args.curve_threshold is not None else "auto"
        mode_str += f" + MIXED (threshold={thresh_str})"

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
    # SIMP uses E0=1.0 for numerical stability (topology is E-invariant).
    # Rescale compliance to frame FEM E so values are directly comparable:
    #   C ∝ 1/E  →  C_frame_scale = C_simp × (E_simp / E_frame)
    E_frame = 2.1e5
    top3d_hist = []
    try:
        _npz = np.load(npz_path, allow_pickle=True)
        if 'compliance_history' in _npz:
            E_simp = float(_npz['E0']) if 'E0' in _npz else 1.0
            scale = E_simp / E_frame
            top3d_hist = [float(v) * scale for v in _npz['compliance_history']]
            if abs(scale - 1.0) > 1e-6:
                print(f"[Report] Rescaled SIMP compliance: E_simp={E_simp:.0f} → E_frame={E_frame:.0f} (×{scale:.6f})")
    except Exception:
        pass

    # ── Optional: Export Top3D result as STL ──────────────────────────────
    if args.export_stl:
        from src.export.npz_to_stl import export_top3d_stl
        stl_path = os.path.join(args.output_dir, f"{base_name}_top3d.stl")
        export_top3d_stl(npz_path, stl_path, vol_thresh=args.vol_thresh, pitch=args.pitch)

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
            curve_threshold=args.curve_threshold,
            smooth_iters=args.smooth_iters,
            smooth_decimate=args.smooth_decimate,
            visualize=args.visualize,
            symmetry=args.symmetry,
            sym_tol=args.sym_tol,
            skip_extrema=(args.mesh_input is not None),
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

    # Subtract plate volume from beam target: plates are fixed geometry,
    # so the beam optimizer should only fill the remaining volume budget.
    if plates_data and target_volume is not None:
        plate_vol = sum(p.get('volume', 0.0) for p in plates_data)
        if plate_vol > 0:
            print(f"[Volume] Total solid: {target_volume:.2f} mm³, "
                  f"plates: {plate_vol:.2f} mm³ ({plate_vol/target_volume*100:.1f}%)")
            target_volume = max(target_volume - plate_vol, target_volume * 0.05)
            print(f"[Volume] Beam target (total − plates): {target_volume:.2f} mm³")

    # Compute baseline compliance
    problem_config = TaggedProblem(load_vector=load_vec)
    if args.top3d_npz:
        problem_config.set_load_position_from_npz(args.top3d_npz, args.pitch)
    problem_config.load_tags_from_json(stage1_out)
    baseline_plates = plates_data if (args.hybrid and plates_data) else None
    c_baseline, _baseline_beam_se = _compute_compliance(
        baseline_data, problem_config, E=2.1e5, return_beam_se=True,
        plates=baseline_plates)

    # ── Symmetry setup ──────────────────────────────────────────────
    sym_data = None
    if args.symmetry and design_bounds:
        from src.optimization.symmetry import (
            parse_symmetry_planes, find_symmetric_node_pairs,
            find_symmetric_edge_pairs,
        )
        sym_tol = args.sym_tol if args.sym_tol is not None else 1.5 * args.pitch
        sym_planes = parse_symmetry_planes(args.symmetry, design_bounds)

        _bl_nodes = np.array(baseline_data['graph']['nodes'])
        _bl_edges_raw = baseline_data['graph']['edges']
        _bl_edges = np.array([[e[0], e[1]] for e in _bl_edges_raw], dtype=int)

        sym_node_info = find_symmetric_node_pairs(
            _bl_nodes, sym_planes, tol=sym_tol)
        sym_edge_info = find_symmetric_edge_pairs(_bl_edges, sym_node_info)

        sym_data = {
            'planes': sym_planes,
            'tol': sym_tol,
            'weight': args.sym_weight,
        }

        for plane in sym_planes:
            pname = plane['name']
            n_pairs = len(sym_node_info[pname]['node_pairs'])
            n_on_plane = len(sym_node_info[pname]['on_plane_nodes'])
            n_unmatched = len(sym_node_info[pname]['unmatched_nodes'])
            e_pairs = len(sym_edge_info[pname])
            print(f"[Symmetry] Plane {pname.upper()}: "
                  f"{n_pairs} node pairs, {n_on_plane} on-plane, "
                  f"{n_unmatched} unmatched, {e_pairs} edge pairs")

    # ── Continuum FEM comparison (same solver for SIMP and frame) ─────
    continuum_results = {}
    _cont_eval = None
    _rho_simp_binary = None
    try:
        from src.optimization.voxelize_frame import voxelize_beam_frame
        from src.optimization.top3d import Top3D as _Top3D

        _npz_data = np.load(npz_path, allow_pickle=True)
        _bc_tags = _npz_data['bc_tags']
        _c_nely, _c_nelx, _c_nelz = _bc_tags.shape
        _c_pitch = float(_npz_data['pitch']) if 'pitch' in _npz_data else args.pitch
        _c_origin = np.array(_npz_data['origin'], dtype=float) if 'origin' in _npz_data else np.zeros(3)
        _rho_simp = _npz_data['rho']

        _cont_eval = _Top3D(_c_nelx, _c_nely, _c_nelz, volfrac=0.0,
                            penal=args.penal, rmin=1.5)
        _cont_eval.setup_from_tags(_bc_tags, load_vec)

        # SIMP continuum compliance — threshold to binary so it matches
        # the binary voxelised frames (no penalisation artefact).
        # This is the same density field that feeds skeleton extraction.
        # Yin et al. (2020): re-evaluate with p=1 on binary field, then
        # rescale from E0=1000 to E=210k for direct comparison with frame FEM.
        _rho_simp_binary = (_rho_simp >= args.vol_thresh).astype(float)
        c_simp_cont, se_simp, u_simp, vm_simp = _cont_eval.evaluate(
            _rho_simp_binary, return_fields=True, return_stress=True,
            density_mask_thresh=args.vol_thresh, exclude_void=True)
        continuum_results['simp'] = c_simp_cont
        # Rescale to E=210 GPa (SIMP runs at E0=1000; C ∝ 1/E)
        c_simp_p1_rescaled = c_simp_cont * (1000.0 / E_frame)
        continuum_results['simp_p1_rescaled'] = c_simp_p1_rescaled

        # ── Hi-res interpolation for 3D rendering (geometry-only upsample) ──
        _up = args.render_upsample
        if _up > 1:
            from scipy.ndimage import zoom as _zoom
            _hr_nelx, _hr_nely, _hr_nelz = _c_nelx * _up, _c_nely * _up, _c_nelz * _up
            _hr_pitch = _c_pitch / _up
            # Upsample density (order=0, binary stays binary), SE and VM stress (order=1, smooth)
            _hr_rho_simp = _zoom(_rho_simp_binary, _up, order=0)
            _hr_se_simp = _zoom(se_simp, _up, order=1)
            _hr_vm_simp = _zoom(vm_simp, _up, order=1) if vm_simp is not None else None
            # Displacement: interpolate node grid (nely+1, nelx+1, nelz+1) then re-flatten
            if u_simp is not None and len(u_simp) > 0:
                _nn = (_c_nely+1, _c_nelx+1, _c_nelz+1)
                _hr_nn = (_hr_nely+1, _hr_nelx+1, _hr_nelz+1)
                _hr_u_components = []
                for _comp in range(3):
                    _u_grid = u_simp[_comp::3].reshape(_nn, order='F')
                    _u_hr = _zoom(_u_grid, [_hr_nn[i]/_nn[i] for i in range(3)], order=1)
                    _hr_u_components.append(_u_hr)
                _hr_u_simp = np.zeros(_hr_nn[0]*_hr_nn[1]*_hr_nn[2]*3)
                for _comp in range(3):
                    _hr_u_simp[_comp::3] = _hr_u_components[_comp].flatten(order='F')
            else:
                _hr_u_simp = u_simp
            print(f"[Render] Upsampled {_c_nelx}x{_c_nely}x{_c_nelz} → "
                  f"{_hr_nelx}x{_hr_nely}x{_hr_nelz} ({_up}x) for 3D renders (interpolated, no FEM)")
            _r_se_simp, _r_rho_simp, _r_u_simp = _hr_se_simp, _hr_rho_simp, _hr_u_simp
            _r_vm_simp = _hr_vm_simp
            _r_nelx, _r_nely, _r_nelz, _r_pitch, _r_origin = _hr_nelx, _hr_nely, _hr_nelz, _hr_pitch, _c_origin
        else:
            _r_se_simp, _r_rho_simp, _r_u_simp = se_simp, _rho_simp_binary, u_simp
            _r_vm_simp = vm_simp
            _r_nelx, _r_nely, _r_nelz, _r_pitch, _r_origin = _c_nelx, _c_nely, _c_nelz, _c_pitch, _c_origin

        # Save SIMP strain energy plot (2D fallback)
        try:
            from src.pipelines.baseline_yin.visualization import save_strain_energy_plot
            _se_base = os.path.join(args.output_dir, f"{base_name}_se_simp")
            save_strain_energy_plot(se_simp, _rho_simp_binary, _c_pitch, _c_origin,
                                   _se_base, title="Strain Energy — SIMP (binary)")
        except Exception as _e:
            print(f"[Warning] SIMP strain energy plot skipped: {_e}")

        # 3D FEA visualization (PyVista)
        try:
            from src.export.vtk_export import render_pipeline_stage
            render_pipeline_stage(
                "SIMP Binary", _r_se_simp, _r_rho_simp, _r_u_simp,
                _r_nelx, _r_nely, _r_nelz, _r_pitch, _r_origin,
                args.output_dir, base_name,
                export_vtr=args.export_vtk, render_3d=args.render_3d,
                stress_field=_r_vm_simp,
            )
        except Exception as _e:
            print(f"[Warning] 3D SIMP visualization skipped: {_e}")

        # Yin-style stages: collect all frame compliance values at E=210k
        # for direct comparison with SIMP binary compliance
        continuum_results['yin_stages'] = []

        # Baseline frame compliance (already at E=210k)
        _bl_label = 'Baseline skeleton (beam+plate FEM)' if baseline_plates else 'Baseline skeleton (beam FEM)'
        continuum_results['yin_stages'].append({
            'label': _bl_label,
            'compliance': c_baseline,
        })

        # Save SIMP strain energy plot (already done above)
        # Save baseline frame strain energy via voxelization (for visual only)
        _b_nodes = np.array(baseline_data['graph']['nodes'])
        _b_edges_raw = baseline_data['graph']['edges']
        _b_edges = np.array([[e[0], e[1]] for e in _b_edges_raw], dtype=int)
        _b_radii = np.array([e[2] for e in _b_edges_raw])

        try:
            rho_base_frame = voxelize_beam_frame(
                _b_nodes, _b_edges, _b_radii, (_c_nely, _c_nelx, _c_nelz),
                pitch=_c_pitch, origin=_c_origin)
            _, se_base, u_base, vm_base = _cont_eval.evaluate(
                rho_base_frame, return_fields=True, return_stress=True,
                density_mask_thresh=0.5, exclude_void=True)
            from src.pipelines.baseline_yin.visualization import save_strain_energy_plot
            _se_base_f = os.path.join(args.output_dir, f"{base_name}_se_baseline_frame")
            save_strain_energy_plot(se_base, rho_base_frame, _c_pitch, _c_origin,
                                   _se_base_f, title="Strain Energy — Baseline Frame (voxelised)")

            # Hi-res interpolation for 3D rendering (no FEM re-solve)
            if _up > 1:
                from scipy.ndimage import zoom as _zoom
                _r_rho_base = _zoom(rho_base_frame, _up, order=0)
                _r_se_base = _zoom(se_base, _up, order=1)
                _r_vm_base = _zoom(vm_base, _up, order=1) if vm_base is not None else None
                if u_base is not None and len(u_base) > 0:
                    _nn = (_c_nely+1, _c_nelx+1, _c_nelz+1)
                    _hr_nn = (_r_nely+1, _r_nelx+1, _r_nelz+1)
                    _hr_u_comps = []
                    for _comp in range(3):
                        _u_g = u_base[_comp::3].reshape(_nn, order='F')
                        _hr_u_comps.append(_zoom(_u_g, [_hr_nn[i]/_nn[i] for i in range(3)], order=1))
                    _r_u_base = np.zeros(_hr_nn[0]*_hr_nn[1]*_hr_nn[2]*3)
                    for _comp in range(3):
                        _r_u_base[_comp::3] = _hr_u_comps[_comp].flatten(order='F')
                else:
                    _r_u_base = u_base
            else:
                _r_se_base, _r_rho_base, _r_u_base = se_base, rho_base_frame, u_base
                _r_vm_base = vm_base

            # 3D FEA visualization (PyVista)
            try:
                from src.export.vtk_export import render_pipeline_stage
                render_pipeline_stage(
                    "Baseline Frame", _r_se_base, _r_rho_base, _r_u_base,
                    _r_nelx, _r_nely, _r_nelz, _r_pitch, _r_origin,
                    args.output_dir, base_name,
                    export_vtr=args.export_vtk, render_3d=args.render_3d,
                    frame_data={
                        'nodes': _b_nodes, 'edges': _b_edges, 'radii': _b_radii,
                        'node_tags': baseline_data['graph'].get('node_tags', {}),
                        'beam_se': _baseline_beam_se,
                    },
                    stress_field=_r_vm_base,
                )
            except Exception as _e2:
                print(f"[Warning] 3D baseline frame visualization skipped: {_e2}")
        except Exception as _e:
            print(f"[Warning] Baseline frame strain energy plot skipped: {_e}")

        # Store grid params for final-stage strain energy only
        continuum_results['grid_shape'] = (_c_nely, _c_nelx, _c_nelz)
        continuum_results['pitch'] = _c_pitch
        continuum_results['origin'] = _c_origin
        # Hi-res render params (same as original if no upsampling)
        continuum_results['render_grid_shape'] = (_r_nely, _r_nelx, _r_nelz)
        continuum_results['render_pitch'] = _r_pitch
        continuum_results['render_upsample'] = _up

        pct_base = ((c_baseline - c_simp_p1_rescaled) / c_simp_p1_rescaled * 100.0) if c_simp_p1_rescaled > 0 else 0.0
        print(f"\n{'─'*70}")
        print(f"  DIRECT COMPLIANCE (Yin et al. 2020)  —  J = f·u, E = 210 GPa")
        print(f"{'─'*70}")
        print(f"  SIMP binary (hex FEM, p=1):    {_fmt_c(c_simp_p1_rescaled)}")
        print(f"  Baseline skeleton (beam FEM):  {_fmt_c(c_baseline)}  ({pct_base:>+.1f}% vs SIMP)")
        print(f"{'─'*70}")

    except Exception as e:
        print(f"[Warning] Continuum FEM comparison skipped: {e}")
        import traceback
        traceback.print_exc()

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

    opt_plate_thicknesses = None  # Initialized here for layout access

    best_compliance = float('inf')
    best_state = None  # (nodes, edges, radii, tags, ctrl_pts, json_path, curves)
    best_json_path = None

    for loop_idx in range(args.opt_loops):
        loop_num = loop_idx + 1
        suffix = f"_loop{loop_num}"

        # Reduce snap distance in later loops to prevent collapsing optimised nodes
        loop_snap = args.snap if loop_idx == 0 else args.snap * 0.5

        print(f"\n{'='*60}")
        print(f"  ITERATION {loop_num}: Size + Layout Optimisation")
        print(f"{'='*60}")
        if loop_idx > 0 and loop_snap != args.snap:
            print(f"  [Snap] Using reduced snap distance: {loop_snap:.1f} mm (was {args.snap:.1f})")

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
            if args.top3d_npz:
                size_problem.set_load_position_from_npz(args.top3d_npz, args.pitch)
            size_problem.load_tags_from_json(current_json)

            # Pass plates for shell FEM in hybrid mode
            size_plates = plates_data if (args.hybrid and plates_data) else None

            # Curved FEM: only for beam-only mode (no plates) with long beams.
            # Short beams (<3 voxels) produce ill-conditioned IGA stiffness.
            # Hybrid mode always uses straight beams + shell plates (solve_curved_frame
            # does not support plates).
            size_ctrl_pts = None
            if args.curved and _CURVES_AVAILABLE and not size_plates:
                _MIN_CURVED_LEN = 3.0 * args.pitch  # minimum chord for IGA
                size_curves_data = size_input.get('curves', [])
                _raw_cp = _extract_ctrl_pts(size_curves_data, len(edges_size))
                if _raw_cp is not None:
                    size_ctrl_pts = []
                    for ei, cp in enumerate(_raw_cp):
                        u_i, v_i = int(edges_size[ei, 0]), int(edges_size[ei, 1])
                        chord = np.linalg.norm(nodes_size[u_i] - nodes_size[v_i])
                        size_ctrl_pts.append(cp if (cp is not None and chord >= _MIN_CURVED_LEN) else None)
                    if all(cp is None for cp in size_ctrl_pts):
                        size_ctrl_pts = None

            size_result = optimize_size(
                nodes_size, edges_size, radii_size, size_problem,
                E=2.1e5, vol_fraction=1.0, max_iter=args.iters,
                visualize=args.visualize, target_volume_abs=target_volume,
                ctrl_pts=size_ctrl_pts, r_min=args.r_min, r_max=args.r_max,
                plates=size_plates,
                sym_data=sym_data,
            )
            radii_sized, c_size_init, c_size_final, size_hist = size_result[:4]
            opt_plate_thicknesses = size_result[4] if len(size_result) > 4 else None

            # Update plate thicknesses in plates_data
            if opt_plate_thicknesses is not None and plates_data:
                for p_idx, p in enumerate(plates_data):
                    if p_idx < len(opt_plate_thicknesses):
                        new_h = float(opt_plate_thicknesses[p_idx])
                        p["thickness"] = new_h
                        if "mid_surface" in p:
                            p["mid_surface"]["mean_thickness"] = new_h
                print(f"[Shell] Updated plate thicknesses: "
                      f"{[f'{h:.3f}' for h in opt_plate_thicknesses]}")

            # ctrl_pts unchanged after size opt (only radii change)
            size_prev_curves = size_input.get('curves', [])
            curves_sized = _refit_curves(nodes_size, edges_size, radii_sized,
                                         args.curved, ctrl_pts_list=size_ctrl_pts,
                                         prev_curves=size_prev_curves)

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

            # ── Yin-style: record frame compliance for direct comparison ──
            if 'yin_stages' in continuum_results:
                continuum_results['yin_stages'].append({
                    'label': f'Size {loop_num} (beam FEM)',
                    'compliance': c_size_final,
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
            if args.top3d_npz:
                layout_problem.set_load_position_from_npz(args.top3d_npz, args.pitch)
            layout_problem.load_tags_from_json(sized_json)

            # Pass plates for shell FEM in hybrid mode
            layout_plates = plates_data if (args.hybrid and plates_data) else None

            # Curved FEM: only for beam-only mode (same logic as size opt)
            layout_ctrl_pts = None
            if args.curved and _CURVES_AVAILABLE and not layout_plates:
                _MIN_CURVED_LEN = 3.0 * args.pitch
                layout_curves_data = layout_input.get('curves', [])
                _raw_cp = _extract_ctrl_pts(layout_curves_data, len(edges_layout))
                if _raw_cp is not None:
                    layout_ctrl_pts = []
                    for ei, cp in enumerate(_raw_cp):
                        u_i, v_i = int(edges_layout[ei, 0]), int(edges_layout[ei, 1])
                        chord = np.linalg.norm(nodes_layout[u_i] - nodes_layout[v_i])
                        layout_ctrl_pts.append(cp if (cp is not None and chord >= _MIN_CURVED_LEN) else None)
                    if all(cp is None for cp in layout_ctrl_pts):
                        layout_ctrl_pts = None
            layout_plate_h = None
            if layout_plates and opt_plate_thicknesses is not None:
                layout_plate_h = list(opt_plate_thicknesses)

            layout_result = optimize_layout(
                nodes_layout, edges_layout, radii_layout, layout_problem,
                E=2.1e5, move_limit=args.limit, visualize=args.visualize,
                target_volume_abs=target_volume, snap_dist=loop_snap,
                design_bounds=design_bounds, node_tags=node_tags,
                ctrl_pts=layout_ctrl_pts,
                ctrl_move_limit=args.ctrl_limit,
                geo_reg=args.geo_reg,
                plates=layout_plates, plate_thicknesses=layout_plate_h,
                sym_data=sym_data,
                vol_weight=args.vol_weight,
            )

            # Unpack result — last element is always layout_history,
            # 8-tuple when curved (includes ctrl_pts), 7-tuple otherwise
            layout_hist = list(layout_result[-1])  # always last
            if len(layout_result) == 8:
                nodes_opt, edges_opt, radii_opt, tags_opt, c_layout_init, c_layout_final, layout_ctrl_pts_opt, _ = layout_result
            else:
                nodes_opt, edges_opt, radii_opt, tags_opt, c_layout_init, c_layout_final = layout_result[:6]
                layout_ctrl_pts_opt = None

            # --- Early stopping: revert if compliance degraded ---
            if c_layout_final > 0 and c_layout_final < best_compliance:
                best_compliance = c_layout_final
                best_state = (nodes_opt.copy(), edges_opt.copy(), radii_opt.copy(),
                              dict(tags_opt), layout_ctrl_pts_opt,
                              curves_sized)  # curves from the preceding size stage
                # best_json_path updated after we write layout_json below
            elif loop_idx > 0 and best_compliance > 0 and best_state is not None:
                pct = (c_layout_final - best_compliance) / best_compliance * 100
                print(f"\n[Opt] Loop {loop_num} Layout compliance {c_layout_final:.6e} > "
                      f"best {best_compliance:.6e} (+{pct:.0f}%) — reverting and stopping.")
                nodes_opt, edges_opt, radii_opt, tags_opt, layout_ctrl_pts_opt, final_curves = best_state
                current_json = best_json_path
                break

            # Remap plate connection_node_ids if snap_nodes renumbered nodes
            if plates_data and len(nodes_opt) != len(nodes_layout):
                from scipy.spatial import cKDTree as _cKDTree
                _tree = _cKDTree(nodes_opt)
                for _p in plates_data:
                    old_conns = _p.get("connection_node_ids", [])
                    new_conns = []
                    for _cid in old_conns:
                        if _cid < len(nodes_layout):
                            _dist, _nid = _tree.query(nodes_layout[_cid], k=1)
                            if _dist < 1.0:  # snap tol is typically 2-5mm
                                new_conns.append(int(_nid))
                    _p["connection_node_ids"] = new_conns

            # --- Post-Optimization Dead-Weight Pruning ---
            if args.prune_opt_thresh > 0.0 and len(radii_opt) > 0:
                max_r = np.max(radii_opt)
                cutoff = max_r * args.prune_opt_thresh

                keep_edges = []
                keep_radii = []
                keep_ctrl_pts = []
                keep_node_indices = set(tags_opt.keys())  # Always keep tagged nodes

                # 1. Filter edges
                for idx, r in enumerate(radii_opt):
                    if r > cutoff:
                        keep_edges.append(edges_opt[idx])
                        keep_radii.append(r)
                        if layout_ctrl_pts_opt is not None:
                            keep_ctrl_pts.append(layout_ctrl_pts_opt[idx])
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
                    if layout_ctrl_pts_opt is not None:
                        layout_ctrl_pts_opt = keep_ctrl_pts

                    # Remap plate connection_node_ids through old→new mapping
                    if plates_data:
                        for _p in plates_data:
                            old_conns = _p.get("connection_node_ids", [])
                            _p["connection_node_ids"] = [
                                old_to_new[c] for c in old_conns if c in old_to_new
                            ]

                    print(f"[Prune] Graph reduced to {len(nodes_opt)} nodes, {len(edges_opt)} edges.")

            # ── Final symmetry enforcement (after snap + pruning) ──────
            if sym_data is not None:
                from src.optimization.symmetry import (
                    find_symmetric_node_pairs as _fsnp,
                    find_symmetric_edge_pairs as _fsep,
                    enforce_exact_node_symmetry as _eens,
                    average_symmetric_radii as _asr,
                )
                _locked = set(tags_opt.keys()) if tags_opt else set()
                _sym_ni = _fsnp(nodes_opt, sym_data['planes'],
                                tol=sym_data['tol'], locked_nodes=_locked)
                nodes_opt = _eens(nodes_opt, _sym_ni, sym_data['planes'],
                                  locked_nodes=_locked)
                _sym_ei = _fsep(edges_opt, _sym_ni)
                radii_opt = _asr(radii_opt, _sym_ei)

            layout_prev_curves = layout_input.get('curves', [])
            curves_layout = _refit_curves(nodes_opt, edges_opt, radii_opt,
                                          args.curved, ctrl_pts_list=layout_ctrl_pts_opt,
                                          prev_curves=layout_prev_curves)

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

            # Update best-so-far JSON path (for early stopping revert)
            if c_layout_final <= best_compliance:
                best_json_path = layout_json

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

            # ── Yin-style: record frame compliance for direct comparison ──
            if 'yin_stages' in continuum_results:
                continuum_results['yin_stages'].append({
                    'label': f'Layout {loop_num} (beam FEM)',
                    'compliance': c_layout_final,
                })

            # ── Strain energy plot for final layout stage only ────────
            if (_cont_eval is not None and 'grid_shape' in continuum_results
                    and loop_idx == args.opt_loops - 1):
                try:
                    from src.optimization.voxelize_frame import voxelize_beam_frame as _vbf
                    from src.pipelines.baseline_yin.visualization import save_strain_energy_plot
                    gs = continuum_results['grid_shape']
                    rho_final = _vbf(
                        nodes_opt, edges_opt, radii_opt, gs,
                        pitch=continuum_results['pitch'],
                        origin=continuum_results['origin'])
                    _, se_final, u_final, vm_final = _cont_eval.evaluate(
                        rho_final, return_fields=True, return_stress=True,
                        density_mask_thresh=0.5, exclude_void=True)
                    _se_path = os.path.join(args.output_dir, f"{base_name}_se_final")
                    save_strain_energy_plot(se_final, rho_final, continuum_results['pitch'],
                                           continuum_results['origin'], _se_path,
                                           title="Strain Energy — Final Optimised Frame")

                    # Per-beam strain energy from beam FEM (for cylinder colouring)
                    _final_loads, _final_bcs = problem_config.apply(nodes_opt)
                    _final_plates = plates_data if (args.hybrid and plates_data) else None
                    if _final_plates:
                        _fp_h = [p.get("thickness", p.get("mid_surface", {}).get("mean_thickness", 1.0))
                                 for p in _final_plates]
                        _res = solve_frame(
                            nodes_opt, edges_opt, radii_opt, E=E_frame,
                            loads=_final_loads, bcs=_final_bcs,
                            plates=_final_plates, plate_thicknesses=_fp_h)
                        _u_beam, _elems_beam = _res[0], _res[2]
                    else:
                        _u_beam, _, _elems_beam = solve_frame(
                            nodes_opt, edges_opt, radii_opt, E=E_frame,
                            loads=_final_loads, bcs=_final_bcs)
                    _final_beam_se = compute_beam_strain_energy(_u_beam, _elems_beam)

                    # Hi-res interpolation for 3D rendering (no FEM re-solve)
                    _rf_up = continuum_results.get('render_upsample', 1)
                    if _rf_up > 1:
                        from scipy.ndimage import zoom as _zoom
                        _rf_rho = _zoom(rho_final, _rf_up, order=0)
                        _rf_se = _zoom(se_final, _rf_up, order=1)
                        _rf_vm = _zoom(vm_final, _rf_up, order=1) if vm_final is not None else None
                        if u_final is not None and len(u_final) > 0:
                            _nn = (gs[0]+1, gs[1]+1, gs[2]+1)
                            _hr_gs = continuum_results['render_grid_shape']
                            _hr_nn = (_hr_gs[0]+1, _hr_gs[1]+1, _hr_gs[2]+1)
                            _hr_u_comps = []
                            for _comp in range(3):
                                _u_g = u_final[_comp::3].reshape(_nn, order='F')
                                _hr_u_comps.append(_zoom(_u_g, [_hr_nn[i]/_nn[i] for i in range(3)], order=1))
                            _rf_u = np.zeros(_hr_nn[0]*_hr_nn[1]*_hr_nn[2]*3)
                            for _comp in range(3):
                                _rf_u[_comp::3] = _hr_u_comps[_comp].flatten(order='F')
                        else:
                            _rf_u = u_final
                        _rf_nely, _rf_nelx, _rf_nelz = continuum_results['render_grid_shape']
                        _rf_pitch = continuum_results['render_pitch']
                    else:
                        _rf_se, _rf_rho, _rf_u = se_final, rho_final, u_final
                        _rf_vm = vm_final
                        _rf_nely, _rf_nelx, _rf_nelz = gs
                        _rf_pitch = continuum_results['pitch']

                    # 3D FEA visualization (PyVista)
                    try:
                        from src.export.vtk_export import render_pipeline_stage
                        render_pipeline_stage(
                            "Final Optimised", _rf_se, _rf_rho, _rf_u,
                            _rf_nelx, _rf_nely, _rf_nelz,
                            _rf_pitch, continuum_results['origin'],
                            args.output_dir, base_name,
                            export_vtr=args.export_vtk, render_3d=args.render_3d,
                            frame_data={
                                'nodes': nodes_opt, 'edges': edges_opt, 'radii': radii_opt,
                                'node_tags': {str(k): v for k, v in tags_opt.items()},
                                'beam_se': _final_beam_se,
                            },
                            stress_field=_rf_vm,
                        )
                    except Exception as _e2:
                        print(f"  [Warning] 3D final visualization skipped: {_e2}")
                except Exception as _e:
                    print(f"  [Warning] Final strain energy plot skipped: {_e}")

            current_json = layout_json

        except Exception as e:
            print(f"[ERROR] Layout Optimisation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Use current_json which tracks the last successful output
    # (may differ from expected loop count if early stopping triggered)
    stage2_out = current_json
    stage3_out = os.path.join(args.output_dir, f"{base_name}_3_sized_loop{args.opt_loops}.json")

    # ── Set optimised_frame from the last Yin stage ──
    if (continuum_results.get('yin_stages')
            and len(continuum_results['yin_stages']) > 0):
        continuum_results['optimised_frame'] = continuum_results['yin_stages'][-1]['compliance']

    # ========================================
    # Print Comparison Table
    # ========================================
    if args.opt_loops > 1 or len(metrics_list) > 2:
        _print_comparison_table(metrics_list, c_baseline, target_volume,
                                top3d_hist=top3d_hist,
                                continuum_results=continuum_results)

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
        _figsize = None
        if args.fig_size:
            try:
                w, h = args.fig_size.lower().split('x')
                _figsize = (float(w), float(h))
            except ValueError:
                print(f"[Warning] Invalid --fig_size '{args.fig_size}', using default. Expected 'WxH' e.g. '3.5x2.6'")

        if convergence_stages and np.isfinite(c_baseline):
            plot_combined_convergence(
                top3d_hist, convergence_stages, c_baseline,
                f"{fig_base}_fig_combined",
                mesh_size=(args.nelx, args.nely, args.nelz),
                volfrac=args.volfrac,
                figsize=_figsize,
            )

        # Text + JSON pipeline report
        report_data = _build_report_data(
            args, baseline_data, metrics_list, convergence_stages,
            top3d_hist, target_volume, c_baseline,
            nodes_opt=nodes_opt,
            edges_opt=edges_opt,
            radii_opt=radii_opt,
            baseline_nodes=baseline_nodes,
            continuum_results=continuum_results,
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
