#!/usr/bin/env python3
"""Evaluate two-signal zone classification accuracy with ablation study.

Runs the classifier in three modes (both signals, Signal A only, Signal B only)
on each test case and prints:
  1. Extended classification table (Signal A/B breakdown)
  2. Ablation comparison table (single-signal vs combined)
  3. Volume-fraction sensitivity sweep
  4. LaTeX-ready versions of tables 1-3

Usage:
    python scripts/evaluate_classification.py
"""

import sys
import time
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scipy.ndimage import distance_transform_edt
from src.pipelines.baseline_yin.thinning import thin_grid_yin
from src.pipelines.baseline_yin.graph import classify_skeleton_post_thinning

# ── Primary test cases (diverse topologies) ─────────────────────
TEST_CASES = [
    # ─── Mixed (plate + beam) structures ───
    {
        "name": "Clear Beam-Plate",
        "short": "CBP",
        "npz": "output/hybrid_v2/Clear_Beam_Plate_Test_top3d.npz",
        "expected": "mixed",
    },
    {
        "name": "Wall Bracket",
        "short": "WB",
        "npz": "output/hybrid_v2/Wall_Bracket_top3d.npz",
        "expected": "mixed",
    },
    {
        "name": "Elevated Slab",
        "short": "ES",
        "npz": "output/hybrid_v2/elevated_slab_top3d.npz",
        "expected": "mixed",
    },
    {
        "name": "Curved Shell",
        "short": "CS",
        "npz": "output/hybrid_v2/curved_shell_top3d.npz",
        "expected": "mixed",
    },
    {
        "name": "Pipe Bracket",
        "short": "PB",
        "npz": "output/hybrid_v2/pipe_bracket_test_top3d.npz",
        "expected": "mixed",
    },
    {
        "name": "Bridge",
        "short": "Br",
        "npz": "output/hybrid_v2/bridge_top3d.npz",
        "expected": "beam",
        "min_plate_size": 30,
    },
    {
        "name": "Roof Slab 60x60",
        "short": "RS",
        "npz": "output/hybrid_v2/roof_slab_60x60_top3d.npz",
        "expected": "mixed",
    },
    # ─── Pure beam structures ───
    {
        "name": "Cantilever",
        "short": "Cant",
        "npz": "output/hybrid_v2/matlab_replicated_top3d.npz",
        "expected": "beam",
        "min_plate_size": 30,
    },
    {
        "name": "Thick Beam",
        "short": "TB",
        "npz": "output/hybrid_v2/full_control_beam_top3d.npz",
        "expected": "beam",
    },
]

# ── Volume-fraction sweep (same geometry, varying VF) ───────────
VF_SWEEP_CASES = [
    {
        "name": "CBP VF=4%",
        "short": "VF04",
        "npz": "output/hybrid_v2/Clear_Beam_Plate_Test_top3d_0.04_VF.npz",
        "vf": 0.04,
    },
    {
        "name": "CBP VF=5%",
        "short": "VF05",
        "npz": "output/hybrid_v2/Clear_Beam_Plate_Test_top3d_0.05_VF.npz",
        "vf": 0.05,
    },
    {
        "name": "CBP VF=8%",
        "short": "VF08",
        "npz": "output/hybrid_v2/Clear_Beam_Plate_Test_top3d_0.08_VF.npz",
        "vf": 0.08,
    },
    {
        "name": "CBP VF=12%",
        "short": "VF12",
        "npz": "output/hybrid_v2/Clear_Beam_Plate_Test_top3d.npz",
        "vf": 0.12,
    },
    {
        "name": "CBP VF=20%",
        "short": "VF20",
        "npz": "output/hybrid_v2/Clear_Beam_Plate_Test_top3d_HIGHER_VF.npz",
        "vf": 0.196,
    },
]


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    rho = d["rho"]
    bc_tags = d["bc_tags"].astype(np.int32) if "bc_tags" in d else np.zeros_like(rho, dtype=np.int32)
    return rho, bc_tags


def run_thinning(rho, bc_tags, vol_thresh=0.3):
    solid = (rho > vol_thresh).astype(np.uint8)
    edt = distance_transform_edt(solid)
    skel3 = thin_grid_yin(solid.copy(), tags=bc_tags, max_iters=50, mode=3, edt=edt)
    skel0 = thin_grid_yin(solid.copy(), tags=bc_tags, max_iters=50, mode=0, edt=edt)
    return solid, skel3, skel0


def classify(skel3, skel0, solid, signal_mode='both', min_plate_size=None):
    nelz = skel3.shape[2]
    if min_plate_size is None:
        min_plate_size = 10 if nelz <= 6 else 3
    zone_mask, plate_labels, stats = classify_skeleton_post_thinning(
        skel3, min_plate_size=min_plate_size, flatness_ratio=3.0,
        skeleton_curve=skel0, solid=solid, signal_mode=signal_mode,
    )
    return stats


def run_test_case(tc, verbose=True):
    """Run thinning + 3-mode classification on a single test case. Returns row dict or None."""
    npz_path = REPO / tc["npz"]
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  {tc['name']}  ({npz_path.name})")

    if not npz_path.exists():
        if verbose:
            print(f"  [SKIP] NPZ not found")
        return None

    rho, bc_tags = load_npz(npz_path)
    domain = f"{rho.shape[0]}x{rho.shape[1]}x{rho.shape[2]}"
    n_solid = int(np.sum(rho > 0.3))

    t0 = time.perf_counter()
    solid, skel3, skel0 = run_thinning(rho, bc_tags)
    t_thin = time.perf_counter() - t0

    mps = tc.get("min_plate_size")

    t1 = time.perf_counter()
    stats_both = classify(skel3, skel0, solid, 'both', mps)
    t_cls = time.perf_counter() - t1

    stats_a = classify(skel3, skel0, solid, 'a_only', mps)
    stats_b = classify(skel3, skel0, solid, 'b_only', mps)

    row = {
        "name": tc["name"],
        "short": tc["short"],
        "expected": tc.get("expected", "mixed"),
        "domain": domain,
        "n_solid": n_solid,
        "skel": stats_both["n_skeleton_voxels"],
        "mode0": int(np.sum(skel0 > 0)),
        "sig_a": stats_both["n_signal_a"],
        "sig_b": stats_both["n_signal_b"],
        "overlap": stats_both["n_overlap"],
        "combined": stats_both["n_combined"],
        "growth": stats_both["n_growth"],
        "plates_both": stats_both["n_plate_voxels"],
        "beams_both": stats_both["n_beam_voxels"],
        "regions_both": stats_both["n_plate_regions"],
        "plates_a": stats_a["n_plate_voxels"],
        "plates_b": stats_b["n_plate_voxels"],
        "t_thin": t_thin,
        "t_cls": t_cls,
    }

    if verbose:
        print(f"  Domain: {domain},  Solid: {n_solid}")
        print(f"  Skeleton: {row['skel']} (mode=3), {row['mode0']} (mode=0)")
        print(f"  Signal A: {row['sig_a']},  Signal B: {row['sig_b']},  "
              f"Overlap: {row['overlap']}")
        print(f"  Combined: {row['combined']} -> Plates: {row['plates_both']} "
              f"({row['regions_both']} regions) + Growth: {row['growth']}")
        print(f"  Ablation:  A-only -> {row['plates_a']} plates,  "
              f"B-only -> {row['plates_b']} plates")
        print(f"  Time:  thinning={t_thin:.2f}s,  classification={t_cls:.3f}s")

    return row


def print_table1(results):
    """Table 1: Extended signal breakdown."""
    print(f"\n\n{'=' * 90}")
    print("  TABLE 1: Signal Breakdown (Primary Test Cases)")
    print("=" * 90)
    hdr = (f"{'Test Case':<22s} {'Domain':<14s} {'Skel':>5s} {'SigA':>5s} "
           f"{'SigB':>5s} {'A^B':>4s} {'Comb':>5s} {'Filt':>5s} {'Grow':>5s} "
           f"{'Plate':>6s} {'Beam':>6s} {'Reg':>4s} {'t_cls':>6s}")
    print(hdr)
    print("─" * len(hdr))
    for r in results:
        filtered = r["plates_both"] - r["growth"]
        print(f"{r['name']:<22s} {r['domain']:<14s} {r['skel']:5d} "
              f"{r['sig_a']:5d} {r['sig_b']:5d} "
              f"{r['overlap']:4d} {r['combined']:5d} {filtered:5d} {r['growth']:5d} "
              f"{r['plates_both']:6d} {r['beams_both']:6d} {r['regions_both']:4d} "
              f"{r['t_cls']:5.3f}s")


def print_table2(results):
    """Table 2: Single-signal ablation."""
    print(f"\n\n{'=' * 90}")
    print("  TABLE 2: Single-Signal Ablation")
    print("=" * 90)
    hdr2 = (f"{'Test Case':<22s} {'A+B':>6s} {'A only':>7s} {'B only':>7s} "
            f"{'A miss%':>8s} {'B miss%':>8s} {'Expected':>9s} {'Correct':>8s}")
    print(hdr2)
    print("─" * len(hdr2))
    for r in results:
        ref = max(r["plates_both"], 1)
        a_miss = 100.0 * (1.0 - r["plates_a"] / ref) if r["plates_both"] > 0 else 0.0
        b_miss = 100.0 * (1.0 - r["plates_b"] / ref) if r["plates_both"] > 0 else 0.0
        correct = (
            (r["expected"] == "mixed" and r["plates_both"] > 0 and r["beams_both"] > 0) or
            (r["expected"] == "beam" and r["plates_both"] == 0)
        )
        mark = "Y" if correct else "N"
        print(f"{r['name']:<22s} {r['plates_both']:6d} {r['plates_a']:7d} "
              f"{r['plates_b']:7d} {a_miss:7.1f}% {b_miss:7.1f}% "
              f"{r['expected']:>9s} {mark:>8s}")


def print_table3(vf_results):
    """Table 3: Volume-fraction sensitivity sweep."""
    print(f"\n\n{'=' * 90}")
    print("  TABLE 3: Volume-Fraction Sensitivity (Clear Beam-Plate geometry)")
    print("=" * 90)
    hdr3 = (f"{'VF':>6s} {'Solid':>6s} {'Skel':>5s} {'SigA':>5s} "
            f"{'SigB':>5s} {'Plate':>6s} {'Beam':>6s} {'Reg':>4s} "
            f"{'Plate%':>7s} {'t_thin':>7s} {'t_cls':>6s}")
    print(hdr3)
    print("─" * len(hdr3))
    for r in vf_results:
        plate_pct = 100.0 * r["plates_both"] / max(r["skel"], 1)
        print(f"{r['vf']:5.1f}% {r['n_solid']:6d} {r['skel']:5d} "
              f"{r['sig_a']:5d} {r['sig_b']:5d} "
              f"{r['plates_both']:6d} {r['beams_both']:6d} {r['regions_both']:4d} "
              f"{plate_pct:6.1f}% {r['t_thin']:6.2f}s {r['t_cls']:5.3f}s")


def print_latex_table1(results):
    """LaTeX: Extended classification table."""
    print(f"\n\n{'=' * 72}")
    print("  LATEX: Extended Classification Table")
    print("=" * 72)
    print(r"""\begin{table*}[t]
\centering
\caption{Two-signal zone classification results across nine test structures.
  $|\mathcal{S}_3|$: mode-3 skeleton size.
  $|A|$: Signal~A candidates (set difference).
  $|B|$: Signal~B candidates (plane-octant count).
  $|A{\cap}B|$: overlap.
  Growth: edge voxels added in boundary expansion pass.
  $t$: classification time (excluding thinning).}
\label{tab:zone_class_extended}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\footnotesize
\begin{tabular}{@{}l l r r r r r r r r r@{}}
\toprule
\textbf{Test case} & \textbf{Domain}
  & {$|\mathcal{S}_3|$}
  & {$|A|$} & {$|B|$} & {$|A{\cap}B|$}
  & {Grow.}
  & {Plate} & {Beam} & {Reg.}
  & {$t$ (s)} \\
\midrule""")
    for r in results:
        check = r"\cmark" if (
            (r["expected"] == "mixed" and r["plates_both"] > 0 and r["beams_both"] > 0) or
            (r["expected"] == "beam" and r["plates_both"] == 0)
        ) else r"\xmark"
        print(f"{r['name']} {check}"
              f"  & {r['domain']}"
              f"  & {r['skel']}"
              f"  & {r['sig_a']}"
              f"  & {r['sig_b']}"
              f"  & {r['overlap']}"
              f"  & {r['growth']}"
              f"  & {r['plates_both']}"
              f"  & {r['beams_both']}"
              f"  & {r['regions_both']}"
              f"  & {r['t_cls']:.3f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table*}""")


def print_latex_table2(results):
    """LaTeX: Ablation table."""
    print(f"\n\n{'=' * 72}")
    print("  LATEX: Ablation Table")
    print("=" * 72)
    print(r"""\begin{table}[t]
\centering
\caption{Single-signal ablation study.  Plate voxel counts under
  Signal~A only, Signal~B only, and the combined classifier ($A{\cup}B$).
  Miss\,\%: fraction of combined-mode plate voxels lost when using a single
  signal.}
\label{tab:zone_ablation}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{4pt}
\footnotesize
\begin{tabular}{@{}l r r r r r@{}}
\toprule
\textbf{Test case}
  & {$A{\cup}B$}
  & {$A$ only} & {miss}
  & {$B$ only} & {miss} \\
\midrule""")
    for r in results:
        ref = max(r["plates_both"], 1)
        a_miss = 100.0 * (1.0 - r["plates_a"] / ref) if r["plates_both"] > 0 else 0.0
        b_miss = 100.0 * (1.0 - r["plates_b"] / ref) if r["plates_both"] > 0 else 0.0
        a_miss_s = f"{a_miss:.0f}\\%" if r["plates_both"] > 0 else "---"
        b_miss_s = f"{b_miss:.0f}\\%" if r["plates_both"] > 0 else "---"
        print(f"{r['name']}"
              f"  & {r['plates_both']}"
              f"  & {r['plates_a']} & {a_miss_s}"
              f"  & {r['plates_b']} & {b_miss_s} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def print_latex_table3(vf_results):
    """LaTeX: VF sensitivity table."""
    print(f"\n\n{'=' * 72}")
    print("  LATEX: Volume-Fraction Sensitivity Table")
    print("=" * 72)
    print(r"""\begin{table}[t]
\centering
\caption{Volume-fraction sensitivity for the Clear Beam-Plate geometry
  (40$\times$40$\times$20 domain).  As VF increases, solid voxels grow
  and plate regions become thicker, shifting the plate/beam ratio.}
\label{tab:zone_vf_sweep}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{4pt}
\footnotesize
\begin{tabular}{@{}r r r r r r r r r@{}}
\toprule
{VF (\%)} & {Solid} & {$|\mathcal{S}_3|$}
  & {$|A|$} & {$|B|$}
  & {Plate} & {Beam} & {Reg.}
  & {Plate\,\%} \\
\midrule""")
    for r in vf_results:
        plate_pct = 100.0 * r["plates_both"] / max(r["skel"], 1)
        print(f"{r['vf']:.1f}"
              f"  & {r['n_solid']}"
              f"  & {r['skel']}"
              f"  & {r['sig_a']}"
              f"  & {r['sig_b']}"
              f"  & {r['plates_both']}"
              f"  & {r['beams_both']}"
              f"  & {r['regions_both']}"
              f"  & {plate_pct:.1f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main():
    print("=" * 90)
    print("  Two-Signal Classification Evaluation  (v2 — expanded)")
    print("=" * 90)

    # ── Part 1: Primary test cases ──────────────────────────────
    results = []
    for tc in TEST_CASES:
        row = run_test_case(tc)
        if row is not None:
            results.append(row)

    if not results:
        print("\nNo test cases found. Ensure NPZ files exist.")
        return

    print_table1(results)
    print_table2(results)

    # ── Part 2: VF sensitivity sweep ────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("  VOLUME-FRACTION SWEEP")
    print("=" * 90)

    vf_results = []
    for tc in VF_SWEEP_CASES:
        row = run_test_case(tc, verbose=True)
        if row is not None:
            row["vf"] = tc["vf"] * 100  # store as percentage
            vf_results.append(row)

    if vf_results:
        print_table3(vf_results)

    # ── Part 3: Summary statistics ──────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("  SUMMARY STATISTICS")
    print("=" * 90)

    n_total = len(results)
    n_correct = sum(1 for r in results if (
        (r["expected"] == "mixed" and r["plates_both"] > 0 and r["beams_both"] > 0) or
        (r["expected"] == "beam" and r["plates_both"] == 0)
    ))
    print(f"  Classification accuracy: {n_correct}/{n_total} "
          f"({100*n_correct/n_total:.0f}%)")

    mixed_results = [r for r in results if r["expected"] == "mixed"]
    if mixed_results:
        avg_overlap = np.mean([r["overlap"] / max(r["combined"], 1) * 100
                               for r in mixed_results if r["combined"] > 0])
        print(f"  Mean signal overlap (mixed cases): {avg_overlap:.1f}%")

        # Signal dominance
        a_dominant = sum(1 for r in mixed_results
                         if r["sig_a"] > r["sig_b"] and r["plates_both"] > 0)
        b_dominant = sum(1 for r in mixed_results
                         if r["sig_b"] > r["sig_a"] and r["plates_both"] > 0)
        codominant = sum(1 for r in mixed_results
                         if r["sig_a"] == r["sig_b"] and r["plates_both"] > 0)
        print(f"  Signal dominance (mixed): A-dominant={a_dominant}, "
              f"B-dominant={b_dominant}, co-dominant={codominant}")

    t_total_thin = sum(r["t_thin"] for r in results)
    t_total_cls = sum(r["t_cls"] for r in results)
    print(f"  Total thinning time: {t_total_thin:.1f}s  |  "
          f"Total classification time: {t_total_cls:.2f}s")
    print(f"  Mean classification time: {t_total_cls/len(results):.3f}s/case")

    # ── Part 4: LaTeX output ────────────────────────────────────
    print_latex_table1(results)
    print_latex_table2(results)
    if vf_results:
        print_latex_table3(vf_results)

    print(f"\n{'=' * 90}")
    print("  Done.")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
