#!/usr/bin/env python3
"""
Batch Comparison Study for Dissertation
========================================
Runs beam_mode x hybrid (6 configs) across multiple test cases, then
aggregates results into master tables for the dissertation comparison section.

Usage:
    # Full run (~60 min)
    python run_batch_comparison.py --opt_loops 2

    # Subset of cases
    python run_batch_comparison.py --cases matlab_replicated,elevated_slab

    # Dry run — print commands without executing
    python run_batch_comparison.py --dry_run

    # Re-aggregate existing results without re-running
    python run_batch_comparison.py --aggregate_only output/batch_comparison/20260410_1500
"""

import argparse
import datetime
import glob
import json
import math
import os
import subprocess
import sys
import time


# ── Test case registry ─────────────────────────────────────────────────────

TEST_CASES = {
    "matlab_replicated": {
        "npz": "output/hybrid_v2/matlab_replicated_New_E_top3d.npz",
        "category": "beam",
        "description": "Yin et al. cantilever benchmark (150x40x4)",
        "args": [
            "--snap", "2.0", "--prune_len", "2.0",
            "--collapse_thresh", "2.0", "--rdp", "1.0",
            "--load_fy", "-100",
            "--no_render_3d",
        ],
    },
    "simply_supported": {
        "npz": "output/simply_supported_top3d.npz",
        "category": "beam",
        "description": "Simply-supported beam (2 supports)",
        "args": [
            "--snap", "1.5", "--prune_len", "1.5",
            "--collapse_thresh", "1.5", "--rdp", "1.0",
            "--load_fy", "-1000",
            "--no_render_3d",
        ],
    },
    "elevated_slab": {
        "npz": "output/hybrid_v2/elevated_slab_v3_top3d.npz",
        "category": "plate",
        "description": "Elevated slab on columns (40x40x30)",
        "args": [
            "--min_plate_size", "8", "--flatness_ratio", "5",
            "--min_avg_neighbors", "3",
            "--snap", "2.0", "--prune_len", "1.5",
            "--collapse_thresh", "2.0", "--rdp", "1.5",
            "--no_render_3d",
        ],
    },
    "curved_shell": {
        "npz": "output/hybrid_v2/curved_shell_top3d.npz",
        "category": "plate",
        "description": "Curved shell (10x40x20)",
        "args": [
            "--min_plate_size", "6", "--flatness_ratio", "3",
            "--min_avg_neighbors", "3",
            "--snap", "1.5", "--prune_len", "1.0",
            "--collapse_thresh", "1.5", "--rdp", "1.0",
            "--no_render_3d",
        ],
    },
    "clear_beam_plate": {
        "npz": "output/hybrid_v2/Clear_Beam_Plate_Test_top3d_0.05_VF.npz",
        "category": "mixed",
        "description": "Tagged beam+plate regions (40x40x20)",
        "args": [
            "--min_plate_size", "8", "--flatness_ratio", "7",
            "--min_avg_neighbors", "3",
            "--snap", "1.5", "--prune_len", "1.5",
            "--collapse_thresh", "2.0", "--rdp", "2.0",
            "--symmetry", "xz,yz", "--sym_weight", "0.1",
            "--no_render_3d",
        ],
    },
}


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser():
    case_names = ", ".join(TEST_CASES.keys())
    parser = argparse.ArgumentParser(
        description="Batch comparison: beam_mode x hybrid across multiple test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available test cases: {case_names}",
    )
    parser.add_argument(
        "--cases", default=None,
        help="Comma-separated case names to run (default: all)")
    parser.add_argument(
        "--opt_loops", type=int, default=2,
        help="Number of optimisation loops per run (default: 2)")
    parser.add_argument(
        "--base_dir", default="output/batch_comparison",
        help="Root output directory (default: output/batch_comparison)")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing")
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Per-run timeout in seconds (default: 3600)")
    parser.add_argument(
        "--aggregate_only", default=None, metavar="DIR",
        help="Skip runs; aggregate results from an existing batch directory")
    return parser


# ── Helpers ────────────────────────────────────────────────────────────────

def _fc(v):
    """Format compliance (scientific notation)."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{v:.4e}"


def _fp(v):
    """Format percentage."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{v:+.1f}%"


def _ff(v, d=2):
    """Format float."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{v:.{d}f}"


def _fi(v):
    """Format integer."""
    return str(v) if v is not None else "--"


def _json_default(obj):
    """Handle non-serializable types for json.dump."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


# ── Run execution ──────────────────────────────────────────────────────────

def resolve_cases(args):
    """Return ordered dict of {name: config} for cases to run."""
    if args.cases:
        names = [n.strip() for n in args.cases.split(",")]
        missing = [n for n in names if n not in TEST_CASES]
        if missing:
            print(f"[FATAL] Unknown case(s): {', '.join(missing)}")
            print(f"  Available: {', '.join(TEST_CASES.keys())}")
            sys.exit(1)
        return {n: TEST_CASES[n] for n in names}
    return dict(TEST_CASES)


def run_all_cases(args):
    """Run run_comparison.py for each test case. Return batch_dir."""
    cases = resolve_cases(args)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    batch_dir = os.path.join(args.base_dir, timestamp)
    os.makedirs(batch_dir, exist_ok=True)

    total = len(cases)
    total_t0 = time.time()

    print(f"\n{'=' * 100}")
    print(f"  BATCH COMPARISON STUDY — {total} test cases x 6 configs = {total * 6} runs")
    print(f"  Output: {batch_dir}")
    print(f"  opt_loops: {args.opt_loops}  |  timeout: {args.timeout}s")
    print(f"{'=' * 100}")

    for i, (case_name, case_config) in enumerate(cases.items()):
        cat = case_config["category"]
        desc = case_config["description"]
        npz = case_config["npz"]
        case_dir = os.path.join(batch_dir, case_name)

        print(f"\n{'─' * 80}")
        print(f"  [{i + 1}/{total}] {case_name} ({cat}) — {desc}")
        print(f"  NPZ: {npz}")
        print(f"{'─' * 80}")

        if not os.path.exists(npz):
            print(f"  SKIPPED — NPZ not found: {npz}")
            continue

        cmd = [
            sys.executable, "run_comparison.py",
            "--npz", npz,
            "--opt_loops", str(args.opt_loops),
            "--base_dir", case_dir,
            "--timeout", str(args.timeout),
        ]
        if args.dry_run:
            cmd.append("--dry_run")
        cmd.extend(case_config["args"])

        print(f"  cmd: {' '.join(cmd)}")

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, timeout=args.timeout * 7)
            elapsed = time.time() - t0
            status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            print(f"\n  {case_name}: {status} in {elapsed / 60:.1f}m")
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"\n  {case_name}: TIMEOUT after {elapsed / 60:.1f}m")
        except Exception as e:
            print(f"\n  {case_name}: ERROR — {e}")

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 100}")
    print(f"  All cases complete in {total_elapsed / 60:.1f}m")
    print(f"{'=' * 100}")

    return batch_dir


# ── Aggregation ────────────────────────────────────────────────────────────

def find_summary_json(case_dir):
    """Find the latest summary.json inside a case directory."""
    # run_comparison.py creates case_dir/<timestamp>/summary.json
    matches = glob.glob(os.path.join(case_dir, "*", "summary.json"))
    if not matches:
        return None
    # Pick the latest by modification time
    return max(matches, key=os.path.getmtime)


def aggregate(batch_dir, cases):
    """Load all per-case summary.json files. Return master dict."""
    master = {}  # {case_name: {"config": ..., "summary": parsed_json}}

    for case_name, case_config in cases.items():
        case_dir = os.path.join(batch_dir, case_name)
        summary_path = find_summary_json(case_dir)
        if not summary_path:
            print(f"  [WARN] No summary.json for {case_name}")
            master[case_name] = {"config": case_config, "data": None}
            continue

        with open(summary_path) as f:
            data = json.load(f)
        master[case_name] = {"config": case_config, "data": data}
        n_ok = sum(1 for r in data.get("runs", {}).values()
                   if r.get("report") is not None)
        print(f"  Loaded {case_name}: {n_ok} successful runs")

    return master


# ── Master Table 1: Final Compliance Matrix ────────────────────────────────

CONFIG_LABELS = [
    ("straight", False, "str_beam"),
    ("straight", True, "str_hybrid"),
    ("curved", False, "cur_beam"),
    ("curved", True, "cur_hybrid"),
    ("mixed", False, "mix_beam"),
    ("mixed", True, "mix_hybrid"),
]

# Labels used in run_comparison.py's summary.json
RUN_LABELS = [
    "straight_beamonly", "straight_hybrid",
    "curved_beamonly", "curved_hybrid",
    "mixed_beamonly", "mixed_hybrid",
]


def _get_metric(master, case_name, run_label, *keys):
    """Extract a nested metric from a run's report."""
    data = master.get(case_name, {}).get("data")
    if not data:
        return None
    run = data.get("runs", {}).get(run_label)
    if not run:
        return None
    report = run.get("report")
    if not report:
        return None
    obj = report
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return None
    return obj


def format_compliance_matrix(master):
    """Table 1: C_final for each (case, config). Best per row bolded with *."""
    lines = []
    W = 120
    lines.append("=" * W)
    lines.append(f"{'TABLE 1: FINAL COMPLIANCE MATRIX':^{W}}")
    lines.append(f"{'(C_final — lower is better. * = best in row)':^{W}}")
    lines.append("=" * W)

    # Header
    hdr1 = f"{'':30} | {'Straight':^23} | {'Curved':^23} | {'Mixed':^23}"
    hdr2 = (f"{'Test Case':<20} {'Cat':>3} {'VF%':>5} |"
            f" {'Beam':>10} {'Hybrid':>11} |"
            f" {'Beam':>10} {'Hybrid':>11} |"
            f" {'Beam':>10} {'Hybrid':>11}")
    lines.append(hdr1)
    lines.append(hdr2)
    lines.append("-" * W)

    for case_name, case_info in master.items():
        config = case_info["config"]
        data = case_info.get("data")
        cat = config["category"][0].upper()  # B, P, M

        # Get volume fraction from first successful run's top3d data
        vf = "--"
        if data:
            for run in data.get("runs", {}).values():
                r = run.get("report")
                if r:
                    vf_val = r.get("top3d", {}).get("volfrac")
                    if vf_val is not None:
                        vf = f"{vf_val * 100:.0f}" if vf_val <= 1 else f"{vf_val:.0f}"
                    break

        # Collect C_final for all 6 configs
        c_vals = {}
        for rl in RUN_LABELS:
            c_vals[rl] = _get_metric(master, case_name, rl, "overall", "final_compliance")

        # Find best (min) non-None value
        valid_vals = {k: v for k, v in c_vals.items() if v is not None}
        best_label = min(valid_vals, key=valid_vals.get) if valid_vals else None

        def fmt_cell(rl):
            v = c_vals.get(rl)
            if v is None:
                return f"{'--':>10}"
            s = f"{v:.4e}"
            if rl == best_label:
                s = f"*{s}"
            return f"{s:>10}"

        row = (f"{case_name:<20} {cat:>3} {vf:>5} |"
               f" {fmt_cell('straight_beamonly')} {fmt_cell('straight_hybrid'):>11} |"
               f" {fmt_cell('curved_beamonly')} {fmt_cell('curved_hybrid'):>11} |"
               f" {fmt_cell('mixed_beamonly')} {fmt_cell('mixed_hybrid'):>11}")
        lines.append(row)

    lines.append("=" * W)
    return lines


# ── Master Table 2: Hybrid Advantage ───────────────────────────────────────

def format_hybrid_advantage(master):
    """Table 2: Delta% = (C_hybrid - C_beamonly) / C_beamonly. Negative = hybrid helps."""
    lines = []
    W = 100
    lines.append("=" * W)
    lines.append(f"{'TABLE 2: HYBRID ADVANTAGE':^{W}}")
    lines.append(f"{'Delta = (C_hybrid - C_beamonly) / C_beamonly. Negative = hybrid is better.':^{W}}")
    lines.append("=" * W)

    hdr = (f"{'Test Case':<20} {'Cat':>4} |"
           f" {'Straight':>12} | {'Curved':>12} | {'Mixed':>12} |"
           f" {'Verdict':<20}")
    lines.append(hdr)
    lines.append("-" * W)

    beam_modes = ["straight", "curved", "mixed"]

    for case_name, case_info in master.items():
        config = case_info["config"]
        cat = config["category"][0].upper()

        deltas = {}
        for bm in beam_modes:
            c_beam = _get_metric(master, case_name, f"{bm}_beamonly", "overall", "final_compliance")
            c_hyb = _get_metric(master, case_name, f"{bm}_hybrid", "overall", "final_compliance")
            if c_beam and c_hyb and c_beam > 0:
                deltas[bm] = (c_hyb - c_beam) / c_beam * 100
            else:
                deltas[bm] = None

        # Verdict: if majority of deltas are negative, "Hybrid helps"
        neg_count = sum(1 for v in deltas.values() if v is not None and v < 0)
        pos_count = sum(1 for v in deltas.values() if v is not None and v >= 0)
        total_valid = neg_count + pos_count
        if total_valid == 0:
            verdict = "No data"
        elif neg_count > pos_count:
            verdict = "Hybrid helps"
        elif neg_count == pos_count:
            verdict = "Mixed"
        else:
            verdict = "Beam-only better"

        row = (f"{case_name:<20} {cat:>4} |"
               f" {_fp(deltas.get('straight')):>12} |"
               f" {_fp(deltas.get('curved')):>12} |"
               f" {_fp(deltas.get('mixed')):>12} |"
               f" {verdict:<20}")
        lines.append(row)

    lines.append("=" * W)
    return lines


# ── Master Table 3: Structural Element Counts ──────────────────────────────

def format_element_counts(master):
    """Table 3: N/E/P per config."""
    lines = []
    W = 120
    lines.append("=" * W)
    lines.append(f"{'TABLE 3: STRUCTURAL ELEMENT COUNTS (Nodes / Edges / Plates)':^{W}}")
    lines.append("=" * W)

    hdr = (f"{'Test Case':<20} |"
           f" {'str_beam':>12} | {'str_hybrid':>12} |"
           f" {'cur_beam':>12} | {'cur_hybrid':>12} |"
           f" {'mix_beam':>12} | {'mix_hybrid':>12}")
    lines.append(hdr)
    lines.append("-" * W)

    for case_name in master:
        cells = []
        for rl in RUN_LABELS:
            n = _get_metric(master, case_name, rl, "reconstruction", "nodes")
            e = _get_metric(master, case_name, rl, "reconstruction", "edges")
            p = _get_metric(master, case_name, rl, "reconstruction", "plates")
            if n is not None:
                cells.append(f"{_fi(n)}/{_fi(e)}/{_fi(p)}")
            else:
                cells.append("--")

        row = f"{case_name:<20} |"
        for c in cells:
            row += f" {c:>12} |"
        # Strip trailing |
        row = row.rstrip(" |")
        lines.append(row)

    lines.append("=" * W)
    return lines


# ── LaTeX table ────────────────────────────────────────────────────────────

def format_latex_table(master):
    """Generate LaTeX tabular for Table 1 (compliance matrix)."""
    lines = []
    lines.append("% Auto-generated by run_batch_comparison.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Final compliance $C_{\\text{final}}$ across beam representation "
                 "and hybrid configurations. Bold indicates best per test case. Lower is better.}")
    lines.append("\\label{tab:beam-hybrid-comparison}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}lcc|cc|cc|cc@{}}")
    lines.append("\\toprule")
    lines.append("& & & \\multicolumn{2}{c|}{\\textbf{Straight}}"
                 " & \\multicolumn{2}{c|}{\\textbf{Curved}}"
                 " & \\multicolumn{2}{c}{\\textbf{Mixed}} \\\\")
    lines.append("\\textbf{Test Case} & Cat. & VF\\%"
                 " & Beam & Hybrid & Beam & Hybrid & Beam & Hybrid \\\\")
    lines.append("\\midrule")

    for case_name, case_info in master.items():
        config = case_info["config"]
        cat = config["category"][0].upper()

        # VF
        vf = "--"
        data = case_info.get("data")
        if data:
            for run in data.get("runs", {}).values():
                r = run.get("report")
                if r:
                    vf_val = r.get("top3d", {}).get("volfrac")
                    if vf_val is not None:
                        vf = f"{vf_val * 100:.0f}" if vf_val <= 1 else f"{vf_val:.0f}"
                    break

        # Collect C_final
        c_vals = {}
        for rl in RUN_LABELS:
            c_vals[rl] = _get_metric(master, case_name, rl, "overall", "final_compliance")

        valid_vals = {k: v for k, v in c_vals.items() if v is not None}
        best_label = min(valid_vals, key=valid_vals.get) if valid_vals else None

        def tex_cell(rl):
            v = c_vals.get(rl)
            if v is None:
                return "--"
            # Format as X.XXe-Y
            s = f"{v:.2e}"
            if rl == best_label:
                return f"\\textbf{{{s}}}"
            return s

        # Escape underscores in case name
        safe_name = case_name.replace("_", "\\_")
        row = (f"{safe_name} & {cat} & {vf}"
               f" & {tex_cell('straight_beamonly')} & {tex_cell('straight_hybrid')}"
               f" & {tex_cell('curved_beamonly')} & {tex_cell('curved_hybrid')}"
               f" & {tex_cell('mixed_beamonly')} & {tex_cell('mixed_hybrid')} \\\\")
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return lines


# ── Save master outputs ────────────────────────────────────────────────────

def save_outputs(batch_dir, master):
    """Write master_summary.txt, .json, and .tex."""
    all_lines = []
    all_lines.append("")
    all_lines.extend(format_compliance_matrix(master))
    all_lines.append("")
    all_lines.extend(format_hybrid_advantage(master))
    all_lines.append("")
    all_lines.extend(format_element_counts(master))
    all_lines.append("")

    text = "\n".join(all_lines)
    print(text)

    # master_summary.txt
    txt_path = os.path.join(batch_dir, "master_summary.txt")
    with open(txt_path, "w") as f:
        f.write(text + "\n")
    print(f"\nMaster tables saved to: {txt_path}")

    # master_summary.json
    json_data = {}
    for case_name, case_info in master.items():
        data = case_info.get("data")
        if not data:
            json_data[case_name] = {"category": case_info["config"]["category"], "runs": {}}
            continue
        runs_summary = {}
        for rl in RUN_LABELS:
            run = data.get("runs", {}).get(rl)
            if not run or not run.get("report"):
                runs_summary[rl] = None
                continue
            overall = run["report"].get("overall", {})
            recon = run["report"].get("reconstruction", {})
            runs_summary[rl] = {
                "c_baseline": overall.get("baseline_compliance"),
                "c_final": overall.get("final_compliance"),
                "vol_error_pct": overall.get("volume_error_pct"),
                "geo_similarity": overall.get("geometric_similarity"),
                "nodes": recon.get("nodes"),
                "edges": recon.get("edges"),
                "plates": recon.get("plates"),
                "elapsed_s": run.get("elapsed_s"),
            }
        json_data[case_name] = {
            "category": case_info["config"]["category"],
            "runs": runs_summary,
        }

    json_path = os.path.join(batch_dir, "master_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=_json_default)
    print(f"JSON data saved to: {json_path}")

    # master_summary.tex
    tex_lines = format_latex_table(master)
    tex_path = os.path.join(batch_dir, "master_summary.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"LaTeX table saved to: {tex_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.aggregate_only:
        # Re-aggregate mode: skip runs, just re-build master tables
        batch_dir = args.aggregate_only
        if not os.path.isdir(batch_dir):
            print(f"[FATAL] Directory not found: {batch_dir}")
            return 1
        cases = resolve_cases(args)
        print(f"\n  Aggregating results from: {batch_dir}")
        master = aggregate(batch_dir, cases)
        save_outputs(batch_dir, master)
        return 0

    # Normal mode: run all cases then aggregate
    batch_dir = run_all_cases(args)

    if args.dry_run:
        print(f"\n[DRY RUN] No runs executed. Commands printed above.")
        return 0

    # Aggregate
    cases = resolve_cases(args)
    print(f"\n{'─' * 80}")
    print(f"  AGGREGATING RESULTS")
    print(f"{'─' * 80}")
    master = aggregate(batch_dir, cases)
    save_outputs(batch_dir, master)

    return 0


if __name__ == "__main__":
    sys.exit(main())
