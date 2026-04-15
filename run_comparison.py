#!/usr/bin/env python3
"""
Beam Mode x Hybrid Comparison Runner
=====================================
Runs all combinations of beam_mode (straight, curved, mixed) x hybrid (on/off)
and produces comparison tables.

Usage:
    python run_comparison.py \\
        --npz output/hybrid_v2/Clear_Beam_Plate_Test_top3d_0.05_VF.npz \\
        --opt_loops 2 \\
        --min_plate_size 8 --flatness_ratio 7 --min_avg_neighbors 3 \\
        --snap 1.5 --prune_len 1.5 --collapse_thresh 2.0 --rdp 2.0 \\
        --symmetry xz,yz --sym_weight 0.1
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


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run beam_mode x hybrid comparison matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All unrecognised arguments are forwarded to run_pipeline.py.",
    )
    parser.add_argument("--npz", required=True,
                        help="Path to .npz file (maps to --top3d_npz)")
    parser.add_argument("--base_dir", default="output/comparison",
                        help="Root output directory (default: output/comparison)")
    parser.add_argument("--opt_loops", type=int, default=2,
                        help="Number of optimisation loops per run (default: 2)")
    parser.add_argument("--beam_modes", default="straight,curved,mixed",
                        help="Comma-separated beam modes to test (default: straight,curved,mixed)")
    parser.add_argument("--skip_beamonly", action="store_true",
                        help="Skip beam-only (non-hybrid) runs")
    parser.add_argument("--skip_hybrid", action="store_true",
                        help="Skip hybrid runs")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-run timeout in seconds (default: 3600)")
    return parser


# ── Combination builder ───────────────────────────────────────────────────

def build_combinations(args):
    """Return list of (beam_mode, is_hybrid, label) tuples."""
    modes = [m.strip() for m in args.beam_modes.split(",")]
    combos = []
    for mode in modes:
        if not args.skip_hybrid:
            combos.append((mode, True, f"{mode}_hybrid"))
        if not args.skip_beamonly:
            combos.append((mode, False, f"{mode}_beamonly"))
    return combos


# ── Command builder ───────────────────────────────────────────────────────

def build_command(args, passthrough, beam_mode, is_hybrid, sub_dir):
    """Build the subprocess command list."""
    cmd = [
        sys.executable, "run_pipeline.py",
        "--skip_top3d",
        "--top3d_npz", args.npz,
        "--beam_mode", beam_mode,
        "--output", "result.json",
        "--output_dir", sub_dir,
        "--optimize",
        "--opt_loops", str(args.opt_loops),
    ]
    if is_hybrid:
        cmd.append("--hybrid")
    cmd.extend(passthrough)
    return cmd


# ── Runner ────────────────────────────────────────────────────────────────

def run_all(args, passthrough):
    """Execute all combinations, return (run_dir, results_dict)."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(args.base_dir, timestamp)

    combos = build_combinations(args)
    # Use ordered dict behaviour (Python 3.7+)
    results = {}

    print(f"\n{'=' * 100}")
    print(f"  BEAM MODE x HYBRID COMPARISON — {len(combos)} runs")
    print(f"  NPZ: {args.npz}")
    print(f"  Output: {run_dir}")
    print(f"  Passthrough: {' '.join(passthrough)}")
    print(f"{'=' * 100}")

    for i, (beam_mode, is_hybrid, label) in enumerate(combos):
        sub_dir = os.path.join(run_dir, label)
        os.makedirs(sub_dir, exist_ok=True)

        cmd = build_command(args, passthrough, beam_mode, is_hybrid, sub_dir)

        print(f"\n{'─' * 80}")
        print(f"  [{i + 1}/{len(combos)}] {label}")
        print(f"  cmd: {' '.join(cmd)}")
        print(f"{'─' * 80}")

        if args.dry_run:
            results[label] = {
                "returncode": -999, "error": "dry_run",
                "beam_mode": beam_mode, "hybrid": is_hybrid,
                "dir": sub_dir,
            }
            continue

        log_path = os.path.join(sub_dir, "pipeline.log")
        t0 = time.time()
        try:
            with open(log_path, "w") as log_f:
                proc = subprocess.run(
                    cmd, stdout=log_f, stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
            elapsed = time.time() - t0
            results[label] = {
                "returncode": proc.returncode,
                "beam_mode": beam_mode,
                "hybrid": is_hybrid,
                "log": log_path,
                "dir": sub_dir,
                "elapsed_s": elapsed,
            }
            status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            print(f"  {status} in {elapsed:.0f}s")
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            results[label] = {
                "returncode": -1, "error": "timeout",
                "beam_mode": beam_mode, "hybrid": is_hybrid,
                "dir": sub_dir, "elapsed_s": elapsed,
            }
            print(f"  TIMEOUT after {elapsed:.0f}s")
        except Exception as e:
            results[label] = {
                "returncode": -1, "error": str(e),
                "beam_mode": beam_mode, "hybrid": is_hybrid,
                "dir": sub_dir, "elapsed_s": 0,
            }
            print(f"  ERROR: {e}")

    return run_dir, results


# ── Report collection ─────────────────────────────────────────────────────

def collect_reports(results):
    """Load _report.json for each successful run."""
    for label, info in results.items():
        info["report"] = None
        if info.get("returncode", -1) != 0:
            continue
        report_files = glob.glob(os.path.join(info["dir"], "*_report.json"))
        if report_files:
            try:
                with open(report_files[0]) as f:
                    info["report"] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass


# ── Formatters ────────────────────────────────────────────────────────────

def _fc(v):
    """Format compliance value."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{v:.5e}"


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


# ── Table A: Per-loop optimisation detail ─────────────────────────────────

def format_detail_table(results):
    """Per-loop detail: every Size/Layout stage for every run."""
    W = 150
    lines = []
    lines.append("=" * W)
    lines.append(f"{'PER-LOOP OPTIMISATION DETAIL':^{W}}")
    lines.append("=" * W)

    hdr = (f"{'Run':<25} | {'Loop':>4} | {'Stage':<7} | "
           f"{'C_init':>14} | {'C_final':>14} | {'Iters':>5} | "
           f"{'r_min':>7} | {'r_max':>7} | {'r_mean':>7} | "
           f"{'max_disp':>10} | {'mean_disp':>10}")
    lines.append(hdr)
    lines.append("-" * W)

    for label, info in results.items():
        report = info.get("report")
        if not report:
            err = info.get("error", f"exit {info.get('returncode', '?')}")
            lines.append(f"{label:<25} | {'FAILED: ' + str(err)}")
            lines.append("-" * W)
            continue

        first_row = True
        for loop_data in report.get("optimization_loops", []):
            lp = loop_data["loop"]

            # Size stage
            s = loop_data.get("size", {})
            if s:
                run_col = label if first_row else ""
                first_row = False
                lines.append(
                    f"{run_col:<25} | {lp:>4} | {'Size':<7} | "
                    f"{_fc(s.get('c_initial')):>14} | {_fc(s.get('c_final')):>14} | "
                    f"{_fi(s.get('iterations')):>5} | "
                    f"{_ff(s.get('radius_min')):>7} | {_ff(s.get('radius_max')):>7} | "
                    f"{_ff(s.get('radius_mean')):>7} | "
                    f"{'--':>10} | {'--':>10}"
                )

            # Layout stage
            lo = loop_data.get("layout", {})
            if lo:
                run_col = label if first_row else ""
                first_row = False
                lines.append(
                    f"{run_col:<25} | {lp:>4} | {'Layout':<7} | "
                    f"{_fc(lo.get('c_initial')):>14} | {_fc(lo.get('c_final')):>14} | "
                    f"{_fi(lo.get('iterations')):>5} | "
                    f"{'--':>7} | {'--':>7} | {'--':>7} | "
                    f"{_ff(lo.get('max_node_disp'), 4):>10} | "
                    f"{_ff(lo.get('mean_node_disp'), 4):>10}"
                )

        lines.append("-" * W)

    return lines


# ── Table B: Final summary ───────────────────────────────────────────────

def format_summary_table(results):
    """One row per run with final metrics."""
    W = 150
    n_ok = sum(1 for r in results.values() if r.get("report"))
    lines = []
    lines.append("=" * W)
    lines.append(f"{'COMPARISON SUMMARY — ' + str(n_ok) + ' successful run(s)':^{W}}")
    lines.append("=" * W)

    hdr = (f"{'Run':<25} | {'C_baseline':>14} | {'C_final':>14} | "
           f"{'Delta C%':>9} | {'VolErr%':>8} | {'GeoSim':>7} | "
           f"{'Nodes':>5} | {'Edges':>5} | {'Plates':>6} | "
           f"{'vs SIMP':>9} | {'Time':>7}")
    lines.append(hdr)
    lines.append("-" * W)

    for label, info in results.items():
        report = info.get("report")
        if not report:
            err = info.get("error", f"exit {info.get('returncode', '?')}")
            lines.append(f"{label:<25} | FAILED: {err}")
            continue

        overall = report.get("overall", {})
        recon = report.get("reconstruction", {})
        continuum = report.get("continuum", {})

        c_b = overall.get("baseline_compliance")
        c_f = overall.get("final_compliance")
        reduction = None
        if c_b and c_f and c_b > 0:
            if not (isinstance(c_f, float) and math.isnan(c_f)):
                reduction = (c_f - c_b) / c_b * 100  # negative = compliance decreased

        c_simp = continuum.get("simp_p1_rescaled")
        yin = continuum.get("yin_stages", [])
        c_yin_final = yin[-1]["compliance"] if yin else None
        delta_simp = None
        if c_simp and c_yin_final and c_simp > 0:
            if not (isinstance(c_yin_final, float) and math.isnan(c_yin_final)):
                delta_simp = (c_yin_final - c_simp) / c_simp * 100

        elapsed = info.get("elapsed_s", 0)
        time_str = f"{elapsed:.0f}s" if elapsed else "--"

        lines.append(
            f"{label:<25} | {_fc(c_b):>14} | {_fc(c_f):>14} | "
            f"{_fp(reduction):>9} | "
            f"{_ff(overall.get('volume_error_pct')):>8} | "
            f"{_ff(overall.get('geometric_similarity'), 3):>7} | "
            f"{_fi(recon.get('nodes')):>5} | {_fi(recon.get('edges')):>5} | "
            f"{_fi(recon.get('plates')):>6} | "
            f"{_fp(delta_simp):>9} | {time_str:>7}"
        )

    lines.append("=" * W)
    return lines


# ── Table C: Yin-stage cross-comparison ───────────────────────────────────

def format_yin_table(results):
    """Compliance at each pipeline stage, columns = runs."""
    lines = []

    # Collect valid runs and yin data
    valid_labels = []
    yin_data = {}
    all_stage_labels = []

    for label, info in results.items():
        report = info.get("report")
        if not report:
            continue
        stages = report.get("continuum", {}).get("yin_stages", [])
        if not stages:
            continue
        valid_labels.append(label)
        yin_data[label] = {s["label"]: s["compliance"] for s in stages}
        for s in stages:
            if s["label"] not in all_stage_labels:
                all_stage_labels.append(s["label"])

    if not valid_labels:
        return lines

    # Calculate dynamic width
    col_w = max(max(len(l) for l in valid_labels), 14)
    stage_col = 45
    W = stage_col + (col_w + 3) * len(valid_labels) + 1

    lines.append("=" * W)
    lines.append(f"{'DIRECT COMPLIANCE COMPARISON (Yin et al. 2020)':^{W}}")
    lines.append("-" * W)

    # Header
    hdr = f"{'Stage':<{stage_col}}"
    for label in valid_labels:
        hdr += f" | {label:>{col_w}}"
    lines.append(hdr)
    lines.append("-" * W)

    # SIMP row
    simp_row = f"{'SIMP binary (hex FEM, p=1)':<{stage_col}}"
    for label in valid_labels:
        report = results[label].get("report", {})
        c_simp = report.get("continuum", {}).get("simp_p1_rescaled")
        simp_row += f" | {_fc(c_simp):>{col_w}}"
    lines.append(simp_row)

    # Each Yin stage
    for stage_label in all_stage_labels:
        row = f"{stage_label:<{stage_col}}"
        for label in valid_labels:
            c = yin_data.get(label, {}).get(stage_label)
            row += f" | {_fc(c):>{col_w}}"
        lines.append(row)

    lines.append("=" * W)
    return lines


# ── JSON helper ───────────────────────────────────────────────────────────

def _json_default(obj):
    """Handle non-serializable types."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


# ── Save outputs ──────────────────────────────────────────────────────────

def save_outputs(run_dir, results, args, passthrough):
    """Print tables, save summary.txt and summary.json."""
    all_lines = []
    all_lines.append("")
    all_lines.extend(format_detail_table(results))
    all_lines.append("")
    all_lines.extend(format_summary_table(results))
    all_lines.append("")
    all_lines.extend(format_yin_table(results))
    all_lines.append("")

    text = "\n".join(all_lines)
    print(text)

    # summary.txt
    txt_path = os.path.join(run_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write(text + "\n")

    # summary.json
    json_path = os.path.join(run_dir, "summary.json")
    summary_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "npz": args.npz,
        "opt_loops": args.opt_loops,
        "common_args": passthrough,
        "runs": {},
        "summary": [],
    }

    for label, info in results.items():
        summary_data["runs"][label] = {
            "beam_mode": info.get("beam_mode"),
            "hybrid": info.get("hybrid"),
            "returncode": info.get("returncode"),
            "elapsed_s": info.get("elapsed_s"),
            "report": info.get("report"),
        }

        report = info.get("report")
        if report:
            overall = report.get("overall", {})
            recon = report.get("reconstruction", {})
            continuum = report.get("continuum", {})

            c_b = overall.get("baseline_compliance")
            c_f = overall.get("final_compliance")
            reduction = None
            if c_b and c_f and c_b > 0:
                if not (isinstance(c_f, float) and math.isnan(c_f)):
                    reduction = (c_f - c_b) / c_b * 100  # negative = improvement

            c_simp = continuum.get("simp_p1_rescaled")
            yin = continuum.get("yin_stages", [])
            c_yin_final = yin[-1]["compliance"] if yin else None
            delta_simp = None
            if c_simp and c_yin_final and c_simp > 0:
                if not (isinstance(c_yin_final, float) and math.isnan(c_yin_final)):
                    delta_simp = (c_yin_final - c_simp) / c_simp * 100

            summary_data["summary"].append({
                "label": label,
                "beam_mode": info.get("beam_mode"),
                "hybrid": info.get("hybrid"),
                "c_baseline": c_b,
                "c_final": c_f,
                "reduction_pct": reduction,
                "vol_error_pct": overall.get("volume_error_pct"),
                "geo_similarity": overall.get("geometric_similarity"),
                "nodes": recon.get("nodes"),
                "edges": recon.get("edges"),
                "plates": recon.get("plates"),
                "delta_vs_simp_pct": delta_simp,
                "elapsed_s": info.get("elapsed_s"),
            })

    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=_json_default)

    print(f"\nSummary saved to: {txt_path}")
    print(f"JSON data saved to: {json_path}")
    print(f"Per-run outputs in: {run_dir}/{{label}}/")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    if not os.path.exists(args.npz):
        print(f"[FATAL] NPZ file not found: {args.npz}")
        return 1

    if args.skip_hybrid and args.skip_beamonly:
        print("[FATAL] Cannot skip both hybrid and beam-only — nothing to run.")
        return 1

    run_dir, results = run_all(args, passthrough)

    if args.dry_run:
        print(f"\n[DRY RUN] No runs executed. Commands printed above.")
        return 0

    collect_reports(results)
    save_outputs(run_dir, results, args, passthrough)

    any_ok = any(r.get("returncode") == 0 for r in results.values())
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
