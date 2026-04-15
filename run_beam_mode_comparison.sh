#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Run the hybrid pipeline for each beam_mode and produce a summary table.
# Usage:  bash run_beam_mode_comparison.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

NPZ="output/hybrid_v2/Clear_Beam_Plate_Test_top3d_0.05_VF.npz"
BASE_DIR="output/beam_mode_comparison"
MODES=("straight" "curved" "mixed")

COMMON_ARGS=(
  --skip_top3d
  --top3d_npz "$NPZ"
  --min_plate_size 8
  --flatness_ratio 7
  --min_avg_neighbors 3
  --snap 1.5
  --prune_len 1.5
  --collapse_thresh 2.0
  --rdp 2.5
  --optimize
  --hybrid
  --opt_loops 5
)

mkdir -p "$BASE_DIR"

SUMMARY_FILE="$BASE_DIR/summary.txt"
SUMMARY_JSON="$BASE_DIR/summary.json"

# Header
printf "\n%s\n" "$(printf '=%.0s' {1..100})"
printf "  BEAM MODE COMPARISON — %s\n" "$(date '+%Y-%m-%d %H:%M')"
printf "%s\n\n" "$(printf '=%.0s' {1..100})"

# JSON array start
echo "[" > "$SUMMARY_JSON"

for i in "${!MODES[@]}"; do
  MODE="${MODES[$i]}"
  OUT_DIR="$BASE_DIR/$MODE"
  mkdir -p "$OUT_DIR"
  OUT_JSON="result.json"
  LOG_FILE="$OUT_DIR/pipeline.log"

  printf "\n%s\n" "$(printf '─%.0s' {1..80})"
  printf "  Running beam_mode=%s  →  %s\n" "$MODE" "$OUT_DIR"
  printf "%s\n" "$(printf '─%.0s' {1..80})"

  # Run pipeline, tee to log
  python run_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --beam_mode "$MODE" \
    --output "$OUT_JSON" \
    --output_dir "$OUT_DIR" \
    2>&1 | tee "$LOG_FILE"

  printf "\n  ✓ %s complete — log: %s\n" "$MODE" "$LOG_FILE"

  # Add comma separator between JSON entries
  if [ "$i" -gt 0 ]; then
    # Insert comma before this entry
    printf ",\n" >> "$SUMMARY_JSON"
  fi

  # Extract the report JSON if it exists
  REPORT_JSON=$(find "$OUT_DIR" -name "*_report.json" -type f | head -1)
  if [ -n "$REPORT_JSON" ]; then
    printf '{"beam_mode": "%s", "report": ' "$MODE" >> "$SUMMARY_JSON"
    cat "$REPORT_JSON" >> "$SUMMARY_JSON"
    printf "}" >> "$SUMMARY_JSON"
  else
    printf '{"beam_mode": "%s", "report": null}' "$MODE" >> "$SUMMARY_JSON"
  fi
done

echo "]" >> "$SUMMARY_JSON"

# ──────────────────────────────────────────────────────────────────────
# Parse report JSONs and build summary table
# ──────────────────────────────────────────────────────────────────────
printf "\n\n" > "$SUMMARY_FILE"
python3 - "$BASE_DIR" "${MODES[@]}" >> "$SUMMARY_FILE" <<'PYEOF'
import json, sys, os, glob

base_dir = sys.argv[1]
modes = sys.argv[2:]

sep = "=" * 110
thin = "-" * 110

print(sep)
print(f"{'BEAM MODE COMPARISON SUMMARY':^110}")
print(sep)

# Collect data
rows = []
for mode in modes:
    report_files = glob.glob(os.path.join(base_dir, mode, "*_report.json"))
    if not report_files:
        rows.append({"mode": mode, "error": "No report found"})
        continue
    with open(report_files[0]) as f:
        data = json.load(f)

    overall = data.get("overall", {})
    recon = data.get("reconstruction", {})
    continuum = data.get("continuum", {})
    yin_stages = continuum.get("yin_stages", [])

    c_baseline = overall.get("baseline_compliance")
    c_final = overall.get("final_compliance")
    vol_target = overall.get("volume_target")
    vol_final = overall.get("volume_final")
    vol_err = overall.get("volume_error_pct")
    geo_sim = overall.get("geometric_similarity")
    n_nodes = recon.get("nodes")
    n_edges = recon.get("edges")
    n_plates = recon.get("plates")

    reduction = None
    if c_baseline and c_final and c_baseline > 0:
        reduction = (c_baseline - c_final) / c_baseline * 100

    # SIMP binary from yin stages
    c_simp = continuum.get("simp_p1_rescaled")
    c_final_yin = yin_stages[-1]["compliance"] if yin_stages else None

    delta_vs_simp = None
    if c_simp and c_final_yin:
        delta_vs_simp = (c_final_yin - c_simp) / c_simp * 100

    rows.append({
        "mode": mode,
        "c_baseline": c_baseline,
        "c_final": c_final,
        "c_simp": c_simp,
        "reduction": reduction,
        "delta_vs_simp": delta_vs_simp,
        "vol_target": vol_target,
        "vol_final": vol_final,
        "vol_err": vol_err,
        "geo_sim": geo_sim,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_plates": n_plates,
    })

def fmt_c(v):
    return f"{v:.5e}" if v is not None else "—"

def fmt_f(v, d=2):
    return f"{v:.{d}f}" if v is not None else "—"

def fmt_pct(v):
    return f"{v:+.1f}%" if v is not None else "—"

def fmt_int(v):
    return str(v) if v is not None else "—"

# Table
hdr = (f"{'Mode':<10} | {'C_baseline':>13} | {'C_final':>13} | {'Reduction':>10} | "
       f"{'Δ vs SIMP':>10} | {'Vol Err%':>9} | {'Geo Sim':>8} | "
       f"{'Nodes':>6} | {'Edges':>6} | {'Plates':>7}")
print()
print(hdr)
print(thin)

for r in rows:
    if "error" in r:
        print(f"{r['mode']:<10} | {r['error']}")
        continue
    print(f"{r['mode']:<10} | {fmt_c(r['c_baseline']):>13} | {fmt_c(r['c_final']):>13} | "
          f"{fmt_pct(r['reduction']):>10} | {fmt_pct(r['delta_vs_simp']):>10} | "
          f"{fmt_f(r['vol_err']):>9}% | {fmt_f(r['geo_sim']):>8} | "
          f"{fmt_int(r['n_nodes']):>6} | {fmt_int(r['n_edges']):>6} | {fmt_int(r['n_plates']):>7}")

print(thin)

# Yin comparison per mode
print(f"\n{'DIRECT COMPLIANCE COMPARISON (Yin et al. 2020)':^110}")
print(thin)
print(f"{'Stage':<45}", end="")
for r in rows:
    if "error" not in r:
        print(f" | {r['mode']:>13}", end="")
print()
print(thin)

# Collect all unique stage labels across modes
all_labels = []
yin_data = {}
for mode in modes:
    report_files = glob.glob(os.path.join(base_dir, mode, "*_report.json"))
    if not report_files:
        continue
    with open(report_files[0]) as f:
        data = json.load(f)
    stages = data.get("continuum", {}).get("yin_stages", [])
    yin_data[mode] = {s["label"]: s["compliance"] for s in stages}
    for s in stages:
        if s["label"] not in all_labels:
            all_labels.append(s["label"])

for label in all_labels:
    print(f"{label:<45}", end="")
    for r in rows:
        if "error" in r:
            continue
        c = yin_data.get(r["mode"], {}).get(label)
        print(f" | {fmt_c(c):>13}", end="")
    print()

print(thin)
print()
PYEOF

cat "$SUMMARY_FILE"

printf "\nSummary saved to: %s\n" "$SUMMARY_FILE"
printf "JSON data saved to: %s\n" "$SUMMARY_JSON"
printf "Per-mode outputs in: %s/{straight,curved,mixed}/\n\n" "$BASE_DIR"
