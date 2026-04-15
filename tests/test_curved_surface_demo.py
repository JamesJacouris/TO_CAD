#!/usr/bin/env python3
"""
Curved Surface Demo — Parabolic Arch via Topology Optimisation
===============================================================

Runs topology optimisation to produce a naturally curved arch structure,
then processes it through the hybrid pipeline and verifies that curved
plates and beam supports are correctly identified.

The curved_shell problem (wide edge supports + full top-surface load) at
60x20x30 with volfrac=0.18 produces two parabolic arch walls — genuine
curved surfaces with 5-6 voxel thick members.

Usage:
    python tests/test_curved_surface_demo.py           # full run (~2min)
    python tests/test_curved_surface_demo.py --reuse    # skip TO if NPZ exists
"""

import numpy as np
import json
import sys
import os
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# 1. Topology optimisation
# ---------------------------------------------------------------------------

def run_topology_optimisation(npz_path, reuse=False):
    """
    Run curved_shell problem: 60×20×30, volfrac=0.18, rmin=2.5.

    At this resolution and volume fraction, the optimizer produces two
    parabolic arch walls instead of a flat slab.  Members are 5-6 voxels
    thick — sufficient for two-pass thinning surface detection.

    If reuse=True and npz_path exists, skip the optimisation.
    """
    if reuse and os.path.exists(npz_path):
        print(f"[TO] Reusing existing NPZ: {npz_path}")
        d = np.load(npz_path, allow_pickle=True)
        n_solid = int(np.sum(d['rho'] > 0.3))
        print(f"  Solid voxels: {n_solid}")
        return

    import subprocess
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)

    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "run_top3d.py"),
        "--problem", "curved_shell",
        "--nelx", "60", "--nely", "20", "--nelz", "30",
        "--volfrac", "0.18",
        "--rmin", "2.5",
        "--max_loop", "80",
        "--output", npz_path,
    ]
    print(f"[TO] Running topology optimisation (60x20x30, VF=0.18)...")
    print(f"  This takes ~2 minutes. Use --reuse to skip on subsequent runs.")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[TO] FAILED (exit {result.returncode})")
        print(result.stderr[-500:] if result.stderr else "no stderr")
        sys.exit(1)

    # Extract key info from output
    for line in result.stdout.splitlines():
        if "Converged" in line:
            print(f"  {line.strip()}")
    print(f"[TO] Completed in {elapsed:.0f}s → {npz_path}")


# ---------------------------------------------------------------------------
# 2. Pipeline run + verification
# ---------------------------------------------------------------------------

def run_and_verify(npz_path, json_path):
    """
    Run the hybrid pipeline and verify curved surface detection.
    Returns True if all checks pass.
    """
    from src.pipelines.baseline_yin.reconstruct import reconstruct_npz

    print("\n" + "=" * 70)
    print("  RUNNING HYBRID PIPELINE")
    print("=" * 70)
    t0 = time.time()

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    reconstruct_npz(
        npz_path, json_path,
        hybrid=True,
        min_plate_size=8,
        flatness_ratio=5.0,
        min_avg_neighbors=3.0,
        vol_thresh=0.3,
        prune_len=1.5,
        collapse_thresh=2.0,
        plate_mode='bspline',
        visualize=False,
    )
    elapsed = time.time() - t0
    print(f"\n[pipeline] Completed in {elapsed:.1f}s → {json_path}")

    # --- Load output ---
    with open(json_path) as f:
        data = json.load(f)

    plates = data.get('plates', [])
    graph = data.get('graph', {})
    edges = graph.get('edges', [])
    nodes = graph.get('nodes', [])

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("  CURVED SURFACE DEMO — RESULTS")
    print("=" * 70)
    print(f"{'Plate':>6} {'Curved':>8} {'NormDev':>9} {'SkelVox':>9}"
          f" {'MS Verts':>10} {'MS Tris':>9} {'Thickness':>10}")
    print("-" * 70)
    for p in plates:
        ms = p.get('mid_surface')
        n_skel = len(p.get('voxels', []))
        n_verts = len(ms['vertices']) if ms else 0
        n_tris = len(ms['triangles']) if ms else 0
        thk = f"{ms.get('mean_thickness', 0):.2f}" if ms else "--"
        dev = p.get('max_normal_deviation_deg', 0)
        print(f"{p.get('id', '?'):>6} {'YES' if p.get('is_curved') else 'no':>8}"
              f" {dev:>8.1f}° {n_skel:>9} {n_verts:>10} {n_tris:>9} {thk:>10}")
    print(f"\n  Beam graph: {len(nodes)} nodes, {len(edges)} edges")
    print("=" * 70)

    # --- Verification checks ---
    checks = []

    def check(name, condition, msg=""):
        checks.append((name, condition))
        mark = "  [PASS]" if condition else "  [FAIL]"
        detail = f" — {msg}" if msg else ""
        print(f"{mark} {name}{detail}")

    print()
    check("Plates extracted",
          len(plates) >= 1,
          f"{len(plates)} plate(s)")

    curved_plates = [p for p in plates if p.get('is_curved', False)]
    check("Curved plate detected",
          len(curved_plates) >= 1,
          f"{len(curved_plates)} curved plate(s)")

    # Find the largest curved plate for detailed checks
    if curved_plates:
        largest_cp = max(curved_plates,
                         key=lambda p: len(p.get('mid_surface', {}).get('vertices', [])))
        dev = largest_cp.get('max_normal_deviation_deg', 0)
        check("Normal deviation > 15deg",
              dev > 15.0,
              f"{dev:.1f}deg")

        ms = largest_cp.get('mid_surface')
        if ms:
            verts = ms.get('vertices', [])
            vnormals = ms.get('vertex_normals', [])
            check("Mid-surface vertices >= 20",
                  len(verts) >= 20,
                  f"{len(verts)} vertices")

            check("Vertex normals present",
                  len(vnormals) == len(verts),
                  f"{len(vnormals)} normals for {len(verts)} verts")

            if vnormals:
                norms = [np.linalg.norm(n) for n in vnormals]
                all_unit = all(0.9 < n < 1.1 for n in norms)
                check("Normals are unit vectors",
                      all_unit,
                      f"range [{min(norms):.3f}, {max(norms):.3f}]")

            thk = ms.get('thickness_per_vertex', [])
            n_pos = sum(1 for t in thk if t > 0)
            frac_pos = n_pos / len(thk) if thk else 0
            mean_thk = sum(thk) / len(thk) if thk else 0
            check("Mean thickness > 0",
                  mean_thk > 0.5,
                  f"mean={mean_thk:.2f}, {frac_pos:.0%} positive" if thk else "missing")
        else:
            check("Mid-surface data", False, "mid_surface is None")
    else:
        for name in ["Normal deviation > 15deg", "Mid-surface vertices >= 20",
                      "Vertex normals present", "Normals are unit vectors",
                      "Mean thickness > 0"]:
            check(name, False, "no curved plates to check")

    check("Beam edges present",
          len(edges) >= 4,
          f"{len(edges)} edges")

    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    print(f"\n  Result: {n_pass}/{n_total} checks passed")
    return n_pass == n_total


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Curved surface demo test")
    parser.add_argument("--reuse", action="store_true",
                        help="Skip topology optimisation if NPZ already exists")
    args = parser.parse_args()

    out_dir = os.path.join(REPO_ROOT, "output", "curved_demo")
    npz_path = os.path.join(out_dir, "curved_shell_v2_top3d.npz")
    json_path = os.path.join(out_dir, "curved_shell_v2.json")

    print("=" * 70)
    print("  CURVED SURFACE DEMO")
    print("  Parabolic arch via topology optimisation (60x20x30, VF=0.18)")
    print("=" * 70)

    # Step 1: topology optimisation
    run_topology_optimisation(npz_path, reuse=args.reuse)

    # Step 2: hybrid pipeline + verify
    all_ok = run_and_verify(npz_path, json_path)

    if all_ok:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED — see details above")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
