#!/usr/bin/env python3
"""
Baseline Yin + Top3D Hybrid Pipeline V2
========================================

Full pipeline with Yin-style BC tag propagation:
  Stage 0: Python Top3D → .npz (with bc_tags)
  Stage 1: Reconstruction → JSON (skeleton, graph, node_tags)
  Stage 2: Layout Optimization → JSON (optimized positions)
  Stage 3: Size Optimization → JSON (optimized radii)

Usage:
  python run_hybrid.py \\
    --nelx 50 --nely 10 --nelz 2 \\
    --volfrac 0.3 \\
    --load_x 50 --load_y 5 --load_z 1 \\
    --load_fx 0.0 --load_fy -1.0 --load_fz 0.0 \\
    --prune_len 2.0 \\
    --collapse_thresh 2 \\
    --rdp 1 \\
    --radius_mode uniform \\
    --limit 5.0 \\
    --snap 5.0 \\
    --visualize \\
    --output "full_control_beam.json"
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


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
        description="Baseline Yin + Top3D Hybrid Pipeline V2 (with BC Tag Propagation)",
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
    
    # === Layout & Size Optimisation ===
    g_opt = parser.add_argument_group("Layout & Size Optimisation")
    g_opt.add_argument("--limit", type=float, default=5.0, help="Layout move limit (mm)")
    g_opt.add_argument("--snap", type=float, default=5.0, help="Snap distance for node merging (mm)")
    g_opt.add_argument("--iters", type=int, default=50, help="Size opt iterations")
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
    
    # Derive output base name from --output
    base_name = os.path.splitext(os.path.basename(args.output))[0]
    
    print("=" * 60)
    print("   BASELINE YIN + TOP3D HYBRID PIPELINE V2")
    print("   (with Yin-style BC Tag Propagation)")
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
        # else use the default derived path
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
    if args.visualize: cmd.append("--visualize")
    
    if not run_stage(cmd, "STAGE 1: Skeleton Reconstruction (with BC Tag Propagation)"):
        return 1
    
    # ========================================
    # STAGE 2: Layout Optimisation
    # ========================================
    stage2_out = os.path.join(args.output_dir, f"{base_name}_2_layout.json")
    
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "src/optimization/layout_opt.py"),
        stage1_out, stage2_out,
        "--problem", args.problem,
        "--limit", str(args.limit),
        "--snap", str(args.snap),
    ]
    if args.visualize: cmd.append("--visualise")
    
    if not run_stage(cmd, "STAGE 2: Layout Optimisation"):
        return 1
    
    # ========================================
    # STAGE 3: Size Optimisation
    # ========================================
    stage3_out = os.path.join(args.output_dir, f"{base_name}_3_sized.json")
    
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "src/optimization/size_opt.py"),
        stage2_out, stage3_out,
        "--problem", args.problem,
        "--iters", str(args.iters),
    ]
    if args.visualize: cmd.append("--visualize")
    
    if not run_stage(cmd, "STAGE 3: Size Optimisation"):
        return 1
    
    # ========================================
    # FINAL: Copy final output to requested name
    # ========================================
    final_path = os.path.join(args.output_dir, args.output)
    
    import shutil
    shutil.copy2(stage3_out, final_path)
    
    # ========================================
    # Pipeline History (for FreeCAD)
    # ========================================
    try:
        with open(stage1_out, 'r') as f: s1 = json.load(f)
        with open(stage2_out, 'r') as f: s2 = json.load(f)
        with open(stage3_out, 'r') as f: s3 = json.load(f)
        
        history = {
            "metadata": s3.get("metadata", {}),
            "history": s1.get("history", []),
            "stages": [
                {"name": "1. Reconstructed", "curves": s1.get("curves", [])},
                {"name": "2. Layout Optimised", "curves": s2.get("curves", [])},
                {"name": "3. Size Optimised", "curves": s3.get("curves", [])},
            ],
            "curves": s3.get("curves", [])
        }
        
        hist_path = os.path.join(args.output_dir, f"{base_name}_history.json")
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n[Export] Pipeline history: {hist_path}")
    except Exception as e:
        print(f"[Warning] Could not create history: {e}")
    
    # === Summary ===
    print(f"\n{'='*60}")
    print("              PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Top3D:         {npz_path}")
    print(f"  Reconstructed: {stage1_out}")
    print(f"  Layout Opt:    {stage2_out}")
    print(f"  Size Opt:      {stage3_out}")
    print(f"  Final Output:  {final_path}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
