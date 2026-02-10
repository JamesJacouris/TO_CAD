import os
import sys
import subprocess
import json

def run_command(cmd, desc):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd[:3])}...")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"[ERROR] {desc} failed with exit code {result.returncode}")
        return False
    return True

def run_hybrid_logic(args, top3d_result):
    """
    Orchestrates the three stages of the pipeline, starting from Top3D result.
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames based on problem name
    base_name = args.problem
    stage1_output = os.path.join(args.output_dir, f"{base_name}_1_reconstruction.json")
    stage2_output = os.path.join(args.output_dir, f"{base_name}_2_layout_optimized.json")
    stage3_output = os.path.join(args.output_dir, f"{base_name}_3_fully_optimized.json")
    
    print("=" * 60)
    print("           HYBRID OPTIMIZATION PIPELINE")
    print("=" * 60)
    print(f"Input:   {top3d_result}")
    print(f"Problem: {args.problem}")
    print(f"Output:  {args.output_dir}")
    print("=" * 60)
    
    # Get repo root directory (Relative to THIS file in Baseline_Yin_Top3D_Pipeline_V2/src/pipelines/unified)
    # This file is deep, so we go up 3 levels to get to Baseline_Yin_Top3D_Pipeline_V2 root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    
    # ========================================
    # STAGE 1: Baseline Reconstruction
    # ========================================
    # Pass .npz instead of mesh
    cmd_stage1 = [
        sys.executable, 
        os.path.join(repo_root, "src/pipelines/baseline_yin/reconstruct.py"),
        top3d_result,
        stage1_output,
        "--pitch", str(args.pitch),
        "--max_iters", str(args.max_iters),
        "--collapse_thresh", str(args.collapse_thresh),
        "--prune_len", str(args.prune_len),
        "--rdp_epsilon", str(args.rdp_epsilon),
        "--radius_mode", args.radius_mode,
    ]
    if args.visualise:
        cmd_stage1.append("--visualize")
    
    if not run_command(cmd_stage1, "STAGE 1: Baseline Reconstruction"):
        return 1
    
    # ========================================
    # STAGE 2: Layout Optimization
    # ========================================
    cmd_stage2 = [
        sys.executable,
        os.path.join(repo_root, "src/optimization/layout_opt.py"),
        stage1_output,
        stage2_output,
        "--problem", args.problem,
        "--limit", str(args.limit),
        "--snap", str(args.snap),
    ]
    if args.visualise:
        cmd_stage2.append("--visualise")
    
    if not run_command(cmd_stage2, "STAGE 2: Layout Optimization"):
        return 1
    
    # ========================================
    # STAGE 3: Size Optimization
    # ========================================
    cmd_stage3 = [
        sys.executable,
        os.path.join(repo_root, "src/optimization/size_opt.py"),
        stage2_output,
        stage3_output,
        "--problem", args.problem,
        "--iters", str(args.iters),
    ]
    if args.visualise:
        cmd_stage3.append("--visualize")
    
    if not run_command(cmd_stage3, "STAGE 3: Size Optimization"):
        return 1
    
    # ========================================
    # EXPORT COMBINED HISTORY FOR FREECAD
    # ========================================
    print(f"\n{'='*60}")
    print("  EXPORTING PIPELINE HISTORY")
    print(f"{'='*60}")
    
    history_path = os.path.join(args.output_dir, "pipeline_history.json")
    
    try:
        with open(stage1_output, 'r') as f:
            stage1_data = json.load(f)
        with open(stage2_output, 'r') as f:
            stage2_data = json.load(f)
        with open(stage3_output, 'r') as f:
            stage3_data = json.load(f)
        
        stages = []
        stage1_history = stage1_data.get("history", [])
        for sub in stage1_history:
            step_name = sub.get("step", "Unknown")
            step_type = sub.get("type", "graph")
            
            if step_type == "graph":
                nodes = sub.get("nodes", [])
                edges = sub.get("edges", [])
                curves = []
                for e in edges:
                    if len(e) >= 3:
                        u, v = int(e[0]), int(e[1])
                        rad = float(e[2]) if len(e) > 2 else 1.0
                        pts_list = e[3] if len(e) > 3 else []
                        if u < len(nodes) and v < len(nodes):
                            p1 = nodes[u]
                            p2 = nodes[v]
                            if len(pts_list) == 0:
                                curve_pts = [p1 + [rad], p2 + [rad]]
                            else:
                                curve_pts = [p1 + [rad]] + [list(p) + [rad] for p in pts_list] + [p2 + [rad]]
                            curves.append({"points": curve_pts})
                stages.append({"name": f"1. {step_name}", "curves": curves})
        
        stages.append({"name": "1. Final Reconstruction", "curves": stage1_data.get("curves", [])})
        stages.append({"name": "2. Layout Optimized", "curves": stage2_data.get("curves", [])})
        stages.append({"name": "3. Size Optimized", "curves": stage3_data.get("curves", [])})
        
        history = {
            "metadata": stage3_data.get("metadata", {}),
            "history": stage1_data.get("history", []),
            "stages": stages,
            "curves": stage3_data.get("curves", [])
        }
        
        print(f"[Export] Including {len(history['history'])} history snapshots + {len(stages)} stages")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"[Export] Saved pipeline history to {history_path}")
        
    except Exception as e:
        print(f"[Warning] Could not create pipeline history: {e}")
    
    print(f"\n{'='*60}")
    print("           PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Stage 1: {stage1_output}")
    print(f"Stage 2: {stage2_output}")
    print(f"Stage 3: {stage3_output}")
    print(f"History: {history_path}")
    print(f"{'='*60}")
    
    return 0
