"""
Pipeline runner wrapper for parameter tuning.

Runs Stage 1 (Reconstruction) subprocess and handles I/O.
"""
import subprocess
import sys
import os
import tempfile
import json


def run_reconstruction_trial(npz_path, params, output_dir=None, timeout=60):
    """
    Run Stage 1 reconstruction with given parameters.
    
    Args:
        npz_path: Path to Top3D .npz file
        params: dict with keys: prune_len, collapse_thresh, rdp, radius_mode, vol_thresh
        output_dir: Optional output directory (creates temp dir if None)
        timeout: Maximum runtime in seconds
    
    Returns:
        output_json_path: Path to Stage 1 JSON output
        success: bool
    """
    # Create temp output if needed
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='optuna_trial_')
        output_dir = temp_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_json = os.path.join(output_dir, "reconstruction.json")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reconstruct_script = os.path.join(script_dir, "src/pipelines/baseline_yin/reconstruct.py")
    
    # Build command
    cmd = [
        sys.executable,
        reconstruct_script,
        npz_path,
        output_json,
        "--prune_len", str(params['prune_len']),
        "--collapse_thresh", str(params['collapse_thresh']),
        "--rdp_epsilon", str(params['rdp']),
        "--radius_mode", params['radius_mode'],
        "--vol_thresh", str(params.get('vol_thresh', 0.3))
    ]
    
    try:
        # Run subprocess with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=script_dir
        )
        
        # Check if successful
        success = result.returncode == 0 and os.path.exists(output_json)
        
        if not success:
            print(f"[Trial Failed] Return code: {result.returncode}")
            if result.stderr:
                print(f"[Stderr]: {result.stderr[:500]}")
        
        return output_json, success
        
    except subprocess.TimeoutExpired:
        print(f"[Trial Failed] Timeout after {timeout}s")
        return None, False
    except Exception as e:
        print(f"[Trial Failed] Exception: {e}")
        return None, False


def cleanup_trial_dir(trial_dir):
    """
    Clean up temporary trial directory.
    """
    import shutil
    try:
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)
    except Exception as e:
        print(f"[Warning] Could not cleanup {trial_dir}: {e}")
