#!/usr/bin/env python3
"""
Automated parameter tuning using Optuna.

Optimizes skeletonization parameters (prune_len, collapse_thresh, rdp, radius_mode)
based on geometry fidelity metrics.
"""
import argparse
import os
import sys
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.tuning.pipeline_runner import run_reconstruction_trial, cleanup_trial_dir
from src.tuning.metrics import extract_metrics


class ParameterTuner:
    """
    Optuna-based parameter tuner for skeletonization.
    """
    def __init__(self, npz_path, vol_thresh=0.3, output_dir="output/tuning"):
        self.npz_path = npz_path
        self.vol_thresh = vol_thresh
        self.output_dir = output_dir
        self.trial_counter = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def objective(self, trial):
        """
        Optuna objective function.
        
        Minimizes composite score:
            score = volume_error + 0.1 * complexity_penalty + 0.2 * coverage_penalty
        """
        # Sample parameters
        params = {
            'prune_len': trial.suggest_float('prune_len', 0.5, 5.0),
            'collapse_thresh': trial.suggest_float('collapse_thresh', 1.0, 5.0),
            'rdp': trial.suggest_float('rdp', 0.5, 5.0),
            'radius_mode': trial.suggest_categorical('radius_mode', ['uniform', 'edt']),
            'vol_thresh': self.vol_thresh
        }
        
        # Create trial output directory
        trial_dir = os.path.join(self.output_dir, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Run reconstruction
        output_json, success = run_reconstruction_trial(
            self.npz_path,
            params,
            output_dir=trial_dir,
            timeout=60
        )
        
        if not success:
            # Return high penalty for failed trials
            return 1e6
        
        try:
            # Extract metrics
            metrics = extract_metrics(self.npz_path, output_json, self.vol_thresh)
            
            # Primary: Volume error
            volume_error = metrics['volume_error']
            
            # Secondary: Complexity penalty
            complexity = (metrics['num_nodes'] + 0.5 * metrics['num_edges']) / 100.0
            
            # Secondary: Coverage penalty (normalized)
            coverage = metrics['hausdorff_dist'] / metrics['domain_size'] if metrics['domain_size'] > 0 else 0.0
            
            # Composite score
            score = volume_error + 0.1 * complexity + 0.2 * coverage
            
            # Save metrics to trial directory
            metrics['params'] = params
            metrics['score'] = score
            with open(os.path.join(trial_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Report intermediate values for monitoring
            trial.set_user_attr('volume_error', volume_error)
            trial.set_user_attr('num_nodes', metrics['num_nodes'])
            trial.set_user_attr('num_edges', metrics['num_edges'])
            trial.set_user_attr('hausdorff_dist', metrics['hausdorff_dist'])
            
            return score
            
        except Exception as e:
            print(f"[Trial {trial.number}] Error extracting metrics: {e}")
            return 1e6
    
    def run_study(self, n_trials=100, study_name="param_tuning", storage_path=None):
        """
        Run Optuna study.
        
        Args:
            n_trials: Number of trials to run
            study_name: Name of the study
            storage_path: Optional SQLite database path for persistent storage
        """
        # Create storage
        if storage_path is None:
            storage_path = os.path.join(self.output_dir, "optuna_study.db")
        
        storage = f"sqlite:///{storage_path}"
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        print(f"Starting Optuna study: {study_name}")
        print(f"Storage: {storage_path}")
        print(f"Trials: {n_trials}")
        print(f"Input: {self.npz_path}")
        print("=" * 60)
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Print results
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best score: {study.best_value:.6f}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Save best params to JSON
        best_params_path = os.path.join(self.output_dir, "best_params.json")
        with open(best_params_path, 'w') as f:
            json.dump({
                'trial_number': study.best_trial.number,
                'score': study.best_value,
                'params': study.best_params,
                'user_attrs': study.best_trial.user_attrs
            }, f, indent=2)
        
        print(f"\nBest parameters saved to: {best_params_path}")
        
        # Generate visualizations
        try:
            import plotly
            
            fig_history = plot_optimization_history(study)
            fig_history.write_html(os.path.join(self.output_dir, "optimization_history.html"))
            
            fig_importance = plot_param_importances(study)
            fig_importance.write_html(os.path.join(self.output_dir, "param_importances.html"))
            
            print(f"Visualizations saved to: {self.output_dir}")
        except ImportError:
            print("[Warning] Plotly not installed. Skipping visualizations.")
        
        return study


def main():
    parser = argparse.ArgumentParser(
        description="Automated parameter tuning for skeletonization"
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to Top3D .npz file"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/tuning",
        help="Output directory for results"
    )
    parser.add_argument(
        "--vol_thresh",
        type=float,
        default=0.3,
        help="Volume threshold for NPZ (default: 0.3)"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="param_tuning",
        help="Optuna study name"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.npz_path):
        print(f"Error: NPZ file not found: {args.npz_path}")
        return 1
    
    # Create tuner
    tuner = ParameterTuner(
        npz_path=args.npz_path,
        vol_thresh=args.vol_thresh,
        output_dir=args.output_dir
    )
    
    # Run study
    study = tuner.run_study(
        n_trials=args.trials,
        study_name=args.study_name
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
