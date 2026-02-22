.. _testing:

Testing & Validation
=====================

The project does not yet have a formal test suite.  The following manual
validation commands serve as regression checks.

Smoke test — import check
--------------------------

::

   python -c "
   from src.optimization.top3d import Top3D
   from src.pipelines.baseline_yin.reconstruct import reconstruct_npz
   from src.optimization.size_opt import optimize_size
   from src.optimization.layout_opt import optimize_layout
   from src.problems import load_problem_config
   print('All imports OK')
   "

Beam-only regression
---------------------

Using an existing NPZ (replace path as needed)::

   python run_pipeline.py \
       --skip_top3d \
       --top3d_npz output/hybrid_v2/matlab_replicated_top3d.npz \
       --output regression_beam.json \
       --opt_loops 1

Expected: pipeline prints ``PIPELINE COMPLETE`` with no tracebacks.

Hybrid regression
------------------

::

   python run_pipeline.py \
       --skip_top3d \
       --top3d_npz output/hybrid_v2/matlab_replicated_top3d.npz \
       --hybrid \
       --output regression_hybrid.json

Expected: ``PIPELINE COMPLETE (Unoptimized Hybrid)`` with plate count ≥ 1.

Parameter tuning
-----------------

::

   python tune_parameters.py \
       output/hybrid_v2/matlab_replicated_top3d.npz \
       --trials 10 \
       --output_dir output/tuning_test

Expected: Optuna prints trial results and saves best parameters.

Checking the docs build
------------------------

::

   cd docs && make clean html 2>&1 | grep -E "(WARNING|ERROR|error)"

A clean build should report zero errors and minimal warnings.
