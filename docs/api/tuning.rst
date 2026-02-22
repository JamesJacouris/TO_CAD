.. _api_tuning:

Tuning — ``src.tuning``
========================

Optuna-based parameter search for skeletonisation quality.

.. contents:: On this page
   :local:

pipeline\_runner
----------------

Subprocess wrapper that runs a single reconstruction trial with a given
parameter set and returns the output JSON path.

.. automodule:: src.tuning.pipeline_runner
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

metrics
-------

Geometry fidelity metric extraction from reconstruction output.

.. automodule:: src.tuning.metrics
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
