.. _cli_tune_parameters:

``tune_parameters.py`` — Parameter Tuning
==========================================

Uses `Optuna <https://optuna.org/>`_ to search for the best skeletonisation
parameters for a given NPZ file.

Synopsis
--------

.. code-block:: bash

   python tune_parameters.py NPZ_PATH [OPTIONS]

Examples
--------

**100-trial search**::

   python tune_parameters.py \
       output/hybrid_v2/matlab_replicated_top3d.npz \
       --trials 100

**Short 10-trial validation run**::

   python tune_parameters.py \
       output/hybrid_v2/matlab_replicated_top3d.npz \
       --trials 10 \
       --output_dir output/tuning_test

Argument reference
------------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``npz_path`` (positional)
     - —
     - Path to Top3D ``.npz`` density field
   * - ``--trials``
     - 100
     - Number of Optuna trials
   * - ``--output_dir``
     - ``output/tuning``
     - Directory for trial outputs

Search space
------------

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Search range
     - Maps to
   * - ``prune_len``
     - 0.5 – 5.0 (float)
     - ``--prune_len``
   * - ``collapse_thresh``
     - 1.0 – 5.0 (float)
     - ``--collapse_thresh``
   * - ``rdp``
     - 0.5 – 5.0 (float)
     - ``--rdp``
   * - ``radius_mode``
     - ``uniform`` / ``edt``
     - ``--radius_mode``

Objective function
------------------

Minimises a composite geometry fidelity score:

.. code-block:: text

   score = volume_error + 0.1 × complexity_penalty + 0.2 × coverage_penalty

where:

* ``volume_error`` — relative error between frame volume and target volume
* ``complexity_penalty`` — normalised edge count (penalises over-complex graphs)
* ``coverage_penalty`` — skeleton voxel coverage fraction

Best parameters are printed to stdout and saved to the Optuna study database.

Internals
---------

Tuning uses :mod:`src.tuning.pipeline_runner` (subprocess wrapper for
:func:`src.pipelines.baseline_yin.reconstruct.reconstruct_npz`) and
:mod:`src.tuning.metrics` (geometry fidelity extraction).
