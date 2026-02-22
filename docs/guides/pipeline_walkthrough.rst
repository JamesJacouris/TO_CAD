.. _pipeline_walkthrough:

Pipeline Walkthrough
====================

This page walks through every stage of the ``run_pipeline.py`` orchestrator
in execution order, explaining inputs, outputs, and key parameters.

.. contents:: On this page
   :local:
   :depth: 2

Stage 0 — Topology Optimisation (Top3D)
-----------------------------------------

**Entry point**: ``run_top3d.py`` (called as subprocess from ``run_pipeline.py``
when ``--skip_top3d`` is *not* given).

**Algorithm**: Pure-Python port of Sigmund's 88-line MATLAB ``top3d`` code
(:mod:`src.optimization.top3d`).  Uses a SIMP material model with OC density
updates and a mesh-independence filter.

**Boundary conditions**: Set via ``--problem``:

* ``cantilever`` — fixed at ``x = 0``, point load at ``x = nelx``
* ``roof`` — four corner supports, central downward load
* ``bridge`` — bottom surface fixed, distributed top load
* ``tagged`` *(pipeline default)* — BC tags propagated from ``run_top3d.py``
  into the ``bc_tags`` array stored in the NPZ

**Key outputs** (stored in ``.npz``):

.. code-block:: python

   npz['rho']     # shape (nely, nelx, nelz), float64, density ∈ [0, 1]
   npz['bc_tags'] # shape (nely, nelx, nelz), int32; 1=fixed, 2=loaded

Stage 1 — Skeleton Reconstruction
------------------------------------

**Entry point**: :func:`src.pipelines.baseline_yin.reconstruct.reconstruct_npz`
(called directly from ``run_pipeline.py``; can also be used as a Python API).

**Sub-steps**:

1. **Threshold** — ``solid = rho > vol_thresh`` creates a binary voxel mask.
2. **EDT** — ``scipy.ndimage.distance_transform_edt(solid)`` pre-computes the
   Euclidean distance transform used by thinning and radius estimation.
3. **Thinning** — :func:`src.pipelines.baseline_yin.thinning.thin_grid_yin`
   iteratively removes simple points (Yin Definition 3.14) until a medial axis
   remains.  Mode selects which topology class is preserved:

   * ``mode=0`` — curve-preserving (beam-only path)
   * ``mode=3`` — surface+curve-preserving (hybrid path)

4. **Graph extraction** — :func:`src.pipelines.baseline_yin.graph.extract_graph`
   converts the skeleton voxels into a graph of nodes and edges.
5. **Post-processing** — prune short branches, collapse short edges, simplify
   edge polylines, compute per-edge radii.
6. **[Hybrid only]** — zone classification, plate extraction, beam-plate joint
   creation (see :ref:`guides/pipeline_walkthrough:hybrid beam+plate mode`).
7. **Export** — :func:`src.pipelines.baseline_yin.reconstruct.export_to_json`
   writes the graph + plate data to a ``.json`` file.

**Key parameters**:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``--vol_thresh``
     - 0.3
     - Density threshold; lower → more voxels survive
   * - ``--prune_len``
     - 2.0 mm
     - Remove branch tips shorter than this
   * - ``--collapse_thresh``
     - 2.0 mm
     - Merge nodes connected by edges shorter than this
   * - ``--rdp``
     - 1.0 mm
     - Ramer–Douglas–Peucker epsilon for polyline simplification
   * - ``--radius_mode``
     - ``uniform``
     - ``uniform`` = volume-matching; ``edt`` = geometric from EDT

Stage 2 — Size Optimisation
------------------------------

**Entry point**: :func:`src.optimization.size_opt.optimize_size`

Uses the Optimality Criteria (OC) method to resize beam cross-section radii
so that the total frame volume matches ``target_volume`` while minimising
compliance (maximising stiffness).

Iterates until the relative change in radii is below ``1e-3`` or
``--iters`` is reached.

Stage 3 — Layout Optimisation
--------------------------------

**Entry point**: :func:`src.optimization.layout_opt.optimize_layout`

Uses ``scipy.optimize.minimize`` with the L-BFGS-B method to move node
positions along compliance gradients, subject to:

* A move limit (``--limit``, default 5 mm per iteration)
* Design-domain box constraints (nodes stay inside the voxel bounding box)
* Node snapping (``--snap``) to merge nodes that drift within tolerance

Stages 2 and 3 are interleaved for ``--opt_loops`` iterations.  Each loop
runs *Size first, then Layout* to ensure volume constraint enforcement before
geometry update.

Hybrid Beam+Plate Mode
------------------------

Enabled with ``--hybrid``.

After thinning (``mode=3``), the skeleton is classified into beam and plate
zones using Yin's topological predicates
(:func:`src.pipelines.baseline_yin.graph.classify_skeleton_post_thinning`):

* **Plate voxels** (``zone == 1``): surface-like topology — high neighbour
  count, planar PCA shape
* **Beam voxels** (``zone == 2``): curve-like topology — low neighbour count,
  linear PCA shape

Plate regions are expanded back to full thickness via
:func:`src.pipelines.baseline_yin.plate_extraction.extract_plates_v2` and
their skeleton mid-surfaces are fitted with B-spline surfaces.  Beam endpoints
near a plate boundary are snapped to a shared node by
:func:`src.pipelines.baseline_yin.joint_creation.create_beam_plate_joints`.

Optimisation Stages 2 and 3 operate on **beam edges only**; plates are treated
as rigid and carried through unchanged.

Curved Beam Mode
-----------------

``--curved`` fits cubic Bézier curves through skeleton waypoints for smooth
visualisation.  FEM always uses straight-beam elements — ``--curved`` does not
change the structural model.  See :ref:`guides/pipeline_walkthrough:stage 1 — skeleton reconstruction`
for implementation notes.
