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
   npz['bc_tags'] # shape (nely, nelx, nelz), int32; 1=fixed, 2=loaded, 3=passive void

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
   classifies skeleton voxels into endpoints (≤1 neighbour), body (exactly 2),
   and junctions (≥3), then traces edges between non-body voxels.  BC-tagged
   voxel clusters are consolidated into single nodes.
5. **Laplacian smoothing** — edge waypoints are smoothed in-place to remove
   staircase artefacts from the voxel grid.
6. **Post-processing** — collapse short edges (``--collapse_thresh``), prune
   short branch tips (``--prune_len``), simplify edge polylines via
   Ramer–Douglas–Peucker (``--rdp``), compute per-edge radii.
7. **Initial radius** — a uniform radius is assigned so that the total beam
   volume matches the solid volume:
   ``r₀ = sqrt(V_solid / (π · ΣLᵢ))``.
8. **[Hybrid only]** — two-signal zone classification, plate extraction,
   beam-plate joint creation
   (see :ref:`guides/pipeline_walkthrough:hybrid beam+plate mode`).
9. **Export** — :func:`src.pipelines.baseline_yin.reconstruct.export_to_json`
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
zones using a two-signal approach
(:func:`src.pipelines.baseline_yin.graph.classify_skeleton_post_thinning`):

* **Signal A** — skeleton set-difference: voxels present in the ``mode=3``
  skeleton but absent in the ``mode=0`` skeleton are plate candidates.
* **Signal B** — octant plane pattern count ≥ 4 with ≥ 3 foreground
  neighbours identifies surface-like voxels directly.

Combined candidates are filtered by connected-component size and global
linearity to reject noise.  Final classification:

* **Plate voxels** (``zone == 1``): surface-like topology
* **Beam voxels** (``zone == 2``): curve-like topology

Plate regions are expanded back to full thickness via morphological dilation
into the original solid
(:func:`src.pipelines.baseline_yin.plate_extraction.extract_plates_v2`),
smoothed with Taubin's λ/μ filter, and assigned EDT-based thickness
(``2 × EDT × pitch``).  Beam endpoints near a plate boundary are snapped to a
shared node by KD-tree lookup
(:func:`src.pipelines.baseline_yin.joint_creation.create_beam_plate_joints`).

Optimisation Stages 2 and 3 operate on beam radii and plate thicknesses
jointly under a shared volume constraint via the OC update.

Curved Beam Mode
-----------------

``--curved`` fits cubic Bézier curves through skeleton waypoints for smooth
geometry.  When enabled, the FEM assembler can use IGA Timoshenko curved-beam
elements (cubic Bernstein basis, 24-DOF, statically condensed to 12-DOF) for
beams with significant curvature, while short or straight beams continue to
use standard Euler–Bernoulli elements.  After each optimisation stage, control
points are re-fitted from updated node positions using
:func:`src.curves.spline.fit_cubic_bezier`.

Bézier control points are sanitised to enforce chord monotonicity
(``0 ≤ t₁ ≤ t₂ ≤ L``) and a perpendicular bulge limit (≤ 50 % of chord
length) to prevent self-intersecting curves.

External Mesh Input
---------------------

``--mesh_input <path>`` allows the pipeline to accept an external STL or OBJ
mesh instead of running Top3D.  The mesh is voxelised at the pitch specified
by ``--mesh_pitch`` (default 1.0 mm) using trimesh, producing an NPZ with a
binary ``rho`` field (no ``bc_tags``).  The skeleton reconstruction then
proceeds identically, with EDT-based radius assignment used by default since
no volume-matching target is available.

Symmetry Enforcement
----------------------

``--symmetry x|y|z`` enforces mirror symmetry about the specified axis.  The
skeleton is split at the symmetry plane, edges crossing the plane are split at
the intersection, one half is retained for optimisation, and the result is
reflected back.  Nodes on the symmetry plane are fixed in the normal direction
during layout optimisation to prevent drift.
