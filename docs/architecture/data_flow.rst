.. _data_flow:

Data Flow
=========

This page traces how data is transformed at each stage.

.. contents:: On this page
   :local:

Stage 0 → Stage 1: NPZ format
-------------------------------

``run_top3d.py`` writes a NumPy ``.npz`` archive:

.. code-block:: python

   np.savez(output_path,
       rho=rho,        # shape (nely, nelx, nelz), float64 ∈ [0, 1]
       bc_tags=bc_tags # shape (nely, nelx, nelz), int32; 1=fixed, 2=loaded
   )

**Index order**: Top3D uses ``(nely, nelx, nelz)`` (row-major MATLAB convention).
When converting to world coordinates, indices are reordered to ``[nelx, nely, nelz]``
by taking ``indices[:, [1, 0, 2]]``.

Stage 1 output: JSON schema
----------------------------

:func:`src.pipelines.baseline_yin.reconstruct.export_to_json` produces a
``.json`` file.  See :ref:`architecture/json_schema:json output schema` for the
full field reference.

Key top-level keys:

.. code-block:: text

   {
     "metadata": { "pitch": 1.0, "units": "mm", "vol_thresh": 0.3, ... },
     "graph":    { "nodes": [...], "edges": [...], "node_tags": {...} },
     "curves":   [...],
     "plates":   [...],
     "joints":   [...],
     "history":  [...]
   }

Stages 2 & 3: in-memory arrays
--------------------------------

Size and layout optimisation operate on NumPy arrays extracted from the JSON:

.. code-block:: python

   nodes  = np.array(data['graph']['nodes'])   # (N, 3) float64, mm
   edges  = np.array([[e[0], e[1]] for e in data['graph']['edges']], dtype=int)  # (M, 2)
   radii  = np.array([e[2] for e in data['graph']['edges']])  # (M,) float64, mm

After each stage the updated graph is written back to a new ``.json`` file.

Pipeline history object
------------------------

``run_pipeline.py`` assembles a history object for FreeCAD:

.. code-block:: text

   {
     "metadata": { ... },
     "history":  [ { "type": "voxels", "step": "1_Initial_Voxels", ... }, ... ],
     "stages":   [
       { "name": "1. Reconstructed", "curves": [...], "plates": [...] },
       { "name": "Size Loop 1",      "curves": [...], "plates": [...] },
       { "name": "Layout Loop 1",    "curves": [...], "plates": [...] }
     ],
     "curves":  [...],
     "plates":  [...],
     "joints":  [...],
     "graph":   { ... }
   }

The ``history`` list contains per-voxel snapshots (type ``"voxels"``) and
per-graph snapshots (type ``"graph"``).  The ``stages`` list contains the full
geometry at each optimisation stage for the FreeCAD timeline feature.

External mesh input: NPZ format
---------------------------------

When ``--mesh_input`` is given, the external STL/OBJ mesh is voxelised by
:mod:`src.mesh_import.mesh_voxelizer` at the specified ``--mesh_pitch``.  The
resulting NPZ contains only ``rho`` (binary, float32); no ``bc_tags`` are
generated.  Downstream modules detect the absence of ``bc_tags`` and skip
BC-related processing (no node tagging, no tagged-problem BCs).

.. code-block:: python

   np.savez(output_path,
       rho=rho  # shape (nely, nelx, nelz), float32, binary {0.0, 1.0}
   )

Symmetry data flow
--------------------

When ``--symmetry`` is specified, the symmetry module
(:mod:`src.optimization.symmetry`) operates on the extracted skeleton *before*
optimisation Stages 2/3:

1. Edges crossing the symmetry plane are split at the plane intersection.
2. One half of the skeleton is retained.
3. After optimisation, the result is reflected and merged back.
4. Nodes on the symmetry plane are constrained in the plane-normal direction
   during layout optimisation.

BC tag propagation
-------------------

BC tags flow from Top3D through the entire pipeline:

1. ``run_top3d.py`` sets ``bc_tags[voxel] = 1`` (fixed) or ``2`` (loaded).
2. :func:`src.pipelines.baseline_yin.thinning.thin_grid_yin` preserves tagged
   voxels from deletion.
3. :func:`src.pipelines.baseline_yin.graph.extract_graph` assigns ``node_tags``
   by majority vote of voxel tags at each node cluster.
4. :class:`src.problems.tagged_problem.TaggedProblem` reads ``node_tags`` from
   the JSON and applies fixed DOFs / loads accordingly.
