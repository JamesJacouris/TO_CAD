.. _quickstart:

Quick Start
===========

This page gets you from zero to a FreeCAD skeleton in four commands.

Step 1 — Run topology optimisation
------------------------------------

Optimise a 60 × 20 × 4 cantilever domain::

   python run_top3d.py \
       --nelx 60 --nely 20 --nelz 4 \
       --volfrac 0.3 --penal 3.0 --rmin 1.5 --max_loop 50 \
       --problem cantilever \
       --output output/hybrid_v2/quickstart_top3d.npz

Output: ``output/hybrid_v2/quickstart_top3d.npz`` (density field + BC tags).

Step 2 — Reconstruct the skeleton
-----------------------------------

Convert the density field to a beam graph::

   python run_pipeline.py \
       --skip_top3d \
       --top3d_npz output/hybrid_v2/quickstart_top3d.npz \
       --prune_len 2.0 --collapse_thresh 2.0 --rdp 1.0 \
       --radius_mode uniform \
       --output quickstart.json

Output: ``output/hybrid_v2/quickstart.json`` (full pipeline history + final graph).

Step 3 — (Optional) Open the 3-D preview
------------------------------------------

Add ``--visualize`` to either command above to open an interactive Open3D
window at each stage of processing.

Step 4 — Import into FreeCAD
------------------------------

Open FreeCAD, then run the macro::

   Macro → Macros… → freecad_reconstruct → Execute

Point the dialog at ``output/hybrid_v2/quickstart.json``.

.. tip::

   Running the full pipeline in one shot (Stages 0–3)::

      python run_pipeline.py \
          --nelx 60 --nely 20 --nelz 4 \
          --volfrac 0.3 \
          --opt_loops 2 \
          --output quickstart_optimised.json

   Omit ``--skip_top3d`` and the pipeline runs Stage 0 automatically.

.. tip::

   **External mesh input** — start from an existing STL or OBJ mesh instead
   of running Top3D::

      python run_pipeline.py \
          --mesh_input models/rocker_arm.stl \
          --mesh_pitch 0.5 \
          --output rocker_arm.json

   The mesh is voxelised at the given pitch and fed into the skeleton
   reconstruction pipeline.  No BC tags are generated; EDT-based radius
   assignment is used by default.

Typical output files
---------------------

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - File
     - Description
   * - ``<name>_top3d.npz``
     - Raw density array + BC tag array from Top3D
   * - ``<name>_1_reconstructed.json``
     - Stage 1 skeleton graph
   * - ``<name>_3_sized_loop1.json``
     - After size optimisation (loop 1)
   * - ``<name>_2_layout_loop1.json``
     - After layout optimisation (loop 1)
   * - ``<name>.json``
     - Final output (copy of last layout stage)
   * - ``<name>_history.json``
     - Full history for FreeCAD timeline playback
