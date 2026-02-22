.. _cli_run_pipeline:

``run_pipeline.py`` — Main Pipeline
=====================================

``run_pipeline.py`` is the primary entry point.  It orchestrates all four
pipeline stages and writes the history JSON consumed by the FreeCAD macro.

Synopsis
--------

.. code-block:: bash

   python run_pipeline.py [OPTIONS]

Examples
--------

**Full beam-only run (Stages 0–3)**::

   python run_pipeline.py \
       --nelx 60 --nely 20 --nelz 4 \
       --volfrac 0.3 \
       --opt_loops 2 \
       --output cantilever.json

**Skip Stage 0 (reuse existing NPZ)**::

   python run_pipeline.py \
       --skip_top3d \
       --top3d_npz output/hybrid_v2/my_top3d.npz \
       --output my_result.json

**Hybrid beam+plate mode**::

   python run_pipeline.py \
       --skip_top3d \
       --top3d_npz output/hybrid_v2/roof_top3d.npz \
       --hybrid \
       --min_plate_size 8 --flatness_ratio 7 \
       --output roof_hybrid.json

**With interactive visualisation**::

   python run_pipeline.py \
       --skip_top3d --top3d_npz output/hybrid_v2/my.npz \
       --visualize --output debug.json

Argument reference
------------------

Top3D Design Domain
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 12 63
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--nelx``
     - 60
     - Elements in X (domain length)
   * - ``--nely``
     - 20
     - Elements in Y (domain height)
   * - ``--nelz``
     - 4
     - Elements in Z (domain depth)
   * - ``--volfrac``
     - 0.3
     - Target volume fraction (0–1)
   * - ``--penal``
     - 3.0
     - SIMP penalisation exponent
   * - ``--rmin``
     - 1.5
     - Density filter radius (elements)
   * - ``--max_loop``
     - 50
     - Max Top3D iterations

Load Definition
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 12 63
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--load_x``
     - ``nelx``
     - Load node X index
   * - ``--load_y``
     - ``nely``
     - Load node Y index
   * - ``--load_z``
     - ``nelz/2``
     - Load node Z index
   * - ``--load_fx``
     - —
     - Force X component (N)
   * - ``--load_fy``
     - —
     - Force Y component (N)
   * - ``--load_fz``
     - —
     - Force Z component (N)
   * - ``--load_dist``
     - ``point``
     - Distribution: ``point``, ``surface_top``, ``surface_bottom``

Skeletonisation
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 12 63
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--pitch``
     - 1.0
     - Voxel size in mm
   * - ``--max_iters``
     - 50
     - Max thinning iterations
   * - ``--vol_thresh``
     - 0.3
     - Density threshold for binarisation
   * - ``--prune_len``
     - 2.0
     - Prune branch tips shorter than X mm
   * - ``--collapse_thresh``
     - 2.0
     - Collapse edges shorter than X mm
   * - ``--rdp``
     - 1.0
     - RDP polyline simplification epsilon (0 = off)
   * - ``--radius_mode``
     - ``uniform``
     - ``uniform`` = volume-matching; ``edt`` = geometric from EDT

Hybrid Beam+Plate Mode
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--hybrid``
     - off
     - Enable beam+plate hybrid pipeline
   * - ``--detect_plates``
     - ``auto``
     - ``auto``, ``off``, or ``force``
   * - ``--plate_mode``
     - ``bspline``
     - Plate representation: ``bspline``, ``voxel``, ``mesh``
   * - ``--plate_thickness_ratio``
     - 0.15
     - Max plate half-thickness as fraction of domain diagonal
   * - ``--min_plate_size``
     - 4
     - Min skeleton voxels to classify as plate
   * - ``--flatness_ratio``
     - 3.0
     - PCA eigenvalue ratio threshold for planarity
   * - ``--junction_thresh``
     - 4
     - Min neighbour count to classify as junction/plate
   * - ``--min_avg_neighbors``
     - 3.0
     - Min average neighbour count for plate classification

Optimisation
~~~~~~~~~~~~~

.. list-table::
   :widths: 25 12 63
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--optimize``
     - off
     - Enable Stages 2 & 3 (auto-enabled for beam-only)
   * - ``--opt_loops``
     - 2
     - Size + Layout iteration loops
   * - ``--iters``
     - 50
     - Max iterations per optimisation stage
   * - ``--limit``
     - 5.0
     - Move limit for layout optimisation (mm)
   * - ``--snap``
     - 5.0
     - Node snap distance for post-layout merging (mm)
   * - ``--prune_opt_thresh``
     - 0.0
     - Post-opt pruning: remove edges with radius < X × max_radius
   * - ``--problem``
     - ``tagged``
     - BC problem config: ``tagged``, ``cantilever``, ``rocker_arm``

Curved Beams
~~~~~~~~~~~~~

.. list-table::
   :widths: 25 12 63
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--curved``
     - off
     - Fit cubic Bézier curves to skeleton edges (geometry only; FEM always uses straight beams)

Output & Control
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 12 63
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--output``
     - ``full_control_beam.json``
     - Output filename (placed in ``--output_dir``)
   * - ``--output_dir``
     - ``output/hybrid_v2``
     - Output directory
   * - ``--visualize``
     - off
     - Open interactive Open3D windows at each stage
   * - ``--skip_top3d``
     - off
     - Skip Stage 0; use ``--top3d_npz``
   * - ``--top3d_npz``
     - —
     - Path to existing ``.npz`` (required with ``--skip_top3d``)
