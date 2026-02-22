.. _cli_run_top3d:

``run_top3d.py`` — Topology Optimisation
==========================================

Standalone script that runs Python Top3D and writes a ``.npz`` density field.
Used as Stage 0 by ``run_pipeline.py``, or independently.

Synopsis
--------

.. code-block:: bash

   python run_top3d.py [OPTIONS]

Examples
--------

**Short cantilever (fast)**::

   python run_top3d.py \
       --nelx 60 --nely 20 --nelz 4 \
       --volfrac 0.3 --penal 3.0 --rmin 1.5 \
       --problem cantilever \
       --output output/hybrid_v2/cantilever_top3d.npz

**Roof structure (4-corner supports, central load)**::

   python run_top3d.py \
       --nelx 40 --nely 40 --nelz 20 \
       --volfrac 0.05 --penal 3.0 --rmin 1.5 --max_loop 100 \
       --load_fz -100.0 \
       --problem roof \
       --output output/hybrid_v2/roof_top3d.npz

**Rocker arm (dedicated script)**::

   python run_top3d_rocker_arm.py

   # Then reconstruct:
   python run_pipeline.py \
       --skip_top3d \
       --top3d_npz output/hybrid_v2/rocker_arm_top3d.npz \
       --vol_thresh 0.28 \
       --output rocker_arm.json

Argument reference
------------------

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--nelx``
     - 60
     - Elements in X
   * - ``--nely``
     - 20
     - Elements in Y
   * - ``--nelz``
     - 4
     - Elements in Z
   * - ``--volfrac``
     - 0.3
     - Target volume fraction
   * - ``--penal``
     - 3.0
     - SIMP penalisation exponent
   * - ``--rmin``
     - 1.5
     - Filter radius (elements)
   * - ``--max_loop``
     - 50
     - Max solver iterations
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
     - 0.0
     - Force X (N)
   * - ``--load_fy``
     - -1.0
     - Force Y (N)
   * - ``--load_fz``
     - 0.0
     - Force Z (N)
   * - ``--problem``
     - ``cantilever``
     - Problem type: ``cantilever``, ``roof``, ``bridge``, ``deck``
   * - ``--load_dist``
     - ``point``
     - ``point``, ``surface_top``, ``surface_bottom``
   * - ``--output``
     - ``python_top3d_result.npz``
     - Output file path

Problem types
-------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Problem
     - Boundary conditions
   * - ``cantilever``
     - All DOFs fixed at ``x=0``; point load at ``(load_x, load_y, load_z)``
   * - ``roof``
     - 4 corner columns (z=0 face) fixed; central downward load
   * - ``bridge``
     - Bottom surface (z=0) fixed; distributed load at top surface
   * - ``deck``
     - Same as bridge but with symmetric load points
