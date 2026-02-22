.. _json_schema:

JSON Output Schema
==================

Every stage of the pipeline writes or reads a ``.json`` file.  This page
documents the canonical schema.

Top-level structure
--------------------

.. code-block:: text

   {
     "metadata": { ... },
     "graph":    { ... },
     "curves":   [ ... ],
     "plates":   [ ... ],
     "joints":   [ ... ],
     "history":  [ ... ]
   }

``metadata``
~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``method``
     - str
     - ``"Baseline Yin"``
   * - ``pitch``
     - float
     - Voxel size in mm
   * - ``units``
     - str
     - ``"mm"``
   * - ``target_volume``
     - float
     - Total solid volume (mm³) from Stage 0
   * - ``design_bounds``
     - [[x,y,z],[x,y,z]]
     - Axis-aligned bounding box of the domain
   * - ``vol_thresh``
     - float
     - Density threshold used to binarise the NPZ
   * - ``plate_mode``
     - str
     - ``"bspline"``, ``"voxel"``, or ``"mesh"``
   * - ``load_force``
     - [fx, fy, fz]
     - Applied load vector (from CLI ``--load_f*``)

``graph``
~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``nodes``
     - ``[[x, y, z], …]``
     - Node world positions (mm), length N
   * - ``edges``
     - ``[[u, v, r], …]``
     - Edges: u and v are 0-based node indices; r is radius (mm)
   * - ``node_tags``
     - ``{"i": tag, …}``
     - Integer tag per node: 1 = fixed, 2 = loaded, 0 = free

``curves``
~~~~~~~~~~~

One entry per edge, in the same order as ``graph.edges``.

**Straight beam** (no ``--curved``):

.. code-block:: text

   { "points": [[x, y, z, r], …] }

**Curved beam** (``--curved``):

.. code-block:: text

   {
     "ctrl_pts": [[x1,y1,z1], [x2,y2,z2]],
     "points":   [[x, y, z, r], …],
     "radius":   r
   }

``plates``
~~~~~~~~~~~

Each plate entry:

.. code-block:: text

   {
     "id":                   0,
     "voxels":               [[x,y,z], …],
     "vertices":             [[x,y,z], …],
     "faces":                [[i,j,k], …],
     "thickness":            [t0, t1, …],
     "connection_node_ids":  [n0, n1, …],
     "bspline_surface":      { "grid_u": […], "grid_v": […], "points": […] }
   }

``joints``
~~~~~~~~~~~

.. code-block:: text

   {
     "plate_id":  0,
     "node_id":   3,
     "location":  [x, y, z],
     "direction": [dx, dy, dz],
     "radius":    r,
     "type":      "fillet"
   }

``history``
~~~~~~~~~~~~

A list of snapshots, one per processing step.  Two types:

**Voxel snapshot** (initial solid, skeleton, zone classification):

.. code-block:: text

   {
     "type":   "voxels",
     "step":   "1_Initial_Voxels",
     "points": [[x,y,z], …],
     "colors": [[r,g,b], …]
   }

**Graph snapshot** (after each postprocessing step):

.. code-block:: text

   {
     "type":   "graph",
     "step":   "3_Raw_Graph",
     "nodes":  [[x,y,z], …],
     "edges":  [[u, v, r, [waypoints…]], …],
     "plates": [ … ]
   }
