.. _freecad_export:

FreeCAD Export
==============

The FreeCAD macro (:mod:`src.export.freecad_reconstruct`) reads the pipeline
``.json`` output and creates 3-D CAD geometry inside a FreeCAD document.

Geometry created
-----------------

* **Beams** — swept circular tubes (one per edge), with hemispherical end caps
  at every node.  Beams are coloured with a radius heatmap (blue → green →
  yellow → red) where blue = thinnest and red = thickest.
* **Curved beams** — ``Part.BezierCurve`` piped through the cross-section
  circle (only when the JSON contains ``ctrl_pts`` entries).
* **Plates** — polygon faces built from the plate vertex lists, or B-spline
  surfaces when ``bspline_surface`` data is present.  Curved plates use
  offset mid-surfaces along vertex normals.
* **Fused body** — a single ``Fused_Body`` solid created by batched
  ``multiFuse`` (groups of 30) and ``removeSplitter()`` of all beam and plate
  shapes, suitable for direct STEP export.
* **History timeline** — one FreeCAD group per pipeline stage so you can
  toggle visibility and compare before/after.
* **Density legend** — for the initial-voxels stage, 10 coloured reference
  cubes (red → green) are placed below the model.

Running the macro
------------------

1. Open FreeCAD (0.21+).
2. **Macro → Macros…** → Browse to ``src/export/freecad_reconstruct.py`` →
   **Execute**.
3. A file-picker dialog opens; select your ``<name>_history.json`` file.
4. Wait for geometry creation.  Large models may take 30–60 s.

.. tip::

   The macro batches GUI updates every 20 objects to avoid freezing.  If the
   FreeCAD window seems unresponsive, wait for the console to print ``Done``.

Supported JSON fields
----------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Field
     - Description
   * - ``metadata.pitch``
     - Voxel size in mm; sets the global scale
   * - ``graph.nodes``
     - Node positions ``[[x, y, z], …]``
   * - ``graph.edges``
     - Edge list ``[[u, v, r], …]`` (u, v = node indices; r = radius mm)
   * - ``curves``
     - Per-edge visualisation data (straight or Bézier)
   * - ``plates``
     - Plate geometry (vertices, connectivity, optional B-spline surface)
   * - ``history``
     - Voxel snapshots at each processing stage
   * - ``stages``
     - Graph states at each optimisation loop

Known limitations
------------------

* Mid-surface plate geometry requires the Open3D mesh library, which is not
  available inside the FreeCAD Python interpreter.  Plates fall back to
  flat-face polygon meshes.
* The fused body uses per-beam Boolean unions and ``removeSplitter()``.  These
  OCC operations can be slow for very large models (> 2000 beams) and may fail
  on degenerate geometry; individual beams/plates are always available as a
  fallback.
* Very large models (> 5000 beams) may exceed FreeCAD's undo-history buffer;
  save before running.
