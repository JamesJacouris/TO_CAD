.. _api_reference:

API Reference
=============

This section documents every public module in the ``src/`` package tree.
Pages are organised by package, mirroring the repository layout.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Page
     - Contents
   * - :doc:`pipelines`
     - ``src.pipelines.baseline_yin.*`` — reconstruction, thinning, graph,
       postprocessing, plates, joints, surface fitting, visualisation
   * - :doc:`optimization`
     - ``src.optimization.*`` — Top3D solver, FEA, size opt, layout opt
   * - :doc:`problems`
     - ``src.problems.*`` — boundary condition problem configs
   * - :doc:`curves`
     - ``src.curves.spline`` — cubic Bézier fitting
   * - :doc:`export`
     - ``src.export.freecad_reconstruct`` — FreeCAD macro
   * - :doc:`tuning`
     - ``src.tuning.*`` — parameter search pipeline runner and metrics

Docstring conventions
----------------------

Public functions use **Google-style** docstrings parsed by
``sphinx.ext.napoleon``.  Example::

   def foo(x: int) -> float:
       """One-line summary.

       Args:
           x: Description of x.

       Returns:
           Description of return value.

       Raises:
           ValueError: If x is negative.
       """
