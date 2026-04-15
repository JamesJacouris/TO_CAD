.. _api_curves:

Curves — ``src.curves``
========================

Cubic Bézier curve utilities used for smooth beam geometry and, optionally,
curved-beam finite element analysis.

.. note::

   When ``--curved`` is enabled, beams with significant curvature can use an
   IGA Timoshenko element with cubic Bernstein shape functions.  Short or
   straight beams continue to use standard Euler–Bernoulli elements.  Control
   points are sanitised to enforce chord monotonicity and a perpendicular
   bulge limit.  See :ref:`architecture/algorithms:curved beam element (iga timoshenko)` for
   the formulation.

.. automodule:: src.curves.spline
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
