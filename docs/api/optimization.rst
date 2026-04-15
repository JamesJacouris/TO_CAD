.. _api_optimization:

Optimization — ``src.optimization``
=====================================

.. contents:: On this page
   :local:
   :depth: 1

top3d
-----

Pure-Python 3-D topology optimiser (SIMP + OC + mesh-independence filter).

.. automodule:: src.optimization.top3d
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

fem
---

3-D frame FEA with straight Euler–Bernoulli and curved IGA Timoshenko beam
elements.  Curved elements use cubic Bernstein shape functions with static
condensation to a 12-DOF interface.

.. automodule:: src.optimization.fem
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

size\_opt
---------

Optimality Criteria beam radius (cross-section) optimisation.

.. automodule:: src.optimization.size_opt
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

layout\_opt
-----------

L-BFGS-B node position layout optimisation.

.. automodule:: src.optimization.layout_opt
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

symmetry
--------

Mirror-half skeleton symmetry enforcement for layout optimisation.

.. automodule:: src.optimization.symmetry
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
