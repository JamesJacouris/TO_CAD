.. _api_pipelines:

Pipelines — ``src.pipelines.baseline_yin``
===========================================

.. contents:: On this page
   :local:
   :depth: 1

reconstruct
-----------

Main Stage 1 orchestrator.  Loads a Top3D ``.npz``, runs thinning and graph
extraction, and exports the result to ``.json``.

.. automodule:: src.pipelines.baseline_yin.reconstruct
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

thinning
--------

Yin 3-D parallel medial-axis thinning.

.. automodule:: src.pipelines.baseline_yin.thinning
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

topology
--------

Low-level topological predicates (Numba-accelerated).

.. automodule:: src.pipelines.baseline_yin.topology
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

graph
-----

Skeleton → graph extraction and zone classification.

.. automodule:: src.pipelines.baseline_yin.graph
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

postprocessing
--------------

Graph refinement: pruning, edge collapse, radius computation.

.. automodule:: src.pipelines.baseline_yin.postprocessing
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

plate\_extraction
-----------------

Plate region recovery for hybrid beam+plate mode.

.. automodule:: src.pipelines.baseline_yin.plate_extraction
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

joint\_creation
---------------

Beam-plate interface joint snapping.

.. automodule:: src.pipelines.baseline_yin.joint_creation
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

surface\_fitting
----------------

B-spline plate mid-surface fitting.

.. automodule:: src.pipelines.baseline_yin.surface_fitting
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

visualization
-------------

Open3D and Matplotlib visualisation utilities.

.. note::

   Visualisation functions require ``open3d``; they are mocked during doc
   builds.  See :ref:`guides/installation:python dependencies` for setup.

.. automodule:: src.pipelines.baseline_yin.visualization
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

mesh\_voxelizer
----------------

External STL/OBJ mesh voxelisation for the ``--mesh_input`` pipeline path.

.. automodule:: src.mesh_import.mesh_voxelizer
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
