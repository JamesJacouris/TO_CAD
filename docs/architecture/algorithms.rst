.. _algorithms:

Algorithms
==========

.. contents:: On this page
   :local:
   :depth: 2

Top3D — Topology Optimisation
-------------------------------

:mod:`src.optimization.top3d` implements the 3-D extension of Sigmund's
88-line MATLAB ``top`` code.

**Material model**: SIMP (Solid Isotropic Material with Penalisation)

.. math::

   E(\rho_e) = E_{\min} + \rho_e^p (E_0 - E_{\min})

where :math:`p` is the penalisation exponent (``--penal``, default 3).

**Density update**: Optimality Criteria

.. math::

   \rho_e^{new} = \rho_e \cdot B_e^{\eta}

where :math:`B_e = -\partial c/\partial \rho_e / (\lambda \partial V/\partial \rho_e)`
and :math:`\lambda` is found by bisection to satisfy the volume constraint.

**Mesh-independence filter**: Weighted average over a ball of radius ``--rmin``

.. math::

   \hat{\rho}_e = \frac{\sum_{f} H_{ef} \rho_f}{\sum_f H_{ef}}, \quad
   H_{ef} = \max(0, r_{\min} - \Delta(e,f))

**Element**: 8-node hexahedral H8 with trilinear shape functions.
Stiffness matrix pre-computed by ``lk_H8()`` and assembled via vectorised
indexing.  Sparse system solved with ``scipy.sparse.linalg.spsolve``.

Yin Medial-Axis Thinning
--------------------------

:mod:`src.pipelines.baseline_yin.thinning` implements the 3-D parallel thinning
algorithm by Yin (2002).

**Core idea**: Iteratively remove *simple points* (points whose removal does
not change the topology of the binary object) until no more can be removed.

**Simple point** (Yin Definition 3.14): A voxel is simple if the 26-connected
foreground and the 6-connected background are both unchanged (connected-component
count preserved) after its removal.  Tested by
:func:`src.pipelines.baseline_yin.topology.is_simple_point`.

**Directional sweeps**: Each iteration sweeps in 6 directions (±x, ±y, ±z).
Within each direction, voxels not tagged as BC nodes are candidates if they are:

* Simple (topological predicate)
* Not an end voxel (``is_end_voxel``) — endpoints of curves are preserved
* [mode=1/3] Not a surface voxel (``is_surface_point``) — plates are preserved

**Numba acceleration**: Critical loops use ``@njit(parallel=True)`` for
per-voxel parallelism.

**Thinning modes**:

.. list-table::
   :widths: 10 20 70
   :header-rows: 1

   * - Mode
     - Name
     - Preserved topology
   * - 0
     - Curve-preserving
     - Curve endpoints; plates collapsed to curves
   * - 1
     - Surface-preserving
     - Surface voxels; curves collapsed to surfaces
   * - 3
     - Hybrid
     - Both curve endpoints and surface voxels

Graph Extraction
-----------------

:func:`src.pipelines.baseline_yin.graph.extract_graph` converts the
skeleton to a graph:

1. **Node detection**: skeleton voxels with ≠ 2 neighbours are junctions or
   endpoints → become graph nodes.
2. **Clustering**: nearby nodes within ``pitch`` are merged into a single node
   using union-find.
3. **Edge tracing**: BFS from each node through degree-2 voxels traces the
   skeleton path between nodes; waypoints are stored as the edge polyline.
4. **BC tag assignment**: each node inherits the dominant BC tag of its cluster.

Post-thinning Zone Classification
-----------------------------------

For hybrid mode, :func:`src.pipelines.baseline_yin.graph.classify_skeleton_post_thinning`
classifies each skeleton voxel using topological properties:

* **Plate voxel**: ≥ ``junction_thresh`` 26-connected skeleton neighbours AND
  average connectivity in its local neighbourhood ≥ ``min_avg_neighbors``
* **Beam voxel**: otherwise

Connected components of plate voxels below ``min_plate_size`` are reclassified
as beams.

Frame FEA
----------

:func:`src.optimization.fem.solve_frame` assembles a 3-D Euler-Bernoulli
frame stiffness matrix.

**Element**: 2-node 3-D beam with 6 DOF per node (3 translations + 3 rotations).
Local stiffness :math:`K_e` (12×12) is computed analytically from beam length,
Young's modulus ``E``, and cross-section area :math:`A = \pi r^2`.

**Assembly**: Global :math:`K = \sum_e T_e^T K_e T_e` where :math:`T_e` is the
12×12 rotation matrix aligning the local beam axis to the global frame.

**Solve**: ``scipy.sparse.linalg.spsolve(K_{ff}, f_{f})`` where subscript
:math:`f` denotes free (non-constrained) DOFs.

**Compliance**: :math:`c = f^T u` (scalar, minimised by optimisers).

**Gradient**: :math:`\partial c / \partial r_e = -2 \pi r_e L_e E (u_e^T k_e u_e) / (EA)^2`
(chain rule from cross-section area dependence).

Optimality Criteria Sizing
----------------------------

:func:`src.optimization.size_opt.optimize_size` uses OC updates identical
in form to Top3D but operating on beam radii instead of voxel densities:

.. math::

   r_e^{new} = \text{clip}\!\left( r_e \cdot \left(\frac{-\partial c / \partial r_e}{\lambda \partial V / \partial r_e}\right)^{\eta}, r_{\min}, r_{\max} \right)

:math:`\lambda` is found by bisection to satisfy the total volume constraint.
Move limits of ±20% are applied each iteration.

L-BFGS-B Layout Optimisation
------------------------------

:func:`src.optimization.layout_opt.optimize_layout` minimises compliance
with respect to free node positions:

.. math::

   \min_{\mathbf{x}} \; c(\mathbf{x}) \quad \text{s.t.} \; \mathbf{x}_{\min} \le \mathbf{x} \le \mathbf{x}_{\max}

Gradient :math:`\partial c / \partial x_n` is assembled analytically from
per-element sensitivities.  Fixed (BC) nodes and loaded nodes have their
positions frozen.

After convergence, nodes within ``snap_dist`` of each other are merged to
maintain clean graph topology.
