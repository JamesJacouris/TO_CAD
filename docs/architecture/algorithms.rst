.. _algorithms:

Algorithms
==========

.. contents:: On this page
   :local:
   :depth: 2

Top3D — Topology Optimisation (Stage 0)
----------------------------------------

:mod:`src.optimization.top3d` implements the 3-D extension of Sigmund's
88-line MATLAB ``top`` code, extended with BC tag generation, a
conjugate-gradient solver, and convergence history tracking.

**Material model**: SIMP (Solid Isotropic Material with Penalisation)

.. math::

   E(\rho_e) = E_{\min} + \rho_e^p (E_0 - E_{\min})

where :math:`p` is the penalisation exponent (``--penal``, default 3),
:math:`E_0` is the solid modulus, and :math:`E_{\min} = 10^{-9}`.

**Objective**: Minimise compliance :math:`C = \mathbf{F}^\top \mathbf{u}`
subject to a volume fraction limit :math:`V_f`:

.. math::

   \min_{\boldsymbol{\rho}} \; C = \mathbf{F}^\top \mathbf{u}
   \quad \text{s.t.} \quad
   \mathbf{K}\mathbf{u} = \mathbf{F}, \quad
   \frac{1}{V_0}\sum_e \rho_e v_e \le V_f, \quad
   0 \le \rho_e \le 1

**Sensitivity**:

.. math::

   \frac{\partial C}{\partial \rho_e}
   = -p\,(E_0 - E_{\min})\,\rho_e^{p-1}\,
     \mathbf{u}_e^\top \mathbf{K}_E \mathbf{u}_e

where :math:`\mathbf{K}_E` is the unit-modulus element stiffness matrix.
Sensitivities are averaged over each element's neighbourhood within radius
:math:`r_{\min}` to suppress checkerboard patterns.

**Density update**: Optimality Criteria

.. math::

   \rho_e^{\,\text{new}} = \rho_e \cdot B_e^{\eta}

where :math:`B_e = -\partial C/\partial \rho_e / (\lambda\,\partial V/\partial \rho_e)`
and :math:`\lambda` is found by bisection to satisfy the volume constraint.
Iteration continues until :math:`\max_e |\Delta\rho_e| < 0.01`.

**Mesh-independence filter**: Weighted average over a ball of radius ``--rmin``

.. math::

   \hat{\rho}_e = \frac{\sum_{f} H_{ef} \rho_f}{\sum_f H_{ef}}, \quad
   H_{ef} = \max(0, r_{\min} - \Delta(e,f))

**Element**: 8-node hexahedral H8 with trilinear shape functions.
Stiffness matrix pre-computed by ``lk_H8()`` and assembled via vectorised
indexing.  Sparse system solved with ``scipy.sparse.linalg.spsolve``.

**Binarisation**: The converged density field is binarised before Stage 1:

.. math::

   \text{solid}_e =
   \begin{cases}
   1 & \rho_e > v_t \\
   0 & \text{otherwise}
   \end{cases}

where :math:`v_t = 0.30` (``--vol_thresh``).  BC-tagged voxels are retained
regardless of density.

Boundary-Condition Tag Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before optimisation, each element is assigned an integer tag encoding its
structural role:

* ``tag = 1`` **(support)**: shares a node with a fixed-displacement DOF.
* ``tag = 2`` **(load)**: shares a node with an applied force.
* ``tag = 3`` **(passive)**: excluded from the design domain (fixed void).

Tags 1 and 2 propagate through Stages 1--3, serving two roles:
(1) thinning protection — tagged voxels are never deleted during
skeletonisation; and (2) frame boundary conditions — tagged nodes supply
fixed DOFs and loads to the frame solver.  Tag 3 elements remain passive
and do not enter downstream stages.


Skeletonisation (Stage 1)
--------------------------

Yin Medial-Axis Thinning
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:mod:`src.pipelines.baseline_yin.thinning` implements the 3-D parallel
directional thinning algorithm by Yin (2002).

**Core idea**: Iteratively peel voxels from the surface of the binary
solid, one layer at a time, until only a one-voxel-wide skeleton remains.
The critical requirement is that every connected component, tunnel, and
cavity is preserved.

**Simple point test**: A voxel is only removed if it is a *simple point*:
its deletion does not disconnect any material or merge any void regions.
The test examines the voxel's :math:`3 \times 3 \times 3` neighbourhood:
the surrounding 26-connected material voxels must form a single connected
cluster, and the surrounding 6-connected void voxels must also form a
single connected cluster.  If either condition fails, the voxel is a
topological bridge whose removal would sever a load path.
Tested by :func:`src.pipelines.baseline_yin.topology.is_simple_point`.

**Directional sweeps**: Each iteration sweeps in 6 axis-aligned directions
(:math:`\pm X, \pm Y, \pm Z`).  For each direction, border voxels that
face the current sweep direction are collected as candidates.  Each
candidate is re-checked before deletion because an earlier removal in the
same pass can change a later candidate's neighbourhood.  Iteration
continues until a full sweep produces no deletions.

**Unconditionally excluded voxels**: Three classes of voxels are never
deleted, regardless of simple-point status:

1. **BC-tagged voxels** (``tag > 0``): support and load voxels from
   Stage 0 are preserved; without this protection, boundary voxels
   passing the simple-point test would be removed, severing load paths.
2. **End voxels** (:math:`\le 1` 26-neighbour): protecting endpoints
   prevents thin chains from eroding entirely, preserving the
   curve-skeleton topology.
3. **Surface points** (mode 1/3 only): voxels that locally bound a
   two-dimensional medial surface are retained, allowing surface-like
   regions to thin to sheets rather than curves.

**Numba acceleration**: Critical loops use ``@njit(parallel=True)`` for
per-voxel parallelism.

**Thinning modes**:

.. list-table::
   :widths: 10 25 65
   :header-rows: 1

   * - Mode
     - Name
     - Preserved topology
   * - 0
     - Curve-preserving
     - Curve endpoints; surface regions collapsed to medial curves
   * - 1
     - Surface-preserving
     - Surface voxels retained; curves collapsed to surfaces
   * - 3
     - Hybrid (surface + curve)
     - Both curve endpoints and surface voxels preserved

Graph Extraction
^^^^^^^^^^^^^^^^^^

:func:`src.pipelines.baseline_yin.graph.extract_graph` converts the
skeleton voxels into a graph of nodes and edges:

1. **BC tag cluster collapsing**: Connected clusters of BC-tagged
   skeleton voxels are collapsed to single centroid nodes via
   ``consolidate_tagged_voxels()``, preventing a support or load point
   from fragmenting into multiple graph nodes.  Each centroid is
   reconnected to its skeleton neighbours by 3-D line rasterisation.

2. **Node classification**: Each remaining skeleton voxel is classified
   by counting its 26-connected neighbours:

   * **Endpoint** (:math:`\le 1` neighbour): terminal graph node.
   * **Body** (exactly 2 neighbours): interior waypoint along an edge.
   * **Junction** (:math:`\ge 3` neighbours): branching graph node.

3. **Edge tracing**: BFS from each node through degree-2 voxels traces
   the skeleton path between nodes; body voxels are stored as waypoints.

4. **Laplacian smoothing**: Each edge polyline is smoothed by three
   iterations of Laplacian averaging: every interior waypoint is moved
   halfway toward the midpoint of its two neighbours, while endpoint
   nodes remain fixed.  This reduces voxel-grid staircase noise.

5. **BC tag assignment**: each node inherits the dominant BC tag of its
   cluster.

Graph Post-processing
^^^^^^^^^^^^^^^^^^^^^^^

The raw graph inherits noise from voxel discretisation.  Three operations
clean it:

1. **Edge collapse**: Node pairs connected by an edge shorter than
   ``--collapse_thresh`` are merged; BC-tagged nodes anchor the merge.
   Resulting degree-2 nodes are absorbed into neighbouring edges.

2. **Branch pruning**: Degree-1 leaf nodes on edges shorter than
   ``--prune_len`` are removed iteratively; BC-tagged endpoints are
   always retained.

3. **Geometric simplification**: The Ramer--Douglas--Peucker algorithm
   reduces waypoints on each edge to within ``--rdp`` perpendicular
   distance tolerance; node positions and graph topology are unchanged.

Initial Radius Assignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each edge receives an initial cross-sectional radius by uniform
volume-matching:

.. math::

   r_0 = \sqrt{\frac{V_{\text{solid}}}{\pi \sum_i L_i}}

where :math:`V_{\text{solid}}` is the total solid voxel volume and
:math:`L_i` the smoothed polyline length of edge :math:`i`, ensuring
Stage 2 begins from a volume budget consistent with the original density
field.

For external mesh inputs (``--mesh_input``), where no density field is
available, the median EDT value along each edge is used instead, giving
each member a radius proportional to its local thickness in the original
solid.


Two-Signal Zone Classification (Hybrid)
-----------------------------------------

For hybrid mode,
:func:`src.pipelines.baseline_yin.graph.classify_skeleton_post_thinning`
partitions the skeleton into surface and beam zones by combining two
complementary topological signals.

**Signal A** (set-difference): Compares two thinning passes.  Voxels
present in the hybrid skeleton (mode 3) but absent from a curve-only
skeleton (mode 0) are identified as medial-surface voxels that standard
thinning would have collapsed:

.. math::

   A = \mathcal{S}_3 \setminus \mathcal{S}_0

Signal A dominates for thick structures where the two thinning modes
diverge.

**Signal B** (octant plane patterns): Each voxel's :math:`3 \times 3 \times 3`
neighbourhood is divided into 8 octants.  A voxel is a surface candidate
if :math:`\ge 4` octants match one of 12 known planar configurations AND
the voxel has :math:`\ge 3` skeleton neighbours.  Signal B compensates for
thin structures where both thinning modes produce similar skeletons.

**Combination**: Plate candidates are the union of both signals:

.. math::

   \text{candidates} = A \cup B

The union is segmented into 26-connected regions, filtered by minimum size
(``min_plate_size``) and elongation (global linearity eigenvalue ratio),
and assigned as surface zones.  All remaining voxels become beam zones.
When neither signal produces candidates, the classifier returns a pure-beam
skeleton.


Hybrid Surface Extraction and Joint Creation
----------------------------------------------

**Surface extraction**: Each surface region identified during
classification is a one-voxel-thick medial sheet.  Full-thickness regions
are recovered by morphological dilation into the original solid,
constrained to avoid beam zones
(:func:`src.pipelines.baseline_yin.plate_extraction.recover_plate_regions_from_skeleton`).
The recovered boundary is meshed into a watertight triangular surface,
smoothed by Taubin filtering to reduce voxelisation artefacts, and
annotated with per-vertex thickness derived from the EDT:
:math:`t_v = 2 \times \text{EDT}(v) \times \text{pitch}`.

**Joint creation**: Beam-zone voxels are passed to graph extraction.
Beam nodes near a surface boundary are snapped onto the nearest surface
vertex using a KD-tree
(:func:`src.pipelines.baseline_yin.joint_creation.create_beam_plate_joints`),
creating shared nodes at each beam-to-surface interface.  Isolated beam
endpoints within a search range receive a new junction node connected by a
short bridging edge.  BC tags are propagated through all joints.


Frame FEA
----------

Straight Beam Element
^^^^^^^^^^^^^^^^^^^^^^^

:func:`src.optimization.fem.solve_frame` assembles a 3-D Euler--Bernoulli
frame stiffness matrix.

**Element**: 2-node 3-D beam with 6 DOF per node (3 translations
:math:`u_x, u_y, u_z` + 3 rotations :math:`\theta_x, \theta_y, \theta_z`).
Local stiffness :math:`\mathbf{K}_e` (12x12) is computed analytically from
beam length, Young's modulus :math:`E`, and cross-section radius :math:`r`
(area :math:`A = \pi r^2`, moment of inertia :math:`I = \pi r^4/4`).

**Assembly**: Global :math:`\mathbf{K} = \sum_e \mathbf{T}_e^\top \mathbf{K}_e \mathbf{T}_e`
where :math:`\mathbf{T}_e` is the 12x12 rotation matrix aligning the local
beam axis to the global frame.

**Solve**: Sparse LU factorisation via ``scipy.sparse.linalg.spsolve(K_ff, f_f)``
where subscript :math:`f` denotes free (non-constrained) DOFs.

**Compliance**: :math:`C = \mathbf{F}^\top \mathbf{u}` (scalar, minimised
by optimisers).

**Gradient**: :math:`\partial C / \partial r_e = -\mathbf{u}_e^\top (\partial \mathbf{K}_e / \partial r_e) \mathbf{u}_e`
evaluated analytically from the cross-section area dependence.

Curved Beam Element (IGA Timoshenko)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``--curved`` is enabled, a cubic Bezier curve is fitted to each
skeleton edge and a curved finite element is built on the same Bezier
basis (isogeometric analysis).

**Bezier fitting**: With endpoints :math:`\mathbf{P}_0` and
:math:`\mathbf{P}_3` fixed, interior control points :math:`\mathbf{P}_1`
and :math:`\mathbf{P}_2` are found by least-squares fitting, parameterised
by cumulative arc-length fraction
(:func:`src.curves.spline.fit_cubic_bezier`).
Two constraints prevent degenerate curves:

* **Chord monotonicity**: both control points project between the endpoints
  along the chord (:math:`0 \le t_1 \le t_2 \le L`), preventing the curve
  from looping.
* **Perpendicular bulge limit**: lateral deviation is clamped to prevent
  excessive curvature that would produce ill-conditioned elements.

**Timoshenko formulation**: A Timoshenko formulation is adopted because
centreline curvature introduces axial-bending coupling that
Euler--Bernoulli cannot capture.  The cubic Bernstein basis describes both
geometry and displacement (isogeometric analysis).  The four control points
give 24 DOFs (4 points x 6 DOF each).

**Static condensation**: To maintain compatibility with the 12x12 straight
element format, the 12 interior DOFs (at :math:`\mathbf{P}_1` and
:math:`\mathbf{P}_2`) are eliminated:

.. math::

   \mathbf{K}_{\text{cond}} = \mathbf{K}_{bb}
   - \mathbf{K}_{bi}\,\mathbf{K}_{ii}^{-1}\,\mathbf{K}_{ib}

where :math:`b` and :math:`i` denote boundary
(:math:`\mathbf{P}_0, \mathbf{P}_3`) and interior
(:math:`\mathbf{P}_1, \mathbf{P}_2`) DOFs, making the curved element a
drop-in replacement in the global assembly.

**Mixed assembly**: :func:`src.optimization.fem.solve_curved_frame`
handles mixed straight and curved elements in the same global system.
Elements without control points use the standard Euler--Bernoulli
stiffness; elements with control points use the condensed IGA stiffness.


Frame Re-optimisation (Stages 2--3)
-------------------------------------

Optimality Criteria Sizing (Stage 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`src.optimization.size_opt.optimize_size` adjusts every beam radius
to minimise compliance while keeping total frame volume equal to the solid
volume from Stage 1.  In hybrid mode the constraint includes shell
contributions:
:math:`\sum_i \pi r_i^2 L_i + \sum_j A_j h_j = V_{\text{target}}`.

.. math::

   r_i^{\,\text{new}} = \operatorname{clip}_{[r_i(1-m),\; r_i(1+m)]}
   \left( r_i \left( \frac{-\partial C / \partial r_i}
   {\lambda\;\partial V / \partial r_i} \right)^{\!\eta} \right)

where :math:`m = 0.2` limits change per iteration, :math:`\eta = 0.5` is a
damping exponent, and :math:`\lambda` enforces the volume constraint via
bisection.  Both sensitivities
(:math:`\partial C/\partial r_i`, :math:`\partial V/\partial r_i = 2\pi r_i L_i`)
are evaluated analytically.

**Convergence**: Iteration terminates when the maximum radius change falls
below :math:`10^{-3}`, or after 50 iterations (``--iters``).

L-BFGS-B Layout Optimisation (Stage 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`src.optimization.layout_opt.optimize_layout` repositions the free
nodes of the beam graph to minimise compliance:

.. math::

   \min_{\mathbf{x}} \; C(\mathbf{x}) \quad \text{s.t.} \;
   \mathbf{x}_{\min} \le \mathbf{x} \le \mathbf{x}_{\max}

Each free node may move a limited distance from its initial position,
scaled as a fraction of the design domain size.  BC-tagged and loaded nodes
are excluded and remain fixed.  Compliance gradients are computed
analytically from per-element sensitivities.  When curved beams are
enabled, the design vector is extended with six control-point coordinates
per curved edge, subject to the monotonicity and bulge constraints.

After convergence, nodes within ``snap_dist`` of each other are merged to
prevent near-zero-length edges; BC-tagged nodes are never merged or
relocated.

Alternating Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^

Size and layout optimisation are interdependent: each assumes the other's
variables are fixed.  The two stages therefore alternate in cycles of size
(Stage 2) then layout (Stage 3), controlled by ``--opt_loops`` (default 2).
Size runs first because the uniform radii from Stage 1 give all members
near-identical sensitivities; once a non-uniform distribution is
established, layout can meaningfully reposition nodes.  Compliance
reductions beyond the second cycle are marginal in practice.


External Mesh Input
--------------------

The pipeline is not limited to density fields from the built-in SIMP solver.
Topology-optimised results from external solvers (SolidWorks, Abaqus, ANSYS)
can be imported as surface meshes (STL, OBJ, or PLY) via
:mod:`src.mesh_import.mesh_voxelizer`.

The mesh is voxelised at a user-specified pitch (``--mesh_pitch``) using
trimesh, with the interior automatically filled.  The resulting boolean grid
is transposed from trimesh's native ``(nx, ny, nz)`` ordering to the pipeline
convention ``(nely, nelx, nelz)``.

The saved NPZ contains only ``rho`` (no ``bc_tags``), since standard mesh
formats do not embed boundary-condition information.  EDT-based radius
assignment is used instead of uniform volume-matching.  Support and load
locations must be re-specified manually to enable frame re-optimisation in
Stages 2--3.


Symmetry Enforcement
---------------------

:mod:`src.optimization.symmetry` detects symmetric node/edge pairs by
reflecting positions across user-specified planes (``--symmetry xz,yz``),
then enforces symmetry via radius averaging (size opt) and a soft penalty
plus exact projection (layout opt).

For geometry-only runs (no optimisation), a mirror-half approach is
available: all nodes on one side of the symmetry plane are kept, the other
side is discarded, and the kept nodes and edges are mirrored to create a
perfectly symmetric skeleton.  Edges crossing the symmetry plane are split
at the intersection point to prevent member loss.
