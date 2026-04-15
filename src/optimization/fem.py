"""3-D frame finite element analysis (straight Euler-Bernoulli + curved IGA Timoshenko).

Assembles a global sparse stiffness matrix from circular cross-section beam
elements, applies boundary conditions, solves for displacements and
compliance, and computes per-element compliance sensitivities for gradient-
based optimisation.

Straight beams use the classical 12-DOF Euler-Bernoulli element.
Curved beams use a 24-DOF isogeometric Timoshenko element with cubic
Bernstein basis functions, integrated via Gauss-Legendre quadrature
and statically condensed to 12 DOFs for global assembly.

Main entry points
-----------------
- :func:`solve_frame` — straight-beam static analysis
- :func:`solve_curved_frame` — mixed straight/curved static analysis
- :func:`compute_frame_gradients` — ``dC/dr`` per edge (straight)
- :func:`compute_curved_size_gradients` — ``dC/dr`` per edge (curved)
- :func:`compute_curved_ctrl_gradients` — ``dC/dP`` per control point
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

from src.curves.spline import bernstein_basis, bernstein_basis_d1

def compute_element_stiffness(E, A, Iy, Iz, J, G, L):
    """
    Computes 12x12 local stiffness matrix for a 3D Euler-Bernoulli beam.
    Axis: Beam along local X-axis.
    """
    k = np.zeros((12, 12))
    
    # Axial (u)
    k[0, 0] = k[6, 6] = E * A / L
    k[0, 6] = k[6, 0] = -E * A / L
    
    # Torsion (theta_x)
    k[3, 3] = k[9, 9] = G * J / L
    k[3, 9] = k[9, 3] = -G * J / L
    
    # Bending about Z (v, theta_z)
    k[1, 1] = k[7, 7] = 12 * E * Iz / L**3
    k[1, 7] = k[7, 1] = -12 * E * Iz / L**3
    k[1, 5] = k[5, 1] = 6 * E * Iz / L**2
    k[1, 11] = k[11, 1] = 6 * E * Iz / L**2
    k[5, 5] = k[11, 11] = 4 * E * Iz / L
    k[5, 11] = k[11, 5] = 2 * E * Iz / L
    k[7, 11] = k[11, 7] = -6 * E * Iz / L**2
    k[5, 7] = k[7, 5] = -6 * E * Iz / L**2
    
    # Bending about Y (w, theta_y)
    k[2, 2] = k[8, 8] = 12 * E * Iy / L**3
    k[2, 8] = k[8, 2] = -12 * E * Iy / L**3
    k[2, 4] = k[4, 2] = -6 * E * Iy / L**2
    k[2, 10] = k[10, 2] = -6 * E * Iy / L**2
    k[4, 4] = k[10, 10] = 4 * E * Iy / L
    k[4, 10] = k[10, 4] = 2 * E * Iy / L
    k[8, 10] = k[10, 8] = 6 * E * Iy / L**2
    k[4, 8] = k[8, 4] = 6 * E * Iy / L**2
    
    return k

def rotation_matrix(vec, roll=0.0):
    """
    Computes 12x12 rotation matrix from Local to Global [T].
    vec: Vector from node 1 to node 2.
    """
    L = np.linalg.norm(vec)
    if L < 1e-8: return np.eye(12)
    
    Cx, Cy, Cz = vec / L
    
    # Handle vertical alignment cases for robustness
    if np.abs(Cz) > 0.999: # Almost vertical
        # Align global Y with local Y (approximation)
        R = np.array([
            [0, 0, np.sign(Cz)],
            [0, 1, 0],
            [-np.sign(Cz), 0, 0]
        ])
    else:
        # Standard formulation
        D = np.sqrt(Cx**2 + Cy**2)
        R = np.array([
            [Cx, Cy, Cz],
            [-Cy/D, Cx/D, 0],
            [-Cx*Cz/D, -Cy*Cz/D, D]
        ])
        
    # Build 12x12
    T = np.zeros((12, 12))
    T[0:3, 0:3] = T[3:6, 3:6] = T[6:9, 6:9] = T[9:12, 9:12] = R
    
    return T


# =========================================================================
# Flat triangular shell element (MITC3 — membrane + bending, 6 DOF/node)
# =========================================================================

def _triangle_area_and_normal(v0, v1, v2):
    """Return (area, unit_normal) for a triangle defined by three vertices."""
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    twice_area = np.linalg.norm(cross)
    if twice_area < 1e-30:
        return 0.0, np.array([0.0, 0.0, 1.0])
    return 0.5 * twice_area, cross / twice_area


def compute_shell_rotation(v0, v1, v2):
    """Build 3x3 rotation from local (x',y',z') to global (X,Y,Z).

    Local frame::

        x' = (v1-v0) / ||v1-v0||
        z' = normal to the triangle
        y' = z' x x'

    Returns R such that ``v_global = R @ v_local``.
    """
    e1 = v1 - v0
    L1 = np.linalg.norm(e1)
    if L1 < 1e-14:
        return np.eye(3)
    xp = e1 / L1
    e2 = v2 - v0
    zp = np.cross(e1, e2)
    Lz = np.linalg.norm(zp)
    if Lz < 1e-14:
        return np.eye(3)
    zp /= Lz
    yp = np.cross(zp, xp)
    # R rows = local axes in global coords
    R = np.array([xp, yp, zp])  # (3,3)
    return R


def compute_shell_element_stiffness(E, nu, thickness, v0, v1, v2):
    """Compute 18x18 flat shell stiffness (membrane + plate bending).

    Uses constant-strain triangle (CST) for membrane and discrete
    Kirchhoff triangle (DKT-like) bending with 6 DOF per node
    ``[u, v, w, theta_x, theta_y, theta_z]``.

    The element is formulated in a local coordinate system aligned with
    the triangle plane, then rotated to global coordinates.

    Parameters
    ----------
    E : float — Young's modulus
    nu : float — Poisson's ratio
    thickness : float — shell thickness
    v0, v1, v2 : (3,) arrays — triangle vertex positions (global coords)

    Returns
    -------
    K_global : (18, 18) ndarray — element stiffness in global coordinates
    """
    v0, v1, v2 = np.asarray(v0, float), np.asarray(v1, float), np.asarray(v2, float)
    area, _ = _triangle_area_and_normal(v0, v1, v2)
    if area < 1e-20:
        return np.zeros((18, 18))

    h = max(thickness, 1e-6)

    # Local coordinate system
    R = compute_shell_rotation(v0, v1, v2)  # local→global

    # Transform vertices to local coordinates
    p0 = R @ v0
    p1 = R @ v1
    p2 = R @ v2

    # Local 2D coordinates (z' ≈ constant for flat triangle)
    x1, y1 = 0.0, 0.0
    x2, y2 = p1[0] - p0[0], p1[1] - p0[1]
    x3, y3 = p2[0] - p0[0], p2[1] - p0[1]

    # ── Membrane stiffness (CST) ─────────────────────────────────
    # Plane stress constitutive matrix
    Dm = (E * h / (1.0 - nu**2)) * np.array([
        [1.0, nu,  0.0],
        [nu,  1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])

    # CST strain-displacement matrix (constant over element)
    # B_m maps [u1,v1, u2,v2, u3,v3] → [eps_xx, eps_yy, gamma_xy]
    det_J = (x2 * y3 - x3 * y2)
    if abs(det_J) < 1e-20:
        return np.zeros((18, 18))

    inv_det = 1.0 / det_J
    b1 = (y2 - y3) * inv_det
    b2 = (y3 - y1) * inv_det
    b3 = (y1 - y2) * inv_det
    c1 = (x3 - x2) * inv_det
    c2 = (x1 - x3) * inv_det
    c3 = (x2 - x1) * inv_det

    Bm = np.array([
        [b1, 0,  b2, 0,  b3, 0],
        [0,  c1, 0,  c2, 0,  c3],
        [c1, b1, c2, b2, c3, b3],
    ])

    Km_local = area * (Bm.T @ Dm @ Bm)  # (6,6) in [u1,v1,u2,v2,u3,v3]

    # ── Bending stiffness (simplified DKT) ────────────────────────
    # Plate bending constitutive matrix
    Db = (E * h**3 / (12.0 * (1.0 - nu**2))) * np.array([
        [1.0, nu,  0.0],
        [nu,  1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])

    # For a simplified flat-shell bending element, use the analogy with
    # the CST membrane element but for curvatures:
    # kappa_xx = -d²w/dx², kappa_yy = -d²w/dy², kappa_xy = -2 d²w/dxdy
    # With linear interpolation of rotations theta_x, theta_y per node,
    # we get a constant-curvature bending element.
    # DOFs per node: [w, theta_x, theta_y] = [w, dw/dy, -dw/dx]
    # Bb maps [w1,θx1,θy1, w2,θx2,θy2, w3,θx3,θy3] → [κ_xx, κ_yy, 2κ_xy]

    # Using the standard linear triangle bending formulation:
    # θx = dw/dy, θy = -dw/dx
    # κ_xx = -dθy/dx = d²w/dx²  → use b_i for θy
    # κ_yy = dθx/dy             → use c_i for θx
    # κ_xy = dθx/dx - dθy/dy    → use b_i for θx, c_i for θy

    Bb = np.zeros((3, 9))
    # Node 1
    Bb[0, 2] = -b1   # κ_xx from θy1
    Bb[1, 1] = c1     # κ_yy from θx1
    Bb[2, 1] = b1     # κ_xy from θx1
    Bb[2, 2] = -c1    # κ_xy from θy1
    # Node 2
    Bb[0, 5] = -b2
    Bb[1, 4] = c2
    Bb[2, 4] = b2
    Bb[2, 5] = -c2
    # Node 3
    Bb[0, 8] = -b3
    Bb[1, 7] = c3
    Bb[2, 7] = b3
    Bb[2, 8] = -c3

    Kb_local = area * (Bb.T @ Db @ Bb)  # (9,9) in [w1,θx1,θy1, ...]

    # ── Assemble into 18x18 local shell stiffness ─────────────────
    # Local DOF order per node: [u', v', w', θx', θy', θz']
    # Membrane DOFs: u'(0), v'(1) → indices 0,1, 6,7, 12,13
    # Bending DOFs: w'(2), θx'(3), θy'(4) → indices 2,3,4, 8,9,10, 14,15,16
    # θz'(5) = drilling rotation → add small stiffness for stability

    K_local = np.zeros((18, 18))

    # Membrane: scatter Km_local (6x6) into K_local
    mem_dofs = [0, 1, 6, 7, 12, 13]  # u',v' for nodes 0,1,2
    for i in range(6):
        for j in range(6):
            K_local[mem_dofs[i], mem_dofs[j]] += Km_local[i, j]

    # Bending: scatter Kb_local (9x9) into K_local
    bend_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16]  # w',θx',θy' for nodes 0,1,2
    for i in range(9):
        for j in range(9):
            K_local[bend_dofs[i], bend_dofs[j]] += Kb_local[i, j]

    # Drilling DOF stabilisation (small diagonal stiffness)
    drill_stiffness = 1e-3 * E * h * area  # scale with element
    for n in range(3):
        K_local[n * 6 + 5, n * 6 + 5] += drill_stiffness

    # ── Rotate to global coordinates ──────────────────────────────
    # Build 18x18 transformation: T = block_diag(R, R, R, R, R, R)
    # Each 3x3 block rotates [u,v,w] or [θx,θy,θz]
    T = np.zeros((18, 18))
    Rt = R.T  # local→global = R^T (since R maps global→local as rows)
    for b in range(6):
        T[b*3:b*3+3, b*3:b*3+3] = Rt

    K_global = T.T @ K_local @ T
    K_global = 0.5 * (K_global + K_global.T)  # symmetrise
    return K_global


def compute_shell_element_stiffness_derivative(E, nu, thickness, v0, v1, v2):
    """Compute dK_shell/d(thickness) for a flat triangular shell element.

    Since the membrane stiffness scales as h and bending as h³,
    dK/dh = K_membrane/h + 3*K_bending/h.

    Returns (18, 18) ndarray.
    """
    v0, v1, v2 = np.asarray(v0, float), np.asarray(v1, float), np.asarray(v2, float)
    area, _ = _triangle_area_and_normal(v0, v1, v2)
    if area < 1e-20:
        return np.zeros((18, 18))

    h = max(thickness, 1e-6)

    R = compute_shell_rotation(v0, v1, v2)
    p0 = R @ v0
    p1 = R @ v1
    p2 = R @ v2

    x1, y1 = 0.0, 0.0
    x2, y2 = p1[0] - p0[0], p1[1] - p0[1]
    x3, y3 = p2[0] - p0[0], p2[1] - p0[1]

    det_J = (x2 * y3 - x3 * y2)
    if abs(det_J) < 1e-20:
        return np.zeros((18, 18))
    inv_det = 1.0 / det_J

    b1 = (y2 - y3) * inv_det
    b2 = (y3 - y1) * inv_det
    b3 = (y1 - y2) * inv_det
    c1 = (x3 - x2) * inv_det
    c2 = (x1 - x3) * inv_det
    c3 = (x2 - x1) * inv_det

    # ── Membrane derivative: dK_m/dh = K_m / h (linear in h) ─────
    # Recompute Dm at unit thickness
    Dm_unit = (E / (1.0 - nu**2)) * np.array([
        [1.0, nu,  0.0],
        [nu,  1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])
    Bm = np.array([
        [b1, 0,  b2, 0,  b3, 0],
        [0,  c1, 0,  c2, 0,  c3],
        [c1, b1, c2, b2, c3, b3],
    ])
    dKm_local = area * (Bm.T @ Dm_unit @ Bm)  # dK_m/dh (constant)

    # ── Bending derivative: dK_b/dh = 3h² * K_b_unit ─────────────
    Db_unit = (E / (12.0 * (1.0 - nu**2))) * np.array([
        [1.0, nu,  0.0],
        [nu,  1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])
    Bb = np.zeros((3, 9))
    Bb[0, 2] = -b1; Bb[1, 1] = c1; Bb[2, 1] = b1; Bb[2, 2] = -c1
    Bb[0, 5] = -b2; Bb[1, 4] = c2; Bb[2, 4] = b2; Bb[2, 5] = -c2
    Bb[0, 8] = -b3; Bb[1, 7] = c3; Bb[2, 7] = b3; Bb[2, 8] = -c3

    dKb_local = area * 3.0 * h**2 * (Bb.T @ Db_unit @ Bb)  # (9,9)

    # ── Scatter into 18x18 ────────────────────────────────────────
    dK_local = np.zeros((18, 18))

    mem_dofs = [0, 1, 6, 7, 12, 13]
    for i in range(6):
        for j in range(6):
            dK_local[mem_dofs[i], mem_dofs[j]] += dKm_local[i, j]

    bend_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16]
    for i in range(9):
        for j in range(9):
            dK_local[bend_dofs[i], bend_dofs[j]] += dKb_local[i, j]

    # Drilling derivative: d/dh of drill_stiffness = 1e-3 * E * area
    drill_deriv = 1e-3 * E * area
    for n in range(3):
        dK_local[n * 6 + 5, n * 6 + 5] += drill_deriv

    # Rotate to global
    T = np.zeros((18, 18))
    Rt = R.T
    for b in range(6):
        T[b*3:b*3+3, b*3:b*3+3] = Rt

    dK_global = T.T @ dK_local @ T
    dK_global = 0.5 * (dK_global + dK_global.T)
    return dK_global


def _fallback_mid_surface(plate):
    """Generate a mid-surface triangulation from plate voxel centers.

    Used when the reconstruction's alpha-shape / Delaunay mid-surface
    extraction fails (e.g. coplanar/cocircular points).  Projects voxel
    centers onto the plate's principal plane and triangulates in 2D.

    Returns dict with 'vertices', 'triangles', 'mean_thickness', 'node_tags',
    or None if insufficient points.
    """
    from scipy.spatial import Delaunay

    voxels = plate.get("voxels", [])
    if len(voxels) < 3:
        return None

    pts = np.array(voxels, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA to find principal plane
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals sorted ascending; eigvecs[:,0] = normal (smallest variance)
    normal = eigvecs[:, 0]

    # Project onto 2D plane (use eigvecs[:,1] and eigvecs[:,2] as axes)
    ax1 = eigvecs[:, 1]
    ax2 = eigvecs[:, 2]
    pts_2d = np.column_stack([centered @ ax1, centered @ ax2])

    # Add small jitter to break cocircular degeneracy
    rng = np.random.RandomState(42)
    pts_2d += rng.randn(*pts_2d.shape) * 1e-6

    try:
        tri = Delaunay(pts_2d)
        triangles = tri.simplices.tolist()
    except Exception:
        return None

    if len(triangles) == 0:
        return None

    h = plate.get("thickness", 1.0)
    return {
        "vertices": pts.tolist(),
        "triangles": triangles,
        "mean_thickness": h,
        "node_tags": {},
    }


def _get_plate_mid_surface(plate):
    """Get mid-surface data, using fallback triangulation if needed."""
    ms = plate.get("mid_surface")
    if ms is not None and len(ms.get("triangles", [])) > 0:
        return ms
    return _fallback_mid_surface(plate)


def _build_plate_node_map(plates, n_beam_nodes, nodes, bcs, loads):
    """Build a mapping from (plate_idx, local_vertex_idx) → global DOF node index.

    Beam-plate joint nodes (listed in ``connection_node_ids``) reuse the
    existing beam node index.  All other plate mid-surface vertices are
    assigned new global node indices starting at ``n_beam_nodes``.

    Loads and BCs are propagated to plate-only nodes based on:
    1. ``node_tags`` from the mid-surface (tag=1 → fixed, tag=2 → loaded)
    2. Fallback: infer from connection node membership in *bcs* / *loads*

    For loaded plates, the total force from loaded connection nodes is
    distributed equally across **all** plate nodes (connection + interior)
    to simulate a distributed surface load.

    Returns
    -------
    plate_node_map : list of ndarray
        ``plate_node_map[p]`` is an int array of length ``n_verts_p`` mapping
        each local vertex index to a global node index.
    all_nodes : ndarray, shape (N_total, 3)
        Expanded node array (beam nodes + new plate-only nodes).
    plate_bcs : dict
        ``{global_node_idx: [0,1,2,3,4,5]}`` for plate vertices needing BCs.
    plate_loads : dict
        ``{global_node_idx: load_vector}`` for plate vertices needing loads.
    """
    from scipy.spatial import cKDTree

    extra_nodes = []
    plate_node_map = []
    plate_bcs = {}
    plate_loads = {}
    next_id = n_beam_nodes

    for p_idx, plate in enumerate(plates):
        ms = _get_plate_mid_surface(plate)
        if ms is None:
            plate_node_map.append(np.array([], dtype=int))
            continue

        verts = np.array(ms["vertices"])
        n_v = len(verts)
        local_to_global = np.full(n_v, -1, dtype=int)

        # Joint nodes: match plate vertices to beam nodes by position
        conn_ids = plate.get("connection_node_ids", [])
        conn_ids = [c for c in conn_ids if c < n_beam_nodes]  # guard stale IDs
        if len(conn_ids) > 0 and len(nodes) > 0:
            beam_joint_pos = nodes[conn_ids]
            tree = cKDTree(beam_joint_pos)
            for vi in range(n_v):
                dist, idx = tree.query(verts[vi], k=1)
                if dist < 1e-3:  # close enough to be same node
                    local_to_global[vi] = conn_ids[idx]

        # Determine plate role from connection nodes' bcs/loads membership
        conn_has_bc = [cid for cid in conn_ids if cid in bcs]
        conn_has_load = [cid for cid in conn_ids if cid in loads]
        plate_is_fixed = len(conn_has_bc) > 0
        plate_is_loaded = len(conn_has_load) > 0

        # Mid-surface node tags (tag=1 fixed, tag=2 loaded)
        node_tags_ms = ms.get("node_tags", {})

        # Remaining vertices get new global IDs
        for vi in range(n_v):
            if local_to_global[vi] < 0:
                local_to_global[vi] = next_id
                extra_nodes.append(verts[vi])

                # Check mid-surface tag first
                tag = node_tags_ms.get(str(vi)) or node_tags_ms.get(vi, 0)

                if tag == 1 or (tag == 0 and plate_is_fixed and not plate_is_loaded):
                    # Fixed plate: fix all DOFs for interior nodes
                    plate_bcs[next_id] = [0, 1, 2, 3, 4, 5]

                next_id += 1

        plate_node_map.append(local_to_global)

        # Loaded plates: interior nodes don't need external loads — they're
        # mechanically connected to loaded connection nodes through shell
        # stiffness.  The shell elements transfer load internally.
        # No additional force distribution needed.

    if len(extra_nodes) > 0:
        all_nodes = np.vstack([nodes, np.array(extra_nodes)])
    else:
        all_nodes = nodes

    return plate_node_map, all_nodes, plate_bcs, plate_loads


def solve_frame(nodes, edges, radii, E=1.0, nu=0.3, loads={}, bcs={},
                plates=None, plate_thicknesses=None):
    """Solve a 3-D frame with optional shell plate elements.

    Assembles the global stiffness matrix ``K`` from beam elements and
    (optionally) triangular shell elements for plates.  Applies Dirichlet
    BCs, solves ``K_ff u_f = f_f``, and returns displacements + compliance.

    Args:
        nodes (numpy.ndarray): Beam node positions, shape ``(N, 3)``, mm.
        edges (numpy.ndarray): Beam element connectivity, shape ``(M, 2)``.
        radii (numpy.ndarray): Beam cross-section radii, shape ``(M,)``, mm.
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        loads (dict): ``{node_idx: [Fx, Fy, Fz, Mx, My, Mz]}``.
        bcs (dict): ``{node_idx: [dof_0, dof_1, …]}``.
        plates (list of dict or None): Plate data from reconstruction.
            Each must have ``mid_surface`` with ``vertices``, ``triangles``,
            ``thickness`` (or per-vertex via ``thickness_per_vertex``),
            and ``connection_node_ids``.
        plate_thicknesses (list of float or None): Override thickness per
            plate for optimisation.  If None, uses plate's own thickness.

    Returns:
        tuple:
            - **u** (ndarray): Full displacement vector.
            - **compliance** (float): ``c = f^T u``.
            - **elements** (list): Per-beam element data for gradients.
            - **shell_elements** (list): Per-shell-triangle element data
              (only present when plates are provided).
    """
    # ── Plate node mapping ────────────────────────────────────────
    has_plates = plates is not None and len(plates) > 0
    n_beam_nodes = len(nodes)

    if has_plates:
        plate_node_map, all_nodes, plate_bcs_extra, plate_loads_extra = \
            _build_plate_node_map(plates, n_beam_nodes, nodes, bcs, loads)
        # Merge plate BCs/loads with beam BCs/loads
        merged_bcs = dict(bcs)
        merged_bcs.update(plate_bcs_extra)
        merged_loads = dict(loads)
        merged_loads.update(plate_loads_extra)
    else:
        all_nodes = nodes
        merged_bcs = bcs
        merged_loads = loads
        plate_node_map = []

    n_nodes_total = len(all_nodes)
    n_dof = n_nodes_total * 6

    # ── Sparse matrix builder ─────────────────────────────────────
    I_idx, J_idx, V_val = [], [], []
    G = E / (2 * (1 + nu))
    elements = []

    # ── Beam elements ─────────────────────────────────────────────
    for e_i, (u, v) in enumerate(edges[:, :2]):
        u, v = int(u), int(v)
        p1, p2 = all_nodes[u], all_nodes[v]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        if L < 1e-6:
            continue

        r = radii[e_i]
        A = np.pi * r**2
        Iy = Iz = np.pi * r**4 / 4
        J = np.pi * r**4 / 2

        k_local = compute_element_stiffness(E, A, Iy, Iz, J, G, L)
        T = rotation_matrix(vec)
        k_global = T.T @ k_local @ T

        dof_indices = np.concatenate([
            np.arange(u*6, u*6+6),
            np.arange(v*6, v*6+6)
        ])

        for r_i in range(12):
            for c_j in range(12):
                val = k_global[r_i, c_j]
                if abs(val) > 1e-12:
                    I_idx.append(dof_indices[r_i])
                    J_idx.append(dof_indices[c_j])
                    V_val.append(val)

        elements.append({
            'k_global': k_global,
            'dofs': dof_indices
        })

    # ── Shell elements (plates) ───────────────────────────────────
    shell_elements = []
    n_shell_tris = 0

    if has_plates:
        for p_idx, plate in enumerate(plates):
            ms = _get_plate_mid_surface(plate)
            if ms is None:
                continue

            verts = np.array(ms["vertices"])
            tris = ms["triangles"]
            node_map = plate_node_map[p_idx]
            if len(node_map) == 0:
                continue

            # Determine thickness
            if plate_thicknesses is not None and p_idx < len(plate_thicknesses):
                h = plate_thicknesses[p_idx]
            else:
                h = plate.get("thickness", ms.get("mean_thickness", 1.0))
            h = max(h, 1e-3)

            for tri in tris:
                i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
                if max(i0, i1, i2) >= len(verts):
                    continue

                g0 = node_map[i0]
                g1 = node_map[i1]
                g2 = node_map[i2]

                v0 = all_nodes[g0]
                v1 = all_nodes[g1]
                v2 = all_nodes[g2]

                k_shell = compute_shell_element_stiffness(E, nu, h, v0, v1, v2)

                dof_indices = np.concatenate([
                    np.arange(g0*6, g0*6+6),
                    np.arange(g1*6, g1*6+6),
                    np.arange(g2*6, g2*6+6),
                ])

                for r_i in range(18):
                    for c_j in range(18):
                        val = k_shell[r_i, c_j]
                        if abs(val) > 1e-12:
                            I_idx.append(dof_indices[r_i])
                            J_idx.append(dof_indices[c_j])
                            V_val.append(val)

                shell_elements.append({
                    'k_global': k_shell,
                    'dofs': dof_indices,
                    'plate_idx': p_idx,
                    'thickness': h,
                    'vertex_global_ids': (g0, g1, g2),
                })
                n_shell_tris += 1

        if n_shell_tris > 0:
            print(f"     [FEA] Assembled {n_shell_tris} shell triangles "
                  f"from {len(plates)} plate(s)")

    # ── Build K ───────────────────────────────────────────────────
    K_global = sp.coo_matrix(
        (V_val, (I_idx, J_idx)), shape=(n_dof, n_dof)).tocsc()

    # ── Load vector ───────────────────────────────────────────────
    F_global = np.zeros(n_dof)
    for n_id, load in merged_loads.items():
        start = n_id * 6
        for i, val in enumerate(load):
            if i < 6:
                F_global[start + i] += val

    # ── Apply BCs ─────────────────────────────────────────────────
    fixed_dofs = []
    for n_id, dofs in merged_bcs.items():
        base = n_id * 6
        for d in dofs:
            fixed_dofs.append(base + d)

    fixed_dofs = np.array(fixed_dofs) if len(fixed_dofs) > 0 else np.array([], dtype=int)
    free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

    # ── Solve ─────────────────────────────────────────────────────
    K_free = K_global[free_dofs, :][:, free_dofs]
    F_free = F_global[free_dofs]

    # Stabilise zero-stiffness DOFs (common with shell drilling DOFs
    # or plate-only nodes connected to a single degenerate triangle)
    if has_plates:
        diag = np.array(K_free.diagonal()).ravel()
        zero_diag = np.where(np.abs(diag) < 1e-12)[0]
        if len(zero_diag) > 0:
            # Add small stiffness to prevent singularity
            stab = max(np.max(np.abs(diag)) * 1e-6, 1e-3)
            stab_diag = sp.lil_matrix(K_free.shape, dtype=float)
            for zd in zero_diag:
                stab_diag[zd, zd] = stab
            K_free = K_free + stab_diag.tocsc()
            print(f"     [FEA+Shell] Stabilised {len(zero_diag)} zero-stiffness DOFs")

    label = "[FEA]" if not has_plates else "[FEA+Shell]"
    print(f"     {label} Solving system with {len(free_dofs)} DOFs "
          f"({len(elements)} beams, {n_shell_tris} shells)...")
    u_free = scipy.sparse.linalg.spsolve(K_free, F_free)

    u_total = np.zeros(n_dof)
    u_total[free_dofs] = u_free

    compliance = np.dot(F_global, u_total)

    if has_plates:
        return u_total, compliance, elements, shell_elements
    return u_total, compliance, elements


def compute_beam_strain_energy(u, elements):
    """Compute per-beam strain energy from solve_frame outputs.

    Args:
        u: (N*6,) full displacement vector from solve_frame.
        elements: list of {'k_global', 'dofs'} from solve_frame.

    Returns:
        numpy.ndarray: (M,) per-beam strain energy (u_e^T K_e u_e).
    """
    se = np.zeros(len(elements))
    for i, elem in enumerate(elements):
        dofs = elem['dofs']
        u_e = u[dofs]
        se[i] = u_e @ elem['k_global'] @ u_e
    return se


def compute_element_stiffness_derivative(E, r, L, nu):
    """
    Computes derivative of stiffness matrix w.r.t radius r.
    dk/dr.
    """
    # Dependencies:
    # A = pi*r^2  => dA/dr = 2*pi*r
    # I = pi*r^4/4 => dI/dr = pi*r^3
    # J = pi*r^4/2 => dJ/dr = 2*pi*r^3
    
    A = np.pi * r**2
    dA = 2 * np.pi * r
    
    I = np.pi * r**4 / 4
    dI = np.pi * r**3
    
    J = np.pi * r**4 / 2
    dJ = 2 * np.pi * r**3
    
    G = E / (2 * (1 + nu))
    
    # We can reuse the same structure as compute_element_stiffness
    # but pass in the DERIVATIVES of properties instead of properties
    # Because K is linear in A, I, J.
    # dk/dr = K(dA/dr, dI/dr, dJ/dr) - wait, is it purely linear?
    # Yes, terms are like E*A/L. d/dr(E*A/L) = E/L * dA/dr. 
    
    dk = compute_element_stiffness(E, dA, dI, dI, dJ, G, L)
    return dk

def compute_frame_gradients(nodes, edges, radii, u_total, E=1.0, nu=0.3):
    """
    Computes sensitivity of compliance w.r.t radius of each member.
    dC/dr_i = - u^T * (dK_global/dr_i) * u
    
    Returns: gradients (M,) array.
    """
    gradients = np.zeros(len(edges))
    
    for i, (u_idx, v_idx) in enumerate(edges[:, :2]):
        u_idx, v_idx = int(u_idx), int(v_idx)
        
        # 1. Get Element Displacements
        # Indices in global vector
        dofs = np.concatenate([
            np.arange(u_idx*6, u_idx*6+6),
            np.arange(v_idx*6, v_idx*6+6)
        ])
        
        u_elem = u_total[dofs]
        
        # 2. Geometry
        p1, p2 = nodes[u_idx], nodes[v_idx]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        
        if L < 1e-6: continue
        
        r = radii[i]
        
        # 3. Compute dk_local / dr
        dk_local = compute_element_stiffness_derivative(E, r, L, nu)
        
        # 4. Rotate to Global: dK_global = T.T * dk_local * T
        # (T is constant w.r.t radius, only depends on coord)
        T = rotation_matrix(vec)
        dK_global = T.T @ dk_local @ T
        
        # 5. Sensitivity = - u_e.T * dK_e * u_e
        # Energy term.
        # Note: Compliance C = F.u. 
        # Gradients usually defined as dC/drho.
        # dC/dx = - u^T * dK/dx * u
        
        term = u_elem.T @ dK_global @ u_elem
        gradients[i] = -term

    return gradients


def compute_shell_thickness_gradients(plates, plate_thicknesses, shell_elements,
                                       all_nodes, u_total, E=1.0, nu=0.3):
    """Compute dC/d(thickness) for each plate.

    Each plate has a single thickness variable.  The gradient is the sum
    of per-triangle sensitivities for all triangles belonging to that plate:

        dC/dh_p = - sum_{tri in plate_p} u_tri^T (dK_tri/dh) u_tri

    Parameters
    ----------
    plates : list of dict — plate data (for indexing)
    plate_thicknesses : list of float — current thickness per plate
    shell_elements : list of dict — from solve_frame (shell_elements output)
    all_nodes : ndarray — expanded node array (beam + plate nodes)
    u_total : ndarray — full displacement vector from solve_frame
    E, nu : material properties

    Returns
    -------
    gradients : ndarray, shape (n_plates,) — dC/dh per plate
    """
    n_plates = len(plates)
    gradients = np.zeros(n_plates)

    for se in shell_elements:
        p_idx = se['plate_idx']
        h = se['thickness']
        dofs = se['dofs']
        g0, g1, g2 = se['vertex_global_ids']

        v0 = all_nodes[g0]
        v1 = all_nodes[g1]
        v2 = all_nodes[g2]

        u_e = u_total[dofs]

        dK = compute_shell_element_stiffness_derivative(E, nu, h, v0, v1, v2)
        gradients[p_idx] += -(u_e @ dK @ u_e)

    return gradients


# =========================================================================
# IGA Curved Beam Element (Timoshenko, cubic Bernstein, Gauss quadrature)
# =========================================================================

def _compute_local_frame(tangent):
    """Compute 3x3 rotation matrix R whose rows are [e1, e2, e3].

    Uses the same near-vertical handling as :func:`rotation_matrix`.
    """
    L = np.linalg.norm(tangent)
    if L < 1e-12:
        return np.eye(3)
    e1 = tangent / L
    Cx, Cy, Cz = e1
    if np.abs(Cz) > 0.999:
        R = np.array([
            [0, 0, np.sign(Cz)],
            [0, 1, 0],
            [-np.sign(Cz), 0, 0],
        ])
    else:
        D = np.sqrt(Cx ** 2 + Cy ** 2)
        R = np.array([
            [Cx, Cy, Cz],
            [-Cy / D, Cx / D, 0],
            [-Cx * Cz / D, -Cy * Cz / D, D],
        ])
    return R


def compute_iga_element_stiffness(E, r, p0, p1, p2, p3, nu, n_gauss=4):
    """Compute 24x24 IGA Timoshenko element stiffness via Gauss quadrature.

    Parameters
    ----------
    E : float — Young's modulus
    r : float — cross-section radius
    p0, p1, p2, p3 : (3,) arrays — Bezier control points
    nu : float — Poisson's ratio
    n_gauss : int — number of Gauss-Legendre quadrature points

    Returns
    -------
    K_full : (24, 24) ndarray — element stiffness in global coordinates
    """
    p0, p1, p2, p3 = (np.asarray(x, dtype=float) for x in (p0, p1, p2, p3))
    ctrl = np.array([p0, p1, p2, p3])  # (4, 3)

    # Section properties (circular cross-section)
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    G = E / (2.0 * (1.0 + nu))
    kappa_s = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)  # Timoshenko shear correction

    D_diag = np.array([E * A, kappa_s * G * A, kappa_s * G * A,
                        G * J, E * I, E * I])  # (6,)

    # Gauss-Legendre quadrature on [0, 1]
    xi_gl, w_gl = np.polynomial.legendre.leggauss(n_gauss)
    xi_pts = 0.5 * (xi_gl + 1.0)
    w_pts = 0.5 * w_gl

    K_full = np.zeros((24, 24))

    for g in range(n_gauss):
        xi = xi_pts[g]
        w = w_pts[g]

        # Bernstein basis values and first derivatives at xi
        N = bernstein_basis(xi)      # (4,)
        dN = bernstein_basis_d1(xi)  # (4,)

        # Tangent vector x'(xi) = sum dN_i * P_i
        x_prime = dN @ ctrl  # (3,)
        Jac = np.linalg.norm(x_prime)
        if Jac < 1e-14:
            continue

        # Arc-length derivatives of basis functions
        N_s = dN / Jac  # dBi/ds = dBi/dxi / J

        # Local frame at this Gauss point
        R = _compute_local_frame(x_prime)  # rows: e1, e2, e3
        e1 = R[0]
        e2 = R[1]
        e3 = R[2]

        # Build 6x24 strain-displacement matrix B = [B0 | B1 | B2 | B3]
        B = np.zeros((6, 24))
        for i in range(4):
            col = i * 6  # start column for control point i
            Ns = N_s[i]  # scalar: dBi/ds
            Ni = N[i]    # scalar: Bi(xi)

            # Translations (cols 0:3 of each 6-DOF block)
            B[0, col:col + 3] = Ns * e1             # axial strain
            B[1, col:col + 3] = Ns * e2             # shear gamma_2
            B[2, col:col + 3] = Ns * e3             # shear gamma_3

            # Rotations (cols 3:6 of each 6-DOF block)
            B[1, col + 3:col + 6] = -Ni * e3        # shear gamma_2 coupling
            B[2, col + 3:col + 6] = Ni * e2         # shear gamma_3 coupling
            B[3, col + 3:col + 6] = Ns * e1         # twist rate
            B[4, col + 3:col + 6] = Ns * e2         # bending curvature kappa_2
            B[5, col + 3:col + 6] = Ns * e3         # bending curvature kappa_3

        # K += w * B^T * D * B * J
        # D is diagonal so D @ B = D_diag[:, None] * B
        DB = D_diag[:, np.newaxis] * B  # (6, 24)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            contrib = B.T @ DB
        # Replace any NaN/inf from overflow with 0 (does not affect
        # well-conditioned beams; degenerate cases are caught downstream)
        contrib = np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
        K_full += w * Jac * contrib

    # Symmetrise (numerical noise)
    K_full = 0.5 * (K_full + K_full.T)
    return K_full


def compute_iga_element_stiffness_derivative(E, r, p0, p1, p2, p3, nu, n_gauss=4):
    """Compute dK_full/dr (24x24) for the IGA Timoshenko element.

    Since D is polynomial in r and B does not depend on r,
    dK/dr = integral of B^T * (dD/dr) * B * J dxi.
    """
    p0, p1, p2, p3 = (np.asarray(x, dtype=float) for x in (p0, p1, p2, p3))
    ctrl = np.array([p0, p1, p2, p3])

    G = E / (2.0 * (1.0 + nu))
    kappa_s = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)

    # dD/dr: derivatives of section properties w.r.t. r
    dA_dr = 2.0 * np.pi * r
    dI_dr = np.pi * r ** 3
    dJ_dr = 2.0 * np.pi * r ** 3
    dD_diag = np.array([E * dA_dr, kappa_s * G * dA_dr, kappa_s * G * dA_dr,
                         G * dJ_dr, E * dI_dr, E * dI_dr])

    xi_gl, w_gl = np.polynomial.legendre.leggauss(n_gauss)
    xi_pts = 0.5 * (xi_gl + 1.0)
    w_pts = 0.5 * w_gl

    dK = np.zeros((24, 24))

    for g in range(n_gauss):
        xi = xi_pts[g]
        w = w_pts[g]

        N = bernstein_basis(xi)
        dN = bernstein_basis_d1(xi)
        x_prime = dN @ ctrl
        Jac = np.linalg.norm(x_prime)
        if Jac < 1e-14:
            continue

        N_s = dN / Jac
        R = _compute_local_frame(x_prime)
        e1, e2, e3 = R[0], R[1], R[2]

        B = np.zeros((6, 24))
        for i in range(4):
            col = i * 6
            Ns = N_s[i]
            Ni = N[i]
            B[0, col:col + 3] = Ns * e1
            B[1, col:col + 3] = Ns * e2
            B[2, col:col + 3] = Ns * e3
            B[1, col + 3:col + 6] = -Ni * e3
            B[2, col + 3:col + 6] = Ni * e2
            B[3, col + 3:col + 6] = Ns * e1
            B[4, col + 3:col + 6] = Ns * e2
            B[5, col + 3:col + 6] = Ns * e3

        dDB = dD_diag[:, np.newaxis] * B
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            contrib = B.T @ dDB
        contrib = np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
        dK += w * Jac * contrib

    dK = 0.5 * (dK + dK.T)
    return dK


# Boundary / interior DOF indices for 24-DOF → 12-DOF condensation.
# P0 = DOFs 0-5, P1 = 6-11, P2 = 12-17, P3 = 18-23.
_BOUNDARY_DOFS = np.array([0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23])
_INTERIOR_DOFS = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])


def condense_element(K_full):
    """Statically condense a 24x24 element to 12x12 (P0 + P3 DOFs only).

    K_cond = K_bb - K_bi * K_ii^-1 * K_ib

    Returns (12, 12) condensed stiffness, or None if K_ii is singular.
    """
    K_bb = K_full[np.ix_(_BOUNDARY_DOFS, _BOUNDARY_DOFS)]
    K_bi = K_full[np.ix_(_BOUNDARY_DOFS, _INTERIOR_DOFS)]
    K_ib = K_full[np.ix_(_INTERIOR_DOFS, _BOUNDARY_DOFS)]
    K_ii = K_full[np.ix_(_INTERIOR_DOFS, _INTERIOR_DOFS)]

    cond_num = np.linalg.cond(K_ii)
    if cond_num > 1e12:
        print(f"     [IGA] WARNING: K_ii condition number {cond_num:.1e}, "
              "falling back to straight beam")
        return None

    K_cond = K_bb - K_bi @ np.linalg.solve(K_ii, K_ib)
    K_cond = 0.5 * (K_cond + K_cond.T)
    return K_cond


def recover_internal_displacements(K_full, u_boundary):
    """Recover P1/P2 displacement DOFs from boundary displacements.

    u_internal = -K_ii^-1 * K_ib * u_boundary

    Parameters
    ----------
    K_full : (24, 24) ndarray
    u_boundary : (12,) ndarray — [u_P0(6), u_P3(6)]

    Returns
    -------
    u_full : (24,) ndarray — [u_P0(6), u_P1(6), u_P2(6), u_P3(6)]
    """
    K_ii = K_full[np.ix_(_INTERIOR_DOFS, _INTERIOR_DOFS)]
    K_ib = K_full[np.ix_(_INTERIOR_DOFS, _BOUNDARY_DOFS)]
    u_int = -np.linalg.solve(K_ii, K_ib @ u_boundary)
    u_full = np.zeros(24)
    u_full[_BOUNDARY_DOFS] = u_boundary
    u_full[_INTERIOR_DOFS] = u_int
    return u_full


def solve_curved_frame(nodes, edges, radii, ctrl_pts, E=1.0, nu=0.3,
                        loads={}, bcs={}):
    """Solve a frame with mixed straight and curved (IGA) beams.

    Parameters
    ----------
    nodes, edges, radii, E, nu, loads, bcs : same as :func:`solve_frame`
    ctrl_pts : list of (2, 3) arrays or None per edge.
        ``ctrl_pts[i]`` is the [P1, P2] interior control points for edge i,
        or None to use the straight-beam element for that edge.

    Returns
    -------
    u_total : (N*6,) displacements
    compliance : float
    element_data : list of dicts with 'k_global'/'K_full'/'dofs' per element
    """
    n_nodes = len(nodes)
    n_dof = n_nodes * 6

    I_idx, J_idx, V_val = [], [], []
    G = E / (2.0 * (1.0 + nu))

    element_data = []

    for e_i, (u, v) in enumerate(edges[:, :2]):
        u, v = int(u), int(v)
        p_start, p_end = nodes[u], nodes[v]
        vec = p_end - p_start
        L = np.linalg.norm(vec)
        if L < 1e-6:
            element_data.append(None)
            continue

        r = radii[e_i]
        dof_indices = np.concatenate([
            np.arange(u * 6, u * 6 + 6),
            np.arange(v * 6, v * 6 + 6),
        ])

        cp = ctrl_pts[e_i] if ctrl_pts is not None else None

        if cp is not None:
            # IGA curved element
            p1_ctrl, p2_ctrl = cp[0], cp[1]
            K_full_24 = compute_iga_element_stiffness(
                E, r, p_start, p1_ctrl, p2_ctrl, p_end, nu)
            K_cond = condense_element(K_full_24)
            if K_cond is None:
                # Fallback to straight beam
                A = np.pi * r ** 2
                Iy = Iz = np.pi * r ** 4 / 4.0
                Jt = np.pi * r ** 4 / 2.0
                k_local = compute_element_stiffness(E, A, Iy, Iz, Jt, G, L)
                T = rotation_matrix(vec)
                k_global = T.T @ k_local @ T
                element_data.append({
                    'k_global': k_global, 'dofs': dof_indices,
                    'K_full': None, 'curved': False,
                })
            else:
                k_global = K_cond
                element_data.append({
                    'k_global': k_global, 'dofs': dof_indices,
                    'K_full': K_full_24, 'curved': True,
                })
        else:
            # Standard straight beam
            A = np.pi * r ** 2
            Iy = Iz = np.pi * r ** 4 / 4.0
            Jt = np.pi * r ** 4 / 2.0
            k_local = compute_element_stiffness(E, A, Iy, Iz, Jt, G, L)
            T = rotation_matrix(vec)
            k_global = T.T @ k_local @ T
            element_data.append({
                'k_global': k_global, 'dofs': dof_indices,
                'K_full': None, 'curved': False,
            })

        # Assemble into global sparse triplets
        for r_i in range(12):
            for c_j in range(12):
                val = k_global[r_i, c_j]
                if abs(val) > 1e-12:
                    I_idx.append(dof_indices[r_i])
                    J_idx.append(dof_indices[c_j])
                    V_val.append(val)

    K_global = sp.coo_matrix(
        (V_val, (I_idx, J_idx)), shape=(n_dof, n_dof)).tocsc()

    # Load vector
    F_global = np.zeros(n_dof)
    for n_id, load in loads.items():
        start = n_id * 6
        for i, val in enumerate(load):
            if i < 6:
                F_global[start + i] += val

    # BCs
    fixed_dofs = []
    for n_id, dofs in bcs.items():
        base = n_id * 6
        for d in dofs:
            fixed_dofs.append(base + d)
    fixed_dofs = np.array(fixed_dofs)
    free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

    K_free = K_global[free_dofs, :][:, free_dofs]
    F_free = F_global[free_dofs]

    print(f"     [FEA-IGA] Solving system with {len(free_dofs)} DOFs "
          f"({sum(1 for ed in element_data if ed and ed.get('curved'))} "
          f"curved elements)...")
    u_free = scipy.sparse.linalg.spsolve(K_free, F_free)

    u_total = np.zeros(n_dof)
    u_total[free_dofs] = u_free
    compliance = np.dot(F_global, u_total)

    return u_total, compliance, element_data


def compute_curved_size_gradients(nodes, edges, radii, ctrl_pts,
                                   u_total, E=1.0, nu=0.3):
    """Compute dC/dr for each element, supporting curved (IGA) beams.

    For curved beams: recovers internal displacements, then uses the full
    24-DOF adjoint formula with dK_full/dr.

    For straight beams: delegates to the standard E-B derivative.
    """
    gradients = np.zeros(len(edges))
    G = E / (2.0 * (1.0 + nu))

    for i, (u_idx, v_idx) in enumerate(edges[:, :2]):
        u_idx, v_idx = int(u_idx), int(v_idx)
        dofs = np.concatenate([
            np.arange(u_idx * 6, u_idx * 6 + 6),
            np.arange(v_idx * 6, v_idx * 6 + 6),
        ])
        u_boundary = u_total[dofs]  # (12,)

        p_start, p_end = nodes[u_idx], nodes[v_idx]
        vec = p_end - p_start
        L = np.linalg.norm(vec)
        if L < 1e-6:
            continue

        r = radii[i]
        cp = ctrl_pts[i] if ctrl_pts is not None else None

        if cp is not None:
            # IGA curved beam: recover internal DOFs, use full adjoint
            p1_ctrl, p2_ctrl = cp[0], cp[1]
            K_full = compute_iga_element_stiffness(
                E, r, p_start, p1_ctrl, p2_ctrl, p_end, nu)

            cond_num = np.linalg.cond(
                K_full[np.ix_(_INTERIOR_DOFS, _INTERIOR_DOFS)])
            if cond_num > 1e12:
                # Fallback to straight beam gradient
                dk_local = compute_element_stiffness_derivative(E, r, L, nu)
                T = rotation_matrix(vec)
                dK = T.T @ dk_local @ T
                gradients[i] = -(u_boundary @ dK @ u_boundary)
                continue

            u_full = recover_internal_displacements(K_full, u_boundary)

            dK_full = compute_iga_element_stiffness_derivative(
                E, r, p_start, p1_ctrl, p2_ctrl, p_end, nu)
            gradients[i] = -(u_full @ dK_full @ u_full)
        else:
            # Standard straight beam
            dk_local = compute_element_stiffness_derivative(E, r, L, nu)
            T = rotation_matrix(vec)
            dK = T.T @ dk_local @ T
            gradients[i] = -(u_boundary @ dK @ u_boundary)

    return gradients


def compute_curved_ctrl_gradients(nodes, edges, radii, ctrl_pts,
                                   u_total, E=1.0, nu=0.3, eps=1e-5):
    """Compute dC/dP for interior control points via semi-analytical FD.

    For each curved beam, perturbs each of the 6 ctrl_pt components
    (P1_x, P1_y, P1_z, P2_x, P2_y, P2_z) by +/-eps, recomputes the
    condensed stiffness, and applies the adjoint formula.

    Returns
    -------
    dC_dctrl : (M, 2, 3) ndarray — gradient for each beam's P1 and P2.
        Zero for straight beams (no ctrl_pts).
    """
    M = len(edges)
    dC_dctrl = np.zeros((M, 2, 3))

    for i, (u_idx, v_idx) in enumerate(edges[:, :2]):
        cp = ctrl_pts[i] if ctrl_pts is not None else None
        if cp is None:
            continue

        u_idx, v_idx = int(u_idx), int(v_idx)
        dofs = np.concatenate([
            np.arange(u_idx * 6, u_idx * 6 + 6),
            np.arange(v_idx * 6, v_idx * 6 + 6),
        ])
        u_b = u_total[dofs]  # (12,) boundary displacements

        p_start, p_end = nodes[u_idx], nodes[v_idx]
        r = radii[i]

        # Central finite difference on each of the 6 ctrl_pt components
        for cp_idx in range(2):        # P1 (0) or P2 (1)
            for axis in range(3):       # x, y, z
                cp_plus = [c.copy() for c in cp]
                cp_minus = [c.copy() for c in cp]
                cp_plus[cp_idx][axis] += eps
                cp_minus[cp_idx][axis] -= eps

                K_plus = compute_iga_element_stiffness(
                    E, r, p_start, cp_plus[0], cp_plus[1], p_end, nu)
                Kc_plus = condense_element(K_plus)

                K_minus = compute_iga_element_stiffness(
                    E, r, p_start, cp_minus[0], cp_minus[1], p_end, nu)
                Kc_minus = condense_element(K_minus)

                if Kc_plus is None or Kc_minus is None:
                    continue

                dKc = (Kc_plus - Kc_minus) / (2.0 * eps)
                dC_dctrl[i, cp_idx, axis] = -(u_b @ dKc @ u_b)

    return dC_dctrl
