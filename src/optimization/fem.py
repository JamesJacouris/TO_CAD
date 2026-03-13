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

def solve_frame(nodes, edges, radii, E=1.0, nu=0.3, loads={}, bcs={}):
    """Solve a 3-D Euler-Bernoulli frame under static loading.

    Assembles the global stiffness matrix ``K`` from per-element local
    stiffness matrices rotated into the global frame, applies Dirichlet BCs
    by zeroing constrained rows/columns, and solves ``K_ff u_f = f_f`` with
    ``scipy.sparse.linalg.spsolve``.

    Args:
        nodes (numpy.ndarray): Node positions, shape ``(N, 3)``, mm.
        edges (numpy.ndarray): Element connectivity, shape ``(M, 2)``,
            integer node indices.
        radii (numpy.ndarray): Circular cross-section radii per element,
            shape ``(M,)``, mm.
        E (float): Young's modulus (consistent units with nodes/radii).
        nu (float): Poisson's ratio (used for shear modulus
            ``G = E / (2(1+nu))``).
        loads (dict): ``{node_idx: [Fx, Fy, Fz, Mx, My, Mz]}``.
        bcs (dict): ``{node_idx: [dof_0, dof_1, …]}``, list of fixed DOF
            indices (0–5 per node).

    Returns:
        tuple:
            - **u** (``numpy.ndarray``, shape ``(N*6,)``): Full displacement
              vector (translations + rotations per node).
            - **compliance** (``float``): Structural compliance
              ``c = f^T u`` (lower = stiffer).
            - **f** (``numpy.ndarray``, shape ``(N*6,)``): Applied force
              vector.
    """
    n_nodes = len(nodes)
    n_dof = n_nodes * 6
    
    # Initialize Sparse Matrix Builder
    I_idx, J_idx, V_val = [], [], []
    
    # Shear Modulus
    G = E / (2 * (1 + nu))
    
    # Element Data Storage (for efficient gradient calc later)
    elements = []
    
    for e_i, (u, v) in enumerate(edges[:, :2]):
        u, v = int(u), int(v)
        p1, p2 = nodes[u], nodes[v]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        
        if L < 1e-6: continue
        
        r = radii[e_i]
        A = np.pi * r**2
        Iy = Iz = np.pi * r**4 / 4
        J = np.pi * r**4 / 2
        
        # Local Stiffness
        k_local = compute_element_stiffness(E, A, Iy, Iz, J, G, L)
        
        # Rotation
        T = rotation_matrix(vec)
        
        # Global Stiffness: K = T.T * k * T
        k_global = T.T @ k_local @ T
        
        # Assemble
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
                    
        # Store for compliance tracking
        elements.append({
            'k_global': k_global,
            'dofs': dof_indices
        })
                    
    # Build K
    K_global = sp.coo_matrix((V_val, (I_idx, J_idx)), shape=(n_dof, n_dof)).tocsc()
    
    # Load Vector
    F_global = np.zeros(n_dof)
    for n_id, load in loads.items():
        # load can be size 3 (Force) or 6 (Moment)
        start = n_id * 6
        for i, val in enumerate(load):
             if i < 6: F_global[start + i] += val
             
    # Apply BCs (Penalty Method or Partitioning)
    # Using simple partitioning (masking free DOFs)
    fixed_dofs = []
    for n_id, dofs in bcs.items():
        base = n_id * 6
        for d in dofs:
            fixed_dofs.append(base + d)
            
    fixed_dofs = np.array(fixed_dofs)
    free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)
    
    # Solve Reduced System
    K_free = K_global[free_dofs, :][:, free_dofs]
    F_free = F_global[free_dofs]
    
    print(f"     [FEA] Solving System with {len(free_dofs)} DOFs...")
    u_free = scipy.sparse.linalg.spsolve(K_free, F_free)
    
    # Full Displacement Vector
    u_total = np.zeros(n_dof)
    u_total[free_dofs] = u_free
    
    # Compute Compliance
    compliance = np.dot(F_global, u_total)
    
    return u_total, compliance, elements

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
