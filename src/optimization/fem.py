"""3-D Euler-Bernoulli frame finite element analysis.

Assembles a global sparse stiffness matrix from circular cross-section beam
elements, applies boundary conditions, solves for displacements and
compliance, and computes per-element compliance sensitivities for gradient-
based optimisation.

Main entry points
-----------------
- :func:`solve_frame` — static analysis → ``(displacements, compliance, forces)``
- :func:`compute_frame_gradients` — ``dC/dr`` per edge
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

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
