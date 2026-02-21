import numpy as np

def compute_cst_stiffness(nodes_2d, E, nu, t):
    """
    Computes 6x6 membrane stiffness matrix for Constant Strain Triangle (CST).
    nodes_2d: (3, 2) local coordinates [[x1,y1], [x2,y2], [x3,y3]]
    """
    x, y = nodes_2d[:, 0], nodes_2d[:, 1]
    
    # Area
    A = 0.5 * abs(x[0]*(y[1] - y[2]) + x[1]*(y[2] - y[0]) + x[2]*(y[0] - y[1]))
    if A < 1e-12: return np.zeros((6, 6))

    # B matrix (Strain-Displacement)
    b = y[1] - y[2]; c = x[2] - x[1]
    d = y[2] - y[0]; e = x[0] - x[2]
    f = y[0] - y[1]; g = x[1] - x[0]
    
    B = np.array([
        [b, 0, d, 0, f, 0],
        [0, c, 0, e, 0, g],
        [c, b, e, d, g, f]
    ]) / (2 * A)
    
    # D matrix (Constitutive - Plane Stress)
    factor = E / (1 - nu**2)
    D = factor * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    
    # K = t * A * B.T * D * B
    K_mem = t * A * (B.T @ D @ B)
    
    return K_mem

def compute_dkt_stiffness(nodes_2d, E, nu, t):
    """
    Computes 9x9 bending stiffness matrix for a triangle.
    Uses a Mindlin plate formulation with 3-point integration.
    nodes_2d: (3, 2) local coordinates
    DOFs: [w1, tx1, ty1, w2, tx2, ty2, w3, tx3, ty3]
    """
    x, y = nodes_2d[:, 0], nodes_2d[:, 1]
    A = 0.5 * abs(x[0]*(y[1] - y[2]) + x[1]*(y[2] - y[0]) + x[2]*(y[0] - y[1]))
    if A < 1e-12: return np.zeros((9, 9))

    # CST Gradient coefficients
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / (2*A)
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / (2*A)
    
    # Bending D Matrix
    D_fac = (E * t**3) / (12 * (1 - nu**2))
    D_b = D_fac * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    
    # Shear D Matrix
    G = E / (2 * (1 + nu))
    ks = 5.0/6.0
    D_s = np.eye(2) * G * t * ks

    # B_bending matrix (Constant for linear beta interpolation)
    B_b = np.zeros((3, 9))
    for i in range(3):
        # kappa_x = d(-theta_y)/dx
        B_b[0, 3*i + 2] = -b[i]
        # kappa_y = d(theta_x)/dy
        B_b[1, 3*i + 1] = c[i]
        # kappa_xy = d(-theta_y)/dy + d(theta_x)/dx
        B_b[2, 3*i + 2] = -c[i]
        B_b[2, 3*i + 1] = b[i]

    K_bend = np.zeros((9, 9))
    
    # Integration (Hammer 3-point)
    p_int = [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]
    w_int = 1.0 / 3.0
    
    for pt in p_int:
        B_s_pt = np.zeros((2, 9))
        for i in range(3):
            Ni = pt[i]
            B_s_pt[0, 3*i] = b[i]
            B_s_pt[1, 3*i] = c[i]
            B_s_pt[0, 3*i+2] = Ni
            B_s_pt[1, 3*i+1] = -Ni
            
        K_s_pt = A * (B_s_pt.T @ D_s @ B_s_pt)
        K_b_pt = A * (B_b.T @ D_b @ B_b)
        K_bend += w_int * (K_b_pt + K_s_pt)
        
    return K_bend

def compute_shell_stiffness_matrix(nodes, t, E, nu):
    """
    Computes 18x18 global stiffness matrix for a triangular shell element.
    """
    p1, p2, p3 = nodes[0], nodes[1], nodes[2]
    v12 = p2 - p1
    v13 = p3 - p1
    
    n = np.cross(v12, v13)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12: return np.zeros((18, 18))
    
    lz = n / norm_n
    lx = v12 / np.linalg.norm(v12)
    ly = np.cross(lz, lx)
    
    R = np.vstack([lx, ly, lz])
    
    # Projection
    nodes_2d = np.zeros((3, 2))
    nodes_2d[0] = [0, 0]
    nodes_2d[1] = [np.dot(p2-p1, lx), 0]
    nodes_2d[2] = [np.dot(p3-p1, lx), np.dot(p3-p1, ly)]
    
    # Local K
    k_mem = compute_cst_stiffness(nodes_2d, E, nu, t)
    k_bend = compute_dkt_stiffness(nodes_2d, E, nu, t)
    
    kl = np.zeros((18, 18))
    mem_indices = [0, 1, 6, 7, 12, 13]
    for i in range(6):
        for j in range(6):
            kl[mem_indices[i], mem_indices[j]] = k_mem[i, j]
            
    bend_indices = [2, 3, 4, 8, 9, 10, 14, 15, 16]
    for i in range(9):
        for j in range(9):
            kl[bend_indices[i], bend_indices[j]] = k_bend[i, j]
            
    # Drilling Stiffness
    Area = norm_n / 2.0
    k_drill = 0.001 * E * t * Area
    drill_indices = [5, 11, 17]
    for idx in drill_indices:
        kl[idx, idx] = k_drill
        
    # Transform
    T = np.zeros((18, 18))
    for i in range(6):
        T[i*3:(i+1)*3, i*3:(i+1)*3] = R
        
    return T.T @ kl @ T
