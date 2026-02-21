import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.spatial import cKDTree
from .fem import compute_element_stiffness, rotation_matrix
from .shell_element import compute_shell_stiffness_matrix

def solve_coupled_system(nodes, beam_edges, beam_radii, plate_tris, plate_thickness, loads={}, bcs={}, E=1.0, nu=0.3):
    """
    Assembles and solves a unified stiffness matrix for mixed beam-shell systems.
    
    nodes: (N, 3) coordinates
    beam_edges: (M_b, 2) connectivity
    beam_radii: (M_b,) radii
    plate_tris: (M_p, 3) triangle indices
    plate_thickness: scalar or (M_p,) thickness
    """
    n_nodes = len(nodes)
    n_dof = n_nodes * 6
    I_idx, J_idx, V_val = [], [], []
    
    # Shear Modulus
    G = E / (2 * (1 + nu))
    
    # --- 1. Assemble Beams ---
    for e_i, (u, v) in enumerate(beam_edges):
        u, v = int(u), int(v)
        p1, p2 = nodes[u], nodes[v]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        if L < 1e-6: continue
        
        # Handle out-of-bounds radii for bridge edges
        if e_i < len(beam_radii):
            r = beam_radii[e_i]
        else:
            r = 1.0 # Default radius for bridge edges

        A = np.pi * r**2
        Iy = Iz = (np.pi * r**4) / 4.0
        J = (np.pi * r**4) / 2.0
        
        k_local = compute_element_stiffness(E, A, Iy, Iz, J, G, L)
        T = rotation_matrix(vec) # 12x12
        k_global = T.T @ k_local @ T
        
        # DOF Indices
        dof_indices = np.concatenate([
            np.arange(u*6, u*6+6),
            np.arange(v*6, v*6+6)
        ])
        
        rows, cols = np.meshgrid(dof_indices, dof_indices)
        I_idx.extend(rows.flatten())
        J_idx.extend(cols.flatten())
        V_val.extend(k_global.flatten())

    # --- 2. Assemble Plates ---
    if np.isscalar(plate_thickness):
        t_vals = np.full(len(plate_tris), plate_thickness)
    else:
        t_vals = plate_thickness
        
    for p_i, tri in enumerate(plate_tris):
        n1, n2, n3 = int(tri[0]), int(tri[1]), int(tri[2])
        el_nodes = nodes[[n1, n2, n3]]
        t = t_vals[p_i]
        
        k_shell_global = compute_shell_stiffness_matrix(el_nodes, t, E, nu)
        
        dof_indices = np.concatenate([
            np.arange(n1*6, n1*6+6),
            np.arange(n2*6, n2*6+6),
            np.arange(n3*6, n3*6+6)
        ])
        
        rows, cols = np.meshgrid(dof_indices, dof_indices)
        I_idx.extend(rows.flatten())
        J_idx.extend(cols.flatten())
        V_val.extend(k_shell_global.flatten())

    # --- 3. Build Global Systems ---
    K_global = sp.coo_matrix((V_val, (I_idx, J_idx)), shape=(n_dof, n_dof)).tocsc()
    
    # --- Soft Grounding ---
    # Apply small stabilization to all DOFs to handle rigid body modes of disconnected parts
    grounding = sp.diags([E * 1e-10] * n_dof, format='csc')
    K_global += grounding
    
    # Load Vector
    F_global = np.zeros(n_dof)
    for n_id, load in loads.items():
        start = n_id * 6
        for i, val in enumerate(load):
            if i < 6: F_global[start + i] += val
            
    # --- 4. Apply BCs ---
    fixed_dofs = []
    for n_id, dofs in bcs.items():
        base = n_id * 6
        for d in dofs:
            fixed_dofs.append(base + d)
    fixed_dofs = np.unique(fixed_dofs)
    free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)
    
    # --- 5. Solve ---
    if len(free_dofs) == 0:
        return np.zeros(n_dof), 0.0
        
    K_free = K_global[free_dofs, :][:, free_dofs]
    F_free = F_global[free_dofs]
    
    try:
        u_free = scipy.sparse.linalg.spsolve(K_free, F_free)
    except Exception:
        return np.zeros(n_dof), 0.0
    
    u_total = np.zeros(n_dof)
    u_total[free_dofs] = u_free
    compliance = np.dot(F_global, u_total)
    
    return u_total, compliance

def compute_coupled_gradients(nodes, beam_edges, beam_radii, u_total, E=1.0, nu=0.3):
    """
    Computes compliance gradients with respect to beam radii.
    dC/dr = -u^T * (dK/dr) * u
    Currently only supports gradients for beams (plate thickness constant).
    """
    n_beams = len(beam_edges)
    gradients = np.zeros(n_beams)
    G = E / (2 * (1 + nu))
    
    for e_i, (u, v) in enumerate(beam_edges):
        u, v = int(u), int(v)
        p1, p2 = nodes[u], nodes[v]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        if L < 1e-6: continue
        
        # Handle out-of-bounds radii
        if e_i < len(beam_radii):
            r = beam_radii[e_i]
        else:
            r = 1.0 # Default radius
        
        # dK_local/dr
        dA_dr = 2 * np.pi * r
        dI_dr = np.pi * r**3
        dJ_dr = 2 * np.pi * r**3
        
        # We reuse compute_element_stiffness by passing derivative values instead of E*A...
        # Local stiffness is linear in [EA, EI, GJ]. 
        # So dK/dr = K(E*dA_dr, E*dI_dr, G*dJ_dr)
        dk_local = compute_element_stiffness(1.0, dA_dr, dI_dr, dI_dr, dJ_dr, 1.0, L)
        
        T = rotation_matrix(vec)
        dk_global = T.T @ dk_local @ T
        
        u_el = np.concatenate([
            u_total[u*6 : u*6+6],
            u_total[v*6 : v*6+6]
        ])
        
        gradients[e_i] = -np.dot(u_el, dk_global @ u_el)
        
    return gradients

def merge_meshes(beam_nodes, beam_edges, plates_data, beam_node_tags=None, tol=0.5):
    """
    Merges beam nodes with plate mid-surface meshes, handling node re-indexing.
    Consolidates beam_node_tags with any tags found in plate mid-surfaces.

    Returns:
       new_nodes, new_beam_edges, new_plate_tris, all_thickness, merged_node_tags
    """
    nodes = list(beam_nodes)
    n_beam_orig = len(nodes)
    plate_tri_indices = []
    plate_thicknesses = []
    plate_node_tags = {} # global_idx -> tag
    
    n_current = n_beam_orig
    
    for p in plates_data:
        if 'mid_surface' not in p: continue
        ms = p['mid_surface']
        p_tris = np.array(ms['triangles']) + n_current
        nodes.extend(ms['vertices'])
        plate_tri_indices.append(p_tris)
        plate_thicknesses.extend([ms['mean_thickness']] * len(p_tris))
        
        # Capture tags from mid-surface
        if 'node_tags' in ms:
            for local_nid, tag in ms['node_tags'].items():
                global_nid = n_current + int(local_nid)
                plate_node_tags[global_nid] = tag
                
        n_current += len(ms['vertices'])
        
    all_nodes = np.array(nodes)
    
    # Init merged tags with beam tags
    merged_tags = {}
    if beam_node_tags:
        for nid, tag in beam_node_tags.items():
            merged_tags[int(nid)] = tag
            
    # Add plate tags
    for nid, tag in plate_node_tags.items():
        if nid not in merged_tags:
            merged_tags[nid] = tag
        # If both exist, we could decide precedence. Original tags should align.
    
    if not plate_tri_indices:
        return all_nodes, beam_edges, np.zeros((0,3), dtype=int), np.zeros(0), merged_tags

    all_tris = np.vstack(plate_tri_indices)
    all_thickness = np.array(plate_thicknesses)
    
    # Fuse duplicates
    kdt = cKDTree(all_nodes)
    pairs = kdt.query_pairs(r=tol)
    remap = np.arange(len(all_nodes))
    for i, j in pairs:
        root_i = remap[i]; root_j = remap[j]
        if root_i != root_j:
            target = min(root_i, root_j)
            remap[remap == root_i] = target
            remap[remap == root_j] = target
            
        
    unique_ids, inv_map = np.unique(remap, return_inverse=True)
    new_nodes = all_nodes[unique_ids]
    new_beam_edges = inv_map[beam_edges]
    new_plate_tris = inv_map[all_tris]
    
    # Remap Merged Tags
    final_merged_tags = {}
    for old_nid, tag in merged_tags.items(): # Original code used 'merged_tags' here
        new_nid = inv_map[old_nid]
        if new_nid not in final_merged_tags:
            final_merged_tags[int(new_nid)] = tag

    # Robust Anchoring: Force fix any node near Z=0 (the floor)
    # This addresses cases where skeletonization doesn't perfectly reach the Z=0.5 voxel center
    z_min_all = np.min(new_nodes[:, 2])
    for i, p in enumerate(new_nodes):
        if p[2] < z_min_all + 0.1: # Very close to floor
            if i not in final_merged_tags or final_merged_tags[i] != 2: # Don't override loads
                final_merged_tags[int(i)] = 1 # Force FIXED

    # --- Connectivity Diagnosis ---
    # Build an adjacency list for the merged system
    adj = {i: set() for i in range(len(new_nodes))}
    for u, v in new_beam_edges:
        adj[u].add(v); adj[v].add(u)
    for t in new_plate_tris:
        u, v, w = t
        adj[u].add(v); adj[v].add(u); adj[u].add(w); adj[w].add(u); adj[v].add(w); adj[w].add(v)
    
    # Simple BFS to find components
    visited = set()
    components = []
    for i in range(len(new_nodes)):
        if i not in visited:
            comp = set()
            stack = [i]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr); comp.add(curr)
                    stack.extend(adj[curr] - visited)
            components.append(comp)
    
    if len(components) > 1:
        print(f"[Warning] Merged System has {len(components)} disconnected components.")
        # Attempt minimal bridging between distinct components
        for c_idx in range(1, len(components)):
            comp_nodes = list(components[c_idx])
            base_nodes = list(components[0])
            
            c_coords = new_nodes[comp_nodes]
            b_coords = new_nodes[base_nodes]
            
            # Find closest pair between this component and the "main" component (idx 0)
            from scipy.spatial.distance import cdist
            dists = cdist(c_coords, b_coords)
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            min_dist = dists[min_idx]
            
            if min_dist < 10.0: # Bridge if within 10mm
                u_idx = comp_nodes[min_idx[0]]
                v_idx = base_nodes[min_idx[1]]
                print(f"          [Bridge] Adding connection between node {u_idx} and node {v_idx} (dist={min_dist:.2f})")
                # Add a stiff connecting edge
                new_beam_edges = np.vstack([new_beam_edges, [u_idx, v_idx]])

    # Re-apply Robust Anchoring to ALL components
    z_min_all = np.min(new_nodes[:, 2])
    for i, p in enumerate(new_nodes):
        if p[2] < z_min_all + 0.1: # Very close to floor
            if i not in final_merged_tags or final_merged_tags[i] != 2:
                final_merged_tags[int(i)] = 1

    return new_nodes, new_beam_edges, new_plate_tris, all_thickness, final_merged_tags, inv_map[:n_beam_orig]
