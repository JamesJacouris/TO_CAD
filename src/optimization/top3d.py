import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time

def lk_H8(nu):
    """
    Generates the element stiffness matrix for an 8-node brick element.
    """
    A = np.array([
        [32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
        [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]
    ])
    k = 1/144 * A.T @ np.array([1, nu])
    
    K1 = np.array([
        [k[0], k[1], k[1], k[2], k[4], k[4]],
        [k[1], k[0], k[1], k[3], k[5], k[6]],
        [k[1], k[1], k[0], k[3], k[6], k[5]],
        [k[2], k[3], k[3], k[0], k[7], k[7]],
        [k[4], k[5], k[6], k[7], k[0], k[1]],
        [k[4], k[6], k[5], k[7], k[1], k[0]]
    ])
    K2 = np.array([
        [k[8], k[7], k[11], k[5], k[3], k[6]],
        [k[7], k[8], k[11], k[4], k[2], k[4]],
        [k[9], k[9], k[12], k[6], k[3], k[5]],
        [k[5], k[4], k[10], k[8], k[1], k[9]],
        [k[3], k[2], k[4], k[1], k[8], k[11]],
        [k[10], k[3], k[5], k[11], k[9], k[12]]
    ])
    K3 = np.array([
        [k[5], k[6], k[3], k[8], k[11], k[7]],
        [k[6], k[5], k[3], k[9], k[12], k[9]],
        [k[4], k[4], k[2], k[7], k[11], k[8]],
        [k[8], k[9], k[1], k[5], k[10], k[4]],
        [k[11], k[12], k[9], k[10], k[5], k[3]],
        [k[1], k[11], k[8], k[3], k[4], k[2]]
    ])
    K4 = np.array([
        [k[13], k[10], k[10], k[12], k[9], k[9]],
        [k[10], k[13], k[10], k[11], k[8], k[7]],
        [k[10], k[10], k[13], k[11], k[7], k[8]],
        [k[12], k[11], k[11], k[13], k[6], k[6]],
        [k[9], k[8], k[7], k[6], k[13], k[10]],
        [k[9], k[7], k[8], k[6], k[10], k[13]]
    ])
    K5 = np.array([
        [k[0], k[1], k[7], k[2], k[4], k[3]],
        [k[1], k[0], k[7], k[3], k[5], k[10]],
        [k[7], k[7], k[0], k[4], k[10], k[5]],
        [k[2], k[3], k[4], k[0], k[7], k[1]],
        [k[4], k[5], k[10], k[7], k[0], k[7]],
        [k[3], k[10], k[5], k[1], k[7], k[0]]
    ])
    K6 = np.array([
        [k[13], k[10], k[6], k[12], k[9], k[11]],
        [k[10], k[13], k[6], k[11], k[8], k[1]],
        [k[6], k[6], k[13], k[9], k[1], k[8]],
        [k[12], k[11], k[9], k[13], k[6], k[10]],
        [k[9], k[8], k[1], k[6], k[13], k[6]],
        [k[11], k[1], k[8], k[10], k[6], k[13]]
    ])
    
    KE = 1 / ((nu + 1) * (1 - 2 * nu)) * np.block([
        [K1, K2, K3, K4],
        [K2.T, K5, K6, K3.T],
        [K3.T, K6, K5.T, K2.T],
        [K4, K3, K2, K1.T]
    ])
    
    return KE

class Top3D:
    def __init__(self, nelx, nely, nelz, volfrac, penal, rmin):
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        
        self.nele = nelx * nely * nelz
        self.ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
        
        # User defined loads and BCs
        self.f = np.zeros((self.ndof, 1))
        self.fixed_dofs = np.array([], dtype=int)
        
        # BC Tags Grid (0=None, 1=Fixed, 2=Loaded)
        # Note: Matlab uses (nely, nelx, nelz). We will stick to (nely, nelx, nelz) to match.
        self.bc_tags = np.zeros((nely, nelx, nelz), dtype=np.uint8)
        
        self.x = np.full((nely, nelx, nelz), volfrac)
        self.xPhys = self.x.copy()
        
    def set_load(self, load_dof, magnitude):
        self.f[load_dof] = magnitude
        
        # Tag Loading Region
        # Map DOF -> Node -> Elements
        node_idx = load_dof // 3
        self._tag_elements_around_node(node_idx, tag_value=2)

    def set_fixed_dofs(self, fixed):
        self.fixed_dofs = fixed
        
        # Tag Fixed Regions
        unique_nodes = np.unique(fixed // 3)
        for nid in unique_nodes:
            self._tag_elements_around_node(nid, tag_value=1)

    def _tag_elements_around_node(self, node_idx, tag_value):
        """
        Marks elements touching the given node with tag_value.
        """
        nx, ny, nz = self.nelx, self.nely, self.nelz
        
        # Decompose Node Index
        # nID = k * (nx+1)*(ny+1) + i * (ny+1) + j
        slice_size = (nx + 1) * (ny + 1)
        k = node_idx // slice_size
        rem = node_idx % slice_size
        i = rem // (ny + 1)
        j = rem % (ny + 1)
        
        # Node (i,j,k) touches elements in ranges [i-1, i], [j-1, j], [k-1, k]
        # Element grid is (nely, nelx, nelz) -> (j, i, k) indices
        
        for dk in [-1, 0]:
            for di in [-1, 0]:
                for dj in [-1, 0]:
                    ek = k + dk
                    ei = i + di
                    ej = j + dj
                    
                    # Check bounds
                    if 0 <= ek < nz and 0 <= ei < nx and 0 <= ej < ny:
                        # Mark it
                        # bc_tags index order: (y, x, z) -> (ej, ei, ek)
                        self.bc_tags[ej, ei, ek] = tag_value

    def optimize(self, max_loop=200, tolx=0.01):
        KE = lk_H8(0.3)
        
        # Prepare Filter
        print("Preparing Filter...")
        H, Hs = self._prepare_filter()
        
        # Precompute EdofMat
        print("Preparing EdofMat...")
        edofMat, iK, jK = self._prepare_edof_mat()
        
        # Main Loop
        loop = 0
        change = 1.0
        
        print(f"Starting Optimization (Mesh: {self.nelx}x{self.nely}x{self.nelz})")
        
        start_time = time.time()
        
        while change > tolx and loop < max_loop:
            loop += 1
            
            # FE Analysis
            # sK = (KE(:) * (Emin + xPhys(:)'.^penal * (E0-Emin)))
            # Simplified: E0=1, Emin=1e-9 -> approx x^p
            
            # flattened xPhys (column-major to match MATLAB if needed, but here consisteny matters)
            # construction of iK/jK assumes edofMat order. 
            # edofMat was built iterating (k, i, j). 
            # xPhys is (ny, nx, nz). 
            # We need to flatten xPhys in the SAME order as the element iteration in edofMat.
            # In edofMat calc: for k.. for i.. for j..
            # xPhys[j, i, k] correspond to that order.
            # So xPhys.flatten(order='F') walks j, then i, then k?
            # Numpy 'F' order: first index changes fastest.
            # xPhys shape (ny, nx, nz). 'F' order: j changes, then i, then k.
            # YES. This matches edofMat loop order!
            
            x_flat = self.xPhys.flatten(order='F')
            
            # E0=1, Emin=1e-9
            Emin = 1e-9
            E0 = 1.0
            
            # sK shape: (24*24) x nele
            # We want 1D array of all values
            filt_stiff = Emin + x_flat**self.penal * (E0 - Emin) 
            sK = (KE.flatten()[:, np.newaxis] @ filt_stiff[np.newaxis, :]).flatten(order='F')
            
            K = sp.coo_matrix((sK, (iK, jK)), shape=(self.ndof, self.ndof)).tocsc()
            K = (K + K.T) / 2.0
            
            # Solve
            # Partitioning
            free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
            
            # Solve K_ff * u_f = f_f
            # Note: For large systems, use iterative solver or cholesky?
            # spsolve is direct LU. Fine for small/med.
            
            u_free = spsolve(K[free_dofs, :][:, free_dofs], self.f[free_dofs])
            
            u = np.zeros(self.ndof)
            u[free_dofs] = u_free
            
            # Objective & Sensitivity
            # ce = sum((U(edofMat)*KE).*U(edofMat), 2)
            
            # u[edofMat] shape: (nele, 24)
            u_ele = u[edofMat] 
            
            # (u_ele @ KE) * u_ele -> element-wise mult sum?
            # Matlab: sum((U(edofMat)*KE).*U(edofMat), 2)
            # U*KE is matrix mult (nele x 24) * (24 x 24) -> (nele x 24)
            # .* is elementwise. sum(, 2) is sum rows.
            
            ce = np.sum((u_ele @ KE) * u_ele, axis=1)
            c = np.sum((Emin + x_flat**self.penal * (E0 - Emin)) * ce)
            
            dc = -self.penal * (E0 - Emin) * x_flat**(self.penal - 1) * ce
            dv = np.ones(self.nele)
            
            # Filtering Sensitivities
            # dc(:) = H*(dc(:)./Hs);
            dc[:] = H @ (dc / Hs)
            dv[:] = H @ (dv / Hs)
            
            # Optimality Criteria
            xnew_flat = self._optimality_criteria(x_flat, dc, dv)
            xnew = xnew_flat.reshape((self.nely, self.nelx, self.nelz), order='F')
            
            # Filter Design Variable
            # xPhys(:) = (H*xnew(:))./Hs
            xPhys_flat = H @ xnew_flat / Hs
            self.xPhys = xPhys_flat.reshape((self.nely, self.nelx, self.nelz), order='F')
            
            change = np.max(np.abs(xnew - self.x))
            self.x = xnew
            
            print(f" It.: {loop:4d} Obj.: {c:10.4f} Vol.: {np.mean(self.xPhys):6.3f} ch.: {change:6.3f}")
            
        print(f"Optimization Converged in {time.time() - start_time:.2f}s")
        return self.xPhys

    def _prepare_edof_mat(self):
        # Python port of edofMat generation
        # Careful with 0-based indexing and C vs F ordering
        
        # Node Grid (nodegrd in matlab)
        # In MATLAB: reshape(1:(nely+1)*(nelx+1), nely+1, nelx+1)
        # Means node IDs increase down columns (Y), then across rows (X).
        
        nx, ny, nz = self.nelx, self.nely, self.nelz
        
        # 3D Node Grid
        # Create 3D array of Node IDs matching MATLAB ordering
        node_ids = np.arange((ny + 1) * (nx + 1) * (nz + 1)).reshape((nx+1, nz+1, ny+1)) 
        # Wait, Matlab:
        # nodegrd is (nely+1) x (nelx+1). 
        # nodeids = reshape(nodegrd(1:end-1,1:end-1), nely*nelx, 1)
        # This implies standard column-major element walking.
        
        # Let's replicate MATLAB's vector math exactly for safety
        # Elements are indexed 1..nele walking Y, then X, then Z
        
        elx = np.repeat(np.arange(nx), ny)
        ely = np.tile(np.arange(ny), nx)
        # elx, ely for one slice (Z=0)
        
        # Nodes for element at (x,y,z)
        # n1 = (nely+1)*x + y + 1 (in matlab 1-based, here 0-based: (nely+1)*x + y)
        # Let's verify.
        
        n_per_slice = (nx + 1) * (ny + 1)
        
        edofMat = np.zeros((self.nele, 24), dtype=int)
        
        elem_idx = 0
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    # Base node (top-left-front of element)
                    # Coordinates in node grid: i, j, k
                    # ID = k * n_per_slice + i * (ny + 1) + j
                    
                    n1 = k * n_per_slice + i * (ny + 1) + j
                    n2 = k * n_per_slice + (i + 1) * (ny + 1) + j
                    n3 = k * n_per_slice + (i + 1) * (ny + 1) + (j + 1)
                    n4 = k * n_per_slice + i * (ny + 1) + (j + 1)
                    
                    n5 = (k + 1) * n_per_slice + i * (ny + 1) + j
                    n6 = (k + 1) * n_per_slice + (i + 1) * (ny + 1) + j
                    n7 = (k + 1) * n_per_slice + (i + 1) * (ny + 1) + (j + 1)
                    n8 = (k + 1) * n_per_slice + i * (ny + 1) + (j + 1)
                    
                    # DOFs: 3*n, 3*n+1, 3*n+2
                    nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
                    dofs = []
                    for n in nodes:
                        dofs.extend([3*n, 3*n+1, 3*n+2])
                        
                    edofMat[elem_idx, :] = dofs
                    elem_idx += 1
                    
        iK = np.kron(edofMat, np.ones((24, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 24))).flatten()
        
        return edofMat, iK, jK

    def _prepare_filter(self):
        # Standard distance filter
        nele = self.nele
        nx, ny, nz = self.nelx, self.nely, self.nelz
        rmin = self.rmin
        
        iH = []
        jH = []
        sH = []
        
        # To speed up, stick to loops or numba. Pure python loop sets are slow for large mesh.
        # But for 60x20x4 type meshes, it's fine.
        
        # Using element centroid distance
        # Current element e1 at (x,y,z)
        # Neighbor e2 at (i,j,k)
        
        elem_idx = 0
        for k1 in range(nz):
            for i1 in range(nx):
                for j1 in range(ny):
                    e1 = elem_idx
                    
                    # Search window
                    min_i = max(i1 - int(np.floor(rmin)) - 1, 0)
                    max_i = min(i1 + int(np.floor(rmin)) + 1, nx)
                    min_j = max(j1 - int(np.floor(rmin)) - 1, 0)
                    max_j = min(j1 + int(np.floor(rmin)) + 1, ny)
                    min_k = max(k1 - int(np.floor(rmin)) - 1, 0)
                    max_k = min(k1 + int(np.floor(rmin)) + 1, nz)
                    
                    for k2 in range(min_k, max_k):
                        for i2 in range(min_i, max_i):
                            for j2 in range(min_j, max_j):
                                e2 = k2 * (nx * ny) + i2 * ny + j2
                                dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2 + (k1 - k2)**2)
                                
                                if dist < rmin:
                                    iH.append(e1)
                                    jH.append(e2)
                                    sH.append(rmin - dist)
                                    
                    elem_idx += 1
                    
        H = sp.coo_matrix((sH, (iH, jH)), shape=(nele, nele)).tocsc()
        Hs = np.array(H.sum(axis=1)).flatten()
        return H, Hs

    def _optimality_criteria(self, x, dc, dv):
        l1 = 0
        l2 = 1e9
        move = 0.2
        
        xnew = np.zeros_like(x)
        
        # Clip sensitivity to ensure non-negative under square root
        # (dc should be negative, so -dc should be positive)
        # dv should be positive (volume derivative)
        sensitivity = np.maximum(1e-10, -dc / dv)

        while (l2 - l1) > 1e-3 * (l1 + l2):
            lmid = 0.5 * (l2 + l1)
            
            # xnew = max(0, max(x - move, min(1, min(x + move, x * sqrt(-dc / dv / lmid)))))
            # Term B: x * sqrt(...)
            # Avoid division by zero if lmid is extremely small
            if lmid < 1e-20:
                 # If lambda is highly permissive, we tend to max density
                 term_B = np.ones_like(x) 
            else:
                 term_B = x * np.sqrt(sensitivity / lmid)
            
            # Min 1 with x+move
            term_C = np.minimum(x + move, term_B)
            term_D = np.minimum(1.0, term_C)
            
            # Max with x-move
            term_E = np.maximum(x - move, term_D)
            
            # Max with 0
            xnew = np.maximum(0.0, term_E)
            
            if np.sum(xnew) > self.volfrac * self.nele:
                l1 = lmid
            else:
                l2 = lmid
                
        return xnew
