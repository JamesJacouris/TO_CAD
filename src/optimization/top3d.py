import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, cg, LinearOperator
from numba import njit, prange
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
        [k[12], k[11], np.array(k[9]), k[13], k[6], k[10]],
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

    def set_passive_void(self, passive):
        """
        Mark voxels as permanently void (forced to density = 0).

        Args:
            passive: boolean ndarray of shape (nely, nelx, nelz), True = forced void.
                     Must be called before optimize().
        """
        self._passive_mask = passive.flatten(order='F').astype(bool)
        # Initialise those elements to zero so they don't count toward volfrac
        self.x.flatten(order='F')[self._passive_mask] = 0.0
        self.xPhys[passive] = 0.0
        # Tag passive voxels in bc_tags (tag=3)
        self.bc_tags[passive] = 3

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
        """Run the topology optimisation loop (OC method).

        Iteratively updates element densities using an optimality criteria (OC)
        update scheme with a density filter.  Convergence is declared when the
        maximum density change between successive iterations falls below
        ``tolx``.

        Args:
            max_loop (int): Maximum number of OC iterations.  Defaults to 200.
            tolx (float): Convergence threshold on max density change.
                Iteration stops when ``change <= tolx``.  Defaults to 0.01.

        Returns:
            numpy.ndarray: Physical density field ``xPhys`` with shape
            ``(nely, nelx, nelz)``, values in ``[0, 1]``.  Voxels above the
            volume fraction threshold represent retained material.
        """
        KE = lk_H8(0.3)
        
        # Prepare Filter (Vectorized KD-Tree)
        print("Preparing Filter...")
        t0 = time.time()
        H, Hs = self._prepare_filter()
        print(f"Filter prepared in {time.time()-t0:.2f}s")
        
        # Precompute EdofMat (Vectorized)
        print("Preparing EdofMat...")
        edofMat, iK, jK = self._prepare_edof_mat_vectorized()
        
        # Main Loop
        loop = 0
        change = 1.0
        
        print(f"Starting Optimization (Mesh: {self.nelx}x{self.nely}x{self.nelz})")

        start_time = time.time()
        self.compliance_history = []

        while change > tolx and loop < max_loop:
            loop += 1
            
            # 1. Setup Stiffness Matrix
            # xPhys is (ny, nx, nz), flatten F-order matches element iteration
            x_flat = self.xPhys.flatten(order='F')
            
            Emin = 1e-9
            E0 = 1000.0  # Match frame FEM (size_opt / layout_opt default E=1000)
            
            # Element stiffness scaling
            filt_stiff = Emin + x_flat**self.penal * (E0 - Emin) 
            
            # Efficient K construction using broadcasting
            sK = (KE.flatten()[:, np.newaxis] @ filt_stiff[np.newaxis, :]).flatten(order='F')
            K = sp.coo_matrix((sK, (iK, jK)), shape=(self.ndof, self.ndof)).tocsc()
            K = (K + K.T) / 2.0
            
            # 2. Solve System: K * u = f
            free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
            
            # Use Iterative Solver (CG) with Jacobi Preconditioner
            K_free = K[free_dofs, :][:, free_dofs]
            f_free = self.f[free_dofs].flatten()
            
            # u_free = spsolve(K[free_dofs, :][:, free_dofs], self.f[free_dofs])
            
            K_ff = K[free_dofs, :][:, free_dofs]
            f_f = self.f[free_dofs]
            
            # Simple Jacobi Preconditioner
            diag_K = K_ff.diagonal()
            # Avoid division by zero
            diag_K[diag_K == 0] = 1.0
            M_inv = sp.diags(1.0 / diag_K)
            
            # Solve using Conjugate Gradient
            u_free, info = cg(K_ff, f_f, M=M_inv, rtol=1e-5, maxiter=2000)
            if info > 0:
                print(f"      [Warning] CG did not converge after {info} iterations.")
            
            u = np.zeros(self.ndof)
            u[free_dofs] = u_free
            
            # 3. Sensitivity Analysis
            u_ele = u[edofMat] 
            ce = np.sum((u_ele @ KE) * u_ele, axis=1)
            c = np.sum((Emin + x_flat**self.penal * (E0 - Emin)) * ce)
            self.compliance_history.append(float(c))

            dc = -self.penal * (E0 - Emin) * x_flat**(self.penal - 1) * ce
            dv = np.ones(self.nele)
            
            # 4. Filtering Sensitivities
            dc[:] = H @ (dc / Hs)
            dv[:] = H @ (dv / Hs)
            
            # 5. Optimality Criteria Update
            xnew_flat = self._optimality_criteria(x_flat, dc, dv)
            # Clamp passive voids before filtering
            if hasattr(self, '_passive_mask'):
                xnew_flat[self._passive_mask] = 0.0
            xnew = xnew_flat.reshape((self.nely, self.nelx, self.nelz), order='F')

            # 6. Filter Design Variable
            xPhys_flat = H @ xnew_flat / Hs
            self.xPhys = xPhys_flat.reshape((self.nely, self.nelx, self.nelz), order='F')
            # Clamp again after filter (filter can bleed density into passive voxels)
            if hasattr(self, '_passive_mask'):
                passive_3d = self._passive_mask.reshape(
                    (self.nely, self.nelx, self.nelz), order='F')
                self.xPhys[passive_3d] = 0.0
            
            change = np.max(np.abs(xnew - self.x))
            self.x = xnew
            
            print(f" It.: {loop:4d} Obj.: {c:10.4f} Vol.: {np.mean(self.xPhys):6.3f} ch.: {change:6.3f}")
            
        print(f"Optimization Converged in {time.time() - start_time:.2f}s")
        return self.xPhys, self.compliance_history

    def _prepare_edof_mat_vectorized(self):
        """
        Vectorized generation of edofMat (Top3D style)
        """
        nx, ny, nz = self.nelx, self.nely, self.nelz
        
        # Node IDs grid
        # MATLAB ordering: columns (Y) -> rows (X) -> depth (Z)
        n_node = (nx+1)*(ny+1)*(nz+1)
        node_grid = np.arange(n_node).reshape((ny+1, nx+1, nz+1), order='F')
        
        # Element base nodes (top-left-front corner of each voxel)
        # We need (ny, nx, nz) base nodes
        node_base = node_grid[:-1, :-1, :-1].flatten(order='F')
        
        # Offsets to the 8 nodes of a voxel
        # strides in F-order: 1 (y), ny+1 (x), (ny+1)*(nx+1) (z)
        sy = 1
        sx = ny + 1
        sz = (ny + 1) * (nx + 1)
        
        offsets = np.array([
            0,          # n1
            sx,         # n2
            sx + sy,    # n3
            sy,         # n4
            sz,         # n5
            sx + sz,    # n6
            sx + sy + sz, # n7
            sy + sz     # n8
        ])
        
        edofMat = np.zeros((self.nele, 24), dtype=int)
        
        for i in range(8):
            # Nodes for all elements
            nodes = node_base + offsets[i]
            # DOFs
            edofMat[:, 3*i]   = 3*nodes
            edofMat[:, 3*i+1] = 3*nodes + 1
            edofMat[:, 3*i+2] = 3*nodes + 2
            
        iK = np.kron(edofMat, np.ones((24, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 24))).flatten()
        
        return edofMat, iK, jK

    def _prepare_filter(self):
        # Standard distance filter accelerated with Numba
        nx, ny, nz = self.nelx, self.nely, self.nelz
        rmin = self.rmin
        nele = self.nele
        
        # We'll use a dense-ish pre-calculation or a focused loop
        # For Numba compatibility, we pass necessary dimensions
        iH, jH, sH = fast_filter_prep(nx, ny, nz, rmin)
        
        H = sp.coo_matrix((sH, (iH, jH)), shape=(self.nele, self.nele)).tocsc()
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

@njit(parallel=True)
def fast_filter_prep(nx, ny, nz, rmin):
    """
    Parallel preparation of filter indices and weights.
    """
    nele = nx * ny * nz
    # Heuristic for space allocation (rmin dependent)
    # Approx volume of sphere of radius rmin / vol of element
    est_entries = nele * int(4.18 * rmin**3) 
    
    iH = np.zeros(est_entries, dtype=np.int32)
    jH = np.zeros(est_entries, dtype=np.int32)
    sH = np.zeros(est_entries, dtype=np.float32)
    
    ptr = 0
    rmin_int = int(np.floor(rmin))
    
    # We can't easily sync 'ptr' in a parallel loop without a mutex or block-wise allocation.
    # Instead, we'll do a 2-pass approach or just use a serial loop for ptr-safety, 
    # but the inner check can be optimized or we can parallelize outer k1.
    
    # Accurate count pass per k1 slice
    counts_per_k = np.zeros(nz, dtype=np.int32)
    for k1 in prange(nz):
        c = 0
        for i1 in range(nx):
            for j1 in range(ny):
                for k2 in range(max(k1 - rmin_int, 0), min(k1 + rmin_int + 1, nz)):
                    for i2 in range(max(i1 - rmin_int, 0), min(i1 + rmin_int + 1, nx)):
                        for j2 in range(max(j1 - rmin_int, 0), min(j1 + rmin_int + 1, ny)):
                            dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2 + (k1 - k2)**2)
                            if dist < rmin:
                                c += 1
        counts_per_k[k1] = c
        
    # Offsets for writing
    offsets = np.zeros(nz + 1, dtype=np.int32)
    for k in range(nz):
        offsets[k+1] = offsets[k] + counts_per_k[k]
        
    total_entries = offsets[nz]
    iH_final = np.zeros(total_entries, dtype=np.int32)
    jH_final = np.zeros(total_entries, dtype=np.int32)
    sH_final = np.zeros(total_entries, dtype=np.float32)
    
    # Writing pass (Parallel)
    for k1 in prange(nz):
        write_ptr = offsets[k1]
        for i1 in range(nx):
            for j1 in range(ny):
                e1 = k1 * (nx * ny) + i1 * ny + j1
                for k2 in range(max(k1 - rmin_int, 0), min(k1 + rmin_int + 1, nz)):
                    for i2 in range(max(i1 - rmin_int, 0), min(i1 + rmin_int + 1, nx)):
                        for j2 in range(max(j1 - rmin_int, 0), min(j1 + rmin_int + 1, ny)):
                            dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2 + (k1 - k2)**2)
                            if dist < rmin:
                                e2 = k2 * (nx * ny) + i2 * ny + j2
                                iH_final[write_ptr] = e1
                                jH_final[write_ptr] = e2
                                sH_final[write_ptr] = rmin - dist
                                write_ptr += 1
                                
    return iH_final, jH_final, sH_final
