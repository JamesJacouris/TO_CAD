import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree
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

        # Passive void elements (always zero density)
        self.passive_void = None

    def set_passive_void(self, mask):
        """
        Set a boolean mask (nely, nelx, nelz) of elements that are forced to
        zero density (void) throughout optimization.
        """
        self.passive_void = mask.astype(bool)
        # Pre-zero the initial densities so the first FE solve is consistent
        self.x[self.passive_void] = 0.0
        self.xPhys[self.passive_void] = 0.0

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

        # Prepare Filter (KD-Tree vectorized)
        print("Preparing Filter...")
        t0 = time.time()
        H, Hs = self._prepare_filter_kdtree()
        print(f"Filter prepared in {time.time()-t0:.2f}s")

        # Precompute EdofMat (vectorized)
        print("Preparing EdofMat...")
        edofMat, iK, jK = self._prepare_edof_mat_vectorized()

        # Main Loop
        loop = 0
        change = 1.0

        print(f"Starting Optimization (Mesh: {self.nelx}x{self.nely}x{self.nelz})")

        start_time = time.time()

        while change > tolx and loop < max_loop:
            loop += 1

            # xPhys flattened in F-order matches edofMat element ordering
            x_flat = self.xPhys.flatten(order='F')

            Emin = 1e-9
            E0 = 1.0

            filt_stiff = Emin + x_flat**self.penal * (E0 - Emin)
            sK = (KE.flatten()[:, np.newaxis] @ filt_stiff[np.newaxis, :]).flatten(order='F')

            K = sp.coo_matrix((sK, (iK, jK)), shape=(self.ndof, self.ndof)).tocsc()
            K = (K + K.T) / 2.0

            # Solve with CG + Jacobi preconditioner
            free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
            K_free = K[free_dofs, :][:, free_dofs]
            f_free = self.f[free_dofs].flatten()
            M = sp.diags(1.0 / K_free.diagonal())
            u_free, info = cg(K_free, f_free, M=M, rtol=1e-5, maxiter=5000)
            if info > 0:
                print(f"  Warning: CG did not converge (info={info})")

            u = np.zeros(self.ndof)
            u[free_dofs] = u_free

            # Sensitivity Analysis
            u_ele = u[edofMat]
            ce = np.sum((u_ele @ KE) * u_ele, axis=1)
            c = np.sum((Emin + x_flat**self.penal * (E0 - Emin)) * ce)

            dc = -self.penal * (E0 - Emin) * x_flat**(self.penal - 1) * ce
            dv = np.ones(self.nele)

            # Filter Sensitivities
            dc[:] = H @ (dc / Hs)
            dv[:] = H @ (dv / Hs)
            
            # Optimality Criteria
            xnew_flat = self._optimality_criteria(x_flat, dc, dv)
            xnew = xnew_flat.reshape((self.nely, self.nelx, self.nelz), order='F')

            # Enforce passive void (always zero density)
            if self.passive_void is not None:
                xnew[self.passive_void] = 0.0
                xnew_flat = xnew.flatten(order='F')

            # Filter Design Variable
            # xPhys(:) = (H*xnew(:))./Hs
            xPhys_flat = H @ xnew_flat / Hs
            self.xPhys = xPhys_flat.reshape((self.nely, self.nelx, self.nelz), order='F')

            # Re-enforce passive void on physical density (filter can smear)
            if self.passive_void is not None:
                self.xPhys[self.passive_void] = 1e-9
            
            change = np.max(np.abs(xnew - self.x))
            self.x = xnew
            
            print(f" It.: {loop:4d} Obj.: {c:10.4f} Vol.: {np.mean(self.xPhys):6.3f} ch.: {change:6.3f}")
            
        print(f"Optimization Converged in {time.time() - start_time:.2f}s")
        return self.xPhys

    def _prepare_edof_mat_vectorized(self):
        """Vectorized DOF matrix construction (8 passes, one per voxel corner)."""
        nx, ny, nz = self.nelx, self.nely, self.nelz
        n_node = (nx + 1) * (ny + 1) * (nz + 1)
        # F-order reshape: index order (ny+1, nx+1, nz+1)
        node_grid = np.arange(n_node).reshape((ny + 1, nx + 1, nz + 1), order='F')
        # Base node (corner 0) for every element
        node_base = node_grid[:-1, :-1, :-1].flatten(order='F')

        # Stride offsets in F-order node numbering
        sy = 1            # step in Y (fastest index)
        sx = ny + 1       # step in X
        sz = (ny + 1) * (nx + 1)  # step in Z
        offsets = np.array([0, sx, sx + sy, sy, sz, sx + sz, sx + sy + sz, sy + sz])

        edofMat = np.zeros((self.nele, 24), dtype=int)
        for i in range(8):
            nodes = node_base + offsets[i]
            edofMat[:, 3 * i]     = 3 * nodes
            edofMat[:, 3 * i + 1] = 3 * nodes + 1
            edofMat[:, 3 * i + 2] = 3 * nodes + 2

        iK = np.kron(edofMat, np.ones((24, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 24))).flatten()
        return edofMat, iK, jK

    def _prepare_filter_kdtree(self):
        """KD-Tree filter preparation — replaces 6-nested-loop version."""
        nx, ny, nz = self.nelx, self.nely, self.nelz
        rmin = self.rmin
        nele = self.nele

        # Element centroids in F-order (matches xPhys.flatten(order='F'))
        y_range = np.arange(ny) + 0.5
        x_range = np.arange(nx) + 0.5
        z_range = np.arange(nz) + 0.5
        yy, xx, zz = np.meshgrid(y_range, x_range, z_range, indexing='ij')
        centroids = np.column_stack([
            xx.flatten(order='F'),
            yy.flatten(order='F'),
            zz.flatten(order='F'),
        ])

        tree = cKDTree(centroids)
        indices = tree.query_ball_point(centroids, rmin)

        iH, jH, sH = [], [], []
        for i, neighbors in enumerate(indices):
            if not neighbors:
                continue
            dists = np.linalg.norm(centroids[neighbors] - centroids[i], axis=1)
            weights = np.maximum(0, rmin - dists)
            iH.extend([i] * len(neighbors))
            jH.extend(neighbors)
            sH.extend(weights)

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
