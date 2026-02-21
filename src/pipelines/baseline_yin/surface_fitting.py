import numpy as np
import warnings
from scipy.interpolate import bisplrep, bisplev
from scipy.spatial.transform import Rotation

def fit_bspline_surface(points, degree=3, target_control_points=10):
    """
    Fits a B-Spline surface to a cloud of 3D points.
    
    Strategy (Yin's Medial Surface Fitting):
    1. PCA to align points to the XY plane (z' is the thin direction).
    2. Project points to (x', y').
    3. Fit a smooth B-Spline z' = f(x', y').
    4. Evaluate on a grid and transform back to global coordinates.
    
    Args:
        points (np.ndarray): (N, 3) array of voxel coordinates.
        degree (int): B-Spline degree (usually 3).
        target_control_points (int): Approximate number of control points in each dim.
        
    Returns:
        dict: {
            'ctrl_grid': [[x,y,z]...],  # Grid of 3D points for FreeCAD
            'degree_u': int,
            'degree_v': int
        } or None if fitting fails.
    """
    if len(points) < 16: # Need enough points for cubic fit
        return None
        
    # 1. PCA for alignment
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    rot = vh  # (3, 3) orthogonal matrix
    
    local_points = centered @ rot.T
    
    x = local_points[:, 0]
    y = local_points[:, 1]
    z = local_points[:, 2]
    
    # 2. Fit Spline z = f(x, y)
    try:
        w = np.ones_like(z)
        
        # Adaptive Degree Selection
        # Check number of unique grid points in local x/y to prevent "m > (k+1)(k+1)" errors
        # or rank deficiency for narrow plates.
        unique_x = len(np.unique(np.round(x, 4)))
        unique_y = len(np.unique(np.round(y, 4)))
        
        kx = min(degree, unique_x - 1)
        ky = min(degree, unique_y - 1)
        
        # Ensure minimum degree 1 (linear) and max 5
        kx = max(1, min(kx, 5))
        ky = max(1, min(ky, 5))
        
        # Smoothing factor: higher = smoother
        smoothing = len(points) * 1.0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Create the spline
            tck = bisplrep(x, y, z, w=w, kx=kx, ky=ky, s=smoothing)
            tx, ty, c, kx_out, ky_out = tck

        # Adaptive grid resolution: scale with point count
        # Small plates (16 pts): 8x8, large plates (500+ pts): 20x20
        res = max(8, min(20, int(np.sqrt(len(points)))))
        
        u_grid = np.linspace(np.min(x), np.max(x), res)
        v_grid = np.linspace(np.min(y), np.max(y), res)
        
        tck = (tx, ty, c, kx, ky)
        z_grid = bisplev(u_grid, v_grid, tck)  # (res, res)
        
        # Transform back to global coordinates
        grid_points = []
        for i in range(res):
            row = []
            for j in range(res):
                p_local = np.array([u_grid[i], v_grid[j], z_grid[i, j]])
                p_global = p_local @ rot + centroid
                row.append(p_global.tolist())
            grid_points.append(row)

        return {
            'ctrl_grid': grid_points,
            'degree_u': kx,
            'degree_v': ky
        }
        
    except Exception as e:
        print(f"B-Spline fit failed: {e}")
        return None


def evaluate_bspline(model, u_grid, v_grid):
    """
    Evaluates the B-Spline model to get 3D points.
    Useful for generating mesh triangulation fallback or debug.
    """
    tx, ty, c, kx, ky = model['tck']
    
    # Evaluate z grid
    z_grid = bisplev(u_grid, v_grid, (tx, ty, c, kx, ky))
    
    # Transform back to global 3D
    # Grid shapes: (Nu), (Nv) -> Z is (Nu, Nv)?? Check bisplev docs.
    # bisplev returns 2D array of shape (len(u), len(v))
    
    nu, nv = len(u_grid), len(v_grid)
    local_xyz = np.zeros((nu, nv, 3))
    
    # Create meshgrid for x, y
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij') # (Nu, Nv)
    
    local_xyz[:, :, 0] = U
    local_xyz[:, :, 1] = V
    local_xyz[:, :, 2] = z_grid
    
    # Reshape to list of points for rotation
    flat_local = local_xyz.reshape(-1, 3)
    
    rot = np.array(model['rotation'])
    centroid = np.array(model['centroid'])
    
    # Inverse transform: p_global = (p_local @ rot) + centroid
    # Since local = centered @ rot.T => centered = local @ rot
    # p_global = centered + centroid
    
    global_points = flat_local @ rot + centroid
    
    return global_points.reshape(nu, nv, 3)
