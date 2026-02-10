import numpy as np

class OptimizationProblem:
    """
    Base class for defining an optimization setup (Loads & BCs).
    Subclasses must implement apply().
    """
    def __init__(self, name):
        self.name = name
        
    def apply(self, nodes):
        """
        Given the array of reconstructed nodes (N,3),
        Return:
          loads: {node_idx: [Fx, Fy, Fz, ...]}
          bcs:   {node_idx: [fixed_dofs]}
        """
        raise NotImplementedError

# ----------------------------------------------------------------------------
# Geometric Helpers
# ----------------------------------------------------------------------------

def get_nodes_in_box(nodes, min_corner, max_corner):
    """
    Returns indices of nodes inside the axis-aligned box.
    min_corner: [x,y,z]
    max_corner: [x,y,z]
    """
    # Vectorized Check
    mask = np.all((nodes >= min_corner) & (nodes <= max_corner), axis=1)
    return np.flatnonzero(mask)

def get_nodes_near_point(nodes, center, radius):
    """
    Returns indices of nodes within distance R of center.
    """
    dists = np.linalg.norm(nodes - np.array(center), axis=1)
    return np.flatnonzero(dists <= radius)

def get_nodes_on_plane(nodes, axis, value, tolerance=0.1):
    """
    Returns indices of nodes on a plane (e.g., Z = 0).
    axis: 0(X), 1(Y), 2(Z)
    """
    return np.flatnonzero(np.abs(nodes[:, axis] - value) <= tolerance)

def distribute_load(node_indices, total_force):
    """
    Distributes a total force vector equally among the listed nodes.
    total_force: [Fx, Fy, Fz]
    Returns: dict {node_id: [Fx, Fy, Fz, 0, 0, 0]}
    """
    if len(node_indices) == 0:
        return {}
        
    f_per_node = np.array(total_force) / len(node_indices)
    load_vec = list(f_per_node) + [0, 0, 0] # Add moments (0)
    
    loads = {}
    for idx in node_indices:
        loads[idx] = load_vec
    return loads
