import numpy as np

class CantileverSetup:
    """
    Standard Cantilever Beam Configuration.
    - Fixed at x=0 (Wall).
    - Load at x=max, y=min (Bottom-Right tip).
    """
    
    def __init__(self):
        self.name = "Cantilever (Generic)"

    def apply(self, nodes):

        """
        Generic Cantilever Setup.
        1. Detects Long Axis (X, Y, or Z).
        2. Fixes 'Base' Face (at Min Coordinate).
        3. Loads 'Tip' Point (at Max Coordinate) with -1000N transverse force.
        """
        bcs = {}
        loads = {}
        
        # 1. Detect Dimensions
        mins = np.min(nodes, axis=0)
        maxs = np.max(nodes, axis=0)
        dims = maxs - mins
        
        long_axis = np.argmax(dims) # 0=X, 1=Y, 2=Z
        axes = ['X', 'Y', 'Z']
        print(f"[Setup] Detected Long Axis: {axes[long_axis]} (Len={dims[long_axis]:.1f})")
        
        # 2. Fix Base (Min End of Long Axis - X=0 usually)
        # Fix all nodes within tolerance of MIN coord on long axis
        tolerance = 2.0
        min_val = mins[long_axis]
        
        base_indices = [i for i, p in enumerate(nodes) if p[long_axis] <= min_val + tolerance]
        
        for idx in base_indices:
            bcs[idx] = [0, 1, 2, 3, 4, 5] # Fix All
            
        print(f"[Setup] Fixed {len(base_indices)} nodes at {axes[long_axis]}={min_val:.1f} (Base)")

        # 3. Load Tip (Max End of Long Axis)
        # Load the single most centered node at the MAX coordinate.
        max_val = maxs[long_axis]
        tip_candidates = [i for i, p in enumerate(nodes) if p[long_axis] >= max_val - tolerance]
        
        if not tip_candidates:
             tip_candidates = [np.argmax(nodes[:, long_axis])]
             
        # 4. Determine Load Direction
        # Transverse Load. Usually -Y. 
        force_vec = [0.0, 0.0, 0.0]
        match_load = 1000.0
        
        if long_axis == 0: # X Long
            force_vec = [0.0, -match_load, 0.0] # Load -Y
        elif long_axis == 1: # Y Long
            force_vec = [match_load, 0.0, 0.0]  # Load +X
        else: # Z Long
            force_vec = [0.0, -match_load, 0.0] # Load -Y
            
        # Distribute Load among ALL tip candidates
        # This handles cases where the tip skeletonizes into multiple points (e.g. a fork).
        # By applying F/N to each of N nodes, the resultant is F at the centroid.
        
        num_tips = len(tip_candidates)
        f_per_node = [f / num_tips for f in force_vec]
        load_6d = f_per_node + [0,0,0]
        
        for idx in tip_candidates:
            if idx not in loads: loads[idx] = [0.0]*6
            # Add to existing (though usually initialized empty)
            loads[idx] = [sum(x) for x in zip(loads[idx], load_6d)]
            
        print(f"[Setup] Distributed Total Load {force_vec} across {num_tips} tip nodes at {axes[long_axis]}={max_val:.1f}")
        
        return loads, bcs
