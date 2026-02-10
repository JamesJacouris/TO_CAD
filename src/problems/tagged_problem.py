import numpy as np
import json
import os


class TaggedProblem:
    """
    Problem setup that reads BCs directly from node_tags in the JSON.
    
    Tags:
        1 = Fixed node (all 6 DOFs constrained)
        2 = Loaded node (force applied from load_vector)
    
    This replaces geometric heuristics (CantileverSetup) with exact
    tag-based BC assignment, following Yin's approach.
    """
    
    def __init__(self, load_vector=None, load_magnitude=1000.0):
        self.name = "Tagged (Yin BC Propagation)"
        self.load_vector = load_vector  # [fx, fy, fz] total force
        self.load_magnitude = load_magnitude
        self._node_tags = {}  # Populated from JSON
    
    def load_tags_from_json(self, json_path):
        """Load node_tags from a pipeline JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        tags = data.get('graph', {}).get('node_tags', {})
        # JSON keys are strings, convert to int
        self._node_tags = {int(k): v for k, v in tags.items()}
        
        n_fixed = sum(1 for v in self._node_tags.values() if v == 1)
        n_loaded = sum(1 for v in self._node_tags.values() if v == 2)
        print(f"[TaggedProblem] Loaded {n_fixed} fixed, {n_loaded} loaded nodes from JSON")
        
        return self._node_tags
    
    def set_tags(self, node_tags):
        """Set node_tags directly (dict {node_id: tag_value})."""
        self._node_tags = node_tags
    
    def apply(self, nodes):
        """
        Apply BCs based on node_tags.
        Returns: (loads, bcs) dicts
        """
        bcs = {}
        loads = {}
        
        if not self._node_tags:
            print("[TaggedProblem] Warning: No node_tags loaded! Falling back to empty BCs.")
            return loads, bcs
        
        # Fixed nodes (tag=1)
        for node_id, tag in self._node_tags.items():
            if tag == 1 and node_id < len(nodes):
                bcs[node_id] = [0, 1, 2, 3, 4, 5]  # Fix all 6 DOFs
        
        # Loaded nodes (tag=2)
        loaded_ids = [nid for nid, t in self._node_tags.items() if t == 2 and nid < len(nodes)]
        
        if loaded_ids:
            # Determine load vector
            if self.load_vector is not None:
                total_force = self.load_vector
            else:
                # Auto-detect: transverse load on long axis (like CantileverSetup)
                mins = np.min(nodes, axis=0)
                maxs = np.max(nodes, axis=0)
                dims = maxs - mins
                long_axis = np.argmax(dims)
                
                total_force = [0.0, 0.0, 0.0]
                if long_axis == 0:
                    total_force[1] = -self.load_magnitude  # Load -Y
                elif long_axis == 1:
                    total_force[0] = self.load_magnitude   # Load +X
                else:
                    total_force[1] = -self.load_magnitude  # Load -Y
                    
                print(f"[TaggedProblem] Auto-detected load direction: {total_force}")
            
            # Distribute force among all loaded nodes
            n_loaded = len(loaded_ids)
            f_per_node = [f / n_loaded for f in total_force]
            load_6d = f_per_node + [0.0, 0.0, 0.0]  # [Fx, Fy, Fz, Mx, My, Mz]
            
            for nid in loaded_ids:
                loads[nid] = load_6d
        
        print(f"[TaggedProblem] Applied: {len(bcs)} fixed nodes, {len(loads)} loaded nodes")
        if loaded_ids:
            print(f"  Total force: {total_force}, distributed across {len(loaded_ids)} nodes")
        
        return loads, bcs
