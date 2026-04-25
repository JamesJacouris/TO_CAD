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
        self._load_position = None  # Centroid of original load-tagged voxels
    
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

    def set_load_position_from_npz(self, npz_path, pitch=1.0, origin=None):
        """Compute centroid of tag==2 voxels from NPZ bc_tags for fallback."""
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'bc_tags' not in data:
                return
            bc = data['bc_tags']
            p = float(data['pitch']) if 'pitch' in data else pitch
            o = data['origin'].astype(float) if 'origin' in data else (origin or np.zeros(3))
            load_ijk = np.argwhere(bc == 2)  # [nely_idx, nelx_idx, nelz_idx]
            if len(load_ijk) == 0:
                return
            centroid_ijk = load_ijk.mean(axis=0)  # [nely, nelx, nelz]
            # Reorder to world coords [nelx, nely, nelz] to match graph node positions
            centroid_world = centroid_ijk[[1, 0, 2]]
            self._load_position = centroid_world * p + np.asarray(o) + p * 0.5
        except Exception:
            pass
    
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
        
        # Determine load vector
        if self.load_vector is not None:
            total_force = self.load_vector
        else:
            mins = np.min(nodes, axis=0)
            maxs = np.max(nodes, axis=0)
            dims = maxs - mins
            short_axis = np.argmin(dims)
            total_force = [0.0, 0.0, 0.0]
            total_force[short_axis] = -self.load_magnitude
            print(f"[TaggedProblem] Auto-detected load direction: {total_force} (short_axis={short_axis})")

        # Fallback: if no tag==2 nodes, apply load to nearest node to original position
        if not loaded_ids and any(f != 0 for f in total_force):
            fixed_ids = set(bcs.keys())
            free_ids = [i for i in range(len(nodes)) if i not in fixed_ids]
            if free_ids:
                if self._load_position is not None:
                    dists = np.linalg.norm(nodes[free_ids] - self._load_position, axis=1)
                    best = free_ids[np.argmin(dists)]
                else:
                    # Farthest free node from centroid of fixed nodes
                    if fixed_ids:
                        fix_centroid = np.mean(nodes[list(fixed_ids)], axis=0)
                        dists = np.linalg.norm(nodes[free_ids] - fix_centroid, axis=1)
                        best = free_ids[np.argmax(dists)]
                    else:
                        best = free_ids[-1]
                loaded_ids = [best]
                print(f"[TaggedProblem] No tag-2 nodes found; fallback load on node {best} "
                      f"at {nodes[best].tolist()}")

        if loaded_ids:
            n_loaded = len(loaded_ids)
            f_per_node = [f / n_loaded for f in total_force]
            load_6d = f_per_node + [0.0, 0.0, 0.0]  # [Fx, Fy, Fz, Mx, My, Mz]
            for nid in loaded_ids:
                loads[nid] = load_6d

        print(f"[TaggedProblem] Applied: {len(bcs)} fixed nodes, {len(loads)} loaded nodes")
        if loaded_ids:
            print(f"  Total force: {total_force}, distributed across {len(loaded_ids)} nodes")

        return loads, bcs
