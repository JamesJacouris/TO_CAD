"""L-BFGS-B node-position layout optimisation for beam frames.

Moves free node positions along compliance gradients subject to box constraints
(design domain) and a move limit.  Fixed (BC tag 1) and loaded (BC tag 2)
nodes are frozen throughout.

Main entry point
----------------
:func:`optimize_layout`
"""
import numpy as np
import argparse
import json
import sys
import os
from scipy.optimize import minimize

# Adjust path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.optimization.fem import solve_frame, solve_curved_frame
from src.problems import load_problem_config
from src.pipelines.baseline_yin.visualization import viz_graph_radii, show_step, viz_loads
from src.curves.spline import sanitize_bezier_ctrl_pts, fit_cubic_bezier
# Removed collapse_short_edges import as we implementing local logic
# from src.pipelines.baseline_yin.postprocessing import collapse_short_edges, recheck_graph, graph_to_arrays

def generate_report(c_init, c_final, nodes_init, nodes_final, edges, radii, iterations, message, filename="optimization_report.txt", target_volume_abs=None):
    """
    Prints and saves a summary of the Layout Optimization.
    """
    # 1. Compliance Stats
    reduction = c_init - c_final
    pct = (reduction / c_init) * 100.0 if c_init > 0 else 0.0
    
    # 2. Movement Stats
    displacements = np.linalg.norm(nodes_final - nodes_init, axis=1)
    max_d = np.max(displacements)
    idx_max = np.argmax(displacements)
    mean_d = np.mean(displacements)
    
    # 3. Volume Stats (Check if conserved)
    def calc_volume(ns):
        vol = 0.0
        for i, (u, v) in enumerate(edges):
            length = np.linalg.norm(ns[u] - ns[v])
            area = np.pi * (radii[i]**2)
            vol += area * length
        return vol
        
    v_init = calc_volume(nodes_init)
    v_final = calc_volume(nodes_final)
    v_change = v_final - v_init
    v_pct = (v_change / v_init) * 100.0 if v_init > 0 else 0.0
    
    v_target_row = ""
    if target_volume_abs:
        v_diff = v_final - target_volume_abs
        v_diff_pct = (v_diff / target_volume_abs) * 100.0
        v_target_row = f"\n  Target (Solid):   {target_volume_abs:.4f} mm^3 (Diff: {v_diff_pct:+.2f}%)"
    
    # 4. Format Report
    lines = [
        "="*50,
        "          LAYOUT OPTIMIZATION REPORT",
        "="*50,
        f"Status:             {message}",
        f"Iterations:         {iterations}",
        "-"*50,
        "COMPLIANCE (Stiffness Inverse):",
        f"  Initial:          {c_init:.4f}",
        f"  Final:            {c_final:.4f}",
        f"  Reduction:        {reduction:.4f} ({pct:.2f}%)",
        "-"*50,
        "VOLUME CHECK:",
        f"  Initial Volume:   {v_init:.4f} mm^3",
        f"  Final Volume:     {v_final:.4f} mm^3",
        f"  Change:           {v_change:.4f} mm^3 ({v_pct:.2f}%)" + v_target_row,
        "-"*50,
        "NODE MOVEMENT (Geometry Change):",
        f"  Max Displacement: {max_d:.4f} mm (Node {idx_max})",
        f"  Mean Displacement:{mean_d:.4f} mm",
        "="*50
    ]
    
    report_text = "\n".join(lines)
    
    # Print to Console
    print("\n" + report_text + "\n")
    
    # Save to File
    try:
        with open(filename, "w") as f:
            f.write(report_text)
        print(f"[Report] Saved detailed report to {filename}")
    except Exception as e:
        print(f"[Report] Failed to save file: {e}")



def obj_compliance(x_flat, nodes_shape, edges, radii, problem, E):
    """
    Objective function for Layout Optimization.
    x_flat: Flattened node coordinates.
    """
    # 1. Reconstruct Nodes
    nodes = x_flat.reshape(nodes_shape)
    
    # 2. Setup Loads/BCs (Re-evaluate on new geometry? 
    # Usually BCs are geometric. If nodes move, BCs might change?
    # For now, we assume standard geometric mapping holds, or we fix the INDICES.
    # If we move a fixed node, it's bad. 
    # We will constrain fixed nodes, so their position won't change much.
    # But strictly, the Problem class finds nodes by boxes.
    # Let's assume Problem definition relies on initial geometry or we pass 'bcs' directly?
    # Better: Recalculate Loads/BCs to be safe, BUT optimization solvers dislike discontinuous changes.
    # If a node moves out of a "Fixed Box", it suddenly becomes free -> Discontinuity.
    # STRATEGY: Calculate BCs ONCE on start, and lock those NODE INDICES. 
    # Loads: If geometric (e.g. surface load), might need update. 
    # But for rocker arm, loads are on specific tips.
    # Let's simple-pass: We calculate BCs/Loads once outside and pass them in?
    # No, 'solve_frame' needs them.
    # Let's cache them.
    pass

    # Actually, we'll use a cached wrapper.
    return 0.0

class LayoutOptimizer:
    def __init__(self, nodes_init, edges, radii, problem, E=1000.0,
                 ctrl_pts=None, geo_reg=0.0, plates=None, plate_thicknesses=None,
                 sym_data=None, vol_weight=0.0, target_volume=None):
        self.nodes_init = nodes_init.copy()
        self.nodes_ref = nodes_init.copy()  # Reference geometry for regularization
        self.nodes_shape = nodes_init.shape
        self.edges = edges
        self.radii = radii
        self.problem = problem
        self.E = E
        self.ctrl_pts = ctrl_pts
        self.use_curved = ctrl_pts is not None
        self.n_nodes = len(nodes_init)
        self.n_edges = len(edges)
        self.geo_reg = geo_reg
        self.plates = plates
        self.plate_thicknesses = plate_thicknesses

        # Volume penalty: penalize deviation from target volume
        self._vol_weight = vol_weight if target_volume is not None else 0.0
        self._target_vol = target_volume if target_volume is not None else 0.0
        self._c_ref = None  # set on first objective() call for penalty scaling
        self._last_compliance = 0.0  # pure FEM compliance (no penalties)
        # Fixed plate volume contribution (plates don't change during layout opt)
        self._plate_vol_fixed = 0.0
        if plates and plate_thicknesses:
            for p, h in zip(plates, plate_thicknesses):
                area = p.get("area", 0.0)
                self._plate_vol_fixed += area * h

        # Pre-calc BCs/Loads to lock topology
        self.loads_init, self.bcs_init = problem.apply(self.nodes_init)

        # Identify Fixed Nodes & Loaded Nodes (for constraints)
        self.fixed_node_indices = set(self.bcs_init.keys())
        self.loaded_node_indices = set(self.loads_init.keys())
        self.locked_node_indices = self.fixed_node_indices.union(self.loaded_node_indices)

        # Build free-node mask for regularization (exclude locked nodes)
        self._free_mask = np.ones(self.n_nodes, dtype=bool)
        for idx in self.locked_node_indices:
            if idx < self.n_nodes:
                self._free_mask[idx] = False
        self._n_free = int(self._free_mask.sum())

        # ── Symmetry penalty setup ──────────────────────────────────
        self._sym_pairs = []
        self._sym_weight = 0.0
        self._sym_node_info = None
        if sym_data is not None:
            from src.optimization.symmetry import find_symmetric_node_pairs
            self._sym_node_info = find_symmetric_node_pairs(
                nodes_init, sym_data['planes'], tol=sym_data['tol'],
                locked_nodes=self.locked_node_indices)
            self._sym_weight = sym_data.get('weight', 0.1)

            for plane in sym_data['planes']:
                pname = plane['name']
                info = self._sym_node_info[pname]
                # Include pairs where at least one node is free
                pairs = [(i, j) for i, j in info['node_pairs']
                         if i not in self.locked_node_indices
                         or j not in self.locked_node_indices]
                if pairs:
                    self._sym_pairs.append({
                        'axis': plane['axis'],
                        'center': plane['center'],
                        'pairs': np.array(pairs, dtype=int),
                    })

            n_sym = sum(len(sp['pairs']) for sp in self._sym_pairs)
            print(f"[Layout] Symmetry: {n_sym} node pair(s), "
                  f"weight={self._sym_weight:.3f}")

    def _solve_compliance(self, nodes, cp_list=None):
        """Return raw FEM compliance (no penalties)."""
        if self.use_curved and cp_list is not None:
            u, compliance, _ = solve_curved_frame(
                nodes, self.edges, self.radii, cp_list,
                E=self.E, loads=self.loads_init, bcs=self.bcs_init)
        elif self.plates is not None and len(self.plates) > 0:
            result = solve_frame(
                nodes, self.edges, self.radii, self.E,
                loads=self.loads_init, bcs=self.bcs_init,
                plates=self.plates,
                plate_thicknesses=self.plate_thicknesses)
            u, compliance = result[0], result[1]
        else:
            u, compliance, _ = solve_frame(
                nodes, self.edges, self.radii, self.E,
                loads=self.loads_init, bcs=self.bcs_init)
        return compliance

    def objective(self, x_flat):
        # Split design vector into nodes + (optional) ctrl_pts
        nodes = x_flat[:self.n_nodes * 3].reshape(self.nodes_shape)

        cp_list = None
        if self.use_curved:
            cp_flat = x_flat[self.n_nodes * 3:]
            cp_list = self._unpack_ctrl_pts(cp_flat, nodes)

        compliance = self._solve_compliance(nodes, cp_list)
        self._last_compliance = compliance  # track pure compliance

        total = compliance

        # Geometric regularization: penalize deviation from reference topology
        if self.geo_reg > 0 and self._n_free > 0:
            disp = nodes[self._free_mask] - self.nodes_ref[self._free_mask]
            total += self.geo_reg * float(np.sum(disp ** 2)) / self._n_free

        # Symmetry penalty: ||node[j] - mirror(node[i])||^2
        if self._sym_weight > 0 and self._sym_pairs:
            sym_penalty = 0.0
            for sp in self._sym_pairs:
                ax = sp['axis']
                ctr = sp['center']
                pairs = sp['pairs']  # (K, 2)

                pos_i = nodes[pairs[:, 0]]  # (K, 3)
                pos_j = nodes[pairs[:, 1]]  # (K, 3)

                mirror_i = pos_i.copy()
                mirror_i[:, ax] = 2 * ctr - mirror_i[:, ax]

                diff = pos_j - mirror_i
                sym_penalty += float(np.sum(diff ** 2))

            n_total_pairs = sum(len(sp['pairs']) for sp in self._sym_pairs)
            if n_total_pairs > 0:
                sym_penalty /= n_total_pairs

            total += self._sym_weight * sym_penalty

        # Volume penalty: scaled relative to initial compliance so both terms
        # are the same order of magnitude.  penalty = vol_weight * C_ref * vol_err^2
        if self._vol_weight > 0 and self._target_vol > 0:
            lengths = np.linalg.norm(
                nodes[self.edges[:, 0]] - nodes[self.edges[:, 1]], axis=1)
            beam_vol = float(np.sum(np.pi * self.radii**2 * lengths))
            current_vol = beam_vol + self._plate_vol_fixed
            vol_err = (current_vol - self._target_vol) / self._target_vol
            # Scale by reference compliance (set on first call) so penalty
            # magnitude is comparable to compliance regardless of problem size
            if not hasattr(self, '_c_ref') or self._c_ref is None:
                self._c_ref = max(abs(compliance), 1e-12)
            total += self._vol_weight * self._c_ref * vol_err**2

        return total

    def _unpack_ctrl_pts(self, cp_flat, nodes):
        """Reconstruct ctrl_pts list from flat vector, applying sanitization."""
        cp_list = []
        offset = 0
        for idx in range(self.n_edges):
            if self.ctrl_pts[idx] is not None:
                p1 = cp_flat[offset:offset + 3].copy()
                p2 = cp_flat[offset + 3:offset + 6].copy()
                offset += 6
                u_idx, v_idx = int(self.edges[idx, 0]), int(self.edges[idx, 1])
                p1s, p2s = sanitize_bezier_ctrl_pts(
                    nodes[u_idx], nodes[v_idx], p1, p2)
                cp_list.append(np.array([p1s, p2s]))
            else:
                cp_list.append(None)
        return cp_list

def optimize_layout(nodes, edges, radii, problem, E=1000.0, move_limit=5.0,
                    visualize=False, target_volume_abs=None, snap_dist=2.0,
                    design_bounds=None, node_tags=None, ctrl_pts=None,
                    ctrl_move_limit=None, geo_reg=0.0,
                    plates=None, plate_thicknesses=None,
                    sym_data=None, vol_weight=10.0):
    """Optimise node positions to minimise frame compliance (L-BFGS-B).

    Free node positions are updated by ``scipy.optimize.minimize`` with
    the ``L-BFGS-B`` method.  Each function evaluation performs a full
    FEA solve and assembles analytical compliance gradients.

    BC nodes (``tag=1``) and loaded nodes (``tag=2``) are excluded from
    the optimisation variables.  Box bounds are set to
    ``initial_position ± move_limit`` clipped to ``design_bounds``.

    After convergence, nodes within ``snap_dist`` of each other are merged
    to maintain a clean graph topology.

    Args:
        nodes (numpy.ndarray): Initial node positions, shape ``(N, 3)``, mm.
        edges (numpy.ndarray): Element connectivity, shape ``(M, 2)``.
        radii (numpy.ndarray): Per-element radii, shape ``(M,)``, mm.
        problem: Problem config with ``apply(nodes) → (loads, bcs)``.
        E (float): Young's modulus.
        move_limit (float): Maximum allowed node displacement per step, mm.
        visualize (bool): Show Open3D radius visualisation after optimisation.
        target_volume_abs (float or None): Volume constraint for reporting.
        snap_dist (float): Distance threshold for post-opt node merging, mm.
        design_bounds (list or None): ``[[x_min, y_min, z_min],
            [x_max, y_max, z_max]]`` — hard box constraints on node positions.
        node_tags (dict or None): ``{node_idx: tag}`` — tag 1=fixed,
            tag 2=loaded; tagged nodes are excluded from optimisation.
        ctrl_move_limit (float or None): Maximum allowed displacement of
            interior Bézier control points per step, mm.  Defaults to
            ``0.3 * move_limit`` when ``ctrl_pts`` is not None.  Use a
            smaller value (e.g. 1.0) to prevent extreme curvature.
        geo_reg (float): Geometric regularization weight.  The objective
            becomes ``C_compliance + geo_reg * mean(||node - node_ref||²)``
            where ``node_ref`` is the initial skeleton position.  Penalizes
            drift from the original TO topology.  0.0 = off (default).
            Typical range: 0.01–1.0 relative to compliance scale.

    Returns:
        tuple:
            - **nodes_opt** (``numpy.ndarray``, shape ``(N', 3)``): Optimised
              node positions (N' ≤ N after snapping).
            - **edges_opt** (``numpy.ndarray``, shape ``(M', 2)``): Updated
              edge connectivity after snapping.
            - **radii_opt** (``numpy.ndarray``, shape ``(M',)``): Radii
              corresponding to ``edges_opt``.
            - **tags_opt** (``dict``): Updated node tags after snapping.
            - **c_initial** (``float``): Compliance before optimisation.
            - **c_final** (``float``): Compliance after optimisation.
    """
    optimizer = LayoutOptimizer(nodes, edges, radii, problem, E, ctrl_pts=ctrl_pts,
                                geo_reg=geo_reg, plates=plates,
                                plate_thicknesses=plate_thicknesses,
                                sym_data=sym_data,
                                vol_weight=vol_weight,
                                target_volume=target_volume_abs)
    use_curved = ctrl_pts is not None
    if ctrl_move_limit is None:
        ctrl_move_limit = move_limit * 0.3

    # Build design vector: [node_positions, ctrl_pt_positions (if curved)]
    x0_parts = [nodes.flatten()]
    if use_curved:
        cp_parts = []
        for idx in range(len(edges)):
            if ctrl_pts[idx] is not None:
                cp_parts.append(ctrl_pts[idx].flatten())  # (6,): P1xyz + P2xyz
        if cp_parts:
            x0_parts.append(np.concatenate(cp_parts))
    x0 = np.concatenate(x0_parts)
    
    # Pre-compute max radius per node (for geometry envelope constraint)
    node_max_radii = np.zeros(len(nodes))
    for i, (u, v) in enumerate(edges):
        node_max_radii[u] = max(node_max_radii[u], radii[i])
        node_max_radii[v] = max(node_max_radii[v], radii[i])
    
    # Bounds
    # Fixed nodes: Exact position (or very tight bounds)
    # Free nodes: +/- move_limit, AND within design_bounds (accounting for radius)
    bounds = []
    
    # Pre-compute locked set
    locked_nodes = optimizer.locked_node_indices
    
    # Convert design bounds to numpy for easier use
    if design_bounds is not None:
        domain_min = np.array(design_bounds[0])
        domain_max = np.array(design_bounds[1])
        print(f"[Layout] Design Domain: [{domain_min[0]:.1f}, {domain_max[0]:.1f}] x [{domain_min[1]:.1f}, {domain_max[1]:.1f}] x [{domain_min[2]:.1f}, {domain_max[2]:.1f}]")
    else:
        domain_min = None
        domain_max = None
    
    # On-plane symmetry constraints: lock across-plane coordinate to center
    on_plane_overrides = {}  # {node_idx: {axis: value}}
    if sym_data is not None and optimizer._sym_node_info is not None:
        from src.optimization.symmetry import get_on_plane_bound_overrides
        on_plane_overrides = get_on_plane_bound_overrides(
            nodes, optimizer._sym_node_info, sym_data['planes'])
        if on_plane_overrides:
            print(f"[Layout] Symmetry: {len(on_plane_overrides)} on-plane node(s) constrained")

    for i in range(len(nodes)):
        if i in locked_nodes:
            # Fixed or Loaded Node - constrain to initial position
            c = nodes[i]
            bounds.append((c[0], c[0]))
            bounds.append((c[1], c[1]))
            bounds.append((c[2], c[2]))
        else:
            # Free Node
            c = nodes[i]
            r = node_max_radii[i]  # Max radius of connected edges

            # Start with move_limit bounds
            b_min = [c[0] - move_limit, c[1] - move_limit, c[2] - move_limit]
            b_max = [c[0] + move_limit, c[1] + move_limit, c[2] + move_limit]

            # Apply design domain bounds (accounting for beam radius)
            if domain_min is not None:
                for axis in range(3):
                    # Node center must be at least r away from domain boundary
                    # so the cylinder doesn't exceed the design envelope
                    domain_width = domain_max[axis] - domain_min[axis]

                    if domain_width < 2 * r:
                        # Domain too thin for this beam - clamp to center
                        center = (domain_min[axis] + domain_max[axis]) / 2.0
                        b_min[axis] = max(b_min[axis], center)
                        b_max[axis] = min(b_max[axis], center)
                    else:
                        b_min[axis] = max(b_min[axis], domain_min[axis] + r)
                        b_max[axis] = min(b_max[axis], domain_max[axis] - r)

                    # Final safety: ensure min <= max
                    if b_min[axis] > b_max[axis]:
                        mid = (b_min[axis] + b_max[axis]) / 2.0
                        b_min[axis] = mid
                        b_max[axis] = mid

            # Override bounds for on-plane symmetry nodes
            if i in on_plane_overrides:
                for axis, val in on_plane_overrides[i].items():
                    b_min[axis] = val
                    b_max[axis] = val

            bounds.append((b_min[0], b_max[0]))
            bounds.append((b_min[1], b_max[1]))
            bounds.append((b_min[2], b_max[2]))
            
    # Append bounds for ctrl_pt variables (if curved)
    # ctrl_move_limit is intentionally tighter than node move_limit to prevent
    # extreme arch curvature — defaults to 30% of node move limit.
    if use_curved:
        for idx in range(len(edges)):
            if ctrl_pts[idx] is not None:
                cp = ctrl_pts[idx]  # (2, 3)
                for cp_idx in range(2):
                    for axis in range(3):
                        val = cp[cp_idx, axis]
                        bounds.append((val - ctrl_move_limit, val + ctrl_move_limit))

    print(f"[Layout] Starting L-BFGS-B Optimization on {len(x0)} variables "
          f"({len(x0) - len(nodes)*3} ctrl_pt DOFs)...")
    print(f"[Layout] Move Limit: +/- {move_limit} mm"
          + (f", Ctrl Pt Limit: +/- {ctrl_move_limit:.2f} mm" if use_curved else "")
          + (f", Geo Reg: {geo_reg:.4f}" if geo_reg > 0 else ""))
    
    if visualize:
         # Show Initial
         # Viz 1: Loads & Initial State
         load_geoms = viz_loads(nodes, optimizer.loads_init, optimizer.bcs_init, scale=10.0)
         graph_geoms = viz_graph_radii(nodes, edges, radii, ctrl_pts=ctrl_pts)
         show_step("Initial Layout", load_geoms + graph_geoms)
         
    # Callback — record per-iteration pure compliance (not total objective)
    layout_history = []
    def callback(xk):
        optimizer.objective(xk)  # updates _last_compliance
        layout_history.append(float(optimizer._last_compliance))

    # Initial Compliance (pure FEM, no penalties)
    obj_init = optimizer.objective(x0)
    c_init = float(optimizer._last_compliance)
    print(f"[Layout] Initial Compliance: {c_init:.6e}  (total objective: {obj_init:.6e})")
    layout_history.append(c_init)

    if np.isnan(c_init):
        print("[ERROR] Beam Graph is singular or disconnected (NaN Compliance). Skipping Layout Optimization.")
        return nodes, edges, radii, node_tags, c_init, c_init, layout_history

    # Optimization
    res = minimize(optimizer.objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'disp': True, 'maxiter': 50, 'eps': 1e-4}, # epsilon step for FD
                   callback=callback)
                   
    # Evaluate pure compliance at the final point
    optimizer.objective(res.x)  # updates _last_compliance
    c_final = float(optimizer._last_compliance)
    print(f"[Layout] Done. Final Compliance: {c_final:.6e}  (total objective: {res.fun:.6e})")

    nodes_new = res.x[:len(nodes) * 3].reshape(nodes.shape)

    # Exact symmetry enforcement (project onto symmetric manifold)
    if sym_data is not None and optimizer._sym_node_info is not None:
        from src.optimization.symmetry import enforce_exact_node_symmetry
        nodes_new = enforce_exact_node_symmetry(
            nodes_new, optimizer._sym_node_info, sym_data['planes'],
            locked_nodes=optimizer.locked_node_indices)
        print(f"[Layout] Symmetry: Enforced exact node symmetry")

    # Recover optimised ctrl_pts from the design vector
    ctrl_pts_opt = None
    if use_curved:
        cp_flat = res.x[len(nodes) * 3:]
        ctrl_pts_opt = optimizer._unpack_ctrl_pts(cp_flat, nodes_new)

    # --- Final Report ---
    generate_report(c_init, c_final, nodes, nodes_new, edges, radii, getattr(res, 'nit', 0), res.message, target_volume_abs=target_volume_abs)
    # --------------------

    if visualize:
        final_geoms = viz_graph_radii(nodes_new, edges, radii,
                                      ctrl_pts=ctrl_pts_opt if ctrl_pts_opt is not None else ctrl_pts)
        show_step("Optimized Layout", final_geoms)

    # Snap Nodes (Collapse Short Edges)
    print(f"[Layout] Snapping Clean-up (Limit={snap_dist}mm)...")

    nodes_clean, edges_clean, radii_clean, tags_clean, cp_clean = snap_nodes(
        nodes_new, edges, radii, snap_dist,
        locked_nodes=optimizer.locked_node_indices,
        node_tags=node_tags, ctrl_pts=ctrl_pts_opt)

    # Append final pure compliance to history
    layout_history.append(c_final)

    if len(nodes_clean) < len(nodes_new):
        print(f"[Layout] Snapped! Nodes reduced from {len(nodes_new)} to {len(nodes_clean)}")
        if use_curved:
            return nodes_clean, edges_clean, radii_clean, tags_clean, c_init, c_final, cp_clean, layout_history
        return nodes_clean, edges_clean, radii_clean, tags_clean, c_init, c_final, layout_history
    else:
        if use_curved:
            return nodes_new, edges, radii, node_tags if node_tags else {}, c_init, c_final, ctrl_pts_opt, layout_history
        return nodes_new, edges, radii, node_tags if node_tags else {}, c_init, c_final, layout_history

def snap_nodes(nodes, edges, radii, tol, locked_nodes=None, node_tags=None,
               ctrl_pts=None):
    """
    Merges nodes closer than tol connected by an edge.
    Updates edges, radii, node_tags, and ctrl_pts.
    Prioritizes keeping the position of locked_nodes (anchors).
    """
    if locked_nodes is None:
        locked_nodes = set()
    else:
        locked_nodes = set(locked_nodes)

    if node_tags is None:
        node_tags = {}
        
    # PROTECT EXTREMA: Add nodes at min/max of each axis to locked set
    # ... (rest of extrema logic stays same)
    mins = np.min(nodes, axis=0)
    maxs = np.max(nodes, axis=0)
    dims = maxs - mins
    long_axis = np.argmax(dims)
    
    tol_extrema = 3.0
    min_val_long = mins[long_axis]
    tip_indices = [i for i, p in enumerate(nodes) if p[long_axis] <= min_val_long + tol_extrema]
    for idx in tip_indices: locked_nodes.add(idx)
    
    max_val_long = maxs[long_axis]
    base_indices = [i for i, p in enumerate(nodes) if p[long_axis] >= max_val_long - tol_extrema]
    for idx in base_indices: locked_nodes.add(idx)
    
    # 1. Identify short edges to collapse (using UFD)
    N = len(nodes)
    parent = np.arange(N)
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    for i, (u, v) in enumerate(edges):
        p1, p2 = nodes[u], nodes[v]
        if np.linalg.norm(p1 - p2) < tol:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                # Merge if tags are compatible
                tag_u, tag_v = node_tags.get(u), node_tags.get(v)
                if tag_u and tag_v and tag_u != tag_v:
                    continue # Block merge of different BCs
                parent[root_v] = root_u

    # 2. Compute Centroids for merged groups
    group_nodes = {}
    for i in range(N):
        root = find(i)
        if root not in group_nodes: group_nodes[root] = []
        group_nodes[root].append(i)
        
    new_coords = {}
    new_node_tags = {}
    
    unique_roots = sorted(group_nodes.keys())
    for new_id, root in enumerate(unique_roots):
        members = group_nodes[root]
        locked_members = [m for m in members if m in locked_nodes]
        
        # Coordinate
        if len(locked_members) > 0:
            new_coords[new_id] = np.mean([nodes[m] for m in locked_members], axis=0)
        else:
            new_coords[new_id] = np.mean([nodes[m] for m in members], axis=0)
            
        # Tag (collect tags from any member)
        for m in members:
            if m in node_tags:
                new_node_tags[new_id] = node_tags[m]
                break
                
    # 3. Build Outputs
    nodes_out = np.array([new_coords[i] for i in range(len(unique_roots))])
    
    root_to_new_id = {root: i for i, root in enumerate(unique_roots)}
    edges_out = []
    radii_out = []
    cp_out = []
    for i, (u, v) in enumerate(edges):
        ru, rv = find(u), find(v)
        if ru != rv:
            edges_out.append([root_to_new_id[ru], root_to_new_id[rv]])
            radii_out.append(radii[i])
            if ctrl_pts is not None:
                cp_out.append(ctrl_pts[i])
            else:
                cp_out.append(None)

    # Re-fit ctrl_pts for edges whose endpoints moved due to merging
    if ctrl_pts is not None:
        for j in range(len(edges_out)):
            eu, ev = edges_out[j]
            if cp_out[j] is None:
                continue
            # Re-fit if the merged endpoints differ significantly
            p0, p3 = nodes_out[eu], nodes_out[ev]
            cp_out[j] = np.array([
                *sanitize_bezier_ctrl_pts(p0, p3, cp_out[j][0], cp_out[j][1])
            ]).reshape(2, 3)

    return nodes_out, np.array(edges_out), np.array(radii_out), new_node_tags, cp_out

def main():
    parser = argparse.ArgumentParser(description="Run Layout Optimization.")
    parser.add_argument("input_json", help="Path to input.json (with radii)")
    parser.add_argument("output_json", help="Path to save optimized.json")
    parser.add_argument("--problem", type=str, required=True, help="Problem Config")
    parser.add_argument("--limit", type=float, default=5.0, help="Move limit (mm)")
    parser.add_argument("--snap", type=float, default=2.0, help="Snap distance for merging close nodes (mm), set to 0 to disable")
    parser.add_argument("--visualise", action="store_true", help="Visualise Result")
    args = parser.parse_args()
    
    # Init Problem
    problem = load_problem_config(args.problem)
    if not problem: return

    # Load JSON
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    # If tagged problem, load node_tags from JSON
    if hasattr(problem, 'load_tags_from_json'):
        problem.load_tags_from_json(args.input_json)
        
    nodes_data = data['graph']['nodes']
    edges_data = data['graph']['edges']
    
    # Manually parse edges to handle mixed types (lists inside rows)
    nodes = np.array(nodes_data)
    edges_list = []
    radii_list = []
    
    for e in edges_data:
        # e is [u, v, len, pts, r] or [u, v, r] or [u, v]
        u, v = int(e[0]), int(e[1])
        edges_list.append([u, v])
        
        # Radius extraction - Be Flexible
        r = 1.0
        if len(e) >= 5: r = float(e[4]) # Index 4 is Radius in Clean Graph
        elif len(e) >= 3: r = float(e[2]) # Index 2 might be Radius in some formats
        radii_list.append(r)
        
    edges = np.array(edges_list)
    radii = np.array(radii_list)
    
    # If tagged problem, load node_tags from JSON
    if hasattr(problem, 'load_tags_from_json'):
        problem.load_tags_from_json(args.input_json)
    
    node_tags_input = problem._node_tags if hasattr(problem, '_node_tags') else {}
        
    nodes_data = data['graph']['nodes']
    edges_data = data['graph']['edges']
    
    # Manually parse edges to handle mixed types (lists inside rows)
    nodes = np.array(nodes_data)
    edges_list = []
    radii_list = []
    
    for e in edges_data:
        # e is [u, v, len, pts, r] or [u, v, r] or [u, v]
        u, v = int(e[0]), int(e[1])
        edges_list.append([u, v])
        if len(e) >= 5:
            radii_list.append(float(e[4]))
        elif len(e) >= 3:
            radii_list.append(float(e[2]))
        else:
            radii_list.append(1.0)
            
    edges = np.array(edges_list)
    radii = np.array(radii_list)
    
    # Retrieve Target Volume if exists
    target_vol_meta = None
    if 'metadata' in data and 'target_volume' in data['metadata']:
        target_vol_meta = data['metadata']['target_volume']
        print(f"[Layout] Found Target Volume in Metadata: {target_vol_meta:.2f}")

    # Retrieve Design Bounds if exists
    design_bounds = None
    if 'metadata' in data and 'design_bounds' in data['metadata']:
        design_bounds = data['metadata']['design_bounds']
        print(f"[Layout] Found Design Bounds in Metadata")

    # Run Layout Opt
    res = optimize_layout(nodes, edges, radii, problem, move_limit=args.limit, 
                          visualize=args.visualise, target_volume_abs=target_vol_meta, 
                          snap_dist=args.snap, design_bounds=design_bounds,
                          node_tags=node_tags_input)
    
    if isinstance(res, tuple) and len(res) >= 6:
        nodes_new, edges_new, radii_new, node_tags_new = res[0], res[1], res[2], res[3]
    elif isinstance(res, tuple) and len(res) == 4:
        nodes_new, edges_new, radii_new, node_tags_new = res
    elif isinstance(res, tuple) and len(res) == 3:
        nodes_new, edges_new, radii_new = res
        node_tags_new = node_tags_input
    else:
        nodes_new = res
        edges_new = edges
        radii_new = radii
        node_tags_new = node_tags_input
    
    # Save
    edges_out_list = []
    for i, (u, v) in enumerate(edges_new):
        r = radii_new[i]
        edges_out_list.append([int(u), int(v), 1.0, [], float(r)])

    data['graph']['nodes'] = nodes_new.tolist()
    data['graph']['edges'] = edges_out_list
    data['graph']['node_tags'] = {str(k): v for k, v in node_tags_new.items()}
    # Remove old 'radii' key if exists to avoid confusion
    if 'radii' in data['graph']:
        del data['graph']['radii']
    
    # Update Curves (Points)
    # The curves correspond to edges. If we moved nodes, the curves (polylines) need to move.
    # However, our edge collapses/pruning might have made curves complex.
    # Linear interpolation for now: Just move endpoints? 
    # The intermediate points in 'curves' are explicit coordinates.
    # If we only optimize Nodes, the intermediates are "stuck".
    # This might look weird. 
    # Correct approach: Re-interpolate intermediates or Optimize Intermediates too? 
    # Optimizing intermediates explodes variables (16 edges * ~10 pts/edge).
    # Simple approach: Re-linearize edges (straight lines) for output to avoid broken polylines.
    # Or map intermediates proportional to node movement.
    # Let's just Linearize curves (remove intermediates) for the Optimized result. 
    # Optimization tends to straighten members anyway.
    
    new_curves = []
    # Use edges_new and radii_new, as nodes_new is re-indexed
    for i, e in enumerate(edges_new):
        u, v = int(e[0]), int(e[1])
        p1 = nodes_new[u]
        p2 = nodes_new[v]
        r = radii_new[i]
        
        # Create 2-point curve (Straight line)
        # Format: [ [x,y,z,r], [x,y,z,r] ]
        pts = [
            list(p1) + [r],
            list(p2) + [r]
        ]
        new_curves.append({'points': pts})
        
    data['curves'] = new_curves
    
    with open(args.output_json, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Layout] Saved to {args.output_json}")
    
    if args.visualise:
        print("[Viz] Showing Final Snapped Layout...")
        final_geoms_snapped = viz_graph_radii(nodes_new, edges_new, radii_new)
        show_step("Final Snapped Layout", final_geoms_snapped)

if __name__ == "__main__":
    main()
