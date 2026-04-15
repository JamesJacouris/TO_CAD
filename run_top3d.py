import numpy as np
import scipy.io
import argparse
from src.optimization.top3d import Top3D

def main():
    parser = argparse.ArgumentParser(description="Python Top3D Optimization (Cantilever)")
    parser.add_argument("--nelx", type=int, default=60, help="Elements in X")
    parser.add_argument("--nely", type=int, default=20, help="Elements in Y")
    parser.add_argument("--nelz", type=int, default=4, help="Elements in Z")
    parser.add_argument("--volfrac", type=float, default=0.3, help="Volume Fraction")
    parser.add_argument("--penal", type=float, default=3.0, help="Penalization Factor")
    parser.add_argument("--rmin", type=float, default=1.5, help="Filter Radius")
    parser.add_argument("--load_x", type=int, default=None, help="Load X Index (default=nelx)")
    parser.add_argument("--load_y", type=int, default=None, help="Load Y Index (default=nely)")
    parser.add_argument("--load_z", type=int, default=None, help="Load Z Index (default=nelz/2)")
    
    # Load Vector
    parser.add_argument("--load_fx", type=float, default=0.0, help="Load X Magnitude")
    parser.add_argument("--load_fy", type=float, default=-1.0, help="Load Y Magnitude")
    parser.add_argument("--load_fz", type=float, default=0.0, help="Load Z Magnitude")
    # Output and Control
    parser.add_argument("--max_loop", type=int, default=50, help="Max Iterations")
    parser.add_argument("--output", default="python_top3d_result.npz", help="Output .npz file")
    parser.add_argument("--export_stl", action="store_true",
                        help="Also export result as watertight STL (marching cubes isosurface)")
    # Problem Type
    parser.add_argument("--problem", type=str, default="cantilever", choices=["cantilever", "roof", "roof_slab", "elevated_slab", "bridge", "deck", "quadcopter", "simply_supported", "l_bracket", "obstacle_course", "vault", "curved_shell", "pipe_bracket"], help="Problem type")
    parser.add_argument("--load_dist", type=str, default="point", choices=["point", "surface_top", "surface_bottom"], help="Load distribution")
    # Quadcopter-specific
    parser.add_argument("--motor_arm_frac", type=float, default=0.1,
        help="Motor mount position as fraction of nelx/nely inset from each corner (default: 0.1 — motors near corners for long arms)")
    parser.add_argument("--load_patch_frac", type=float, default=0.1,
        help="Half-width of centre payload patch as fraction of nelx/nely (default: 0.1)")
    parser.add_argument("--motor_radius", type=int, default=0,
        help="[Quadcopter] Radius (elements) of circular passive void at each motor mount. 0 = no void (default).")
    parser.add_argument("--motor_bolt_spacing", type=int, default=0,
        help="[Quadcopter] Split each motor mount into 2 bolt columns separated by this many elements "
             "perpendicularly to the arm axis. Creates bending moment → parallel arm branches. 0 = single column (default).")
    parser.add_argument("--arm_load_n", type=int, default=0,
        help="[Quadcopter] Number of distributed load columns per arm placed between centre and motor. "
             "Creates arm bending moment → X-bracing. 0 = centre load only (default).")
    parser.add_argument("--arm_load_frac", type=float, default=0.3,
        help="[Quadcopter] Fraction of total load applied to arm load columns (remainder goes to centre patch). Default: 0.3.")
    parser.add_argument("--arm_void_width", type=int, default=0,
        help="[Quadcopter] Width (elements) of passive void strip along each arm centreline. "
             "Physically separates the two chord members so thinning produces two distinct skeleton branches. "
             "Recommended: 3-5. 0 = disabled (default).")

    args = parser.parse_args()
    
    print(f"=== Python Top3D Optimization: {args.problem.upper()} ===")
    print(f"Mesh: {args.nelx}x{args.nely}x{args.nelz}, VolFrac: {args.volfrac}")
    
    # Initialize Solver
    solver = Top3D(args.nelx, args.nely, args.nelz, args.volfrac, args.penal, args.rmin)
    
    # --- Define Problem Boundary Conditions ---
    # Top3D node ordering (F-order): Y varies fastest, then X, then Z
    # node_grid = np.arange(n_node).reshape((ny+1, nx+1, nz+1), order='F')
    # So node[n] has coords (j, i, k) where n = j + i*(ny+1) + k*(ny+1)*(nx+1)
    jl, il, kl = np.meshgrid(np.arange(args.nely+1), np.arange(args.nelx+1), np.arange(args.nelz+1), indexing='ij')
    il_flat = il.flatten(order='F')
    jl_flat = jl.flatten(order='F')
    kl_flat = kl.flatten(order='F')

    if args.problem == "roof":
        # 1. Fixed BC: 4 Bottom Corners (z=0) set inset by 1 to create a 2x2 element support pillar
        nx, ny = args.nelx, args.nely
        corners = [(1, 1, 0), (nx-1, 1, 0), (1, ny-1, 0), (nx-1, ny-1, 0)]
        fixed_node_indices = []
        for (cx, cy, cz) in corners:
            dist = (il_flat - cx)**2 + (jl_flat - cy)**2 + (kl_flat - cz)**2
            fixed_node_indices.append(np.argmin(dist))
        
        fixed_dofs_list = []
        for n in fixed_node_indices:
            fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"Fixed {len(fixed_node_indices)} Corner Nodes at Z=0 (2x2 Support Pillars).")
        
        # Default Load for Roof: Distributed across top surface
        if args.load_x is None and args.load_dist == "point":
            args.load_dist = "surface_top"

    elif args.problem == "roof_slab":
        # Thin slab (roof) with interior point supports — classic hybrid plate+beam case.
        # Fixed BC: Interior point supports in a grid pattern on bottom surface (z=0)
        nx, ny = args.nelx, args.nely

        # 3x3 grid of interior supports (excludes boundaries)
        support_positions = []
        for i in np.linspace(0.25, 0.75, 3):
            for j in np.linspace(0.25, 0.75, 3):
                support_positions.append((int(i * nx), int(j * ny), 0))

        fixed_node_indices = []
        for (cx, cy, cz) in support_positions:
            dist = (il_flat - cx)**2 + (jl_flat - cy)**2 + (kl_flat - cz)**2
            fixed_node_indices.append(np.argmin(dist))

        fixed_dofs_list = []
        for n in fixed_node_indices:
            fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"Fixed {len(fixed_node_indices)} Interior Nodes at Z=0 (3x3 Support Grid).")

        # Default Load for Roof Slab: Distributed across top surface (downward)
        if args.load_x is None and args.load_dist == "point":
            args.load_dist = "surface_top"
            if args.load_fy == -1.0:  # If using default load
                args.load_fy = -100.0  # Heavier load for slab

    elif args.problem == "elevated_slab":
        # Elevated slab on 4 pad columns — showcase for hybrid beam+plate pipeline.
        # Physical analogy: table / building floor slab on 4 columns.
        # Pad supports (radius=2 nodes, ~13 nodes each) instead of single-node
        # supports prevent stress-concentration lattice artifacts and guide SIMP
        # toward clean column formation.
        #
        # Load is in -Z (through the slab and columns) for a natural gravity-like
        # path that produces balanced plate + beam topology.
        #
        # Recommended:
        #   python run_top3d.py --problem elevated_slab \
        #       --nelx 40 --nely 40 --nelz 30 \
        #       --volfrac 0.12 --rmin 2.0 --max_loop 80 \
        #       --output elevated_slab.npz
        nx, ny, nz = args.nelx, args.nely, args.nelz

        # Pad centres: 20% inset from each corner, bottom face (z=0)
        inset_x = max(2, int(round(0.20 * nx)))
        inset_y = max(2, int(round(0.20 * ny)))
        pad_centres = [
            (inset_x,      inset_y,      0),
            (nx - inset_x, inset_y,      0),
            (inset_x,      ny - inset_y, 0),
            (nx - inset_x, ny - inset_y, 0),
        ]

        # Radius-2 pad: ~13 fixed nodes per corner (disc in XY at z=0).
        pad_radius = 2
        fixed_dofs_list = []
        total_fixed_nodes = 0
        for (cx, cy, cz) in pad_centres:
            pad_mask = (
                ((il_flat - cx)**2 + (jl_flat - cy)**2 <= pad_radius**2)
                & (kl_flat == cz)
            )
            pad_nodes = np.where(pad_mask)[0]
            total_fixed_nodes += len(pad_nodes)
            for n in pad_nodes:
                fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"[elevated_slab] Fixed {total_fixed_nodes} nodes across 4 pad supports "
              f"(radius={pad_radius}, inset=({inset_x},{inset_y})).")
        for (cx, cy, cz) in pad_centres:
            print(f"  Pad at ({cx}, {cy}, {cz})")

        # Load: distributed across top surface (z=nz), in -Z direction.
        # This creates a gravity-like load path: slab -> columns -> supports.
        top_mask = (kl_flat == nz)
        top_nodes = np.where(top_mask)[0]
        n_top = len(top_nodes)
        fz_total = args.load_fz if args.load_fz != 0.0 else -1.0
        for n in top_nodes:
            solver.set_load(3*n + 2, fz_total / n_top)
        print(f"[elevated_slab] Applied {fz_total:.1f} N (-Z) distributed across "
              f"{n_top} top-surface nodes (z={nz}).")
        args.load_dist = "already_applied"

    elif args.problem == "quadcopter":
        # X-config quadcopter frame (corrected formulation):
        # - 4 motor mounts = FULL Z-COLUMNS at XY corners (or bolt pairs for branching arms)
        # - Centre payload  = FULL Z-COLUMN at XY centre (+ optional arm intermediate loads)
        #
        # Fixing/loading full Z-columns forces SIMP to route material IN the XY
        # plane (diagonal arms) rather than through-Z (diagonal pillar arches).
        nx, ny = args.nelx, args.nely
        frac = args.motor_arm_frac
        mx = max(1, int(round(frac * nx)))
        my = max(1, int(round(frac * ny)))
        cx_hub, cy_hub = nx // 2, ny // 2

        motor_xy = [
            (mx,      my),
            (nx - mx, my),
            (mx,      ny - my),
            (nx - mx, ny - my),
        ]

        # --- Motor mount columns ---
        # motor_bolt_spacing > 0: split each motor into 2 bolt columns perpendicular
        # to the arm axis.  This creates a moment couple at the motor attachment,
        # which forces SIMP to route two parallel chord members along each arm
        # (Warren/Pratt truss) rather than a single diagonal.
        if args.motor_bolt_spacing > 0:
            S = args.motor_bolt_spacing
            fixed_cols = []
            for (cx, cy) in motor_xy:
                arm_x = cx - cx_hub
                arm_y = cy - cy_hub
                arm_len = np.sqrt(arm_x**2 + arm_y**2)
                # Unit perpendicular (90° CCW rotation of arm direction)
                perp_x = -arm_y / arm_len
                perp_y =  arm_x / arm_len
                for sign in (+1, -1):
                    bx = int(round(cx + sign * S / 2.0 * perp_x))
                    by = int(round(cy + sign * S / 2.0 * perp_y))
                    bx = int(np.clip(bx, 1, nx - 1))
                    by = int(np.clip(by, 1, ny - 1))
                    fixed_cols.append((bx, by))
            print(f"Motor bolt pairs (spacing={S}): {fixed_cols}")
        else:
            fixed_cols = motor_xy[:]

        # Fix complete Z-column at each motor/bolt position (all z-layers)
        fixed_dofs_list = []
        total_fixed_nodes = 0
        for (cx, cy) in fixed_cols:
            col_mask = (il_flat == cx) & (jl_flat == cy)
            col_nodes = np.where(col_mask)[0]
            total_fixed_nodes += len(col_nodes)
            for n in col_nodes:
                fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"Fixed {len(fixed_cols)} motor Z-columns (arm_frac={frac:.2f}): {motor_xy}")
        print(f"  {total_fixed_nodes} nodes fixed across {args.nelz+1} Z-layers.")

        # --- Passive voids (unified block) ---
        # Build one passive array; contributions from motor corners and arm
        # centreline strips are accumulated before a single set_passive_void call.
        passive = np.zeros((ny, nx, args.nelz), dtype=bool)
        xi_arr = np.arange(nx)
        yi_arr = np.arange(ny)
        XX, YY = np.meshgrid(xi_arr, yi_arr)

        # Motor corner cutouts: void centred at corner keeps arm-tip column outside.
        if args.motor_radius > 0:
            r = args.motor_radius
            min_tip_dist = min(mx, my)
            if min_tip_dist <= r:
                print(f"WARNING: motor_radius={r} >= arm tip distance={min_tip_dist}. "
                      f"Increase --motor_arm_frac or decrease --motor_radius.")
            for (cx, cy) in motor_xy:
                corner_x = 0 if cx < nx // 2 else nx - 1
                corner_y = 0 if cy < ny // 2 else ny - 1
                mask2d = (XX - corner_x)**2 + (YY - corner_y)**2 <= r**2
                passive[mask2d, :] = True
            print(f"Passive motor cutouts: radius={r} elements at corners, "
                  f"{int(passive.sum())} void voxels.")

        # Arm centreline void strips: a thin void band along each arm's centreline
        # physically forces SIMP to route two chord members on either side of the gap.
        # This guarantees the skeletoniser sees two separate paths per arm.
        if args.arm_void_width > 0:
            W = args.arm_void_width
            half_w = W / 2.0
            pr_tmp = args.load_patch_frac
            hub_excl = max(int(round(pr_tmp * nx)) + 3, 3)   # stay outside hub plate
            motor_excl = max(min(mx, my) // 2, 3)             # stay clear of bolt cols
            n_before = int(passive.sum())
            for (mx_c, my_c) in motor_xy:
                arm_vec = np.array([mx_c - cx_hub, my_c - cy_hub], dtype=float)
                arm_len = float(np.linalg.norm(arm_vec))
                arm_unit = arm_vec / arm_len
                perp_unit = np.array([-arm_unit[1], arm_unit[0]])
                dx = (XX - cx_hub).astype(float)
                dy = (YY - cy_hub).astype(float)
                proj_arm  = dx * arm_unit[0]  + dy * arm_unit[1]
                proj_perp = dx * perp_unit[0] + dy * perp_unit[1]
                strip = (
                    (proj_arm  > hub_excl) &
                    (proj_arm  < arm_len - motor_excl) &
                    (np.abs(proj_perp) <= half_w)
                )
                passive[strip, :] = True
            n_strip = int(passive.sum()) - n_before
            print(f"Arm centreline void strips: width={W} elements, {n_strip} new void voxels "
                  f"(hub_excl={hub_excl}, motor_excl={motor_excl}).")

        if passive.any():
            solver.set_passive_void(passive)
            print(f"Total passive void: {int(passive.sum())} voxels.")

        # --- Load application ---
        # Total load split: arm_load_frac → arm columns; 1-arm_load_frac → hub patch.
        pr = args.load_patch_frac
        px = max(0, int(round(pr * nx)))
        py = max(0, int(round(pr * ny)))
        patch_mask = (
            (np.abs(il_flat - cx_hub) <= px) &
            (np.abs(jl_flat - cy_hub) <= py)
        )
        patch_nodes = np.where(patch_mask)[0]
        n_patch = len(patch_nodes)
        if n_patch == 0:
            raise ValueError("Centre-patch contains 0 nodes — increase --load_patch_frac or domain size.")

        centre_frac = 1.0 - args.arm_load_frac if args.arm_load_n > 0 else 1.0
        for n in patch_nodes:
            if args.load_fx != 0.0: solver.set_load(3*n,     args.load_fx * centre_frac / n_patch)
            if args.load_fy != 0.0: solver.set_load(3*n + 1, args.load_fy * centre_frac / n_patch)
            if args.load_fz != 0.0: solver.set_load(3*n + 2, args.load_fz * centre_frac / n_patch)
        print(f"Applied centre-patch load ({centre_frac*100:.0f}%) to {n_patch} nodes "
              f"(patch ±{px}x{py} around XY centre, all Z-layers).")

        # --- Distributed arm loads ---
        # When motor_bolt_spacing > 0, arm loads are applied to PAIRED columns at the
        # same perpendicular offset as the bolt columns — not on the centreline.
        # This keeps each chord independently loaded and avoids loading void voxels.
        if args.arm_load_n > 0:
            N = args.arm_load_n
            arm_frac = args.arm_load_frac
            S = args.motor_bolt_spacing
            use_pairs = S > 0

            arm_load_cols = []   # list of (cx, cy) XY positions to load
            for (mx_c, my_c) in motor_xy:
                arm_vec = np.array([mx_c - cx_hub, my_c - cy_hub], dtype=float)
                arm_len = float(np.linalg.norm(arm_vec))
                arm_unit = arm_vec / arm_len
                perp_unit = np.array([-arm_unit[1], arm_unit[0]])
                for k in range(1, N + 1):
                    t = k / (N + 1)
                    ax = cx_hub + t * (mx_c - cx_hub)
                    ay = cy_hub + t * (my_c - cy_hub)
                    if use_pairs:
                        # Paired columns at ±S/2 perp — one per chord member.
                        for sign in (+1, -1):
                            bx = int(round(ax + sign * S / 2.0 * perp_unit[0]))
                            by = int(round(ay + sign * S / 2.0 * perp_unit[1]))
                            bx = int(np.clip(bx, 0, nx))
                            by = int(np.clip(by, 0, ny))
                            arm_load_cols.append((bx, by))
                    else:
                        arm_load_cols.append((int(round(ax)), int(round(ay))))

            all_arm_nodes = []
            for (ax, ay) in arm_load_cols:
                col_mask = (il_flat == ax) & (jl_flat == ay)
                all_arm_nodes.extend(np.where(col_mask)[0].tolist())

            n_arm = len(all_arm_nodes)
            if n_arm == 0:
                print("WARNING: arm load columns contain 0 nodes — skipping arm loads.")
            else:
                for n in all_arm_nodes:
                    if args.load_fx != 0.0: solver.set_load(3*n,     args.load_fx * arm_frac / n_arm)
                    if args.load_fy != 0.0: solver.set_load(3*n + 1, args.load_fy * arm_frac / n_arm)
                    if args.load_fz != 0.0: solver.set_load(3*n + 2, args.load_fz * arm_frac / n_arm)
                mode_str = f"paired (bolt_spacing={S})" if use_pairs else "centreline"
                print(f"Applied arm loads ({arm_frac*100:.0f}%) [{mode_str}] to {n_arm} nodes at "
                      f"{len(arm_load_cols)} columns ({N}×{'2' if use_pairs else '1'} per arm × 4 arms).")

        print(f"Total force: [{args.load_fx}, {args.load_fy}, {args.load_fz}]")
        if args.load_fy == -1.0:
            print("WARNING: using default load_fy=-1.0; consider --load_fy -100.0 for meaningful results.")
        args.load_dist = "already_applied"

    elif args.problem == "bridge" or args.problem == "deck":
        # 2. Fixed BC: Entire Bottom Surface (z=0)
        fixed_node_indices = np.where(kl_flat == 0)[0]
        fixed_dofs_list = []
        for n in fixed_node_indices:
            fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"Fixed Entire Bottom Surface (Z=0) (Deck Support).")
        
        # If problem is 'deck', apply 4 corners load IF no specific load is given
        if args.problem == "deck" and args.load_x is None and args.load_dist == "point":
            nx, ny, nz = args.nelx, args.nely, args.nelz
            top_corners = [(0, 0, nz), (nx, 0, nz), (0, ny, nz), (nx, ny, nz)]
            for (cx, cy, cz) in top_corners:
                dist_sq = (il_flat - cx)**2 + (jl_flat - cy)**2 + (kl_flat - cz)**2
                n = np.argmin(dist_sq)
                if args.load_fx != 0.0: solver.set_load(3*n, args.load_fx / 4.0)
                if args.load_fy != 0.0: solver.set_load(3*n+1, args.load_fy / 4.0)
                if args.load_fz != 0.0: solver.set_load(3*n+2, args.load_fz / 4.0)
            print(f"Applied Distributed Loads to 4 Top Corners (Z={nz}).")
            # Set flag to skip standard point load application
            args.load_dist = "already_applied"

    elif args.problem == "simply_supported":
        # Simply-supported span: end-column supports at (x=0, z=0) and (x=nelx, z=0),
        # central top load at (x=nelx//2, z=nelz).
        # Produces classic arch/Michell truss topology — ideal for curved beam IGA testing:
        # two diagonal compression members arc from each support to the central load point;
        # members are 20-35 voxels long, naturally curved, and stable for IGA condensation.

        # Fixed: left and right bottom edge columns (x=0 or x=nelx, z=0, all y)
        fixed_mask = ((il_flat == 0) | (il_flat == args.nelx)) & (kl_flat == 0)
        fixed_node_indices = np.where(fixed_mask)[0]
        fixed_dofs_list = []
        for n in fixed_node_indices:
            fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"[simply_supported] Fixed {len(fixed_node_indices)} nodes at "
              f"(x=0, z=0) and (x={args.nelx}, z=0) — end-column supports.")

        # Loaded: centre top column (x=nelx//2, z=nelz, all y) — downward (-Z)
        cx = args.nelx // 2
        load_mask = (il_flat == cx) & (kl_flat == args.nelz)
        load_node_indices = np.where(load_mask)[0]
        n_load = len(load_node_indices)
        if n_load == 0:
            raise ValueError("[simply_supported] No load nodes found at centre top — check nelx/nelz.")
        fz_total = args.load_fz if args.load_fz != 0.0 else -1000.0  # default: 1000 N downward
        for n in load_node_indices:
            solver.set_load(3*n + 2, fz_total / n_load)
        print(f"[simply_supported] Applied {fz_total:.1f} N (-Z) across {n_load} nodes "
              f"at x={cx}, z={args.nelz} (centre top).")
        args.load_dist = "already_applied"

    elif args.problem == "l_bracket":
        # L-bracket: classic TO benchmark that produces curved inner-corner members.
        # Full rectangular domain with upper-right quadrant removed (passive void).
        # Fixed at top of vertical arm (x=0, z=nelz); loaded at tip of horizontal
        # arm (x=nelx, z=0) pointing downward.  The stress concentration at the
        # inner corner (x≈nelx/2, z≈nelz/2) always develops a curved fillet member.
        # Recommended: --nelx 40 --nely 10 --nelz 40 --volfrac 0.30
        nx, ny, nz = args.nelx, args.nely, args.nelz

        # Passive void: upper-right quadrant (removes 25% of domain)
        passive = np.zeros((ny, nx, nz), dtype=bool)
        x_cut = nx // 2
        z_cut = nz // 2
        passive[:, x_cut:, z_cut:] = True
        solver.set_passive_void(passive)
        n_void = int(passive.sum())
        print(f"[l_bracket] Passive void: x∈[{x_cut},{nx}), z∈[{z_cut},{nz}) — "
              f"{n_void} voxels ({100*n_void/(nx*ny*nz):.0f}% of domain)")

        # Fixed: entire top face of vertical arm (z=nz, x∈[0, x_cut], all y)
        # Must fix a face, not just an edge — single column leaves rigid body modes.
        fixed_mask = (kl_flat == nz) & (il_flat <= x_cut)
        fixed_nodes = np.where(fixed_mask)[0]
        fixed_dofs_list = []
        for n in fixed_nodes:
            fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"[l_bracket] Fixed {len(fixed_nodes)} nodes at z={nz}, x∈[0,{x_cut}] (top face of vertical arm)")

        # Load: tip of horizontal arm — column at (x=nx, z=0, all y), downward (-Z)
        load_mask = (il_flat == nx) & (kl_flat == 0)
        load_nodes = np.where(load_mask)[0]
        n_load = len(load_nodes)
        if n_load == 0:
            raise ValueError("[l_bracket] No load nodes found at (x=nelx, z=0).")
        fz_total = args.load_fz if args.load_fz != 0.0 else -1000.0
        for n in load_nodes:
            solver.set_load(3*n + 2, fz_total / n_load)
        print(f"[l_bracket] Applied {fz_total:.1f} N (-Z) across {n_load} nodes "
              f"at x={nx}, z=0")
        args.load_dist = "already_applied"

    elif args.problem == "obstacle_course":
        # Cantilever with staggered cylindrical voids that force S-curved load paths.
        # Alternating high/low obstacles compel diagonal web members to weave
        # around them, producing genuinely curved skeleton edges after thinning.
        # Top3D convention: Y is vertical (load direction), Z is depth (thin).
        # Recommended: --nelx 80 --nely 40 --nelz 10 --volfrac 0.25 --rmin 2.0
        #
        #  y=ny ┌────┐    ○       ○       ○       ┌────┐
        #       │    │       ○       ○       ○     │    │
        #       │    │    ○       ○       ○       │    │
        #  y=0  └────┘       ○       ○       ○     └────┘ ↓ Load (-Y)
        #       x=0   voids alternate high/low      x=nx
        nx, ny, nz = args.nelx, args.nely, args.nelz

        # Staggered cylindrical voids (through-Z depth)
        passive = np.zeros((ny, nx, nz), dtype=bool)
        r_hole = max(3, min(nx, ny) // 10)  # proportional to span and height

        # 6 voids in alternating high/low pattern along the span
        n_holes = 6
        spacing = nx / (n_holes + 1)
        holes = []
        for i in range(n_holes):
            cx = int((i + 1) * spacing)
            cy = int(ny * 0.67) if i % 2 == 0 else int(ny * 0.33)
            holes.append((cx, cy))

        for (cx, cy) in holes:
            for xi in range(max(0, cx - r_hole), min(nx, cx + r_hole + 1)):
                for yi in range(max(0, cy - r_hole), min(ny, cy + r_hole + 1)):
                    if (xi - cx)**2 + (yi - cy)**2 <= r_hole**2:
                        passive[yi, xi, :] = True  # through all Z (depth) layers

        solver.set_passive_void(passive)
        n_void = int(passive.sum())
        print(f"[obstacle_course] {n_holes} cylindrical voids (r={r_hole}) at:")
        for i, (cx, cy) in enumerate(holes):
            print(f"  Hole {i}: x={cx}, y={cy}")
        print(f"  {n_void} void voxels ({100*n_void/(nx*ny*nz):.0f}% of domain)")

        # Fixed: entire left face (x=0)
        fixed_mask = (il_flat == 0)
        fixed_nodes = np.where(fixed_mask)[0]
        fixed_dofs_list = []
        for n in fixed_nodes:
            fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"[obstacle_course] Fixed {len(fixed_nodes)} nodes at x=0 (left face)")

        # Load: right tip at mid-height (x=nx, y=ny//2, all z) — downward (-Y)
        load_mask = (il_flat == nx) & (jl_flat == ny // 2)
        load_nodes = np.where(load_mask)[0]
        n_load = len(load_nodes)
        if n_load == 0:
            raise ValueError("[obstacle_course] No load nodes found.")
        fy_total = args.load_fy if args.load_fy != -1.0 else -1000.0
        for n in load_nodes:
            solver.set_load(3*n + 1, fy_total / n_load)
        print(f"[obstacle_course] Applied {fy_total:.1f} N (-Y, downward) at x={nx}, "
              f"y={ny//2} ({n_load} nodes)")
        args.load_dist = "already_applied"

    elif args.problem == "vault":
        # Barrel vault: wide bottom-edge supports + narrow centre-top load strip.
        # The offset between wide supports and narrow load forces material into
        # curved arch paths (parabolic/catenary compression lines).
        # Recommended: --nelx 60 --nely 10 --nelz 30 --volfrac 0.12
        nx, ny, nz = args.nelx, args.nely, args.nelz

        # Fixed: wide bottom-edge supports (x≤3 and x≥nx-3, z=0, all y)
        pad = max(2, nx // 15)  # ~3-4 elements wide support base
        fixed_mask = ((il_flat <= pad) | (il_flat >= nx - pad)) & (kl_flat == 0)
        fixed_nodes = np.where(fixed_mask)[0]
        fixed_dofs_list = []
        for n in fixed_nodes:
            fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"[vault] Fixed {len(fixed_nodes)} nodes at bottom edges "
              f"(x≤{pad} and x≥{nx-pad}, z=0, all y)")

        # Load: narrow centre strip on top surface (x ∈ [cx-strip, cx+strip], z=nz)
        cx = nx // 2
        strip = max(1, nx // 10)  # ~10% of span width
        load_mask = (
            (il_flat >= cx - strip) & (il_flat <= cx + strip) &
            (kl_flat == nz)
        )
        load_nodes = np.where(load_mask)[0]
        n_load = len(load_nodes)
        if n_load == 0:
            raise ValueError("[vault] No load nodes found at centre top strip.")
        fy_total = args.load_fy if args.load_fy != -1.0 else -1000.0
        for n in load_nodes:
            solver.set_load(3*n + 1, fy_total / n_load)
        print(f"[vault] Applied {fy_total:.1f} N (-Y) across {n_load} nodes "
              f"at centre strip x∈[{cx-strip},{cx+strip}], z={nz}")
        args.load_dist = "already_applied"

    elif args.problem == "curved_shell":
        # Curved shell roof: wide edge supports + full top surface load.
        # Produces barrel vault / catenary shell topology — ideal for testing
        # the curved surface classification and FreeCAD offset shell pipeline.
        # Recommended: --nelx 60 --nely 20 --nelz 30 --volfrac 0.10 --rmin 2.0
        nx, ny, nz = args.nelx, args.nely, args.nelz

        # Fixed: wide pad supports at both X-edges, bottom surface (z=0), all Y
        pad = max(3, nx // 15)
        fixed_mask = ((il_flat <= pad) | (il_flat >= nx - pad)) & (kl_flat == 0)
        fixed_nodes = np.where(fixed_mask)[0]
        fixed_dofs_list = []
        for n in fixed_nodes:
            fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"[curved_shell] Fixed {len(fixed_nodes)} nodes at bottom edge pads "
              f"(x<={pad} and x>={nx-pad}, z=0, all y)")

        # Load: entire top surface (z=nz), distributed downward
        load_mask = (kl_flat == nz)
        load_nodes = np.where(load_mask)[0]
        n_load = len(load_nodes)
        fy_total = args.load_fy if args.load_fy != -1.0 else -1000.0
        for n in load_nodes:
            solver.set_load(3*n + 1, fy_total / n_load)
        print(f"[curved_shell] Applied {fy_total:.1f} N (-Y) across {n_load} nodes "
              f"on full top surface (z={nz})")
        args.load_dist = "already_applied"

    elif args.problem == "pipe_bracket":
        # Pipe bracket from Yin's paper: two cylindrical pipe openings in a cuboid.
        # Fixed at 4 vertical corner edges, loaded at 4 cardinal points per pipe.
        # Produces mixed beam-plate topology with curved shell surfaces around pipes.
        # Paper params: nelx=120, nely=60, nelz=40, p=3, rmin=3, Vf=0.1, 100 iters.
        # Cylinder centres at x=0.3*nelx and x=0.7*nelx, y=nely/2, R=0.3*nely.
        # For faster testing: --nelx 60 --nely 30 --nelz 20
        nx, ny, nz = args.nelx, args.nely, args.nelz

        # Cylinder parameters (scale proportionally with domain)
        R = max(3, int(round(0.3 * ny)))
        cx1 = int(round(0.3 * nx))
        cx2 = int(round(0.7 * nx))
        cy_ctr = ny // 2
        print(f"[pipe_bracket] Domain: {nx}x{ny}x{nz}")
        print(f"[pipe_bracket] Cylinders: R={R}, centres ({cx1},{cy_ctr}) and ({cx2},{cy_ctr})")

        # Passive voids: two through-Z cylinders
        passive = np.zeros((ny, nx, nz), dtype=bool)
        xi_arr = np.arange(nx)
        yi_arr = np.arange(ny)
        XX, YY = np.meshgrid(xi_arr, yi_arr)
        for cxp in [cx1, cx2]:
            mask2d = (XX - cxp)**2 + (YY - cy_ctr)**2 <= R**2
            passive[mask2d, :] = True
        solver.set_passive_void(passive)
        n_void = int(passive.sum())
        print(f"[pipe_bracket] Passive voids: {n_void} elements "
              f"({100*n_void/(nx*ny*nz):.1f}% of domain)")

        # Fixed: 4 vertical corner edges (running in Y direction, all nely+1 nodes)
        fixed_mask = (
            ((il_flat == 0) | (il_flat == nx)) &
            ((kl_flat == 0) | (kl_flat == nz))
        )
        fixed_nodes = np.where(fixed_mask)[0]
        fixed_dofs_list = []
        for n in fixed_nodes:
            fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"[pipe_bracket] Fixed {len(fixed_nodes)} nodes at 4 vertical corner edges")

        # Loads: 4 cardinal points per pipe, each applying F=100 downward (-Y)
        F_per_point = 100.0
        load_points = []
        for cxp in [cx1, cx2]:
            load_points.extend([
                (cxp, cy_ctr + R, "top"),
                (cxp, cy_ctr - R, "bottom"),
                (cxp - R, cy_ctr, "left"),
                (cxp + R, cy_ctr, "right"),
            ])

        total_loaded = 0
        for (lx, ly, label) in load_points:
            lx = int(np.clip(lx, 0, nx))
            ly = int(np.clip(ly, 0, ny))
            load_mask = (il_flat == lx) & (jl_flat == ly)
            load_nodes_line = np.where(load_mask)[0]
            n_line = len(load_nodes_line)
            if n_line == 0:
                print(f"  WARNING: No nodes at ({lx},{ly}) [{label}] — skipping")
                continue
            for n in load_nodes_line:
                solver.set_load(3*n + 1, -F_per_point / n_line)
            total_loaded += n_line
            print(f"  Load {label}: ({lx},{ly}) -> {n_line} Z-nodes, "
                  f"{F_per_point:.0f} N (-Y)")

        total_force = F_per_point * len(load_points)
        print(f"[pipe_bracket] Total: {total_force:.0f} N across {total_loaded} nodes "
              f"({len(load_points)} support points)")
        args.load_dist = "already_applied"

    else:
        # Default: Cantilever (Fixed Left Wall x=0)
        fixed_node_indices = np.where(il_flat == 0)[0]
        fixed_dofs_list = []
        for n in fixed_node_indices:
            fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        solver.set_fixed_dofs(np.array(fixed_dofs_list))
        print(f"Fixed {len(fixed_node_indices)} nodes on Left Wall (x=0) (Cantilever Setup).")
    
    # 2. Load Application
    if args.load_dist == "surface_top":
        top_node_indices = np.where(kl_flat == args.nelz)[0]
        n_nodes = len(top_node_indices)
        for n in top_node_indices:
            if args.load_fx != 0.0: solver.set_load(3*n, args.load_fx / n_nodes)
            if args.load_fy != 0.0: solver.set_load(3*n+1, args.load_fy / n_nodes)
            if args.load_fz != 0.0: solver.set_load(3*n+2, args.load_fz / n_nodes)
        print(f"Applied Distributed Load to Top Surface ({n_nodes} nodes).")
        print(f"Total Force Vector: [{args.load_fx}, {args.load_fy}, {args.load_fz}]")

    elif args.load_dist == "surface_bottom":
        bottom_node_indices = np.where(kl_flat == 0)[0]
        n_nodes = len(bottom_node_indices)
        for n in bottom_node_indices:
            if args.load_fx != 0.0: solver.set_load(3*n, args.load_fx / n_nodes)
            if args.load_fy != 0.0: solver.set_load(3*n+1, args.load_fy / n_nodes)
            if args.load_fz != 0.0: solver.set_load(3*n+2, args.load_fz / n_nodes)
        print(f"Applied Distributed Load to Bottom Surface ({n_nodes} nodes).")
    
    elif args.load_dist == "already_applied":
        pass
    else:
        # Point Load (Standard)
        load_x = args.load_x if args.load_x is not None else args.nelx
        load_y = args.load_y if args.load_y is not None else args.nely
        load_z = args.load_z if args.load_z is not None else (args.nelz // 2)
        
        # Find closest node
        dist = (il_flat - load_x)**2 + (jl_flat - load_y)**2 + (kl_flat - load_z)**2
        load_node_idx = np.argmin(dist)
        
        if args.load_fx != 0.0: solver.set_load(3*load_node_idx, args.load_fx)
        if args.load_fy != 0.0: solver.set_load(3*load_node_idx + 1, args.load_fy)
        if args.load_fz != 0.0: solver.set_load(3*load_node_idx + 2, args.load_fz)
        
        print(f"Applied Point Load at Node {load_node_idx} (Target: {load_x},{load_y},{load_z}).")
        print(f"Force Vector: [{args.load_fx}, {args.load_fy}, {args.load_fz}]")

    # Change Iterations
    xPhys, compliance_history = solver.optimize(max_loop=args.max_loop)
    
    # Export
    # Save rho, bc_tags, pitch, origin as .npz (compressed dict)
    
    # Compute total load vector from the solver's force array (3 DOFs per node)
    f = solver.f
    total_fx = np.sum(f[0::3])
    total_fy = np.sum(f[1::3])
    total_fz = np.sum(f[2::3])
    load_vector_total = np.array([total_fx, total_fy, total_fz])
    print(f"Total Load Vector: [{total_fx:.2f}, {total_fy:.2f}, {total_fz:.2f}]")

    output_dict = {
        'rho': xPhys,
        'bc_tags': solver.bc_tags,
        'pitch': 1.0,
        'origin': [0, 0, 0],
        'load_vector': load_vector_total,
        'compliance_history': np.array(compliance_history),
        'E0': 1e3,
    }
    
    np.savez(args.output, **output_dict)

    print(f"Saved results to {args.output}")
    print(f"BC Tags Info: Fixed={np.sum(solver.bc_tags==1)}, Loaded={np.sum(solver.bc_tags==2)}")

    if args.export_stl:
        from src.export.npz_to_stl import export_top3d_stl
        stl_path = args.output.replace('.npz', '.stl')
        export_top3d_stl(args.output, stl_path, vol_thresh=0.3)

if __name__ == "__main__":
    main()
