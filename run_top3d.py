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
    parser.add_argument("--top3d_iters", "--max_loop", type=int, default=50, help="Max Iterations")
    parser.add_argument("--output", default="python_top3d_result.npz", help="Output .npz file")
    # Problem Type
    parser.add_argument("--problem", type=str, default="cantilever", choices=["cantilever", "roof", "roof_slab", "bridge", "deck"], help="Problem type")
    parser.add_argument("--load_dist", type=str, default="point", choices=["point", "surface_top", "surface_bottom"], help="Load distribution")
    # Quality improvements
    parser.add_argument("--no-p-continuation", dest="use_p_continuation", action="store_false",
                        help="Disable p-continuation penalty ramp (enabled by default)")
    parser.set_defaults(use_p_continuation=True)
    
    args = parser.parse_args()
    
    print(f"=== Python Top3D Optimization: {args.problem.upper()} ===")
    print(f"Mesh: {args.nelx}x{args.nely}x{args.nelz}, VolFrac: {args.volfrac}")
    
    # Initialize Solver
    solver = Top3D(args.nelx, args.nely, args.nelz, args.volfrac, args.penal, args.rmin,
                   use_p_continuation=args.use_p_continuation)
    
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
    xPhys = solver.optimize(max_loop=args.top3d_iters) 
    
    # Export
    # Save rho, bc_tags, pitch, origin as .npz (compressed dict)
    
    output_dict = {
        'rho': xPhys,
        'bc_tags': solver.bc_tags,
        'pitch': 1.0,
        'origin': [0, 0, 0]
    }
    
    np.savez(args.output, **output_dict)
    
    print(f"Saved results to {args.output}")
    print(f"BC Tags Info: Fixed={np.sum(solver.bc_tags==1)}, Loaded={np.sum(solver.bc_tags==2)}")

if __name__ == "__main__":
    main()
