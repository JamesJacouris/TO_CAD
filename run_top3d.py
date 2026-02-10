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
    # Re-adding missing args
    parser.add_argument("--max_loop", type=int, default=50, help="Max Iterations")
    parser.add_argument("--output", default="python_top3d_result.npz", help="Output .npz file")
    
    args = parser.parse_args()
    
    print("=== Python Top3D Optimization ===")
    print(f"Mesh: {args.nelx}x{args.nely}x{args.nelz}, VolFrac: {args.volfrac}")
    
    # Initialize Solver
    solver = Top3D(args.nelx, args.nely, args.nelz, args.volfrac, args.penal, args.rmin)
    
    # --- Define Problem (Cantilever) ---

    # 1. Fixed BC: Left Wall (x=0)
    # Re-implmenting robust node finding
    il, jl, kl = np.meshgrid(np.arange(args.nelx+1), np.arange(args.nely+1), np.arange(args.nelz+1), indexing='ij')
    il_flat = il.flatten()
    jl_flat = jl.flatten()
    kl_flat = kl.flatten()
    
    # Find indices where x=0
    fixed_node_indices = np.where(il_flat == 0)[0]
    
    fixed_dofs_list = []
    for n in fixed_node_indices:
        fixed_dofs_list.extend([3*n, 3*n+1, 3*n+2])
        
    fixed_dofs = np.array(fixed_dofs_list)
    solver.set_fixed_dofs(fixed_dofs)
    print(f"Fixed {len(fixed_node_indices)} nodes on Left Wall (x=0).")
    
    # 2. Load
    load_x = args.load_x if args.load_x is not None else args.nelx
    load_y = args.load_y if args.load_y is not None else args.nely
    load_z = args.load_z if args.load_z is not None else (args.nelz // 2)
    
    # Find closest node
    dist = (il_flat - load_x)**2 + (jl_flat - load_y)**2 + (kl_flat - load_z)**2
    load_node_idx = np.argmin(dist)
    
    # Apply Load (Vector)
    # DOFs: 3*node, 3*node+1, 3*node+2
    dof_x = 3*load_node_idx
    dof_y = 3*load_node_idx + 1
    dof_z = 3*load_node_idx + 2
    
    if args.load_fx != 0.0: solver.set_load(dof_x, args.load_fx)
    if args.load_fy != 0.0: solver.set_load(dof_y, args.load_fy)
    if args.load_fz != 0.0: solver.set_load(dof_z, args.load_fz)
    
    print(f"Applied Point Load at Node {load_node_idx} (Target: {load_x},{load_y},{load_z}).")
    print(f"Force Vector: [{args.load_fx}, {args.load_fy}, {args.load_fz}]")

    # Change Iterations
    xPhys = solver.optimize(max_loop=args.max_loop) 
    
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
