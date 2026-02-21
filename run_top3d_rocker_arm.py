"""
Rocker Arm Topology Optimization (after Yin et al.)

Domain orientation:
  - X = 75 elements  (depth: x=0 = back face,  x=75 = front face)
  - Y = 60 elements  (width: y=0 = left side,   y=60 = right side)
  - Z = 12 elements  (height: z=0 = bottom,     z=12 = top surface)

Design space — figure (a):
  A flat rectangular block with a U-shaped through-hole opening at the FRONT face
  (x=75), cut back 30 elements toward the back. The two 'legs' are the left-side
  and right-side strips that remain below (in Y) and above (in Y) the U opening.

Boundary conditions:
  BACK FACE clamped supports (all DOFs fixed):
    Two nodes on the back face (x=0), symmetric in Y, at mid-height Z.
    Default: (x=0, y=15, z=6) and (x=0, y=45, z=6)

  FRONT EDGE leg-tip nodes (Z DOF fixed + loads applied in X and Y):
    Left leg tip:  (x=75, y≈7,  z=6) — Fx=+65 N (outward),  Fy=+100 N (inward, +Y toward void), Uz=0
    Right leg tip: (x=75, y≈53, z=6) — Fx=+65 N (outward),  Fy=−100 N (inward, −Y toward void), Uz=0

Loads:
  TOP SURFACE (z=12), two symmetric nodes at (x≈30, y=15) and (x≈30, y=45):
    Each node:  Fz = +100 N (upward)   →  total 200 N upward
                Fx = − 50 N (toward back, −X)  →  total 100 N

  FRONT EDGE leg tips (as above): 65 N outward (+X) + 100 N inward (±Y) per node.

Passive void (U cutout, through full Z):
  Elements with x_idx ∈ [45, 75), y_idx ∈ [15, 45), all Z.
"""

import numpy as np
import argparse
import os
from src.optimization.top3d import Top3D


def _node_xyz(node_idx, nelx, nely):
    """Decode a flat node index → physical (ix, iy, iz) grid position."""
    slice_sz = (nelx + 1) * (nely + 1)
    iz = node_idx // slice_sz
    rem = node_idx % slice_sz
    ix = rem // (nely + 1)
    iy = rem % (nely + 1)
    return np.array([ix, iy, iz], dtype=float)


def show_bc_setup(solver, nelx, nely, nelz):
    """
    Matplotlib 3D visualization of the BC / load setup.
    Fixed nodes → blue squares.
    Load arrows → coloured by component (X=red, Y=green, Z=blue).
    Axes match physical coordinates: X=depth(0=back), Y=width, Z=height.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    except ImportError:
        print("[Viz] matplotlib not available — skipping BC visualisation.")
        return

    f_flat = solver.f.flatten()
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Rocker Arm — BC Setup (force arrows before optimisation)", pad=14)
    ax.set_xlabel("X  (depth, 0=back)")
    ax.set_ylabel("Y  (width)")
    ax.set_zlabel("Z  (height)")
    ax.set_xlim(0, nelx); ax.set_ylim(0, nely); ax.set_zlim(0, nelz)

    arrow_len = max(nelx, nely, nelz) * 0.12   # scale factor for quiver

    # Fixed nodes — blue scatter
    fixed_nodes = np.unique(solver.fixed_dofs // 3)
    if len(fixed_nodes):
        fp = np.array([_node_xyz(n, nelx, nely) for n in fixed_nodes])
        ax.scatter(fp[:, 0], fp[:, 1], fp[:, 2],
                   color='royalblue', s=120, marker='s', label='Fixed DOF', zorder=6)

    # Loaded nodes — one arrow per non-zero force component
    loaded_dofs = np.where(np.abs(f_flat) > 1e-10)[0]
    loaded_nodes = np.unique(loaded_dofs // 3)
    comp_colors = {0: ('X', 'red'), 1: ('Y', 'limegreen'), 2: ('Z', 'dodgerblue')}
    labeled = set()
    for nid in loaded_nodes:
        pos = _node_xyz(nid, nelx, nely)
        for comp, (axis_name, color) in comp_colors.items():
            val = f_flat[3 * nid + comp]
            if abs(val) < 1e-10:
                continue
            # Unit direction × arrow_len, sign preserved
            d = np.zeros(3); d[comp] = np.sign(val)
            lbl = f"F{axis_name}" if f"F{axis_name}" not in labeled else None
            if lbl:
                labeled.add(lbl)
            ax.quiver(pos[0], pos[1], pos[2],
                      d[0], d[1], d[2],
                      length=arrow_len, color=color, arrow_length_ratio=0.35,
                      label=lbl if lbl else '')
            ax.text(pos[0] + d[0] * arrow_len * 1.1,
                    pos[1] + d[1] * arrow_len * 1.1,
                    pos[2] + d[2] * arrow_len * 1.1,
                    f"{val:+.0f}N", fontsize=7, color=color)

    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    print("[Viz] Showing BC setup — close window to continue.")
    plt.show()


def node_at(nelx, nely, ix, iy, iz):
    """Flat node index for grid position (ix=X, iy=Y, iz=Z)."""
    n_per_slice = (nelx + 1) * (nely + 1)
    return iz * n_per_slice + ix * (nely + 1) + iy


def main():
    parser = argparse.ArgumentParser(
        description="Rocker Arm Topology Optimization (Yin et al.)"
    )

    # Domain
    parser.add_argument("--nelx",     type=int,   default=75,   help="Elements in X / depth (default: 75)")
    parser.add_argument("--nely",     type=int,   default=60,   help="Elements in Y / width (default: 60)")
    parser.add_argument("--nelz",     type=int,   default=12,   help="Elements in Z / height (default: 12)")
    parser.add_argument("--volfrac",  type=float, default=0.085)
    parser.add_argument("--penal",    type=float, default=3.0)
    parser.add_argument("--rmin",     type=float, default=3.0)
    parser.add_argument("--max_loop", type=int,   default=100)

    # U-void (opens at front face x=nelx, extends back void_depth elements)
    parser.add_argument("--void_depth", type=int, default=30,
                        help="U-void depth in X elements from front face (default: 30)")
    parser.add_argument("--void_y0", type=float, default=0.25,
                        help="U-void inner-left Y boundary, fraction of nely (default: 0.25 → y=15)")
    parser.add_argument("--void_y1", type=float, default=0.75,
                        help="U-void inner-right Y boundary, fraction of nely (default: 0.75 → y=45)")

    # Back-face clamped supports (two symmetric nodes on x=0 face)
    parser.add_argument("--rear_y1", type=float, default=0.25,
                        help="Left back-support Y, fraction of nely (default: 0.25 → y=15)")
    parser.add_argument("--rear_y2", type=float, default=0.75,
                        help="Right back-support Y, fraction of nely (default: 0.75 → y=45)")
    parser.add_argument("--rear_z",  type=float, default=0.5,
                        help="Back-support Z, fraction of nelz (default: 0.5 → z=6, mid-height)")

    # Top-surface load nodes (two symmetric nodes on z=nelz face)
    parser.add_argument("--top_x",  type=float, default=0.40,
                        help="Top-load X position, fraction of nelx (default: 0.40 → x=30)")
    parser.add_argument("--top_y1", type=float, default=0.25,
                        help="Left top-load Y, fraction of nely (default: 0.25 → y=15)")
    parser.add_argument("--top_y2", type=float, default=0.75,
                        help="Right top-load Y, fraction of nely (default: 0.75 → y=45)")
    parser.add_argument("--top_fz", type=float, default=200.0,
                        help="Total upward (+Z) force on top nodes, N (default: 200)")
    parser.add_argument("--top_fx", type=float, default=-100.0,
                        help="Total backward (−X) force on top nodes, N (default: −100)")

    # Front-edge leg-tip load nodes
    parser.add_argument("--front_y_left",  type=float, default=0.125,
                        help="Left leg-tip Y, fraction of nely (default: 0.125 → y=7)")
    parser.add_argument("--front_y_right", type=float, default=0.875,
                        help="Right leg-tip Y, fraction of nely (default: 0.875 → y=53)")
    parser.add_argument("--front_z", type=float, default=0.5,
                        help="Leg-tip Z, fraction of nelz (default: 0.5 → z=6, mid-height)")
    parser.add_argument("--front_fx",     type=float, default=65.0,
                        help="Outward (+X) force per leg-tip node, N (default: 65)")
    parser.add_argument("--front_fy_mag", type=float, default=100.0,
                        help="Inward (±Y) force magnitude per leg-tip node, N (default: 100)")

    # Output / visualisation
    parser.add_argument("--output", default="output/hybrid_v2/rocker_arm_top3d.npz")
    parser.add_argument("--no_void", action="store_true", help="Disable the U-void")
    parser.add_argument("--visualize", action="store_true",
                        help="Show matplotlib BC setup (force arrows) before optimisation")

    args = parser.parse_args()
    nelx, nely, nelz = args.nelx, args.nely, args.nelz

    print("=== Rocker Arm Topology Optimization (Yin) ===")
    print(f"Domain: X={nelx} (depth)  Y={nely} (width)  Z={nelz} (height)")
    print(f"volfrac={args.volfrac},  penal={args.penal},  rmin={args.rmin},  max_loop={args.max_loop}")

    solver = Top3D(nelx, nely, nelz, args.volfrac, args.penal, args.rmin)

    # ------------------------------------------------------------------ #
    # 1. Passive void — U-shaped through-hole opening at the front face
    # ------------------------------------------------------------------ #
    if not args.no_void:
        void_x0 = nelx - args.void_depth   # void starts here (element x index)
        vy0 = int(round(args.void_y0 * nely))
        vy1 = int(round(args.void_y1 * nely))

        passive = np.zeros((nely, nelx, nelz), dtype=bool)
        # passive shape: (nely, nelx, nelz) = (Y, X, Z)
        passive[vy0:vy1, void_x0:nelx, :] = True
        solver.set_passive_void(passive)

        void_elems = int(passive.sum())
        print(f"\nU-void: x=[{void_x0}..{nelx}), y=[{vy0}..{vy1}), all Z  →  {void_elems} void elements")
        print(f"  Left leg:  y=[0..{vy0}]  ({vy0} elements wide)")
        print(f"  Right leg: y=[{vy1}..{nely}]  ({nely - vy1} elements wide)")

    # ------------------------------------------------------------------ #
    # 2. Boundary conditions
    # ------------------------------------------------------------------ #
    fixed_dofs = []

    # --- Back-face clamped supports (all DOFs) ---
    by1 = int(round(args.rear_y1 * nely))
    by2 = int(round(args.rear_y2 * nely))
    bz  = int(round(args.rear_z  * nelz))
    rear_nodes = [
        node_at(nelx, nely, 0, by1, bz),
        node_at(nelx, nely, 0, by2, bz),
    ]
    for nid in rear_nodes:
        fixed_dofs.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])
    print(f"\nBack clamped supports (all DOFs):  y=[{by1}, {by2}], z={bz}  (x=0)")

    # --- Front leg-tip nodes: Z DOF fixed ---
    fy_left  = int(round(args.front_y_left  * nely))
    fy_right = int(round(args.front_y_right * nely))
    fz       = int(round(args.front_z       * nelz))
    front_left  = node_at(nelx, nely, nelx, fy_left,  fz)
    front_right = node_at(nelx, nely, nelx, fy_right, fz)
    for nid in [front_left, front_right]:
        fixed_dofs.append(3 * nid + 2)   # Z DOF only
    print(f"Front leg-tip Z-fixed:  left y={fy_left}, right y={fy_right}, z={fz}  (x={nelx})")

    solver.set_fixed_dofs(np.unique(np.array(fixed_dofs, dtype=int)))

    # ------------------------------------------------------------------ #
    # 3. Loads
    # ------------------------------------------------------------------ #

    # --- Top-surface loads: 200 N upward (+Z) and 100 N backward (−X) ---
    tx  = int(round(args.top_x  * nelx))
    ty1 = int(round(args.top_y1 * nely))
    ty2 = int(round(args.top_y2 * nely))
    top_nodes = [
        node_at(nelx, nely, tx, ty1, nelz),
        node_at(nelx, nely, tx, ty2, nelz),
    ]
    n_top = len(top_nodes)
    for nid in top_nodes:
        solver.set_load(3 * nid + 2, args.top_fz / n_top)   # +Z (upward)
        solver.set_load(3 * nid + 0, args.top_fx / n_top)   # −X (toward back)
    print(f"\nTop-surface load:  x={tx}, y=[{ty1},{ty2}], z={nelz}")
    print(f"  Fz = {args.top_fz/n_top:+.1f} N per node (+Z upward),  "
          f"Fx = {args.top_fx/n_top:+.1f} N per node (−X toward back)")

    # --- Front leg-tip loads: 65 N outward (+X) and 100 N inward (±Y) ---
    solver.set_load(3 * front_left  + 0, +args.front_fx)          # +X outward
    solver.set_load(3 * front_left  + 1, +args.front_fy_mag)       # +Y inward (left→center)
    solver.set_load(3 * front_right + 0, +args.front_fx)          # +X outward
    solver.set_load(3 * front_right + 1, -args.front_fy_mag)       # −Y inward (right→center)
    print(f"Front leg-tip loads:")
    print(f"  Left  (y={fy_left}):  Fx=+{args.front_fx:.0f} N (+X outward),  Fy=+{args.front_fy_mag:.0f} N (+Y inward)")
    print(f"  Right (y={fy_right}): Fx=+{args.front_fx:.0f} N (+X outward),  Fy=−{args.front_fy_mag:.0f} N (−Y inward)")

    # ------------------------------------------------------------------ #
    # 4. (Optional) BC visualisation
    # ------------------------------------------------------------------ #
    if args.visualize:
        show_bc_setup(solver, nelx, nely, nelz)

    # ------------------------------------------------------------------ #
    # 5. Run optimisation
    # ------------------------------------------------------------------ #
    print()
    xPhys = solver.optimize(max_loop=args.max_loop)

    # ------------------------------------------------------------------ #
    # 6. Save
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        rho=xPhys,
        bc_tags=solver.bc_tags,
        pitch=1.0,
        origin=[0, 0, 0],
    )

    n_solid = int(np.sum(xPhys > 0.5))
    total = nelx * nely * nelz
    print(f"\nSaved → {args.output}")
    print(f"Solid elements (ρ > 0.5): {n_solid} / {total}  ({100*n_solid/total:.1f}%)")
    print(f"BC Tags: Fixed={int(np.sum(solver.bc_tags == 1))},  Loaded={int(np.sum(solver.bc_tags == 2))}")


if __name__ == "__main__":
    main()
