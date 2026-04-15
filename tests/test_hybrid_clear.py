#!/usr/bin/env python3
"""
Test Case: Clear Beam-Plate Separation
=======================================

Creates a Top3D structure with obviously distinct beam and plate regions:
- Thin flat roof plate (top face, distributed load)
- Four corner columns (thin beams, point supports at bottom)

This geometry should be easy to classify:
- Roof: Planarity >> Linearity → PLATE
- Columns: Linearity >> Planarity → BEAM
"""

import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from src.optimization.top3d import Top3D


def test_clear_separation():
    """
    Test case with very clear beam vs plate geometry.

    Domain: 40x40x20 (width x depth x height)
    Volume fraction: 0.08 (sparse structure)
    """

    nelx, nely, nelz = 40, 40, 20
    volfrac = 0.08
    penal = 3.0
    rmin = 1.5

    print("="*60)
    print("HYBRID TEST: Clear Beam-Plate Separation")
    print("="*60)
    print(f"Domain: {nelx} x {nely} x {nelz}")
    print(f"Volume fraction: {volfrac}")
    print("="*60)

    # Initialize solver
    solver = Top3D(nelx, nely, nelz, volfrac, penal, rmin)

    # Node grid (F-order)
    jl, il, kl = np.meshgrid(
        np.arange(nely+1), np.arange(nelx+1), np.arange(nelz+1),
        indexing='ij'
    )
    il_flat = il.flatten(order='F')
    jl_flat = jl.flatten(order='F')
    kl_flat = kl.flatten(order='F')

    # ===== SUPPORTS: 4 corner points at bottom =====
    print(f"\n[Supports] Fixed at 4 corners of bottom...")
    corners = [(1, 1, 0), (nelx-1, 1, 0), (1, nely-1, 0), (nelx-1, nely-1, 0)]
    fixed_dofs = []
    for (cx, cy, cz) in corners:
        dist = (il_flat - cx)**2 + (jl_flat - cy)**2 + (kl_flat - cz)**2
        n = np.argmin(dist)
        fixed_dofs.extend([3*n, 3*n+1, 3*n+2])
        print(f"  → Node {n} at ({cx}, {cy}, {cz})")

    solver.set_fixed_dofs(np.array(fixed_dofs))

    # ===== LOADS: Distributed on top surface =====
    print(f"\n[Loads] Distributed load on top surface...")
    top_nodes = np.where(kl_flat == nelz)[0]
    n_top = len(top_nodes)
    load_per_node = -1.0 / n_top  # Total downward load = -1.0

    for n in top_nodes:
        solver.set_load(3*n+2, load_per_node)  # Z-direction

    print(f"  → {n_top} nodes on top face")
    print(f"  → Load per node: {load_per_node:.6f}")

    # ===== OPTIMIZE =====
    print(f"\n[Optimization] Running...")
    xPhys, compliance_history = solver.optimize(max_loop=100)

    # ===== SAVE =====
    output_dir = "output/hybrid_v2"
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "Clear_Beam_Plate_Test_top3d.npz")

    np.savez_compressed(
        npz_path,
        rho=xPhys,
        bc_tags=solver.bc_tags,
        pitch=1.0,
        origin=[0, 0, 0],
        compliance_history=np.array(compliance_history),
        E0=1e3,
    )

    print(f"\n[Output] Saved to: {npz_path}")
    print(f"  Mean density: {np.mean(xPhys):.4f}")

    print(f"\n[Next] Run reconstruction:")
    print(f"  python run_pipeline.py --skip_top3d --top3d_npz {npz_path} --hybrid --output Clear_Test.json --visualize")

    return npz_path


if __name__ == "__main__":
    test_clear_separation()
