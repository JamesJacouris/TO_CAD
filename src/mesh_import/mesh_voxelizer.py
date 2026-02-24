"""
mesh_voxelizer.py
Convert an external solid mesh (STL / OBJ / PLY) to a voxel density array
compatible with the TO_CAD reconstruction pipeline.

The output ``rho`` array has the same convention as Top3D NPZ output:
  shape (nely, nelx, nelz) — solid voxels = 1.0, empty = 0.0

Supported input formats (via trimesh):
  STL (.stl), OBJ (.obj), PLY (.ply), and any other format trimesh handles.

Typical usage
-------------
From the CLI (via run_pipeline.py --mesh_input):
    python run_pipeline.py --mesh_input ansys_result.stl --pitch 1.5 --output geo.json

Programmatically:
    from src.import.mesh_voxelizer import save_mesh_as_npz
    save_mesh_as_npz("model.stl", "model_voxelized.npz", pitch=1.0)
"""

import numpy as np


def voxelize_mesh_to_density(mesh_path: str, pitch: float = 1.0) -> np.ndarray:
    """
    Load a mesh file and voxelize it into a binary density array.

    The mesh is voxelized at the given ``pitch`` and its interior is filled
    automatically.  The resulting boolean grid is transposed from trimesh's
    native (nx, ny, nz) ordering to the pipeline convention (nely, nelx, nelz).

    Parameters
    ----------
    mesh_path : str
        Path to STL / OBJ / PLY (or any trimesh-supported format).
    pitch : float
        Voxel edge length in mm.  Smaller values produce denser grids.

    Returns
    -------
    rho : np.ndarray, shape (nely, nelx, nelz), dtype float32
        1.0 for solid voxels, 0.0 for empty.

    Raises
    ------
    ImportError
        If trimesh is not installed.
    ValueError
        If the mesh is empty or cannot be voxelized.
    """
    try:
        import trimesh
        from trimesh.voxel.creation import voxelize
    except ImportError:
        raise ImportError(
            "trimesh is required for mesh input. Install it with:\n"
            "    pip install trimesh"
        )

    print(f"[MeshVox] Loading: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')

    if mesh is None or (hasattr(mesh, 'is_empty') and mesh.is_empty):
        raise ValueError(f"[MeshVox] Could not load a valid mesh from: {mesh_path}")

    print(f"[MeshVox] Bounds:  {mesh.bounds}")
    print(f"[MeshVox] Faces:   {len(mesh.faces):,}")
    print(f"[MeshVox] Pitch:   {pitch} mm")

    # Voxelise and fill interior
    vgrid = voxelize(mesh, pitch)
    vgrid.fill()

    # trimesh matrix convention: (nx, ny, nz)
    # pipeline convention:        (nely, nelx, nelz) = (ny, nx, nz)
    matrix = vgrid.matrix                                      # bool (nx, ny, nz)
    rho = matrix.transpose(1, 0, 2).astype(np.float32)        # → (ny, nx, nz)

    print(f"[MeshVox] Grid (nely, nelx, nelz): {rho.shape}")
    print(f"[MeshVox] Solid voxels: {int(rho.sum()):,} / {rho.size:,}  "
          f"({100.0 * rho.mean():.1f}% fill)")
    return rho


def save_mesh_as_npz(mesh_path: str, npz_path: str, pitch: float = 1.0) -> str:
    """
    Voxelize a mesh and write a minimal NPZ compatible with ``reconstruct_npz()``.

    The saved NPZ contains only ``rho`` — no ``bc_tags`` (those are inferred
    from the ``--load_*`` CLI flags later) and no ``compliance_history``
    (not applicable for external meshes).

    Parameters
    ----------
    mesh_path : str
        Path to input mesh file.
    npz_path  : str
        Destination ``.npz`` file path.
    pitch : float
        Voxel size in mm.

    Returns
    -------
    npz_path : str
        Same as the input parameter (for convenient chaining).
    """
    rho = voxelize_mesh_to_density(mesh_path, pitch)
    np.savez(npz_path, rho=rho)
    print(f"[MeshVox] Saved: {npz_path}")
    return npz_path
