"""
npz_to_stl.py
Export a Top3D density field as a watertight STL mesh.

Uses marching cubes to extract a smooth isosurface from the density field,
then exports via trimesh. The resulting STL can be imported into SolidWorks
(or any CAD tool) for FEA comparison.

Usage
-----
CLI::

    python -m src.export.npz_to_stl input.npz output.stl --vol_thresh 0.3

Pipeline integration::

    python run_pipeline.py --skip_top3d --top3d_npz result.npz --export_stl ...

Programmatic::

    from src.export.npz_to_stl import export_top3d_stl
    export_top3d_stl("result.npz", "result.stl", vol_thresh=0.3, pitch=1.0)
"""

import numpy as np


def export_top3d_stl(npz_path, stl_path, vol_thresh=0.3, pitch=1.0,
                     smooth_iters=3, gaussian_sigma=0.7):
    """Convert a Top3D NPZ density field to a watertight STL mesh.

    Parameters
    ----------
    npz_path : str
        Path to Top3D ``.npz`` file (must contain ``rho`` array).
    stl_path : str
        Destination STL file path.
    vol_thresh : float
        Density threshold for the isosurface (default 0.3).
    pitch : float
        Voxel size in mm — scales the output mesh to real-world dimensions.
    smooth_iters : int
        Laplacian smoothing iterations on the output mesh (0 to disable).
    gaussian_sigma : float
        Gaussian pre-smoothing of the density field before marching cubes.
        Produces smoother surfaces than raw voxel edges. 0 to disable.

    Returns
    -------
    stl_path : str
        Same as input (for convenient chaining).
    """
    from skimage.measure import marching_cubes
    import trimesh

    # Load density field
    npz = np.load(npz_path, allow_pickle=True)
    rho = npz['rho']  # shape: (nely, nelx, nelz)
    npz_pitch = float(npz['pitch']) if 'pitch' in npz else pitch
    if npz_pitch != pitch and pitch == 1.0:
        pitch = npz_pitch

    print(f"[NPZ→STL] Density field: {rho.shape} "
          f"({int(np.sum(rho > vol_thresh)):,} solid voxels at threshold {vol_thresh})")

    # Pre-smooth the density field for a smoother isosurface
    if gaussian_sigma > 0:
        from scipy.ndimage import gaussian_filter
        rho_smooth = gaussian_filter(rho.astype(np.float64), sigma=gaussian_sigma)
    else:
        rho_smooth = rho.astype(np.float64)

    # Pad with zeros on all sides so marching cubes produces a closed surface
    rho_padded = np.pad(rho_smooth, pad_width=1, mode='constant', constant_values=0.0)

    # Marching cubes — extract isosurface at vol_thresh
    # rho shape is (nely, nelx, nelz); marching cubes treats axis 0,1,2 as spatial
    verts, faces, normals, _ = marching_cubes(
        rho_padded, level=vol_thresh, spacing=(pitch, pitch, pitch)
    )
    # Shift vertices back to account for padding offset
    verts -= pitch

    print(f"[NPZ→STL] Marching cubes: {len(verts):,} vertices, {len(faces):,} faces")

    # Create trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Keep only the largest connected component (removes isolated density islands)
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.faces))
        print(f"[NPZ→STL] Kept largest component ({len(mesh.faces):,} faces) "
              f"of {len(components)} disconnected pieces")

    # Laplacian smoothing for cleaner surface
    if smooth_iters > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters)
        print(f"[NPZ→STL] Applied {smooth_iters} Laplacian smoothing iterations")

    # Ensure consistent winding
    mesh.fix_normals()

    watertight = mesh.is_watertight
    print(f"[NPZ→STL] Watertight: {watertight}, "
          f"Volume: {mesh.volume:.1f} mm³" if watertight else
          f"[NPZ→STL] Watertight: {watertight} (mesh may have small holes)")

    mesh.export(stl_path)
    print(f"[NPZ→STL] Exported: {stl_path} ({len(faces):,} triangles)")

    return stl_path


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Export Top3D density field as watertight STL mesh")
    parser.add_argument("npz_path", help="Input Top3D .npz file")
    parser.add_argument("stl_path", help="Output .stl file")
    parser.add_argument("--vol_thresh", type=float, default=0.3,
                        help="Density threshold (default: 0.3)")
    parser.add_argument("--pitch", type=float, default=1.0,
                        help="Voxel size in mm (default: 1.0)")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Laplacian smoothing iterations (default: 3, 0=off)")
    parser.add_argument("--sigma", type=float, default=0.7,
                        help="Gaussian pre-smoothing sigma (default: 0.7, 0=off)")
    args = parser.parse_args()

    export_top3d_stl(args.npz_path, args.stl_path,
                     vol_thresh=args.vol_thresh, pitch=args.pitch,
                     smooth_iters=args.smooth, gaussian_sigma=args.sigma)


if __name__ == "__main__":
    main()
