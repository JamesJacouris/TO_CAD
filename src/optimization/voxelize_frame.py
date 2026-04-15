"""Voxelise a beam frame as cylinders onto a hex grid.

Used to evaluate beam structures with the same continuum FEM solver as SIMP
topology optimisation, enabling direct compliance comparison.
"""

import numpy as np


def voxelize_beam_frame(nodes, edges, radii, grid_shape, pitch=1.0, origin=None):
    """Rasterise beam frame as cylinders onto a voxel grid.

    Each beam is treated as a cylinder of the given radius.  Voxel centres
    within that radius of the beam centreline are set to density 1.0.

    Args:
        nodes: (N, 3) world-space node coordinates [X, Y, Z].
        edges: (M, 2) int node index pairs.
        radii: (M,) beam radii.
        grid_shape: (nely, nelx, nelz) — matches Top3D density array shape.
        pitch: voxel size in world units.
        origin: (3,) world coordinates of grid origin.  Defaults to [0, 0, 0].

    Returns:
        rho: (nely, nelx, nelz) float64 array with values 0.0 or 1.0.
    """
    nely, nelx, nelz = grid_shape
    rho = np.zeros(grid_shape, dtype=np.float64)

    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin, dtype=float)

    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)
    radii = np.asarray(radii, dtype=float)

    for idx in range(len(edges)):
        u, v = edges[idx]
        p0 = nodes[u]
        p1 = nodes[v]
        r = float(radii[idx])

        # Minimum radius covers at least the centreline voxels
        r_eff = max(r, pitch * 0.5)

        # Bounding box in world coords, padded by radius + 1 pitch
        pad = r_eff + pitch
        bbox_min = np.minimum(p0, p1) - pad
        bbox_max = np.maximum(p0, p1) + pad

        # Convert to grid indices
        # World X -> nelx_idx, World Y -> nely_idx, World Z -> nelz_idx
        ix_lo = max(0, int(np.floor((bbox_min[0] - origin[0]) / pitch)))
        ix_hi = min(nelx - 1, int(np.ceil((bbox_max[0] - origin[0]) / pitch)))
        iy_lo = max(0, int(np.floor((bbox_min[1] - origin[1]) / pitch)))
        iy_hi = min(nely - 1, int(np.ceil((bbox_max[1] - origin[1]) / pitch)))
        iz_lo = max(0, int(np.floor((bbox_min[2] - origin[2]) / pitch)))
        iz_hi = min(nelz - 1, int(np.ceil((bbox_max[2] - origin[2]) / pitch)))

        if ix_lo > ix_hi or iy_lo > iy_hi or iz_lo > iz_hi:
            continue

        # Meshgrid of candidate voxel centres within bbox
        iy_rng = np.arange(iy_lo, iy_hi + 1)
        ix_rng = np.arange(ix_lo, ix_hi + 1)
        iz_rng = np.arange(iz_lo, iz_hi + 1)
        IY, IX, IZ = np.meshgrid(iy_rng, ix_rng, iz_rng, indexing='ij')

        # World coordinates of voxel centres
        cx = origin[0] + IX * pitch + pitch * 0.5
        cy = origin[1] + IY * pitch + pitch * 0.5
        cz = origin[2] + IZ * pitch + pitch * 0.5

        # Vectorised point-to-line-segment distance
        pts = np.stack([cx, cy, cz], axis=-1).reshape(-1, 3)
        d = p1 - p0
        seg_len_sq = np.dot(d, d)

        if seg_len_sq < 1e-12:
            # Degenerate beam — treat as sphere
            dist = np.linalg.norm(pts - p0, axis=1)
        else:
            t = np.clip(np.dot(pts - p0, d) / seg_len_sq, 0.0, 1.0)
            closest = p0 + t[:, np.newaxis] * d
            dist = np.linalg.norm(pts - closest, axis=1)

        mask = (dist <= r_eff).reshape(IY.shape).astype(np.float64)
        sub = rho[iy_lo:iy_hi + 1, ix_lo:ix_hi + 1, iz_lo:iz_hi + 1]
        np.maximum(sub, mask, out=sub)

    return rho
