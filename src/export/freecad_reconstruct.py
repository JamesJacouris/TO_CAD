"""
FreeCAD Macro: Hybrid Beam-Plate Reconstruction (Stable Version)
Crash-resistant version with proper error handling and batching.
Includes history visualization, pipeline stages, and final geometry.

Curved beam support: if a curve entry contains 'ctrl_pts' (two interior
cubic Bézier control points), the beam is reconstructed as a smooth
swept tube using Part.BezierCurve + makePipeShell instead of the
ball-and-stick polyline approximation used for straight beams.
"""

import FreeCAD
import Part
import json
import math
import sys
import os

try:
    import Points
except ImportError:
    pass

from FreeCAD import Vector

# Cross-version PySide import
try:
    from PySide import QtGui, QtCore
except ImportError:
    try:
        from PySide2 import QtWidgets as QtGui, QtCore
    except ImportError:
        from PySide6 import QtWidgets as QtGui, QtCore


def validate_radius(r, default=0.5):
    """Validate radius, handling NaN and invalid values."""
    try:
        r = float(r)
        if math.isnan(r) or r <= 0:
            return default
        return max(0.1, r)
    except:
        return default

def get_color_from_radius(r, r_min, r_max):
    """
    Map radius to a colormap (Blue -> Green -> Yellow -> Red) for structural heatmap.
    """
    if r_max <= r_min:
        return (0.8, 0.0, 0.0)
    
    t = (float(r) - r_min) / (r_max - r_min)
    t = max(0.0, min(1.0, t))
    
    if t < 0.333:
        local_t = t / 0.333
        return (0.0, local_t, 1.0)
    elif t < 0.666:
        local_t = (t - 0.333) / 0.333
        return (local_t, 1.0, 1.0 - local_t)
    else:
        local_t = (t - 0.666) / 0.334
        return (1.0, 1.0 - local_t, 0.0)


def _sanitize_ctrl_pts_freecad(p0, p1, p2, p3, max_bulge_ratio=0.2):
    """
    Clamp interior Bézier control points (FreeCAD Vectors) to prevent
    the cubic curve from self-intersecting (looping).

    Enforces chord monotonicity: 0 ≤ t1 ≤ t2 ≤ L for the chord-direction
    projections of P1/P2, and limits perpendicular bulge to
    max_bulge_ratio * chord_length.

    Returns (p1_safe, p2_safe) as FreeCAD Vectors.
    """
    chord = p3 - p0
    L = chord.Length
    if L < 1e-10:
        return p1, p2

    d = chord / L  # unit chord direction (FreeCAD Vector, scalar division)

    # Chord-direction projections
    t1 = (p1 - p0).dot(d)
    t2 = (p2 - p0).dot(d)
    # Perpendicular components
    n1 = (p1 - p0) - d * t1
    n2 = (p2 - p0) - d * t2

    # Clamp chord projections to [0, L]
    t1 = max(0.0, min(float(t1), L))
    t2 = max(0.0, min(float(t2), L))

    # Ensure t1 ≤ t2
    if t1 > t2:
        t1, t2 = t2, t1
        n1, n2 = n2, n1

    # Clamp perpendicular bulge
    max_perp = max_bulge_ratio * L
    if n1.Length > max_perp:
        n1 = n1 * (max_perp / n1.Length)
    if n2.Length > max_perp:
        n2 = n2 * (max_perp / n2.Length)

    p1_safe = p0 + d * t1 + n1
    p2_safe = p0 + d * t2 + n2
    return p1_safe, p2_safe


def create_curved_beam_sweep(p0, p1, p2, p3, radius):
    """
    Create a solid swept tube along a cubic Bézier curve.

    Uses Part.BezierCurve with the four control points (p0 endpoint,
    p1/p2 interior, p3 endpoint), then sweeps a circular cross-section
    of the given radius along it using makePipeShell in Frenet mode.

    Control points are sanitized before use to prevent self-intersecting
    sweep paths. Falls back to a straight cylinder along the chord if
    the sweep fails.

    Parameters
    ----------
    p0, p1, p2, p3 : FreeCAD.Vector
    radius          : float

    Returns
    -------
    list of Part.Shape  (single element on success, may be empty on total failure)
    """
    # Sanitize interior control points to prevent looping sweep paths
    p1, p2 = _sanitize_ctrl_pts_freecad(p0, p1, p2, p3)

    try:
        # 1. Build the cubic Bézier path
        bezier = Part.BezierCurve()
        bezier.setPoles([p0, p1, p2, p3])
        path_edge = bezier.toShape()
        path_wire = Part.Wire([path_edge])

        # 2. Profile circle at p0.
        #    Tangent at t=0 of a cubic Bézier: P'(0) = 3*(P1 - P0)
        tan_vec = p1 - p0
        if tan_vec.Length < 1e-8:
            tan_vec = p3 - p0
        if tan_vec.Length < 1e-8:
            raise ValueError("Degenerate Bézier: all control points coincide")
        tan_vec = tan_vec.normalize()

        circle = Part.makeCircle(radius, p0, tan_vec)
        profile_wire = Part.Wire([Part.Edge(circle)])

        # 3. Sweep along path (Frenet=True keeps profile ⊥ to tangent)
        shell = path_wire.makePipeShell([profile_wire], True, True)

        if shell and shell.isValid():
            # Add joint spheres at both endpoints so beams connect cleanly where
            # multiple curved pipes meet at a node (makePipeShell gives open ends).
            shapes = [shell]
            for pt in (p0, p3):
                try:
                    shapes.append(Part.makeSphere(radius, pt))
                except Exception:
                    pass
            return shapes
        else:
            FreeCAD.Console.PrintWarning("  Curved sweep: result invalid, falling back\n")

    except Exception as e:
        FreeCAD.Console.PrintWarning(f"  Curved sweep failed: {str(e)[:60]}, using cylinder\n")

    # Fallback: straight cylinder along the chord p0→p3
    try:
        vec = p3 - p0
        height = vec.Length
        if height > 1e-4:
            cyl = Part.makeCylinder(radius, height)
            z_axis = Vector(0, 0, 1)
            direction = vec.normalize()
            angle = z_axis.getAngle(direction)
            if abs(angle) < 1e-5:
                rot = FreeCAD.Rotation()
            elif abs(angle - math.pi) < 1e-5:
                rot = FreeCAD.Rotation(Vector(1, 0, 0), 180)
            else:
                rot = FreeCAD.Rotation(z_axis, direction)
            cyl.Placement = FreeCAD.Placement(p0, rot)
            shapes = [cyl]
            for pt in (p0, p3):
                try:
                    shapes.append(Part.makeSphere(radius, pt))
                except Exception:
                    pass
            return shapes
    except:
        pass

    return []


def create_rod_geometry_ball_stick(points):
    """
    Create a Ball-and-Stick representation for a chain of points.
    - Spheres at every node (joint).
    - Cones (Tapered Cylinders) between nodes.
    Returns a list of shapes.
    """
    if len(points) < 2:
        return []

    shapes = []

    # Helper for Vector
    def get_vec(idx):
        return Vector(points[idx][0], points[idx][1], points[idx][2])

    num = len(points)

    # 1. Create Spheres at every node
    for i in range(num):
        try:
            pos = get_vec(i)
            r = points[i][3] if len(points[i]) > 3 else 0.5
            r = validate_radius(r, 0.5)

            s = Part.makeSphere(r, pos)
            shapes.append(s)
        except Exception as e:
            if i < 3:  # Only log first few errors
                FreeCAD.Console.PrintWarning(f"  Sphere {i} failed: {str(e)[:40]}\n")
            continue

    # 2. Create Cones between nodes
    for i in range(num - 1):
        try:
            p1 = get_vec(i)
            p2 = get_vec(i+1)
            r1 = points[i][3] if len(points[i]) > 3 else 0.5
            r2 = points[i+1][3] if len(points[i+1]) > 3 else 0.5

            r1 = validate_radius(r1, 0.5)
            r2 = validate_radius(r2, 0.5)

            vec = p2 - p1
            height = vec.Length

            if height > 1e-4:
                solid_shape = None

                # Case A: Cylinder (Radii are equal)
                if abs(r1 - r2) < 1e-4:
                    try:
                        solid_shape = Part.makeCylinder(r1, height)
                    except:
                        pass

                # Case B: Cone (Tapered)
                if not solid_shape:
                    try:
                        solid_shape = Part.makeCone(r1, r2, height)
                    except:
                        # Fallback to average radius cylinder
                        avg_r = (r1 + r2) / 2.0
                        try:
                            solid_shape = Part.makeCylinder(avg_r, height)
                        except:
                            continue

                if solid_shape:
                    # Orientation
                    z_axis = Vector(0,0,1)
                    direction = vec.normalize()
                    angle = z_axis.getAngle(direction)

                    if abs(angle) < 1e-5:
                        rot = FreeCAD.Rotation()
                    elif abs(angle - math.pi) < 1e-5:
                        rot = FreeCAD.Rotation(Vector(1,0,0), 180)
                    else:
                        rot = FreeCAD.Rotation(z_axis, direction)

                    solid_shape.Placement = FreeCAD.Placement(p1, rot)
                    shapes.append(solid_shape)
        except Exception as e:
            if i < 3:
                FreeCAD.Console.PrintWarning(f"  Cone {i} failed: {str(e)[:40]}\n")
            continue

    return shapes


def create_joint_geometry(location, direction, radius):
    """
    Creates a small gusset cone at the beam-plate junction.
    DISABLED per user request (remove all cones).
    """
    return None

    try:
        # Simple Cone Strategy
        # h = 1.05 * r to ensure overlap/penetration for fusion
        h = radius * 1.05
        r_beam = radius
        r_plate = radius * 1.5
        
        cone = Part.makeCone(r_plate, r_beam, h)
        
        z_axis = Vector(0,0,1)
        dir_vec = Vector(direction[0], direction[1], direction[2]).normalize()
        rot = FreeCAD.Rotation(z_axis, dir_vec)
        cone.Placement = FreeCAD.Placement(
            Vector(location[0], location[1], location[2]), rot
        )
        return cone

    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Joint creation failed: {e}\n")
        return None




def create_cuboid_geometry(cuboid_data):
    """
    Creates a Part.Box. Supports OBB (center, rotation, dimensions) and AABB.
    """
    try:
        # 1. Extract Data
        c_type = cuboid_data.get('type', 'obb')
        
        if c_type == 'aabb':
            p_min = Vector(*cuboid_data['p_min'])
            dims = cuboid_data['dimensions']
            return Part.makeBox(dims[0], dims[1], dims[2], p_min)
            
        # OBB Strategy
        center = Vector(*cuboid_data['center'])
        rot_axes = cuboid_data['rotation']
        dims = cuboid_data['dimensions']
        
        dx, dy, dz = dims[0], dims[1], dims[2]
        
        m = FreeCAD.Matrix()
        # Columns define the orientation of the box's local axes (X, Y, Z)
        m.A11, m.A21, m.A31 = rot_axes[0][0], rot_axes[1][0], rot_axes[2][0]
        m.A12, m.A22, m.A32 = rot_axes[0][1], rot_axes[1][1], rot_axes[2][1]
        m.A13, m.A23, m.A33 = rot_axes[0][2], rot_axes[1][2], rot_axes[2][2]
        
        box = Part.makeBox(dx, dy, dz)
        box.translate(Vector(-dx/2, -dy/2, -dz/2))
        
        rot = FreeCAD.Rotation(m)
        box.Placement = FreeCAD.Placement(center, rot)
        
        return box
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Cuboid creation failed: {e}\n")
        return None


def create_voxelized_geometry(voxel_centers, voxel_size):
    """
    Creates a compound of boxes representing individual voxels.
    This is the most robust way to ensure 100% geometric fidelity.
    """
    try:
        if not voxel_centers:
            return None
            
        boxes = []
        for pt in voxel_centers:
            # Create box at grid position
            # Voxel center pt is world coord, we need p_min
            p_min = Vector(*pt) - Vector(voxel_size/2, voxel_size/2, voxel_size/2)
            boxes.append(Part.makeBox(voxel_size, voxel_size, voxel_size, p_min))
            
        if not boxes:
            return None
            
        compound = Part.Compound(boxes)
        return compound
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Voxelized geometry creation failed: {e}\n")
        return None


def create_bspline_surface_from_data(bspline_data, thickness=0.0):
    """
    Yin's Medial Surface Reconstruction:
    Creates a Part.BSplineSurface from a pre-fitted control grid,
    then optionally offsets it to create a manifold solid.

    bspline_data: dict with keys:
        'ctrl_grid': list[list[[x,y,z]]]  — NxM grid of 3D points
        'degree_u': int (default 3)
        'degree_v': int (default 3)
    thickness: float — plate thickness for offset (0 = surface only)
    
    Returns: Part.Shape (solid if offset succeeds, face otherwise) or None
    """
    try:
        grid = bspline_data.get('ctrl_grid', [])
        deg_u = min(bspline_data.get('degree_u', 3), 3)
        deg_v = min(bspline_data.get('degree_v', 3), 3)

        if not grid or len(grid) < 2 or len(grid[0]) < 2:
            return None

        n_rows = len(grid)
        n_cols = len(grid[0])

        # Need at least degree+1 points in each direction
        if n_rows <= deg_u or n_cols <= deg_v:
            deg_u = min(deg_u, n_rows - 1)
            deg_v = min(deg_v, n_cols - 1)

        # Build the point grid as FreeCAD Vectors
        pts = []
        for row in grid:
            row_pts = []
            for p in row:
                row_pts.append(Vector(float(p[0]), float(p[1]), float(p[2])))
            pts.append(row_pts)

        # Create B-Spline surface via interpolation through the grid points
        bs = Part.BSplineSurface()
        bs.interpolate(pts)

        face = bs.toShape()
        
        # Offset to create solid (Yin's thickness step)
        if thickness > 0.01:
            half_t = thickness / 2.0
            try:
                solid = face.makeOffsetShape(half_t, 0.01, fill=True)
                if solid and solid.isValid():
                    return solid
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"    Offset failed ({e}), using face only\n")

        return face

    except Exception as e:
        FreeCAD.Console.PrintWarning(f"B-Spline surface creation failed: {e}\n")
        return None


def create_bspline_from_voxels(voxel_pts, pitch=1.0):
    """
    Fit a B-Spline surface directly from plate skeleton voxel centers.

    Instead of using the pre-computed ctrl_grid (scipy bisplrep → evaluated
    grid → FreeCAD interpolation — two approximation steps), this function
    works from the raw voxel coordinates stored in plate["voxels"]:

      1. PCA to find the plate's principal plane (major/minor/normal axes).
      2. Project voxels onto the u-v plane.
      3. Build a regular grid whose resolution matches the voxel density
         (≈ one grid point per voxel pitch), capped at 10×10 to stay safe.
      4. IDW interpolation to estimate the out-of-plane (normal) offset at
         each grid point — handles slightly non-flat plates.
      5. Back-project grid points to world coordinates and call
         Part.BSplineSurface.interpolate().

    For perfectly flat plates (z = const) the result is exact. For nearly-flat
    plates the IDW gives a smooth, faithful approximation.
    """
    try:
        import numpy as np
        from scipy.spatial import cKDTree

        pts = np.array(voxel_pts, dtype=float)
        if len(pts) < 4:
            return None

        # --- PCA ---
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        u_axis = vh[0]   # major in-plane axis
        v_axis = vh[1]   # minor in-plane axis
        n_axis = vh[2]   # plate normal

        u_coords = centered @ u_axis
        v_coords = centered @ v_axis
        n_coords = centered @ n_axis

        u_min, u_max = u_coords.min(), u_coords.max()
        v_min, v_max = v_coords.min(), v_coords.max()

        if (u_max - u_min) < 1e-3 or (v_max - v_min) < 1e-3:
            return None  # degenerate (line / point)

        # Grid resolution: ~1 point per pitch, capped at [3, 10]
        n_u = max(3, min(10, int(round((u_max - u_min) / pitch)) + 1))
        n_v = max(3, min(10, int(round((v_max - v_min) / pitch)) + 1))

        u_grid_vals = np.linspace(u_min, u_max, n_u)
        v_grid_vals = np.linspace(v_min, v_max, n_v)

        # --- IDW interpolation for out-of-plane offset ---
        tree = cKDTree(np.column_stack([u_coords, v_coords]))

        grid_pts = []
        for u_val in u_grid_vals:
            row = []
            for v_val in v_grid_vals:
                dists, idxs = tree.query([u_val, v_val], k=min(4, len(pts)))
                if np.isscalar(dists):
                    dists, idxs = np.array([dists]), np.array([idxs])
                if dists[0] < 1e-8:
                    n_val = n_coords[idxs[0]]
                else:
                    w = 1.0 / (dists + 1e-8)
                    n_val = np.dot(w, n_coords[idxs]) / w.sum()
                wp = centroid + u_val * u_axis + v_val * v_axis + n_val * n_axis
                row.append(Vector(float(wp[0]), float(wp[1]), float(wp[2])))
            grid_pts.append(row)

        FreeCAD.Console.PrintMessage(
            f"      [Voxel BSpline] {n_u}×{n_v} grid from {len(pts)} voxels\n")

        bs = Part.BSplineSurface()
        bs.interpolate(grid_pts)
        return bs.toShape()

    except Exception as e:
        FreeCAD.Console.PrintWarning(f"    Voxel B-Spline failed: {e}\n")
        return None


def create_extruded_voxel_plate(voxel_pts, pitch=1.0, thickness=2.0, normal=None):
    """
    Create a plate solid as a compound of axis-aligned boxes — one per skeleton
    voxel, each pitch×pitch in-plane and `thickness` deep along the plate normal.

    The normal is snapped to the nearest principal axis (x/y/z) so every box is
    perfectly axis-aligned with no rotation matrix involved.  This gives an exact
    replica of the voxel grid — no approximation, no floating-point drift.
    """
    try:
        import numpy as np

        pts = np.array(voxel_pts, dtype=float)
        if len(pts) < 1:
            return None

        # Snap normal to nearest principal axis (0=x, 1=y, 2=z)
        if normal and len(normal) == 3:
            n = np.array(normal, dtype=float)
            n = n / (np.linalg.norm(n) + 1e-8)
        else:
            centroid = pts.mean(axis=0)
            centered = pts - centroid
            if len(pts) >= 3:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                n = vh[2]
            else:
                n = np.array([0.0, 0.0, 1.0])

        dominant = int(np.argmax(np.abs(n)))  # 0=x, 1=y, 2=z
        ax0, ax1 = [i for i in range(3) if i != dominant]  # in-plane axes

        half_p = pitch / 2.0
        half_t = thickness / 2.0

        u = pts[:, ax0]   # in-plane axis 0 coordinates
        v = pts[:, ax1]   # in-plane axis 1 coordinates
        w = pts[:, dominant]
        w_center = float(w.mean())

        # --- Boundary polygon extrusion (flat plates) ---
        # Traces the exact outer edge of the voxel pixel grid and extrudes once
        # along the normal — produces one seamless solid with no boolean ops.
        is_flat = (float(w.max()) - float(w.min())) < pitch * 2
        if is_flat:
            u0 = float(u.min())
            v0 = float(v.min())
            i_cells = np.round((u - u0) / pitch).astype(int)
            j_cells = np.round((v - v0) / pitch).astype(int)
            cell_set = set(zip(i_cells.tolist(), j_cells.tolist()))

            # Build directed boundary edges (CCW per cell, exterior sides only).
            # Grid corner (gi, gj) → world: ax0 = u0 - half_p + gi*pitch,
            #                               ax1 = v0 - half_p + gj*pitch
            edge_map = {}  # start_corner → end_corner (one outgoing edge per corner)
            for (ci, cj) in cell_set:
                if (ci,   cj-1) not in cell_set: edge_map[(ci,   cj  )] = (ci+1, cj  )  # bottom →
                if (ci+1, cj  ) not in cell_set: edge_map[(ci+1, cj  )] = (ci+1, cj+1)  # right  ↑
                if (ci,   cj+1) not in cell_set: edge_map[(ci+1, cj+1)] = (ci,   cj+1)  # top    ←
                if (ci-1, cj  ) not in cell_set: edge_map[(ci,   cj+1)] = (ci,   cj  )  # left   ↓

            if edge_map:
                # Walk ALL boundary loops (outer boundary + any holes).
                # The outer boundary comes out CCW; hole loops come out CW —
                # exactly what Part.Face([outer, hole, ...]) expects.
                remaining = dict(edge_map)
                all_loops = []
                while remaining:
                    start = next(iter(remaining))
                    loop = [start]
                    cur = start
                    while True:
                        nxt = remaining.pop(cur, None)
                        if nxt is None or nxt == start:
                            break
                        loop.append(nxt)
                        cur = nxt
                    if len(loop) >= 3:
                        all_loops.append(loop)

                if all_loops:
                    def g2w(gi, gj):
                        coords = [0.0, 0.0, 0.0]
                        coords[ax0] = u0 - half_p + gi * pitch
                        coords[ax1] = v0 - half_p + gj * pitch
                        coords[dominant] = w_center - half_t
                        return Vector(*coords)

                    def loop_to_solid(loop_pts, reverse=False):
                        """Extrude a grid-corner loop into a solid."""
                        ordered = list(reversed(loop_pts)) if reverse else list(loop_pts)
                        wpts = [g2w(gi, gj) for (gi, gj) in ordered]
                        wpts.append(wpts[0])  # close
                        wire = Part.makePolygon(wpts)
                        face = Part.Face(wire)
                        exv = [0.0, 0.0, 0.0]
                        exv[dominant] = thickness
                        return face.extrude(Vector(*exv))

                    # Sort: largest loop = outer boundary; smaller loops = holes
                    all_loops.sort(key=lambda L: -len(L))

                    FreeCAD.Console.PrintMessage(
                        f"      [VoxelPlate] {len(pts)} voxels → "
                        f"{len(all_loops)} loop(s) {'(+hole)' if len(all_loops)>1 else ''}, "
                        f"{'xyz'[dominant]}-normal\n")

                    # Outer loop is CCW → extrude directly.
                    # Hole loops are CW → reverse to CCW for Part.Face, extrude,
                    # then cut from the outer solid. Boolean cut is always reliable.
                    result = loop_to_solid(all_loops[0])
                    for hole_loop in all_loops[1:]:
                        hole_solid = loop_to_solid(hole_loop, reverse=True)
                        result = result.cut(hole_solid)
                    return result

        # --- Fallback: single bounding-box solid (non-flat plates) ---
        dims = [0.0, 0.0, 0.0]
        pmin_coords = [0.0, 0.0, 0.0]
        dims[ax0]      = float(u.max() - u.min()) + pitch
        dims[ax1]      = float(v.max() - v.min()) + pitch
        dims[dominant] = thickness
        pmin_coords[ax0]      = float(u.min()) - half_p
        pmin_coords[ax1]      = float(v.min()) - half_p
        pmin_coords[dominant] = w_center - half_t
        box = Part.makeBox(dims[0], dims[1], dims[2], Vector(*pmin_coords))
        FreeCAD.Console.PrintMessage(
            f"      [VoxelPlate] BBox fallback "
            f"{dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f}, "
            f"{'xyz'[dominant]}-normal → 1 solid\n")
        return box

    except Exception as e:
        FreeCAD.Console.PrintWarning(f"    Extruded voxel plate failed: {e}\n")
        return None


def create_plate_shell_batched(verts, tris, batch_size=500):
    """Create shell from triangles with batching to avoid memory issues."""
    if not (verts and tris):
        return None

    try:
        # Process in batches
        all_faces = []
        num_batches = (len(tris) + batch_size - 1) // batch_size

        FreeCAD.Console.PrintMessage(f"    Processing {len(tris)} triangles in {num_batches} batches...\n")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(tris))
            batch_tris = tris[start_idx:end_idx]

            for tri in batch_tris:
                try:
                    if len(tri) < 3:
                        continue

                    idx0, idx1, idx2 = int(tri[0]), int(tri[1]), int(tri[2])
                    if not (0 <= idx0 < len(verts) and 0 <= idx1 < len(verts) and 0 <= idx2 < len(verts)):
                        continue

                    p1 = Vector(float(verts[idx0][0]), float(verts[idx0][1]), float(verts[idx0][2]))
                    p2 = Vector(float(verts[idx1][0]), float(verts[idx1][1]), float(verts[idx1][2]))
                    p3 = Vector(float(verts[idx2][0]), float(verts[idx2][1]), float(verts[idx2][2]))

                    # Skip degenerate triangles
                    if ((p1 - p2).Length < 1e-6 or (p2 - p3).Length < 1e-6 or (p3 - p1).Length < 1e-6):
                        continue

                    wire = Part.makePolygon([p1, p2, p3, p1])
                    face = Part.Face(wire)
                    all_faces.append(face)
                except:
                    continue

            # GUI update between batches
            if batch_idx % 3 == 0:
                try:
                    FreeCAD.Gui.updateGui()
                except:
                    pass

        if not all_faces:
            return None

        FreeCAD.Console.PrintMessage(f"    Created {len(all_faces)} valid faces\n")

        # Build shell
        shell = Part.makeShell(all_faces)
        return shell

    except Exception as e:
        FreeCAD.Console.PrintError(f"    Shell creation failed: {str(e)[:60]}\n")
        return None


def import_hybrid_json(json_path=None):
    # 0. File Selection
    if not json_path:
        filename, _ = QtGui.QFileDialog.getOpenFileName(None, "Select Hybrid Reconstruction JSON", "", "JSON Files (*.json)")
        json_path = filename

    if not json_path:
        FreeCAD.Console.PrintMessage("No file selected. Aborting.\n")
        return

    doc = FreeCAD.activeDocument()
    if not doc:
        doc = FreeCAD.newDocument("Hybrid_Reconstruction")

    FreeCAD.Console.PrintMessage(f"Loading {json_path}...\n")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        FreeCAD.Console.PrintMessage("✓ JSON loaded successfully\n")
    except Exception as e:
        FreeCAD.Console.PrintError(f"Failed to load JSON: {e}\n")
        return

    # ---------------------------------------------------------
    # 0. PROCESS HISTORY (Pipeline Stages & Voxels)
    # ---------------------------------------------------------
    history = data.get("history", [])
    if history:
        FreeCAD.Console.PrintMessage(f"Found {len(history)} history snapshots. Creating visualization groups...\n")
        hist_root = doc.addObject("App::DocumentObjectGroup", "Pipeline_History")

        for idx, snap in enumerate(history):
            if idx % 5 == 0:
                FreeCAD.Console.PrintMessage(f"  Processing history {idx+1}/{len(history)}...\n")
                try:
                    FreeCAD.Gui.updateGui()
                except:
                    pass

            step_name = snap.get("step", f"Step_{idx}")
            snap_type = snap.get("type", "graph")

            params = f"{idx+1}_{step_name}"
            safe_params = params.replace(" ", "_").replace(".", "")

            # --- HANDLE VOXELS (Points -> Boxes) ---
            if snap_type == "voxels":
                pts_data = snap.get("points", [])
                colors_data = snap.get("colors", None)  # Get color array if available

                if pts_data:
                    pitch = data.get("metadata", {}).get("pitch", 1.0)
                    box_size = pitch * 1.0
                    offset = box_size / 2.0

                    # Limit count to avoid crash
                    if len(pts_data) > 30000:
                        FreeCAD.Console.PrintWarning(f"Voxel count {len(pts_data)} high, subsampling...\n")
                        pts_data = pts_data[::2]
                        if colors_data:
                            colors_data = colors_data[::2]

                    try:
                        # Check if we have per-voxel colors
                        has_colors = colors_data is not None and len(colors_data) == len(pts_data)
                        
                        if has_colors:
                            # Create individual objects for each voxel to support different colors
                            FreeCAD.Console.PrintMessage(f"  Creating {len(pts_data)} color-coded voxels...\n")
                            
                            # Group voxels by color to reduce object count
                            color_groups = {}
                            for idx, p in enumerate(pts_data):
                                color = tuple(colors_data[idx])
                                if color not in color_groups:
                                    color_groups[color] = []
                                corner = Vector(p[0]-offset, p[1]-offset, p[2]-offset)
                                box = Part.makeBox(box_size, box_size, box_size, corner)
                                color_groups[color].append(box)
                            
                            # Create one object per color group
                            for color_idx, (color, boxes) in enumerate(color_groups.items()):
                                if boxes:
                                    comp = Part.makeCompound(boxes)
                                    # Use semantic names for known zone colours; density index for gradient colours
                                    if color == (1.0, 0.0, 0.0):
                                        color_label = "Beam_Red"
                                    elif color == (0.0, 1.0, 1.0):
                                        color_label = "Plate_Cyan"
                                    else:
                                        color_label = f"Density_{color_idx:02d}"
                                    obj = doc.addObject("Part::Feature", f"{safe_params}_{color_label}")
                                    obj.Shape = comp
                                    hist_root.addObject(obj)
                                    obj.ViewObject.ShapeColor = (float(color[0]), float(color[1]), float(color[2]))

                            # Density colorbar legend — only for the initial voxel snapshot
                            if "Initial" in step_name:
                                try:
                                    n_bins = 10
                                    vol_thresh_meta = data.get("metadata", {}).get("vol_thresh", 0.3)
                                    legend_pitch = pitch * 2.0
                                    legend_gap = legend_pitch * 0.3
                                    legend_y = -legend_pitch * 4.0
                                    legend_grp = doc.addObject(
                                        "App::DocumentObjectGroup",
                                        f"{safe_params}_DensityLegend"
                                    )
                                    hist_root.addObject(legend_grp)
                                    for bin_i in range(n_bins):
                                        t = bin_i / n_bins
                                        if t <= 0.5:
                                            s = t * 2.0
                                            r_c, g_c, b_c = 1.0, s, 0.0
                                        else:
                                            s = (t - 0.5) * 2.0
                                            r_c, g_c, b_c = 1.0 - s, 1.0, 0.0
                                        x_start = bin_i * (legend_pitch + legend_gap)
                                        corner = Vector(x_start, legend_y, 0.0)
                                        cube = Part.makeBox(legend_pitch, legend_pitch, legend_pitch, corner)
                                        lo_d = vol_thresh_meta + t * (1.0 - vol_thresh_meta)
                                        hi_d = vol_thresh_meta + (t + 1.0 / n_bins) * (1.0 - vol_thresh_meta)
                                        leg_obj = doc.addObject(
                                            "Part::Feature",
                                            f"{safe_params}_Leg_rho{lo_d:.2f}-{hi_d:.2f}"
                                        )
                                        leg_obj.Shape = cube
                                        leg_obj.ViewObject.ShapeColor = (r_c, g_c, b_c)
                                        legend_grp.addObject(leg_obj)
                                except Exception as leg_e:
                                    FreeCAD.Console.PrintWarning(f"  Density legend failed: {leg_e}\n")
                        else:
                            # Original behavior: single compound with default color
                            voxel_boxes = []
                            for p in pts_data:
                                corner = Vector(p[0]-offset, p[1]-offset, p[2]-offset)
                                box = Part.makeBox(box_size, box_size, box_size, corner)
                                voxel_boxes.append(box)

                            if voxel_boxes:
                                comp = Part.makeCompound(voxel_boxes)
                                obj = doc.addObject("Part::Feature", safe_params)
                                obj.Shape = comp
                                hist_root.addObject(obj)

                                obj.ViewObject.ShapeColor = (0.7, 0.7, 0.7)
                                if "Skeleton" in step_name:
                                    obj.ViewObject.ShapeColor = (1.0, 0.0, 0.0)
                                elif "Initial" in step_name:
                                    obj.ViewObject.Transparency = 80

                    except Exception as e:
                        FreeCAD.Console.PrintError(f"Error creating voxel boxes: {e}\n")

            # --- HANDLE GRAPH (Wireframe) ---
            elif snap_type == "graph":
                try:
                    grp = doc.addObject("App::DocumentObjectGroup", safe_params)
                    hist_root.addObject(grp)

                    h_nodes = snap["nodes"]
                    h_edges = snap["edges"]

                    # Draw Nodes (as compound for efficiency)
                    node_shapes = []
                    for n_i, n_pos in enumerate(h_nodes):
                        try:
                            v = Part.Vertex(Vector(n_pos[0], n_pos[1], n_pos[2]))
                            node_shapes.append(v)
                        except:
                            continue

                    if node_shapes:
                        node_comp = Part.makeCompound(node_shapes)
                        node_obj = doc.addObject("Part::Feature", f"{safe_params}_Nodes")
                        node_obj.Shape = node_comp
                        node_obj.ViewObject.PointSize = 4.0
                        node_obj.ViewObject.ShapeColor = (0.5, 0.5, 0.5)
                        grp.addObject(node_obj)

                    # Draw Edges
                    edge_lines = []
                    for e_i, edge in enumerate(h_edges):
                        try:
                            u, v_idx = int(edge[0]), int(edge[1])
                            pts = edge[3] if len(edge) >= 4 else []

                            if pts:
                                # Polyline
                                for k in range(len(pts) + 1):
                                    if k == 0:
                                        p_start = h_nodes[u]
                                        p_end = pts[0]
                                    elif k == len(pts):
                                        p_start = pts[-1]
                                        p_end = h_nodes[v_idx]
                                    else:
                                        p_start = pts[k-1]
                                        p_end = pts[k]

                                    line = Part.makeLine(
                                        Vector(p_start[0], p_start[1], p_start[2]),
                                        Vector(p_end[0], p_end[1], p_end[2])
                                    )
                                    edge_lines.append(line)
                            elif u < len(h_nodes) and v_idx < len(h_nodes):
                                # Straight line
                                p1 = h_nodes[u]
                                p2 = h_nodes[v_idx]
                                line = Part.makeLine(
                                    Vector(p1[0], p1[1], p1[2]),
                                    Vector(p2[0], p2[1], p2[2])
                                )
                                edge_lines.append(line)
                        except:
                            continue

                    if edge_lines:
                        edge_comp = Part.makeCompound(edge_lines)
                        edge_obj = doc.addObject("Part::Feature", f"{safe_params}_Edges")
                        edge_obj.Shape = edge_comp
                        edge_obj.ViewObject.LineWidth = 2.0

                        # Color coding
                        if "Raw" in step_name:
                            edge_obj.ViewObject.ShapeColor = (1.0, 0.0, 0.0)
                        elif "Collapse" in step_name:
                            edge_obj.ViewObject.ShapeColor = (1.0, 0.5, 0.0)
                        elif "Pruned" in step_name:
                            edge_obj.ViewObject.ShapeColor = (0.0, 0.0, 1.0)
                        elif "Simplified" in step_name:
                            edge_obj.ViewObject.ShapeColor = (0.0, 1.0, 1.0)
                        else:
                            edge_obj.ViewObject.ShapeColor = (1.0, 1.0, 1.0)

                        grp.addObject(edge_obj)

                    # --- PLATES IN SNAPSHOT ---
                    h_plates = snap.get("plates", [])
                    for p_idx, plate in enumerate(h_plates):
                        verts = plate.get("vertices", [])
                        tris = plate.get("triangles", [])
                        if verts and tris:
                            shell = create_plate_shell_batched(verts, tris[:500])  # Limit for history
                            if shell:
                                try:
                                    sh_obj = doc.addObject("Part::Feature", f"{safe_params}_Plate_{p_idx}")
                                    sh_obj.Shape = shell
                                    sh_obj.ViewObject.ShapeColor = (0.0, 0.6, 0.9)
                                    sh_obj.ViewObject.Transparency = 40
                                    grp.addObject(sh_obj)
                                except Exception as e:
                                    FreeCAD.Console.PrintWarning(f"  History Plate failed: {e}\n")
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"  History graph {idx} failed: {str(e)[:50]}\n")

    # ---------------------------------------------------------
    # 0b. PROCESS PIPELINE STAGES
    # ---------------------------------------------------------
    stages = data.get("stages", [])
    if stages:
        FreeCAD.Console.PrintMessage(f"Found {len(stages)} pipeline stages. Creating stage folders...\n")
        stage_root = doc.addObject("App::DocumentObjectGroup", "Ref_Stages")

        stage_colors = [
            (0.2, 0.6, 1.0),   # Blue
            (1.0, 0.6, 0.0),   # Orange
            (0.0, 0.8, 0.2),   # Green
            (0.8, 0.0, 0.8),   # Purple
            (1.0, 0.0, 0.0),   # Red
        ]

        for stage_idx, stage in enumerate(stages):
            try:
                stage_name = stage.get("name", f"Stage_{stage_idx+1}")
                folder_name = stage_name.replace(" ", "_").replace(".", "")

                stage_grp = doc.addObject("App::DocumentObjectGroup", folder_name)
                stage_root.addObject(stage_grp)

                FreeCAD.Console.PrintMessage(f"  Building {stage_name}...\n")

                curves = stage.get("curves", [])
                color = stage_colors[stage_idx % len(stage_colors)]

                all_shapes = []
                if curves:
                    for c_idx, curve in enumerate(curves):
                        pts           = curve.get("points", [])
                        ctrl_pts_json = curve.get("ctrl_pts", None)
                        radius        = curve.get("radius", None)

                        if ctrl_pts_json and radius and len(pts) >= 2:
                            # Curved beam in stage view
                            pv0 = Vector(*pts[0][:3])
                            pv3 = Vector(*pts[-1][:3])
                            pv1 = Vector(*ctrl_pts_json[0])
                            pv2 = Vector(*ctrl_pts_json[1])
                            r = validate_radius(radius)
                            shapes = create_curved_beam_sweep(pv0, pv1, pv2, pv3, r)
                            if not shapes:
                                shapes = create_rod_geometry_ball_stick(pts)
                        elif len(pts) >= 2:
                            shapes = create_rod_geometry_ball_stick(pts)
                        else:
                            shapes = []

                        all_shapes.extend(shapes)

                        if c_idx % 10 == 0:
                            try:
                                FreeCAD.Gui.updateGui()
                            except:
                                pass

                    if all_shapes:
                        compound = Part.makeCompound(all_shapes)
                        obj = doc.addObject("Part::Feature", f"Rods_{folder_name}")
                        obj.Shape = compound
                        obj.ViewObject.ShapeColor = color
                        stage_grp.addObject(obj)

                # --- PLATES IN STAGE ---
                stage_plates = stage.get("plates", [])
                for p_idx, plate in enumerate(stage_plates):
                    verts = plate.get("vertices", [])
                    tris = plate.get("triangles", [])
                    if verts and tris:
                        shell = create_plate_shell_batched(verts, tris)
                        if shell:
                            try:
                                sh_obj = doc.addObject("Part::Feature", f"Plate_{folder_name}_{p_idx}")
                                sh_obj.Shape = shell
                                sh_obj.ViewObject.ShapeColor = color
                                sh_obj.ViewObject.Transparency = 50
                                stage_grp.addObject(sh_obj)
                            except Exception as e:
                                FreeCAD.Console.PrintWarning(f"  Stage Plate failed: {e}\n")
            except Exception as e:
                FreeCAD.Console.PrintError(f"Stage {stage_idx} failed: {str(e)[:50]}\n")

    # ---------------------------------------------------------
    # 1. PROCESS MAIN GEOMETRY (Final Result)
    # ---------------------------------------------------------

    skel_grp = doc.addObject("App::DocumentObjectGroup", "Ref_Skeleton_Final")
    geo_grp = doc.addObject("App::DocumentObjectGroup", "Beams_CSG_Final")

    # --- BEAMS ---
    curves = data.get("curves", [])
    beam_shape_registry = []  # [(shape, start_pos, end_pos, doc_obj_name), ...]
    if curves:
        total_curves = len(curves)
        curved_count = sum(1 for c in curves if 'ctrl_pts' in c)
        
        # Calculate global radius bounds for heatmap coloring
        r_list = []
        for c in curves:
            rad = c.get("radius")
            if rad is not None:
                r_list.append(validate_radius(rad))
        if r_list:
            global_r_min, global_r_max = min(r_list), max(r_list)
        else:
            global_r_min, global_r_max = 0.5, 0.5
            
        FreeCAD.Console.PrintMessage(
            f"Reconstructing {total_curves} beams "
            f"({curved_count} curved Bézier, {total_curves - curved_count} straight)...\n"
            f"Radii Range for Heatmap: [{global_r_min:.3f}, {global_r_max:.3f}] mm\n"
        )

        for i, curve in enumerate(curves):
            if i % 10 == 0:
                FreeCAD.Console.PrintMessage(f"  Processing beam {i+1}/{total_curves}...\n")
                try:
                    FreeCAD.Gui.updateGui()
                except:
                    pass

            pts = curve.get("points", [])
            if not pts:
                continue

            # 1. VISUALIZE SKELETON
            try:
                if len(pts) > 0:
                    # Start Node
                    v_start = doc.addObject("Part::Vertex", f"Ref_Node_S_{i}")
                    v_start.X, v_start.Y, v_start.Z = pts[0][0], pts[0][1], pts[0][2]
                    v_start.ViewObject.PointSize = 5.0
                    v_start.ViewObject.ShapeColor = (0.0, 1.0, 0.0)
                    skel_grp.addObject(v_start)

                    # End Node
                    v_end = doc.addObject("Part::Vertex", f"Ref_Node_E_{i}")
                    v_end.X, v_end.Y, v_end.Z = pts[-1][0], pts[-1][1], pts[-1][2]
                    v_end.ViewObject.PointSize = 5.0
                    v_end.ViewObject.ShapeColor = (0.0, 1.0, 0.0)
                    skel_grp.addObject(v_end)

                # Skeleton Lines
                for j in range(len(pts)-1):
                    p1 = Vector(pts[j][0], pts[j][1], pts[j][2])
                    p2 = Vector(pts[j+1][0], pts[j+1][1], pts[j+1][2])

                    l_obj = doc.addObject("Part::Line", f"Ref_L_{i}_{j}")
                    l_obj.X1, l_obj.Y1, l_obj.Z1 = p1.x, p1.y, p1.z
                    l_obj.X2, l_obj.Y2, l_obj.Z2 = p2.x, p2.y, p2.z
                    l_obj.ViewObject.LineWidth = 2.0
                    l_obj.ViewObject.ShapeColor = (1.0, 1.0, 0.0)
                    skel_grp.addObject(l_obj)
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"  Skeleton {i} failed: {str(e)[:40]}\n")

            # 2. BUILD GEOMETRY (CSG Solids) + Track for plate fusion
            try:
                ctrl_pts_json = curve.get("ctrl_pts", None)
                radius_json   = curve.get("radius", None)
                start_pos = pts[0][:3]
                end_pos   = pts[-1][:3]

                if ctrl_pts_json and radius_json:
                    # Curved beam: use smooth Bézier sweep
                    pv0 = Vector(*pts[0][:3])
                    pv3 = Vector(*pts[-1][:3])
                    pv1 = Vector(*ctrl_pts_json[0])
                    pv2 = Vector(*ctrl_pts_json[1])
                    r   = validate_radius(radius_json)
                    shapes = create_curved_beam_sweep(pv0, pv1, pv2, pv3, r)
                    if not shapes:
                        # Fallback to polyline ball-stick if sweep fails
                        shapes = create_rod_geometry_ball_stick(pts)
                else:
                    # Straight beam: original ball-and-stick
                    shapes = create_rod_geometry_ball_stick(pts)

                for j, shape in enumerate(shapes):
                    obj_name = f"Beam_{i}_P_{j}"
                    obj = doc.addObject("Part::Feature", obj_name)
                    obj.Shape = shape
                    
                    if getattr(obj, "ViewObject", None) is not None:
                        beam_r = validate_radius(radius_json) if radius_json else (global_r_max if 'global_r_max' in locals() else 1.0)
                        if 'global_r_min' in locals() and 'global_r_max' in locals():
                            obj.ViewObject.ShapeColor = get_color_from_radius(beam_r, global_r_min, global_r_max)
                        else:
                            obj.ViewObject.ShapeColor = (0.8, 0.0, 0.0)
                        try:
                            obj.ViewObject.ShapeMaterial = {"SpecularColor": (0.8, 0.8, 0.8), "Shininess": 80.0, "AmbientColor": (0.2, 0.2, 0.2)}
                        except: pass
                        
                    geo_grp.addObject(obj)
                    beam_shape_registry.append((shape.copy(), start_pos, end_pos, obj_name))
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"  Beam geometry {i} failed: {str(e)[:40]}\n")

    # --- PLATES + JOINTS (Fused Assembly) ---
    plates_list = data.get("plates", [])
    plate_shapes = {}  # plate_id -> Part.Shape, for later fusion with joints
    if plates_list:
        plate_grp = doc.addObject("App::DocumentObjectGroup", "Plates_CSG_Final")
        FreeCAD.Console.PrintMessage(f"Reconstructing {len(plates_list)} Plate Regions...\n")

        for p_idx, plate in enumerate(plates_list):
            try:
                FreeCAD.Console.PrintMessage(f"  Plate {p_idx+1}/{len(plates_list)}...\n")
                try:
                    FreeCAD.Gui.updateGui()
                except:
                    pass

                plate_id = plate.get("id", p_idx)
                thickness = plate.get("thickness", 2.0)
                verts = plate.get("vertices", [])
                tris = plate.get("triangles", [])

                if not (tris and verts):
                    FreeCAD.Console.PrintWarning(f"  Plate {plate_id}: No geometry\n")
                    continue

                # === Plate Reconstruction Method ===
                # Read mode from metadata (default: bspline)
                metadata = data.get("metadata", {})
                plate_mode = metadata.get("plate_mode", "bspline")
                
                solid = None
                
                # Helper functions for attempts
                def try_extruded_voxels():
                    voxel_pts = plate.get("voxels", [])
                    if not voxel_pts:
                        return None
                    pitch_val = plate.get("voxel_size",
                                         data.get("metadata", {}).get("pitch", 1.0))
                    normal = plate.get("normal", None)
                    s = create_extruded_voxel_plate(
                        voxel_pts, pitch=pitch_val,
                        thickness=thickness, normal=normal)
                    if s:
                        FreeCAD.Console.PrintMessage(
                            f"    ✓ Extruded voxel plate created "
                            f"({len(voxel_pts)} voxels)\n")
                        return s
                    return None

                def try_voxel_bspline():
                    voxel_pts = plate.get("voxels", [])
                    if not voxel_pts:
                        return None
                    pitch_val = plate.get("voxel_size", data.get("metadata", {}).get("pitch", 1.0))
                    s = create_bspline_from_voxels(voxel_pts, pitch=pitch_val)
                    if s:
                        FreeCAD.Console.PrintMessage(f"    ✓ Voxel B-Spline Surface created\n")
                        return s
                    return None

                def try_bspline():
                    bspline_data = plate.get("bspline_surface", None)
                    if bspline_data:
                        try:
                            s = create_bspline_surface_from_data(bspline_data, thickness)
                            if s:
                                FreeCAD.Console.PrintMessage(f"    ✓ B-Spline Surface created (Yin method)\n")
                                return s
                        except Exception as e:
                            FreeCAD.Console.PrintWarning(f"    B-Spline failed: {e}\n")
                    elif plate_mode == "bspline":
                         FreeCAD.Console.PrintWarning(f"    [Warning] Plate {plate_id}: B-Spline mode active but no surface data found in JSON. Falling back to next method.\n")
                    return None
                    
                def try_voxels():
                    voxel_data = plate.get("voxels", None)
                    if voxel_data:
                        voxel_size = plate.get("voxel_size", 1.0)
                        try:
                            s = create_voxelized_geometry(voxel_data, voxel_size)
                            if s:
                                FreeCAD.Console.PrintMessage(f"    ✓ Voxel Plate created ({len(voxel_data)} voxels)\n")
                                return s
                        except Exception as e:
                            FreeCAD.Console.PrintWarning(f"    Voxel reconstruction failed: {e}\n")
                    return None
                    
                def try_mesh():
                    if solid is None:
                        s = create_plate_shell_batched(verts, tris, batch_size=500)
                        if s:
                            try:
                                if s.isClosed():
                                    s_solid = Part.Solid(s)
                                    FreeCAD.Console.PrintMessage(f"    ✓ Mesh Solid created\n")
                                    return s_solid
                            except Exception as e:
                                FreeCAD.Console.PrintMessage(f"    Could not make mesh solid: {str(e)[:40]}\n")
                            return s # Return shell/face if solid fails
                    return None
                    
                def try_cuboid():
                     cuboid_data = plate.get("cuboid", None)
                     if cuboid_data:
                         try:
                             s = create_cuboid_geometry(cuboid_data)
                             if s:
                                 FreeCAD.Console.PrintMessage(f"    ✓ Cuboid Plate created\n")
                                 return s
                         except Exception as e:
                             FreeCAD.Console.PrintWarning(f"    Cuboid failed: {e}\n")
                     return None

                # Priority: extruded voxel boxes first (geometrically exact),
                # then fallbacks in order of fidelity.
                steps = [try_extruded_voxels, try_voxel_bspline,
                         try_bspline, try_mesh, try_cuboid]
                    
                for step in steps:
                    solid = step()
                    if solid: break


                if solid:
                    # Store plate solid for later fusion with joints
                    plate_shapes[plate_id] = solid
                    FreeCAD.Console.PrintMessage(f"    ✓ Plate {plate_id} geometry ready\n")
                else:
                    FreeCAD.Console.PrintWarning(f"  Plate {plate_id}: Geometry creation failed\n")

            except Exception as e:
                FreeCAD.Console.PrintError(f"  Plate {p_idx} error: {str(e)[:50]}\n")
                import traceback
                FreeCAD.Console.PrintError(f"  {traceback.format_exc()[:200]}\n")

    # ---------------------------------------------------------
    # 3. FUSE BEAMS + JOINTS + PLATES (Boolean Union + Fillet)
    # ---------------------------------------------------------
    joints_list = data.get("joints", [])
    
    # Group joint shapes AND locations by plate_id
    joint_shapes_by_plate = {}
    joint_locs_by_plate = {}
    joint_radii_by_plate = {}
    if joints_list:
        FreeCAD.Console.PrintMessage(f"Building {len(joints_list)} joint connectors...\n")
        for j_idx, joint in enumerate(joints_list):
            try:
                loc = joint.get("location", [0,0,0])
                direction = joint.get("direction", [0,0,1])
                radius = joint.get("radius", 0.5)
                plate_id = joint.get("plate_id", -1)
                
                if len(loc) == 3 and len(direction) == 3:
                    shape = create_joint_geometry(loc, direction, radius)
                    
                    # Always init lists for this plate
                    if plate_id not in joint_shapes_by_plate:
                        joint_shapes_by_plate[plate_id] = []
                        joint_locs_by_plate[plate_id] = []
                        joint_radii_by_plate[plate_id] = []
                    
                    if shape:
                        joint_shapes_by_plate[plate_id].append(shape)
                    
                    # Store location/radius regardless of shape (needed for beam matching)
                    joint_locs_by_plate[plate_id].append(loc)
                    joint_radii_by_plate[plate_id].append(radius)
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"  Joint {j_idx} failed: {e}\n")

    # Add each plate directly — standalone, not fused with beams
    if plate_shapes:
        if 'plate_grp' not in dir():
            plate_grp = doc.addObject("App::DocumentObjectGroup", "Plates_CSG_Final")
        FreeCAD.Console.PrintMessage(f"Adding {len(plate_shapes)} plates to document...\n")
    for plate_id, plate_solid in plate_shapes.items():
        try:
            obj = doc.addObject("Part::Feature", f"Plate_{plate_id}")
            obj.Shape = plate_solid
            
            if getattr(obj, "ViewObject", None) is not None:
                obj.ViewObject.ShapeColor = (0.8, 0.8, 0.95)
                obj.ViewObject.Transparency = 30
                try:
                    obj.ViewObject.ShapeMaterial = {"SpecularColor": (1.0, 1.0, 1.0), "Shininess": 100.0, "AmbientColor": (0.3, 0.3, 0.4)}
                except: pass
                
            plate_grp.addObject(obj)
            FreeCAD.Console.PrintMessage(f"    ✓ Plate {plate_id} added to document\n")
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"  Plate {plate_id} final add failed: {e}\n")

    # Finalize
    FreeCAD.Console.PrintMessage("Finalizing...\n")
    try:
        doc.recompute()
        FreeCAD.Console.PrintMessage("✓ Recompute successful\n")
    except Exception as e:
        FreeCAD.Console.PrintError(f"Recompute failed: {e}\n")

    try:
        if FreeCAD.GuiUp:
            view = FreeCAD.Gui.ActiveDocument.ActiveView
            view.fitAll()
            view.setDrawStyle("Shaded")
            if hasattr(view, "setAxisCross"):
                view.setAxisCross(False)
            try:
                view.setBackgroundColor(
                    FreeCAD.Gui.getMainWindow().palette().color(QtGui.QPalette.Window)
                )
            except: pass
    except: pass

    FreeCAD.Console.PrintMessage("✓✓✓ Reconstruction Complete ✓✓✓\n")


if __name__ == '__main__':
    import_hybrid_json()
