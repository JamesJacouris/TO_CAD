"""
FreeCAD Macro: Hybrid Beam-Plate Reconstruction
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_radius(r, default=0.5):
    """Validate radius, handling NaN and invalid values."""
    try:
        r = float(r)
        if math.isnan(r) or r <= 0:
            return default
        return max(0.1, r)
    except:
        return default


# ---------------------------------------------------------------------------
# Straight beam geometry (ball-and-stick polyline)
# ---------------------------------------------------------------------------

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

    def get_vec(idx):
        return Vector(points[idx][0], points[idx][1], points[idx][2])

    num = len(points)

    # 1. Spheres at every node
    for i in range(num):
        try:
            pos = get_vec(i)
            r = points[i][3] if len(points[i]) > 3 else 0.5
            r = validate_radius(r, 0.5)
            s = Part.makeSphere(r, pos)
            shapes.append(s)
        except Exception as e:
            if i < 3:
                FreeCAD.Console.PrintWarning(f"  Sphere {i} failed: {str(e)[:40]}\n")
            continue

    # 2. Cones/cylinders between nodes
    for i in range(num - 1):
        try:
            p1 = get_vec(i)
            p2 = get_vec(i + 1)
            r1 = points[i][3] if len(points[i]) > 3 else 0.5
            r2 = points[i + 1][3] if len(points[i + 1]) > 3 else 0.5
            r1 = validate_radius(r1, 0.5)
            r2 = validate_radius(r2, 0.5)

            vec = p2 - p1
            height = vec.Length
            if height < 1e-4:
                continue

            solid_shape = None
            if abs(r1 - r2) < 1e-4:
                try:
                    solid_shape = Part.makeCylinder(r1, height)
                except:
                    pass
            if not solid_shape:
                try:
                    solid_shape = Part.makeCone(r1, r2, height)
                except:
                    avg_r = (r1 + r2) / 2.0
                    try:
                        solid_shape = Part.makeCylinder(avg_r, height)
                    except:
                        continue

            if solid_shape:
                z_axis = Vector(0, 0, 1)
                direction = vec.normalize()
                angle = z_axis.getAngle(direction)
                if abs(angle) < 1e-5:
                    rot = FreeCAD.Rotation()
                elif abs(angle - math.pi) < 1e-5:
                    rot = FreeCAD.Rotation(Vector(1, 0, 0), 180)
                else:
                    rot = FreeCAD.Rotation(z_axis, direction)
                solid_shape.Placement = FreeCAD.Placement(p1, rot)
                shapes.append(solid_shape)
        except Exception as e:
            if i < 3:
                FreeCAD.Console.PrintWarning(f"  Cone {i} failed: {str(e)[:40]}\n")
            continue

    return shapes


# ---------------------------------------------------------------------------
# Curved beam geometry (cubic Bézier sweep)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Plate helpers (unchanged from stable version)
# ---------------------------------------------------------------------------

def create_joint_geometry(location, direction, radius):
    """Gusset cone at beam-plate junction. Currently disabled."""
    return None


def create_cuboid_geometry(cuboid_data):
    """Creates a Part.Box from OBB or AABB data."""
    try:
        c_type = cuboid_data.get('type', 'obb')
        if c_type == 'aabb':
            p_min = Vector(*cuboid_data['p_min'])
            dims = cuboid_data['dimensions']
            return Part.makeBox(dims[0], dims[1], dims[2], p_min)

        center = Vector(*cuboid_data['center'])
        rot_axes = cuboid_data['rotation']
        dims = cuboid_data['dimensions']
        dx, dy, dz = dims[0], dims[1], dims[2]

        m = FreeCAD.Matrix()
        m.A11, m.A21, m.A31 = rot_axes[0][0], rot_axes[1][0], rot_axes[2][0]
        m.A12, m.A22, m.A32 = rot_axes[0][1], rot_axes[1][1], rot_axes[2][1]
        m.A13, m.A23, m.A33 = rot_axes[0][2], rot_axes[1][2], rot_axes[2][2]

        box = Part.makeBox(dx, dy, dz)
        box.translate(Vector(-dx / 2, -dy / 2, -dz / 2))
        rot = FreeCAD.Rotation(m)
        box.Placement = FreeCAD.Placement(center, rot)
        return box
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Cuboid creation failed: {e}\n")
        return None


def create_voxelized_geometry(voxel_centers, voxel_size):
    """Creates a compound of boxes representing individual voxels."""
    try:
        if not voxel_centers:
            return None
        boxes = []
        for pt in voxel_centers:
            p_min = Vector(*pt) - Vector(voxel_size / 2, voxel_size / 2, voxel_size / 2)
            boxes.append(Part.makeBox(voxel_size, voxel_size, voxel_size, p_min))
        if not boxes:
            return None
        return Part.Compound(boxes)
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Voxelized geometry creation failed: {e}\n")
        return None


def create_bspline_surface_from_data(bspline_data, thickness=0.0):
    """
    Creates a Part.BSplineSurface from a pre-fitted control grid,
    then optionally offsets it to create a manifold solid.
    """
    try:
        grid = bspline_data.get('ctrl_grid', [])
        deg_u = min(bspline_data.get('degree_u', 3), 3)
        deg_v = min(bspline_data.get('degree_v', 3), 3)

        if not grid or len(grid) < 2 or len(grid[0]) < 2:
            return None

        n_rows = len(grid)
        n_cols = len(grid[0])
        if n_rows <= deg_u or n_cols <= deg_v:
            deg_u = min(deg_u, n_rows - 1)
            deg_v = min(deg_v, n_cols - 1)

        pts = []
        for row in grid:
            row_pts = [Vector(float(p[0]), float(p[1]), float(p[2])) for p in row]
            pts.append(row_pts)

        bs = Part.BSplineSurface()
        bs.interpolate(pts)
        face = bs.toShape()

        if thickness > 0.01:
            try:
                solid = face.makeOffsetShape(thickness / 2.0, 0.01, fill=True)
                if solid and solid.isValid():
                    return solid
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"    Offset failed ({e}), using face only\n")
        return face
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"B-Spline surface creation failed: {e}\n")
        return None


def create_plate_shell_batched(verts, tris, batch_size=500):
    """Create shell from triangles with batching to avoid memory issues."""
    if not (verts and tris):
        return None
    try:
        all_faces = []
        num_batches = (len(tris) + batch_size - 1) // batch_size
        FreeCAD.Console.PrintMessage(f"    Processing {len(tris)} triangles in {num_batches} batches...\n")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            batch_tris = tris[start_idx: start_idx + batch_size]

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
                    if ((p1 - p2).Length < 1e-6 or (p2 - p3).Length < 1e-6 or (p3 - p1).Length < 1e-6):
                        continue
                    wire = Part.makePolygon([p1, p2, p3, p1])
                    face = Part.Face(wire)
                    all_faces.append(face)
                except:
                    continue

            if batch_idx % 3 == 0:
                try:
                    FreeCAD.Gui.updateGui()
                except:
                    pass

        if not all_faces:
            return None
        FreeCAD.Console.PrintMessage(f"    Created {len(all_faces)} valid faces\n")
        return Part.makeShell(all_faces)
    except Exception as e:
        FreeCAD.Console.PrintError(f"    Shell creation failed: {str(e)[:60]}\n")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def import_hybrid_json(json_path=None):
    # 0. File Selection
    if not json_path:
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            None, "Select Hybrid Reconstruction JSON", "", "JSON Files (*.json)")
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

    # -------------------------------------------------------------------------
    # 0. HISTORY (Pipeline Stages & Voxels)
    # -------------------------------------------------------------------------
    history = data.get("history", [])
    if history:
        FreeCAD.Console.PrintMessage(f"Found {len(history)} history snapshots.\n")
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
            safe_params = f"{idx+1}_{step_name}".replace(" ", "_").replace(".", "")

            if snap_type == "voxels":
                pts_data = snap.get("points", [])
                colors_data = snap.get("colors", None)
                if pts_data:
                    pitch = data.get("metadata", {}).get("pitch", 1.0)
                    box_size = pitch * 1.0
                    offset = box_size / 2.0
                    if len(pts_data) > 30000:
                        FreeCAD.Console.PrintWarning(f"Voxel count {len(pts_data)} high, subsampling...\n")
                        pts_data = pts_data[::2]
                        if colors_data:
                            colors_data = colors_data[::2]
                    try:
                        has_colors = colors_data is not None and len(colors_data) == len(pts_data)
                        if has_colors:
                            color_groups = {}
                            for vi, p in enumerate(pts_data):
                                color = tuple(colors_data[vi])
                                if color not in color_groups:
                                    color_groups[color] = []
                                corner = Vector(p[0] - offset, p[1] - offset, p[2] - offset)
                                color_groups[color].append(Part.makeBox(box_size, box_size, box_size, corner))
                            for color, boxes in color_groups.items():
                                if boxes:
                                    comp = Part.makeCompound(boxes)
                                    color_name = "Red" if color == (1.0, 0.0, 0.0) else "Cyan" if color == (0.0, 1.0, 1.0) else "Other"
                                    obj = doc.addObject("Part::Feature", f"{safe_params}_{color_name}")
                                    obj.Shape = comp
                                    hist_root.addObject(obj)
                                    obj.ViewObject.ShapeColor = color
                        else:
                            voxel_boxes = []
                            for p in pts_data:
                                corner = Vector(p[0] - offset, p[1] - offset, p[2] - offset)
                                voxel_boxes.append(Part.makeBox(box_size, box_size, box_size, corner))
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

            elif snap_type == "graph":
                try:
                    grp = doc.addObject("App::DocumentObjectGroup", safe_params)
                    hist_root.addObject(grp)
                    h_nodes = snap["nodes"]
                    h_edges = snap["edges"]

                    node_shapes = []
                    for n_pos in h_nodes:
                        try:
                            node_shapes.append(Part.Vertex(Vector(n_pos[0], n_pos[1], n_pos[2])))
                        except:
                            continue
                    if node_shapes:
                        node_comp = Part.makeCompound(node_shapes)
                        node_obj = doc.addObject("Part::Feature", f"{safe_params}_Nodes")
                        node_obj.Shape = node_comp
                        node_obj.ViewObject.PointSize = 4.0
                        node_obj.ViewObject.ShapeColor = (0.5, 0.5, 0.5)
                        grp.addObject(node_obj)

                    edge_lines = []
                    for edge in h_edges:
                        try:
                            u, v_idx = int(edge[0]), int(edge[1])
                            pts = edge[3] if len(edge) >= 4 else []
                            if pts:
                                for k in range(len(pts) + 1):
                                    p_start = h_nodes[u] if k == 0 else pts[k - 1]
                                    p_end = h_nodes[v_idx] if k == len(pts) else pts[k]
                                    edge_lines.append(Part.makeLine(
                                        Vector(p_start[0], p_start[1], p_start[2]),
                                        Vector(p_end[0], p_end[1], p_end[2])
                                    ))
                            elif u < len(h_nodes) and v_idx < len(h_nodes):
                                p1, p2 = h_nodes[u], h_nodes[v_idx]
                                edge_lines.append(Part.makeLine(
                                    Vector(p1[0], p1[1], p1[2]),
                                    Vector(p2[0], p2[1], p2[2])
                                ))
                        except:
                            continue

                    if edge_lines:
                        edge_comp = Part.makeCompound(edge_lines)
                        edge_obj = doc.addObject("Part::Feature", f"{safe_params}_Edges")
                        edge_obj.Shape = edge_comp
                        edge_obj.ViewObject.LineWidth = 2.0
                        color_map = {"Raw": (1.0, 0.0, 0.0), "Collapse": (1.0, 0.5, 0.0),
                                     "Pruned": (0.0, 0.0, 1.0), "Simplified": (0.0, 1.0, 1.0)}
                        edge_obj.ViewObject.ShapeColor = next(
                            (c for k, c in color_map.items() if k in step_name), (1.0, 1.0, 1.0))
                        grp.addObject(edge_obj)

                    for p_idx, plate in enumerate(snap.get("plates", [])):
                        verts = plate.get("vertices", [])
                        tris = plate.get("triangles", [])
                        if verts and tris:
                            shell = create_plate_shell_batched(verts, tris[:500])
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

    # -------------------------------------------------------------------------
    # 0b. PIPELINE STAGES
    # -------------------------------------------------------------------------
    stages = data.get("stages", [])
    if stages:
        FreeCAD.Console.PrintMessage(f"Found {len(stages)} pipeline stages.\n")
        stage_root = doc.addObject("App::DocumentObjectGroup", "Ref_Stages")
        stage_colors = [
            (0.2, 0.6, 1.0), (1.0, 0.6, 0.0), (0.0, 0.8, 0.2),
            (0.8, 0.0, 0.8), (1.0, 0.0, 0.0),
        ]

        for stage_idx, stage in enumerate(stages):
            try:
                stage_name = stage.get("name", f"Stage_{stage_idx+1}")
                folder_name = stage_name.replace(" ", "_").replace(".", "")
                stage_grp = doc.addObject("App::DocumentObjectGroup", folder_name)
                stage_root.addObject(stage_grp)
                FreeCAD.Console.PrintMessage(f"  Building {stage_name}...\n")

                color = stage_colors[stage_idx % len(stage_colors)]
                all_shapes = []

                for c_idx, curve in enumerate(stage.get("curves", [])):
                    pts = curve.get("points", [])
                    ctrl_pts_json = curve.get("ctrl_pts", None)
                    radius = curve.get("radius", None)

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

                for p_idx, plate in enumerate(stage.get("plates", [])):
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

    # -------------------------------------------------------------------------
    # 1. MAIN GEOMETRY — Beams
    # -------------------------------------------------------------------------
    skel_grp = doc.addObject("App::DocumentObjectGroup", "Ref_Skeleton_Final")
    geo_grp  = doc.addObject("App::DocumentObjectGroup", "Beams_CSG_Final")

    curves = data.get("curves", [])
    beam_shape_registry = []  # [(shape, start_pos, end_pos, obj_name), ...]

    if curves:
        total_curves = len(curves)
        curved_count = sum(1 for c in curves if 'ctrl_pts' in c)
        FreeCAD.Console.PrintMessage(
            f"Reconstructing {total_curves} beams "
            f"({curved_count} curved Bézier, {total_curves - curved_count} straight)...\n"
        )

        for i, curve in enumerate(curves):
            if i % 10 == 0:
                FreeCAD.Console.PrintMessage(f"  Processing beam {i+1}/{total_curves}...\n")
                try:
                    FreeCAD.Gui.updateGui()
                except:
                    pass

            pts          = curve.get("points", [])
            ctrl_pts_json = curve.get("ctrl_pts", None)   # [[x1,y1,z1],[x2,y2,z2]]
            radius_json  = curve.get("radius", None)

            if not pts:
                continue

            # --- Skeleton wireframe ---
            try:
                v_start = doc.addObject("Part::Vertex", f"Ref_Node_S_{i}")
                v_start.X, v_start.Y, v_start.Z = pts[0][0], pts[0][1], pts[0][2]
                v_start.ViewObject.PointSize = 5.0
                v_start.ViewObject.ShapeColor = (0.0, 1.0, 0.0)
                skel_grp.addObject(v_start)

                v_end = doc.addObject("Part::Vertex", f"Ref_Node_E_{i}")
                v_end.X, v_end.Y, v_end.Z = pts[-1][0], pts[-1][1], pts[-1][2]
                v_end.ViewObject.PointSize = 5.0
                v_end.ViewObject.ShapeColor = (0.0, 1.0, 0.0)
                skel_grp.addObject(v_end)

                for j in range(len(pts) - 1):
                    p1 = Vector(pts[j][0], pts[j][1], pts[j][2])
                    p2 = Vector(pts[j + 1][0], pts[j + 1][1], pts[j + 1][2])
                    l_obj = doc.addObject("Part::Line", f"Ref_L_{i}_{j}")
                    l_obj.X1, l_obj.Y1, l_obj.Z1 = p1.x, p1.y, p1.z
                    l_obj.X2, l_obj.Y2, l_obj.Z2 = p2.x, p2.y, p2.z
                    l_obj.ViewObject.LineWidth = 2.0
                    l_obj.ViewObject.ShapeColor = (1.0, 1.0, 0.0)
                    skel_grp.addObject(l_obj)
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"  Skeleton {i} failed: {str(e)[:40]}\n")

            # --- Solid geometry ---
            try:
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

                start_pos = pts[0][:3]
                end_pos   = pts[-1][:3]
                for j, shape in enumerate(shapes):
                    obj_name = f"Beam_{i}_P_{j}"
                    obj = doc.addObject("Part::Feature", obj_name)
                    obj.Shape = shape
                    obj.ViewObject.ShapeColor = (0.8, 0.0, 0.0)
                    geo_grp.addObject(obj)
                    beam_shape_registry.append((shape.copy(), start_pos, end_pos, obj_name))

            except Exception as e:
                FreeCAD.Console.PrintWarning(f"  Beam geometry {i} failed: {str(e)[:40]}\n")

    # -------------------------------------------------------------------------
    # 2. PLATES + JOINTS
    # -------------------------------------------------------------------------
    plates_list = data.get("plates", [])
    plate_shapes = {}

    if plates_list:
        plate_grp = doc.addObject("App::DocumentObjectGroup", "Plates_CSG_Final")
        FreeCAD.Console.PrintMessage(f"Reconstructing {len(plates_list)} Plate Regions...\n")

        metadata   = data.get("metadata", {})
        plate_mode = metadata.get("plate_mode", "bspline")

        for p_idx, plate in enumerate(plates_list):
            try:
                FreeCAD.Console.PrintMessage(f"  Plate {p_idx+1}/{len(plates_list)}...\n")
                try:
                    FreeCAD.Gui.updateGui()
                except:
                    pass

                plate_id  = plate.get("id", p_idx)
                thickness = plate.get("thickness", 2.0)
                verts     = plate.get("vertices", [])
                tris      = plate.get("triangles", [])

                if not (tris and verts):
                    FreeCAD.Console.PrintWarning(f"  Plate {plate_id}: No geometry\n")
                    continue

                def try_bspline():
                    bspline_data = plate.get("bspline_surface", None)
                    if bspline_data:
                        s = create_bspline_surface_from_data(bspline_data, thickness)
                        if s:
                            FreeCAD.Console.PrintMessage("    ✓ B-Spline Surface created\n")
                            return s
                    return None

                def try_voxels():
                    voxel_data = plate.get("voxels", None)
                    if voxel_data:
                        s = create_voxelized_geometry(voxel_data, plate.get("voxel_size", 1.0))
                        if s:
                            FreeCAD.Console.PrintMessage(f"    ✓ Voxel Plate created\n")
                            return s
                    return None

                def try_mesh():
                    s = create_plate_shell_batched(verts, tris, batch_size=500)
                    if s:
                        try:
                            if s.isClosed():
                                return Part.Solid(s)
                        except:
                            pass
                        return s
                    return None

                def try_cuboid():
                    cuboid_data = plate.get("cuboid", None)
                    if cuboid_data:
                        s = create_cuboid_geometry(cuboid_data)
                        if s:
                            FreeCAD.Console.PrintMessage("    ✓ Cuboid Plate created\n")
                            return s
                    return None

                steps_map = {
                    "voxel":   [try_voxels, try_bspline, try_mesh, try_cuboid],
                    "mesh":    [try_mesh, try_bspline, try_voxels, try_cuboid],
                }
                steps = steps_map.get(plate_mode, [try_bspline, try_voxels, try_mesh, try_cuboid])

                solid = None
                for step in steps:
                    try:
                        solid = step()
                    except Exception as e:
                        FreeCAD.Console.PrintWarning(f"    Step failed: {e}\n")
                    if solid:
                        break

                if solid:
                    plate_shapes[plate_id] = solid
                else:
                    FreeCAD.Console.PrintWarning(f"  Plate {plate_id}: Geometry creation failed\n")

            except Exception as e:
                FreeCAD.Console.PrintError(f"  Plate {p_idx} error: {str(e)[:50]}\n")

    # -------------------------------------------------------------------------
    # 3. FUSE beams + joints + plates
    # -------------------------------------------------------------------------
    joints_list = data.get("joints", [])
    joint_shapes_by_plate = {}
    joint_locs_by_plate   = {}
    joint_radii_by_plate  = {}

    if joints_list:
        FreeCAD.Console.PrintMessage(f"Building {len(joints_list)} joint connectors...\n")
        for j_idx, joint in enumerate(joints_list):
            try:
                loc       = joint.get("location", [0, 0, 0])
                direction = joint.get("direction", [0, 0, 1])
                radius    = joint.get("radius", 0.5)
                plate_id  = joint.get("plate_id", -1)
                if len(loc) == 3 and len(direction) == 3:
                    shape = create_joint_geometry(loc, direction, radius)
                    if plate_id not in joint_shapes_by_plate:
                        joint_shapes_by_plate[plate_id] = []
                        joint_locs_by_plate[plate_id]   = []
                        joint_radii_by_plate[plate_id]  = []
                    if shape:
                        joint_shapes_by_plate[plate_id].append(shape)
                    joint_locs_by_plate[plate_id].append(loc)
                    joint_radii_by_plate[plate_id].append(radius)
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"  Joint {j_idx} failed: {e}\n")

    SNAP_DIST = 3.0
    beam_shapes_by_plate = {}
    beams_used_in_plates = set()

    for plate_id, locs in joint_locs_by_plate.items():
        beam_shapes_by_plate[plate_id] = []
        for b_shape, b_start, b_end, b_name in beam_shape_registry:
            for loc in locs:
                d_start = sum((a - b) ** 2 for a, b in zip(b_start, loc)) ** 0.5
                d_end   = sum((a - b) ** 2 for a, b in zip(b_end,   loc)) ** 0.5
                if d_start < SNAP_DIST or d_end < SNAP_DIST:
                    beam_shapes_by_plate[plate_id].append(b_shape)
                    beams_used_in_plates.add(b_name)
                    break

    if plate_shapes:
        if 'plate_grp' not in dir():
            plate_grp = doc.addObject("App::DocumentObjectGroup", "Plates_CSG_Final")
        FreeCAD.Console.PrintMessage("Fusing plates with beams and joints...\n")

    for plate_id, plate_solid in plate_shapes.items():
        try:
            all_shapes = ([plate_solid]
                          + joint_shapes_by_plate.get(plate_id, [])
                          + beam_shapes_by_plate.get(plate_id, []))
            if len(all_shapes) > 1:
                try:
                    fused = all_shapes[0]
                    for s in all_shapes[1:]:
                        fused = fused.fuse(s)
                    fused = fused.removeSplitter()
                    plate_solid = fused if fused.isValid() else Part.Compound(all_shapes)
                except Exception as e:
                    plate_solid = Part.Compound(all_shapes)
                    FreeCAD.Console.PrintWarning(f"    Plate {plate_id}: Fusion failed, using compound\n")

            obj = doc.addObject("Part::Feature", f"Plate_{plate_id}")
            obj.Shape = plate_solid
            obj.ViewObject.ShapeColor = (0.0, 0.6, 0.9)
            obj.ViewObject.Transparency = 20
            plate_grp.addObject(obj)
            FreeCAD.Console.PrintMessage(f"    ✓ Plate {plate_id} added\n")
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"  Plate {plate_id} final add failed: {e}\n")

    for obj_name in beams_used_in_plates:
        try:
            obj = doc.getObject(obj_name)
            if obj:
                obj.ViewObject.Visibility = False
        except:
            pass

    # -------------------------------------------------------------------------
    # Finalise
    # -------------------------------------------------------------------------
    FreeCAD.Console.PrintMessage("Finalizing...\n")
    try:
        doc.recompute()
        FreeCAD.Console.PrintMessage("✓ Recompute successful\n")
    except Exception as e:
        FreeCAD.Console.PrintError(f"Recompute failed: {e}\n")
    try:
        FreeCAD.Gui.SendMsgToActiveView("ViewFit")
    except:
        pass

    FreeCAD.Console.PrintMessage("✓✓✓ Reconstruction Complete ✓✓✓\n")


if __name__ == '__main__':
    import_hybrid_json()
