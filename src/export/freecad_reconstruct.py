"""
FreeCAD Macro: Hybrid Beam-Plate Reconstruction (Stable Version)
Crash-resistant version with proper error handling and batching.
Includes history visualization, pipeline stages, and final geometry.
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

                            # Create one object per color
                            for color, boxes in color_groups.items():
                                if boxes:
                                    comp = Part.makeCompound(boxes)
                                    color_name = "Red" if color == (1.0, 0.0, 0.0) else "Cyan" if color == (0.0, 1.0, 1.0) else "Other"
                                    obj = doc.addObject("Part::Feature", f"{safe_params}_{color_name}")
                                    obj.Shape = comp
                                    hist_root.addObject(obj)
                                    obj.ViewObject.ShapeColor = color
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
                        pts = curve.get("points", [])
                        if len(pts) >= 2:
                            shapes = create_rod_geometry_ball_stick(pts)
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
    if curves:
        total_curves = len(curves)
        FreeCAD.Console.PrintMessage(f"Reconstructing {total_curves} Detailed Rods...\n")

        for i, curve in enumerate(curves):
            if i % 10 == 0:
                FreeCAD.Console.PrintMessage(f"  Processing curve {i+1}/{total_curves}...\n")
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

            # 2. BUILD GEOMETRY (CSG Solids)
            try:
                shapes = create_rod_geometry_ball_stick(pts)
                for j, shape in enumerate(shapes):
                    obj = doc.addObject("Part::Feature", f"Beam_{i}_P_{j}")
                    obj.Shape = shape
                    obj.ViewObject.ShapeColor = (0.8, 0.0, 0.0)
                    geo_grp.addObject(obj)
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"  Beam geometry {i} failed: {str(e)[:40]}\n")

    # --- PLATES (Simplified, no problematic operations) ---
    plates_list = data.get("plates", [])
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

                # Create shell with batching
                shell = create_plate_shell_batched(verts, tris, batch_size=500)

                if shell:
                    # Try to make solid (simple approach, no problematic operations)
                    solid = None
                    try:
                        if shell.isClosed():
                            solid = Part.Solid(shell)
                            FreeCAD.Console.PrintMessage(f"    ✓ Solid created\n")
                    except Exception as e:
                        FreeCAD.Console.PrintMessage(f"    Could not make solid: {str(e)[:40]}\n")

                    # Add to document
                    shape_to_add = solid if solid else shell
                    obj = doc.addObject("Part::Feature", f"Plate_{plate_id}")
                    obj.Shape = shape_to_add
                    obj.ViewObject.ShapeColor = (0.0, 0.6, 0.9)
                    obj.ViewObject.Transparency = 25
                    plate_grp.addObject(obj)
                    FreeCAD.Console.PrintMessage(f"    ✓ Plate {plate_id} added\n")
                else:
                    FreeCAD.Console.PrintWarning(f"  Plate {plate_id}: Shell creation failed\n")

            except Exception as e:
                FreeCAD.Console.PrintError(f"  Plate {p_idx} error: {str(e)[:50]}\n")
                import traceback
                FreeCAD.Console.PrintError(f"  {traceback.format_exc()[:200]}\n")

    # Finalize
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
