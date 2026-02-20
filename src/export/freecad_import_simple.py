"""
FreeCAD Macro: Simple TO_CAD Pipeline Viewer
Ultra-minimal, reliable visualization of pipeline output.
Uses only basic FreeCAD objects (spheres, lines) - no complex geometry.
"""

import FreeCAD
import Part
import json
from FreeCAD import Vector

try:
    from PySide import QtGui
except ImportError:
    try:
        from PySide2 import QtWidgets as QtGui
    except ImportError:
        from PySide6 import QtWidgets as QtGui


def import_pipeline_output(json_path=None):
    """Simple viewer for TO_CAD pipeline output."""

    # File selection
    if not json_path:
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            None, "Select Pipeline JSON", "", "JSON Files (*.json)"
        )
        json_path = filename

    if not json_path:
        print("[IMPORT] No file selected.")
        return

    print(f"[IMPORT] Loading {json_path}")

    # Load JSON
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"[IMPORT] ✓ JSON loaded")
    except Exception as e:
        print(f"[IMPORT] ✗ Load failed: {e}")
        return

    # Get or create document
    doc = FreeCAD.activeDocument()
    if not doc:
        doc = FreeCAD.newDocument("Pipeline_Output")
    print(f"[IMPORT] Document: {doc.Name}")

    # Create groups
    try:
        nodes_grp = doc.addObject("App::DocumentObjectGroup", "Nodes")
        edges_grp = doc.addObject("App::DocumentObjectGroup", "Edges")
        print(f"[IMPORT] Groups created")
    except Exception as e:
        print(f"[IMPORT] ✗ Group creation failed: {e}")
        return

    # Extract graph
    graph = data.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    print(f"[IMPORT] Graph: {len(nodes)} nodes, {len(edges)} edges")

    # Create node spheres
    if nodes:
        print(f"[IMPORT] Creating {len(nodes)} nodes...")
        for i, node in enumerate(nodes):
            try:
                x, y, z = node[0], node[1], node[2]
                radius = node[3] if len(node) > 3 else 1.0

                # Create sphere
                sphere = Part.makeSphere(radius, Vector(x, y, z))
                obj = doc.addObject("Part::Feature", f"Node_{i}")
                obj.Shape = sphere
                obj.ViewObject.ShapeColor = (1.0, 0.0, 0.0)  # Red
                nodes_grp.addObject(obj)

                if i % 20 == 0:
                    print(f"[IMPORT]   {i+1}/{len(nodes)} nodes done")
                    try:
                        FreeCAD.Gui.updateGui()
                    except:
                        pass

            except Exception as e:
                print(f"[IMPORT]   Node {i} failed: {str(e)[:40]}")
                continue

        print(f"[IMPORT] ✓ All nodes created")

    # Create edges (lines between nodes)
    if edges:
        print(f"[IMPORT] Creating {len(edges)} edges...")
        for i, edge in enumerate(edges):
            try:
                u, v = edge[0], edge[1]
                if u >= len(nodes) or v >= len(nodes):
                    continue

                p1 = Vector(nodes[u][0], nodes[u][1], nodes[u][2])
                p2 = Vector(nodes[v][0], nodes[v][1], nodes[v][2])

                # Create line
                line_obj = doc.addObject("Part::Line", f"Edge_{i}")
                line_obj.X1, line_obj.Y1, line_obj.Z1 = p1.x, p1.y, p1.z
                line_obj.X2, line_obj.Y2, line_obj.Z2 = p2.x, p2.y, p2.z
                line_obj.ViewObject.LineWidth = 2.0
                line_obj.ViewObject.ShapeColor = (0.0, 1.0, 0.0)  # Green
                edges_grp.addObject(line_obj)

                if i % 50 == 0:
                    print(f"[IMPORT]   {i+1}/{len(edges)} edges done")
                    try:
                        FreeCAD.Gui.updateGui()
                    except:
                        pass

            except Exception as e:
                print(f"[IMPORT]   Edge {i} failed: {str(e)[:40]}")
                continue

        print(f"[IMPORT] ✓ All edges created")

    # Finalize
    print(f"[IMPORT] Finalizing...")
    try:
        doc.recompute()
        print(f"[IMPORT] ✓ Recompute successful")
    except Exception as e:
        print(f"[IMPORT] Recompute failed: {e}")

    try:
        FreeCAD.Gui.SendMsgToActiveView("ViewFit")
    except:
        pass

    print(f"[IMPORT] ✓✓✓ COMPLETE ✓✓✓")
    print(f"[IMPORT] Nodes: {len(nodes)}, Edges: {len(edges)}")


if __name__ == "__main__":
    import_pipeline_output()
