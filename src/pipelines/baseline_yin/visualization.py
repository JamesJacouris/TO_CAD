import open3d as o3d
import numpy as np

def viz_voxels(mask, pitch, origin, color=[1, 0, 0], radius_scale=1.0):
    """
    Visualize a binary voxel mask as a Point Cloud (faster) or VoxelGrid.
    """
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None
        
    pts = origin + (indices * pitch) + (pitch * 0.5)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color(color)
    return pcd

def viz_graph(node_coords, edges, node_color=[0,1,0], edge_color=[0,0,1]):
    """
    Visualize a graph (Nodes + Edges).
    node_coords: (N, 3) float array
    edges: list of (u, v) tuples
    """
    geoms = []
    
    if len(node_coords) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(node_coords)
        pcd.paint_uniform_color(node_color)
        geoms.append(pcd)
        
    if len(edges) > 0 and len(node_coords) > 0:
        # Open3D expects Nx2 int array for lines (POINT INDICES)
        # We need to construct a single global point cloud containing all Nodes + all Intermediate Points
        # Then indices will reference this global array.
        
        # Current geometry: 
        # Nodes: node_coords (indices 0..N-1)
        # Edges: [u, v, w, intermediates]
        
        all_points = [pt for pt in node_coords]
        all_lines = []
        
        current_idx = len(all_points)
        
        for e in edges:
            u, v = int(e[0]), int(e[1])
            pts = e[3] if len(e) > 3 else []
            
            # Start of chain is Node u
            prev_idx = u
            
            # Add intermediate points
            for pt in pts:
                all_points.append(pt)
                new_idx = current_idx
                all_lines.append([prev_idx, new_idx])
                prev_idx = new_idx
                current_idx += 1
                
            # End of chain is Node v
            all_lines.append([prev_idx, v])
            
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_points)
        line_set.lines = o3d.utility.Vector2iVector(all_lines)
        line_set.paint_uniform_color(edge_color)
        geoms.append(line_set)
        
    return geoms

def show_step(title, geometries):
    """
    Display a list of geometries in a window.
    """
    valid_geoms = []
    for g in geometries:
        if isinstance(g, list):
            valid_geoms.extend(g)
        elif g is not None:
            valid_geoms.append(g)

    if not valid_geoms:
        print(f"[Viz] {title}: Nothing to show.")
        return

    print(f"[Viz] Showing: {title}")
    # Add coordinate frame for reference
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    valid_geoms.append(frame)

    o3d.visualization.draw_geometries(valid_geoms, window_name=title)

def viz_iterative_thinning(iter_map, pitch, origin):
    """
    Visualizes voxels colored by the iteration they were removed.
    iter_map: 3D int array (0 = kept/bg, 1..N = removed iter)
    """
    indices = np.argwhere(iter_map > 0)
    if len(indices) == 0:
        return None
        
    pts = origin + (indices * pitch) + (pitch * 0.5)
    iters = iter_map[indices[:,0], indices[:,1], indices[:,2]]
    
    # Normalize iterations to 0..1 for colormap
    max_it = np.max(iters)
    if max_it == 0: max_it = 1
    
    # Simple heatmap (Blue=Early -> Red=Late)
    # Using matplotlib colormap if available, else manual gradient
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("jet")
        colors = cmap(iters / max_it)[:, :3] # RGBA -> RGB
    except ImportError:
        # Fallback: Blue to Red
        t = (iters / max_it)[:, None]
        colors = np.hstack((t, np.zeros_like(t), 1-t))
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def viz_skeleton_classification(skeleton, pitch, origin):
    """
    Visualizes skeleton voxels colored by connectivity.
    Blue=End (1 neighbor), Grey=Reg (2), Red=Joint (>2).
    """
    indices = np.argwhere(skeleton > 0)
    if len(indices) == 0:
        return None
        
    pts = origin + (indices * pitch) + (pitch * 0.5)
    colors = []
    
    # Compute neighbors for each point in the sparse set
    # Using a set lookup is fastest for sparse data
    voxel_set = set([tuple(idx) for idx in indices])
    
    for idx in indices:
        z, y, x = idx
        neighbors = 0
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz==0 and dy==0 and dx==0: continue
                    if (z+dz, y+dy, x+dx) in voxel_set:
                        neighbors += 1
                        
        if neighbors <= 1:
            colors.append([0, 0, 1]) # Blue (End)
        elif neighbors == 2:
            colors.append([0.7, 0.7, 0.7]) # Grey (Regular)
        else:
            colors.append([1, 0, 0]) # Red (Joint)
            
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def viz_graph_comparison(nodes1, edges1, nodes2, edges2, label1="Before", label2="After"):
    """
    Overlays two graphs. 
    Graph 1: Red (faint), Graph 2: Green (solid).
    """
    # Viz 1
    g1 = viz_graph(nodes1, edges1, node_color=[1, 0.5, 0.5], edge_color=[1, 0, 0]) 
    # Viz 2
    g2 = viz_graph(nodes2, edges2, node_color=[0, 1, 0], edge_color=[0, 1, 0])
    
    return g1 + g2

def viz_loads(nodes, loads, bcs, scale=10.0):
    """
    Draws arrows for Loads and cubes for BCs.
    """
    geoms = []
    
    # Draw Loads (Arrows)
    if loads:
        max_force = 0
        for f in loads.values():
            max_force = max(max_force, np.linalg.norm(f[:3]))
            
        if max_force == 0: max_force = 1.0
            
        for n_idx, force in loads.items():
            if n_idx >= len(nodes): continue
            
            p = nodes[n_idx]
            f_vec = np.array(force[:3])
            mag = np.linalg.norm(f_vec)
            
            if mag < 1e-6: continue
            
            # Start arrow at node, point in force direction
            # Open3D Arrow points in +Z by default
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.5, cone_radius=1.0, 
                cylinder_height=scale*0.7, cone_height=scale*0.3
            )
            arrow.paint_uniform_color([1, 1, 0]) # Yellow
            
            # Rotation to align +Z with f_vec
            z_axis = np.array([0, 0, 1])
            f_dir = f_vec / mag
            
            # Rotate
            # Axis = Z x F
            axis = np.cross(z_axis, f_dir)
            angle = np.arccos(np.dot(z_axis, f_dir))
            
            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                R =  o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                arrow.rotate(R, center=[0,0,0])
            elif np.dot(z_axis, f_dir) < 0:
                 # 180 flip
                 R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
                 arrow.rotate(R, center=[0,0,0])
                 
            arrow.translate(p)
            geoms.append(arrow)

    # Draw BCs (Cubes)
    if bcs:
        for n_idx in bcs.keys():
             if n_idx >= len(nodes): continue
             p = nodes[n_idx]
             box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
             box.paint_uniform_color([0, 1, 1]) # Cyan
             # Center box
             box.translate(p - np.array([1.0, 1.0, 1.0]))
             geoms.append(box)
             
    return geoms

def viz_zone_classification(zone_mask, pitch, origin):
    """
    Visualizes zone classification: Beams in red, Plates in cyan.
    zone_mask: 3D int array (1=plate, 2=beam, 0=background)
    """
    zone1_indices = np.argwhere(zone_mask == 1)
    zone2_indices = np.argwhere(zone_mask == 2)

    geoms = []

    # Zone 1 = Plates (shown as CYAN)
    if len(zone1_indices) > 0:
        pts = origin + (zone1_indices * pitch) + (pitch * 0.5)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0, 1, 1])  # CYAN for plates
        geoms.append(pcd)

    # Zone 2 = Beams (shown as RED)
    if len(zone2_indices) > 0:
        pts = origin + (zone2_indices * pitch) + (pitch * 0.5)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([1, 0, 0])  # RED for beams
        geoms.append(pcd)

    return geoms

def save_zone_visualization(zone_mask, pitch, origin, output_path):
    """
    Save zone classification visualization as PNG image (top view and side views).
    zone_mask: 3D int array with shape (Y, X, Z) matching Top3D convention.
               Values: 1=plate, 2=beam, 0=background
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # Array is (Y, X, Z) — axis 0=Y, axis 1=X, axis 2=Z
        indices = np.argwhere(zone_mask > 0)
        if len(indices) == 0:
            return False

        y_min, x_min, z_min = indices.min(axis=0)
        y_max, x_max, z_max = indices.max(axis=0)

        # Create custom colormap: 0=white (bg), 1=cyan (plate), 2=red (beam)
        colors = ['white', 'cyan', 'red']
        cmap = ListedColormap(colors)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Top View (XY plane): slice at Z midpoint — looking down
        z_mid = (z_min + z_max) // 2
        slice_xy = zone_mask[:, :, z_mid]  # (Y, X) at fixed Z

        im_xy = axes[0].imshow(slice_xy, cmap=cmap, origin='lower', vmin=0, vmax=2,
                               extent=[x_min*pitch, (x_max+1)*pitch, y_min*pitch, (y_max+1)*pitch])
        axes[0].set_title("Top View (XY plane)")
        axes[0].set_xlabel("X (mm)")
        axes[0].set_ylabel("Y (mm)")
        plt.colorbar(im_xy, ax=axes[0], label="Zone (0=bg, 1=plate, 2=beam)", ticks=[0, 1, 2])

        # Side View (XZ plane): slice at Y midpoint — looking from front
        y_mid = (y_min + y_max) // 2
        slice_xz = zone_mask[y_mid, :, :]  # (X, Z) at fixed Y

        im_xz = axes[1].imshow(slice_xz.T, cmap=cmap, origin='lower', vmin=0, vmax=2,
                               extent=[x_min*pitch, (x_max+1)*pitch, z_min*pitch, (z_max+1)*pitch])
        axes[1].set_title("Side View (XZ plane)")
        axes[1].set_xlabel("X (mm)")
        axes[1].set_ylabel("Z (mm)")
        plt.colorbar(im_xz, ax=axes[1], label="Zone (0=bg, 1=plate, 2=beam)", ticks=[0, 1, 2])

        # Side View (YZ plane): slice at X midpoint — looking from side
        x_mid = (x_min + x_max) // 2
        slice_yz = zone_mask[:, x_mid, :]  # (Y, Z) at fixed X

        im_yz = axes[2].imshow(slice_yz.T, cmap=cmap, origin='lower', vmin=0, vmax=2,
                               extent=[y_min*pitch, (y_max+1)*pitch, z_min*pitch, (z_max+1)*pitch])
        axes[2].set_title("Side View (YZ plane)")
        axes[2].set_xlabel("Y (mm)")
        axes[2].set_ylabel("Z (mm)")
        plt.colorbar(im_yz, ax=axes[2], label="Zone (0=bg, 1=plate, 2=beam)", ticks=[0, 1, 2])

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return True
    except Exception as e:
        print(f"[Warning] Could not save zone visualization: {e}")
        return False

def viz_graph_radii(nodes, edges, radii):
    """
    Visualizes graph edges colored by Radius.
    """
    # Reuse viz_graph logic but apply colors to LineSet
    # LineSet colors are per-segment.
    
    geoms = []
    
    if len(edges) == 0: return geoms
    
    # Normalize radii
    r_min = np.min(radii)
    r_max = np.max(radii)
    if r_max == r_min: r_max += 1e-6
    
    all_points = [pt for pt in nodes]
    all_lines = []
    line_colors = []
    
    current_idx = len(all_points)
    
    # Try Matplotlib
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("jet")
    except:
        cmap = None
    
    for i, e in enumerate(edges):
        u, v = int(e[0]), int(e[1])
        pts = e[3] if len(e) > 3 else []
        r = radii[i]
        
        # Color
        val = (r - r_min) / (r_max - r_min)
        if cmap:
            c = cmap(val)[:3]
        else:
            c = [val, 0, 1-val] # Red-Blue
            
        # Segments
        prev_idx = u
        for pt in pts:
            all_points.append(pt)
            new_idx = current_idx
            all_lines.append([prev_idx, new_idx])
            line_colors.append(c)
            prev_idx = new_idx
            current_idx += 1
        all_lines.append([prev_idx, v])
        line_colors.append(c)
        
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(all_lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    
    geoms.append(line_set)
    
    # Add Nodes
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nodes)
    pcd.paint_uniform_color([0, 0, 0])
    geoms.append(pcd)
    
    return geoms
