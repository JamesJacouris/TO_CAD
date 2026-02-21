import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.spatial import cKDTree


def recover_plate_regions_from_skeleton(skeleton_zone_mask, plate_labels, original_solid, edt):
    """
    Dilate skeleton plate voxels back into the original solid to recover
    full-thickness plate regions for mesh extraction.

    Parameters
    ----------
    skeleton_zone_mask : ndarray
        Zone classification of skeleton (0=empty, 1=plate, 2=beam).
    plate_labels : ndarray
        Connected component labels for plate skeleton voxels.
    original_solid : ndarray
        Original binary solid volume.
    edt : ndarray
        EDT of the original solid (voxel units).

    Returns
    -------
    recovered_zone_mask : ndarray
        Zone mask expanded to cover full plate thickness (0=empty, 1=plate, 2=beam).
    recovered_plate_labels : ndarray
        Plate labels expanded to full thickness.
    """
    from scipy.ndimage import label as scipy_label

    recovered_zone = skeleton_zone_mask.copy()
    recovered_labels = plate_labels.copy()

    # For each plate label, dilate skeleton surface voxels into original solid
    s26 = np.ones((3, 3, 3), dtype=bool)
    unique_labels = np.unique(plate_labels[plate_labels > 0])

    beam_mask = (skeleton_zone_mask == 2)

    for lab in unique_labels:
        plate_seed = (plate_labels == lab)

        # Get max EDT in this plate region to determine dilation radius
        plate_edt_vals = edt[plate_seed & (edt > 0)]
        if len(plate_edt_vals) == 0:
            continue
        max_radius = int(np.ceil(np.max(plate_edt_vals))) + 1

        # Iterative dilation within the original solid
        current = plate_seed.copy()
        for _ in range(max_radius):
            dilated = binary_dilation(current, structure=s26)
            # Constrain to original solid and not beam zone
            dilated = dilated & original_solid & (~beam_mask)
            if np.array_equal(dilated, current):
                break
            current = dilated

        # Assign recovered voxels
        new_voxels = current & (recovered_zone == 0)  # Only fill empty space
        recovered_zone[new_voxels] = 1
        recovered_labels[new_voxels] = lab

    n_recovered = int(np.sum(recovered_zone == 1)) - int(np.sum(skeleton_zone_mask == 1))
    print(f"    [PlateRecovery] Recovered {n_recovered} additional plate voxels "
          f"({len(unique_labels)} regions)")

    return recovered_zone, recovered_labels


def extract_plates(skeleton, surface_mask, solid, pitch, origin, beam_node_coords=None):
    """
    Extracts structured plate geometry from the hybrid skeleton using
    voxel boundary mesh extraction with Taubin smoothing.

    skeleton: 3D binary array (medial surface + curves)
    surface_mask: 3D int array (labels for plate regions)
    solid: original source solid (for thickness estimation via EDT)
    pitch: voxel size in mm
    origin: 3D array, world-space origin of the grid
    """
    print(f"    [Extraction] Processing {np.max(surface_mask)} potential plate regions...")
    edt = distance_transform_edt(solid)

    unique_labels = np.unique(surface_mask)
    unique_labels = unique_labels[unique_labels > 0]

    plates_data = []

    for label_id in unique_labels:
        # 1. Isolate plate voxels in the skeleton
        plate_voxels_mask = (skeleton > 0) & (surface_mask == label_id)
        indices = np.argwhere(plate_voxels_mask)
        if len(indices) < 3:
            continue

        # 2. Extract Boundary Mesh (Volumetric)
        voxel_set = set(map(tuple, indices))
        vertices, triangles = _extract_boundary_faces(voxel_set, pitch, origin)

        if len(triangles) < 4:
            print(f"    [Extraction] Plate {label_id}: Too few faces ({len(triangles)}). Skipping.")
            continue

        # 3. Merge close vertices (pitch-relative tolerance)
        vertices, triangles = _merge_close_vertices(vertices, triangles, tolerance=pitch * 0.01)

        if len(triangles) < 4:
            continue

        # 4. Taubin smoothing (shrinkage-free)
        vertices = _taubin_smooth(vertices, triangles, n_iters=5, lam=0.5, mu=-0.53)

        # 5. Remove degenerate triangles (pitch-relative thresholds)
        min_area = (pitch * 0.05) ** 2
        min_edge = pitch * 0.05
        triangles = _filter_degenerate(vertices, triangles, min_area, min_edge)

        # Minimum triangle count (pitch-relative)
        min_tris = max(4, len(voxel_set) // 10)
        if len(triangles) < min_tris:
            print(f"    [Extraction] Plate {label_id}: Too small after cleanup "
                  f"({len(triangles)} tris, need {min_tris}). Skipping.")
            continue

        # 6. Metadata 
        if len(vertices) > 0:
            centroid = np.mean(vertices, axis=0)
        else:
            centroid = np.array([0.0, 0.0, 0.0])
            
        thickness = 0.0 # Flag as solid

        # 7. Connection points (beam-plate interface)
        beam_skeleton = (skeleton > 0) & (surface_mask == 0)
        dilated_plate = binary_dilation(plate_voxels_mask, structure=np.ones((3, 3, 3)))
        intersection = dilated_plate & beam_skeleton
        conn_indices = np.argwhere(intersection)
        connection_points = (origin + (conn_indices * pitch)).tolist()

        plates_data.append({
            "id": int(label_id),
            "type": "solid_mesh",
            "vertices": vertices.tolist() if isinstance(vertices, np.ndarray) else vertices,
            "triangles": triangles,
            "thickness": thickness,
            "normal": [0.0, 0.0, 1.0], # Dummy normal for solid
            "center": centroid.tolist(),
            "connection_points": connection_points,
            "voxel_size": float(pitch),
        })

    print(f"    [Extraction] Successfully extracted {len(plates_data)} plates.")
    return plates_data


def _extract_boundary_faces(voxel_set, pitch, origin):
    """
    Extracts ALL boundary faces of the voxel set to form a closed manifold mesh.
    """
    face_dirs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    all_faces = []

    for (vz, vy, vx) in voxel_set:
        wp = origin + np.array([vz, vy, vx], dtype=float) * pitch
        for d in face_dirs:
            dz, dy, dx = d
            nz, ny, nx = vz + dz, vy + dy, vx + dx
            if (nz, ny, nx) not in voxel_set:
                face_center = wp + np.array([dz, dy, dx], dtype=float) * (pitch * 0.5)
                if dz != 0:
                    t1, t2 = np.array([0.0, pitch, 0.0]), np.array([0.0, 0.0, pitch])
                elif dy != 0:
                    t1, t2 = np.array([pitch, 0.0, 0.0]), np.array([0.0, 0.0, pitch])
                else:
                    t1, t2 = np.array([pitch, 0.0, 0.0]), np.array([0.0, pitch, 0.0])
                p0 = face_center - 0.5 * t1 - 0.5 * t2
                p1 = face_center + 0.5 * t1 - 0.5 * t2
                p2 = face_center + 0.5 * t1 + 0.5 * t2
                p3 = face_center - 0.5 * t1 + 0.5 * t2
                all_faces.append([tuple(p0), tuple(p1), tuple(p2), tuple(p3)])

    v_map, vertices, triangles = {}, [], []
    for corners in all_faces:
        vis = []
        for c in corners:
            if c not in v_map:
                v_map[c] = len(vertices)
                vertices.append(list(c))
            vis.append(v_map[c])
        triangles.append([vis[0], vis[1], vis[2]])
        triangles.append([vis[0], vis[2], vis[3]])

    return np.array(vertices, dtype=float) if vertices else np.zeros((0, 3)), triangles


def _merge_close_vertices(vertices, triangles, tolerance):
    if len(vertices) == 0: return vertices, triangles
    verts = np.asarray(vertices, dtype=float)
    quantized = np.round(verts / tolerance) * tolerance
    unique_verts, inverse = np.unique(quantized, axis=0, return_inverse=True)
    new_tris = []
    for t in triangles:
        nt = [int(inverse[t[0]]), int(inverse[t[1]]), int(inverse[t[2]])]
        if nt[0] != nt[1] and nt[1] != nt[2] and nt[0] != nt[2]:
            new_tris.append(nt)
    return unique_verts, new_tris


def _taubin_smooth(vertices, triangles, n_iters=5, lam=0.5, mu=-0.53):
    verts = np.array(vertices, dtype=float)
    n_verts = len(verts)
    if n_verts == 0: return verts
    adj = [set() for _ in range(n_verts)]
    for t in triangles:
        for i in range(3):
            u, v = t[i], t[(i + 1) % 3]
            adj[u].add(v)
            adj[v].add(u)
    for _ in range(n_iters):
        for step_val in [lam, mu]:
            new_verts = verts.copy()
            for i in range(n_verts):
                if adj[i]:
                    neighbor_avg = np.mean(verts[list(adj[i])], axis=0)
                    new_verts[i] = verts[i] + step_val * (neighbor_avg - verts[i])
            verts = new_verts
    return verts


def _filter_degenerate(vertices, triangles, min_area, min_edge):
    verts = np.asarray(vertices, dtype=float)
    clean = []
    for t in triangles:
        p0, p1, p2 = verts[t[0]], verts[t[1]], verts[t[2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        if area < min_area: continue
        e1, e2, e3 = np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p0 - p2)
        if min(e1, e2, e3) < min_edge: continue
        clean.append(t)
    return clean


def extract_plates_v2(plate_skeleton, plate_labels, solid, edt, pitch, origin,
                      zone_mask=None, bc_tags=None, recovered_labels=None):
    """
    Enhanced plate extraction producing both solid mesh and mid-surface + thickness.

    Parameters
    ----------
    plate_skeleton : ndarray (D, H, W), bool/int
        Surface-preserving thinned skeleton of plate zones only.
    plate_labels : ndarray (D, H, W), int
        Labeled plate regions (0=not plate, 1..N=plate IDs).
    solid : ndarray (D, H, W), bool
        Original binary solid.
    edt : ndarray (D, H, W), float
        EDT of the full solid (voxel units).
    pitch : float
        Voxel size in mm.
    origin : ndarray (3,)
        World-space origin.
    zone_mask : ndarray or None
        Zone classification (1=beam, 2=plate). Used for connection point detection.
    bc_tags : ndarray or None
        Original BC tags (1=fixed, 2=loaded).

    Returns
    -------
    plates_data : list of dict
        Each dict contains solid_mesh, mid_surface, thickness, normal, etc.
    """
    unique_labels = np.unique(plate_labels[plate_labels > 0])
    print(f"    [PlatesV2] Processing {len(unique_labels)} plate regions...")

    plates_data = []

    for label_id in unique_labels:
        # --- Grouping Logic ---
        # plate_labels = thin skeleton labels (Step 2.5)
        # recovered_labels = thick dilated labels (for volume/mesh)
        thinned_region = plate_labels == label_id
        plate_region = recovered_labels == label_id if recovered_labels is not None else thinned_region

        # --- Solid Mesh (boundary of the full plate solid, not just skeleton) ---
        plate_solid_voxels = solid & plate_region
        solid_indices = np.argwhere(plate_solid_voxels)
        if len(solid_indices) < 3:
            continue

        voxel_set = set(map(tuple, solid_indices))
        vertices, triangles = _extract_boundary_faces(voxel_set, pitch, origin)

        if len(triangles) < 4:
            print(f"    [PlatesV2] Plate {label_id}: Too few boundary faces. Skipping.")
            continue

        vertices, triangles = _merge_close_vertices(vertices, triangles, tolerance=pitch * 0.01)
        if len(triangles) < 4:
            continue

        vertices = _taubin_smooth(vertices, triangles, n_iters=5, lam=0.5, mu=-0.53)

        min_area = (pitch * 0.05) ** 2
        min_edge = pitch * 0.05
        triangles = _filter_degenerate(vertices, triangles, min_area, min_edge)

        min_tris = max(4, len(voxel_set) // 10)
        if len(triangles) < min_tris:
            print(f"    [PlatesV2] Plate {label_id}: Too small after cleanup "
                  f"({len(triangles)} tris, need {min_tris}). Skipping.")
            continue

        centroid = np.mean(vertices, axis=0) if len(vertices) > 0 else np.zeros(3)

        # --- Mid-Surface from plate skeleton ---
        skel_voxels = np.argwhere((plate_skeleton > 0) & plate_region)
        mid_surface_data = None
        if len(skel_voxels) >= 10:
            mid_surface_data = _extract_mid_surface(skel_voxels, edt, pitch, origin, bc_tags=bc_tags)

        # --- Thickness from EDT ---
        region_edt_values = edt[plate_solid_voxels]
        mean_thickness = float(2.0 * np.mean(region_edt_values) * pitch) if len(region_edt_values) > 0 else 0.0

        # --- Robust OBB (Oriented Bounding Box) ---
        # PCA on world-space points
        world_centers = origin + (solid_indices * pitch) + (pitch * 0.5)
        mean_w = np.mean(world_centers, axis=0)
        centered_w = world_centers - mean_w
        cov = np.cov(centered_w, rowvar=False) if len(world_centers) > 2 else np.eye(3)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Project into PCA space to find extents
        projs = centered_w @ eigenvectors # N x 3
        p_min = np.min(projs, axis=0)
        p_max = np.max(projs, axis=0)
        
        # Dimensions along PCA axes
        # eigenvectors[:, 0] is thickness direction
        dims = p_max - p_min
        dims[0] = max(dims[0], mean_thickness) # Thickness
        dims[1] = max(dims[1], pitch) # Width
        dims[2] = max(dims[2], pitch) # Length
        
        # Center in World Space
        # The true center is mean_w + eigenvectors * midpoint_of_projections
        mid_proj = (p_min + p_max) / 2.0
        obb_center = mean_w + eigenvectors @ mid_proj
        
        cuboid_data = {
            "center": obb_center.tolist(),
            "rotation": eigenvectors.tolist(), # Columns are the axes
            "dimensions": dims.tolist(),
            "type": "obb"
        }

        # --- Normal direction (Smallest eigenvector) ---
        normal = eigenvectors[:, 0].tolist() 

        # --- Connection points (zone boundary detection) ---
        connection_points = []
        if zone_mask is not None:
            s26 = np.ones((3, 3, 3), dtype=bool)
            plate_dilated = binary_dilation(plate_region, structure=s26)
            beam_at_interface = (zone_mask == 2) & plate_dilated  # zone==2 is beams
            conn_indices = np.argwhere(beam_at_interface)
            connection_points = (origin + (conn_indices * pitch) + (pitch * 0.5)).tolist()

        # --- Skeleton Voxels (Topological / Post-Thinning) ---
        # We strictly follow the thinned_region (Step 2.5) to avoid over-counting
        skel_indices = np.argwhere((plate_skeleton > 0) & thinned_region)
        world_skel_centers = origin + (skel_indices * pitch) + (pitch * 0.5)

        plate_dict = {
            "id": int(label_id),
            "type": "solid_mesh",
            "voxels": world_skel_centers.tolist(), # Precision skeleton representation
            "vertices": vertices.tolist() if isinstance(vertices, np.ndarray) else vertices,
            "triangles": triangles,
            "thickness": mean_thickness,
            "normal": normal,
            "center": centroid.tolist(),
            "connection_points": connection_points,
            "connection_node_ids": [],  # Populated later by joint_creation
            "voxel_size": float(pitch),
        }

        if mid_surface_data is not None:
            plate_dict["mid_surface"] = mid_surface_data
            
        plate_dict["cuboid"] = cuboid_data

        plates_data.append(plate_dict)

    print(f"    [PlatesV2] Successfully extracted {len(plates_data)} plates.")
    return plates_data


def _extract_mid_surface(skel_voxels, edt, pitch, origin, bc_tags=None):
    """
    Extract a mid-surface triangulation from plate skeleton voxels.

    Uses Open3D alpha shapes for triangulation of the skeleton point cloud.
    Falls back to convex hull if alpha shapes fail.

    Returns dict with vertices, triangles, thickness_per_vertex, mean_thickness,
    node_tags (dict mapping vertex index to original BC tag),
    or None if extraction fails.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("    [MidSurface] Open3D not available, skipping mid-surface extraction.")
        return None

    # Convert skeleton voxels to world coordinates (center of voxels)
    points = origin + skel_voxels.astype(np.float64) * pitch + pitch * 0.5
    D, H, W = edt.shape

    if len(points) < 10:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals for surface reconstruction
    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=pitch * 3, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
    except RuntimeError as e:
        # Degenerate geometry (all points coplanar or too few points)
        print(f"      [Warning] Normal estimation failed (degenerate geometry): {e}")
        return None

    # Try alpha shapes with increasing alpha until we get a mesh
    mesh = None
    for alpha_mult in [2.0, 3.0, 5.0, 8.0]:
        alpha = pitch * alpha_mult
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            if len(mesh.triangles) > 0:
                break
        except Exception:
            continue

    if mesh is None or len(mesh.triangles) == 0:
        return None

    ms_vertices = np.asarray(mesh.vertices)
    ms_triangles = np.asarray(mesh.triangles).tolist()

    # Taubin smooth the mid-surface
    ms_vertices = _taubin_smooth(ms_vertices, ms_triangles, n_iters=3, lam=0.5, mu=-0.53)

    # Compute per-vertex thickness from EDT
    thickness_per_vertex = np.zeros(len(ms_vertices))
    for i, v in enumerate(ms_vertices):
        voxel_coord = np.round((v - origin - pitch * 0.5) / pitch).astype(int)
        vz = np.clip(voxel_coord[0], 0, D - 1)
        vy = np.clip(voxel_coord[1], 0, H - 1)
        vx = np.clip(voxel_coord[2], 0, W - 1)
        thickness_per_vertex[i] = 2.0 * edt[vz, vy, vx] * pitch

    mean_thickness = float(np.mean(thickness_per_vertex)) if len(thickness_per_vertex) > 0 else 0.0

    # Propagate BC Tags to mid-surface vertices
    node_tags_ms = {}
    if bc_tags is not None:
        for i, v in enumerate(ms_vertices):
            voxel_coord = np.round((v - origin - pitch * 0.5) / pitch).astype(int)
            vz = np.clip(voxel_coord[0], 0, D - 1)
            vy = np.clip(voxel_coord[1], 0, H - 1)
            vx = np.clip(voxel_coord[2], 0, W - 1)
            tag = int(bc_tags[vz, vy, vx])
            if tag > 0:
                node_tags_ms[i] = tag

    return {
        "vertices": ms_vertices.tolist(),
        "triangles": ms_triangles,
        "thickness_per_vertex": thickness_per_vertex.tolist(),
        "mean_thickness": mean_thickness,
        "node_tags": node_tags_ms
    }
