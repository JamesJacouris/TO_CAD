"""ANSYS/SolidWorks-quality 3D FEA visualization and VTK export.

Produces publication-quality 3D renders of strain energy density,
displacement magnitude, and density fields from the TO_CAD pipeline.
Optionally exports .vtr files for interactive ParaView exploration.

Requires ``pyvista`` for rendering (headless off-screen).
Optionally uses ``pyevtk`` for lightweight .vtr file export.

Usage from pipeline::

    from src.export.vtk_export import render_pipeline_stage
    render_pipeline_stage(
        "SIMP Binary", se_field, density, u,
        nelx, nely, nelz, pitch, origin,
        output_dir, base_name,
    )
"""

import os
import logging
import numpy as np

_PYVISTA_AVAILABLE = False
_PYEVTK_AVAILABLE = False

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    _PYVISTA_AVAILABLE = True
    # Suppress ALL VTK warnings — prevents cascading WARNING:root:WARNING:root:... crash
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()
    vtk.vtkOutputWindow.GetInstance().SetDisplayModeToNever()
except (ImportError, Exception):
    pass

try:
    from pyevtk.hl import gridToVTK
    _PYEVTK_AVAILABLE = True
except ImportError:
    pass


# ── Camera presets ────────────────────────────────────────────────────────
CAMERA_PRESETS = {
    'isometric':  {'name': 'isometric'},
    'iso_rear':   {'name': 'iso_rear'},
    'top':        {'name': 'top'},
    'front':      {'name': 'front'},
    'right':      {'name': 'right'},
}


# ── Internal helpers ──────────────────────────────────────────────────────

def _top3d_to_vtk_cell(arr):
    """Transpose Top3D (nely, nelx, nelz) → VTK (nelx, nely, nelz), flatten F-order."""
    return arr.transpose(1, 0, 2).flatten(order='F')


def _displacement_to_vtk_point(u, nelx, nely, nelz):
    """Convert flat DOF vector to per-node displacement magnitude.

    Args:
        u: (ndof,) displacement vector from Top3D evaluate().
        nelx, nely, nelz: Element counts.

    Returns:
        dict with 'ux', 'uy', 'uz', 'magnitude' as 1D arrays in VTK point order.
    """
    # DOF layout: [ux0, uy0, uz0, ux1, ...].  Node grid: (nely+1, nelx+1, nelz+1) F-order.
    ux = u[0::3].reshape((nely + 1, nelx + 1, nelz + 1), order='F').transpose(1, 0, 2)
    uy = u[1::3].reshape((nely + 1, nelx + 1, nelz + 1), order='F').transpose(1, 0, 2)
    uz = u[2::3].reshape((nely + 1, nelx + 1, nelz + 1), order='F').transpose(1, 0, 2)
    mag = np.sqrt(ux**2 + uy**2 + uz**2)
    return {
        'ux':        ux.flatten(order='F'),
        'uy':        uy.flatten(order='F'),
        'uz':        uz.flatten(order='F'),
        'magnitude': mag.flatten(order='F'),
    }


def _prepare_grid(nelx, nely, nelz, pitch, origin):
    """Create a PyVista ImageData grid matching the Top3D domain."""
    grid = pv.ImageData()
    grid.dimensions = (nelx + 1, nely + 1, nelz + 1)
    grid.spacing = (pitch, pitch, pitch)
    grid.origin = (float(origin[0]), float(origin[1]), float(origin[2]))
    return grid


def _build_frame_mesh(nodes, edges, radii, beam_se=None):
    """Build a merged PyVista mesh of tubes (cylinders) for the beam frame.

    Args:
        nodes: (N, 3) node positions.
        edges: (M, 2) edge connectivity.
        radii: (M,) per-edge radius.
        beam_se: (M,) per-beam strain energy from beam FEM, or None.

    Returns:
        pv.PolyData: Merged tube mesh with 'radius' and optionally
        'strain_energy' cell data for coloring.
    """
    tubes = []
    for i, (u, v) in enumerate(edges):
        r = float(radii[i])
        if r <= 0:
            continue
        p0 = nodes[int(u)]
        p1 = nodes[int(v)]
        length = np.linalg.norm(p1 - p0)
        if length < 1e-12:
            continue
        line = pv.Line(p0, p1, resolution=1)
        tube = line.tube(radius=r, n_sides=20)
        tube.cell_data['radius'] = np.full(tube.n_cells, r)
        if beam_se is not None:
            tube.cell_data['strain_energy'] = np.full(tube.n_cells, float(beam_se[i]))
        tubes.append(tube)

    if not tubes:
        return None

    merged = tubes[0]
    for t in tubes[1:]:
        merged = merged.merge(t)
    return merged


def _set_camera(pl, bounds, camera_preset):
    """Set camera position from bounds and a preset dict. Shared by all renderers."""
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    diag = np.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
    dist = diag * 2.0
    focal = (cx, cy, cz)

    view_name = camera_preset.get('name', 'isometric')
    if view_name == 'top':
        pl.camera.position = (cx, cy, cz + dist)
        pl.camera.focal_point = focal
        pl.camera.up = (0.0, 1.0, 0.0)
    elif view_name == 'front':
        pl.camera.position = (cx, cy - dist, cz)
        pl.camera.focal_point = focal
        pl.camera.up = (0.0, 0.0, 1.0)
    elif view_name == 'right':
        pl.camera.position = (cx + dist, cy, cz)
        pl.camera.focal_point = focal
        pl.camera.up = (0.0, 0.0, 1.0)
    elif view_name == 'iso_rear':
        az, el = np.radians(225), np.radians(30)
        pl.camera.position = (
            cx + dist * np.cos(el) * np.sin(az),
            cy + dist * np.cos(el) * np.cos(az),
            cz + dist * np.sin(el),
        )
        pl.camera.focal_point = focal
        pl.camera.up = (0.0, 0.0, 1.0)
    else:  # isometric (default)
        az, el = np.radians(45), np.radians(30)
        pl.camera.position = (
            cx + dist * np.cos(el) * np.sin(az),
            cy + dist * np.cos(el) * np.cos(az),
            cz + dist * np.sin(el),
        )
        pl.camera.focal_point = focal
        pl.camera.up = (0.0, 0.0, 1.0)

    pl.camera.zoom(1.3)


def _render_single(mesh, scalars, camera_preset, output_path, window_size,
                   cmap='inferno', log_scale=False, scalar_bar_title='',
                   title='', show_edges=False):
    """Render a single view and save as PNG."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('white')

    # Log scaling for strain energy (wide dynamic range)
    render_scalars = scalars
    clim = None
    if log_scale:
        vals = mesh[scalars]
        valid = vals[vals > 0]
        if len(valid) > 0:
            vmin = np.percentile(valid, 1)
            log_vals = np.log10(np.clip(vals, vmin, None))
            mesh['_log_scalars'] = log_vals
            render_scalars = '_log_scalars'
            scalar_bar_title += ' (log10)'
            # Percentile-based color limits to prevent outlier domination
            clim = [float(np.percentile(log_vals, 2)),
                    float(np.percentile(log_vals, 98))]

    sbar_args = dict(
        title=scalar_bar_title,
        title_font_size=14,
        label_font_size=12,
        n_labels=5,
        italic=False,
        fmt='%.2e',
        font_family='times',
        color='black',
        position_x=0.05,
        position_y=0.02,
        width=0.35,
        height=0.06,
    )

    add_kwargs = dict(
        scalars=render_scalars,
        cmap=cmap,
        show_edges=show_edges,
        edge_color='grey',
        opacity=1.0,
        scalar_bar_args=sbar_args,
    )
    if clim is not None:
        add_kwargs['clim'] = clim

    pl.add_mesh(mesh, **add_kwargs)

    if title:
        pl.add_text(title, position='upper_left', font_size=12,
                     color='black', font='times')

    # Camera — use explicit positions to avoid "view plane normal is parallel" VTK crash
    pl.reset_camera()
    _set_camera(pl, mesh.bounds, camera_preset)

    try:
        pl.enable_anti_aliasing('ssaa')
    except Exception:
        pass

    # Suppress VTK warning log cascade during screenshot
    _prev_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    try:
        pl.screenshot(output_path, transparent_background=False, return_img=False)
    finally:
        logging.getLogger().setLevel(_prev_level)
    pl.close()


# ── Public API ────────────────────────────────────────────────────────────

def render_fea_views(
    se_field,
    density,
    displacement_u,
    nelx, nely, nelz,
    pitch, origin,
    output_base,
    title='',
    density_threshold=0.5,
    views=('isometric', 'top', 'front'),
    window_size=(1920, 1080),
    show_edges=False,
    stress_field=None,
):
    """Render ANSYS-quality 3D FEA views and save as PNG.

    Creates one PNG per (field, view) combination.

    Args:
        se_field: (nely, nelx, nelz) per-element strain energy density.
        density: (nely, nelx, nelz) density (for thresholding active material).
        displacement_u: (ndof,) displacement vector, or None.
        nelx, nely, nelz: Element counts (Top3D convention).
        pitch: Voxel size in mm.
        origin: (3,) world-space origin.
        output_base: Path prefix (no extension).
        title: Optional title string.
        density_threshold: Hide cells below this density.
        views: Camera preset names.
        window_size: Render resolution (width, height).
        show_edges: Show element edge wireframe.
        stress_field: (nely, nelx, nelz) Von Mises stress, or None.

    Returns:
        list[str]: Paths to generated PNG files.
    """
    if not _PYVISTA_AVAILABLE:
        print('[VTK] pyvista not installed — skipping 3D FEA visualization.')
        print('[VTK] Install with: pip install pyvista')
        return []

    os.makedirs(os.path.dirname(output_base) or '.', exist_ok=True)
    output_paths = []

    # Build grid
    grid = _prepare_grid(nelx, nely, nelz, pitch, np.asarray(origin))
    grid.cell_data['density'] = _top3d_to_vtk_cell(density)
    grid.cell_data['strain_energy'] = _top3d_to_vtk_cell(se_field)

    # Von Mises stress (cell data)
    has_stress = stress_field is not None
    if has_stress:
        grid.cell_data['von_mises_stress'] = _top3d_to_vtk_cell(stress_field)

    # Displacement (point data)
    has_disp = displacement_u is not None and len(displacement_u) > 0
    if has_disp:
        try:
            disp = _displacement_to_vtk_point(displacement_u, nelx, nely, nelz)
            grid.point_data['displacement_magnitude'] = disp['magnitude']
        except Exception:
            has_disp = False

    # Threshold to active material only
    active = grid.threshold(density_threshold, scalars='density')
    if active.n_cells == 0:
        print('[VTK] No active cells after thresholding — skipping render.')
        return []

    # Render strain energy
    for view_name in views:
        cam = CAMERA_PRESETS.get(view_name, CAMERA_PRESETS['isometric'])
        path = f'{output_base}_se_{view_name}.png'
        view_title = f'{title} — Strain Energy ({view_name})' if title else f'Strain Energy ({view_name})'
        _render_single(active, 'strain_energy', cam, path, window_size,
                       cmap='turbo', log_scale=True,
                       scalar_bar_title='Strain Energy Density',
                       title=view_title, show_edges=show_edges)
        output_paths.append(path)

    # Render displacement magnitude
    if has_disp:
        for view_name in views:
            cam = CAMERA_PRESETS.get(view_name, CAMERA_PRESETS['isometric'])
            path = f'{output_base}_disp_{view_name}.png'
            view_title = f'{title} — Displacement ({view_name})' if title else f'Displacement ({view_name})'
            _render_single(active, 'displacement_magnitude', cam, path, window_size,
                           cmap='coolwarm', log_scale=False,
                           scalar_bar_title='Displacement (mm)',
                           title=view_title, show_edges=show_edges)
            output_paths.append(path)

    # Render Von Mises stress
    if has_stress:
        for view_name in views:
            cam = CAMERA_PRESETS.get(view_name, CAMERA_PRESETS['isometric'])
            path = f'{output_base}_vm_{view_name}.png'
            view_title = f'{title} — Von Mises Stress ({view_name})' if title else f'Von Mises Stress ({view_name})'
            _render_single(active, 'von_mises_stress', cam, path, window_size,
                           cmap='turbo', log_scale=True,
                           scalar_bar_title='Von Mises Stress (MPa)',
                           title=view_title, show_edges=show_edges)
            output_paths.append(path)

    print(f'[VTK] Saved {len(output_paths)} 3D FEA renders: {output_base}_*.png')
    return output_paths


def export_vtk(
    se_field,
    density,
    displacement_u,
    nelx, nely, nelz,
    pitch, origin,
    output_path,
    density_threshold=0.5,
    stress_field=None,
):
    """Export pre-thresholded active material as a VTK file for ParaView.

    Only cells with density >= threshold are included, so the geometry
    is immediately visible in ParaView without manual filtering.
    Exports .vtu (UnstructuredGrid) via PyVista if available, otherwise
    falls back to .vtr (full grid) via pyevtk.

    Args:
        se_field: (nely, nelx, nelz) strain energy density.
        density: (nely, nelx, nelz) density.
        displacement_u: (ndof,) displacement vector, or None.
        nelx, nely, nelz: Element counts.
        pitch: Voxel size in mm.
        origin: (3,) world origin.
        output_path: Base path (extension added automatically).
        density_threshold: Only export cells above this density.
        stress_field: (nely, nelx, nelz) Von Mises stress, or None.

    Returns:
        str: Path to the exported file, or None on failure.
    """
    # Prefer PyVista: threshold first, export only active cells as .vtu
    if _PYVISTA_AVAILABLE:
        grid = _prepare_grid(nelx, nely, nelz, pitch, np.asarray(origin))
        grid.cell_data['density'] = _top3d_to_vtk_cell(density)
        grid.cell_data['strain_energy'] = _top3d_to_vtk_cell(se_field)
        if stress_field is not None:
            grid.cell_data['von_mises_stress'] = _top3d_to_vtk_cell(stress_field)

        if displacement_u is not None and len(displacement_u) > 0:
            try:
                disp = _displacement_to_vtk_point(displacement_u, nelx, nely, nelz)
                grid.point_data['displacement_magnitude'] = disp['magnitude']
                grid.point_data['displacement_x'] = disp['ux']
                grid.point_data['displacement_y'] = disp['uy']
                grid.point_data['displacement_z'] = disp['uz']
            except Exception:
                pass

        active = grid.threshold(density_threshold, scalars='density')
        if active.n_cells == 0:
            print('[VTK] No active cells — skipping export.')
            return None

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        vtu_path = output_path + '.vtu'
        active.save(vtu_path)
        print(f'[VTK] Exported ({active.n_cells} active cells): {vtu_path}')
        return vtu_path

    # Fallback: pyevtk exports full grid as .vtr
    if _PYEVTK_AVAILABLE:
        ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
        x = np.arange(0, nelx + 1, dtype=np.float64) * pitch + ox
        y = np.arange(0, nely + 1, dtype=np.float64) * pitch + oy
        z = np.arange(0, nelz + 1, dtype=np.float64) * pitch + oz

        cellData = {
            'strain_energy': np.asfortranarray(se_field.transpose(1, 0, 2).astype(np.float64)),
            'density':       np.asfortranarray(density.transpose(1, 0, 2).astype(np.float64)),
        }
        if stress_field is not None:
            cellData['von_mises_stress'] = np.asfortranarray(stress_field.transpose(1, 0, 2).astype(np.float64))

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        vtr_path = gridToVTK(output_path, x, y, z, cellData=cellData)
        print(f'[VTK] Exported (full grid, threshold in ParaView): {vtr_path}')
        return vtr_path

    print('[VTK] Neither pyvista nor pyevtk installed — skipping export.')
    return None


def _render_frame_single(frame_mesh, scalars_name, cmap, sbar_title, sbar_fmt,
                         cam, view_title, path, window_size, nodes, node_tags, marker_r,
                         log_scale=False):
    """Render one frame view with given scalars and save as PNG."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('white')

    # Log-scale transform and percentile clim for strain energy
    render_scalars = scalars_name
    clim = None
    if log_scale and scalars_name in frame_mesh.cell_data:
        vals = frame_mesh.cell_data[scalars_name]
        valid = vals[vals > 0]
        if len(valid) > 0:
            vmin = np.percentile(valid, 1)
            log_vals = np.log10(np.clip(vals, vmin, None))
            frame_mesh.cell_data['_log_' + scalars_name] = log_vals
            render_scalars = '_log_' + scalars_name
            sbar_title += ' (log10)'
            clim = [float(np.percentile(log_vals, 2)),
                    float(np.percentile(log_vals, 98))]

    sbar_args = dict(
        title=sbar_title,
        title_font_size=14,
        label_font_size=12,
        n_labels=5,
        italic=False,
        fmt=sbar_fmt,
        font_family='times',
        color='black',
        position_x=0.05,
        position_y=0.02,
        width=0.35,
        height=0.06,
    )

    add_kwargs = dict(
        scalars=render_scalars,
        cmap=cmap,
        show_edges=False,
        opacity=1.0,
        scalar_bar_args=sbar_args,
    )
    if clim is not None:
        add_kwargs['clim'] = clim

    pl.add_mesh(frame_mesh, **add_kwargs)

    # BC markers
    if node_tags:
        for nid_str, tag in node_tags.items():
            nid = int(nid_str)
            if nid >= len(nodes):
                continue
            pos = nodes[nid]
            if tag == 1:  # fixed
                sphere = pv.Sphere(radius=marker_r, center=pos)
                pl.add_mesh(sphere, color='steelblue', opacity=0.9)
            elif tag == 2:  # loaded
                cone = pv.Cone(center=pos, direction=(0, -1, 0),
                               height=marker_r * 3, radius=marker_r * 1.2,
                               resolution=20)
                pl.add_mesh(cone, color='firebrick', opacity=0.9)

    if view_title:
        pl.add_text(view_title, position='upper_left', font_size=12,
                    color='black', font='times')

    pl.reset_camera()
    _set_camera(pl, frame_mesh.bounds, cam)

    try:
        pl.enable_anti_aliasing('ssaa')
    except Exception:
        pass

    _prev_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    try:
        pl.screenshot(path, transparent_background=False, return_img=False)
    finally:
        logging.getLogger().setLevel(_prev_level)
    pl.close()


def render_frame_views(
    nodes,
    edges,
    radii,
    output_base,
    title='',
    node_tags=None,
    beam_se=None,
    views=('isometric', 'top', 'front'),
    window_size=(1920, 1080),
):
    """Render the beam frame as smooth cylinders colored by radius and strain energy.

    Args:
        nodes: (N, 3) node positions.
        edges: (M, 2) edge connectivity (indices into nodes).
        radii: (M,) per-edge radius.
        output_base: Path prefix (no extension).
        title: Optional title string.
        node_tags: dict mapping node index (str or int) → tag (1=fixed, 2=loaded).
        beam_se: (M,) per-beam strain energy from beam FEM, or None.
        views: Camera preset names.
        window_size: Render resolution.

    Returns:
        list[str]: Paths to generated PNG files.
    """
    if not _PYVISTA_AVAILABLE:
        return []

    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)
    radii = np.asarray(radii, dtype=float)
    if beam_se is not None:
        beam_se = np.asarray(beam_se, dtype=float)

    frame_mesh = _build_frame_mesh(nodes, edges, radii, beam_se=beam_se)
    if frame_mesh is None:
        print('[VTK] No valid beams to render.')
        return []

    os.makedirs(os.path.dirname(output_base) or '.', exist_ok=True)
    output_paths = []

    # Marker size relative to mean radius
    marker_r = float(np.mean(radii[radii > 0])) * 0.6

    # Radius-coloured views
    for view_name in views:
        cam = CAMERA_PRESETS.get(view_name, CAMERA_PRESETS['isometric'])
        path = f'{output_base}_frame_{view_name}.png'
        view_title = f'{title} — Frame ({view_name})' if title else f'Frame ({view_name})'
        _render_frame_single(
            frame_mesh, 'radius', 'viridis', 'Beam Radius (mm)', '%.2f',
            cam, view_title, path, window_size, nodes, node_tags, marker_r)
        output_paths.append(path)

    # Strain energy-coloured views (beam FEM) — log scale + percentile clim
    if beam_se is not None and 'strain_energy' in frame_mesh.cell_data:
        for view_name in views:
            cam = CAMERA_PRESETS.get(view_name, CAMERA_PRESETS['isometric'])
            path = f'{output_base}_frame_se_{view_name}.png'
            view_title = f'{title} — Beam SE ({view_name})' if title else f'Beam SE ({view_name})'
            _render_frame_single(
                frame_mesh, 'strain_energy', 'inferno',
                'Beam Strain Energy (beam FEM)', '%.2e',
                cam, view_title, path, window_size, nodes, node_tags, marker_r,
                log_scale=True)
            output_paths.append(path)

    print(f'[VTK] Saved {len(output_paths)} frame renders: {output_base}_frame*.png')
    return output_paths


def export_frame_vtk(nodes, edges, radii, output_path):
    """Export the beam frame as a .vtu file with tube geometry for ParaView.

    Returns:
        str: Path to the exported file, or None on failure.
    """
    if not _PYVISTA_AVAILABLE:
        return None

    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)
    radii = np.asarray(radii, dtype=float)

    frame_mesh = _build_frame_mesh(nodes, edges, radii)
    if frame_mesh is None:
        return None

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    vtp_path = output_path + '_frame.vtp'
    frame_mesh.save(vtp_path)
    print(f'[VTK] Exported frame ({frame_mesh.n_cells} cells): {vtp_path}')
    return vtp_path


def render_pipeline_stage(
    stage_name,
    se_field,
    density,
    displacement_u,
    nelx, nely, nelz,
    pitch, origin,
    output_dir,
    base_name,
    density_threshold=0.5,
    export_vtr=False,
    render_3d=True,
    frame_data=None,
    stress_field=None,
):
    """One-call wrapper: render 3D views + optionally export for a pipeline stage.

    Args:
        stage_name: e.g. "SIMP Binary", "Final Optimised".
        se_field: (nely, nelx, nelz) strain energy density.
        density: (nely, nelx, nelz) density.
        displacement_u: (ndof,) or None.
        nelx, nely, nelz: Element counts.
        pitch: Voxel size in mm.
        origin: (3,) world origin.
        output_dir: Directory for output files.
        base_name: Base filename prefix.
        density_threshold: Hide cells below this.
        export_vtr: Whether to export VTK files for ParaView.
        render_3d: Whether to render 3D PNG views.
        frame_data: Optional dict with 'nodes', 'edges', 'radii', 'node_tags'
                    for parametric cylinder rendering.
        stress_field: (nely, nelx, nelz) Von Mises stress, or None.

    Returns:
        list[str]: All generated file paths.
    """
    safe_stage = stage_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    output_base = os.path.join(output_dir, f'{base_name}_3d_{safe_stage}')

    all_paths = []

    if render_3d:
        png_paths = render_fea_views(
            se_field, density, displacement_u,
            nelx, nely, nelz, pitch, np.asarray(origin),
            output_base,
            title=stage_name,
            density_threshold=density_threshold,
            stress_field=stress_field,
        )
        all_paths.extend(png_paths)

    # Parametric frame rendering (cylinders)
    if render_3d and frame_data is not None:
        frame_paths = render_frame_views(
            frame_data['nodes'], frame_data['edges'], frame_data['radii'],
            output_base,
            title=stage_name,
            node_tags=frame_data.get('node_tags'),
            beam_se=frame_data.get('beam_se'),
        )
        all_paths.extend(frame_paths)

    if export_vtr:
        vtr_path = export_vtk(
            se_field, density, displacement_u,
            nelx, nely, nelz, pitch, np.asarray(origin),
            output_base,
            stress_field=stress_field,
        )
        if vtr_path:
            all_paths.append(vtr_path)

        # Export frame geometry too
        if frame_data is not None:
            frame_vtu = export_frame_vtk(
                frame_data['nodes'], frame_data['edges'], frame_data['radii'],
                output_base,
            )
            if frame_vtu:
                all_paths.append(frame_vtu)

    return all_paths
