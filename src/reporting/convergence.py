"""
Dissertation-quality convergence plots and pipeline report generation.

Produces:
  - Figure (a): Top3D SIMP compliance convergence (red line)
  - Figure (b): Size + Layout optimisation compliance trajectory (blue line,
                stage markers, S/L timeline annotations)
  - Plain-text pipeline report (dissertation-ready statistics)
  - Machine-readable JSON report

All figures are saved as both PDF (vector, for dissertation) and PNG (preview).
Matplotlib is configured for Computer Modern-like serif fonts without requiring
a full LaTeX installation.
"""

import os
import json
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ---------------------------------------------------------------------------
# Global matplotlib style — Computer Modern serif, publication-ready
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':   'cm',
    'axes.labelsize':     11,
    'axes.titlesize':     12,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

_RED    = '#c0392b'
_BLUE   = '#2471a3'
_GREY   = '#95a5a6'
_DARK   = '#2c3e50'


# ---------------------------------------------------------------------------
# Figure (a) — Top3D convergence
# ---------------------------------------------------------------------------

def plot_top3d_convergence(compliance_history, output_path,
                           mesh_size=None, volfrac=None):
    """
    Plot SIMP topology-optimisation compliance convergence.

    Matches the reference figure: red line, clean axes, y-grid only.

    Parameters
    ----------
    compliance_history : list of float
        Compliance value recorded at each SIMP iteration.
    output_path : str
        Destination path (without extension — .pdf and .png are appended).
    mesh_size : tuple (nelx, nely, nelz), optional
        Used to annotate the subtitle.
    volfrac : float, optional
        Volume fraction used in the run.
    """
    if not compliance_history:
        print("[Report] Top3D history empty — skipping Figure (a).")
        return

    hist = [c for c in compliance_history if np.isfinite(c)]
    iters = list(range(1, len(hist) + 1))

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(iters, hist, color=_RED, linewidth=1.5, zorder=3)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost function')
    ax.set_xlim(0, max(iters))
    ax.set_ylim(bottom=0)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', color=_GREY, alpha=0.35, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Subtitle annotation
    parts = []
    if mesh_size:
        nelx, nely, nelz = mesh_size
        parts.append(f'{nelx}×{nely}×{nelz} mesh ({nelx*nely*nelz:,} elements)')
    if volfrac is not None:
        parts.append(f'$V_f$ = {volfrac:.2f}')
    if parts:
        ax.set_title(', '.join(parts), fontsize=8, color=_DARK, pad=4)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure (b) — Size + Layout optimisation convergence
# ---------------------------------------------------------------------------

def plot_size_layout_convergence(convergence_stages, output_path):
    """
    Plot combined Size (S) and Layout (L) compliance trajectory.

    Blue line, square markers only at the START of each stage (not at
    identical end/start transition pairs), compliance annotations with
    staggered vertical offsets, S/L timeline below x-axis.

    Parameters
    ----------
    convergence_stages : list of dict
        Each dict: {'type': 'size'|'layout', 'loop': int, 'history': list}
    output_path : str
        Destination path (without extension).
    """
    if not convergence_stages:
        print("[Report] No optimisation stages — skipping Figure (b).")
        return

    cum_x       = []
    cum_c       = []
    stage_starts = []   # (x, c) — first point of each stage only
    stage_spans  = []   # (x_start, x_end, label) for timeline
    offset = 0

    for stage in convergence_stages:
        hist = [c for c in stage['history'] if np.isfinite(c)]
        if not hist:
            continue
        label   = 'S' if stage['type'] == 'size' else 'L'
        x_start = offset
        stage_starts.append((offset, hist[0]))
        for i, c in enumerate(hist):
            cum_x.append(offset + i)
            cum_c.append(c)
        offset += len(hist)
        stage_spans.append((x_start, offset - 1, label))

    if not cum_x:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(bottom=0.16)

    ax.plot(cum_x, cum_c, color=_BLUE, linewidth=1.2, zorder=3)

    # ── Markers + annotations: stage STARTS only + final point ─────────────
    # Stagger both x and y to spread labels and avoid overlap
    _offsets = [(6, 15), (-25, 22), (6, 8), (-25, 28), (6, 18), (-25, 12)]
    all_annots = stage_starts + [(cum_x[-1], cum_c[-1])]
    for i, (x, c) in enumerate(all_annots):
        xo, yo = _offsets[i % len(_offsets)]
        ax.plot(x, c, 's', color=_BLUE, markersize=5, zorder=5,
                markeredgewidth=0.5, markeredgecolor='white')
        ax.annotate(f'{c:,.0f}',
                    xy=(x, c), xytext=(xo, yo),
                    textcoords='offset points',
                    fontsize=6.5, color=_DARK, rotation=30, va='bottom')

    # ── Vertical separators at stage start positions only ──────────────────
    for (x_start, _, _label) in stage_spans:
        if x_start > 0:
            ax.axvline(x_start, color=_GREY, linewidth=0.5,
                       linestyle='--', alpha=0.5, zorder=1)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Compliance')
    ax.set_xlim(0, max(cum_x))
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', color=_GREY, alpha=0.30, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # ── S / L timeline bar (inside graph, lower left with gap) ──────────────
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([])
    ax2.xaxis.set_visible(False)

    bar_y   = 0.10
    arrow_y = 0.12
    total_x = max(cum_x)

    ax.annotate('', xy=(1.0, arrow_y), xycoords='axes fraction',
                xytext=(0.01, arrow_y),
                arrowprops=dict(arrowstyle='->', color=_DARK, lw=0.8, mutation_scale=8))

    for (x_start, x_end, label) in stage_spans:
        x_frac       = (x_start + x_end) / 2.0 / total_x
        x_frac_start = x_start / total_x
        ax.annotate('', xy=(x_frac_start, arrow_y + 0.025),
                    xycoords='axes fraction',
                    xytext=(x_frac_start, arrow_y - 0.010),
                    arrowprops=dict(arrowstyle='-', color=_DARK, lw=0.6))
        ax.text(x_frac, bar_y - 0.025, label,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=8, color=_DARK, fontweight='bold')

    ax.annotate('', xy=(1.0, arrow_y + 0.025), xycoords='axes fraction',
                xytext=(1.0, arrow_y - 0.010),
                arrowprops=dict(arrowstyle='-', color=_DARK, lw=0.6))

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure (c) — Combined full-pipeline convergence
# ---------------------------------------------------------------------------

def plot_combined_convergence(top3d_hist, convergence_stages, c_baseline,
                               output_path, mesh_size=None, volfrac=None,
                               figsize=None):
    """
    Single figure showing the full pipeline compliance journey:
    SIMP topology optimisation → reconstruction → Size + Layout optimisation.

    SIMP compliance is pre-rescaled to E = 210 GPa by the pipeline (SIMP
    runs at E0=1000 for numerical stability; C ∝ 1/E, so values are scaled by
    1000/210000).  A small additional rescaling maps the SIMP final value to
    ``c_baseline`` (frame baseline) for visual continuity — the continuum
    model is geometrically different from the skeleton frame.

    Parameters
    ----------
    top3d_hist : list of float
        Per-iteration SIMP compliance.
    convergence_stages : list of dict
        {'type': 'size'|'layout', 'loop': int, 'history': list}
    c_baseline : float
        Frame FEM compliance of the reconstructed skeleton (pre-optimisation).
    output_path : str
        Destination path (without extension).
    mesh_size : tuple (nelx, nely, nelz), optional
    volfrac : float, optional
    figsize : tuple (width, height) in inches, optional
        Default (3.5, 2.6) — IEEE single-column width (88.9 mm).
    """
    hist_simp = [c for c in top3d_hist if np.isfinite(c)] if top3d_hist else []
    has_simp  = bool(hist_simp) and np.isfinite(c_baseline) and abs(c_baseline) > 0
    has_frame = any(
        [c for c in s['history'] if np.isfinite(c)] for s in convergence_stages
    ) if convergence_stages else False

    if not has_simp and not has_frame:
        print("[Report] No data for combined figure — skipping.")
        return

    if figsize is None:
        figsize = (3.5, 2.6)

    # Compact mode: scale fonts/markers when figure is narrow (e.g. IEEE column)
    compact = figsize[0] < 5.0
    _lw       = 1.0  if compact else 1.5
    _ms       = 3.5  if compact else 5
    _annot_fs = 5.0  if compact else 6.5
    _label_fs = 8    if compact else 11
    _tick_fs  = 6.5  if compact else 9
    _leg_fs   = 5.5  if compact else 7
    _title_fs = 6.5  if compact else 8
    _bar_fs   = 6    if compact else 8
    _recon_fs = 5.0  if compact else 6.5

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.20 if compact else 0.18)

    n_simp   = 0
    rescaled = []

    # ── SIMP section (rescaled so final = c_baseline) ──────────────────────
    if has_simp:
        scale    = c_baseline / hist_simp[-1]
        rescaled = [c * scale for c in hist_simp]
        n_simp   = len(rescaled)
        ax.plot(range(n_simp), rescaled, color=_RED, linewidth=_lw, zorder=3,
                label='Topology opt. (SIMP, E0=1000, rescaled to E=210 GPa)')

    # ── Frame FEM section ─────────────────────────────────────────────────
    cum_x        = []
    cum_c        = []
    frame_starts = []   # (x, c) first point of each frame stage
    stage_spans  = []
    offset       = n_simp

    if has_frame:
        for stage in convergence_stages:
            hist = [c for c in stage['history'] if np.isfinite(c)]
            if not hist:
                continue
            label   = 'S' if stage['type'] == 'size' else 'L'
            x_start = offset
            frame_starts.append((offset, hist[0]))
            for i, c in enumerate(hist):
                cum_x.append(offset + i)
                cum_c.append(c)
            offset += len(hist)
            stage_spans.append((x_start, offset - 1, label))
        if cum_x:
            ax.plot(cum_x, cum_c, color=_BLUE, linewidth=_lw, zorder=3,
                    label='Size + layout opt. (frame FEM, E = 210 GPa)')

    # ── Reconstruction separator ────────────────────────────────────────────
    all_x   = list(range(n_simp)) + cum_x
    total_x = max(all_x) if all_x else 1

    if n_simp > 0 and cum_x:
        ax.axvline(n_simp, color=_DARK, linewidth=0.9, linestyle=':', alpha=0.6, zorder=4)
        xf = n_simp / total_x
        ax.text(xf + 0.005, 0.97, 'Reconstruction',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=_recon_fs, color=_DARK, style='italic')

    # ── Markers + annotations (skip SIMP end = frame start to avoid duplication) ─
    # Stagger both x and y to spread labels and avoid overlap
    if compact:
        _offsets = [(4, 10), (-18, 14), (4, 6), (-18, 18), (4, 12), (-18, 8)]
    else:
        _offsets = [(6, 15), (-25, 22), (6, 8), (-25, 28), (6, 18), (-25, 12)]

    def _mark(x, c, colour, idx):
        xo, yo = _offsets[idx % len(_offsets)]
        ax.plot(x, c, 's', color=colour, markersize=_ms, zorder=5,
                markeredgewidth=0.5, markeredgecolor='white')
        ax.annotate(f'{c:,.0f}',
                    xy=(x, c), xytext=(xo, yo),
                    textcoords='offset points',
                    fontsize=_annot_fs, color=_DARK, rotation=30, va='bottom')

    idx = 0
    if rescaled:
        _mark(0, rescaled[0], _RED, idx);          idx += 1
    for x, c in frame_starts:
        _mark(x, c, _BLUE, idx); idx += 1
    if cum_x:
        _mark(cum_x[-1], cum_c[-1], _BLUE, idx)

    # ── Axes ────────────────────────────────────────────────────────────────
    ax.set_xlabel('Iterations', fontsize=_label_fs)
    ax.set_ylabel('Compliance (E = 210 GPa)', fontsize=_label_fs)
    ax.tick_params(labelsize=_tick_fs)
    if all_x:
        ax.set_xlim(0, max(all_x))
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', color=_GREY, alpha=0.30, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=_leg_fs, framealpha=0.9)

    parts = []
    if mesh_size:
        nx, ny, nz = mesh_size
        parts.append(f'{nx}×{ny}×{nz}')
    if volfrac is not None:
        parts.append(f'$V_f$={volfrac:.2f}')
    if parts:
        ax.set_title(', '.join(parts), fontsize=_title_fs, color=_DARK, pad=4)

    # ── Timeline bar (SIMP + S/L sections, inside graph, lower left) ──────
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([])
    ax2.xaxis.set_visible(False)

    bar_y   = 0.10
    arrow_y = 0.12

    ax.annotate('', xy=(1.0, arrow_y), xycoords='axes fraction',
                xytext=(0.01, arrow_y),
                arrowprops=dict(arrowstyle='->', color=_DARK, lw=0.8, mutation_scale=8))

    if n_simp > 0:
        ax.text((n_simp / 2) / total_x, bar_y - 0.025, 'SIMP',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=_bar_fs, color=_RED, fontweight='bold')
        xf_r = n_simp / total_x
        ax.annotate('', xy=(xf_r, arrow_y + 0.025), xycoords='axes fraction',
                    xytext=(xf_r, arrow_y - 0.010),
                    arrowprops=dict(arrowstyle='-', color=_DARK, lw=0.6))

    for (x_start, x_end, label) in stage_spans:
        x_frac       = (x_start + x_end) / 2.0 / total_x
        x_frac_start = x_start / total_x
        ax.annotate('', xy=(x_frac_start, arrow_y + 0.025),
                    xycoords='axes fraction',
                    xytext=(x_frac_start, arrow_y - 0.010),
                    arrowprops=dict(arrowstyle='-', color=_DARK, lw=0.6))
        ax.text(x_frac, bar_y - 0.025, label,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=_bar_fs, color=_DARK, fontweight='bold')

    ax.annotate('', xy=(1.0, arrow_y + 0.025), xycoords='axes fraction',
                xytext=(1.0, arrow_y - 0.010),
                arrowprops=dict(arrowstyle='-', color=_DARK, lw=0.6))

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Pipeline report
# ---------------------------------------------------------------------------

def generate_pipeline_report(report_data, output_path, print_to_console=True):
    """
    Generate a structured dissertation-ready pipeline report.

    Saves plain-text to *output_path* and a machine-readable JSON to
    *output_path* with '.json' extension.  Optionally prints to console.

    Parameters
    ----------
    report_data : dict
        Built by ``_build_report_data()`` in run_pipeline.py.
    output_path : str
        Destination .txt path.
    print_to_console : bool
        Mirror output to stdout.
    """
    lines = _format_report(report_data)
    text  = '\n'.join(lines)

    if print_to_console:
        print('\n' + text)

    with open(output_path, 'w') as f:
        f.write(text + '\n')

    json_path = os.path.splitext(output_path)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=_json_default)

    print(f"[Report] Saved: {output_path}")
    print(f"[Report] Saved: {json_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig, base_path, dpi=300):
    """Save figure as PDF + PNG + SVG, stripping any existing extension."""
    base = os.path.splitext(base_path)[0]
    pdf_path = base + '.pdf'
    png_path = base + '.png'
    svg_path = base + '.svg'
    fig.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"[Report] Saved figure: {pdf_path}")
    print(f"[Report] Saved figure: {png_path}")
    print(f"[Report] Saved figure (editable): {svg_path}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _pct(a, b):
    """Percentage change from a to b."""
    if a and abs(a) > 1e-12:
        return (b - a) / abs(a) * 100.0
    return 0.0


def _fmt(v, decimals=4):
    """Format a float with thousands separator; scientific for tiny values."""
    if v is None:
        return 'N/A'
    if v != 0 and abs(v) < 0.01:
        return f'{v:.{decimals + 1}e}'
    return f'{v:,.{decimals}f}'


def _format_report(d):
    W = 68
    sep = '=' * W
    thin = '-' * W

    lines = [
        sep,
        _centre('TO_CAD PIPELINE REPORT', W),
        _centre(d.get('problem_name', ''), W),
        sep,
        '',
    ]

    # ── 1. Topology Optimisation ─────────────────────────────────────────
    t = d.get('top3d', {})
    if t:
        lines += [
            '1. TOPOLOGY OPTIMISATION (SIMP — Top3D)',
            thin,
        ]
        if t.get('mesh_size'):
            nx, ny, nz = t['mesh_size']
            n_elem = nx * ny * nz
            lines.append(f"   Mesh:                  {nx} × {ny} × {nz}  ({n_elem:,} elements)")
        _add_kv(lines, 'Volume fraction',     t.get('volfrac'),      fmt='.2f')
        _add_kv(lines, 'Penalisation (p)',    t.get('penal'),        fmt='.1f')
        _add_kv(lines, 'Filter radius (r)',   t.get('rmin'),         fmt='.1f')
        iters = t.get('iterations')
        max_loop = t.get('max_loop')
        conv = t.get('converged', True)
        if iters is not None:
            status = 'converged' if conv else 'max iterations reached'
            loop_str = f'{iters}' + (f' / {max_loop}' if max_loop else '')
            lines.append(f"   Iterations:            {loop_str}  ({status})")
        _add_kv(lines, 'Initial compliance',  t.get('c_initial'),   fmt=',.4f')
        _add_kv(lines, 'Final compliance',    t.get('c_final'),     fmt=',.4f')
        if t.get('c_initial') and t.get('c_final'):
            lines.append(f"   Reduction:             {_pct(t['c_initial'], t['c_final']):+.2f}%")
        lines.append(f"   {'Note:':<24} SIMP runs at E0=1000 (topology is E-invariant).")
        lines.append(f"   {'':24} Compliance rescaled to E=210 GPa for comparison")
        lines.append(f"   {'':24} with frame FEM (§3). Continuum vs frame model differs.")
        lines.append('')

    # ── 2. Skeleton Reconstruction ────────────────────────────────────────
    r = d.get('reconstruction', {})
    if r:
        lines += [
            '2. SKELETON RECONSTRUCTION (Stage 1)',
            thin,
        ]
        _add_kv(lines, 'Solid voxels',        r.get('solid_voxels'), fmt=',')
        _add_kv(lines, 'Skeleton voxels',     r.get('skeleton_voxels'), fmt=',')
        n_nodes = r.get('nodes')
        n_edges = r.get('edges')
        if n_nodes is not None:
            lines.append(f"   Nodes / Edges:         {n_nodes} / {n_edges}")
        n_plates = r.get('plates')
        if n_plates is not None:
            zone = r.get('zone_stats', {})
            pv = zone.get('plate_voxels', '')
            bv = zone.get('beam_voxels', '')
            detail = f'  ({pv} plate + {bv} beam skeleton voxels)' if pv != '' else ''
            lines.append(f"   Plates:                {n_plates} region(s){detail}")
        _add_kv(lines, 'Target volume',        r.get('target_volume'), fmt=',.4f', unit='mm³')

        # Graph processing stages
        stages = r.get('graph_stages', [])
        if stages:
            lines.append('')
            lines.append('   Graph processing stages:')
            lines.append(f"     {'Stage':<26} {'Nodes':>6}  {'Edges':>6}")
            lines.append('     ' + '─' * 42)
            for s in stages:
                lines.append(f"     {s['name']:<26} {s['nodes']:>6}  {s['edges']:>6}")
        lines.append('')

    # ── 3. Size + Layout Optimisation ────────────────────────────────────
    loops = d.get('optimization_loops', [])
    if loops:
        lines += [
            '3. SIZE + LAYOUT OPTIMISATION  (beam frame FEM, E=210 GPa)',
            thin,
        ]
        # Table header
        col = '  {:<5} {:<8} {:>6}  {:>14}  {:>14}  {:>8}'
        lines.append(col.format('Loop', 'Stage', 'Iters', 'C_init', 'C_final', 'Δ%'))
        lines.append('  ' + '-' * 61)
        for lp in loops:
            loop_n = lp.get('loop', '?')
            for stage_key, stage_label in [('size', 'Size'), ('layout', 'Layout')]:
                s = lp.get(stage_key)
                if s is None:
                    continue
                iters  = s.get('iterations', '?')
                ci     = s.get('c_initial', 0.0)
                cf     = s.get('c_final', 0.0)
                dp     = _pct(ci, cf)
                lines.append(col.format(
                    loop_n, stage_label, iters,
                    _fmt(ci, 4), _fmt(cf, 4), f'{dp:+.1f}%'))
        lines.append('')

        # Radius statistics from last size stage
        last_size = None
        for lp in reversed(loops):
            if lp.get('size'):
                last_size = lp['size']
                break
        if last_size and last_size.get('radius_min') is not None:
            lines.append('   Radii (final size stage):')
            lines.append(f"     min={_fmt(last_size['radius_min'],2)}  "
                         f"max={_fmt(last_size['radius_max'],2)}  "
                         f"mean={_fmt(last_size['radius_mean'],2)}  "
                         f"std={_fmt(last_size.get('radius_std'),2)} mm")

        # Node displacement from last layout stage
        last_layout = None
        for lp in reversed(loops):
            if lp.get('layout'):
                last_layout = lp['layout']
                break
        if last_layout and last_layout.get('max_node_disp') is not None:
            lines.append('   Node displacement (cumulative layout):')
            lines.append(f"     max={_fmt(last_layout['max_node_disp'],2)}  "
                         f"mean={_fmt(last_layout.get('mean_node_disp'),2)} mm")
        lines.append('')

    # ── 4. Overall Pipeline Summary ───────────────────────────────────────
    ov = d.get('overall', {})
    lines += [
        '4. PIPELINE SUMMARY',
        thin,
    ]
    t3d = d.get('top3d', {})
    if t3d.get('c_final') is not None:
        _add_kv(lines, 'SIMP final C (rescaled)*', t3d['c_final'], fmt=',.4f')
    _add_kv(lines, 'Reconstructed skeleton C', ov.get('baseline_compliance'), fmt=',.4f')
    _add_kv(lines, 'Optimised frame C',       ov.get('final_compliance'),    fmt=',.4f')
    if ov.get('baseline_compliance') and ov.get('final_compliance'):
        total = _pct(ov['baseline_compliance'], ov['final_compliance'])
        lines.append(f"   Frame opt. reduction:      {total:+.2f}%  (frame FEM, E=210 GPa)")
    _add_kv(lines, 'Volume target',         ov.get('volume_target'),    fmt=',.4f', unit='mm³')
    _add_kv(lines, 'Volume achieved',       ov.get('volume_final'),     fmt=',.4f', unit='mm³')
    if ov.get('volume_error_pct') is not None:
        lines.append(f"   Volume constraint error:  {ov['volume_error_pct']:.2f}%")
    if ov.get('geometric_similarity') is not None:
        lines.append(f"   Geometric similarity:     {ov['geometric_similarity']:.3f}")
    if t3d.get('c_final') is not None:
        lines.append('')
        lines.append(f"   * SIMP uses continuum FEM (different model to frame FEM).")
        lines.append(f"     Values are E-rescaled but not directly comparable.")

    # ── 5. Direct Compliance Comparison (Yin et al. 2020) ─────────────
    cont = d.get('continuum', {})
    if cont.get('simp_p1_rescaled') is not None:
        lines += ['', '5. DIRECT COMPLIANCE COMPARISON  (Yin et al. 2020)', thin]
        lines.append('   J = f·u at E = 210 GPa.  SIMP re-evaluated with p=1 on')
        lines.append('   binary field (no penalisation artefact).')
        lines.append('')

        c_simp_210k = cont['simp_p1_rescaled']

        col = '   {:<30} {:>14}  {:>10}'
        lines.append(col.format('Stage', 'C (f·u)', 'Δ vs SIMP'))
        lines.append('   ' + '-' * 57)
        lines.append(col.format('SIMP binary (hex FEM, p=1)',
                                _fmt(c_simp_210k, 4), '—'))

        for stage in cont.get('yin_stages', []):
            c_st = stage['compliance']
            pct = _pct(c_simp_210k, c_st)
            lines.append(col.format(stage['label'],
                                    _fmt(c_st, 4),
                                    f'{pct:+.1f}%'))

        lines.append('')
        lines.append('   Note: SIMP uses continuum hex FEM, frame uses beam FEM.')
        lines.append('   Both compute J = f·u (work done by loads) — physically')
        lines.append('   meaningful comparison per Yin et al. (2020).')

    lines += ['', sep]
    return lines


def _centre(text, width):
    return text.center(width)


def _add_kv(lines, label, value, fmt='.2f', unit=''):
    if value is None:
        return
    if fmt == ',':
        val_str = f'{int(value):,}'
    elif value != 0 and abs(value) < 0.01 and 'f' in fmt:
        # Adaptive: switch to scientific for tiny floats
        val_str = f'{value:.3e}'
    elif fmt.startswith(',.'):
        val_str = f'{value:{fmt}}'
    else:
        val_str = f'{value:{fmt}}'
    suffix = f' {unit}' if unit else ''
    lines.append(f"   {label + ':':<24} {val_str}{suffix}")
