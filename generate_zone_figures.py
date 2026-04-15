#!/usr/bin/env python3
"""Generate publication-quality zone classification figures for dissertation.

Produces:
  - 4 individual zone classification PNGs (one per test case)
  - 1 combined 2x2 figure
  - 1 hybrid pipeline progression figure (4-panel horizontal)

All output goes to figures/Results/Zone_Classification/
"""

import sys
import numpy as np
from pathlib import Path

# Ensure repo root on path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from src.pipelines.baseline_yin.thinning import thin_grid_yin
from src.pipelines.baseline_yin.graph import (
    classify_skeleton_post_thinning, extract_graph,
)
from src.pipelines.baseline_yin.plate_extraction import (
    recover_plate_regions_from_skeleton,
)
from src.pipelines.baseline_yin.postprocessing import (
    collapse_short_edges, merge_colocated_nodes, prune_branches,
)

# ── Colour constants ──────────────────────────────────────────────
CYAN       = "#00CCCC"   # surface / plate zones
RED        = "#CC0000"   # beam zones
GREY       = "#888888"   # original solid
CYAN_SOLID = "#00AAAA"   # recovered plate solid

OUT_DIR = REPO / "figures" / "Results" / "Zone_Classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Test case definitions ─────────────────────────────────────────
TEST_CASES = [
    {
        "name":     "Clear_Beam_Plate",
        "npz":      "output/hybrid_v2/Clear_Beam_Plate_Test_top3d.npz",
        "label":    "(a) Clear Beam–Plate",
        "out":      "zone_clear_beam_plate.png",
        "expected": "mixed",
        "elev": 30, "azim": -50,
    },
    {
        "name":     "Thick_Beam",
        "npz":      "output/hybrid_v2/full_control_beam_top3d.npz",
        "label":    "(b) Thick Beam",
        "out":      "zone_thick_beam.png",
        "expected": "beam",
        "elev": 35, "azim": -65,
    },
    {
        "name":     "Cantilever",
        "npz":      "output/hybrid_v2/matlab_replicated_top3d.npz",
        "label":    "(c) Cantilever",
        "out":      "zone_cantilever.png",
        "expected": "beam",
        "elev": 30, "azim": -55,
        "min_plate_size": 30,  # large domain, suppress thin-z artefacts
    },
    {
        "name":     "Frame_Supported_Plate",
        "npz":      "output/hybrid_v2/Roof_Structure_Test_top3d.npz",
        "label":    "(d) Frame-Supported Plate",
        "out":      "zone_frame_supported_plate.png",
        "expected": "mixed",
        "elev": 28, "azim": -50,
    },
]


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def load_npz(path):
    """Load Top3D npz, return (rho, bc_tags, pitch, origin)."""
    d = np.load(path, allow_pickle=True)
    rho = d["rho"]
    bc_tags = d["bc_tags"].astype(np.int32) if "bc_tags" in d else np.zeros_like(rho, dtype=np.int32)
    pitch = float(d["pitch"]) if "pitch" in d else 1.0
    origin = np.array(d["origin"], dtype=float) if "origin" in d else np.zeros(3)
    return rho, bc_tags, pitch, origin


def run_classification(rho, bc_tags, vol_thresh=0.3, min_plate_size=None):
    """Two-pass thinning + post-thinning zone classification.

    Returns (solid, skeleton, skeleton_curve, zone_mask, plate_labels, zone_stats, edt).
    """
    solid = (rho > vol_thresh).astype(np.uint8)
    edt = distance_transform_edt(solid)

    # For thin structures (nelz ≤ 6), raise min_plate_size to suppress
    # spurious plate regions from boundary artefacts in the z-dimension.
    if min_plate_size is None:
        nelz = rho.shape[2]
        min_plate_size = 10 if nelz <= 6 else 3

    print("    Thinning mode=3 (surfaces + curves)...")
    skel3 = thin_grid_yin(solid.copy(), tags=bc_tags, max_iters=50, mode=3, edt=edt)

    print("    Thinning mode=0 (curves only)...")
    skel0 = thin_grid_yin(solid.copy(), tags=bc_tags, max_iters=50, mode=0, edt=edt)

    print(f"    Classifying zones (min_plate_size={min_plate_size})...")
    zone_mask, plate_labels, stats = classify_skeleton_post_thinning(
        skel3,
        min_plate_size=min_plate_size,
        flatness_ratio=3.0,
        skeleton_curve=skel0,
        solid=solid,
    )
    return solid, skel3, skel0, zone_mask, plate_labels, stats, edt


def _voxel_xyz(mask, pitch=1.0, origin=None):
    """Convert a binary/int mask to (N,3) world-space XYZ.

    Array convention is (Y, X, Z) — reorder to (X, Y, Z).
    """
    if origin is None:
        origin = np.zeros(3)
    idx = np.argwhere(mask > 0)
    if len(idx) == 0:
        return np.empty((0, 3))
    return origin + idx[:, [1, 0, 2]].astype(float) * pitch + pitch * 0.5


def _data_extents(pts_list):
    """Return (lo, hi, mid, half_spans) from a list of point arrays."""
    all_pts = np.vstack([p for p in pts_list if len(p) > 0])
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    mid = (lo + hi) / 2
    spans = hi - lo
    return lo, hi, mid, spans


def _set_axes_equal_data(ax, pts_list):
    """Set axis limits proportional to actual data extents (not cubic)."""
    valid = [p for p in pts_list if len(p) > 0]
    if not valid:
        return
    lo, hi, mid, spans = _data_extents(valid)
    half = spans / 2 * 1.15  # 15% padding
    # Enforce a minimum span so very thin dimensions don't collapse
    min_half = max(spans.max() * 0.08, 0.5)
    half = np.maximum(half, min_half)
    ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
    ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
    ax.set_zlim(mid[2] - half[2], mid[2] + half[2])
    # Aspect ratio matching data (not cubic)
    ax.set_box_aspect(half / half.max())


def _clean_3d_axis(ax, elev=25, azim=-60):
    """Remove ticks, labels, grid from a 3D axis."""
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)


def _auto_marker_size(n_pts, fig_inches_sq=16.0):
    """Compute scatter marker size that fills the figure well.

    Targets ~30-50% fill of projected figure area.
    """
    if n_pts <= 0:
        return 4.0
    # Heuristic: scale inversely with sqrt(n), tuned for skeleton voxels
    base = fig_inches_sq * 72 * 72  # points² at 72 ppi
    s = min(40.0, max(2.0, base * 0.06 / max(n_pts, 50)))
    return s


# ── Main zone scatter renderer ────────────────────────────────────

def render_zone_scatter(ax, zone_mask, pitch=1.0, origin=None,
                        elev=25, azim=-60, marker_size=None,
                        fig_inches_sq=16.0):
    """Render zone_mask onto a 3D matplotlib axis using scatter."""
    plate_pts = _voxel_xyz(zone_mask == 1, pitch, origin)
    beam_pts  = _voxel_xyz(zone_mask == 2, pitch, origin)

    total = len(plate_pts) + len(beam_pts)
    if marker_size is None:
        marker_size = _auto_marker_size(total, fig_inches_sq)

    if len(plate_pts) > 0:
        ax.scatter(plate_pts[:, 0], plate_pts[:, 1], plate_pts[:, 2],
                   c=CYAN, s=marker_size, marker="s", linewidths=0,
                   alpha=0.90, label="Surface", rasterized=True,
                   depthshade=True)
    if len(beam_pts) > 0:
        ax.scatter(beam_pts[:, 0], beam_pts[:, 1], beam_pts[:, 2],
                   c=RED, s=marker_size, marker="s", linewidths=0,
                   alpha=0.90, label="Beam", rasterized=True,
                   depthshade=True)

    _clean_3d_axis(ax, elev, azim)

    all_pts = [p for p in [plate_pts, beam_pts] if len(p) > 0]
    if all_pts:
        _set_axes_equal_data(ax, all_pts)


def render_solid_scatter(ax, solid, pitch=1.0, origin=None,
                         colour=GREY, elev=25, azim=-60,
                         marker_size=None, alpha=0.18):
    """Render a binary solid as a faint 3D scatter."""
    pts = _voxel_xyz(solid, pitch, origin)
    if marker_size is None:
        marker_size = _auto_marker_size(len(pts), 16.0) * 0.6
    if len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=colour, s=marker_size, marker="s", linewidths=0,
                   alpha=alpha, rasterized=True, depthshade=True)
    _clean_3d_axis(ax, elev, azim)
    if len(pts) > 0:
        _set_axes_equal_data(ax, [pts])


# ══════════════════════════════════════════════════════════════════
#  Individual figure rendering
# ══════════════════════════════════════════════════════════════════

def render_single_figure(zone_mask, pitch, origin, out_path, title=None,
                         elev=25, azim=-60):
    """Save a single zone classification figure."""
    fig = plt.figure(figsize=(4, 4), dpi=300, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    render_zone_scatter(ax, zone_mask, pitch, origin,
                        elev=elev, azim=azim, fig_inches_sq=16.0)

    if title:
        ax.set_title(title, fontsize=11, pad=-2, fontweight="medium")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════
#  Combined 2x2 figure
# ══════════════════════════════════════════════════════════════════

def render_combined_figure(results, out_path):
    """Render all 4 test cases in a 2x2 grid."""
    fig = plt.figure(figsize=(8, 8), dpi=300, facecolor="white")

    for i, (tc, data) in enumerate(results):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d", facecolor="white")
        elev = tc.get("elev", 25)
        azim = tc.get("azim", -60)
        render_zone_scatter(ax, data["zone_mask"], data["pitch"], data["origin"],
                            elev=elev, azim=azim, fig_inches_sq=16.0)
        ax.set_title(tc["label"], fontsize=11, pad=-2, fontweight="medium")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.96,
                        wspace=0.0, hspace=0.08)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved combined figure: {out_path}")


# ══════════════════════════════════════════════════════════════════
#  Hybrid pipeline progression figure (roof case only)
# ══════════════════════════════════════════════════════════════════

def render_pipeline_progression(solid, zone_mask, plate_labels, edt,
                                pitch, origin, skeleton, out_path):
    """4-panel horizontal: solid → zone skeleton → recovered plates → beam graph + plates."""
    fig = plt.figure(figsize=(14, 4.8), dpi=300, facecolor="white")
    elev, azim = 28, -50

    # Pre-compute shared data
    recovered_zone, _ = recover_plate_regions_from_skeleton(
        zone_mask, plate_labels, solid.astype(bool), edt)
    plate_full_pts = _voxel_xyz(recovered_zone == 1, pitch, origin)
    beam_skel_pts  = _voxel_xyz(zone_mask == 2, pitch, origin)
    solid_pts = _voxel_xyz(solid, pitch, origin)

    # Shared limits from the solid extent (so all panels have same framing)
    all_extent_pts = [solid_pts]
    lo, hi, mid, spans = _data_extents(all_extent_pts)
    half = spans / 2 * 1.15
    min_half = max(spans.max() * 0.08, 0.5)
    half = np.maximum(half, min_half)

    def _apply_limits(ax):
        ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
        ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
        ax.set_zlim(mid[2] - half[2], mid[2] + half[2])
        ax.set_box_aspect(half / half.max())

    ms_solid = _auto_marker_size(len(solid_pts), 16.0) * 0.5
    ms_skel  = _auto_marker_size(len(beam_skel_pts) + len(plate_full_pts), 16.0)

    # ── (a) Original binary solid ─────────────────────────────────
    ax1 = fig.add_subplot(1, 4, 1, projection="3d", facecolor="white")
    if len(solid_pts) > 0:
        ax1.scatter(solid_pts[:, 0], solid_pts[:, 1], solid_pts[:, 2],
                    c=GREY, s=ms_solid, marker="s", linewidths=0,
                    alpha=0.18, rasterized=True, depthshade=True)
    _clean_3d_axis(ax1, elev, azim)
    _apply_limits(ax1)
    ax1.set_title("(a) Binary solid", fontsize=10, pad=-2, fontweight="medium")

    # ── (b) Zone-classified skeleton ──────────────────────────────
    ax2 = fig.add_subplot(1, 4, 2, projection="3d", facecolor="white")
    render_zone_scatter(ax2, zone_mask, pitch, origin,
                        elev=elev, azim=azim, fig_inches_sq=16.0)
    _apply_limits(ax2)
    ax2.set_title("(b) Classified skeleton", fontsize=10, pad=-2, fontweight="medium")

    # ── (c) Recovered plate regions + beam skeleton ───────────────
    ax3 = fig.add_subplot(1, 4, 3, projection="3d", facecolor="white")
    if len(plate_full_pts) > 0:
        ax3.scatter(plate_full_pts[:, 0], plate_full_pts[:, 1], plate_full_pts[:, 2],
                    c=CYAN_SOLID, s=ms_skel, marker="s", linewidths=0,
                    alpha=0.45, rasterized=True, depthshade=True)
    if len(beam_skel_pts) > 0:
        ax3.scatter(beam_skel_pts[:, 0], beam_skel_pts[:, 1], beam_skel_pts[:, 2],
                    c=RED, s=ms_skel * 1.3, marker="s", linewidths=0,
                    alpha=0.90, rasterized=True, depthshade=True)
    _clean_3d_axis(ax3, elev, azim)
    _apply_limits(ax3)
    ax3.set_title("(c) Recovered plates + beams", fontsize=10, pad=-2, fontweight="medium")

    # ── (d) Final beam graph overlaid on plate mesh ───────────────
    ax4 = fig.add_subplot(1, 4, 4, projection="3d", facecolor="white")

    # Extract and clean beam graph
    beam_only_skel = np.zeros_like(zone_mask, dtype=np.uint8)
    beam_only_skel[zone_mask == 2] = 1
    bc_tags_empty = np.zeros_like(zone_mask, dtype=np.int32)

    try:
        nodes_raw, edges_raw, _, node_tags = extract_graph(
            beam_only_skel, pitch,
            origin if origin is not None else np.zeros(3),
            tags=bc_tags_empty, hybrid_mode=True)
        # Convert to lists for postprocessing compatibility
        nodes = list(nodes_raw) if isinstance(nodes_raw, np.ndarray) else list(nodes_raw)
        edges = list(edges_raw)
        # Clean up: collapse short edges → prune → merge
        try:
            if len(edges) > 0:
                nodes, edges = collapse_short_edges(
                    nodes, edges, threshold=pitch * 3.5, node_tags=node_tags)
                nodes, edges = prune_branches(
                    nodes, edges, min_len=pitch * 2.0, node_tags=node_tags)
                nodes, edges = merge_colocated_nodes(
                    nodes, edges, node_tags=node_tags, tol=pitch * 0.5)
        except Exception:
            # Fall back to raw graph if cleaning fails
            nodes, edges = list(nodes_raw), list(edges_raw)
        nodes = np.array(nodes) if not isinstance(nodes, np.ndarray) else nodes
    except Exception as e:
        print(f"    [Warning] Graph extraction failed: {e}")
        nodes, edges = np.empty((0, 3)), []

    # Faint plate fill
    if len(plate_full_pts) > 0:
        ax4.scatter(plate_full_pts[:, 0], plate_full_pts[:, 1], plate_full_pts[:, 2],
                    c=CYAN_SOLID, s=ms_skel * 0.7, marker="s", linewidths=0,
                    alpha=0.20, rasterized=True, depthshade=True)

    # Beam graph edges
    if len(nodes) > 0 and len(edges) > 0:
        for edge in edges:
            u, v = int(edge[0]), int(edge[1])
            if u < len(nodes) and v < len(nodes):
                xs = [nodes[u, 0], nodes[v, 0]]
                ys = [nodes[u, 1], nodes[v, 1]]
                zs = [nodes[u, 2], nodes[v, 2]]
                ax4.plot(xs, ys, zs, color=RED, linewidth=2.0, alpha=0.9)
        ax4.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                    c=RED, s=20, marker="o", linewidths=0.5,
                    edgecolors="black", alpha=1.0, zorder=5)

    _clean_3d_axis(ax4, elev, azim)
    _apply_limits(ax4)
    ax4.set_title("(d) Beam graph + plate mesh", fontsize=10, pad=-2, fontweight="medium")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94,
                        wspace=-0.05, hspace=0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved pipeline progression: {out_path}")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Zone Classification Figure Generator")
    print("=" * 70)

    results = []

    for tc in TEST_CASES:
        npz_path = REPO / tc["npz"]
        print(f"\n{'─' * 50}")
        print(f"  {tc['name']}")
        print(f"  NPZ: {npz_path}")

        if not npz_path.exists():
            print(f"  [SKIP] NPZ not found: {npz_path}")
            continue

        rho, bc_tags, pitch, origin = load_npz(npz_path)
        shape = rho.shape
        solid_count = int((rho > 0.3).sum())
        print(f"  Shape: {shape} (nely={shape[0]}, nelx={shape[1]}, nelz={shape[2]})")
        print(f"  Solid voxels: {solid_count}")

        solid, skel3, skel0, zone_mask, plate_labels, stats, edt = \
            run_classification(rho, bc_tags,
                               min_plate_size=tc.get("min_plate_size"))

        n_plate = stats.get("n_plate_voxels", int((zone_mask == 1).sum()))
        n_beam  = stats.get("n_beam_voxels",  int((zone_mask == 2).sum()))
        n_skel  = int((zone_mask > 0).sum())
        print(f"  Skeleton: {n_skel} voxels")
        print(f"  Classification: {n_plate} surface, {n_beam} beam "
              f"({stats.get('n_plate_regions', '?')} plate regions)")
        print(f"  Expected: {tc['expected']}")

        out_single = OUT_DIR / tc["out"]
        render_single_figure(zone_mask, pitch, origin, out_single,
                             title=tc["label"],
                             elev=tc.get("elev", 25),
                             azim=tc.get("azim", -60))

        results.append((tc, {
            "zone_mask": zone_mask,
            "plate_labels": plate_labels,
            "pitch": pitch,
            "origin": origin,
            "solid": solid,
            "skeleton": skel3,
            "edt": edt,
        }))

    # ── Combined 2x2 figure ───────────────────────────────────────
    if len(results) > 0:
        print(f"\n{'─' * 50}")
        print("  Generating combined 2x2 figure...")
        render_combined_figure(results, OUT_DIR / "zone_classification_2x2.png")

    # ── Pipeline progression (roof case only) ─────────────────────
    roof_data = None
    for tc, data in results:
        if tc["name"] == "Frame_Supported_Plate":
            roof_data = data
            break

    if roof_data is not None:
        print(f"\n{'─' * 50}")
        print("  Generating hybrid pipeline progression figure...")
        render_pipeline_progression(
            roof_data["solid"],
            roof_data["zone_mask"],
            roof_data["plate_labels"],
            roof_data["edt"],
            roof_data["pitch"],
            roof_data["origin"],
            roof_data["skeleton"],
            OUT_DIR / "hybrid_pipeline_progression.png",
        )

    print(f"\n{'=' * 70}")
    print("  Done. Figures saved to:")
    print(f"    {OUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
