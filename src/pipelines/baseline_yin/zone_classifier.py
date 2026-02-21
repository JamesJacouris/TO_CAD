"""
Zone Classifier: Pre-thinning classification of solid voxels into beam vs plate zones.

Uses a multi-signal scoring system combining:
1. PCA eigenvalue analysis (planarity/linearity)
2. Aspect ratio analysis (2D spread vs 1D elongation)
3. BC tag density (load vs support regions)
4. EDT uniformity (plates have uniform thickness)

This classification happens BEFORE thinning to avoid the false positives
that occur with post-thinning surface detection.
"""

import numpy as np
from scipy.ndimage import label, binary_closing, binary_opening, binary_dilation


def classify_zones(solid, edt, plate_threshold, pitch, min_region_ratio=0.01,
                   planarity_thresh=0.65, linearity_thresh=0.55, bc_tags=None):
    """
    Pre-thinning classification of solid voxels into beam zones vs plate zones.

    Parameters
    ----------
    solid : ndarray (D, H, W), bool
        Binary solid mask from topology optimization.
    edt : ndarray (D, H, W), float
        Euclidean distance transform of solid (in voxel units).
    plate_threshold : float
        Max EDT value (voxels) for plate candidates.
    pitch : float
        Voxel size in mm.
    min_region_ratio : float
        Minimum region size as fraction of largest domain face area.
    planarity_thresh : float
        PCA planarity ratio threshold (used as baseline, scoring adjusts dynamically).
    linearity_thresh : float
        PCA linearity ratio threshold (used as baseline, scoring adjusts dynamically).
    bc_tags : ndarray (D, H, W), int32, optional
        BC tag grid (0=none, 1=fixed/support, 2=loaded). If None, BC signal is disabled.

    Returns
    -------
    zone_mask : ndarray (D, H, W), int32
        0=background, 1=plate, 2=beam
    plate_labels : ndarray (D, H, W), int32
        Labeled plate regions (0=not plate, 1..N=plate IDs).
    zone_stats : list of dict
        Per-region classification statistics.
    """
    D, H, W = solid.shape
    s26 = np.ones((3, 3, 3), dtype=np.int32)

    # Step 1: Label ALL connected components in solid
    solid_labeled, n_regions = label(solid, structure=s26)

    # Step 2: Filter small regions
    face_areas = sorted([D * H, D * W, H * W], reverse=True)
    min_voxels = max(20, int(face_areas[0] * min_region_ratio))

    region_ids, region_sizes = np.unique(solid_labeled[solid_labeled > 0], return_counts=True)
    small_regions = region_ids[region_sizes < min_voxels]
    for rid in small_regions:
        solid_labeled[solid_labeled == rid] = 0

    # Step 3: Multi-signal classification per surviving region
    zone_mask = np.zeros_like(solid, dtype=np.int32)
    zone_mask[solid] = 2  # Default: everything solid is beam

    plate_mask = np.zeros_like(solid, dtype=np.int32)
    zone_stats = []

    surviving_ids = np.unique(solid_labeled[solid_labeled > 0])

    bc_info = "with BC tags" if bc_tags is not None else "no BC tags"
    print(f"    [ZoneClassifier] {len(surviving_ids)} regions to classify "
          f"(plate_threshold={plate_threshold:.1f} voxels, min_size={min_voxels}, {bc_info})")

    for rid in surviving_ids:
        region_voxels = np.argwhere(solid_labeled == rid)
        n_voxels = len(region_voxels)
        region_mask_bool = solid_labeled == rid

        if n_voxels > 2000:
            # Large region: use sub-region analysis with multi-signal scoring
            classification, stats = _subregion_analysis(
                region_mask_bool, solid, edt, plate_threshold,
                planarity_thresh, linearity_thresh, bc_tags=bc_tags
            )
            zone_stats.append({"id": int(rid), "n_voxels": n_voxels,
                               "classification": classification, **stats})
            if classification == "plate":
                plate_mask[region_mask_bool] = 1
                zone_mask[region_mask_bool] = 1
            elif classification == "mixed":
                sub_plate = stats.get("plate_submask")
                if sub_plate is not None:
                    plate_mask[sub_plate] = 1
                    zone_mask[sub_plate] = 1
        else:
            # Small/medium region: single multi-signal classification
            eigenvalues, eigenvectors = _compute_region_pca(region_voxels)
            mean_edt = float(np.mean(edt[region_mask_bool]))

            plate_score, beam_score, scores_debug = _compute_region_scores(
                region_voxels, eigenvalues, eigenvectors,
                edt, region_mask_bool, bc_tags=bc_tags,
                plate_threshold=plate_threshold
            )

            if plate_score > beam_score + 0.1:
                classification = "plate"
            elif beam_score > plate_score + 0.05:
                classification = "beam"
            else:
                classification = "beam"  # default ambiguous to beam

            stats = _make_stats(eigenvalues)
            stats['mean_edt'] = mean_edt
            stats['plate_score'] = plate_score
            stats['beam_score'] = beam_score
            stats.update(scores_debug)
            zone_stats.append({"id": int(rid), "n_voxels": n_voxels,
                               "classification": classification, **stats})

            if classification == "plate":
                plate_mask[region_mask_bool] = 1
                zone_mask[region_mask_bool] = 1

        s = zone_stats[-1]
        detail = (f"        PCA: P={s.get('planarity', 0):.2f} L={s.get('linearity', 0):.2f}"
                  f" | A2D={s.get('aspect_2d', 0):.2f}"
                  f" | BC: ld={s.get('load_density', 0):.2f} fx={s.get('support_density', 0):.2f}"
                  f" | EDT_CV={s.get('edt_cv', 0):.2f}")
        scores = f"        Scores: plate={s.get('plate_score', 0):.2f} beam={s.get('beam_score', 0):.2f}"
        print(f"      Region {rid}: {n_voxels} voxels -> {s['classification']}")
        print(detail)
        print(scores)

    # Step 5: zone_mask is already set correctly during classification loop.
    # No morphological cleanup needed — binary_closing destroys thin features
    # and overrides correct classifications. The multi-signal scoring is
    # decisive enough to produce clean zone boundaries.

    # Step 6: Label final plate regions
    plate_labels, n_plates = label(zone_mask == 1, structure=s26)

    n_plate = int(np.sum(zone_mask == 1))
    n_beam = int(np.sum(zone_mask == 2))
    print(f"    [ZoneClassifier] Result: {n_plates} plate regions, "
          f"{n_plate} plate voxels, {n_beam} beam voxels")

    return zone_mask, plate_labels, zone_stats


def _compute_region_pca(indices):
    """
    PCA on voxel coordinates of a region.

    Parameters
    ----------
    indices : ndarray (N, 3)
        Voxel coordinates (z, y, x).

    Returns
    -------
    eigenvalues : ndarray (3,)
        Sorted descending.
    eigenvectors : ndarray (3, 3)
        Column eigenvectors corresponding to eigenvalues.
    """
    coords = indices.astype(np.float64)
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def _compute_region_scores(region_voxels, eigenvalues, eigenvectors,
                            edt, region_mask, bc_tags=None,
                            plate_threshold=None):
    """
    Multi-signal scoring for beam vs plate classification.

    Combines 4 signals:
    1. PCA shape (planarity/linearity) — weight 0.3
    2. Aspect ratio (2D spread vs 1D elongation) — weight 0.3
    3. BC tag density (load vs support regions) — weight 0.2
    4. EDT uniformity (plates have uniform thickness) — weight 0.2

    Returns
    -------
    plate_score : float
        Score for plate classification (0-1).
    beam_score : float
        Score for beam classification (0-1).
    debug : dict
        Per-signal breakdown for logging.
    """
    lam1, lam2, lam3 = eigenvalues
    eps = 1e-10

    # Degenerate case
    if lam1 < eps:
        return 0.0, 0.0, {"aspect_2d": 0, "load_density": 0, "support_density": 0, "edt_cv": 0}

    # --- Signal 1: PCA Shape (existing metrics) ---
    planarity = 1.0 - (lam3 / (lam2 + eps)) if lam2 > eps else 0.0
    linearity = 1.0 - (lam2 / (lam1 + eps))

    # --- Signal 2: Aspect Ratio ---
    # Plates spread in 2 directions: λ1 ≈ λ2 >> λ3 → aspect_2d ≈ 1.0
    # Beams spread in 1 direction: λ1 >> λ2 → aspect_2d << 1.0
    aspect_2d = float(min(lam1, lam2) / (max(lam1, lam2) + eps))

    # --- Signal 3: BC Tag Density ---
    load_density = 0.0
    support_density = 0.0
    if bc_tags is not None:
        n_region = max(1, int(np.sum(region_mask)))
        load_density = float(np.sum((bc_tags == 2) & region_mask)) / n_region
        support_density = float(np.sum((bc_tags == 1) & region_mask)) / n_region

    # --- Signal 4: EDT Uniformity ---
    # Plates have uniform thickness (low coefficient of variation)
    # Beams taper or vary (higher CV)
    edt_values = edt[region_mask]
    edt_values = edt_values[edt_values > 0]
    if len(edt_values) > 0:
        edt_mean = float(np.mean(edt_values))
        edt_std = float(np.std(edt_values))
        edt_cv = edt_std / (edt_mean + eps)  # coefficient of variation
    else:
        edt_cv = 0.5  # neutral

    edt_uniformity = 1.0 - min(1.0, edt_cv)  # high = uniform = plate-like

    # --- Combine into scores ---
    # Weights: shape=0.25, aspect=0.25, bc=0.2, edt=0.2
    # beam_bias adds a flat bonus to beam score (>0 = more beams, <0 = more plates)
    w_shape = 0.25
    w_aspect = 0.25
    w_bc = 0.2
    w_edt = 0.2
    beam_bias = 0.9  # Neutral: let PCA + aspect ratio + BC tags decide

    # Plate score: high planarity + high aspect_2d + high load density + high uniformity
    plate_score = (w_shape * planarity +
                   w_aspect * aspect_2d +
                   w_bc * load_density * 3.0 +  # Scale up (density is typically < 0.1)
                   w_edt * edt_uniformity)

    # Beam score: high linearity + low aspect_2d + high support density + low uniformity
    beam_score = (w_shape * linearity +
                  w_aspect * (1.0 - aspect_2d) +
                  w_bc * support_density * 3.0 +
                  w_edt * (1.0 - edt_uniformity) +
                  beam_bias)

    # Normalize so scores are comparable
    total = plate_score + beam_score
    if total > eps:
        plate_score /= total
        beam_score /= total

    debug = {
        "aspect_2d": aspect_2d,
        "load_density": load_density,
        "support_density": support_density,
        "edt_cv": edt_cv,
        "edt_uniformity": edt_uniformity,
    }

    return float(plate_score), float(beam_score), debug


def _classify_region_shape(eigenvalues, planarity_thresh=0.7, linearity_thresh=0.5,
                           mean_thickness=None, plate_threshold=None):
    """Legacy single-metric classification (kept for block-level fallback)."""
    lam1, lam2, lam3 = eigenvalues
    eps = 1e-10

    if lam1 < eps:
        return "ambiguous"

    planarity = 1.0 - (lam3 / (lam2 + eps)) if lam2 > eps else 0.0
    linearity = 1.0 - (lam2 / (lam1 + eps))

    adjusted_planarity_thresh = planarity_thresh
    if mean_thickness is not None and plate_threshold is not None:
        thickness_ratio = mean_thickness / plate_threshold
        if thickness_ratio > 0.8:
            adjusted_planarity_thresh = min(0.90, planarity_thresh + (thickness_ratio - 0.8) * 0.3)
            linearity_thresh = min(0.7, linearity_thresh + (thickness_ratio - 0.8) * 0.2)

    if planarity > adjusted_planarity_thresh and linearity < linearity_thresh:
        return "plate"
    elif linearity > linearity_thresh:
        return "beam"
    else:
        return "ambiguous"


def _make_stats(eigenvalues):
    """Compute planarity/linearity stats from eigenvalues."""
    lam1, lam2, lam3 = eigenvalues
    eps = 1e-10
    return {
        "planarity": float(1.0 - (lam3 / (lam2 + eps))) if lam2 > eps else 0.0,
        "linearity": float(1.0 - (lam2 / (lam1 + eps))),
        "eigenvalues": [float(v) for v in eigenvalues],
    }


def _subregion_analysis(region_mask, solid, edt, plate_threshold,
                        planarity_thresh, linearity_thresh, block_size=12,
                        bc_tags=None):
    """
    Block-level multi-signal classification for large regions.

    Uses both block-level PCA and overall region scoring to classify.
    For large connected regions, first tries overall multi-signal scoring,
    then falls back to block-level analysis if the region is mixed.

    Returns
    -------
    classification : str
        Overall: 'plate', 'beam', 'mixed', or 'ambiguous'
    stats : dict
        Includes planarity, linearity, scores, and optionally plate_submask.
    """
    indices = np.argwhere(region_mask)
    bbox_min = indices.min(axis=0)
    bbox_max = indices.max(axis=0)

    # Overall PCA and multi-signal scoring first
    eigenvalues, eigenvectors = _compute_region_pca(indices)
    mean_edt = float(np.mean(edt[region_mask]))
    stats = _make_stats(eigenvalues)
    stats['mean_edt'] = mean_edt

    # Compute overall multi-signal score
    plate_score, beam_score, scores_debug = _compute_region_scores(
        indices, eigenvalues, eigenvectors,
        edt, region_mask, bc_tags=bc_tags,
        plate_threshold=plate_threshold
    )
    stats['plate_score'] = plate_score
    stats['beam_score'] = beam_score
    stats.update(scores_debug)

    print(f"      [Overall scoring] plate={plate_score:.2f} beam={beam_score:.2f}")

    # Always do block-level analysis for large regions — the overall score
    # can be misleading when a large plate dominates a few small beam supports
    plate_blocks = np.zeros_like(region_mask, dtype=bool)
    beam_blocks = np.zeros_like(region_mask, dtype=bool)
    n_plate_blocks = 0
    n_beam_blocks = 0

    for z in range(bbox_min[0], bbox_max[0] + 1, block_size):
        for y in range(bbox_min[1], bbox_max[1] + 1, block_size):
            for x in range(bbox_min[2], bbox_max[2] + 1, block_size):
                z1 = min(z + block_size, region_mask.shape[0])
                y1 = min(y + block_size, region_mask.shape[1])
                x1 = min(x + block_size, region_mask.shape[2])

                block_mask = region_mask[z:z1, y:y1, x:x1]
                block_indices = np.argwhere(block_mask)

                if len(block_indices) < 5:
                    continue

                # Offset to global coordinates for PCA
                block_global = block_indices + np.array([z, y, x])
                blk_eig, blk_evec = _compute_region_pca(block_global)

                # Build global block mask for scoring
                global_block_mask = np.zeros_like(region_mask, dtype=bool)
                global_block_mask[z:z1, y:y1, x:x1] = block_mask

                # Multi-signal scoring at block level
                blk_plate, blk_beam, _ = _compute_region_scores(
                    block_global, blk_eig, blk_evec,
                    edt, global_block_mask, bc_tags=bc_tags,
                    plate_threshold=plate_threshold
                )

                if blk_plate > blk_beam:
                    plate_blocks[z:z1, y:y1, x:x1] |= block_mask
                    n_plate_blocks += 1
                else:
                    beam_blocks[z:z1, y:y1, x:x1] |= block_mask
                    n_beam_blocks += 1

    total_blocks = n_plate_blocks + n_beam_blocks
    if total_blocks == 0:
        return "ambiguous", stats

    plate_ratio = n_plate_blocks / total_blocks

    print(f"      [Block-level analysis] {n_plate_blocks} plate blocks, {n_beam_blocks} beam blocks, ratio={plate_ratio:.2f}")

    if plate_ratio > 0.95:
        return "plate", stats
    elif plate_ratio < 0.15:
        return "beam", stats
    else:
        # Mixed region: use overall score as tiebreaker
        # If overall score clearly favors one classification, use that
        if plate_score > beam_score + 0.1:
            stats["plate_submask"] = plate_blocks & region_mask
            stats["plate_block_ratio"] = float(plate_ratio)
            return "mixed", stats  # Apply plate sub-mask
        elif beam_score > plate_score + 0.1:
            return "beam", stats  # Overall says beam, trust it
        else:
            # Truly ambiguous: use block-level plate sub-mask
            stats["plate_submask"] = plate_blocks & region_mask
            stats["plate_block_ratio"] = float(plate_ratio)
            return "mixed", stats


def compute_interface_mask(zone_mask):
    """
    Find voxels at the beam-plate boundary.

    Returns a boolean mask where True indicates a beam voxel (zone==2) that is
    26-adjacent to a plate voxel (zone==1), or vice versa.
    """
    s26 = np.ones((3, 3, 3), dtype=bool)
    plate_zone = zone_mask == 1
    beam_zone = zone_mask == 2

    # Beam voxels adjacent to plate
    plate_dilated = binary_dilation(plate_zone, structure=s26)
    beam_at_interface = beam_zone & plate_dilated

    # Plate voxels adjacent to beam
    beam_dilated = binary_dilation(beam_zone, structure=s26)
    plate_at_interface = plate_zone & beam_dilated

    return beam_at_interface | plate_at_interface
