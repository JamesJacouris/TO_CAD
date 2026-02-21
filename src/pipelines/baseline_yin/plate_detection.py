import numpy as np
from scipy.ndimage import distance_transform_edt, label


def classify_regions(solid, pitch, max_thickness_ratio=0.15, min_region_ratio=0.01):
    """
    Identifies plate-like regions using EDT-based thin region detection.
    All thresholds are relative to domain size, not absolute voxel counts.

    solid: 3D binary array (0/1)
    pitch: voxel size in mm
    max_thickness_ratio: max plate half-thickness as fraction of domain diagonal
    min_region_ratio: minimum region size as fraction of largest domain face area

    Returns: plate_mask (3D int array with labels), num_plates
    """
    D, H, W = solid.shape
    diag = np.sqrt(D**2 + H**2 + W**2)

    # EDT gives distance to nearest background voxel (half-thickness in voxel units)
    edt = distance_transform_edt(solid)

    # Max half-thickness for plate candidates (in voxel units)
    max_half_thickness = max(1.5, diag * max_thickness_ratio / 2.0)

    print(f"    [Detection] EDT-based detection: max half-thickness = {max_half_thickness:.1f} voxels "
          f"(domain diagonal = {diag:.1f})")

    # Thin voxels: those whose EDT is within the plate thickness threshold
    thin_mask = (solid > 0) & (edt <= max_half_thickness)

    # Label connected thin regions
    s26 = np.ones((3, 3, 3), dtype=np.int32)
    labeled, n_regions = label(thin_mask, structure=s26)

    # Filter by minimum region size (pitch-relative)
    # Use largest face area of domain bounding box as reference
    face_areas = sorted([D * H, D * W, H * W], reverse=True)
    min_voxels = max(20, int(face_areas[0] * min_region_ratio))

    kept = 0
    for label_id in range(1, n_regions + 1):
        region_size = np.sum(labeled == label_id)
        if region_size < min_voxels:
            labeled[labeled == label_id] = 0
        else:
            kept += 1

    # Re-label to get contiguous IDs
    if kept < n_regions:
        labeled[labeled > 0] = 1
        labeled, n_final = label(labeled > 0, structure=s26)
    else:
        n_final = kept

    print(f"    [Detection] Identified {n_final} plate regions "
          f"(filtered {n_regions - n_final} small regions, min size = {min_voxels})")

    return labeled, n_final
