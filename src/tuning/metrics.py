"""
Metric extraction utilities for parameter tuning.

Extracts geometry fidelity metrics from Stage 1 (Reconstruction) output.
"""
import numpy as np
import json
from scipy.spatial.distance import directed_hausdorff


def load_npz_voxels(npz_path, vol_thresh=0.3):
    """
    Load voxel data from Top3D .npz file.
    
    Returns:
        solid: 3D binary array (D, H, W)
        metadata: dict with pitch, origin, etc.
    """
    data = np.load(npz_path)
    
    # Try 'rho' first (standard key), fall back to 'x'
    if 'rho' in data.files:
        density = data['rho']
    elif 'x' in data.files:
        density = data['x']
    else:
        raise ValueError(f"NPZ file must contain 'rho' or 'x' key. Found: {data.files}")
    
    # Threshold to binary
    solid = (density > vol_thresh).astype(np.int32)
    
    # Extract metadata
    pitch = float(data['pitch']) if 'pitch' in data else 1.0
    origin = data['origin'] if 'origin' in data else np.array([0.0, 0.0, 0.0])
    
    metadata = {
        'pitch': pitch,
        'origin': origin,
        'shape': solid.shape
    }
    
    return solid, metadata


def compute_voxel_volume(solid, pitch):
    """
    Compute total volume of voxel representation.
    """
    voxel_vol = pitch ** 3
    num_solid = np.sum(solid)
    return num_solid * voxel_vol


def load_stage1_json(json_path):
    """
    Load Stage 1 reconstruction output JSON.
    
    Returns dict with:
        - curves: list of edges
        - metadata: target_volume, etc.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def compute_skeleton_volume(curves):
    """
    Compute total volume of skeleton frame.
    
    Volume = sum(π × r² × L) for all edges
    
    Curves format: list of dicts with 'points' key
    Each point is [x, y, z, radius]
    """
    total_vol = 0.0
    
    for curve in curves:
        points = curve.get('points', [])
        if len(points) < 2:
            continue
        
        # Extract coordinates and radii
        coords = np.array([[p[0], p[1], p[2]] for p in points])
        radii = np.array([p[3] if len(p) > 3 else 1.0 for p in points])
        
        # Compute edge length (polyline)
        diffs = np.diff(coords, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        
        # Use mean radius for each segment
        mean_radii = (radii[:-1] + radii[1:]) / 2.0
        
        # Volume for each segment
        for length, r in zip(segment_lengths, mean_radii):
            area = np.pi * (r ** 2)
            total_vol += area * length
    
    return total_vol


def compute_graph_complexity(curves):
    """
    Compute graph statistics.
    
    Returns:
        num_nodes: int
        num_edges: int
        min_edge_length: float
        mean_edge_length: float
    """
    # Extract unique node positions (endpoints only)
    node_set = set()
    edge_lengths = []
    
    for curve in curves:
        points = curve.get('points', [])
        if len(points) < 2:
            continue
        
        # Add endpoints to node set
        p0 = tuple(points[0][:3])
        pN = tuple(points[-1][:3])
        node_set.add(p0)
        node_set.add(pN)
        
        # Compute edge length
        coords = np.array([[p[0], p[1], p[2]] for p in points])
        diffs = np.diff(coords, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        edge_length = np.sum(lengths)
        edge_lengths.append(edge_length)
    
    num_nodes = len(node_set)
    num_edges = len(curves)
    min_edge = min(edge_lengths) if edge_lengths else 0.0
    mean_edge = np.mean(edge_lengths) if edge_lengths else 0.0
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'min_edge_length': min_edge,
        'mean_edge_length': mean_edge
    }


def compute_hausdorff_coverage(solid, curves, pitch, origin):
    """
    Compute maximum distance from voxel centroids to skeleton.
    
    This measures how well the skeleton "covers" the voxel space.
    Lower is better (skeleton closer to all voxels).
    """
    # Extract voxel centroids
    voxel_indices = np.argwhere(solid > 0)
    if len(voxel_indices) == 0:
        return 0.0
    
    # Convert to world coordinates
    voxel_coords = origin + (voxel_indices + 0.5) * pitch
    
    # Extract skeleton points (all points in all curves)
    skeleton_points = []
    for curve in curves:
        points = curve.get('points', [])
        for p in points:
            skeleton_points.append([p[0], p[1], p[2]])
    
    if len(skeleton_points) == 0:
        return 0.0
    
    skeleton_points = np.array(skeleton_points)
    
    # Compute directed Hausdorff (voxels -> skeleton)
    # This gives the max distance from any voxel to its nearest skeleton point
    hausdorff_dist, _, _ = directed_hausdorff(voxel_coords, skeleton_points)
    
    return hausdorff_dist


def extract_metrics(npz_path, stage1_json_path, vol_thresh=0.3):
    """
    Main function to extract all metrics for parameter tuning.
    
    Returns dict with:
        - voxel_vol: float
        - skeleton_vol: float
        - volume_error: float (relative error)
        - num_nodes: int
        - num_edges: int
        - min_edge_length: float
        - mean_edge_length: float
        - hausdorff_dist: float
        - domain_size: float (for normalization)
    """
    # Load voxel data
    solid, voxel_meta = load_npz_voxels(npz_path, vol_thresh)
    pitch = voxel_meta['pitch']
    origin = voxel_meta['origin']
    
    voxel_vol = compute_voxel_volume(solid, pitch)
    
    # Load skeleton data
    stage1_data = load_stage1_json(stage1_json_path)
    curves = stage1_data.get('curves', [])
    
    skeleton_vol = compute_skeleton_volume(curves)
    
    # Volume error
    if voxel_vol > 0:
        volume_error = abs(skeleton_vol - voxel_vol) / voxel_vol
    else:
        volume_error = 0.0
    
    # Graph complexity
    graph_stats = compute_graph_complexity(curves)
    
    # Hausdorff distance
    hausdorff_dist = compute_hausdorff_coverage(solid, curves, pitch, origin)
    
    # Domain size (for normalization)
    domain_size = max(solid.shape) * pitch
    
    metrics = {
        'voxel_vol': voxel_vol,
        'skeleton_vol': skeleton_vol,
        'volume_error': volume_error,
        'num_nodes': graph_stats['num_nodes'],
        'num_edges': graph_stats['num_edges'],
        'min_edge_length': graph_stats['min_edge_length'],
        'mean_edge_length': graph_stats['mean_edge_length'],
        'hausdorff_dist': hausdorff_dist,
        'domain_size': domain_size
    }
    
    return metrics
