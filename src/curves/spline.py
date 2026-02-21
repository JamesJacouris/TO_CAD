"""
Cubic Bézier curve utilities for curved beam representation.

A cubic Bézier is defined by four control points:
  P(t) = (1-t)³ P0 + 3t(1-t)² P1 + 3t²(1-t) P2 + t³ P3,  t ∈ [0, 1]

P0 and P3 are the beam endpoints (junction nodes).
P1 and P2 are the two interior control points stored per beam.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def sanitize_bezier_ctrl_pts(p0, p3, p1, p2, max_bulge_ratio=0.2):
    """
    Clamp interior control points P1, P2 to prevent cubic Bézier self-intersection.

    Enforces two conditions:
      1. **Chord monotonicity**: 0 ≤ t1 ≤ t2 ≤ L, where t1/t2 are the
         chord-direction projections of P1/P2 onto the chord P0→P3.
         Violations cause the curve to fold back on itself (loop).
      2. **Perpendicular bulge limit**: ||P1⊥||, ||P2⊥|| ≤ max_bulge_ratio * L.
         Prevents extreme lateral deviation that overlaps adjacent beams.
         Default 0.2 (20% of chord) keeps curves visually close to straight.

    Parameters
    ----------
    p0, p3 : (3,) arrays — fixed beam endpoints
    p1, p2 : (3,) arrays — interior control points to sanitize
    max_bulge_ratio : float — max perpendicular bulge as fraction of chord length

    Returns
    -------
    p1_safe, p2_safe : (3,) arrays
    """
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    chord = p3 - p0
    L = np.linalg.norm(chord)
    if L < 1e-10:
        return p1.copy(), p2.copy()

    d = chord / L  # unit chord direction

    # Decompose each ctrl pt into chord + perpendicular components
    t1 = float(np.dot(p1 - p0, d))
    t2 = float(np.dot(p2 - p0, d))
    n1 = (p1 - p0) - t1 * d
    n2 = (p2 - p0) - t2 * d

    # 1. Clamp chord projections to [0, L]
    t1 = float(np.clip(t1, 0.0, L))
    t2 = float(np.clip(t2, 0.0, L))

    # 2. Ensure proper ordering t1 ≤ t2 (swapping perpendicular components too)
    if t1 > t2:
        t1, t2 = t2, t1
        n1, n2 = n2, n1

    # 3. Clamp perpendicular bulge
    max_perp = max_bulge_ratio * L
    n1_len = np.linalg.norm(n1)
    n2_len = np.linalg.norm(n2)
    if n1_len > max_perp:
        n1 = n1 * (max_perp / n1_len)
    if n2_len > max_perp:
        n2 = n2 * (max_perp / n2_len)

    p1_safe = p0 + t1 * d + n1
    p2_safe = p0 + t2 * d + n2
    return p1_safe, p2_safe


def fit_cubic_bezier(p0, p3, interior_pts):
    """
    Fit a cubic Bézier through the given intermediate skeleton points.

    Strategy: minimise least-squares distance from interior_pts to the
    Bézier curve by solving the linear system for P1 and P2 with
    **cumulative chord-length parameterisation** t_i = arc_i / arc_total.
    This avoids the bias that uniform parameter spacing causes when
    skeleton intermediates are unevenly distributed (common on voxel grids).

    If interior_pts is empty or has only one point, fall back to the
    chord-length initialisation (P1 at 1/3, P2 at 2/3 of the chord).

    All return paths pass through `sanitize_bezier_ctrl_pts()` to ensure
    the curve is chord-monotone and does not self-intersect.

    Parameters
    ----------
    p0 : (3,) array   — start endpoint (fixed)
    p3 : (3,) array   — end endpoint (fixed)
    interior_pts : list of (3,) or (K, 3) array — skeleton intermediate pts

    Returns
    -------
    ctrl_pts : (2, 3) array — interior control points [P1, P2]
    """
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    if len(interior_pts) == 0:
        # Straight beam: control points at 1/3 and 2/3 along the chord
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2.0 * (p3 - p0) / 3.0
        p1, p2 = sanitize_bezier_ctrl_pts(p0, p3, p1, p2)
        return np.array([p1, p2])

    pts = np.asarray(interior_pts, dtype=float)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]

    n = len(pts)

    if n == 1:
        # Single intermediate point — place P1, P2 symmetrically around it
        mid = pts[0]
        p1 = p0 + 0.5 * (mid - p0)
        p2 = p3 + 0.5 * (mid - p3)
        p1, p2 = sanitize_bezier_ctrl_pts(p0, p3, p1, p2)
        return np.array([p1, p2])

    # Cumulative chord-length parameterisation: t_i = arc_i / arc_total.
    # Skeleton intermediates on a voxel grid are often unevenly spaced;
    # arc-length t gives each point a weight proportional to its actual
    # position along the path rather than a uniform fraction.
    full_chain = np.vstack([p0, pts, p3])  # (n+2, 3)
    seg_lens = np.linalg.norm(np.diff(full_chain, axis=0), axis=1)  # (n+1,)
    total_len = seg_lens.sum()
    if total_len < 1e-10:
        t_vals = np.linspace(0.0, 1.0, n + 2)[1:-1]
    else:
        t_vals = np.cumsum(seg_lens[:-1]) / total_len  # (n,) interior only

    # Bézier basis: B1(t) = 3t(1-t)², B2(t) = 3t²(1-t)
    B1 = 3.0 * t_vals * (1.0 - t_vals) ** 2   # shape (n,)
    B2 = 3.0 * t_vals ** 2 * (1.0 - t_vals)   # shape (n,)

    # RHS: Q_i = pts[i] - (1-t)³ P0 - t³ P3
    t3 = t_vals ** 3
    t0 = (1.0 - t_vals) ** 3
    rhs = pts - np.outer(t0, p0) - np.outer(t3, p3)  # (n, 3)

    # Build 2x2 linear system for [P1, P2] from the least-squares normal eqs
    # A^T A [P1; P2] = A^T rhs  where A = [[B1_i, B2_i], ...]
    A = np.column_stack([B1, B2])  # (n, 2)
    ATA = A.T @ A                  # (2, 2)
    ATrhs = A.T @ rhs              # (2, 3)

    # Solve for [P1, P2] per axis (3 independent 2x2 systems)
    try:
        sol = np.linalg.solve(ATA, ATrhs)  # (2, 3)
    except np.linalg.LinAlgError:
        # Degenerate: fall back to chord initialisation
        sol = np.array([p0 + (p3 - p0) / 3.0, p0 + 2.0 * (p3 - p0) / 3.0])

    p1_safe, p2_safe = sanitize_bezier_ctrl_pts(p0, p3, sol[0], sol[1])
    return np.array([p1_safe, p2_safe])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def sample_bezier(p0, p1, p2, p3, N):
    """
    Sample N+1 points at uniform parameter spacing t = 0, 1/N, ..., 1.

    Parameters
    ----------
    p0, p1, p2, p3 : (3,) arrays — control points
    N : int — number of sub-intervals (returns N+1 points)

    Returns
    -------
    pts : (N+1, 3) array
    """
    t = np.linspace(0.0, 1.0, N + 1)[:, np.newaxis]  # (N+1, 1)
    p0, p1, p2, p3 = (np.asarray(x, dtype=float) for x in (p0, p1, p2, p3))
    pts = (
        (1 - t) ** 3 * p0
        + 3 * t * (1 - t) ** 2 * p1
        + 3 * t ** 2 * (1 - t) * p2
        + t ** 3 * p3
    )
    return pts  # (N+1, 3)


def bezier_tangent(p0, p1, p2, p3, t):
    """
    First derivative (tangent) of the cubic Bézier at parameter t.

    dP/dt = 3(1-t)²(P1-P0) + 6t(1-t)(P2-P1) + 3t²(P3-P2)

    Parameters
    ----------
    t : float or (K,) array

    Returns
    -------
    tangent : (3,) or (K, 3) array
    """
    p0, p1, p2, p3 = (np.asarray(x, dtype=float) for x in (p0, p1, p2, p3))
    scalar = np.ndim(t) == 0
    t = np.atleast_1d(np.asarray(t, dtype=float))[:, np.newaxis]
    tan = (
        3 * (1 - t) ** 2 * (p1 - p0)
        + 6 * t * (1 - t) * (p2 - p1)
        + 3 * t ** 2 * (p3 - p2)
    )
    return tan[0] if scalar else tan


def bezier_arc_length(p0, p1, p2, p3, n_quad=20):
    """
    Estimate arc length via Gauss-Legendre quadrature:
      L = ∫₀¹ ||dP/dt|| dt

    Parameters
    ----------
    n_quad : int — number of quadrature points (20 is accurate for moderate curvature)

    Returns
    -------
    length : float (mm)
    """
    p0, p1, p2, p3 = (np.asarray(x, dtype=float) for x in (p0, p1, p2, p3))
    t_nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # Map from [-1, 1] to [0, 1]
    t_mapped = 0.5 * (t_nodes + 1.0)
    w_mapped = 0.5 * weights
    tangents = bezier_tangent(p0, p1, p2, p3, t_mapped)  # (n_quad, 3)
    speed = np.linalg.norm(tangents, axis=1)              # (n_quad,)
    return float(np.dot(w_mapped, speed))


# ---------------------------------------------------------------------------
# Helpers for the pipeline
# ---------------------------------------------------------------------------

def ctrl_pts_from_edge(nodes, edge):
    """
    Extract P0, P1, P2, P3 for a single edge dict/list.

    Parameters
    ----------
    nodes : (N, 3) array
    edge  : list [u, v, weight, intermediates, ...]  (Stage 1 format)

    Returns
    -------
    p0, p1, p2, p3 : (3,) arrays
    ctrl_pts       : (2, 3) array  [P1, P2]
    """
    u, v = int(edge[0]), int(edge[1])
    p0, p3 = nodes[u], nodes[v]
    interior = edge[3] if len(edge) >= 4 else []
    ctrl_pts = fit_cubic_bezier(p0, p3, interior)
    p1, p2 = ctrl_pts[0], ctrl_pts[1]
    return p0, p1, p2, p3, ctrl_pts


def sample_curve_points(p0, p1, p2, p3, radius, N=20):
    """
    Sample N+1 visualisation points with radius appended as 4th element.

    Returns list of [x, y, z, r] lists (matching existing `curves` JSON format).
    """
    pts = sample_bezier(p0, p1, p2, p3, N)  # (N+1, 3)
    return [list(pt) + [float(radius)] for pt in pts]
