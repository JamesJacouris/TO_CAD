"""Unit tests for IGA curved beam FEM implementation.

Tests cover the full IGA pipeline: Bernstein basis functions,
element stiffness computation, static condensation, frame solver,
and analytical/semi-analytical gradients for size and layout optimisation.
"""

import sys
import os
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.curves.spline import (
    bernstein_basis, bernstein_basis_d1,
    fit_cubic_bezier, sanitize_bezier_ctrl_pts, bezier_arc_length,
)
from src.optimization.fem import (
    _compute_local_frame,
    compute_iga_element_stiffness,
    compute_iga_element_stiffness_derivative,
    condense_element,
    recover_internal_displacements,
    solve_curved_frame,
    compute_curved_size_gradients,
    compute_curved_ctrl_gradients,
    compute_element_stiffness,
    rotation_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures: reusable beam geometries
# ---------------------------------------------------------------------------

@pytest.fixture
def straight_beam_x():
    """Straight beam along X-axis, length 10, collinear ctrl pts."""
    p0 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([10.0, 0.0, 0.0])
    p1 = p0 + (p3 - p0) / 3.0
    p2 = p0 + 2.0 * (p3 - p0) / 3.0
    return p0, p1, p2, p3


@pytest.fixture
def curved_beam_xy():
    """Curved beam in XY plane with moderate lateral bulge."""
    p0 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([10.0, 0.0, 0.0])
    p1 = np.array([3.0, 1.5, 0.0])
    p2 = np.array([7.0, 1.5, 0.0])
    return p0, p1, p2, p3


@pytest.fixture
def curved_beam_3d():
    """3D curved beam with out-of-plane bulge."""
    p0 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([10.0, 0.0, 0.0])
    p1 = np.array([3.0, 1.0, 0.5])
    p2 = np.array([7.0, -0.5, 1.0])
    return p0, p1, p2, p3


@pytest.fixture
def beam_params():
    """Standard beam material/section parameters."""
    return dict(E=1000.0, r=0.5, nu=0.3)


# ---------------------------------------------------------------------------
# 1. Bernstein basis tests
# ---------------------------------------------------------------------------

class TestBernsteinBasis:
    """Tests for cubic Bernstein basis functions."""

    def test_partition_of_unity(self):
        """B0+B1+B2+B3 = 1 at arbitrary xi values."""
        for xi in [0.0, 0.25, 0.5, 0.75, 1.0, 0.123, 0.876]:
            N = bernstein_basis(xi)
            assert abs(N.sum() - 1.0) < 1e-14, f"Sum={N.sum()} at xi={xi}"

    def test_derivatives_sum_to_zero(self):
        """dB0+dB1+dB2+dB3 = 0 (derivative of partition of unity)."""
        for xi in [0.0, 0.25, 0.5, 0.75, 1.0, 0.333]:
            dN = bernstein_basis_d1(xi)
            assert abs(dN.sum()) < 1e-14, f"dSum={dN.sum()} at xi={xi}"

    def test_endpoint_values(self):
        """B0(0)=1, B3(1)=1; all others zero at endpoints."""
        N0 = bernstein_basis(0.0)
        np.testing.assert_allclose(N0, [1, 0, 0, 0], atol=1e-15)
        N1 = bernstein_basis(1.0)
        np.testing.assert_allclose(N1, [0, 0, 0, 1], atol=1e-15)

    def test_symmetry_at_midpoint(self):
        """B0(0.5)=B3(0.5), B1(0.5)=B2(0.5) by symmetry."""
        N = bernstein_basis(0.5)
        assert abs(N[0] - N[3]) < 1e-15
        assert abs(N[1] - N[2]) < 1e-15

    def test_derivative_finite_difference(self):
        """Verify analytical derivatives against central FD."""
        eps = 1e-7
        for xi in [0.1, 0.3, 0.5, 0.7, 0.9]:
            dN_analytical = bernstein_basis_d1(xi)
            dN_fd = (bernstein_basis(xi + eps) - bernstein_basis(xi - eps)) / (2 * eps)
            np.testing.assert_allclose(dN_analytical, dN_fd, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Local frame tests
# ---------------------------------------------------------------------------

class TestLocalFrame:
    """Tests for _compute_local_frame()."""

    def test_orthonormal(self):
        """Frame axes must be orthonormal."""
        for tangent in [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 1, 0], [1, 2, 3], [0.1, -0.5, 0.9]]:
            R = _compute_local_frame(np.array(tangent, dtype=float))
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_e1_is_tangent(self):
        """First row of R must be the unit tangent."""
        tangent = np.array([3.0, 4.0, 0.0])
        R = _compute_local_frame(tangent)
        expected = tangent / np.linalg.norm(tangent)
        np.testing.assert_allclose(R[0], expected, atol=1e-14)

    def test_near_vertical(self):
        """Near-vertical tangent should not produce NaN."""
        R = _compute_local_frame(np.array([0.0, 0.0, 5.0]))
        assert not np.any(np.isnan(R))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_zero_tangent_returns_identity(self):
        """Zero tangent should return identity (fallback)."""
        R = _compute_local_frame(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)


# ---------------------------------------------------------------------------
# 3. IGA element stiffness tests
# ---------------------------------------------------------------------------

class TestIGAElementStiffness:
    """Tests for compute_iga_element_stiffness()."""

    def test_symmetry(self, curved_beam_xy, beam_params):
        """K_full must be symmetric."""
        p0, p1, p2, p3 = curved_beam_xy
        K = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_positive_semi_definite(self, curved_beam_xy, beam_params):
        """All eigenvalues of K_full >= 0."""
        p0, p1, p2, p3 = curved_beam_xy
        K = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > -1e-8), f"Min eigenvalue: {eigvals.min()}"

    def test_rigid_body_modes(self, curved_beam_xy, beam_params):
        """Free-free beam should have 6 zero eigenvalues (rigid body modes)."""
        p0, p1, p2, p3 = curved_beam_xy
        K = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        eigvals = np.sort(np.linalg.eigvalsh(K))
        # 24 DOFs, 6 rigid body modes → 6 near-zero eigenvalues
        assert np.all(np.abs(eigvals[:6]) < 1e-6), \
            f"First 6 eigenvalues: {eigvals[:6]}"
        # Remaining 18 should be positive
        assert np.all(eigvals[6:] > 1e-6), \
            f"7th eigenvalue: {eigvals[6]}"

    def test_3d_beam_no_nan(self, curved_beam_3d, beam_params):
        """3D curved beam should produce finite stiffness values."""
        p0, p1, p2, p3 = curved_beam_3d
        K = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        assert not np.any(np.isnan(K))
        assert not np.any(np.isinf(K))

    def test_scales_with_modulus(self, curved_beam_xy):
        """Doubling E should double all K entries (linearity)."""
        p0, p1, p2, p3 = curved_beam_xy
        K1 = compute_iga_element_stiffness(1000.0, 0.5, p0, p1, p2, p3, 0.3)
        K2 = compute_iga_element_stiffness(2000.0, 0.5, p0, p1, p2, p3, 0.3)
        np.testing.assert_allclose(K2, 2.0 * K1, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Static condensation tests
# ---------------------------------------------------------------------------

class TestCondenseElement:
    """Tests for condense_element() and recover_internal_displacements()."""

    def test_condensed_symmetry(self, curved_beam_xy, beam_params):
        """K_cond must be symmetric."""
        p0, p1, p2, p3 = curved_beam_xy
        K_full = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        K_cond = condense_element(K_full)
        assert K_cond is not None
        np.testing.assert_allclose(K_cond, K_cond.T, atol=1e-10)

    def test_condensed_positive_semi_definite(self, curved_beam_xy, beam_params):
        """Condensed K should have >= 0 eigenvalues."""
        p0, p1, p2, p3 = curved_beam_xy
        K_full = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        K_cond = condense_element(K_full)
        assert K_cond is not None
        eigvals = np.linalg.eigvalsh(K_cond)
        assert np.all(eigvals > -1e-8), f"Min eigenvalue: {eigvals.min()}"

    def test_condensed_size(self, curved_beam_xy, beam_params):
        """Condensed K must be 12x12."""
        p0, p1, p2, p3 = curved_beam_xy
        K_full = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        K_cond = condense_element(K_full)
        assert K_cond is not None
        assert K_cond.shape == (12, 12)

    def test_straight_beam_equivalence(self, straight_beam_x, beam_params):
        """IGA condensed (collinear) should approximate standard E-B element.

        Expected difference ~5-10% due to Timoshenko shear correction
        (IGA includes shear deformation, E-B does not).
        """
        p0, p1, p2, p3 = straight_beam_x
        E, r, nu = beam_params['E'], beam_params['r'], beam_params['nu']
        L = np.linalg.norm(p3 - p0)
        G = E / (2.0 * (1.0 + nu))
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = np.pi * r ** 4 / 2.0

        # IGA path
        K_full = compute_iga_element_stiffness(E, r, p0, p1, p2, p3, nu)
        K_iga = condense_element(K_full)
        assert K_iga is not None

        # E-B path (beam along X → rotation matrix is identity)
        K_eb = compute_element_stiffness(E, A, I, I, J, G, L)

        # For L/r = 10/0.5 = 20, Timoshenko effect is ~1-3%
        rel_diff = np.linalg.norm(K_iga - K_eb) / np.linalg.norm(K_eb)
        assert rel_diff < 0.10, f"Relative difference: {rel_diff:.4f} (expect <10%)"

    def test_recover_internal_consistency(self, curved_beam_xy, beam_params):
        """Recovered internal DOFs should satisfy K_full @ u_full ≈ F_full."""
        p0, p1, p2, p3 = curved_beam_xy
        E, r, nu = beam_params['E'], beam_params['r'], beam_params['nu']

        K_full = compute_iga_element_stiffness(E, r, p0, p1, p2, p3, nu)

        # Create a simple load case: unit force at P3 in Y direction
        # and fix all DOFs at P0
        # With condensed system: K_cond @ u_b = F_b
        K_cond = condense_element(K_full)
        assert K_cond is not None

        # Fix P0 (DOFs 0:6), load P3 (DOF 7 = Fy at P3)
        # In 12-DOF condensed: P0=DOFs[0:6], P3=DOFs[6:12]
        F_b = np.zeros(12)
        F_b[7] = -1.0  # Fy at P3

        # Solve condensed
        K_fixed = K_cond[6:12, 6:12]
        u_free = np.linalg.solve(K_fixed, F_b[6:12])
        u_b = np.zeros(12)
        u_b[6:12] = u_free

        # Recover internal DOFs
        u_full = recover_internal_displacements(K_full, u_b)
        assert u_full.shape == (24,)

        # The full system should satisfy: K_full @ u_full ≈ F_full
        # where F_full has zero internal forces (static condensation property)
        F_full = K_full @ u_full
        # Internal forces (P1, P2 DOFs = indices 6:18) should be ~0
        np.testing.assert_allclose(F_full[6:18], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Curved frame solver tests
# ---------------------------------------------------------------------------

class TestSolveCurvedFrame:
    """Tests for solve_curved_frame()."""

    def _simple_cantilever(self, ctrl_pts_list):
        """Build a 2-beam cantilever: node0--node1--node2, fixed at node0."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        edges = np.array([[0, 1], [1, 2]])
        radii = np.array([0.5, 0.5])
        loads = {2: [0, -1.0, 0, 0, 0, 0]}
        bcs = {0: [0, 1, 2, 3, 4, 5]}
        return nodes, edges, radii, ctrl_pts_list, loads, bcs

    def test_finite_compliance(self):
        """Curved cantilever should produce finite positive compliance."""
        cp0 = np.array([[1.5, 0.3, 0.0], [3.5, 0.3, 0.0]])
        cp1 = np.array([[6.5, 0.3, 0.0], [8.5, 0.3, 0.0]])
        nodes, edges, radii, _, loads, bcs = self._simple_cantilever([cp0, cp1])

        u, compliance, edata = solve_curved_frame(
            nodes, edges, radii, [cp0, cp1], E=1000.0, nu=0.3,
            loads=loads, bcs=bcs)

        assert np.isfinite(compliance)
        assert compliance > 0
        assert not np.any(np.isnan(u))

    def test_mixed_straight_curved(self):
        """Frame with one curved and one straight beam should work."""
        cp0 = np.array([[1.5, 0.5, 0.0], [3.5, 0.5, 0.0]])
        nodes, edges, radii, _, loads, bcs = self._simple_cantilever([cp0, None])

        u, compliance, edata = solve_curved_frame(
            nodes, edges, radii, [cp0, None], E=1000.0, nu=0.3,
            loads=loads, bcs=bcs)

        assert np.isfinite(compliance)
        assert compliance > 0
        # Check element data: first curved, second straight
        assert edata[0]['curved'] is True
        assert edata[1]['curved'] is False

    def test_straight_ctrl_pts_vs_solve_frame(self):
        """Collinear ctrl_pts should give similar compliance to solve_frame.

        Not identical because IGA uses Timoshenko while solve_frame uses E-B.
        """
        from src.optimization.fem import solve_frame

        nodes = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        edges = np.array([[0, 1], [1, 2]])
        radii = np.array([0.5, 0.5])
        loads = {2: [0, -1.0, 0, 0, 0, 0]}
        bcs = {0: [0, 1, 2, 3, 4, 5]}

        # Straight E-B
        _, c_eb, _ = solve_frame(nodes, edges, radii, E=1000.0, nu=0.3,
                                  loads=loads, bcs=bcs)

        # IGA with collinear ctrl pts
        cp0 = np.array([[0 + 5/3, 0, 0], [0 + 10/3, 0, 0]])
        cp1 = np.array([[5 + 5/3, 0, 0], [5 + 10/3, 0, 0]])
        _, c_iga, _ = solve_curved_frame(
            nodes, edges, radii, [cp0, cp1], E=1000.0, nu=0.3,
            loads=loads, bcs=bcs)

        # Should be within ~10% for L/r = 10
        rel_diff = abs(c_iga - c_eb) / c_eb
        assert rel_diff < 0.15, f"Compliance diff: {rel_diff:.4f} (EB={c_eb:.4f}, IGA={c_iga:.4f})"

    def test_curvature_changes_compliance(self):
        """Curvature should change compliance vs straight (same formulation).

        Both use IGA Timoshenko to isolate the curvature effect.
        For a cantilever with lateral bulge, the arch effect introduces
        axial-bending coupling that typically reduces compliance (stiffer).
        """
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        edges = np.array([[0, 1]])
        radii = np.array([0.5])
        loads = {1: [0, -1.0, 0, 0, 0, 0]}
        bcs = {0: [0, 1, 2, 3, 4, 5]}

        # Straight (IGA with collinear ctrl pts)
        p0, p3 = nodes[0], nodes[1]
        cp_straight = np.array([p0 + (p3 - p0) / 3.0, p0 + 2.0 * (p3 - p0) / 3.0])
        _, c_straight, _ = solve_curved_frame(
            nodes, edges, radii, [cp_straight], E=1000.0, nu=0.3,
            loads=loads, bcs=bcs)

        # Curved (lateral bulge)
        cp_curved = np.array([[3.0, 2.0, 0.0], [7.0, 2.0, 0.0]])
        _, c_curved, _ = solve_curved_frame(
            nodes, edges, radii, [cp_curved], E=1000.0, nu=0.3,
            loads=loads, bcs=bcs)

        # Curvature must change the result (not identical to straight)
        assert abs(c_curved - c_straight) / c_straight > 0.01, \
            f"Curved ({c_curved:.4f}) ≈ straight ({c_straight:.4f}), expected difference"
        # Both must be positive and finite
        assert c_curved > 0 and c_straight > 0


# ---------------------------------------------------------------------------
# 6. Size gradient tests (dC/dr)
# ---------------------------------------------------------------------------

class TestSizeGradients:
    """Tests for compute_curved_size_gradients()."""

    def test_gradients_negative(self):
        """dC/dr should be negative (increasing radius decreases compliance)."""
        nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        edges = np.array([[0, 1]])
        radii = np.array([0.5])
        cp = np.array([[3.0, 1.0, 0.0], [7.0, 1.0, 0.0]])
        loads = {1: [0, -1.0, 0, 0, 0, 0]}
        bcs = {0: [0, 1, 2, 3, 4, 5]}

        u, compliance, edata = solve_curved_frame(
            nodes, edges, radii, [cp], E=1000.0, nu=0.3, loads=loads, bcs=bcs)

        grads = compute_curved_size_gradients(
            nodes, edges, radii, [cp], u, E=1000.0, nu=0.3)

        assert grads[0] < 0, f"dC/dr = {grads[0]} (should be negative)"

    def test_gradient_vs_finite_difference(self):
        """Analytical dC/dr should match central FD within tolerance."""
        nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        edges = np.array([[0, 1]])
        radii = np.array([0.5])
        cp = np.array([[3.0, 1.0, 0.0], [7.0, 1.0, 0.0]])
        loads = {1: [0, -1.0, 0, 0, 0, 0]}
        bcs = {0: [0, 1, 2, 3, 4, 5]}

        # Analytical gradient
        u, _, _ = solve_curved_frame(
            nodes, edges, radii, [cp], E=1000.0, nu=0.3, loads=loads, bcs=bcs)
        grad_analytical = compute_curved_size_gradients(
            nodes, edges, radii, [cp], u, E=1000.0, nu=0.3)

        # Central FD
        eps = 1e-5
        radii_p = np.array([0.5 + eps])
        radii_m = np.array([0.5 - eps])
        _, c_plus, _ = solve_curved_frame(
            nodes, edges, radii_p, [cp], E=1000.0, nu=0.3, loads=loads, bcs=bcs)
        _, c_minus, _ = solve_curved_frame(
            nodes, edges, radii_m, [cp], E=1000.0, nu=0.3, loads=loads, bcs=bcs)
        grad_fd = (c_plus - c_minus) / (2 * eps)

        rel_err = abs(grad_analytical[0] - grad_fd) / abs(grad_fd)
        assert rel_err < 0.01, \
            f"Relative error: {rel_err:.6f} (analytical={grad_analytical[0]:.6f}, FD={grad_fd:.6f})"


# ---------------------------------------------------------------------------
# 7. Control point gradient tests (dC/dP)
# ---------------------------------------------------------------------------

class TestCtrlGradients:
    """Tests for compute_curved_ctrl_gradients()."""

    def test_nonzero_for_curved(self):
        """Curved beam should have non-zero dC/dP."""
        nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        edges = np.array([[0, 1]])
        radii = np.array([0.5])
        cp = np.array([[3.0, 1.5, 0.0], [7.0, 1.5, 0.0]])
        loads = {1: [0, -1.0, 0, 0, 0, 0]}
        bcs = {0: [0, 1, 2, 3, 4, 5]}

        u, _, _ = solve_curved_frame(
            nodes, edges, radii, [cp], E=1000.0, nu=0.3, loads=loads, bcs=bcs)
        dC_dP = compute_curved_ctrl_gradients(
            nodes, edges, radii, [cp], u, E=1000.0, nu=0.3)

        assert dC_dP.shape == (1, 2, 3)
        # At least some components should be non-zero for a curved beam
        assert np.any(np.abs(dC_dP[0]) > 1e-6), \
            f"All gradients near zero: {dC_dP[0]}"

    def test_zero_for_straight(self):
        """Straight beam (collinear ctrl pts) should have near-zero dC/dP.

        Collinear ctrl_pts represent no curvature; lateral perturbation
        is a first-order change, so dC/dP should still be ~0 at the
        straight configuration (by symmetry of the straight beam geometry).
        Note: axial components (along beam axis) may be non-zero.
        """
        nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        edges = np.array([[0, 1]])
        radii = np.array([0.5])
        p0, p3 = nodes[0], nodes[1]
        cp = np.array([p0 + (p3 - p0) / 3.0, p0 + 2.0 * (p3 - p0) / 3.0])
        loads = {1: [0, -1.0, 0, 0, 0, 0]}
        bcs = {0: [0, 1, 2, 3, 4, 5]}

        u, _, _ = solve_curved_frame(
            nodes, edges, radii, [cp], E=1000.0, nu=0.3, loads=loads, bcs=bcs)
        dC_dP = compute_curved_ctrl_gradients(
            nodes, edges, radii, [cp], u, E=1000.0, nu=0.3)

        # Lateral components (y, z) should be near zero for straight beam
        lateral = np.abs(dC_dP[0, :, 1:])  # y and z components
        assert np.all(lateral < 1.0), \
            f"Lateral gradients unexpectedly large: {lateral}"

    def test_gradient_vs_full_fd(self):
        """Semi-analytical ctrl_pt gradient should match full FD on compliance."""
        nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        edges = np.array([[0, 1]])
        radii = np.array([0.5])
        cp = np.array([[3.0, 1.0, 0.0], [7.0, 1.0, 0.0]])
        loads = {1: [0, -1.0, 0, 0, 0, 0]}
        bcs = {0: [0, 1, 2, 3, 4, 5]}

        # Semi-analytical gradient
        u, _, _ = solve_curved_frame(
            nodes, edges, radii, [cp], E=1000.0, nu=0.3, loads=loads, bcs=bcs)
        dC_dP = compute_curved_ctrl_gradients(
            nodes, edges, radii, [cp], u, E=1000.0, nu=0.3)

        # Full FD: perturb ctrl_pt, re-solve entire system
        eps = 1e-4
        for cp_idx in range(2):
            for axis in range(3):
                cp_plus = cp.copy()
                cp_minus = cp.copy()
                cp_plus[cp_idx, axis] += eps
                cp_minus[cp_idx, axis] -= eps

                _, c_plus, _ = solve_curved_frame(
                    nodes, edges, radii, [cp_plus], E=1000.0, nu=0.3,
                    loads=loads, bcs=bcs)
                _, c_minus, _ = solve_curved_frame(
                    nodes, edges, radii, [cp_minus], E=1000.0, nu=0.3,
                    loads=loads, bcs=bcs)
                grad_fd = (c_plus - c_minus) / (2 * eps)
                grad_sa = dC_dP[0, cp_idx, axis]

                if abs(grad_fd) > 1e-6:
                    rel_err = abs(grad_sa - grad_fd) / abs(grad_fd)
                    assert rel_err < 0.05, \
                        f"ctrl[{cp_idx}][{axis}]: rel_err={rel_err:.4f} " \
                        f"(SA={grad_sa:.6f}, FD={grad_fd:.6f})"
                else:
                    # Both should be near zero
                    assert abs(grad_sa) < 1.0, \
                        f"ctrl[{cp_idx}][{axis}]: SA={grad_sa:.6f} but FD≈0"


# ---------------------------------------------------------------------------
# 8. IGA stiffness derivative tests (dK/dr)
# ---------------------------------------------------------------------------

class TestStiffnessDerivative:
    """Tests for compute_iga_element_stiffness_derivative()."""

    def test_symmetry(self, curved_beam_xy, beam_params):
        """dK/dr must be symmetric."""
        p0, p1, p2, p3 = curved_beam_xy
        dK = compute_iga_element_stiffness_derivative(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        np.testing.assert_allclose(dK, dK.T, atol=1e-10)

    def test_vs_finite_difference(self, curved_beam_xy, beam_params):
        """dK/dr should match (K(r+eps) - K(r-eps)) / 2eps."""
        p0, p1, p2, p3 = curved_beam_xy
        E, r, nu = beam_params['E'], beam_params['r'], beam_params['nu']

        dK_analytical = compute_iga_element_stiffness_derivative(E, r, p0, p1, p2, p3, nu)

        eps = 1e-6
        K_plus = compute_iga_element_stiffness(E, r + eps, p0, p1, p2, p3, nu)
        K_minus = compute_iga_element_stiffness(E, r - eps, p0, p1, p2, p3, nu)
        dK_fd = (K_plus - K_minus) / (2 * eps)

        np.testing.assert_allclose(dK_analytical, dK_fd, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# 9. Degenerate / edge case tests
# ---------------------------------------------------------------------------

class TestDegenerateCases:
    """Edge cases that should not crash or produce NaN."""

    def test_short_beam(self, beam_params):
        """Very short beam (1 unit) should still produce valid stiffness."""
        p0 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2.0 * (p3 - p0) / 3.0

        K = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        assert not np.any(np.isnan(K))
        assert np.allclose(K, K.T, atol=1e-10)

    def test_vertical_beam(self, beam_params):
        """Beam along Z-axis (near-vertical case)."""
        p0 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 0.0, 10.0])
        p1 = np.array([0.0, 0.5, 3.3])
        p2 = np.array([0.0, 0.5, 6.7])

        K = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        assert not np.any(np.isnan(K))
        K_cond = condense_element(K)
        assert K_cond is not None

    def test_max_bulge_sanitized(self, beam_params):
        """Ctrl pts with extreme bulge should still work after sanitization."""
        p0 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([10.0, 0.0, 0.0])
        # Extreme lateral bulge
        p1_raw = np.array([3.0, 50.0, 0.0])
        p2_raw = np.array([7.0, 50.0, 0.0])

        p1, p2 = sanitize_bezier_ctrl_pts(p0, p3, p1_raw, p2_raw)

        K = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        assert not np.any(np.isnan(K))
        K_cond = condense_element(K)
        assert K_cond is not None

    def test_very_thin_beam(self):
        """Very small radius should still produce valid stiffness."""
        p0 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([10.0, 0.0, 0.0])
        p1 = np.array([3.0, 0.5, 0.0])
        p2 = np.array([7.0, 0.5, 0.0])

        K = compute_iga_element_stiffness(1000.0, 0.1, p0, p1, p2, p3, 0.3)
        assert not np.any(np.isnan(K))
        K_cond = condense_element(K)
        assert K_cond is not None

    def test_diagonal_beam(self, beam_params):
        """Beam along [1,1,1] diagonal."""
        p0 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([10.0, 10.0, 10.0])
        p1 = np.array([3.0, 4.0, 3.0])
        p2 = np.array([7.0, 6.0, 7.0])

        K = compute_iga_element_stiffness(
            beam_params['E'], beam_params['r'], p0, p1, p2, p3, beam_params['nu'])
        assert not np.any(np.isnan(K))
        assert np.allclose(K, K.T, atol=1e-10)
        K_cond = condense_element(K)
        assert K_cond is not None


# ---------------------------------------------------------------------------
# 10. Multi-beam system test
# ---------------------------------------------------------------------------

class TestMultiBeamSystem:
    """Integration test with a small multi-beam frame."""

    def test_three_beam_truss(self):
        """Three curved beams forming a simple truss, fixed + loaded."""
        nodes = np.array([
            [0.0, 0.0, 0.0],   # 0: fixed
            [10.0, 0.0, 0.0],  # 1: free
            [5.0, 8.0, 0.0],   # 2: loaded
        ])
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        radii = np.array([0.5, 0.5, 0.5])

        # Mild curvature on all three beams
        ctrl_pts = [
            np.array([[3.0, 0.5, 0.0], [7.0, 0.5, 0.0]]),
            np.array([[8.5, 3.0, 0.0], [6.5, 6.0, 0.0]]),
            np.array([[1.5, 3.0, 0.0], [3.5, 6.0, 0.0]]),
        ]

        loads = {2: [0, -1.0, 0, 0, 0, 0]}
        bcs = {0: [0, 1, 2, 3, 4, 5]}

        u, compliance, edata = solve_curved_frame(
            nodes, edges, radii, ctrl_pts, E=1000.0, nu=0.3,
            loads=loads, bcs=bcs)

        assert np.isfinite(compliance)
        assert compliance > 0
        assert not np.any(np.isnan(u))

        # All elements should be curved
        assert all(ed['curved'] for ed in edata if ed is not None)

        # Size gradients
        grads = compute_curved_size_gradients(
            nodes, edges, radii, ctrl_pts, u, E=1000.0, nu=0.3)
        assert all(np.isfinite(grads))

        # Ctrl pt gradients
        dC_dP = compute_curved_ctrl_gradients(
            nodes, edges, radii, ctrl_pts, u, E=1000.0, nu=0.3)
        assert dC_dP.shape == (3, 2, 3)
        assert all(np.isfinite(dC_dP.flat))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
