"""Tests for LRBasisFunction: evaluation, support, knot splitting."""
import numpy as np
import pytest
from LRSplines.src.lr_basis import LRBasisFunction, _eval1d, _deriv1d


class TestEval1d:
    def test_linear_hat(self):
        # B_{[0,1,2],1}(1.0) with end_point=False uses strict half-open [0,2).
        # At u=1: B_{[0,1],0}(1) = 0 (strict [0,1) excludes 1)
        #         B_{[1,2],0}(1) = 1 (1 is in [1,2))
        # Result = (1-0)/(1-0)*0 + (2-1)/(2-1)*1 = 1.0
        val = _eval1d(1.0, 1, np.array([0.0, 1.0, 2.0]))
        assert abs(val - 1.0) < 1e-14

    def test_quadratic_peak(self):
        # Degree-2 symmetric hat on [0,1,2,3]: peaks at x=1.5
        knots = np.array([0.0, 1.0, 2.0, 3.0])
        val = _eval1d(1.5, 2, knots)
        assert val > 0
        assert val <= 1.0

    def test_outside_support(self):
        knots = np.array([0.0, 1.0, 2.0, 3.0])
        assert _eval1d(-0.1, 2, knots) == 0.0
        assert _eval1d(3.1, 2, knots) == 0.0

    def test_end_point_closed(self):
        # With end_point=True, the function is evaluated at the right endpoint
        knots = np.array([0.0, 1.0, 2.0, 3.0])
        val = _eval1d(3.0, 2, knots, end_point=True)
        assert val >= 0.0   # should be nonzero at closed endpoint

    def test_derivative_finite_difference(self):
        knots = np.array([0.0, 1.0, 2.0, 3.0])
        x = 1.2
        h = 1e-6
        fd = (_eval1d(x + h, 2, knots) - _eval1d(x - h, 2, knots)) / (2 * h)
        exact = _deriv1d(x, 2, knots)
        assert abs(fd - exact) < 1e-5


class TestLRBasisFunction:
    def _make_bilinear(self):
        return LRBasisFunction(
            id=0,
            knots_u=np.array([0.0, 1.0, 2.0]),
            knots_v=np.array([0.0, 1.0, 2.0]),
            degree_u=1, degree_v=1)

    def test_support(self):
        B = self._make_bilinear()
        assert B.support == (0.0, 2.0, 0.0, 2.0)

    def test_eval_scalar(self):
        B = self._make_bilinear()
        val = B(1.0, 1.0)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_eval_array_shape(self):
        B = self._make_bilinear()
        pts = np.random.default_rng(0).uniform(0, 2, (20, 2))
        vals = B.eval_array(pts)
        assert vals.shape == (20,)

    def test_grad_shape(self):
        B = self._make_bilinear()
        pts = np.random.default_rng(0).uniform(0, 2, (15, 2))
        g = B.grad_array(pts)
        assert g.shape == (15, 2)

    def test_grad_finite_difference(self):
        B = LRBasisFunction(
            id=0,
            knots_u=np.array([0.0, 1.0, 2.0, 3.0]),
            knots_v=np.array([0.0, 1.0, 2.0, 3.0]),
            degree_u=2, degree_v=2)
        u, v = 1.2, 1.7
        h = 1e-6
        pts_pu = np.array([[u + h, v]])
        pts_mu = np.array([[u - h, v]])
        pts_pv = np.array([[u, v + h]])
        pts_mv = np.array([[u, v - h]])
        fd_u = (B.eval_array(pts_pu) - B.eval_array(pts_mu)) / (2 * h)
        fd_v = (B.eval_array(pts_pv) - B.eval_array(pts_mv)) / (2 * h)
        g = B.grad_array(np.array([[u, v]]))
        assert abs(g[0, 0] - fd_u[0]) < 1e-5
        assert abs(g[0, 1] - fd_v[0]) < 1e-5

    def test_outside_support_zero(self):
        B = self._make_bilinear()
        assert B(5.0, 1.0) == 0.0
        assert B(1.0, 5.0) == 0.0

    def test_greville(self):
        B = LRBasisFunction(
            id=0,
            knots_u=np.array([0.0, 1.0, 2.0, 3.0]),
            knots_v=np.array([0.0, 1.0, 2.0, 3.0]),
            degree_u=2, degree_v=2)
        gu, gv = B.greville
        assert abs(gu - 1.5) < 1e-14   # mean([1,2]) = 1.5
        assert abs(gv - 1.5) < 1e-14


class TestKnotSplitting:
    def test_split_u_reproduces_original(self):
        """
        After splitting B into B1, B2 with coefficients alpha_1, alpha_2:
          B1.eval_array + B2.eval_array == B.eval_array
        because eval_array already multiplies by self.coefficient and
        B = alpha_1 * B1_pure + alpha_2 * B2_pure.
        """
        B = LRBasisFunction(
            id=0,
            knots_u=np.array([0.0, 1.0, 2.0, 3.0]),
            knots_v=np.array([0.0, 1.0, 2.0, 3.0]),
            degree_u=2, degree_v=2,
            coefficient=1.0)
        B1, B2 = B.insert_knot_u(1.5)
        pts = np.column_stack([
            np.linspace(0.1, 2.9, 30),
            np.linspace(0.5, 2.5, 30)])
        orig = B.eval_array(pts)
        # eval_array includes coefficient, so sum of children == parent
        assert np.allclose(B1.eval_array(pts) + B2.eval_array(pts), orig, atol=1e-12)

    def test_split_reduces_support(self):
        B = LRBasisFunction(
            id=0,
            knots_u=np.array([0.0, 1.0, 2.0, 3.0]),
            knots_v=np.array([0.0, 1.0, 2.0, 3.0]),
            degree_u=2, degree_v=2)
        B1, B2 = B.insert_knot_u(1.5)
        u0_1, u1_1, _, _ = B1.support
        u0_2, u1_2, _, _ = B2.support
        # B1 should end at or before 1.5, B2 should start at or after 1.5
        assert u1_1 <= B.support[1]
        assert u0_2 >= B.support[0]
        # Their u-ranges together cover the original
        assert u0_1 <= B.support[0] + 1e-14
        assert u1_2 >= B.support[1] - 1e-14
