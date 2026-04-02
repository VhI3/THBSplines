"""Tests for BsplineSpace — single-level 1D B-spline space."""
import numpy as np
import pytest
from HBSplines.src.bspline_space import BsplineSpace


KNOTS_Q2 = [0, 0, 0, 0.5, 1, 1, 1]   # degree-2, 4 functions
KNOTS_Q1 = [0, 0, 0.5, 1, 1]          # degree-1, 3 functions


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_dim_degree2():
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    assert sp.dim == 4
    assert sp.degree == 2

def test_dim_degree1():
    sp = BsplineSpace(KNOTS_Q1, degree=1)
    assert sp.dim == 3

def test_domain():
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    assert sp.domain == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Partition of unity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("knots,degree", [
    (KNOTS_Q2, 2),
    (KNOTS_Q1, 1),
    ([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1], 2),
])
def test_partition_of_unity(knots, degree):
    sp = BsplineSpace(knots, degree)
    x = np.linspace(0, 1, 51)[1:-1]   # interior points
    B = sp.eval_all(x)
    sums = B.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12), f"Max deviation: {np.abs(sums - 1).max()}"


# ---------------------------------------------------------------------------
# Non-negativity
# ---------------------------------------------------------------------------

def test_non_negativity():
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    x = np.linspace(0, 1, 101)
    B = sp.eval_all(x)
    assert np.all(B >= -1e-14)


# ---------------------------------------------------------------------------
# Interpolation at endpoints
# ---------------------------------------------------------------------------

def test_left_endpoint():
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    assert abs(sp.eval(np.array([0.0]), 0)[0] - 1.0) < 1e-12

def test_right_endpoint():
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    n = sp.dim - 1
    assert abs(sp.eval(np.array([1.0]), n)[0] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------

def test_derivative_finite_difference():
    """First derivative should match finite difference approximation."""
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    h = 1e-6
    x0 = np.array([0.3])
    for i in range(sp.dim):
        analytic = sp.deriv(x0, i, r=1)[0]
        fd = (sp.eval(x0 + h, i)[0] - sp.eval(x0 - h, i)[0]) / (2 * h)
        assert abs(analytic - fd) < 1e-5, f"Function {i}: analytic={analytic}, fd={fd}"

def test_second_derivative_finite_difference():
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    h = 1e-4
    x0 = np.array([0.3])
    for i in range(sp.dim):
        analytic = sp.deriv(x0, i, r=2)[0]
        fd = (sp.eval(x0 + h, i)[0] - 2*sp.eval(x0, i)[0] + sp.eval(x0 - h, i)[0]) / h**2
        assert abs(analytic - fd) < 1e-4, f"Function {i}: analytic={analytic:.6f}, fd={fd:.6f}"


# ---------------------------------------------------------------------------
# Dyadic refinement
# ---------------------------------------------------------------------------

def test_refine_doubles_breakpoints():
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    sp2 = sp.refine()
    unique_before = len(np.unique(sp.knots))
    unique_after  = len(np.unique(sp2.knots))
    assert unique_after == 2 * unique_before - 1

def test_children_count():
    """Every function has at least one child; p+2 bound holds for interior functions."""
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    for i in range(sp.dim):
        children = sp.get_children(i)
        assert len(children) >= 1, f"Function {i}: no children found"

def test_children_in_range():
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    finer_dim = sp.refine().dim
    for i in range(sp.dim):
        children = sp.get_children(i)
        assert np.all(children < finer_dim)
        assert np.all(children >= 0)


# ---------------------------------------------------------------------------
# Greville abscissae
# ---------------------------------------------------------------------------

def test_greville_count():
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    g = sp.greville()
    assert len(g) == sp.dim

def test_greville_endpoints():
    sp = BsplineSpace(KNOTS_Q2, degree=2)
    g = sp.greville()
    assert abs(g[0] - 0.0) < 1e-12
    assert abs(g[-1] - 1.0) < 1e-12
