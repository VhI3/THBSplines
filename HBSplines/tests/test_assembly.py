"""Tests for HB-spline FEM assembly and adaptive solver."""
import numpy as np
import pytest
from HBSplines.src.hb_space import HBsplineSpace
from HBSplines.src.assembly import hb_stiffness_matrix, hb_load_vector_full, hb_solve
from HBSplines.problems import smooth_diffusion, boundary_layer_right, interior_layer
from HBSplines import adaptive_solve, AdaptiveSolverSettings


KNOTS = [0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1]
DEGREE = 2


def make_space():
    return HBsplineSpace(KNOTS, DEGREE)


# ---------------------------------------------------------------------------
# Stiffness matrix
# ---------------------------------------------------------------------------

def test_stiffness_shape():
    sp = make_space()
    A = hb_stiffness_matrix(sp, m=1.0, b_adv=0.0)
    n = sp.nfuncs
    assert A.shape == (n, n)

def test_stiffness_symmetric_pure_diffusion():
    """For b=0 the stiffness matrix should be symmetric."""
    sp = make_space()
    A = hb_stiffness_matrix(sp, m=1.0, b_adv=0.0)
    diff = np.abs(A - A.T).max()
    assert diff < 1e-10, f"Asymmetry: {diff}"

def test_stiffness_not_symmetric_with_advection():
    """With b≠0 the advection term breaks symmetry."""
    sp = make_space()
    A = hb_stiffness_matrix(sp, m=1.0, b_adv=5.0)
    diff = np.abs(A - A.T).max()
    assert diff > 1e-8


# ---------------------------------------------------------------------------
# Smooth diffusion solve: u = sin(pi*x)
# ---------------------------------------------------------------------------

def test_smooth_solution_l2_error():
    """L2 error for smooth problem on a fine uniform space should be small."""
    fine_knots = [0, 0, 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1, 1]
    sp = HBsplineSpace(fine_knots, degree=2)
    prob = smooth_diffusion()

    c_full, _, _ = hb_solve(sp, prob.f, prob.u0, prob.uL, prob.m, prob.b)

    x = np.linspace(0, 1, 201)
    u_h = sp.eval_solution(x, c_full)
    u_ex = prob.u_ex(x)
    l2 = np.sqrt(np.trapezoid((u_h - u_ex) ** 2, x))
    assert l2 < 1e-3, f"L2 error too large: {l2:.4e}"


# ---------------------------------------------------------------------------
# Adaptive solver: smooth problem converges
# ---------------------------------------------------------------------------

def test_adaptive_smooth_converges():
    sp = HBsplineSpace([0, 0, 0, 0.5, 1, 1, 1], degree=2)
    prob = smooth_diffusion()
    settings = AdaptiveSolverSettings(max_dofs=200, tol=1e-3, verbose=False)
    result = adaptive_solve(prob, sp, settings)

    assert len(result.history_eta) > 0
    assert result.history_eta[-1] < result.history_eta[0]  # eta decreased


# ---------------------------------------------------------------------------
# Adaptive solver: boundary layer problem
# ---------------------------------------------------------------------------

def test_adaptive_boundary_layer():
    sp = HBsplineSpace([0, 0, 0, 0.5, 1, 1, 1], degree=2)
    prob = boundary_layer_right(m=1.0, b=10.0)
    settings = AdaptiveSolverSettings(max_dofs=150, tol=1e-4, verbose=False)
    result = adaptive_solve(prob, sp, settings)

    # Adaptive should use more levels near the boundary
    assert result.hspace.nlevels >= 2
    # Error should decrease
    assert result.history_eta[-1] < result.history_eta[0]


# ---------------------------------------------------------------------------
# Problems: exact solutions satisfy the PDE
# ---------------------------------------------------------------------------

def test_boundary_layer_rhs_zero():
    """Right boundary layer exact solution has f=0."""
    prob = boundary_layer_right()
    x = np.linspace(0.01, 0.99, 50)
    assert np.allclose(prob.f(x), 0.0)

def test_smooth_diffusion_rhs():
    prob = smooth_diffusion()
    x = np.array([0.5])
    assert abs(prob.f(x)[0] - np.pi**2) < 1e-10
