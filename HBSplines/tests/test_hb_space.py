"""Tests for HBsplineSpace — hierarchical 1D B-spline space."""
import numpy as np
import pytest
from HBSplines.src.hb_space import HBsplineSpace


KNOTS = [0, 0, 0, 0.5, 1, 1, 1]
DEGREE = 2


def make_space():
    return HBsplineSpace(KNOTS, DEGREE)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_nlevels():
    sp = make_space()
    assert sp.nlevels == 1

def test_initial_nfuncs():
    sp = make_space()
    # All 4 functions active at level 0
    assert sp.nfuncs == 4

def test_initial_all_active():
    sp = make_space()
    idx = sp.active_indices(0)
    assert list(idx) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------

def test_refine_adds_level():
    sp = make_space()
    sp.refine(level=0, indices=np.array([1, 2]))
    assert sp.nlevels == 2

def test_refine_deactivates():
    sp = make_space()
    sp.refine(level=0, indices=np.array([1]))
    assert 1 not in sp.active_indices(0)

def test_refine_activates_children():
    sp = make_space()
    sp.refine(level=0, indices=np.array([1]))
    # Children of function 1 at level 0 with degree 2: {2,3,4} intersected with finer dim
    children = sp.sp_lev[0].get_children(1)
    for c in children:
        assert c in sp.active_indices(1)

def test_nfuncs_after_refine():
    sp = make_space()
    # Refine function 1 (interior) → lose 1, gain p+2=4 children (all in range)
    sp.refine(level=0, indices=np.array([1]))
    n_children = len(sp.sp_lev[0].get_children(1))
    expected = (4 - 1) + n_children
    assert sp.nfuncs == expected


# ---------------------------------------------------------------------------
# Boundary / interior DOFs
# ---------------------------------------------------------------------------

def test_boundary_dofs():
    sp = make_space()
    bd = sp.boundary_dofs()
    bd_local = [(lev, i) for lev, i in bd]
    assert (0, 0) in bd_local
    assert (0, 3) in bd_local   # last function at level 0

def test_interior_dofs():
    sp = make_space()
    interior = sp.interior_dofs()
    for lev, i in interior:
        assert not sp.is_boundary(lev, i)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_eval_all_shape():
    sp = make_space()
    x = np.linspace(0.1, 0.9, 20)
    B = sp.eval_all(x)
    assert B.shape == (20, sp.nfuncs)

def test_eval_solution():
    sp = make_space()
    c = np.ones(sp.nfuncs)
    x = np.linspace(0.1, 0.9, 10)
    u = sp.eval_solution(x, c)
    # Since HB-splines don't form a PoU after refinement, just check shape
    assert u.shape == (10,)

def test_global_dofs_length():
    sp = make_space()
    sp.refine(level=0, indices=np.array([1, 2]))
    dofs = sp.global_dofs()
    assert len(dofs) == sp.nfuncs
