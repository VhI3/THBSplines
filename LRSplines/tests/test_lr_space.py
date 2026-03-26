"""Tests for LRSplineSpace: construction, refinement, partition of unity."""
import numpy as np
import pytest
from LRSplines.src.lr_spline_space import LRSplineSpace
from LRSplines.src.lr_mesh import MeshLine
from LRSplines.src.refinement import refine, check_lli


def make_biquadratic():
    """Standard biquadratic space on [0,3]^2 with clamped knots."""
    return LRSplineSpace(
        knots_u=[0, 0, 0, 1, 2, 3, 3, 3],
        knots_v=[0, 0, 0, 1, 2, 3, 3, 3],
        degree_u=2, degree_v=2)


class TestInitialSpace:
    def test_nfuncs(self):
        space = make_biquadratic()
        # (8 - 2 - 1) * (8 - 2 - 1) = 5 * 5 = 25
        assert space.nfuncs == 25

    def test_partition_of_unity_initial(self):
        space = make_biquadratic()
        assert space.check_partition_of_unity()

    def test_all_basis_functions_nonneg(self):
        space = make_biquadratic()
        rng = np.random.default_rng(7)
        pts = rng.uniform(0, 3, (50, 2))
        B = space.evaluate(pts)
        assert np.all(B >= -1e-14)

    def test_element_supports_populated(self):
        space = make_biquadratic()
        for el in space.mesh.elements:
            assert len(el.active_functions) > 0

    def test_bilinear_space(self):
        # Clamped degree-1 knots [0,0,1,2,2]: 3 basis functions per direction
        # → 3*3 = 9 total
        space = LRSplineSpace([0, 0, 1, 2, 2], [0, 0, 1, 2, 2], 1, 1)
        assert space.nfuncs == 9

    def test_greville_shape(self):
        space = make_biquadratic()
        g = space.greville_points
        assert g.shape == (25, 2)


class TestRefinement:
    def test_full_line_increases_nfuncs(self):
        space = make_biquadratic()
        n_before = space.nfuncs
        ln = MeshLine(axis=0, value=1.5, start=0.0, end=3.0)
        refine(space, ln)
        assert space.nfuncs > n_before

    def test_partition_of_unity_after_refinement(self):
        space = make_biquadratic()
        ln = MeshLine(axis=0, value=1.5, start=0.0, end=3.0)
        refine(space, ln)
        assert space.check_partition_of_unity()

    def test_lli_after_full_line(self):
        space = make_biquadratic()
        ln = MeshLine(axis=0, value=1.5, start=0.0, end=3.0)
        refine(space, ln)
        assert check_lli(space)

    def test_two_refinements_pou(self):
        space = make_biquadratic()
        refine(space, MeshLine(axis=0, value=1.5, start=0.0, end=3.0))
        refine(space, MeshLine(axis=1, value=1.5, start=0.0, end=3.0))
        assert space.check_partition_of_unity()

    def test_partial_line_pou(self):
        space = make_biquadratic()
        # Insert a partial vertical line that only spans part of the domain
        ln = MeshLine(axis=0, value=0.5, start=0.0, end=2.0)
        refine(space, ln)
        assert space.check_partition_of_unity()

    def test_nfuncs_monotone(self):
        """Each refinement should not decrease nfuncs."""
        space = make_biquadratic()
        prev = space.nfuncs
        for v in [0.5, 1.5, 2.5]:
            refine(space, MeshLine(axis=1, value=v, start=0.0, end=3.0))
            assert space.nfuncs >= prev
            prev = space.nfuncs


class TestEvaluation:
    def test_evaluate_shape(self):
        space = make_biquadratic()
        pts = np.random.default_rng(0).uniform(0, 3, (30, 2))
        B = space.evaluate(pts)
        assert B.shape == (30, space.nfuncs)

    def test_evaluate_grad_shape(self):
        space = make_biquadratic()
        pts = np.random.default_rng(0).uniform(0, 3, (10, 2))
        G = space.evaluate_grad(pts)
        assert G.shape == (10, space.nfuncs, 2)
