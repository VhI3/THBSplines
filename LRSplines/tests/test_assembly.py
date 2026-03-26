"""Tests for LR B-spline FEM matrix assembly."""
import numpy as np
import pytest
from LRSplines.src.lr_spline_space import LRSplineSpace
from LRSplines.src.lr_mesh import MeshLine
from LRSplines.src.refinement import refine
from LRSplines.src.assembly import lr_mass_matrix, lr_stiffness_matrix, lr_load_vector


def make_space():
    space = LRSplineSpace([0, 0, 0, 1, 2, 3, 3, 3],
                           [0, 0, 0, 1, 2, 3, 3, 3], 2, 2)
    return space


def make_refined_space():
    space = make_space()
    refine(space, MeshLine(axis=0, value=1.5, start=0.0, end=3.0))
    refine(space, MeshLine(axis=1, value=1.5, start=0.0, end=3.0))
    return space


class TestMassMatrix:
    def test_shape(self):
        space = make_space()
        M = lr_mass_matrix(space)
        assert M.shape == (space.nfuncs, space.nfuncs)

    def test_symmetry(self):
        space = make_space()
        M = lr_mass_matrix(space)
        diff = np.max(np.abs(M.toarray() - M.toarray().T))
        assert diff < 1e-12

    def test_positive_definite(self):
        space = make_space()
        M = lr_mass_matrix(space)
        eigvals = np.linalg.eigvalsh(M.toarray())
        assert np.all(eigvals >= -1e-12)

    def test_row_sums(self):
        """For a partition of unity, sum_j M_ij = ∫ B_i dΩ."""
        space = make_space()
        M = lr_mass_matrix(space)
        # Grand sum should equal area of [0,3]^2 = 9
        total = M.toarray().sum()
        assert abs(total - 9.0) < 1e-6

    def test_row_sums_refined(self):
        space = make_refined_space()
        M = lr_mass_matrix(space)
        total = M.toarray().sum()
        assert abs(total - 9.0) < 1e-6


class TestStiffnessMatrix:
    def test_shape(self):
        space = make_space()
        A = lr_stiffness_matrix(space)
        assert A.shape == (space.nfuncs, space.nfuncs)

    def test_symmetry(self):
        space = make_space()
        A = lr_stiffness_matrix(space)
        diff = np.max(np.abs(A.toarray() - A.toarray().T))
        assert diff < 1e-12

    def test_positive_semidefinite(self):
        space = make_space()
        A = lr_stiffness_matrix(space)
        eigvals = np.linalg.eigvalsh(A.toarray())
        assert np.all(eigvals >= -1e-10)

    def test_symmetry_after_refinement(self):
        space = make_refined_space()
        A = lr_stiffness_matrix(space)
        diff = np.max(np.abs(A.toarray() - A.toarray().T))
        assert diff < 1e-12


class TestLoadVector:
    def test_shape(self):
        space = make_space()
        f = lr_load_vector(space, lambda pts: np.ones(len(pts)))
        assert f.shape == (space.nfuncs,)

    def test_constant_one_sums_to_area(self):
        """∫ B_i * 1 dΩ summed over all i = area (partition of unity)."""
        space = make_space()
        f = lr_load_vector(space, lambda pts: np.ones(len(pts)))
        assert abs(f.sum() - 9.0) < 1e-6
