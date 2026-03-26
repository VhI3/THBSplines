"""Tests for LRMesh: element count, line insertion, element lookup."""
import numpy as np
import pytest
from LRSplines.src.lr_mesh import LRMesh, MeshLine


class TestInitialMesh:
    def test_uniform_element_count(self):
        # 3 breaks in u, 3 breaks in v → 2x2 = 4 elements
        mesh = LRMesh([0, 1, 2], [0, 1, 2])
        assert mesh.nelements == 4

    def test_clamped_knot_vector(self):
        # [0,0,0,1,2,3,3,3] has unique breaks [0,1,2,3] → 3 intervals
        mesh = LRMesh([0, 0, 0, 1, 2, 3, 3, 3], [0, 0, 0, 1, 2, 3, 3, 3])
        assert mesh.nelements == 9   # 3 * 3

    def test_domain(self):
        mesh = LRMesh([0, 1, 2], [0, 1, 3])
        assert mesh.u_domain == (0.0, 2.0)
        assert mesh.v_domain == (0.0, 3.0)

    def test_element_areas_sum_to_domain(self):
        mesh = LRMesh([0, 0, 0, 1, 2, 3, 3, 3], [0, 0, 0, 1, 2, 3, 3, 3])
        total = sum(el.area for el in mesh.elements)
        assert abs(total - 9.0) < 1e-12


class TestLineInsertion:
    def test_vertical_line_splits_elements(self):
        mesh = LRMesh([0, 1, 2], [0, 1, 2])
        assert mesh.nelements == 4
        # Insert a full-width vertical line at u=0.5
        ln = MeshLine(axis=0, value=0.5, start=0.0, end=2.0)
        mesh.insert_line(ln)
        # Each of the two left elements is split → 6 elements
        assert mesh.nelements == 6

    def test_partial_line_no_extra_elements(self):
        # A partial line segment that only cuts one element
        mesh = LRMesh([0, 1, 2], [0, 1, 2])
        ln = MeshLine(axis=0, value=0.5, start=0.0, end=1.0)
        mesh.insert_line(ln)
        # Only the bottom-left element is cut → 5 elements
        assert mesh.nelements == 5

    def test_element_areas_preserved_after_insertion(self):
        mesh = LRMesh([0, 1, 2], [0, 1, 2])
        ln = MeshLine(axis=0, value=0.5, start=0.0, end=2.0)
        mesh.insert_line(ln)
        total = sum(el.area for el in mesh.elements)
        assert abs(total - 4.0) < 1e-12


class TestElementLookup:
    def test_find_element_interior(self):
        mesh = LRMesh([0, 1, 2], [0, 1, 2])
        idx = mesh.find_element(0.5, 0.5)
        el = mesh.elements[idx]
        assert el.u0 == 0.0 and el.u1 == 1.0
        assert el.v0 == 0.0 and el.v1 == 1.0

    def test_find_element_outside_raises(self):
        mesh = LRMesh([0, 1, 2], [0, 1, 2])
        with pytest.raises(ValueError):
            mesh.find_element(5.0, 0.5)


class TestMeshLine:
    def test_contains(self):
        ln = MeshLine(axis=0, value=1.0, start=0.0, end=2.0)
        assert ln.contains(1.0)
        assert not ln.contains(0.0)
        assert not ln.contains(2.0)

    def test_overlaps_open(self):
        ln = MeshLine(axis=1, value=1.5, start=0.0, end=2.0)
        assert ln.overlaps_open(0.0, 1.0)     # [0,1] vs [0,2] → overlap
        assert not ln.overlaps_open(2.0, 3.0)  # no overlap

    def test_bad_start_end_raises(self):
        with pytest.raises(AssertionError):
            MeshLine(axis=0, value=1.0, start=2.0, end=0.0)
