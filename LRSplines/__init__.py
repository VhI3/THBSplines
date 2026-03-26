"""
LRSplines — Pure NumPy/SciPy implementation of Locally Refined (LR) B-splines.

Quick start
-----------
>>> import LRSplines as lr
>>> space = lr.LRSplineSpace([0,0,0,1,2,3,3,3], [0,0,0,1,2,3,3,3], 2, 2)
>>> ln = lr.MeshLine(axis=0, value=1.5, start=0.0, end=3.0)
>>> lr.refine(space, ln)
>>> M = lr.lr_mass_matrix(space)
"""

from LRSplines.src.lr_mesh import LRMesh, MeshLine, Element
from LRSplines.src.lr_basis import LRBasisFunction
from LRSplines.src.lr_spline_space import LRSplineSpace
from LRSplines.src.refinement import refine, refine_region, check_lli
from LRSplines.src.assembly import lr_mass_matrix, lr_stiffness_matrix, lr_load_vector
from LRSplines.src.evaluation import (
    evaluate_lr_basis,
    check_partition_of_unity,
    plot_basis_functions,
    plot_mesh_and_supports,
    plot_partition_of_unity,
)

__all__ = [
    "LRMesh", "MeshLine", "Element",
    "LRBasisFunction",
    "LRSplineSpace",
    "refine", "refine_region", "check_lli",
    "lr_mass_matrix", "lr_stiffness_matrix", "lr_load_vector",
    "evaluate_lr_basis", "check_partition_of_unity",
    "plot_basis_functions", "plot_mesh_and_supports", "plot_partition_of_unity",
]
