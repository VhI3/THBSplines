"""
THBSplines — Truncated Hierarchical B-splines in Python.

Public API
----------
The most commonly used names are re-exported here for convenience:

    from THBSplines import HierarchicalSpace, refine
    from THBSplines import hierarchical_mass_matrix, hierarchical_stiffness_matrix

See the individual submodules in ``THBSplines/src/`` for full documentation.
"""

from THBSplines.src.assembly import (
    hierarchical_mass_matrix,
    hierarchical_stiffness_matrix,
    local_mass_matrix,
    local_stiffness_matrix,
)
from THBSplines.src.b_spline import BSpline, UnivariateBSpline
from THBSplines.src.b_spline_numpy import TensorProductBSpline
from THBSplines.src.cartesian_mesh import CartesianMesh
from THBSplines.src.evaluation import (
    check_partition_of_unity,
    create_subdivision_matrix,
    evaluate_hierarchical_basis,
    plot_basis_functions_1d,
)
from THBSplines.src.hierarchical_mesh import HierarchicalMesh
from THBSplines.src.hierarchical_space import HierarchicalSpace
from THBSplines.src.refinement import refine
from THBSplines.src.tensor_product_space import TensorProductSpace, TensorProductSpace2D
