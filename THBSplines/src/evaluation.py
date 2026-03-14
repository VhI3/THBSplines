"""
Evaluation utilities for THB-spline spaces.

This module provides functions for evaluating the hierarchical basis on a
grid of points, verifying the partition-of-unity property, and visualising
individual basis functions.

These are primarily for analysis and debugging; they are not used in the
core assembly routines.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from THBSplines.src.hierarchical_space import HierarchicalSpace
from THBSplines.src.refinement import refine


def create_subdivision_matrix(hspace: HierarchicalSpace, mode: str = "full") -> dict:
    """
    Module-level convenience wrapper around ``HierarchicalSpace.create_subdivision_matrix``.

    Returns the matrices that express each active THB-spline in terms of the
    finest-level B-splines.  See ``HierarchicalSpace.create_subdivision_matrix``
    for full documentation.

    Parameters
    ----------
    hspace : HierarchicalSpace
    mode   : ``'full'`` or ``'reduced'``

    Returns
    -------
    dict mapping level → sparse matrix
    """
    return hspace.create_subdivision_matrix(mode)


def evaluate_hierarchical_basis(
    hspace: HierarchicalSpace,
    points: np.ndarray,
) -> np.ndarray:
    """
    Evaluate all active THB-spline basis functions at a set of points.

    Uses the subdivision matrix to express each active THB-spline as a linear
    combination of finest-level B-splines:

        B̃_i(x) = Σ_j C_{ji} · B^L_j(x)

    where C is the subdivision matrix at the finest level.

    Parameters
    ----------
    hspace : HierarchicalSpace
    points : array of shape (N, d) — the evaluation points

    Returns
    -------
    np.ndarray of shape (N, nfuncs)
        ``result[k, i]`` is the value of the i-th active basis function at ``points[k]``.
    """
    points = np.asarray(points, dtype=np.float64)
    N      = points.shape[0]
    n      = hspace.nfuncs
    L      = hspace.nlevels - 1  # finest level index

    # Subdivision matrix: expresses active THB-splines in terms of finest B-splines
    C = hspace.create_subdivision_matrix("full")[L].toarray()  # (nfine, nfuncs)

    # Evaluate all finest-level B-splines at all points
    fine_space  = hspace.spaces[L]
    nfine       = fine_space.nfuncs
    B_fine      = np.zeros((N, nfine), dtype=np.float64)
    for j in range(nfine):
        Bj          = fine_space.construct_B_spline(j)
        B_fine[:, j] = Bj(points)  # shape (N,)

    # Linear combination: B̃_i = Σ_j C_{ji} B^L_j
    # B_thb[k, i] = Σ_j B_fine[k, j] * C[j, i]
    return B_fine @ C  # (N, nfuncs)


def check_partition_of_unity(
    hspace: HierarchicalSpace,
    n_pts: int = 20,
) -> bool:
    """
    Verify numerically that the active THB-splines form a partition of unity.

    A set of functions {Bᵢ} is a partition of unity if:

        Σᵢ Bᵢ(x) = 1   for all x ∈ Ω

    This property is guaranteed by construction for THB-splines.

    Parameters
    ----------
    hspace : HierarchicalSpace
    n_pts  : number of evaluation points per direction

    Returns
    -------
    bool
        True if the partition-of-unity condition holds to within 1e-10.
    """
    # Build a regular grid spanning the parametric domain
    fine_mesh = hspace.mesh.meshes[-1]
    axes      = [
        np.linspace(fine_mesh.knots[j][0], fine_mesh.knots[j][-1], n_pts)
        for j in range(hspace.dim)
    ]
    grids  = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    points = grids.reshape(-1, hspace.dim)

    B_vals = evaluate_hierarchical_basis(hspace, points)
    sums   = B_vals.sum(axis=1)  # should be 1 everywhere

    ok = bool(np.allclose(sums, 1.0, atol=1e-10))
    if not ok:
        max_err = float(np.max(np.abs(sums - 1.0)))
        print(f"Partition of unity FAILED  |  max error = {max_err:.3e}")
    return ok


def plot_basis_functions_1d(
    hspace: HierarchicalSpace,
    n_pts: int = 200,
    ax=None,
) -> None:
    """
    Plot all active THB-spline basis functions for a 1-D space.

    Parameters
    ----------
    hspace : HierarchicalSpace with ``dim == 1``
    n_pts  : number of evaluation points
    ax     : optional Matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if hspace.dim != 1:
        raise ValueError("plot_basis_functions_1d requires a 1-D space.")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    fine_mesh = hspace.mesh.meshes[-1]
    x         = np.linspace(float(fine_mesh.knots[0][0]),
                            float(fine_mesh.knots[0][-1]), n_pts)
    pts       = x[:, np.newaxis]  # (n_pts, 1)

    B = evaluate_hierarchical_basis(hspace, pts)  # (n_pts, nfuncs)

    for i in range(hspace.nfuncs):
        ax.plot(x, B[:, i], label=f"B̃_{i}")

    # Plot the sum to verify partition of unity
    ax.plot(x, B.sum(axis=1), "k--", linewidth=2, label="Sum (=1)")
    ax.set_xlabel("x")
    ax.set_ylabel("B(x)")
    ax.set_title("THB-spline basis functions")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()
