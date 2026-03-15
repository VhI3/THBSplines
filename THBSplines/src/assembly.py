"""
Finite-element matrix assembly for THB-spline spaces.

This module assembles the two fundamental matrices needed for Galerkin
finite-element methods with THB-splines:

  **Mass matrix**      M_{ij} = ∫_Ω Bᵢ(x) Bⱼ(x) dx
  **Stiffness matrix** A_{ij} = ∫_Ω ∇Bᵢ(x) · ∇Bⱼ(x) dx

Assembly strategy
-----------------
Because THB-splines live on a *hierarchical* mesh, the integration domain Ω
is partitioned level by level:

    Ω = ∪_ℓ Ω_ℓ   (disjoint union of active cells at each level)

For each level ℓ we assemble a *local* matrix over the tensor-product space
V_ℓ and then transform it to the global hierarchical DOF ordering via the
subdivision matrix C[ℓ]:

    M_global += C[ℓ]ᵀ · M_local[ℓ] · C[ℓ]

Quadrature
----------
Gauss–Legendre quadrature with ``order`` points per direction is used.  If
``order`` is not specified, an exact rule is chosen: ``degree + 1`` points
suffice for the mass matrix (integrand degree ≤ 2p) and the stiffness matrix
(integrand degree ≤ 2p-2).

The reference rule on [-1, 1] is mapped to each physical cell by the affine
map  x = 0.5·(ξ+1)·(b-a) + a  with Jacobian (b-a)/2 per direction.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from THBSplines.src.b_spline_numpy import integrate, integrate_grad


# ---------------------------------------------------------------------------
# High-level assemblers
# ---------------------------------------------------------------------------

def hierarchical_mass_matrix(
    T,
    order: int | None = None,
    integration_region: np.ndarray | None = None,
    mode: str = "reduced",
) -> sp.lil_matrix:
    """
    Assemble the global hierarchical mass matrix.

        M_{ij} = ∫_Ω Bᵢ(x) Bⱼ(x) dx

    Parameters
    ----------
    T                  : HierarchicalSpace
    order              : Gauss quadrature order (points per direction per cell).
                         Defaults to ``degree + 1`` (exact for polynomials up to 2p).
    integration_region : optional array of shape (d, 2).  If given, only cells
                         inside this axis-aligned box are integrated.
    mode               : ``'reduced'`` (faster) or ``'full'`` — passed to
                         ``create_subdivision_matrix``.

    Returns
    -------
    scipy.sparse.lil_matrix of shape (nfuncs, nfuncs)
    """
    return _assemble(T, order, integration_region, mode, matrix_type="mass")


def hierarchical_stiffness_matrix(
    T,
    order: int | None = None,
    integration_region: np.ndarray | None = None,
    mode: str = "reduced",
) -> sp.lil_matrix:
    """
    Assemble the global hierarchical stiffness matrix.

        A_{ij} = ∫_Ω ∇Bᵢ(x) · ∇Bⱼ(x) dx

    Parameters
    ----------
    T                  : HierarchicalSpace
    order              : Gauss quadrature order.
    integration_region : optional integration sub-domain.
    mode               : subdivision matrix mode.

    Returns
    -------
    scipy.sparse.lil_matrix of shape (nfuncs, nfuncs)
    """
    return _assemble(T, order, integration_region, mode, matrix_type="stiffness")


def _assemble(T, order, integration_region, mode, matrix_type):
    """
    Shared assembly loop for both mass and stiffness matrices.

    The algorithm accumulates contributions level by level:
      for ℓ = 0, …, L-1:
          M_global += C[ℓ]ᵀ @ M_local[ℓ] @ C[ℓ]
    """
    mesh = T.mesh
    n    = T.nfuncs
    M    = sp.lil_matrix((n, n), dtype=np.float64)

    C         = T.create_subdivision_matrix(mode)
    ndofs_cum = 0  # cumulative DOF count up to and including current level

    for level in range(mesh.nlevels):
        ndofs_cum += T.nfuncs_level[level]

        if mesh.nel_per_level[level] == 0:
            continue  # no active cells at this level — skip

        # Restrict integration to a sub-region if requested
        elem_indices = (
            T.refine_in_rectangle(integration_region, level)
            if integration_region is not None
            else None
        )

        # Assemble the level-local matrix
        if matrix_type == "mass":
            M_local = local_mass_matrix(T, level, order, elem_indices)
        else:
            M_local = local_stiffness_matrix(T, level, order, elem_indices)

        # Add the global contribution via the subdivision matrix
        dof_range = range(ndofs_cum)
        ix        = np.ix_(dof_range, dof_range)
        M[ix]    += C[level].T @ M_local @ C[level]

    return M


# ---------------------------------------------------------------------------
# Level-local assemblers
# ---------------------------------------------------------------------------

def local_mass_matrix(
    T,
    level: int,
    order: int | None = None,
    element_indices: np.ndarray | None = None,
) -> sp.lil_matrix:
    """
    Assemble the mass matrix for the tensor-product space at ``level``.

        M^ℓ_{ij} = ∫_{Ω_ℓ} Bᵢ(x) Bⱼ(x) dx

    Only the active cells at this level are integrated.

    Parameters
    ----------
    T               : HierarchicalSpace
    level           : refinement level
    order           : Gauss quadrature order (default: ``degree + 1``)
    element_indices : restrict integration to this subset of cells.
                      If None, all active cells at this level are used.

    Returns
    -------
    scipy.sparse.lil_matrix of shape (nfuncs_level, nfuncs_level)
    """
    if element_indices is None:
        element_indices = np.arange(T.mesh.meshes[level].nelems, dtype=np.int64)

    active_cells = np.intersect1d(T.mesh.aelem_level[level], element_indices)
    cells        = T.mesh.meshes[level].cells[active_cells]
    n            = T.spaces[level].nfuncs
    M            = sp.lil_matrix((n, n), dtype=np.float64)

    if order is None:
        order = int(T.spaces[level].degrees[0]) + 1

    pts_ref, wts_ref = np.polynomial.legendre.leggauss(order)

    for cell in tqdm(cells, desc=f"Mass matrix  level={level}"):
        qp, qw, area = _translate_points(pts_ref, cell, wts_ref)
        active_bf    = T.spaces[level].get_functions_on_rectangle(cell)

        # Only the upper triangle is computed; symmetry fills the lower half
        for idx_i, i in enumerate(active_bf):
            Bi     = T.spaces[level].construct_B_spline(i)
            bi_val = Bi(qp)

            for j in active_bf[idx_i:]:
                Bj     = T.spaces[level].construct_B_spline(j)
                bj_val = Bj(qp)
                val    = integrate(bi_val, bj_val, qw, area, cell.shape[0])
                M[i, j] += val
                if i != j:
                    M[j, i] += val

    return M


def local_stiffness_matrix(
    T,
    level: int,
    order: int | None = None,
    element_indices: np.ndarray | None = None,
) -> sp.lil_matrix:
    """
    Assemble the stiffness matrix for the tensor-product space at ``level``.

        A^ℓ_{ij} = ∫_{Ω_ℓ} ∇Bᵢ(x) · ∇Bⱼ(x) dx

    Parameters
    ----------
    T               : HierarchicalSpace
    level           : refinement level
    order           : Gauss quadrature order (default: ``degree + 1``)
    element_indices : restrict to a subset of active cells.

    Returns
    -------
    scipy.sparse.lil_matrix of shape (nfuncs_level, nfuncs_level)
    """
    if element_indices is None:
        element_indices = np.arange(T.mesh.meshes[level].nelems, dtype=np.int64)

    active_cells = np.intersect1d(T.mesh.aelem_level[level], element_indices)
    cells        = T.mesh.meshes[level].cells[active_cells]
    n            = T.spaces[level].nfuncs
    M            = sp.lil_matrix((n, n), dtype=np.float64)

    if order is None:
        order = int(T.spaces[level].degrees[0]) + 1

    pts_ref, wts_ref = np.polynomial.legendre.leggauss(order)

    for cell in tqdm(cells, desc=f"Stiffness matrix  level={level}"):
        qp, qw, area = _translate_points(pts_ref, cell, wts_ref)
        active_bf    = T.spaces[level].get_functions_on_rectangle(cell)

        for idx_i, i in enumerate(active_bf):
            Bi      = T.spaces[level].construct_B_spline(i)
            bi_grad = Bi.grad(qp)          # shape (Q, d)

            for j in active_bf[idx_i:]:
                Bj      = T.spaces[level].construct_B_spline(j)
                bj_grad = Bj.grad(qp)      # shape (Q, d)
                val     = integrate_grad(bi_grad, bj_grad, qw, area, cell.shape[0])
                M[i, j] += val
                if i != j:
                    M[j, i] += val

    return M


# ---------------------------------------------------------------------------
# Quadrature helper
# ---------------------------------------------------------------------------

def _translate_points(
    points: np.ndarray,
    cell: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Map 1-D Gauss points from the reference interval [-1, 1] to a d-dimensional
    physical cell and form the tensor-product quadrature rule.

    The affine map in direction j is:
        x_j = 0.5 * (ξ + 1) * (b_j - a_j) + a_j

    The full d-D Jacobian is  area / 2^d.

    Parameters
    ----------
    points  : 1-D Gauss points on [-1, 1], shape (Q,)
    cell    : physical cell, shape (d, 2), where cell[j] = [a_j, b_j]
    weights : 1-D Gauss weights, shape (Q,)

    Returns
    -------
    qp   : physical quadrature points,  shape (Q^d, d)
    qw   : quadrature weights,           shape (Q^d,)
    area : cell volume = ∏_j (b_j - a_j)
    """
    dim = cell.shape[0]

    # Map reference points to physical coordinates per direction
    phys_pts = np.array([
        0.5 * (points + 1.0) * (cell[j, 1] - cell[j, 0]) + cell[j, 0]
        for j in range(dim)
    ])  # shape (d, Q)

    # Tensor-product: all combinations of physical points
    grids = np.stack(np.meshgrid(*phys_pts), -1).reshape(-1, dim)  # (Q^d, d)

    # Tensor-product weights (reference interval; Jacobian applied below)
    wt_grids = np.stack(np.meshgrid(*[weights] * dim), -1).reshape(-1, dim)
    qw       = np.prod(wt_grids, axis=1)  # (Q^d,)

    area = float(np.prod(cell[:, 1] - cell[:, 0]))  # physical cell volume

    return grids, qw, area
