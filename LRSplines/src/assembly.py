"""
FEM matrix assembly for LR B-spline spaces.

Both the mass matrix  M_{ij} = ∫ B_i B_j dΩ  and the stiffness matrix
A_{ij} = ∫ ∇B_i · ∇B_j dΩ  are assembled element-by-element using
Gauss–Legendre quadrature.

For each element e = [u0, u1] × [v0, v1]:
  - The integration is performed on the reference square [-1, 1]^2.
  - The mapping to physical coordinates is:
        u = 0.5*(u0+u1) + 0.5*(u1-u0)*xi
        v = 0.5*(v0+v1) + 0.5*(v1-v0)*eta
  - The Jacobian determinant is:
        J = 0.25 * (u1-u0) * (v1-v0)
  - Only the basis functions with support on e contribute to the element
    matrices (sparse assembly).

Because the parametric domain equals the physical domain here (no geometry
mapping), ∇_phys B_i = ∇_param B_i.  For problems with a nontrivial
geometry map, the chain rule must be applied via the Jacobian matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import TYPE_CHECKING

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):   # type: ignore[misc]
        return x

if TYPE_CHECKING:
    from LRSplines.src.lr_spline_space import LRSplineSpace


# ---------------------------------------------------------------------------
# Gauss–Legendre nodes and weights on [-1, 1]
# ---------------------------------------------------------------------------

def _gauss_rule(order: int):
    """
    Return (nodes, weights) for Gauss–Legendre quadrature on [-1, 1]
    with ``order`` points.  Exact for polynomials of degree <= 2*order-1.
    """
    return np.polynomial.legendre.leggauss(order)


# ---------------------------------------------------------------------------
# Internal shared assembly
# ---------------------------------------------------------------------------

def _assemble(space: 'LRSplineSpace',
              kind: str,
              gauss_order: int) -> sp.csr_matrix:
    """
    Assemble mass (kind='mass') or stiffness (kind='stiffness') matrix.

    Parameters
    ----------
    space       : LRSplineSpace
    kind        : 'mass' or 'stiffness'
    gauss_order : int  number of Gauss points per direction per element

    Returns
    -------
    scipy.sparse.csr_matrix, shape (nfuncs, nfuncs)
    """
    n = space.nfuncs
    nodes_1d, weights_1d = _gauss_rule(gauss_order)
    # 2-D tensor product rule
    xi, eta = np.meshgrid(nodes_1d, nodes_1d, indexing='ij')
    xi  = xi.ravel()
    eta = eta.ravel()
    w2d = np.outer(weights_1d, weights_1d).ravel()

    rows, cols, vals = [], [], []

    for el in tqdm(space.mesh.elements,
                   desc=f"{kind} matrix",
                   leave=False):
        local_idx = el.active_functions
        if len(local_idx) == 0:
            continue

        # Map quadrature nodes from [-1,1]^2 to [u0,u1]x[v0,v1]
        u_mid = 0.5 * (el.u0 + el.u1)
        v_mid = 0.5 * (el.v0 + el.v1)
        du    = 0.5 * (el.u1 - el.u0)
        dv    = 0.5 * (el.v1 - el.v0)
        jac   = du * dv           # Jacobian det (the 0.5^2 is in du, dv)

        u_q = u_mid + du * xi
        v_q = v_mid + dv * eta
        pts_q = np.column_stack([u_q, v_q])   # (n_q, 2)

        # Evaluate basis functions at quadrature points
        # Only the functions active on this element are non-zero
        n_local = len(local_idx)
        B_local  = np.zeros((len(pts_q), n_local))
        dB_local = np.zeros((len(pts_q), n_local, 2))

        for loc, g_idx in enumerate(local_idx):
            bf = space.basis[g_idx]
            B_local[:, loc] = bf.eval_array(pts_q)
            if kind == 'stiffness':
                dB_local[:, loc, :] = bf.grad_array(pts_q)

        if kind == 'mass':
            # M_ij = ∫ Bi Bj dΩ ≈ Σ_q w_q * Bi(q) * Bj(q) * J
            for a in range(n_local):
                for b in range(a, n_local):
                    val = float(np.dot(w2d, B_local[:, a] * B_local[:, b]) * jac)
                    if abs(val) > 1e-16:
                        gi, gj = local_idx[a], local_idx[b]
                        rows.append(gi);  cols.append(gj);  vals.append(val)
                        if a != b:
                            rows.append(gj); cols.append(gi); vals.append(val)
        else:
            # A_ij = ∫ ∇Bi · ∇Bj dΩ ≈ Σ_q w_q * (∇Bi·∇Bj)(q) * J
            for a in range(n_local):
                for b in range(a, n_local):
                    dot_prod = np.sum(dB_local[:, a, :] * dB_local[:, b, :],
                                      axis=1)
                    val = float(np.dot(w2d, dot_prod) * jac)
                    if abs(val) > 1e-16:
                        gi, gj = local_idx[a], local_idx[b]
                        rows.append(gi);  cols.append(gj);  vals.append(val)
                        if a != b:
                            rows.append(gj); cols.append(gi); vals.append(val)

    mat = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # Duplicates are summed by coo→csr conversion automatically
    return mat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lr_mass_matrix(space: 'LRSplineSpace',
                   gauss_order: int | None = None) -> sp.csr_matrix:
    """
    Assemble the mass matrix  M_{ij} = ∫ B_i(x) B_j(x) dΩ.

    Parameters
    ----------
    space       : LRSplineSpace
    gauss_order : int, optional
        Number of Gauss points per direction per element.
        Defaults to ceil((max_degree + 1) / 2) + 1, which is exact for
        the bilinear products arising from degree-p basis functions.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (nfuncs, nfuncs)
    """
    if gauss_order is None:
        gauss_order = max(space.degree_u, space.degree_v) + 1
    return _assemble(space, 'mass', gauss_order)


def lr_stiffness_matrix(space: 'LRSplineSpace',
                        gauss_order: int | None = None) -> sp.csr_matrix:
    """
    Assemble the stiffness matrix  A_{ij} = ∫ ∇B_i · ∇B_j dΩ.

    Parameters
    ----------
    space       : LRSplineSpace
    gauss_order : int, optional
        Defaults to max_degree + 1 (exact for degree-p gradients).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (nfuncs, nfuncs)
    """
    if gauss_order is None:
        gauss_order = max(space.degree_u, space.degree_v) + 1
    return _assemble(space, 'stiffness', gauss_order)


def lr_load_vector(space: 'LRSplineSpace',
                   f_func,
                   gauss_order: int | None = None) -> np.ndarray:
    """
    Assemble the load vector  f_i = ∫ f(x) B_i(x) dΩ.

    This uses the same Gauss quadrature as the matrix assembly,
    giving an accurate right-hand side.

    Parameters
    ----------
    space       : LRSplineSpace
    f_func      : callable  f(pts) -> ndarray shape (N,)
        The source function evaluated at an (N, 2) array of points.
    gauss_order : int, optional

    Returns
    -------
    ndarray, shape (nfuncs,)
    """
    if gauss_order is None:
        gauss_order = max(space.degree_u, space.degree_v) + 1

    n = space.nfuncs
    nodes_1d, weights_1d = _gauss_rule(gauss_order)
    xi, eta = np.meshgrid(nodes_1d, nodes_1d, indexing='ij')
    xi  = xi.ravel()
    eta = eta.ravel()
    w2d = np.outer(weights_1d, weights_1d).ravel()

    fvec = np.zeros(n)

    for el in tqdm(space.mesh.elements, desc="load vector", leave=False):
        local_idx = el.active_functions
        if len(local_idx) == 0:
            continue

        u_mid = 0.5 * (el.u0 + el.u1)
        v_mid = 0.5 * (el.v0 + el.v1)
        du    = 0.5 * (el.u1 - el.u0)
        dv    = 0.5 * (el.v1 - el.v0)
        jac   = du * dv

        u_q = u_mid + du * xi
        v_q = v_mid + dv * eta
        pts_q = np.column_stack([u_q, v_q])

        f_q = np.asarray(f_func(pts_q), dtype=float)   # (n_q,)

        for g_idx in local_idx:
            bf  = space.basis[g_idx]
            b_q = bf.eval_array(pts_q)
            fvec[g_idx] += float(np.dot(w2d, f_q * b_q) * jac)

    return fvec
