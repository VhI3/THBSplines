"""
Residual-based a-posteriori error estimator for HB-spline FEM.

Local cell estimator (strong residual form):

    eta_j^l  =  h_j * || r_h ||_{L2([x_j, x_{j+1}])}

where  r_h(x) = f(x) + m * u_h''(x) - b * u_h'(x)  is the PDE residual
of the computed solution u_h on the active cell [x_j, x_{j+1}] at level l.

Global estimator:

    eta  =  sqrt( sum_{l,j} (eta_j^l)^2 )

Active cells at level l are the knot intervals [xi_j, xi_{j+1}]  (where
xi_j < xi_{j+1}) that are *not* covered by a finer active level.

Reference: MATLAB source ``hLocRes.m`` / ``hGlobRes.m`` in
           arielsboiardi/adAHBsplineFEM.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING, List

import numpy as np

from LRSplines.src.assembly import _gauss_rule

if TYPE_CHECKING:
    from HBSplines.src.hb_space import HBsplineSpace


# ---------------------------------------------------------------------------
# Active cells per level
# ---------------------------------------------------------------------------

def _active_cells(hspace: "HBsplineSpace") -> list[list[tuple[float, float]]]:
    """
    Return, for each level l, the list of (x_lo, x_hi) intervals that are
    active at that level (not refined further).

    A cell [x_j, x_{j+1}] at level l is *active* if it is not covered by
    any active cell at a finer level.  Because refinement is dyadic, we
    detect this by checking whether any level-l+1 knot breakpoint falls
    strictly inside [x_j, x_{j+1}].
    """
    cells_per_level: list[list[tuple[float, float]]] = []

    for lev in range(hspace.nlevels):
        knots_l = hspace.sp_lev[lev].knots
        unique_l = np.unique(knots_l)
        intervals = list(zip(unique_l[:-1], unique_l[1:]))

        if lev < hspace.nlevels - 1:
            # Collect breakpoints of the next level
            knots_next = hspace.sp_lev[lev + 1].knots
            bp_next = set(np.round(np.unique(knots_next), decimals=15))
            active = []
            for x0, x1 in intervals:
                # If no level-(l+1) breakpoint is strictly inside, this cell is active
                has_finer = any(x0 < t < x1 for t in bp_next)
                if not has_finer:
                    active.append((x0, x1))
        else:
            # Finest level: all intervals are active
            active = list(intervals)

        cells_per_level.append(active)

    return cells_per_level


# ---------------------------------------------------------------------------
# Local residual estimator
# ---------------------------------------------------------------------------

def local_residual(
    hspace: "HBsplineSpace",
    c_full: np.ndarray,
    f: Callable,
    m: float,
    b_adv: float,
    gauss_order: int | None = None,
) -> list[np.ndarray]:
    """
    Compute cell-wise residual estimators.

    Parameters
    ----------
    hspace     : HBsplineSpace
    c_full     : coefficient vector for all active DOFs (from hb_solve)
    f          : forcing function  f(x) -> array
    m          : diffusion coefficient
    b_adv      : advection coefficient
    gauss_order: Gauss points (default: degree+3 for accuracy on residual)

    Returns
    -------
    eta : list of length nlevels, each entry is an array of local estimators
          eta[l][j] = h_j * ||r_h||_{L2(cell_j^l)}
    """
    if gauss_order is None:
        gauss_order = hspace.degree + 3

    nodes_ref, weights_ref = _gauss_rule(gauss_order)
    cells_per_level = _active_cells(hspace)
    eta: list[np.ndarray] = []

    for lev, cells in enumerate(cells_per_level):
        eta_lev = np.zeros(len(cells))
        for j, (x0, x1) in enumerate(cells):
            h = x1 - x0
            nodes = 0.5 * h * nodes_ref + 0.5 * (x0 + x1)
            weights = 0.5 * h * weights_ref

            # PDE residual: r_h = f + m*u_h'' - b*u_h'
            f_vals  = np.asarray(f(nodes), dtype=float)
            u_pp    = hspace.eval_solution_deriv(nodes, c_full, r=2)
            u_p     = hspace.eval_solution_deriv(nodes, c_full, r=1)
            r_h     = f_vals + m * u_pp - b_adv * u_p

            # eta_j = h * ||r_h||_{L2}
            eta_lev[j] = h * np.sqrt(float(np.dot(weights, r_h ** 2)))

        eta.append(eta_lev)

    return eta


# ---------------------------------------------------------------------------
# Global estimator
# ---------------------------------------------------------------------------

def global_residual(eta: list[np.ndarray]) -> float:
    """
    Global error estimator:  eta = sqrt( sum_{l,j} eta[l][j]^2 ).
    """
    total = sum(float(np.sum(e ** 2)) for e in eta)
    return float(np.sqrt(total))


# ---------------------------------------------------------------------------
# Map cells → active function indices (for marking → refine bridge)
# ---------------------------------------------------------------------------

def cells_to_functions(
    hspace: "HBsplineSpace",
    level: int,
    cell_indices: np.ndarray,
) -> np.ndarray:
    """
    Return the indices of *active* basis functions at *level* whose support
    overlaps any of the marked cells.

    Parameters
    ----------
    level        : refinement level
    cell_indices : integer array indexing into active_cells(hspace)[level]
    """
    cells = _active_cells(hspace)[level]
    sp_l  = hspace.sp_lev[level]

    marked_funcs = set()
    for j in cell_indices:
        x0, x1 = cells[j]
        for i in hspace.active_indices(level):
            lk = sp_l._local_knots(i)
            # Support of B_i is [lk[0], lk[-1]]; check overlap with (x0, x1)
            if lk[-1] > x0 + 1e-14 and lk[0] < x1 - 1e-14:
                marked_funcs.add(i)

    return np.array(sorted(marked_funcs), dtype=int)
