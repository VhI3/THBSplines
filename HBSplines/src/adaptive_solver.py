"""
Adaptive HB-spline FEM solver.

Implements the standard AFEM loop:

    while not converged:
        1. SOLVE    — assemble and solve the linear system
        2. ESTIMATE — compute local residual estimators
        3. MARK     — select cells/functions to refine
        4. REFINE   — update the HB-spline space

Stopping criteria (any one triggers termination):
  - max_dofs      : total active DOFs exceed this limit
  - max_iter      : iteration count exceeds this limit
  - tol           : global estimator eta < tol

Usage
-----
    from HBSplines import HBsplineSpace, adaptive_solve, AdaptiveSolverSettings
    from HBSplines.problems import boundary_layer_right

    prob   = boundary_layer_right()
    hspace = HBsplineSpace([0,0,0,0.5,1,1,1], degree=2)
    settings = AdaptiveSolverSettings(max_dofs=300, tol=1e-4)
    result = adaptive_solve(prob, hspace, settings)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from HBSplines.src.assembly import hb_solve
from HBSplines.src.estimator import local_residual, global_residual, cells_to_functions
from HBSplines.src.marking import dorfler_mark, max_mark
from HBSplines.src.hb_space import HBsplineSpace


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveSolverSettings:
    """Configuration for the adaptive solver.

    Parameters
    ----------
    max_dofs    : stop when nfuncs > max_dofs
    max_iter    : stop after this many iterations
    tol         : stop when global estimator < tol
    theta       : marking parameter (Dorfler bulk fraction or max fraction)
    strategy    : 'dorfler' or 'max'
    gauss_order : quadrature order (None = degree+2)
    verbose     : print iteration summary
    """
    max_dofs    : int   = 500
    max_iter    : int   = 30
    tol         : float = 1e-6
    theta       : float = 0.5
    strategy    : str   = "dorfler"   # "dorfler" | "max"
    gauss_order : int | None = None
    verbose     : bool  = True


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveSolverResult:
    """Output from the adaptive solver.

    Attributes
    ----------
    hspace      : final HBsplineSpace after all refinements
    c_full      : final coefficient vector (all active DOFs)
    history_dofs : list of nfuncs at each iteration
    history_eta  : list of global estimator at each iteration
    n_iter       : total iterations performed
    converged    : True if tol was reached
    """
    hspace       : HBsplineSpace
    c_full       : np.ndarray
    history_dofs : List[int]   = field(default_factory=list)
    history_eta  : List[float] = field(default_factory=list)
    n_iter       : int         = 0
    converged    : bool        = False


# ---------------------------------------------------------------------------
# Main adaptive loop
# ---------------------------------------------------------------------------

def adaptive_solve(
    prob,
    hspace: HBsplineSpace,
    settings: AdaptiveSolverSettings | None = None,
) -> AdaptiveSolverResult:
    """
    Run the adaptive AFEM loop for problem *prob* on space *hspace*.

    Parameters
    ----------
    prob     : Problem instance with fields m, b, f, u0, uL
    hspace   : initial HBsplineSpace (modified in-place)
    settings : AdaptiveSolverSettings (uses defaults if None)

    Returns
    -------
    AdaptiveSolverResult
    """
    if settings is None:
        settings = AdaptiveSolverSettings()

    result = AdaptiveSolverResult(hspace=hspace, c_full=np.array([]))

    mark_fn: Callable = (
        lambda eta: dorfler_mark(eta, settings.theta)
        if settings.strategy == "dorfler"
        else lambda eta: max_mark(eta, settings.theta)
    )

    for it in range(1, settings.max_iter + 1):
        result.n_iter = it

        # 1. SOLVE
        c_full, _, _ = hb_solve(
            hspace,
            f=prob.f,
            u0=prob.u0,
            uL=prob.uL,
            m=prob.m,
            b_adv=prob.b,
            gauss_order=settings.gauss_order,
        )
        result.c_full = c_full

        # 2. ESTIMATE
        eta_loc = local_residual(
            hspace, c_full, prob.f, prob.m, prob.b,
            gauss_order=settings.gauss_order,
        )
        eta_glob = global_residual(eta_loc)

        result.history_dofs.append(hspace.nfuncs)
        result.history_eta.append(eta_glob)

        if settings.verbose:
            print(
                f"  iter {it:3d} | DOFs {hspace.nfuncs:4d} | "
                f"eta = {eta_glob:.4e}"
            )

        # 3. Check stopping criteria
        if eta_glob < settings.tol:
            result.converged = True
            break
        if hspace.nfuncs > settings.max_dofs:
            break

        # 4. MARK
        marked_cells = mark_fn(eta_loc)
        if not marked_cells:
            break  # nothing to refine

        # Convert marked cells → marked function indices, then REFINE
        for lev, cell_idx in marked_cells.items():
            func_idx = cells_to_functions(hspace, lev, cell_idx)
            if len(func_idx) > 0:
                hspace.refine(lev, func_idx)

    return result
