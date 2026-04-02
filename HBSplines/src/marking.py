"""
Marking strategies for adaptive HB-spline refinement.

Given local estimators  eta[l][j]  (level l, cell j), these strategies
select which cells to refine.

Dorfler (bulk) criterion
------------------------
Mark the smallest set S such that  sum_{S} eta_j^2 >= theta * eta_global^2.
Greedy: sort cells by decreasing eta_j, accumulate until threshold met.

Maximum criterion
-----------------
Mark all cells where  eta_j >= theta * max_j(eta_j).

Both strategies return a dict  {level: array_of_cell_indices}  so the
caller can convert them to basis function indices via
``estimator.cells_to_functions``.
"""

from __future__ import annotations

import numpy as np


def dorfler_mark(
    eta: list[np.ndarray],
    theta: float = 0.5,
) -> dict[int, np.ndarray]:
    """
    Dorfler (bulk) marking.

    Parameters
    ----------
    eta   : list of per-level estimator arrays (from local_residual)
    theta : marking parameter in (0, 1]

    Returns
    -------
    marked : dict  {level: int_array_of_cell_indices}
             Only levels with at least one marked cell are included.
    """
    # Flatten all (level, cell_idx, eta_value) into a single list for sorting
    all_entries: list[tuple[float, int, int]] = []
    for lev, e in enumerate(eta):
        for j, val in enumerate(e):
            all_entries.append((val, lev, j))

    # Sort descending by estimator value
    all_entries.sort(key=lambda x: x[0], reverse=True)

    eta_global_sq = sum(float(np.sum(e ** 2)) for e in eta)
    target = theta * eta_global_sq

    accumulated = 0.0
    marked: dict[int, list[int]] = {}
    for val, lev, j in all_entries:
        accumulated += val ** 2
        marked.setdefault(lev, []).append(j)
        if accumulated >= target:
            break

    return {lev: np.array(idx, dtype=int) for lev, idx in marked.items()}


def max_mark(
    eta: list[np.ndarray],
    theta: float = 0.5,
) -> dict[int, np.ndarray]:
    """
    Maximum marking: mark all cells with  eta_j >= theta * max(eta).

    Parameters
    ----------
    eta   : list of per-level estimator arrays
    theta : fraction of the maximum estimator value

    Returns
    -------
    marked : dict  {level: int_array_of_cell_indices}
    """
    if not any(len(e) for e in eta):
        return {}

    eta_max = max(float(e.max()) for e in eta if len(e) > 0)
    threshold = theta * eta_max

    marked: dict[int, np.ndarray] = {}
    for lev, e in enumerate(eta):
        idx = np.where(e >= threshold - 1e-14)[0]
        if len(idx) > 0:
            marked[lev] = idx

    return marked
