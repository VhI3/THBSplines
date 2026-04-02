"""
Single-level 1D B-spline space.

Reuses:
  - Cox–de Boor evaluation  : LRSplines.src.lr_basis._eval1d, _deriv1d
  - r-th derivative         : THBSplines.src.b_spline_numpy._deriv_scalar
  - Dyadic refinement       : THBSplines.src.tensor_product_space.insert_midpoints
"""

from __future__ import annotations

import numpy as np

from LRSplines.src.lr_basis import _eval1d, _deriv1d
from THBSplines.src.b_spline_numpy import _deriv_scalar
from THBSplines.src.tensor_product_space import insert_midpoints


class BsplineSpace:
    """
    Single-level 1D B-spline space with a clamped (open) knot vector.

    Parameters
    ----------
    knots : array-like
        Full clamped knot vector, e.g. [0,0,0, 0.5, 1,1,1] for degree 2.
    degree : int
        Polynomial degree p.
    """

    def __init__(self, knots: np.ndarray, degree: int) -> None:
        self._knots = np.asarray(knots, dtype=float)
        self._degree = degree
        self._dim = len(self._knots) - degree - 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def knots(self) -> np.ndarray:
        return self._knots

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def dim(self) -> int:
        """Number of basis functions."""
        return self._dim

    @property
    def domain(self) -> tuple[float, float]:
        return float(self._knots[0]), float(self._knots[-1])

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _local_knots(self, i: int) -> np.ndarray:
        """Local knot vector for function i — length p+2."""
        return self._knots[i : i + self._degree + 2]

    def _end_point(self, i: int) -> bool:
        """True for the last basis function (support includes right boundary)."""
        return i == self._dim - 1

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def eval(self, x: np.ndarray, i: int) -> np.ndarray:
        """Evaluate the i-th basis function at an array of points."""
        x = np.asarray(x, dtype=float)
        lk = self._local_knots(i)
        ep = self._end_point(i)
        return np.array([_eval1d(xi, self._degree, lk, ep) for xi in x])

    def deriv(self, x: np.ndarray, i: int, r: int = 1) -> np.ndarray:
        """Evaluate the r-th derivative of the i-th basis function.

        Uses ``_deriv1d`` (from LRSplines) for r=1 and ``_deriv_scalar``
        (from THBSplines) for r>=2.
        """
        x = np.asarray(x, dtype=float)
        lk = self._local_knots(i)
        ep = self._end_point(i)
        if r == 1:
            return np.array([_deriv1d(xi, self._degree, lk, ep) for xi in x])
        # r >= 2: use the recursive formula from THBSplines
        end_int = int(ep)
        return np.array([_deriv_scalar(xi, self._degree, lk, end_int, r) for xi in x])

    def eval_all(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all basis functions at all points.

        Returns
        -------
        B : ndarray, shape (len(x), dim)
        """
        x = np.asarray(x, dtype=float)
        return np.column_stack([self.eval(x, i) for i in range(self._dim)])

    def deriv_all(self, x: np.ndarray, r: int = 1) -> np.ndarray:
        """Evaluate r-th derivative of all basis functions at all points.

        Returns
        -------
        dB : ndarray, shape (len(x), dim)
        """
        x = np.asarray(x, dtype=float)
        return np.column_stack([self.deriv(x, i, r) for i in range(self._dim)])

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def refine(self) -> BsplineSpace:
        """Return a dyadically refined copy (midpoints of all knot intervals)."""
        new_knots = insert_midpoints(self._knots, self._degree)
        return BsplineSpace(new_knots, self._degree)

    def get_children(self, i: int) -> np.ndarray:
        """
        Indices of children of function i at the next (dyadically refined) level.

        Children are functions at the finer level whose support is fully
        contained within the support of function i.  This is equivalent to
        the formula  {2i + k : k = 0,...,p+1}  for interior functions
        (Höllig 2003), but works correctly for boundary functions where the
        repeated clamping knots shift the effective indices.
        """
        finer = self.refine()
        lk_i = self._local_knots(i)
        a, b = float(lk_i[0]), float(lk_i[-1])
        children = []
        for j in range(finer.dim):
            lk_j = finer._local_knots(j)
            aj, bj = float(lk_j[0]), float(lk_j[-1])
            if aj >= a - 1e-14 and bj <= b + 1e-14:
                children.append(j)
        return np.array(children, dtype=int)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def greville(self) -> np.ndarray:
        """Greville abscissae: g_i = mean(knots[i+1 : i+p+1])."""
        p = self._degree
        return np.array(
            [np.mean(self._knots[i + 1 : i + p + 1]) for i in range(self._dim)]
        )

    def __repr__(self) -> str:
        a, b = self.domain
        return (
            f"BsplineSpace(degree={self._degree}, dim={self._dim}, "
            f"domain=[{a}, {b}], nknots={len(self._knots)})"
        )
