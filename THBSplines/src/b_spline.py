"""
Pure-Python / NumPy B-spline utilities.

This module contains:
  - ``UnivariateBSpline`` : scalar-callable B-spline with LRU caching
  - ``BSpline``           : tensor-product wrapper over ``UnivariateBSpline``
  - ``find_knot_index``   : locate the knot span containing a point
  - ``augment_knots``     : pad a knot vector for the knot-insertion algorithm

These are used internally by ``TensorProductSpace.compute_projection_matrix``
(the Boehm knot-insertion algorithm) and are *not* the performance-critical
evaluation path used during assembly.  The performance path lives in
``b_spline_numpy.py``.

Knot-index convention
---------------------
Given a knot vector  T = [t₀, t₁, …, t_n]  (with possible repeated values),
``find_knot_index(x, T)`` returns the largest index ``i`` such that
``T[i] ≤ x < T[i+1]``.  At the right endpoint, passing ``endpoint=True``
returns the index of the last non-degenerate interval.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------
# Stand-alone helper functions
# ---------------------------------------------------------------------------

def find_knot_index(x: float, knots: np.ndarray, endpoint: bool = False) -> int:
    """
    Return the largest index ``i`` such that ``knots[i] ≤ x < knots[i+1]``.

    Parameters
    ----------
    x        : evaluation point
    knots    : sorted knot vector (may have repeated values)
    endpoint : if True, the right endpoint ``knots[-1]`` is mapped to the last
               valid interior interval rather than returning -1.

    Returns
    -------
    int
        Knot-span index, or -1 when ``x`` is outside the support.
    """
    knots = np.asarray(knots, dtype=np.float64)

    if endpoint and knots[-2] <= x <= knots[-1]:
        # Walk back from the right to find the first strictly-smaller knot
        i = max(int(np.argmax(knots < x)) - 1, 0)
        return len(knots) - i - 2

    if x < knots[0] or x >= knots[-1]:
        return -1  # outside support

    return int(max(np.argmax(knots > x) - 1, 0))


def augment_knots(knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Pad a knot vector with ``degree + 1`` sentinel values on each side.

    The augmented vector is used in the Boehm knot-insertion algorithm
    (see ``TensorProductSpace.compute_projection_matrix``).  The sentinels
    are placed strictly outside the original parameter domain so they are
    never accidentally selected as active knot spans.

    Parameters
    ----------
    knots  : original knot vector
    degree : B-spline degree

    Returns
    -------
    np.ndarray
        Padded knot vector of length ``len(knots) + 2*(degree+1)``.
    """
    knots = np.asarray(knots, dtype=np.float64)
    return np.pad(
        knots,
        pad_width=(degree + 1, degree + 1),
        mode="constant",
        constant_values=(knots[0] - 1.0, knots[-1] + 1.0),
    )


# ---------------------------------------------------------------------------
# UnivariateBSpline
# ---------------------------------------------------------------------------

class UnivariateBSpline:
    """
    A single univariate B-spline defined by its degree and *global* knot vector.

    The function is evaluated using the de Boor triangular algorithm in
    matrix form.  Scalar calls are cached with ``lru_cache`` to avoid
    redundant computation during projection-matrix construction.

    Parameters
    ----------
    degree   : polynomial degree p
    knots    : global knot vector of length ≥ p + 2
    endpoint : if True, include the right endpoint in the support
               (needed for the last B-spline in a clamped space)
    """

    def __init__(self, degree: int, knots: np.ndarray, endpoint: bool = True):
        self.degree           = degree
        self.knots            = np.asarray(knots, dtype=np.float64)
        self._endpoint        = bool(endpoint)
        self._augmented_knots = augment_knots(self.knots, self.degree)

    # ------------------------------------------------------------------
    # Evaluation  (cached – knots and degree are fixed per instance)
    # ------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def __call__(self, x: float) -> float:
        """
        Evaluate the B-spline at scalar point ``x``.

        Uses the de Boor algorithm in triangular-table form, which avoids
        redundant multiplications compared to the naive recursion.

        Parameters
        ----------
        x : evaluation point (must be a Python float for caching to work)

        Returns
        -------
        float
        """
        i = self.knot_index(x)
        if i == -1:
            return 0.0

        t = self._augmented_knots
        # Shift index to account for the augmented prefix of length degree+1
        i = i + self.degree + 1

        # Initialise the coefficient vector: all zeros except a 1 at position i
        c = np.zeros(len(t) - self.degree - 1)
        c[self.degree + 1] = 1.0
        c = c[i - self.degree: i + 1]  # extract the relevant window

        # Triangular de Boor recursion
        for k in range(self.degree, 0, -1):
            t1    = t[i - k + 1: i + 1]
            t2    = t[i + 1:     i + k + 1]
            denom = t2 - t1
            omega = np.divide(
                x - t1, denom,
                out=np.zeros_like(t1),
                where=denom != 0,
            )
            c = (1.0 - omega) * c[:-1] + omega * c[1:]

        return float(c)

    # ------------------------------------------------------------------
    # Knot-span lookup  (cached for the same reason as __call__)
    # ------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def knot_index(self, x: float) -> int:
        """Return the knot-span index for ``x`` (delegates to module-level helper)."""
        return find_knot_index(x, self.knots, self.endpoint)

    # ------------------------------------------------------------------
    # Endpoint property — clears caches when toggled
    # ------------------------------------------------------------------

    @property
    def endpoint(self) -> bool:
        return self._endpoint

    @endpoint.setter
    def endpoint(self, value: bool) -> None:
        self._endpoint = bool(value)
        # Invalidate caches: the knot-span logic changes with the flag
        self.__call__.cache_clear()
        self.knot_index.cache_clear()

    def augment_knots(self) -> np.ndarray:
        return augment_knots(self.knots, self.degree)


# ---------------------------------------------------------------------------
# BSpline  (tensor-product wrapper, used in construct_function)
# ---------------------------------------------------------------------------

class BSpline:
    """
    Tensor-product B-spline built from a list of univariate factors.

    This is a lightweight wrapper used by ``TensorProductSpace.construct_function``
    to assemble a callable that multiplies together univariate B-splines in
    each parametric direction.

    Parameters
    ----------
    degrees : array-like of ints, shape (d,)
    knots   : list of 1-D knot vectors, one per direction
    """

    def __init__(self, degrees: np.ndarray, knots: list[np.ndarray]):
        self.degrees = np.asarray(degrees, dtype=int)
        self.knots   = [np.asarray(k, dtype=np.float64) for k in knots]
        self.basis_functions = [
            UnivariateBSpline(d, t) for d, t in zip(self.degrees, self.knots)
        ]
        self.dimension = len(self.degrees)
        # Support as [[min₁, max₁], [min₂, max₂], …]
        self.support = np.array(
            [[k[0], k[-1]] for k in self.knots], dtype=np.float64
        )
        # Slot for identifying this B-spline within a tensor-product mesh
        self.tensor_product_indices: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the tensor-product B-spline at point ``x``.

        Parameters
        ----------
        x : array-like of length d

        Returns
        -------
        float
            B(x₁, …, x_d) = B₁(x₁) · … · B_d(x_d)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        values = np.array([self.basis_functions[i](x[i]) for i in range(self.dimension)])
        return float(np.prod(values))

    def __repr__(self) -> str:
        return f"BSpline(degrees={self.degrees.tolist()}, knots={[k.tolist() for k in self.knots]})"
