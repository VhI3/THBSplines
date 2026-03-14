"""
Pure NumPy/SciPy replacement for the Cython BSpline extension module.

Provides the same public API that the rest of the package uses:
  - BSpline         : univariate B-spline (vectorised evaluation + derivative)
  - TensorProductBSpline : tensor-product B-spline in arbitrary dimension
  - integrate       : Gauss-quadrature inner product ∫ Bᵢ Bⱼ
  - integrate_grad  : Gauss-quadrature inner product ∫ ∇Bᵢ · ∇Bⱼ

Mathematical background
-----------------------
A B-spline Bᵢ of degree p is defined by a *local* knot vector of length p+2:
    t = [t₀, t₁, …, t_{p+1}]
Its support is [t₀, t_{p+1}].  The function is evaluated using the
Cox–de Boor recursion:

    B(x; 0, [t_j, t_{j+1}]) = 1  if  t_j ≤ x < t_{j+1}  else  0

    B(x; p, t) = (x − t₀)/(t_{p} − t₀) · B(x; p−1, t[:-1])
               + (t_{p+1} − x)/(t_{p+1} − t₁) · B(x; p−1, t[1:])

with the convention 0/0 = 0.

The ``evaluate_end`` flag controls whether the rightmost point is included,
which is needed for the last basis function in a clamped spline space
(B-splines normally use half-open intervals [t_j, t_{j+1})).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal scalar helpers (recursive, match Cython logic exactly)
# ---------------------------------------------------------------------------

def _eval_scalar(x: float, degree: int, knots: np.ndarray, end: int) -> float:
    """
    Evaluate a single B-spline at a single point using Cox–de Boor recursion.

    Parameters
    ----------
    x      : evaluation point
    degree : polynomial degree p
    knots  : local knot vector of length p+2
    end    : 1 → include right endpoint (for last basis function), 0 → exclude
    """
    n = len(knots)

    # --- Find knot span: check whether x is in the support ---
    if end == 0:
        # Standard half-open interval [t₀, t_{p+1})
        in_support = (knots[0] <= x < knots[-1])
    else:
        # Last basis function: closed interval (t₀, t_{p+1}]
        in_support = (knots[0] < x <= knots[-1]) or abs(x - knots[-1]) < 1e-14

    if not in_support:
        return 0.0

    # Base case: degree 0 is the indicator function of [t₀, t₁)
    if degree == 0:
        return 1.0

    # --- Recursive step ---
    left  = _eval_scalar(x, degree - 1, knots[:-1], end)
    right = _eval_scalar(x, degree - 1, knots[1:],  end)

    denom_left  = knots[-2] - knots[0]   # t_{p} − t₀
    denom_right = knots[-1] - knots[1]   # t_{p+1} − t₁

    if denom_left > 1e-14:
        left  *= (x - knots[0]) / denom_left
    else:
        left = 0.0  # 0/0 convention

    if denom_right > 1e-14:
        right *= (knots[-1] - x) / denom_right
    else:
        right = 0.0  # 0/0 convention

    return left + right


def _deriv_scalar(x: float, degree: int, knots: np.ndarray, end: int, r: int) -> float:
    """
    Evaluate the r-th derivative of a B-spline at a single point.

    The derivative formula is:
        D^r B(x; p, t) = p · [ D^{r-1} B(x; p-1, t[:-1]) / (t_p − t₀)
                               − D^{r-1} B(x; p-1, t[1:]) / (t_{p+1} − t₁) ]

    For r=1 the inner terms are evaluations (D^0), for r>1 they recurse.
    """
    n = len(knots)
    denom_left  = knots[-2] - knots[0]
    denom_right = knots[-1] - knots[1]

    if r == 1:
        # Base of the derivative recursion: innermost terms are evaluations
        left  = _eval_scalar(x, degree - 1, knots[:-1], end)
        right = _eval_scalar(x, degree - 1, knots[1:],  end)
    else:
        left  = _deriv_scalar(x, degree - 1, knots[:-1], 0,   r - 1)
        right = _deriv_scalar(x, degree - 1, knots[1:],  end, r - 1)

    if denom_left > 1e-14:
        left /= denom_left
    else:
        left = 0.0

    if denom_right > 1e-14:
        right /= denom_right
    else:
        right = 0.0

    return degree * (left - right)


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class BSpline:
    """
    Univariate B-spline defined by a *local* knot vector of length ``degree + 2``.

    This matches the API of the former Cython ``BSpline`` class so that no
    call-site changes are required elsewhere in the package.

    Parameters
    ----------
    degree       : polynomial degree
    knots        : 1-D array of length ``degree + 2``
    evaluate_end : if 1, the right endpoint t_{p+1} is included in the support.
                   Used for the last basis function in a clamped spline space.
    """

    def __init__(self, degree: int, knots: np.ndarray, evaluate_end: int = 0):
        self.degree       = degree
        self.knots        = np.asarray(knots, dtype=np.float64)
        self.evaluate_end = int(evaluate_end)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the B-spline at one or more points.

        Parameters
        ----------
        x : array-like, shape (n,)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        result = np.empty(len(x), dtype=np.float64)
        for i, xi in enumerate(x):
            result[i] = _eval_scalar(xi, self.degree, self.knots, self.evaluate_end)
        return result

    # ------------------------------------------------------------------
    # Derivative
    # ------------------------------------------------------------------

    def D(self, x: np.ndarray, r: int) -> np.ndarray:
        """
        Evaluate the r-th derivative at one or more points.

        Parameters
        ----------
        x : array-like, shape (n,)
        r : derivative order (1 for first derivative, etc.)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        result = np.empty(len(x), dtype=np.float64)
        for i, xi in enumerate(x):
            result[i] = _deriv_scalar(xi, self.degree, self.knots, self.evaluate_end, r)
        return result

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BSpline):
            return NotImplemented
        if self.degree != other.degree:
            return False
        if len(self.knots) != len(other.knots):
            return False
        return np.allclose(self.knots, other.knots, atol=1e-14)

    def __repr__(self) -> str:
        return f"BSpline(degree={self.degree}, knots={self.knots.tolist()}, end={self.evaluate_end})"


class TensorProductBSpline:
    """
    Tensor-product B-spline in ``d`` parametric dimensions.

    A tensor-product B-spline is the product of ``d`` univariate B-splines,
    one per parametric direction.  Because B-splines have compact support,
    this product is also compactly supported.

    Parameters
    ----------
    degrees        : array-like of ints, shape (d,) — one degree per direction
    knots          : array-like of floats, shape (d, p+2) — local knot vectors
    end_evaluation : array-like of ints, shape (d,) — end-point flags per direction
    """

    def __init__(
        self,
        degrees: np.ndarray,
        knots: np.ndarray,
        end_evaluation: np.ndarray | None = None,
    ):
        self.degrees             = np.asarray(degrees, dtype=int)
        self.parametric_dimension = int(len(self.degrees))
        knots                    = np.asarray(knots, dtype=np.float64)

        if end_evaluation is None:
            end_evaluation = np.zeros(self.parametric_dimension, dtype=int)
        self.end_evaluation = np.asarray(end_evaluation, dtype=int)

        # Build one univariate BSpline per direction
        self.univariate_b_splines: list[BSpline] = [
            BSpline(int(self.degrees[i]), knots[i], int(self.end_evaluation[i]))
            for i in range(self.parametric_dimension)
        ]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the tensor-product B-spline at N points.

        Parameters
        ----------
        x : array-like, shape (N, d)

        Returns
        -------
        np.ndarray, shape (N,)
            B(x₁, …, x_d) = B₁(x₁) · B₂(x₂) · … · B_d(x_d)
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.parametric_dimension)
        n = x.shape[0]

        # Evaluate each univariate factor independently, then multiply
        result = np.ones(n, dtype=np.float64)
        for j in range(self.parametric_dimension):
            result *= self.univariate_b_splines[j](x[:, j])
        return result

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient ∇B at N points.

        The j-th component of the gradient is:

            ∂B/∂x_j = B₁(x₁) · … · B'_j(x_j) · … · B_d(x_d)

        i.e. differentiate in direction j and keep the other factors evaluated.

        Parameters
        ----------
        x : array-like, shape (N, d)

        Returns
        -------
        np.ndarray, shape (N, d)
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.parametric_dimension)
        n = x.shape[0]
        d = self.parametric_dimension

        # Pre-compute function values and first derivatives for each direction
        vals  = np.zeros((n, d), dtype=np.float64)  # Bⱼ(xⱼ)
        grads = np.zeros((n, d), dtype=np.float64)  # B'ⱼ(xⱼ)
        for j in range(d):
            vals[:, j]  = self.univariate_b_splines[j](x[:, j])
            grads[:, j] = self.univariate_b_splines[j].D(x[:, j], 1)

        # Gradient component j = derivative in j × product of values in all other dirs
        gradient = grads.copy()
        for j in range(d):
            for k in range(d):
                if k != j:
                    gradient[:, j] *= vals[:, k]

        return gradient

    def __repr__(self) -> str:
        return (
            f"TensorProductBSpline(degrees={self.degrees.tolist()}, "
            f"dim={self.parametric_dimension})"
        )


# ---------------------------------------------------------------------------
# Quadrature helpers
# ---------------------------------------------------------------------------

def integrate(
    bi_values: np.ndarray,
    bj_values: np.ndarray,
    weights: np.ndarray,
    area: float,
    dim: int,
) -> float:
    """
    Compute the Gauss-quadrature approximation of ∫ Bᵢ(x) Bⱼ(x) dx over a cell.

    The standard Gauss rule on [-1,1] is mapped to the physical cell via the
    Jacobian factor ``area / 2**dim``:

        ∫_cell Bᵢ Bⱼ ≈ (area / 2^d) · Σₖ wₖ Bᵢ(xₖ) Bⱼ(xₖ)

    Parameters
    ----------
    bi_values : B-spline Bᵢ evaluated at all quadrature points, shape (Q,)
    bj_values : B-spline Bⱼ evaluated at all quadrature points, shape (Q,)
    weights   : quadrature weights, shape (Q,)
    area      : cell volume (product of edge lengths)
    dim       : parametric dimension (determines the Jacobian exponent)

    Returns
    -------
    float
    """
    return float(np.dot(weights, bi_values * bj_values) * area / 2**dim)


def integrate_grad(
    bi_grad: np.ndarray,
    bj_grad: np.ndarray,
    weights: np.ndarray,
    area: float,
    dim: int,
) -> float:
    """
    Compute the Gauss-quadrature approximation of ∫ ∇Bᵢ(x) · ∇Bⱼ(x) dx over a cell.

        ∫_cell ∇Bᵢ · ∇Bⱼ ≈ (area / 2^d) · Σₖ wₖ (∇Bᵢ(xₖ) · ∇Bⱼ(xₖ))

    Parameters
    ----------
    bi_grad : gradient of Bᵢ at all quadrature points, shape (Q, d)
    bj_grad : gradient of Bⱼ at all quadrature points, shape (Q, d)
    weights : quadrature weights, shape (Q,)
    area    : cell volume
    dim     : parametric dimension

    Returns
    -------
    float
    """
    # Row-wise dot product, then weighted sum
    dot_products = np.sum(bi_grad * bj_grad, axis=1)  # shape (Q,)
    return float(np.dot(weights, dot_products) * area / 2**dim)
