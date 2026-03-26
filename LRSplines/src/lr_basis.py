"""
Individual LR B-spline basis function.

Each LR basis function is a tensor-product B-spline defined by two
*local* knot vectors — one per parametric direction.  It is evaluated
via the standard Cox–de Boor recursion applied independently in each
direction.

Mathematical background
-----------------------
Given degree p and local knot vector  xi = [xi_0, xi_1, ..., xi_{p+1}]
(length p+2), the univariate B-spline B_{xi,p}(u) is recursively defined:

  B_{xi,0}(u) = 1  if  xi_0 <= u < xi_{p+1},  else 0
              (last basis function: xi_0 < u <= xi_{p+1})

  B_{xi,p}(u) = (u - xi_0)/(xi_p - xi_0) * B_{xi[:-1],p-1}(u)
              + (xi_{p+1} - u)/(xi_{p+1} - xi_1) * B_{xi[1:],p-1}(u)

where 0/0 = 0 by convention.

The bivariate basis function is:
  B(u, v) = B_{xi,p}(u) * B_{eta,q}(v)

and its gradient is:
  dB/du = B'_{xi,p}(u) * B_{eta,q}(v)
  dB/dv = B_{xi,p}(u)  * B'_{eta,q}(v)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Univariate Cox–de Boor evaluation (scalar)
# ---------------------------------------------------------------------------

def _eval1d(u: float, degree: int, knots: np.ndarray,
            end_point: bool = False) -> float:
    """
    Evaluate a single univariate B-spline at scalar ``u``.

    Parameters
    ----------
    u        : evaluation point
    degree   : polynomial degree p
    knots    : local knot vector of length p+2
    end_point: if True, treat the rightmost knot as a *closed* endpoint
               (used for the last basis function in a sequence so that the
               right boundary of the domain is in the support).

    Returns
    -------
    float  value of B_{knots, degree}(u)

    Support convention
    ------------------
    end_point=False : half-open interval  [knots[0], knots[-1])  — strict.
    end_point=True  : closed interval     [knots[0], knots[-1]].

    Using strict half-open for the non-endpoint case is essential for
    partition of unity: at a knot breakpoint t, only the function whose
    support starts at t (not the one whose support ends at t) is nonzero.
    """
    left_bd  = knots[0]
    right_bd = knots[-1]
    tol = 1e-14

    if end_point:
        in_support = (left_bd - tol <= u <= right_bd + tol)
    else:
        # Strict half-open: u must be in [left_bd, right_bd)
        in_support = (left_bd - tol <= u < right_bd - tol)

    if not in_support:
        return 0.0

    if degree == 0:
        return 1.0

    # Recursive Cox–de Boor — propagate end_point to both sub-calls so
    # that the closed-endpoint handling reaches the degree-0 base case.
    denom_l = knots[-2] - knots[0]
    left = ((u - knots[0]) / denom_l * _eval1d(u, degree - 1, knots[:-1], end_point)
            if denom_l > tol else 0.0)

    denom_r = knots[-1] - knots[1]
    right = ((knots[-1] - u) / denom_r * _eval1d(u, degree - 1, knots[1:], end_point)
             if denom_r > tol else 0.0)

    return left + right


def _deriv1d(u: float, degree: int, knots: np.ndarray,
             end_point: bool = False) -> float:
    """
    Evaluate the first derivative of a univariate B-spline at scalar ``u``.

    Uses the standard recursion:
      B'_{p}(u) = p * [ B_{p-1,left}(u) / (xi_p - xi_0)
                       - B_{p-1,right}(u) / (xi_{p+1} - xi_1) ]
    """
    if degree == 0:
        return 0.0

    tol = 1e-14
    result = 0.0

    denom_l = knots[-2] - knots[0]
    if denom_l > tol:
        result += degree / denom_l * _eval1d(u, degree - 1, knots[:-1], end_point)

    denom_r = knots[-1] - knots[1]
    if denom_r > tol:
        result -= degree / denom_r * _eval1d(u, degree - 1, knots[1:], end_point)

    return result


def eval1d_vec(u: np.ndarray, degree: int, knots: np.ndarray,
               end_point: bool = False) -> np.ndarray:
    """Vectorised evaluation of a univariate B-spline over array ``u``."""
    return np.array([_eval1d(float(ui), degree, knots, end_point) for ui in u])


def deriv1d_vec(u: np.ndarray, degree: int, knots: np.ndarray,
                end_point: bool = False) -> np.ndarray:
    """Vectorised first-derivative evaluation over array ``u``."""
    return np.array([_deriv1d(float(ui), degree, knots, end_point) for ui in u])


# ---------------------------------------------------------------------------
# Bivariate LR basis function
# ---------------------------------------------------------------------------

class LRBasisFunction:
    """
    A single LR B-spline basis function B(u, v) = B_u(u) * B_v(v).

    Parameters
    ----------
    id : int
        Unique identifier within the LRSplineSpace.
    knots_u : array-like, shape (p+2,)
        Local knot vector in the u-direction.
    knots_v : array-like, shape (q+2,)
        Local knot vector in the v-direction.
    degree_u : int
        Polynomial degree in u.
    degree_v : int
        Polynomial degree in v.
    end_u : bool
        Whether to treat the right endpoint of knots_u as closed.
    end_v : bool
        Whether to treat the right endpoint of knots_v as closed.
    coefficient : float
        Scalar coefficient (control-point weight) for this basis function.
        Default 1.0.
    """

    def __init__(self,
                 id: int,
                 knots_u: np.ndarray,
                 knots_v: np.ndarray,
                 degree_u: int,
                 degree_v: int,
                 end_u: bool = False,
                 end_v: bool = False,
                 coefficient: float = 1.0):
        self.id = id
        self.knots_u = np.asarray(knots_u, dtype=float)
        self.knots_v = np.asarray(knots_v, dtype=float)
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.end_u = end_u
        self.end_v = end_v
        self.coefficient = coefficient

        assert len(self.knots_u) == degree_u + 2, (
            f"knots_u must have length degree_u+2 = {degree_u+2}, "
            f"got {len(self.knots_u)}")
        assert len(self.knots_v) == degree_v + 2, (
            f"knots_v must have length degree_v+2 = {degree_v+2}, "
            f"got {len(self.knots_v)}")

    # ------------------------------------------------------------------
    # Support
    # ------------------------------------------------------------------

    @property
    def support(self) -> Tuple[float, float, float, float]:
        """
        Bounding box of this function's support:
        (u_min, u_max, v_min, v_max) = (knots_u[0], knots_u[-1],
                                         knots_v[0], knots_v[-1]).
        """
        return (self.knots_u[0], self.knots_u[-1],
                self.knots_v[0], self.knots_v[-1])

    def support_contains(self, u: float, v: float, *, tol: float = 1e-14) -> bool:
        """True if (u, v) lies inside (or on the boundary of) the support."""
        u0, u1, v0, v1 = self.support
        return u0 - tol <= u <= u1 + tol and v0 - tol <= v <= v1 + tol

    def support_overlaps_element(self, u0: float, u1: float,
                                  v0: float, v1: float,
                                  *, tol: float = 1e-14) -> bool:
        """
        True if this function's support overlaps the open interior of the
        element [u0, u1] x [v0, v1].
        """
        su0, su1, sv0, sv1 = self.support
        return (su0 < u1 - tol and su1 > u0 + tol and
                sv0 < v1 - tol and sv1 > v0 + tol)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, u: float, v: float) -> float:
        """
        Evaluate coefficient * B(u, v) at a single point.

        The returned value includes ``self.coefficient`` so that the
        sum of all basis functions in the space equals 1 (partition of unity)
        even after knot splitting (where children carry fractional coefficients).
        """
        bu = _eval1d(float(u), self.degree_u, self.knots_u, self.end_u)
        bv = _eval1d(float(v), self.degree_v, self.knots_v, self.end_v)
        return self.coefficient * bu * bv

    def eval_array(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate coefficient * B at an array of points.

        Parameters
        ----------
        pts : ndarray, shape (N, 2)

        Returns
        -------
        ndarray, shape (N,)
            Each entry is ``self.coefficient * B_u(u_i) * B_v(v_i)``.
        """
        pts = np.asarray(pts, dtype=float)
        bu = eval1d_vec(pts[:, 0], self.degree_u, self.knots_u, self.end_u)
        bv = eval1d_vec(pts[:, 1], self.degree_v, self.knots_v, self.end_v)
        return self.coefficient * bu * bv

    def grad_array(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of (coefficient * B) at an array of points.

        Parameters
        ----------
        pts : ndarray, shape (N, 2)

        Returns
        -------
        ndarray, shape (N, 2)
            Column 0: d(coeff*B)/du, Column 1: d(coeff*B)/dv.
        """
        pts = np.asarray(pts, dtype=float)
        bu  = eval1d_vec( pts[:, 0], self.degree_u, self.knots_u, self.end_u)
        bv  = eval1d_vec( pts[:, 1], self.degree_v, self.knots_v, self.end_v)
        dbu = deriv1d_vec(pts[:, 0], self.degree_u, self.knots_u, self.end_u)
        dbv = deriv1d_vec(pts[:, 1], self.degree_v, self.knots_v, self.end_v)
        return self.coefficient * np.column_stack([dbu * bv, bu * dbv])

    # ------------------------------------------------------------------
    # Knot insertion (used by the refinement algorithm)
    # ------------------------------------------------------------------

    def insert_knot_u(self, t: float) -> Tuple['LRBasisFunction', 'LRBasisFunction']:
        """
        Split this basis function by inserting knot ``t`` into knots_u.

        Uses the one-step Boehm knot insertion formula:
          New functions B1 and B2 satisfy:
            B(u,v) = alpha_1(u)*B1(u,v) + alpha_2(u)*B2(u,v)

        Parameters
        ----------
        t : float
            New knot value; must satisfy knots_u[0] < t < knots_u[-1].

        Returns
        -------
        (B1, B2) : two new LRBasisFunction objects
            B1 has knots_u = [knots_u[0], ..., t]  (left child)
            B2 has knots_u = [t, ..., knots_u[-1]] (right child)

        The *coefficients* of B1 and B2 are set so that
          coeff * B = coeff_1 * B1 + coeff_2 * B2
        where coeff_{1,2} = alpha_{1,2} * coeff (Boehm alpha factors).
        """
        return _split_basis(self, axis='u', t=t)

    def insert_knot_v(self, t: float) -> Tuple['LRBasisFunction', 'LRBasisFunction']:
        """
        Split this basis function by inserting knot ``t`` into knots_v.
        See ``insert_knot_u`` for details.
        """
        return _split_basis(self, axis='v', t=t)

    # ------------------------------------------------------------------
    # Greville abscissa (collocation point)
    # ------------------------------------------------------------------

    @property
    def greville_u(self) -> float:
        """Mean of the interior knots in the u-direction."""
        return float(np.mean(self.knots_u[1:-1]))

    @property
    def greville_v(self) -> float:
        """Mean of the interior knots in the v-direction."""
        return float(np.mean(self.knots_v[1:-1]))

    @property
    def greville(self) -> Tuple[float, float]:
        return (self.greville_u, self.greville_v)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        u0, u1, v0, v1 = self.support
        return (f"LRBasisFunction(id={self.id}, "
                f"supp=[{u0:.3g},{u1:.3g}]x[{v0:.3g},{v1:.3g}], "
                f"p=({self.degree_u},{self.degree_v}))")


# ---------------------------------------------------------------------------
# Knot splitting helper
# ---------------------------------------------------------------------------

def _split_basis(B: LRBasisFunction, axis: str, t: float
                 ) -> Tuple[LRBasisFunction, LRBasisFunction]:
    """
    Split basis function B by inserting knot t in the given axis ('u' or 'v').

    One-step knot insertion (Boehm, 1980):
    Given knot vector xi = [xi_0, ..., xi_{p+1}] and a new knot t in
    (xi_0, xi_{p+1}), we insert t at position k such that xi_{k-1} <= t < xi_k.

    The two child knot vectors are:
      xi_left  = [xi_0, ..., xi_{k-1}, t]
      xi_right = [t, xi_k, ..., xi_{p+1}]

    The Boehm alpha coefficients at the Greville abscissae are:
      alpha_right = (g_right - xi_0) / (xi_p - xi_0)   (for B2)
      alpha_left  = (xi_{p+1} - g_left) / (xi_{p+1} - xi_1) (for B1)
    but since we just need the child functions themselves (not their
    weighted combination), we set both child coefficients to B.coefficient.

    The exact decomposition is:
      B(u,v) = alpha_1 * B1(u,v) + alpha_2 * B2(u,v)
    where alpha_1 + alpha_2 = 1 at any point.  We propagate the coefficient
    so that when assembling the FEM system, the sum is correct.
    """
    if axis == 'u':
        knots = B.knots_u.copy()
        degree = B.degree_u
    else:
        knots = B.knots_v.copy()
        degree = B.degree_v

    tol = 1e-14
    assert knots[0] + tol < t < knots[-1] - tol, (
        f"Inserted knot t={t} must be strictly inside support "
        f"[{knots[0]}, {knots[-1]}]")

    # Insert t into the knot vector
    knots_new = np.sort(np.append(knots, t))   # length p+3

    # Left child uses knots_new[0 : p+2]  (drops the last knot)
    # Right child uses knots_new[1 : p+3] (drops the first knot)
    knots_left  = knots_new[:degree + 2]
    knots_right = knots_new[1:degree + 3]

    # Boehm one-step knot insertion coefficients (Boehm 1980, global formula).
    #
    # For old global knot vector T and new function index i, the identity is:
    #   N_i = alpha[i] * N'_left + (1 - alpha[i+1]) * N'_right
    #
    # where alpha[j] = clip((t - T[j]) / (T[j+p] - T[j]), 0, 1)
    # and T[j] / T[j+p] correspond to xi[0] / xi[-2] and xi[1] / xi[-1]
    # of the local knot vector xi.
    #
    # c_left  = alpha[i]   = clip((t - xi[0])/(xi[-2] - xi[0]), 0, 1)
    # c_right = 1-alpha[i+1] = 1 - clip((t - xi[1])/(xi[-1] - xi[1]), 0, 1)
    #
    # 0/0 convention (from global formula):
    #   alpha[j] = 1 if T[j] <= t else 0   when denominator = 0

    # c_left = alpha[i]
    denom_left = knots[-2] - knots[0]
    if denom_left > tol:
        c_left = float(np.clip((t - knots[0]) / denom_left, 0.0, 1.0))
    else:
        c_left = 1.0 if knots[0] <= t + tol else 0.0

    # c_right = 1 - alpha[i+1]
    denom_right = knots[-1] - knots[1]
    if denom_right > tol:
        alpha_next = float(np.clip((t - knots[1]) / denom_right, 0.0, 1.0))
    else:
        alpha_next = 1.0 if knots[1] <= t + tol else 0.0
    c_right = 1.0 - alpha_next

    coeff = B.coefficient

    if axis == 'u':
        B_left = LRBasisFunction(
            id=-1,
            knots_u=knots_left, knots_v=B.knots_v.copy(),
            degree_u=degree, degree_v=B.degree_v,
            end_u=B.end_u, end_v=B.end_v,
            coefficient=coeff * c_left)
        B_right = LRBasisFunction(
            id=-1,
            knots_u=knots_right, knots_v=B.knots_v.copy(),
            degree_u=degree, degree_v=B.degree_v,
            end_u=B.end_u, end_v=B.end_v,
            coefficient=coeff * c_right)
    else:
        B_left = LRBasisFunction(
            id=-1,
            knots_u=B.knots_u.copy(), knots_v=knots_left,
            degree_u=B.degree_u, degree_v=degree,
            end_u=B.end_u, end_v=B.end_v,
            coefficient=coeff * c_left)
        B_right = LRBasisFunction(
            id=-1,
            knots_u=B.knots_u.copy(), knots_v=knots_right,
            degree_u=B.degree_u, degree_v=degree,
            end_u=B.end_u, end_v=B.end_v,
            coefficient=coeff * c_right)

    return B_left, B_right
