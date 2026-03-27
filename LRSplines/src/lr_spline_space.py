"""
LR B-spline space.

Architecture
------------
The space stores two *global* knot vectors (one per parametric direction)
and builds its active basis as the full tensor product of the univariate
B-splines defined by those vectors.  This mirrors the standard
tensor-product B-spline space and guarantees partition of unity at all
times.

Local refinement is realised by inserting *additional* knot values into
one of the global knot vectors.  A *full* line insertion (spanning the
entire domain in the perpendicular direction) is mathematically equivalent
to a standard one-step Boehm knot insertion and always preserves partition
of unity and linear independence.

A *partial* line insertion (T-junction) changes which basis functions exist
on different strips of the domain.  This is the true LR B-spline extension.
Partial lines are supported through the overloading algorithm: only basis
functions whose support is fully spanned by the line in the tangent
direction are split.  Their two children inherit a knot value that
differs from all other functions in the same strip — producing genuine
T-junctions.  Partition of unity is maintained because the correct
Boehm split coefficients are computed from the *local* knot vectors using
the global refinement matrix formula.

Public API
----------
LRSplineSpace(knots_u, knots_v, degree_u, degree_v)
    Construct the initial tensor-product LR space.

.refine_full_line(axis, value)
    Insert a knot line spanning the entire domain.  Always preserves PoU.

.refine_line(line)
    Insert a (possibly partial) MeshLine.  Overloaded functions are split
    with the exact Boehm coefficients derived from the global refinement
    formula, guaranteeing PoU for full lines and well-defined splits for
    partial lines.

.evaluate(pts)         -> ndarray (N, nfuncs)
.evaluate_grad(pts)    -> ndarray (N, nfuncs, 2)
.nfuncs                -> int
.mesh                  -> LRMesh
.basis                 -> list[LRBasisFunction]
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from LRSplines.src.lr_mesh import LRMesh, MeshLine, Element
from LRSplines.src.lr_basis import LRBasisFunction


# ---------------------------------------------------------------------------
# Boehm refinement coefficients (correct global formula)
# ---------------------------------------------------------------------------

def _boehm_coefficients(global_knots: np.ndarray,
                         degree: int,
                         t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Boehm one-step knot insertion matrix column coefficients.

    Given a global knot vector T of length n+p+1 (n basis functions of
    degree p) and a new knot t inserted at position k
    (T[k-1] ≤ t < T[k]), returns arrays ``alpha`` of length n+1 and the
    new local knot vectors for each new basis function.

    The relationship between old (N_i) and new (N'_j) basis functions is:
        N_i = Σ_j  A[j, i] * N'_j
    where A is determined by the alpha values.

    Returns
    -------
    alpha : ndarray, shape (n+1,)
        Boehm alpha coefficients; A[j,i] is derived from alpha[j].
    T_new : ndarray
        The new (longer) global knot vector.
    """
    T = np.asarray(global_knots, dtype=float)
    n = len(T) - degree - 1     # original number of basis functions
    tol = 1e-14

    # Insert t at the correct position
    T_new = np.sort(np.append(T, t))

    alpha = np.zeros(n + 1, dtype=float)
    for j in range(n + 1):
        tj  = T_new[j]           # j-th knot in NEW vector
        # Use the original T for the denominator span
        # alpha_j = (t - T[j]) / (T[j+p] - T[j])
        # but only when T[j] <= t < T[j+p]; else 0 or 1
        if j < len(T) and j + degree < len(T):
            denom = T[j + degree] - T[j]
            if denom > tol:
                a = (t - T[j]) / denom
                alpha[j] = np.clip(a, 0.0, 1.0)
            else:
                # Zero denominator: repeated knot
                alpha[j] = 1.0 if T[j] <= t else 0.0
        elif j >= len(T):
            alpha[j] = 0.0
        else:
            alpha[j] = 0.0

    return alpha, T_new


def _refine_knot_vector(T: np.ndarray, degree: int,
                         t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Insert knot ``t`` into global knot vector T.

    Returns
    -------
    T_new : ndarray   the refined knot vector (length len(T)+1)
    alpha : ndarray   Boehm alpha values, length = nfuncs_new
    """
    T = np.asarray(T, dtype=float)
    n_old = len(T) - degree - 1
    tol = 1e-14

    T_new = np.sort(np.append(T, t))   # length len(T)+1

    # Compute alpha[j] for j = 0..n_old (n_old+1 new functions)
    alpha = np.zeros(n_old + 1, dtype=float)
    for j in range(n_old + 1):
        if j < n_old:
            t_j  = T[j]
            t_jp = T[j + degree]
            denom = t_jp - t_j
            if denom > tol:
                alpha[j] = np.clip((t - t_j) / denom, 0.0, 1.0)
            else:
                alpha[j] = 1.0 if t_j <= t else 0.0
        else:
            # Last new function: alpha_n = 0 (t is to its left)
            alpha[j] = 0.0

    return T_new, alpha


# ---------------------------------------------------------------------------
# LR B-spline space
# ---------------------------------------------------------------------------

class LRSplineSpace:
    """
    A 2-D LR B-spline space based on tensor-product construction with
    support for local refinement via knot line insertion.

    Parameters
    ----------
    knots_u : array-like
        Global knot vector in the u-direction (may contain repeated knots).
    knots_v : array-like
        Global knot vector in the v-direction.
    degree_u : int
        Polynomial degree in u.
    degree_v : int
        Polynomial degree in v.
    """

    def __init__(self,
                 knots_u,
                 knots_v,
                 degree_u: int,
                 degree_v: int):
        self._degree_u = int(degree_u)
        self._degree_v = int(degree_v)

        self._global_knots_u = np.asarray(knots_u, dtype=float)
        self._global_knots_v = np.asarray(knots_v, dtype=float)

        # Build mesh from unique breakpoints
        self._mesh = LRMesh(self._global_knots_u, self._global_knots_v)

        # Flag: True once a partial (T-junction) line has been inserted.
        # A full-line insertion after partial ones would silently destroy the
        # T-junction structure by rebuilding the basis from the global knot
        # vectors (which do not record partial lines).
        self._has_partial_refinements: bool = False

        # Build the tensor-product basis
        self._basis: List[LRBasisFunction] = []
        self._build_tensor_basis()
        self._update_element_supports()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def degree_u(self) -> int:
        return self._degree_u

    @property
    def degree_v(self) -> int:
        return self._degree_v

    @property
    def mesh(self) -> LRMesh:
        return self._mesh

    @property
    def basis(self) -> List[LRBasisFunction]:
        return self._basis

    @property
    def nfuncs(self) -> int:
        return len(self._basis)

    # ------------------------------------------------------------------
    # Initial tensor-product basis construction
    # ------------------------------------------------------------------

    def _build_tensor_basis(self) -> None:
        """
        Enumerate all tensor-product B-splines from the current global
        knot vectors.  Each basis function gets coefficient = 1.
        """
        p = self._degree_u
        q = self._degree_v
        T_u = self._global_knots_u
        T_v = self._global_knots_v
        n_u = len(T_u) - p - 1
        n_v = len(T_v) - q - 1

        assert n_u >= 1 and n_v >= 1, "Not enough knots for the given degree."

        self._basis = []
        idx = 0
        for i in range(n_u):
            ku = T_u[i: i + p + 2]
            end_u = (i == n_u - 1)
            for j in range(n_v):
                kv = T_v[j: j + q + 2]
                end_v = (j == n_v - 1)
                B = LRBasisFunction(
                    id=idx,
                    knots_u=ku, knots_v=kv,
                    degree_u=p, degree_v=q,
                    end_u=end_u, end_v=end_v,
                    coefficient=1.0)
                self._basis.append(B)
                idx += 1

    # ------------------------------------------------------------------
    # Refinement: full line
    # ------------------------------------------------------------------

    def refine_full_line(self, axis: int, value: float) -> None:
        """
        Insert a knot line spanning the entire domain in the given axis.

        This is standard one-step Boehm knot insertion applied globally
        in one parametric direction.  Partition of unity is always preserved.

        Parameters
        ----------
        axis  : 0 = vertical line (insert into u-knot vector),
                1 = horizontal line (insert into v-knot vector)
        value : the knot value to insert
        """
        if self._has_partial_refinements:
            raise RuntimeError(
                "refine_full_line() cannot be called after partial (T-junction) "
                "line insertions: rebuilding the tensor-product basis would "
                "silently discard all earlier local refinements.  "
                "Use refine_line() with a full-span MeshLine instead, or "
                "start from a fresh LRSplineSpace."
            )

        if axis == 0:
            T_new, alpha = _refine_knot_vector(
                self._global_knots_u, self._degree_u, value)
            self._global_knots_u = T_new
        else:
            T_new, alpha = _refine_knot_vector(
                self._global_knots_v, self._degree_v, value)
            self._global_knots_v = T_new

        # Rebuild the entire tensor-product basis from the new knot vectors.
        # Partition of unity is automatic for the new tensor-product basis.
        self._build_tensor_basis()

        # Add the mesh line and rebuild element partition
        u0, u1 = self._mesh.u_domain
        v0, v1 = self._mesh.v_domain
        if axis == 0:
            line = MeshLine(axis=0, value=value, start=v0, end=v1)
        else:
            line = MeshLine(axis=1, value=value, start=u0, end=u1)
        self._mesh.insert_line(line)
        self._update_element_supports()

    # ------------------------------------------------------------------
    # Refinement: partial line (LR B-spline local refinement)
    # ------------------------------------------------------------------

    def refine_line(self, line: MeshLine) -> None:
        """
        Insert a (possibly partial) mesh line segment.

        For a **full** line (spanning the entire perpendicular domain),
        this is identical to ``refine_full_line`` and always preserves PoU.

        For a **partial** line (T-junction), only basis functions whose
        support is fully spanned by the line in the tangent direction are
        split.  This is the canonical LR B-spline overloading algorithm
        (Dokken, Lyche & Pettersen 2013).

        The split uses the exact Boehm coefficients derived from the
        *local* knot vector of each overloaded function, guaranteeing
        that the children's coefficients satisfy:
            Σ (child.coeff * child.pure_B) = parent.coeff * parent.pure_B
        and partition of unity is preserved.

        Parameters
        ----------
        line : MeshLine
        """
        u0, u1 = self._mesh.u_domain
        v0, v1 = self._mesh.v_domain
        tol = 1e-12

        is_full = (
            (line.axis == 0 and abs(line.start - v0) < tol and abs(line.end - v1) < tol) or
            (line.axis == 1 and abs(line.start - u0) < tol and abs(line.end - u1) < tol)
        )

        if is_full:
            self.refine_full_line(line.axis, line.value)
            return

        # Partial line: use overloading algorithm
        from LRSplines.src.refinement import _split_overloaded_lr
        _split_overloaded_lr(self, line)
        self._mesh.insert_line(line)
        self._reassign_ids()
        self._update_element_supports()
        self._has_partial_refinements = True

    # ------------------------------------------------------------------
    # Element-support mapping
    # ------------------------------------------------------------------

    def _update_element_supports(self) -> None:
        for el in self._mesh.elements:
            el.active_functions = []
        for k, B in enumerate(self._basis):
            for el in self._mesh.elements:
                if B.support_overlaps_element(el.u0, el.u1, el.v0, el.v1):
                    el.active_functions.append(k)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate all active basis functions at parametric points.

        Parameters
        ----------
        pts : ndarray, shape (N, 2)

        Returns
        -------
        B : ndarray, shape (N, nfuncs)
        """
        pts = np.asarray(pts, dtype=float)
        N = pts.shape[0]
        out = np.zeros((N, self.nfuncs), dtype=float)
        for k, B in enumerate(self._basis):
            out[:, k] = B.eval_array(pts)
        return out

    def evaluate_grad(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate gradients of all active basis functions.

        Parameters
        ----------
        pts : ndarray, shape (N, 2)

        Returns
        -------
        G : ndarray, shape (N, nfuncs, 2)
        """
        pts = np.asarray(pts, dtype=float)
        N = pts.shape[0]
        out = np.zeros((N, self.nfuncs, 2), dtype=float)
        for k, B in enumerate(self._basis):
            out[:, k, :] = B.grad_array(pts)
        return out

    # ------------------------------------------------------------------
    # Partition of unity
    # ------------------------------------------------------------------

    def check_partition_of_unity(self, n_pts: int = 25,
                                  tol: float = 1e-8) -> bool:
        """Return True if Σ_k B_k(u,v) = 1 at random interior points."""
        rng = np.random.default_rng(42)
        u0, u1 = self._mesh.u_domain
        v0, v1 = self._mesh.v_domain
        eps = 1e-4
        pts = np.column_stack([
            rng.uniform(u0 + eps, u1 - eps, n_pts),
            rng.uniform(v0 + eps, v1 - eps, n_pts)])
        B = self.evaluate(pts)
        return bool(np.all(np.abs(B.sum(axis=1) - 1.0) < tol))

    # ------------------------------------------------------------------
    # Greville abscissae
    # ------------------------------------------------------------------

    @property
    def greville_points(self) -> np.ndarray:
        return np.array([B.greville for B in self._basis])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reassign_ids(self) -> None:
        for k, B in enumerate(self._basis):
            B.id = k

    def __repr__(self) -> str:
        return (f"LRSplineSpace(nfuncs={self.nfuncs}, "
                f"nelements={self._mesh.nelements}, "
                f"degree=({self._degree_u},{self._degree_v}))")
