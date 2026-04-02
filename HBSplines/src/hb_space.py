"""
Hierarchical B-Spline space (HB-splines) — 1D.

Data model
----------
The space is described by a tuple  (sp_lev, A, D)  where:

  sp_lev[l]  : BsplineSpace at level l (dyadically refined from level 0)
  A[l]       : boolean mask of *active*      basis functions at level l
  D[l]       : boolean mask of *deactivated* basis functions at level l

Active functions are those that are currently "live" (not deactivated) at
some level.  The full active set is the union  ⋃_l { i : A[l][i] }  and
the total DOF count is  sum_l |A[l]|.

Refinement step
---------------
Given a set of active function indices (level l, indices idx):
  1. For each i in idx, deactivate i:  A[l][i] = False, D[l][i] = True.
  2. If level l+1 does not yet exist, append sp_lev[l].refine().
  3. For each deactivated i, activate its p+2 children at level l+1.

Boundary functions
------------------
Function 0 at any level touches the left boundary.
Function sp_lev[l].dim-1 touches the right boundary.
These are treated as Dirichlet DOFs in the assembly.

Note: HB-splines do NOT truncate, so the basis functions are *not* a
partition of unity when the space is non-uniformly refined.  This is the
key difference from THB-splines.  See Giannelli et al. (2012) for the
truncation mechanism that fixes this.
"""

from __future__ import annotations

from typing import List

import numpy as np

from HBSplines.src.bspline_space import BsplineSpace


class HBsplineSpace:
    """
    Hierarchical B-spline space built by successive dyadic refinements.

    Parameters
    ----------
    knots : array-like
        Clamped knot vector for the coarsest level.
    degree : int
        Polynomial degree (same at every level).
    """

    def __init__(self, knots, degree: int) -> None:
        sp0 = BsplineSpace(knots, degree)
        self._sp_lev: List[BsplineSpace] = [sp0]
        # All functions active at level 0, none deactivated
        self._A: List[np.ndarray] = [np.ones(sp0.dim, dtype=bool)]
        self._D: List[np.ndarray] = [np.zeros(sp0.dim, dtype=bool)]

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def nlevels(self) -> int:
        return len(self._sp_lev)

    @property
    def sp_lev(self) -> List[BsplineSpace]:
        return self._sp_lev

    @property
    def degree(self) -> int:
        return self._sp_lev[0].degree

    @property
    def domain(self) -> tuple[float, float]:
        return self._sp_lev[0].domain

    @property
    def nfuncs(self) -> int:
        """Total number of active DOFs (sum of active functions across levels)."""
        return sum(int(A.sum()) for A in self._A)

    def active_indices(self, level: int) -> np.ndarray:
        """Integer indices of active functions at *level*."""
        return np.where(self._A[level])[0]

    def is_boundary(self, level: int, i: int) -> bool:
        """True if function i at level is a Dirichlet boundary DOF."""
        return i == 0 or i == self._sp_lev[level].dim - 1

    # ------------------------------------------------------------------
    # Global DOF ordering
    # ------------------------------------------------------------------

    def global_dofs(self) -> list[tuple[int, int]]:
        """
        Return ordered list of  (level, local_index)  for every active DOF.

        The order is: level 0 active functions, then level 1, etc.
        """
        dofs = []
        for lev in range(self.nlevels):
            for i in self.active_indices(lev):
                dofs.append((lev, int(i)))
        return dofs

    def interior_dofs(self) -> list[tuple[int, int]]:
        """Active DOFs that are not on the boundary (not Dirichlet)."""
        return [
            (lev, i)
            for lev, i in self.global_dofs()
            if not self.is_boundary(lev, i)
        ]

    def boundary_dofs(self) -> list[tuple[int, int]]:
        """Active DOFs on the boundary (Dirichlet)."""
        return [
            (lev, i)
            for lev, i in self.global_dofs()
            if self.is_boundary(lev, i)
        ]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval_dof(self, x: np.ndarray, level: int, i: int) -> np.ndarray:
        """Evaluate the (level, i) basis function at points x."""
        return self._sp_lev[level].eval(x, i)

    def eval_dof_deriv(self, x: np.ndarray, level: int, i: int, r: int = 1) -> np.ndarray:
        """Evaluate r-th derivative of the (level, i) basis function at points x."""
        return self._sp_lev[level].deriv(x, i, r)

    def eval_all(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all active basis functions at points x.

        Returns
        -------
        B : ndarray, shape (len(x), nfuncs)
            Columns ordered as in ``global_dofs()``.
        """
        x = np.asarray(x, dtype=float)
        cols = [self.eval_dof(x, lev, i) for lev, i in self.global_dofs()]
        return np.column_stack(cols) if cols else np.empty((len(x), 0))

    def eval_solution(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate u_h(x) = sum_k c_k * B_k(x) for all active DOFs."""
        B = self.eval_all(x)
        return B @ coeffs

    def eval_solution_deriv(self, x: np.ndarray, coeffs: np.ndarray, r: int = 1) -> np.ndarray:
        """Evaluate u_h^(r)(x) for all active DOFs."""
        x = np.asarray(x, dtype=float)
        cols = [self.eval_dof_deriv(x, lev, i, r) for lev, i in self.global_dofs()]
        dB = np.column_stack(cols) if cols else np.empty((len(x), 0))
        return dB @ coeffs

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def _ensure_level(self, lev: int) -> None:
        """Append level lev if it does not yet exist."""
        while len(self._sp_lev) <= lev:
            sp_new = self._sp_lev[-1].refine()
            self._sp_lev.append(sp_new)
            self._A.append(np.zeros(sp_new.dim, dtype=bool))
            self._D.append(np.zeros(sp_new.dim, dtype=bool))

    def refine(self, level: int, indices: np.ndarray) -> None:
        """
        Deactivate basis functions at *level* and activate their children.

        Parameters
        ----------
        level   : source level (0-indexed)
        indices : integer array of active function indices at *level* to refine
        """
        indices = np.asarray(indices, dtype=int)
        self._ensure_level(level + 1)

        sp_cur = self._sp_lev[level]
        for i in indices:
            if not self._A[level][i]:
                continue  # already deactivated
            # Deactivate at current level
            self._A[level][i] = False
            self._D[level][i] = True
            # Activate children at next level
            children = sp_cur.get_children(i)
            self._A[level + 1][children] = True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def check_partition_of_unity(self, n_pts: int = 50) -> bool:
        """
        Sample n_pts interior points and check sum of active functions ≈ 1.

        Note: HB-splines do NOT form a partition of unity after non-uniform
        refinement — this method quantifies the deviation.
        """
        a, b = self.domain
        x = np.linspace(a, b, n_pts + 2)[1:-1]
        B = self.eval_all(x)
        sums = B.sum(axis=1)
        return bool(np.allclose(sums, 1.0, atol=1e-10))

    def summary(self) -> str:
        lines = [f"HBsplineSpace: degree={self.degree}, nlevels={self.nlevels}, nfuncs={self.nfuncs}"]
        for lev in range(self.nlevels):
            act = int(self._A[lev].sum())
            deact = int(self._D[lev].sum())
            lines.append(f"  level {lev}: dim={self._sp_lev[lev].dim}, active={act}, deactivated={deact}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"HBsplineSpace(degree={self.degree}, nlevels={self.nlevels}, nfuncs={self.nfuncs})"
