"""
LR refinement: knot line insertion and basis function splitting.

Theory
------
Given an LR B-spline space V and a new mesh line segment L, the refined
space V' is obtained by:

  1. For each basis function B in V, test whether L *overloads* B.
     A line segment L overloads B if:
       - L is contained in the *interior* of B's support in the normal direction.
       - L spans the *entire* width of B's support in the tangent direction.
     (This is the definition from Dokken, Lyche & Pettersen 2013.)

  2. If L overloads B:
     - Replace B with two children B1, B2 obtained by inserting the line's
       fixed-axis value into B's corresponding local knot vector (one-step
       Boehm knot insertion).
     - B1 and B2 reproduce B exactly:  B = alpha_1*B1 + alpha_2*B2.
     - Mark B as removed.

  3. Insert L into the mesh (LRMesh.insert_line).

  4. Reassign IDs and rebuild element-support mapping.

Local Linear Independence (LLI)
---------------------------------
After inserting a partial line, the LLI condition may be violated on some
element: it requires that the number of basis functions with support on an
element does not exceed (p_u + 1) * (p_v + 1).

**WARNING**: This implementation does NOT automatically enforce LLI.
After a partial insertion, ``check_lli`` should be called to verify the
condition.  If it fails, the user must choose a compatible refinement
strategy (e.g. extend lines manually or use only full-line insertions).

Reference:  Bressan & Jüttler (2018) "Inf-sup stable finite element
pairs of arbitrary order on structured meshes", IMA J. Numer. Anal.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import TYPE_CHECKING, List

from LRSplines.src.lr_mesh import MeshLine
from LRSplines.src.lr_basis import LRBasisFunction

if TYPE_CHECKING:
    from LRSplines.src.lr_spline_space import LRSplineSpace


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def refine(space: 'LRSplineSpace', line: MeshLine) -> None:
    """
    Insert ``line`` into ``space``, splitting overloaded basis functions.

    This is the main public function.  It modifies ``space`` in-place.
    Delegates to ``space.refine_line`` which correctly handles full lines
    via global knot insertion (guaranteeing PoU) and partial lines via
    the overloading algorithm.

    Parameters
    ----------
    space : LRSplineSpace
    line  : MeshLine
        The knot line segment to insert.
    """
    space.refine_line(line)

    # Warn if the insertion produced a space that violates LLI.
    if not check_lli(space):
        warnings.warn(
            "LLI violated after partial line insertion.  "
            "Call check_lli(space) / lli_report(space) for details.  "
            "Consider extending lines manually to restore LLI.",
            stacklevel=2,
        )


def refine_region(space: 'LRSplineSpace',
                  u0: float, u1: float,
                  v0: float, v1: float,
                  n_lines_u: int = 1,
                  n_lines_v: int = 1) -> None:
    """
    Uniformly refine a rectangular parametric region [u0,u1] x [v0,v1].

    Inserts ``n_lines_u`` equally-spaced vertical line segments and
    ``n_lines_v`` equally-spaced horizontal line segments, each clipped
    to the given rectangle.

    Parameters
    ----------
    space            : LRSplineSpace
    u0, u1, v0, v1   : float  — rectangle corners
    n_lines_u        : int    — number of new vertical lines
    n_lines_v        : int    — number of new horizontal lines
    """
    # Vertical lines (axis=0, fixed u)
    if n_lines_u > 0:
        for u in np.linspace(u0, u1, n_lines_u + 2)[1:-1]:
            ln = MeshLine(axis=0, value=u, start=v0, end=v1)
            refine(space, ln)

    # Horizontal lines (axis=1, fixed v)
    if n_lines_v > 0:
        for v in np.linspace(v0, v1, n_lines_v + 2)[1:-1]:
            ln = MeshLine(axis=1, value=v, start=u0, end=u1)
            refine(space, ln)


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

def _split_overloaded(space: 'LRSplineSpace', line: MeshLine) -> None:
    """
    Scan all basis functions and replace those overloaded by ``line`` with
    their two children.  Modifies ``space._basis`` in-place.
    """
    new_basis: List[LRBasisFunction] = []

    for B in space.basis:
        if _is_overloaded(B, line):
            # Split B by inserting the line's fixed value into the
            # appropriate local knot vector.
            if line.axis == 0:
                # Vertical line (fixed u = line.value): insert into knots_u
                B1, B2 = B.insert_knot_u(line.value)
            else:
                # Horizontal line (fixed v = line.value): insert into knots_v
                B1, B2 = B.insert_knot_v(line.value)

            new_basis.append(B1)
            new_basis.append(B2)
        else:
            new_basis.append(B)

    space._basis = new_basis


# Alias used by LRSplineSpace.refine_line for partial line insertion
_split_overloaded_lr = _split_overloaded


def _is_overloaded(B: LRBasisFunction, line: MeshLine,
                   *, tol: float = 1e-14) -> bool:
    """
    Test whether mesh line ``line`` overloads basis function ``B``.

    A line segment L overloads B if and only if:
      (a) L's fixed-axis value is strictly inside B's support in that direction.
      (b) L's extent in the free-axis direction covers the *entire* projection
          of B's support onto that axis (i.e. L spans B fully in the free direction).

    Condition (b) ensures the split produces two functions that together
    reproduce B exactly.

    Parameters
    ----------
    B    : LRBasisFunction
    line : MeshLine

    Returns
    -------
    bool
    """
    u0, u1, v0, v1 = B.support

    if line.axis == 0:
        # Vertical line: fixed u = line.value, spans v in [line.start, line.end]
        # (a) line.value is strictly inside B's u-support
        if not (u0 + tol < line.value < u1 - tol):
            return False
        # (b) line spans the full v-projection of B's support
        if not (line.start <= v0 + tol and line.end >= v1 - tol):
            return False
    else:
        # Horizontal line: fixed v = line.value, spans u in [line.start, line.end]
        # (a) line.value is strictly inside B's v-support
        if not (v0 + tol < line.value < v1 - tol):
            return False
        # (b) line spans the full u-projection of B's support
        if not (line.start <= u0 + tol and line.end >= u1 - tol):
            return False

    return True


# ---------------------------------------------------------------------------
# LLI check utility
# ---------------------------------------------------------------------------

def check_lli(space: 'LRSplineSpace') -> bool:
    """
    Verify the Local Linear Independence (LLI) condition.

    LLI requires that for every element e, the number of basis functions
    with support on e satisfies:
        |{B : supp(B) ∩ int(e) ≠ ∅}| <= (p_u + 1) * (p_v + 1)

    Returns
    -------
    bool  True if LLI holds on every element.
    """
    max_allowed = (space.degree_u + 1) * (space.degree_v + 1)
    for el in space.mesh.elements:
        if len(el.active_functions) > max_allowed:
            return False
    return True


def lli_report(space: 'LRSplineSpace') -> dict:
    """
    Return a dict mapping element index to the number of active functions.
    Elements violating LLI are flagged.
    """
    max_allowed = (space.degree_u + 1) * (space.degree_v + 1)
    report = {}
    for i, el in enumerate(space.mesh.elements):
        n = len(el.active_functions)
        report[i] = {
            'element': el,
            'n_funcs': n,
            'violates_lli': n > max_allowed,
        }
    return report
