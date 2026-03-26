"""
LR mesh: knot line segments and rectangular elements forming a T-mesh.

An LR mesh over the parametric domain [u0, u1] x [v0, v1] consists of:
  - A set of horizontal knot line *segments*: constant v-value over an
    x-interval [xa, xb].
  - A set of vertical knot line *segments*: constant u-value over a
    y-interval [ya, yb].

The intersections of all segments partition the domain into rectangular
*elements*.  Unlike a full tensor-product mesh, segments need not span
the entire domain (T-junctions are allowed), which is what enables
*local* refinement.

Key references
--------------
Dokken, Lyche, Pettersen (2013) "Polynomial splines over locally refined
box-partitions", CAGD 30, pp. 331-356.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MeshLine:
    """
    A single knot line segment.

    For a *horizontal* segment (constant v = value):
        axis  = 1   (the v-axis is fixed)
        value = v0
        start = x0,  end = x1   (x0 < x1)

    For a *vertical* segment (constant u = value):
        axis  = 0   (the u-axis is fixed)
        value = u0
        start = y0,  end = y1   (y0 < y1)

    multiplicity : int
        Knot multiplicity (1 = C^{p-1} continuity, p = C^0, etc.)
    """
    axis: int          # 0 = vertical line (fixed u), 1 = horizontal (fixed v)
    value: float       # the fixed coordinate
    start: float       # lower bound along the free axis
    end: float         # upper bound along the free axis
    multiplicity: int = 1

    def __post_init__(self):
        assert self.start < self.end, "MeshLine: start must be < end"
        assert self.axis in (0, 1), "axis must be 0 (vertical) or 1 (horizontal)"
        assert self.multiplicity >= 1

    def contains(self, t: float, *, tol: float = 1e-14) -> bool:
        """True if t lies in (start, end) strictly (excluding endpoints)."""
        return self.start + tol < t < self.end - tol

    def overlaps_open(self, a: float, b: float, *, tol: float = 1e-14) -> bool:
        """True if the segment's open interior overlaps the open interval (a, b)."""
        return self.start < b - tol and self.end > a + tol

    def __repr__(self) -> str:
        direction = "horizontal" if self.axis == 1 else "vertical"
        return (f"MeshLine({direction}, fixed={'v' if self.axis==1 else 'u'}="
                f"{self.value:.4g}, range=[{self.start:.4g},{self.end:.4g}], "
                f"mult={self.multiplicity})")


@dataclass
class Element:
    """
    A rectangular parametric element [u0, u1] x [v0, v1].

    active_functions : list[int]
        Indices (into the LRSplineSpace.basis list) of basis functions
        whose support overlaps this element's *interior*.
    """
    u0: float
    u1: float
    v0: float
    v1: float
    active_functions: List[int] = field(default_factory=list)

    @property
    def area(self) -> float:
        return (self.u1 - self.u0) * (self.v1 - self.v0)

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.u0 + self.u1), 0.5 * (self.v0 + self.v1))

    def contains_point(self, u: float, v: float, *, tol: float = 1e-14) -> bool:
        """True if (u, v) lies strictly inside this element."""
        return (self.u0 - tol <= u <= self.u1 + tol and
                self.v0 - tol <= v <= self.v1 + tol)

    def __repr__(self) -> str:
        return (f"Element([{self.u0:.4g},{self.u1:.4g}]"
                f"x[{self.v0:.4g},{self.v1:.4g}],"
                f" nfuncs={len(self.active_functions)})")


# ---------------------------------------------------------------------------
# LR Mesh
# ---------------------------------------------------------------------------

class LRMesh:
    """
    T-mesh for a 2-D LR B-spline space.

    Parameters
    ----------
    u_knots : array-like
        Initial *full* knot lines in the u-direction (vertical lines).
        These span the entire v-domain.
    v_knots : array-like
        Initial *full* knot lines in the v-direction (horizontal lines).
        These span the entire u-domain.

    The constructor creates the initial full tensor-product mesh from the
    *unique* breakpoints of ``u_knots`` and ``v_knots``.
    """

    def __init__(self, u_knots: np.ndarray, v_knots: np.ndarray):
        u_knots = np.asarray(u_knots, dtype=float)
        v_knots = np.asarray(v_knots, dtype=float)

        self._u_domain = (u_knots[0], u_knots[-1])
        self._v_domain = (v_knots[0], v_knots[-1])

        # Unique breakpoints (without repetition)
        u_breaks = np.unique(u_knots)
        v_breaks = np.unique(v_knots)

        # Knot multiplicities for the initial full lines
        # (a repeated knot in the global vector → higher multiplicity)
        u_mult = {val: int(np.sum(u_knots == val)) for val in u_breaks}
        v_mult = {val: int(np.sum(v_knots == val)) for val in v_breaks}

        # Build initial mesh lines (full-width / full-height)
        self._lines: List[MeshLine] = []

        # Vertical lines (axis=0, fixed u, span full v-domain)
        for u in u_breaks:
            self._lines.append(MeshLine(
                axis=0, value=u,
                start=self._v_domain[0], end=self._v_domain[1],
                multiplicity=u_mult[u]))

        # Horizontal lines (axis=1, fixed v, span full u-domain)
        for v in v_breaks:
            self._lines.append(MeshLine(
                axis=1, value=v,
                start=self._u_domain[0], end=self._u_domain[1],
                multiplicity=v_mult[v]))

        # Build the initial element partition
        self._elements: List[Element] = []
        self._rebuild_elements()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def u_domain(self) -> Tuple[float, float]:
        return self._u_domain

    @property
    def v_domain(self) -> Tuple[float, float]:
        return self._v_domain

    @property
    def lines(self) -> List[MeshLine]:
        return list(self._lines)

    @property
    def elements(self) -> List[Element]:
        return list(self._elements)

    @property
    def nelements(self) -> int:
        return len(self._elements)

    def u_breaks(self) -> np.ndarray:
        """Unique u-values of all vertical mesh lines."""
        vals = sorted({ln.value for ln in self._lines if ln.axis == 0})
        return np.array(vals)

    def v_breaks(self) -> np.ndarray:
        """Unique v-values of all horizontal mesh lines."""
        vals = sorted({ln.value for ln in self._lines if ln.axis == 1})
        return np.array(vals)

    # ------------------------------------------------------------------
    # Element lookup
    # ------------------------------------------------------------------

    def find_element(self, u: float, v: float) -> int:
        """
        Return the index of the element containing (u, v).
        Raises ValueError if (u, v) is outside the domain.
        """
        for i, el in enumerate(self._elements):
            if el.contains_point(u, v):
                return i
        raise ValueError(f"Point ({u}, {v}) is outside the mesh domain.")

    # ------------------------------------------------------------------
    # Mesh line insertion (used by refinement)
    # ------------------------------------------------------------------

    def insert_line(self, line: MeshLine) -> None:
        """
        Add a new knot line segment to the mesh and rebuild the element
        partition.  Does *not* update basis function supports — that is
        the responsibility of LRSplineSpace.refine().
        """
        self._lines.append(line)
        self._rebuild_elements()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_elements(self) -> None:
        """
        Recompute the element partition from the current set of mesh lines.

        Strategy
        --------
        Consider ALL pairs of u-breakpoints and ALL pairs of v-breakpoints as
        candidate rectangle corners (not only consecutive pairs).  A rectangle
        [u_a, u_b] x [v_a, v_b] is a valid element if and only if:

          1. All four sides are each covered by at least one mesh line segment
             of the matching axis (boundary-coverage check).  This prevents
             phantom elements at T-junctions.

          2. No mesh line segment passes through the interior of the rectangle
             (interior-cut check).  This prevents non-minimal elements: if a
             full interior line exists, the rectangle is already sub-divided.

        By considering all breakpoint pairs (not just consecutive ones), merged
        elements that span across a partial-line T-junction are correctly
        discovered.  For example, a partial vertical line at u=0.5 spanning
        only v ∈ [0,1] should not split the element [0,1] x [1,2] — that
        rectangle is still valid because u=0.5 is not a boundary for v ∈ [1,2]
        and does not cut its interior.
        """
        u_vals = sorted({ln.value for ln in self._lines if ln.axis == 0})
        v_vals = sorted({ln.value for ln in self._lines if ln.axis == 1})

        if len(u_vals) < 2 or len(v_vals) < 2:
            self._elements = []
            return

        tol = 1e-14
        elements = []

        for ia in range(len(u_vals)):
            for ib in range(ia + 1, len(u_vals)):
                u0, u1 = u_vals[ia], u_vals[ib]

                for ja in range(len(v_vals)):
                    for jb in range(ja + 1, len(v_vals)):
                        v0, v1 = v_vals[ja], v_vals[jb]

                        # ── Boundary coverage check ───────────────────────
                        left_ok = any(
                            ln.axis == 0 and abs(ln.value - u0) < tol and
                            ln.start <= v0 + tol and ln.end >= v1 - tol
                            for ln in self._lines)
                        if not left_ok:
                            continue
                        right_ok = any(
                            ln.axis == 0 and abs(ln.value - u1) < tol and
                            ln.start <= v0 + tol and ln.end >= v1 - tol
                            for ln in self._lines)
                        if not right_ok:
                            continue
                        bottom_ok = any(
                            ln.axis == 1 and abs(ln.value - v0) < tol and
                            ln.start <= u0 + tol and ln.end >= u1 - tol
                            for ln in self._lines)
                        if not bottom_ok:
                            continue
                        top_ok = any(
                            ln.axis == 1 and abs(ln.value - v1) < tol and
                            ln.start <= u0 + tol and ln.end >= u1 - tol
                            for ln in self._lines)
                        if not top_ok:
                            continue

                        # ── Interior cut check ────────────────────────────
                        cut = False
                        for ln in self._lines:
                            if ln.axis == 0:
                                if (u0 + tol < ln.value < u1 - tol and
                                        ln.overlaps_open(v0, v1)):
                                    cut = True
                                    break
                            else:
                                if (v0 + tol < ln.value < v1 - tol and
                                        ln.overlaps_open(u0, u1)):
                                    cut = True
                                    break

                        if not cut:
                            elements.append(Element(u0=u0, u1=u1, v0=v0, v1=v1))

        self._elements = elements

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, ax=None, *, title: str = "LR Mesh"):
        """
        Draw all mesh line segments on a matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If None, a new figure is created and returned.
        title : str
            Axes title.

        Returns
        -------
        fig, ax  (only when ax was None)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)

        # Draw element rectangles as light fill
        for el in self._elements:
            rect = patches.Rectangle(
                (el.u0, el.v0), el.u1 - el.u0, el.v1 - el.v0,
                linewidth=0, facecolor='#e8f4f8', alpha=0.6)
            ax.add_patch(rect)

        # Draw mesh line segments
        for ln in self._lines:
            lw = 0.8 + 0.6 * (ln.multiplicity - 1)
            color = '#1d3557' if ln.multiplicity == 1 else '#c1121f'
            if ln.axis == 0:   # vertical: fixed u
                ax.plot([ln.value, ln.value], [ln.start, ln.end],
                        color=color, lw=lw)
            else:              # horizontal: fixed v
                ax.plot([ln.start, ln.end], [ln.value, ln.value],
                        color=color, lw=lw)

        ax.set_xlim(self._u_domain[0] - 0.05, self._u_domain[1] + 0.05)
        ax.set_ylim(self._v_domain[0] - 0.05, self._v_domain[1] + 0.05)
        ax.set_aspect('equal')
        ax.set_xlabel('$u$')
        ax.set_ylabel('$v$')
        ax.set_title(title)

        if created:
            return fig, ax
