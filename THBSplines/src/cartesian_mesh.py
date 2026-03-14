"""
Cartesian (single-level, uniform) mesh.

A ``CartesianMesh`` represents a regular tensor-product grid in ``d``
parametric dimensions.  Each cell is an axis-aligned bounding box (AABB)
stored as an array of shape ``(d, 2)``:

    cell[i] = [left_i, right_i]   for i = 0, …, d-1

All cells have the same volume (``cell_area``), which is exploited during
finite-element assembly.

The mesh is refined dyadically: every knot interval is split at its midpoint,
doubling the number of cells in each direction.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from THBSplines.src.abstract_mesh import Mesh


class CartesianMesh(Mesh):
    """
    Regular Cartesian grid in ``parametric_dimension`` dimensions.

    Parameters
    ----------
    knots               : list of 1-D knot vectors, one per direction.
                          Repeated knots are silently de-duplicated.
    parametric_dimension: spatial dimension d (1, 2, …)
    """

    def __init__(self, knots: list, parametric_dimension: int):
        # De-duplicate knots in each direction (unique keeps sorted order)
        self.knots = np.array(
            [np.unique(np.asarray(kv, dtype=np.float64)) for kv in knots],
            dtype=object,
        )
        self.dim    = parametric_dimension
        self.cells  = self._compute_cells()
        self.nelems = len(self.cells)
        # All cells are the same size; pre-compute it once
        self.cell_area = float(np.prod(np.diff(self.cells[0])))

    # ------------------------------------------------------------------
    # Cell construction
    # ------------------------------------------------------------------

    def _compute_cells(self) -> np.ndarray:
        """
        Build the array of all cells as AABBs.

        For a 1-D mesh with knots [0, 1, 2] the cells are:
            [[0, 1]],  [[1, 2]]   → shape (2, 1, 2)

        For a 2-D mesh the cells come from the tensor product of the 1-D
        intervals.  The Cartesian product is computed via ``np.meshgrid``
        and then reshaped into the canonical ``(N, d, 2)`` layout.

        Returns
        -------
        np.ndarray, shape (N_cells, d, 2)
        """
        knots_left  = [k[:-1] for k in self.knots]   # left  endpoints of intervals
        knots_right = [k[1:]  for k in self.knots]   # right endpoints of intervals

        # Use meshgrid to form all combinations of left/right endpoints
        bl = np.stack(np.meshgrid(*knots_left),  -1).reshape(-1, self.dim)
        tr = np.stack(np.meshgrid(*knots_right), -1).reshape(-1, self.dim)

        # cells[i, j, :] = [left_j, right_j] for cell i in direction j
        cells = np.concatenate((bl, tr), axis=1).reshape(-1, self.dim, 2)

        # For d ≥ 2 meshgrid returns coordinates in (y, x, …) order; swap to (x, y, …)
        # For d == 1 the swap is a no-op, so we can apply it unconditionally.
        if self.dim != 1:
            cells = np.swapaxes(cells, 1, 2)

        return cells

    # Alias used by the rest of the package (old name)
    def compute_cells(self) -> np.ndarray:
        return self._compute_cells()

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def refine(self) -> "CartesianMesh":
        """
        Dyadically refine the mesh by inserting the midpoint of every knot
        interval in each direction.

        Returns
        -------
        CartesianMesh
            A new mesh with twice as many cells per direction.
        """
        refined_knots = [
            np.sort(np.concatenate((kv, (kv[:-1] + kv[1:]) / 2.0)))
            for kv in self.knots
        ]
        return CartesianMesh(refined_knots, self.dim)

    # ------------------------------------------------------------------
    # Sub-element lookup
    # ------------------------------------------------------------------

    def get_sub_elements(self, box: np.ndarray) -> list[int]:
        """
        Return the indices of all cells contained within ``box``.

        A cell is considered *contained* when its lower-left corner is
        ≥ box's lower-left corner and its upper-right corner is
        ≤ box's upper-right corner (in every direction).

        Parameters
        ----------
        box : array of shape (d, 2), where box[i] = [lo_i, hi_i]

        Returns
        -------
        list[int]
        """
        indices = []
        for i, cell in enumerate(self.cells):
            # cell[:, 0] are left edges, cell[:, 1] are right edges
            if np.all((cell[:, 0] >= box[:, 0]) & (cell[:, 1] <= box[:, 1])):
                indices.append(i)
        return indices

    # ------------------------------------------------------------------
    # Quadrature  (implements abstract method)
    # ------------------------------------------------------------------

    def get_gauss_points(
        self, cell_indices: np.ndarray, order: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss–Legendre quadrature points and weights for a set of cells.

        The reference-interval Gauss rule on [-1, 1] is mapped affinely to
        each physical cell [a, b] via:

            x_phys = 0.5 * (x_ref + 1) * (b - a) + a

        The Jacobian of this map is ``(b - a) / 2`` per direction, so the
        full d-dimensional Jacobian is ``cell_area / 2^d``.

        Parameters
        ----------
        cell_indices : 1-D integer array
        order        : number of Gauss points per direction

        Returns
        -------
        points  : (N_q, d) array of physical quadrature points
        weights : (N_q,) array of quadrature weights (Jacobian already applied)
        """
        ref_pts, ref_wts = np.polynomial.legendre.leggauss(order)

        all_points  = []
        all_weights = []

        for ci in cell_indices:
            cell = self.cells[ci]  # shape (d, 2)
            dim  = cell.shape[0]

            # Map reference points to physical coordinates in each direction
            phys_pts_per_dir = [
                0.5 * (ref_pts + 1.0) * (cell[j, 1] - cell[j, 0]) + cell[j, 0]
                for j in range(dim)
            ]

            # Tensor-product quadrature points
            grids = np.meshgrid(*phys_pts_per_dir, indexing="ij")
            pts   = np.stack(grids, axis=-1).reshape(-1, dim)

            # Tensor-product weights (reference), then scale by Jacobian
            wt_grids = np.meshgrid(*[ref_wts] * dim, indexing="ij")
            wts_ref  = np.prod(np.stack(wt_grids, axis=-1).reshape(-1, dim), axis=1)
            jacobian = float(np.prod(cell[:, 1] - cell[:, 0])) / 2**dim
            wts      = wts_ref * jacobian

            all_points.append(pts)
            all_weights.append(wts)

        points  = np.concatenate(all_points,  axis=0)
        weights = np.concatenate(all_weights, axis=0)
        return points, weights

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_cells(self, ax=None) -> None:
        """
        Plot the mesh cells using Matplotlib.

        Only meaningful for 1-D and 2-D meshes.

        Parameters
        ----------
        ax : optional Matplotlib Axes object.  If None, ``plt.gca()`` is used.
        """
        if ax is None:
            ax = plt.gca()

        if self.dim == 1:
            y = 0.0
            for cell in self.cells:
                x0, x1 = cell[0]
                ax.plot([x0, x1], [y, y], color="black")
        elif self.dim == 2:
            for cell in self.cells:
                x = cell[0, [0, 1, 1, 0, 0]]
                y = cell[1, [0, 0, 1, 1, 0]]
                ax.plot(x, y, color="black", linewidth=0.5)
        else:
            raise NotImplementedError("plot_cells only supports d ≤ 2")
