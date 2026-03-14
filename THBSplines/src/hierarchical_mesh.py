"""
Hierarchical mesh: a sequence of nested Cartesian grids.

A ``HierarchicalMesh`` maintains a list of increasingly fine
``CartesianMesh`` objects (levels 0, 1, 2, …).  At any point in time,
each cell of each level is classified as:

  - *active*      (``aelem_level[ℓ]``) : the cell is part of the current mesh
  - *deactivated* (``delem_level[ℓ]``) : the cell has been replaced by finer children

The union of all active cells over all levels forms a non-overlapping
covering of the parametric domain.

Refinement procedure
--------------------
When a set of active cells at level ℓ is marked for refinement:

1. Their 2^d children on the level-(ℓ+1) mesh are activated.
2. The marked cells are deactivated.

If level ℓ+1 does not yet exist, it is created by dyadically refining
the level-ℓ mesh.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from THBSplines.src.abstract_mesh import Mesh
from THBSplines.src.cartesian_mesh import CartesianMesh


class HierarchicalMesh(Mesh):
    """
    Multi-level hierarchically refined mesh.

    Parameters
    ----------
    knots : list of 1-D knot vectors defining the coarsest (level-0) mesh
    dim   : parametric dimension d
    """

    def __init__(self, knots: list, dim: int):
        # Level-0 Cartesian mesh
        self.meshes: list[CartesianMesh] = [CartesianMesh(knots, dim)]
        self.nlevels = 1

        n0 = self.meshes[0].nelems
        # Initially all level-0 cells are active
        self.aelem_level: dict[int, np.ndarray] = {
            0: np.arange(n0, dtype=np.int64)
        }
        # No cells deactivated yet
        self.delem_level: dict[int, np.ndarray] = {
            0: np.array([], dtype=np.int64)
        }
        self.nel_per_level: dict[int, int] = {0: n0}
        self.nel: int = n0
        self.cell_area_per_level: dict[int, float] = {0: self.meshes[0].cell_area}

    # ------------------------------------------------------------------
    # Level management
    # ------------------------------------------------------------------

    def add_level(self) -> None:
        """
        Append a new refinement level by dyadically refining the finest mesh.

        The new level starts with no active and no deactivated cells.
        Cells are activated as marked cells from the previous level are refined.
        """
        self.nlevels += 1
        self.meshes.append(self.meshes[-1].refine())

        lev = self.nlevels - 1
        self.aelem_level[lev]       = np.array([], dtype=np.int64)
        self.delem_level[lev]       = np.array([], dtype=np.int64)
        self.cell_area_per_level[lev] = self.meshes[-1].cell_area

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def refine(self, marked_cells: dict[int, list[int] | np.ndarray]) -> dict[int, np.ndarray]:
        """
        Refine the hierarchical mesh by deactivating marked cells and
        activating their children on the next level.

        Parameters
        ----------
        marked_cells : mapping  level → array of cell indices to refine.
                       All levels up to ``max(marked_cells.keys())`` must
                       exist (or will be created automatically).

        Returns
        -------
        dict
            ``new_elements[ℓ+1]`` contains the newly activated child-cell
            indices at level ℓ+1 for every refined level ℓ.
        """
        # Ensure enough levels exist
        while self.nlevels - 1 <= max(marked_cells.keys()):
            self.add_level()

        return self._update_active_cells(marked_cells)

    def _update_active_cells(
        self, marked_cells: dict[int, list[int] | np.ndarray]
    ) -> dict[int, np.ndarray]:
        """
        Apply the cell-state transitions implied by ``marked_cells``.

        For each level ℓ with marked cells:
          - Remove the marked cells from the active set at level ℓ.
          - Add them to the deactivated set at level ℓ.
          - Activate their children at level ℓ+1.

        Parameters
        ----------
        marked_cells : level → cell indices

        Returns
        -------
        new_cells : level+1 → newly activated cell indices
        """
        new_cells: dict[int, np.ndarray] = {}

        for level in range(len(marked_cells)):
            if level not in marked_cells:
                continue

            marked = np.asarray(marked_cells[level], dtype=np.int64)

            # Only refine cells that are currently active
            active_mask    = np.isin(marked, self.aelem_level[level])
            marked_active  = marked[active_mask]

            # Deactivate the marked active cells at this level
            self.aelem_level[level] = np.setdiff1d(
                self.aelem_level[level], marked_active
            ).astype(np.int64)
            self.delem_level[level] = np.union1d(
                self.delem_level[level], marked
            ).astype(np.int64)

            # Find and activate children at the next level
            children = self._get_children(level, marked)
            new_cells[level + 1] = children
            self.aelem_level[level + 1] = np.union1d(
                self.aelem_level[level + 1], children
            ).astype(np.int64)

        # Recompute total active element counts
        self.nel_per_level = {
            lev: len(self.aelem_level[lev])
            for lev in range(self.nlevels)
        }
        self.nel = sum(self.nel_per_level.values())

        return new_cells

    def _get_children(
        self, level: int, cell_indices: np.ndarray
    ) -> np.ndarray:
        """
        Return the indices of the fine-level cells that are children of
        the given coarse-level cells.

        A fine cell is a *child* of a coarse cell when it is geometrically
        contained inside it.  We check this with a small tolerance ``ε``
        to guard against floating-point boundary effects.

        Parameters
        ----------
        level       : coarse level index
        cell_indices: indices of coarse cells whose children are requested

        Returns
        -------
        np.ndarray of int64
        """
        children   = np.array([], dtype=np.int64)
        fine_cells = self.meshes[level + 1].cells  # shape (N_fine, d, 2)
        eps        = np.spacing(1)                  # machine epsilon

        for ci in cell_indices:
            coarse = self.meshes[level].cells[ci]  # shape (d, 2)
            # fine_cells[:, :, 0] are left edges, fine_cells[:, :, 1] right edges
            contained = np.all(
                (coarse[:, 0] <= fine_cells[:, :, 0] + eps) &
                (coarse[:, 1] + eps >= fine_cells[:, :, 1]),
                axis=1,
            )
            children = np.union1d(children, np.flatnonzero(contained))

        return children.astype(np.int64)

    # ------------------------------------------------------------------
    # Quadrature  (delegates to individual CartesianMesh objects)
    # ------------------------------------------------------------------

    def get_gauss_points(
        self, cell_indices: np.ndarray, order: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss quadrature points for a flat list of (level, cell_index)
        pairs.

        This is a thin wrapper; callers typically interact with individual
        level meshes directly via ``self.meshes[level].get_gauss_points``.

        Parameters
        ----------
        cell_indices : not used at this level — see CartesianMesh.get_gauss_points
        order        : Gauss order per direction

        Notes
        -----
        For the hierarchical case, assembly loops over levels and calls each
        level's ``CartesianMesh`` directly.  This method exists to satisfy
        the abstract interface.
        """
        raise NotImplementedError(
            "Call get_gauss_points on individual level meshes: "
            "self.meshes[level].get_gauss_points(cell_indices, order)"
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_cells(self, ax=None, return_fig: bool = False):
        """
        Plot all active cells across every refinement level.

        Each level's active cells are drawn with the same line style.
        The plot clearly shows where the mesh has been locally refined.

        Parameters
        ----------
        ax         : optional Matplotlib Axes.  Created if not provided.
        return_fig : if True, return the Figure object instead of calling plt.show().
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        for level in range(self.nlevels):
            active_cells = self.meshes[level].cells[self.aelem_level[level]]
            for cell in active_cells:
                x = cell[0, [0, 1, 1, 0, 0]]
                y = cell[1, [0, 0, 1, 1, 0]]
                ax.plot(x, y, color="black", linewidth=0.8)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if return_fig:
            return fig
        plt.show()
