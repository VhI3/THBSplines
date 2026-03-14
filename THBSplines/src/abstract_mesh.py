"""
Abstract base class for all mesh objects.

A *mesh* in this package partitions the parametric domain into non-overlapping
axis-aligned cells (elements).  Concrete subclasses are:

  - ``CartesianMesh``    : a single-level regular grid
  - ``HierarchicalMesh`` : a multi-level locally refined grid
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Mesh(ABC):
    """Abstract interface that every mesh class must satisfy."""

    @abstractmethod
    def plot_cells(self) -> None:
        """
        Visualise the mesh cells.

        Only sensible for 1-D and 2-D parametric domains.
        Concrete implementations should use Matplotlib.
        """

    @abstractmethod
    def get_gauss_points(
        self, cell_indices: np.ndarray, order: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss–Legendre quadrature points and weights for a list of cells.

        Points from the reference interval [-1, 1] are mapped affinely to
        each physical cell.

        Parameters
        ----------
        cell_indices : 1-D integer array of cell indices to integrate over
        order        : number of Gauss points per direction per cell

        Returns
        -------
        points  : array of shape (N_cells * order^d, d)
        weights : array of shape (N_cells * order^d,)
        """
