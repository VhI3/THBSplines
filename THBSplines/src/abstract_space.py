"""
Abstract base class for all spline space objects.

A *space* associates basis functions with mesh cells.  The two key maps are:

  cell → basis : which basis functions are supported over a given cell?
  basis → cell : which cells lie inside the support of a given basis function?

Concrete subclasses are:

  - ``TensorProductSpace``  : full tensor-product B-spline space on a Cartesian mesh
  - ``HierarchicalSpace``   : truncated hierarchical space over a hierarchical mesh
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class Space(ABC):
    """Abstract interface that every spline space must satisfy."""

    @abstractmethod
    def cell_to_basis(
        self, cell_indices: Union[np.ndarray, List[int]]
    ) -> np.ndarray:
        """
        For each cell in ``cell_indices``, return the indices of all basis
        functions whose support contains that cell.

        Parameters
        ----------
        cell_indices : 1-D integer array of length N

        Returns
        -------
        np.ndarray of dtype object, shape (N,)
            The i-th entry is itself a 1-D integer array listing the basis
            functions supported over cell i.
        """

    @abstractmethod
    def basis_to_cell(
        self, basis_indices: Union[np.ndarray, List[int]]
    ) -> np.ndarray:
        """
        For each basis function in ``basis_indices``, return the indices of all
        cells contained in its support.

        Parameters
        ----------
        basis_indices : 1-D integer array of length N

        Returns
        -------
        np.ndarray of dtype object, shape (N,)
            The i-th entry is itself a 1-D integer array listing the cells
            inside the support of basis function i.
        """
