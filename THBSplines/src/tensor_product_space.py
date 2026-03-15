"""
Tensor-product B-spline space over a Cartesian mesh.

A ``TensorProductSpace`` in ``d`` dimensions is spanned by all possible
tensor products of univariate B-spline basis functions:

    {B_{i₁,p₁} ⊗ B_{i₂,p₂} ⊗ … ⊗ B_{i_d,p_d}}

where each ``B_{i_j,p_j}`` is defined over a *local* knot vector of
length ``p_j + 2`` extracted from the global knot vector in direction j.

The total number of basis functions is ∏_j (n_j - p_j - 1), where n_j is
the length of the knot vector in direction j.

Key responsibilities
--------------------
- ``construct_basis``        : build the support array and end-evaluation flags
- ``cell_to_basis``          : find which basis functions cover a given cell
- ``basis_to_cell``          : find which cells lie in a given support
- ``get_basis_functions``    : like cell_to_basis but for a list of cells → union
- ``get_cells``              : like basis_to_cell but returns a dict too
- ``refine``                 : dyadic refinement + knot-insertion matrix
- ``compute_projection_matrix``: Boehm's algorithm (static, used in refinement)
- ``construct_B_spline``     : return a callable TensorProductBSpline

Note: for 2-D domains, ``TensorProductSpace2D`` is used instead.  It has
a more efficient basis construction but the same public interface.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import scipy.sparse as sp

from THBSplines.src.abstract_space import Space
from THBSplines.src.b_spline import augment_knots, find_knot_index
from THBSplines.src.b_spline_numpy import TensorProductBSpline
from THBSplines.src.cartesian_mesh import CartesianMesh


# ---------------------------------------------------------------------------
# Helper: dyadic knot refinement
# ---------------------------------------------------------------------------

def insert_midpoints(knots: np.ndarray, p: int) -> np.ndarray:
    """
    Insert the midpoint of every interior knot interval in a p+1-regular
    knot vector (i.e. a knot vector with p+1 identical values at each end).

    This is the dyadic refinement step: the resulting knot vector has
    approximately twice as many unique knot values.

    Parameters
    ----------
    knots : 1-D array, p+1-regular knot vector
    p     : polynomial degree (used only to clarify the regularity assumption)

    Returns
    -------
    np.ndarray
        Sorted knot vector with midpoints inserted.
    """
    knots        = np.asarray(knots, dtype=np.float64)
    unique_knots = np.unique(knots)
    midpoints    = (unique_knots[:-1] + unique_knots[1:]) / 2.0
    return np.sort(np.concatenate((knots, midpoints)))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TensorProductSpace(Space):
    """
    Full tensor-product B-spline space over a Cartesian mesh.

    Parameters
    ----------
    knots   : list of 1-D knot vectors, one per parametric direction
    degrees : list of polynomial degrees, one per direction
    dim     : parametric dimension (must equal ``len(knots)`` and ``len(degrees)``)
    """

    def __init__(self, knots: list, degrees: list, dim: int):
        self.knots   = np.array(knots,   dtype=np.float64)
        self.degrees = np.array(degrees, dtype=np.int32)
        self.dim     = dim
        self.mesh    = CartesianMesh(knots, dim)

        # Populated by construct_basis()
        self.basis_supports:   np.ndarray | None = None  # (nfuncs, d, 2)
        self.basis:            list | None        = None
        self.basis_end_evals:  list | None        = None

        self._construct_basis()
        self.nfuncs        = len(self.basis)
        # Number of basis functions per direction: n_j = len(knots_j) - degree_j - 1
        self.nfuncs_onedim = [
            len(k) - int(d) - 1 for k, d in zip(self.knots, self.degrees)
        ]
        self.cell_area = self.mesh.cell_area

    # ------------------------------------------------------------------
    # Basis construction
    # ------------------------------------------------------------------

    def _construct_basis(self) -> None:
        """
        Build the list of all basis functions for the tensor-product space.

        For each tensor-product combination of univariate B-spline indices,
        we store:
          - the local knot vector (one per direction) → ``self.basis``
          - the support bounding box                  → ``self.basis_supports``
          - the end-evaluation flags                  → ``self.basis_end_evals``

        The end-evaluation flag for direction j is True if this basis function
        is the *last* B-spline in that direction (so the right endpoint of the
        global knot vector must be included in its support).
        """
        dim     = self.dim
        degrees = self.degrees

        # Index ranges per direction: start[j] = [0, 1, …, n_j-1]
        idx_start = [list(range(len(self.knots[j]) - degrees[j] - 1)) for j in range(dim)]
        idx_stop  = [[s + degrees[j] + 2 for s in idx_start[j]] for j in range(dim)]

        # Tensor product of all (start, stop) index pairs
        start_grid = np.stack(np.meshgrid(*idx_start), -1).reshape(-1, dim)
        stop_grid  = np.stack(np.meshgrid(*idx_stop),  -1).reshape(-1, dim)

        n          = len(start_grid)
        b_splines  = []
        b_support  = np.zeros((n, dim, 2), dtype=np.float64)
        end_evals  = []

        for i in range(n):
            local_knots = []
            end_flags   = []
            for j in range(dim):
                kv = self.knots[j][start_grid[i, j]: stop_grid[i, j]]
                local_knots.append(kv)
                # End flag: True when this B-spline reaches the last knot
                end_flags.append(int(stop_grid[i, j] == len(self.knots[j])))

            local_knots = np.array(local_knots, dtype=np.float64)
            b_splines.append(local_knots)
            b_support[i]  = [[lk[0], lk[-1]] for lk in local_knots]
            end_evals.append(np.array(end_flags, dtype=np.int32).ravel())

        self.basis           = np.array(b_splines)
        self.basis_supports  = b_support
        self.basis_end_evals = end_evals
        self.nfuncs          = n

    # keep old name callable for backward compatibility
    def construct_basis(self) -> None:
        self._construct_basis()

    # ------------------------------------------------------------------
    # Support queries
    # ------------------------------------------------------------------

    def cell_to_basis(
        self, cell_indices: Union[np.ndarray, List[int]]
    ) -> np.ndarray:
        """
        For each cell index, return an array of basis-function indices whose
        support contains that cell.

        A basis function B covers cell Q when:
            B.support[j, 0] ≤ Q[j, 0]  and  B.support[j, 1] ≥ Q[j, 1]
        for all directions j.

        Parameters
        ----------
        cell_indices : 1-D integer array

        Returns
        -------
        np.ndarray of dtype object, shape (N,)
        """
        result = []
        for ci in cell_indices:
            cell = self.mesh.cells[ci]  # (d, 2)
            # Vectorised check over all basis functions simultaneously
            covered = np.all(
                (self.basis_supports[:, :, 0] <= cell[:, 0]) &
                (self.basis_supports[:, :, 1] >= cell[:, 1]),
                axis=1,
            )
            result.append(np.flatnonzero(covered))
        return np.array(result, dtype=object)

    def basis_to_cell(
        self, basis_indices: Union[np.ndarray, List[int]]
    ) -> np.ndarray:
        """
        For each basis-function index, return an array of cell indices that
        lie inside the support of that basis function.

        Parameters
        ----------
        basis_indices : 1-D integer array

        Returns
        -------
        np.ndarray of dtype object, shape (N,)
        """
        result = []
        for bi in basis_indices:
            supp = self.basis_supports[bi]  # (d, 2)
            inside = np.all(
                (supp[:, 0] <= self.mesh.cells[:, :, 0]) &
                (supp[:, 1] >= self.mesh.cells[:, :, 1]),
                axis=1,
            )
            result.append(np.flatnonzero(inside))
        return np.array(result, dtype=object)

    def get_basis_functions(self, cell_list: np.ndarray) -> np.ndarray:
        """
        Return the union of all basis-function indices supported over any
        cell in ``cell_list``.

        Parameters
        ----------
        cell_list : 1-D integer array of cell indices

        Returns
        -------
        np.ndarray of int64
        """
        basis = np.array([], dtype=np.int64)
        for ci in cell_list:
            cell = self.mesh.cells[ci]
            covered = np.all(
                (self.basis_supports[:, :, 0] <= cell[:, 0]) &
                (self.basis_supports[:, :, 1] >= cell[:, 1]),
                axis=1,
            )
            basis = np.union1d(basis, np.flatnonzero(covered))
        return basis

    def get_cells(
        self, basis_function_list: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Return the union of support cells for a list of basis functions,
        along with a per-function mapping.

        Parameters
        ----------
        basis_function_list : 1-D integer array

        Returns
        -------
        cells     : sorted union of all cell indices
        cells_map : dict  basis_index → 1-D array of cell indices
        """
        cells     = np.array([], dtype=np.int64)
        cells_map = {}
        eps       = np.spacing(1)

        for fi in basis_function_list:
            supp = self.basis_supports[fi]  # (d, 2)
            inside = np.all(
                (self.mesh.cells[:, :, 0] + eps >= supp[:, 0]) &
                (self.mesh.cells[:, :, 1]        <= supp[:, 1] + eps),
                axis=1,
            )
            idx            = np.flatnonzero(inside)
            cells          = np.union1d(cells, idx)
            cells_map[fi]  = idx

        return cells, cells_map

    def get_functions_on_rectangle(self, cell: np.ndarray) -> np.ndarray:
        """
        Return the indices of all basis functions whose support contains ``cell``.

        This is equivalent to ``cell_to_basis`` for a single cell, but
        returns a flat array rather than a length-1 object array.

        Parameters
        ----------
        cell : array of shape (d, 2)

        Returns
        -------
        np.ndarray of int64
        """
        covered = np.all(
            (self.basis_supports[:, :, 0] <= cell[:, 0]) &
            (self.basis_supports[:, :, 1] >= cell[:, 1]),
            axis=1,
        )
        return np.flatnonzero(covered)

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def refine(self) -> Tuple["TensorProductSpace", list]:
        """
        Dyadically refine the space and compute the knot-insertion matrix.

        Returns
        -------
        fine_space       : the refined TensorProductSpace
        projection_onedim: list of d sparse matrices (one per direction)
                           that express coarse basis functions in terms of
                           fine basis functions via Boehm's algorithm.
        """
        fine_knots        = [insert_midpoints(kv, int(p)) for kv, p in zip(self.knots, self.degrees)]
        projection_onedim = self.compute_projection_matrix(self.knots, fine_knots, self.degrees)
        fine_space        = TensorProductSpace(fine_knots, self.degrees, self.dim)
        return fine_space, projection_onedim

    @staticmethod
    def compute_projection_matrix(
        coarse_knots: np.ndarray,
        fine_knots:   list,
        degrees:      np.ndarray,
    ) -> list:
        """
        Compute the 1-D knot-insertion matrices (one per direction) using
        Boehm's algorithm.

        Boehm's algorithm expresses each coarse B-spline as a linear
        combination of fine B-splines.  The result is a sparse matrix
        ``A`` where ``A[i, j]`` is the coefficient of fine basis function i
        in the expansion of coarse basis function j.

        Parameters
        ----------
        coarse_knots : array of shape (d, n_coarse_j)
        fine_knots   : list of 1-D arrays, one per direction
        degrees      : array of polynomial degrees, shape (d,)

        Returns
        -------
        list of d scipy.sparse.lil_matrix objects
        """
        matrices = []

        for fine_kv, coarse_kv, degree in zip(fine_knots, coarse_knots, degrees):
            degree = int(degree)
            # Augment both knot vectors with sentinel values
            t_coarse = augment_knots(coarse_kv, degree)
            t_fine   = augment_knots(fine_kv,   degree)

            m = len(t_fine)   - (degree + 1)   # number of fine B-splines + padding
            n = len(t_coarse) - (degree + 1)   # number of coarse B-splines + padding

            A = sp.lil_matrix((m, n), dtype=np.float64)
            t_fine   = np.asarray(t_fine,   dtype=np.float64)
            t_coarse = np.asarray(t_coarse, dtype=np.float64)

            for i in range(m):
                # mu = span index such that t_coarse[mu] ≤ t_fine[i] < t_coarse[mu+1]
                mu = find_knot_index(t_fine[i], t_coarse)

                # Initialise the Oslo algorithm coefficient vector
                b = np.array([1.0])
                for k in range(1, degree + 1):
                    tau1  = t_coarse[mu - k + 1: mu + 1]
                    tau2  = t_coarse[mu + 1:     mu + k + 1]
                    omega = (t_fine[i + k] - tau1) / (tau2 - tau1)
                    b     = np.append((1.0 - omega) * b, 0.0) + np.insert(omega * b, 0, 0.0)

                A[i, mu - degree: mu + 1] = b

            # Strip padding rows and columns introduced by augment_knots
            matrices.append(A[degree + 1: -degree - 1, degree + 1: -degree - 1])

        return matrices

    # ------------------------------------------------------------------
    # Callable basis functions
    # ------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def construct_B_spline(self, i: int) -> TensorProductBSpline:
        """
        Return a callable ``TensorProductBSpline`` for basis function ``i``.

        The result is cached so repeated calls with the same index are free.

        Parameters
        ----------
        i : global basis-function index

        Returns
        -------
        TensorProductBSpline
        """
        return TensorProductBSpline(
            self.degrees,
            self.basis[i],
            self.basis_end_evals[i],
        )

    def construct_function(self, coefficients: np.ndarray):
        """
        Build a callable that evaluates the linear combination:

            f(x) = Σᵢ cᵢ Bᵢ(x)

        Parameters
        ----------
        coefficients : 1-D array of length ``nfuncs``

        Returns
        -------
        Callable[[np.ndarray], float]
        """
        assert len(coefficients) == self.nfuncs, (
            f"Expected {self.nfuncs} coefficients, got {len(coefficients)}"
        )
        def f(x: np.ndarray):
            x = np.asarray(x, dtype=np.float64)
            pointwise_input = x.ndim == 1
            x_eval = x.reshape(1, -1) if pointwise_input else x

            values = np.zeros(x_eval.shape[0], dtype=np.float64)
            for i, c in enumerate(coefficients):
                if c != 0:
                    values += c * self.construct_B_spline(i)(x_eval)

            if pointwise_input:
                return float(values[0])
            return values
        return f


# ---------------------------------------------------------------------------
# 2-D optimised subclass
# ---------------------------------------------------------------------------

class TensorProductSpace2D(TensorProductSpace):
    """
    Optimised tensor-product B-spline space for the 2-D case.

    The basis construction uses direct index arithmetic instead of the
    general meshgrid approach, which is slightly faster for large 2-D grids.

    The public interface is identical to ``TensorProductSpace``.
    """

    def _construct_basis(self) -> None:
        """
        Build basis supports and end-evaluation flags for the 2-D case.

        Basis functions are indexed as B_{i,j} → global index = j*n + i
        where n = nfuncs in u-direction and m = nfuncs in v-direction.
        """
        kv_u   = self.knots[0]
        kv_v   = self.knots[1]
        deg_u  = int(self.degrees[0])
        deg_v  = int(self.degrees[1])

        n = len(kv_u) - deg_u - 1   # # of B-splines in u
        m = len(kv_v) - deg_v - 1   # # of B-splines in v

        b_support        = np.zeros((n * m, 2, 2), dtype=np.float64)
        b_splines_end    = np.zeros((n * m, 2),    dtype=np.int32)

        idx = 0
        for j in range(m):            # v-direction
            for i in range(n):        # u-direction
                # Support in u: [kv_u[i], kv_u[i+deg_u+1]]
                # Support in v: [kv_v[j], kv_v[j+deg_v+1]]
                b_support[idx] = [
                    [kv_u[i], kv_u[i + deg_u + 1]],
                    [kv_v[j], kv_v[j + deg_v + 1]],
                ]
                # End flag is True when we reach the last knot in that direction
                b_splines_end[idx] = [
                    int(len(kv_u) == i + deg_u + 2),
                    int(len(kv_v) == j + deg_v + 2),
                ]
                idx += 1

        self.basis_supports  = b_support
        self.basis_end_evals = b_splines_end
        self.nfuncs          = n * m
        self.dim_u           = n
        self.dim_v           = m
        self.nfuncs_onedim   = [n, m]
        # basis stores only a placeholder; local knots are rebuilt in construct_B_spline
        self.basis           = [None] * (n * m)

    def refine(self) -> Tuple["TensorProductSpace2D", list]:
        fine_knots        = [insert_midpoints(kv, int(p)) for kv, p in zip(self.knots, self.degrees)]
        projection_onedim = self.compute_projection_matrix(self.knots, fine_knots, self.degrees)
        fine_space        = TensorProductSpace2D(fine_knots, self.degrees, self.dim)
        return fine_space, projection_onedim

    @lru_cache(maxsize=None)
    def construct_B_spline(self, i: int) -> TensorProductBSpline:
        """
        Return a callable for basis function i, reconstructing local knots
        on-the-fly from the global knot vectors.

        Parameters
        ----------
        i : global index (row-major: i = ind_v * dim_u + ind_u)
        """
        ind_v  = i // self.dim_u
        ind_u  = i %  self.dim_u
        deg_u  = int(self.degrees[0])
        deg_v  = int(self.degrees[1])

        local_knots = np.array([
            self.knots[0][ind_u: ind_u + deg_u + 2],
            self.knots[1][ind_v: ind_v + deg_v + 2],
        ], dtype=np.float64)

        return TensorProductBSpline(
            self.degrees,
            local_knots,
            self.basis_end_evals[i],
        )
