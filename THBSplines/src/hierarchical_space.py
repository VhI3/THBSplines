"""
Truncated Hierarchical B-spline (THB-spline) space.

Overview
--------
A ``HierarchicalSpace`` organises basis functions from a *sequence* of nested
tensor-product spaces  V₀ ⊂ V₁ ⊂ … ⊂ V_L  into a single hierarchical basis.

At each level ℓ, every basis function is classified as:
  - *active*      (``afunc_level[ℓ]``) : contributes to the current hierarchical basis
  - *deactivated* (``dfunc_level[ℓ]``) : was active but has been replaced by finer functions

The *truncation* mechanism ensures that the resulting set of active functions:
  1. Has compact support (each function vanishes outside a region of the domain).
  2. Forms a partition of unity.
  3. Is linearly independent.

This is achieved by "zeroing out" any coarse-level contribution that overlaps
with the support of a finer-level active function (see ``get_basis_conversion_matrix``
and the ``truncated`` flag).

Refinement workflow
-------------------
1. Mark cells for refinement: ``marked_cells = {0: [3, 4, 5]}``.
2. Call ``refine(hspace, marked_cells)`` (in ``refinement.py``).
3. Internally:
   a. ``HierarchicalMesh.refine`` deactivates cells and activates their children.
   b. ``functions_to_deactivate_from_cells`` finds functions whose support no
      longer contains any active cell.
   c. ``HierarchicalSpace.refine`` updates active/deactivated function sets and
      promotes children to the next level.

References
----------
Giannelli, C., Jüttler, B., & Speleers, H. (2012).
  *THB-splines: The truncated basis for hierarchical splines.*
  Computer Aided Geometric Design, 29(7), 485–498.

Buffa, A., Giannelli, C., Morgenstern, P., & Peterseim, D. (2016).
  *Complexity of hierarchical refinement for a class of admissible mesh
  configurations.* Computer Aided Geometric Design, 47, 83–92.
"""

from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional, Union

import numpy as np
import scipy.sparse as sp

from THBSplines.src.abstract_space import Space
from THBSplines.src.hierarchical_mesh import HierarchicalMesh
from THBSplines.src.tensor_product_space import TensorProductSpace, TensorProductSpace2D


class HierarchicalSpace(Space):
    """
    Truncated Hierarchical B-spline space.

    Parameters
    ----------
    knots   : list of 1-D knot vectors for the coarsest level
    degrees : list of polynomial degrees, one per direction
    dim     : parametric dimension (1 or 2)
    """

    def __init__(self, knots: list, degrees: list, dim: int):
        self.nlevels = 1
        self.degrees = list(degrees)
        self.dim     = dim

        # Choose the specialised 2-D class when possible (faster basis construction)
        SpaceClass = TensorProductSpace2D if dim == 2 else TensorProductSpace
        self.spaces: list[TensorProductSpace] = [SpaceClass(knots, degrees, dim)]

        self.mesh = HierarchicalMesh(knots, dim)

        n0 = self.spaces[0].nfuncs
        # Initially all functions at level 0 are active
        self.afunc_level: dict[int, np.ndarray] = {
            0: np.arange(n0, dtype=np.int64)
        }
        # No functions deactivated yet
        self.dfunc_level: dict[int, np.ndarray] = {
            0: np.array([], dtype=np.int64)
        }
        self.nfuncs_level: dict[int, int] = {0: n0}
        self.nfuncs: int = n0

        # 1-D projection matrices (one list per level transition)
        # projections_onedim[ℓ] is a list of d sparse matrices
        self.projections_onedim: list[list] = []

        # Global active/deactivated function arrays (union over all levels)
        self.afunc = np.array([], dtype=np.int64)
        self.dfunc = np.array([], dtype=np.int64)

        # When True, apply the truncation that gives THB-splines their name.
        # Setting this to False gives plain hierarchical B-splines (HB-splines),
        # which are not a partition of unity.
        self.truncated: bool = True

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def cell_to_basis(self, cell_indices):
        # Delegates to the per-level spaces; here we just satisfy the ABC.
        raise NotImplementedError(
            "Use HierarchicalSpace.spaces[level].cell_to_basis instead."
        )

    def basis_to_cell(self, basis_indices):
        raise NotImplementedError(
            "Use HierarchicalSpace.spaces[level].basis_to_cell instead."
        )

    # ------------------------------------------------------------------
    # Level management
    # ------------------------------------------------------------------

    def add_level(self) -> None:
        """
        Append a new refinement level.

        Refines the finest existing tensor-product space dyadically and
        stores the resulting 1-D projection matrices (used in
        ``get_basis_conversion_matrix``).
        """
        expected_level = self.mesh.nlevels - 1
        if len(self.spaces) != expected_level:
            raise ValueError(
                f"Space has {len(self.spaces)} levels but mesh has {self.mesh.nlevels}. "
                "These must differ by exactly 1 before calling add_level()."
            )

        refined_space, projector_onedim = self.spaces[-1].refine()
        self.spaces.append(refined_space)
        self.projections_onedim.append(projector_onedim)
        self.nlevels += 1

        lev = self.mesh.nlevels - 1
        self.afunc_level[lev] = np.array([], dtype=np.int64)
        self.dfunc_level[lev] = np.array([], dtype=np.int64)
        self.nfuncs_level[lev] = 0

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def refine(
        self,
        marked_functions: dict[int, np.ndarray],
        new_cells: dict[int, np.ndarray],
    ) -> None:
        """
        Update the hierarchical space after mesh refinement.

        This method is called *after* ``HierarchicalMesh.refine`` has already
        deactivated mesh cells.  It deactivates the corresponding basis functions
        and activates their children on the next level.

        Parameters
        ----------
        marked_functions : level → indices of functions to deactivate
        new_cells        : level → newly activated cell indices (from mesh.refine)
        """
        # Make sure we have a space for the new mesh level
        if len(self.spaces) < self.mesh.nlevels:
            self.add_level()
            marked_functions[self.mesh.nlevels - 1] = np.array([], dtype=np.int64)

        self._update_active_functions(marked_functions, new_cells)

    # ------------------------------------------------------------------
    # Projection and conversion matrices
    # ------------------------------------------------------------------

    def compute_full_projection_matrix(self, level: int) -> sp.lil_matrix:
        """
        Build the full (coarse → fine) projection matrix at ``level`` via
        the Kronecker product of the 1-D projection matrices.

        The resulting matrix ``C`` satisfies:

            B_coarse_j(x) = Σᵢ C[i, j] · B_fine_i(x)

        for all coarse basis functions j.

        Parameters
        ----------
        level : index of the coarse level (the fine level is ``level + 1``)

        Returns
        -------
        scipy.sparse.lil_matrix of shape (nfuncs_fine, nfuncs_coarse)
        """
        C = 1
        for d in range(self.dim):
            C = sp.kron(self.projections_onedim[level][d], C)
        return C.tolil()

    def get_basis_conversion_matrix(
        self,
        level: int,
        coarse_indices: Optional[np.ndarray] = None,
    ) -> sp.lil_matrix:
        """
        Build the (possibly truncated) conversion matrix from level ``level``
        to level ``level + 1``.

        If ``truncated=True``, all rows corresponding to active or deactivated
        functions at level ``level + 1`` are zeroed.  This is the core of the
        THB-spline truncation: coarse contributions are removed wherever finer
        basis functions are already active.

        Parameters
        ----------
        level          : coarse level index
        coarse_indices : if provided, only compute columns for these coarse
                         functions (saves time when only a subset is needed).

        Returns
        -------
        scipy.sparse.lil_matrix of shape (nfuncs_{level+1}, nfuncs_{level})
        """
        if coarse_indices is None:
            C = self.compute_full_projection_matrix(level)
        else:
            # Efficient partial construction: only columns in coarse_indices
            nfine   = self.spaces[level + 1].nfuncs
            ncoarse = len(coarse_indices)

            # Pre-allocate COO arrays.  Each coarse function contributes at most
            # ∏(degree_j + 2) non-zero fine functions.
            max_nnz = int(np.prod(self.spaces[level + 1].degrees + 2)) * ncoarse
            rows    = np.zeros(max_nnz, dtype=np.int64)
            cols    = np.zeros(max_nnz, dtype=np.int64)
            vals    = np.zeros(max_nnz, dtype=np.float64)
            ptr     = 0

            # Convert flat indices to per-direction indices.
            # np.unravel_index uses C order (last index changes fastest), which
            # matches the tensor-product ordering used in TensorProductSpace2D
            # (inner loop over u, outer over v).  For dim == 2 the axes returned
            # are (v, u), so we swap them to (u, v) to match our storage.
            sub = np.unravel_index(coarse_indices, self.spaces[level].nfuncs_onedim)
            if self.dim == 2:
                sub = [sub[1], sub[0]]  # swap to (u, v) order

            for col_idx, fi in enumerate(range(len(coarse_indices))):
                # Build column as a Kronecker product of 1-D projections
                col = 1
                for d in range(self.dim):
                    col = sp.kron(
                        self.projections_onedim[level][d][:, sub[d][fi]], col
                    )
                ir, _, iv = sp.find(col)
                end = ptr + len(ir)
                rows[ptr:end] = ir
                cols[ptr:end] = col_idx
                vals[ptr:end] = iv
                ptr = end

            C = sp.coo_matrix(
                (vals[:ptr], (rows[:ptr], cols[:ptr])),
                shape=(nfine, self.spaces[level].nfuncs),
            ).tolil()

        if self.truncated:
            # Zero out rows that correspond to functions already handled at level+1.
            # This is the truncation: coarse contributions vanish where fine functions live.
            already_handled = np.union1d(
                self.afunc_level[level + 1], self.dfunc_level[level + 1]
            )
            C[already_handled, :] = 0

        return C

    # ------------------------------------------------------------------
    # Active function updates
    # ------------------------------------------------------------------

    def _update_active_functions(
        self,
        marked_entities: dict[int, np.ndarray],
        new_cells: dict[int, np.ndarray],
    ) -> None:
        """
        Update active and deactivated function sets after mesh refinement.

        Algorithm (per level ℓ):
        1. Remove ``marked_entities[ℓ]`` from ``afunc_level[ℓ]``.
        2. Add them to ``dfunc_level[ℓ]``.
        3. Compute the children of the deactivated functions at level ℓ+1.
        4. Activate children that are not already active or deactivated.
        5. Activate any further functions at level ℓ+1 whose entire support is
           covered by the union of active and deactivated cells at that level.

        Step 5 ensures that the hierarchical basis remains *complete*: whenever
        a fine cell becomes active, all basis functions supported there that
        are not yet handled at any finer level must be activated.

        Parameters
        ----------
        marked_entities : level → function indices to deactivate
        new_cells       : level → newly active cell indices (from mesh.refine)
        """
        afunc = self.afunc_level
        dfunc = self.dfunc_level

        for level in range(self.nlevels - 1):
            marked_at_level = marked_entities.get(level, np.array([], dtype=np.int64))

            # --- Step 1 & 2: deactivate marked functions ---
            afunc[level] = np.setdiff1d(afunc[level], marked_at_level).astype(np.int64)
            dfunc[level] = np.union1d(marked_at_level, dfunc[level]).astype(np.int64)

            # --- Step 3: find children (fine functions that overlap the coarse support) ---
            children = self._get_children(level, marked_at_level)

            # Activate children not yet active or deactivated at the next level
            already_present = np.union1d(afunc[level + 1], dfunc[level + 1])
            new_children    = np.setdiff1d(children, already_present)
            afunc[level + 1] = np.union1d(afunc[level + 1], new_children).astype(np.int64)

            # --- Step 5: activate additional functions at level+1 with full coverage ---
            # A function at level+1 can be activated if all cells in its support
            # are either active or deactivated (i.e. the support is "covered").
            new_at_next = new_cells.get(level + 1, np.array([], dtype=np.int64))
            candidate_funcs = self.spaces[level + 1].get_basis_functions(new_at_next)
            candidate_funcs = np.setdiff1d(candidate_funcs, afunc[level + 1])

            _, func_cells_map = self.spaces[level + 1].get_cells(candidate_funcs)
            covered_cells     = np.union1d(
                self.mesh.aelem_level[level + 1],
                self.mesh.delem_level[level + 1],
            )

            newly_active = np.array([
                fi for fi in candidate_funcs
                if np.all(np.isin(func_cells_map[fi], covered_cells))
            ], dtype=np.int64)
            afunc[level + 1] = np.union1d(afunc[level + 1], newly_active).astype(np.int64)

        # Compute global active / deactivated arrays (union over all levels)
        self.afunc = reduce(np.union1d, afunc.values())
        self.dfunc = reduce(np.union1d, dfunc.values())

        # Safety check: a function should never be simultaneously active and
        # deactivated.  This can happen in edge cases (e.g. a function that was
        # promoted to active at level ℓ+1 and also appears in dfunc[ℓ+1]).
        # We resolve it by giving priority to the deactivated set.
        for key in dfunc:
            afunc[key] = np.setdiff1d(afunc[key], dfunc[key]).astype(np.int64)

        # Recompute per-level counts and total DOF count
        self.nfuncs_level = {
            lev: len(self.afunc_level[lev])
            for lev in range(self.nlevels)
        }
        self.nfuncs = sum(self.nfuncs_level.values())

    def _get_children(
        self,
        level: int,
        marked_functions: np.ndarray,
    ) -> np.ndarray:
        """
        Return all fine-level (level+1) functions that are non-zero on the
        support of at least one function in ``marked_functions``.

        Uses the full projection matrix: a fine function is a "child" if
        it has a non-zero coefficient when the coarse function is expressed
        in the fine basis.

        Parameters
        ----------
        level            : coarse level index
        marked_functions : 1-D integer array of coarse function indices

        Returns
        -------
        np.ndarray of int64
        """
        children   = np.array([], dtype=np.int64)
        projection = self.compute_full_projection_matrix(level)

        for fi in marked_functions:
            # Non-zero rows of the fi-th column are the children
            col_nz = np.flatnonzero(projection[:, fi].toarray())
            children = np.union1d(children, col_nz)

        return children

    # ------------------------------------------------------------------
    # Cell / function marking utilities
    # ------------------------------------------------------------------

    def functions_to_deactivate_from_cells(
        self, marked_cells: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        """
        Identify which active basis functions should be deactivated given a
        set of marked cells.

        A basis function at level ℓ is deactivated when *all* mesh cells that
        overlap its support have been marked for refinement (i.e. the function
        has no surviving active cells).

        Parameters
        ----------
        marked_cells : level → 1-D array of cell indices

        Returns
        -------
        dict
            level → 1-D array of function indices to deactivate
        """
        marked_functions: dict[int, np.ndarray] = {}

        for level in range(self.nlevels):
            # Skip levels that have no marked cells
            if level not in marked_cells or len(marked_cells[level]) == 0:
                marked_functions[level] = np.array([], dtype=np.int64)
                continue

            # Functions in the support of any marked cell
            candidates = self.spaces[level].get_basis_functions(marked_cells[level])
            # Keep only currently active functions
            candidates = np.intersect1d(candidates, self.afunc_level[level])

            # A function should NOT be deactivated if it still has at least one
            # active cell in its support (outside the marked region)
            _, func_cells_map = self.spaces[level].get_cells(candidates)
            to_keep = np.array([
                fi for fi in candidates
                if np.intersect1d(func_cells_map[fi], self.mesh.aelem_level[level]).size > 0
            ], dtype=np.int64)

            marked_functions[level] = np.setdiff1d(candidates, to_keep).astype(np.int64)

        return marked_functions

    def refine_in_rectangle(
        self, rectangle: np.ndarray, level: int
    ) -> np.ndarray:
        """
        Return the indices of active cells at ``level`` that lie inside
        ``rectangle``.

        Useful for region-based refinement: mark all cells in a rectangular
        sub-domain for refinement.

        Parameters
        ----------
        rectangle : array of shape (d, 2), where rectangle[j] = [lo_j, hi_j]
        level     : mesh level to query

        Returns
        -------
        np.ndarray of int64
        """
        eps   = np.spacing(1)
        cells = self.mesh.meshes[level].cells  # (N, d, 2)

        # A cell is inside the rectangle when every edge falls within the box
        inside = np.all(
            (rectangle[:, 0] <= cells[:, :, 0] + eps) &
            (cells[:, :, 1]  <= rectangle[:, 1] + eps),
            axis=1,
        )
        return np.intersect1d(np.flatnonzero(inside), self.mesh.aelem_level[level])

    # ------------------------------------------------------------------
    # Subdivision matrix
    # ------------------------------------------------------------------

    def create_subdivision_matrix(self, mode: str = "reduced") -> dict[int, sp.spmatrix]:
        """
        Build the matrices that express each active THB-spline at level ℓ in
        terms of the finest-level (level L-1) B-splines.

        These matrices are used in the assembly routines to pull back the
        fine-level local mass/stiffness contributions to the hierarchical DOF
        ordering.

        Two modes are available:

        ``'full'``
            Uses the *full* projection matrix at each level (no column
            reduction).  The result is a complete change-of-basis.

        ``'reduced'``
            Only includes columns corresponding to functions on the
            "deactivation front" (functions on deactivated elements).
            This is more efficient when many functions are far from the
            refinement zone.

        Parameters
        ----------
        mode : ``'reduced'`` (default) or ``'full'``

        Returns
        -------
        dict
            ``C[ℓ]`` is a sparse matrix of shape
            (nfuncs_{level L}, nfuncs_active_up_to_ℓ).
        """
        mesh = self.mesh
        C: dict[int, sp.spmatrix] = {}

        # Level 0: identity restricted to active functions
        C[0] = sp.identity(self.spaces[0].nfuncs, format="lil")
        C[0] = C[0][:, self.afunc_level[0]]

        if mode == "reduced":
            # Track the set of "relevant" coarse functions (those on the
            # boundary between active and deactivated regions)
            relevant = self.spaces[0].get_basis_functions(mesh.aelem_level[0])
            relevant = np.union1d(
                relevant,
                self.spaces[0].get_basis_functions(mesh.delem_level[0]),
            )

            for level in range(1, self.nlevels):
                # Identity matrix selecting active functions at this level
                active_idx = self.afunc_level[level]
                nact       = self.nfuncs_level[level]
                data       = np.ones(nact)
                I = sp.coo_matrix(
                    (data, (active_idx, np.arange(nact))),
                    shape=(self.spaces[level].nfuncs, nact),
                )

                # Conversion matrix (with truncation applied)
                aux     = self.get_basis_conversion_matrix(level - 1, coarse_indices=relevant)
                C[level] = sp.hstack([aux @ C[level - 1], I])

                # Update the relevant set for the next level
                relevant = np.union1d(
                    self.spaces[level].get_basis_functions(mesh.aelem_level[level]),
                    self.spaces[level].get_basis_functions(mesh.delem_level[level]),
                )

        else:  # mode == 'full'
            for level in range(1, self.nlevels):
                I   = sp.identity(self.spaces[level].nfuncs, format="lil")
                aux = self.get_basis_conversion_matrix(level - 1)
                C[level] = sp.hstack([aux @ C[level - 1], I[:, self.afunc_level[level]]])

        return C

    # ------------------------------------------------------------------
    # Internal support utilities
    # ------------------------------------------------------------------

    def _get_all_cells(self) -> np.ndarray:
        """Return all active cells across all levels as a single array."""
        parts = [
            self.mesh.meshes[lev].cells[self.mesh.aelem_level[lev]]
            for lev in range(self.mesh.nlevels)
        ]
        return np.concatenate(parts, axis=0)

    def _get_truncated_supports(self) -> dict[int, np.ndarray]:
        """
        For each active THB-spline (indexed 0 … nfuncs-1), return the
        cells at the finest level that form its effective (truncated) support.
        """
        C = self.create_subdivision_matrix(mode="full")[self.nlevels - 1].toarray()
        result = {}
        for i, coeffs in enumerate(C.T):
            fine_funcs   = np.flatnonzero(coeffs)
            support_cells = self._get_fine_basis_support_cells(fine_funcs)
            result[i]    = self.mesh.meshes[-1].cells[support_cells]
        return result

    def _get_fine_basis_support_cells(self, fine_funcs: np.ndarray) -> np.ndarray:
        """
        Return the indices of finest-level cells that lie in the union of
        supports of ``fine_funcs``.
        """
        cells      = np.array([], dtype=np.int64)
        fine_cells = self.mesh.meshes[-1].cells  # (N, d, 2)

        for fi in fine_funcs:
            supp = self.spaces[-1].basis_supports[fi]  # (d, 2)
            in_support = np.all(
                (supp[:, 0] <= fine_cells[:, :, 0]) &
                (supp[:, 1] >= fine_cells[:, :, 1]),
                axis=1,
            )
            cells = np.union1d(cells, np.flatnonzero(in_support))

        return cells

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_overloading(
        self,
        ax=None,
        filename: Optional[str] = None,
        text: bool = False,
        fontsize: Optional[float] = None,
    ) -> None:
        """
        Visualise cells where more THB-spline functions are active than
        the theoretical maximum (``∏(degree_j + 1)``).

        In a conforming hierarchical basis, the number of active functions
        per cell should never exceed ``∏(degree_j + 1)``.  This method
        colours over-loaded cells and optionally annotates them with the
        excess count — useful for debugging refinement strategies.

        Parameters
        ----------
        ax       : optional Matplotlib Axes
        filename : save the figure to this path instead of displaying it
        text     : if True, annotate each over-loaded cell with the excess count
        fontsize : font size for annotations (relative to cell width)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if ax is None:
            _, ax = plt.subplots()

        C           = self.create_subdivision_matrix("full")
        max_per_cell = int(np.prod([d + 1 for d in self.degrees]))

        for level in range(self.nlevels):
            mesh_l  = self.mesh.meshes[level]
            Csub    = sp.lil_matrix(C[level])

            for cell_idx in self.mesh.aelem_level[level]:
                # Basis functions active on this cell (via the subdivision matrix)
                basis_on_cell = self.spaces[level].cell_to_basis([cell_idx])
                if len(basis_on_cell) == 0 or len(basis_on_cell[0]) == 0:
                    continue

                _, col, _ = sp.find(Csub[basis_on_cell[0], :])
                n_active  = len(np.unique(col))

                cell  = mesh_l.cells[cell_idx]
                w     = cell[0, 1] - cell[0, 0]
                h     = cell[1, 1] - cell[1, 0]

                overloaded = n_active > max_per_cell
                ax.add_patch(patches.Rectangle(
                    (cell[0, 0], cell[1, 0]), w, h,
                    fill=overloaded,
                    color="red" if overloaded else "black",
                    alpha=0.3 if overloaded else 1.0,
                    linewidth=0.5,
                ))

                if text and overloaded:
                    fs = fontsize if fontsize is not None else 8
                    ax.text(
                        cell[0, 0] + w / 2, cell[1, 0] + h / 2,
                        str(n_active - max_per_cell),
                        ha="center", va="center", fontsize=fs,
                    )

        # Set axis limits from mesh extents
        all_cells = self._get_all_cells()
        ax.set_xlim(all_cells[:, 0, 0].min(), all_cells[:, 0, 1].max())
        ax.set_ylim(all_cells[:, 1, 0].min(), all_cells[:, 1, 1].max())
        ax.set_aspect("equal")

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
