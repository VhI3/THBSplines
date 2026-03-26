"""
Evaluation utilities for LR B-spline spaces.

Functions
---------
evaluate_lr_basis(space, pts)
    Evaluate all active basis functions at an array of points.
    Returns an (N, nfuncs) array.

check_partition_of_unity(space, n_pts, tol)
    Verify sum_i B_i(x) == 1 on random interior points.

plot_basis_functions(space, n_pts, ...)
    Plot all active LR basis functions as filled contour plots.

plot_mesh_and_supports(space, ...)
    Overlay the T-mesh and the support boxes of all basis functions.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from LRSplines.src.lr_spline_space import LRSplineSpace


# ---------------------------------------------------------------------------
# Basis evaluation
# ---------------------------------------------------------------------------

def evaluate_lr_basis(space: 'LRSplineSpace',
                      pts: np.ndarray) -> np.ndarray:
    """
    Evaluate all active LR basis functions at parametric points.

    Parameters
    ----------
    space : LRSplineSpace
    pts   : ndarray, shape (N, 2)
        Each row is (u, v).

    Returns
    -------
    B : ndarray, shape (N, nfuncs)
        B[i, k] = B_k(u_i, v_i)
    """
    return space.evaluate(np.asarray(pts, dtype=float))


def evaluate_lr_grad(space: 'LRSplineSpace',
                     pts: np.ndarray) -> np.ndarray:
    """
    Evaluate gradients of all active LR basis functions.

    Parameters
    ----------
    space : LRSplineSpace
    pts   : ndarray, shape (N, 2)

    Returns
    -------
    G : ndarray, shape (N, nfuncs, 2)
    """
    return space.evaluate_grad(np.asarray(pts, dtype=float))


# ---------------------------------------------------------------------------
# Partition of unity
# ---------------------------------------------------------------------------

def check_partition_of_unity(space: 'LRSplineSpace',
                              n_pts: int = 50,
                              tol: float = 1e-10) -> bool:
    """
    Verify that sum_i B_i(u, v) = 1 at ``n_pts`` random interior points.

    Parameters
    ----------
    space : LRSplineSpace
    n_pts : int
    tol   : float  maximum allowed deviation from 1

    Returns
    -------
    bool  True if partition of unity holds within tolerance.
    """
    rng = np.random.default_rng(0)
    u0, u1 = space.mesh.u_domain
    v0, v1 = space.mesh.v_domain
    eps = 1e-6
    u_pts = rng.uniform(u0 + eps, u1 - eps, n_pts)
    v_pts = rng.uniform(v0 + eps, v1 - eps, n_pts)
    pts = np.column_stack([u_pts, v_pts])

    B = space.evaluate(pts)
    row_sums = B.sum(axis=1)
    max_err = float(np.max(np.abs(row_sums - 1.0)))
    return max_err < tol


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_basis_functions(space: 'LRSplineSpace',
                         n_pts: int = 40,
                         max_funcs: int = 16,
                         cmap: str = 'cividis',
                         ax_array=None,
                         title: Optional[str] = None):
    """
    Plot individual LR basis functions as filled contour plots.

    Parameters
    ----------
    space    : LRSplineSpace
    n_pts    : int  resolution per axis of the evaluation grid
    max_funcs: int  maximum number of functions to show
    cmap     : str  matplotlib colormap name
    ax_array : optional pre-created array of axes
    title    : str  figure suptitle

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes
    """
    u0, u1 = space.mesh.u_domain
    v0, v1 = space.mesh.v_domain

    u1d = np.linspace(u0, u1, n_pts)
    v1d = np.linspace(v0, v1, n_pts)
    U, V = np.meshgrid(u1d, v1d)
    pts = np.column_stack([U.ravel(), V.ravel()])

    n_show = min(space.nfuncs, max_funcs)
    ncols = min(n_show, 4)
    nrows = int(np.ceil(n_show / ncols))

    if ax_array is None:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.2 * ncols, 3.0 * nrows),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
    else:
        axes = np.atleast_2d(ax_array)
        fig = axes.flat[0].get_figure()

    for k, ax in enumerate(axes.flat):
        if k >= n_show:
            ax.axis('off')
            continue
        Z = space.basis[k].eval_array(pts).reshape(n_pts, n_pts)
        ax.contourf(U, V, Z, levels=14, cmap=cmap)
        ax.set_title(fr'$B_{{{k}}}$', fontsize=9, pad=2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=13)

    return fig, axes


def plot_mesh_and_supports(space: 'LRSplineSpace',
                           show_supports: bool = True,
                           ax=None):
    """
    Draw the T-mesh and (optionally) the support boxes of every basis
    function.

    Parameters
    ----------
    space           : LRSplineSpace
    show_supports   : bool  if True, draw support bounding boxes
    ax              : matplotlib Axes, optional

    Returns
    -------
    fig, ax  (only when ax was None)
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)

    # Support boxes
    if show_supports:
        palette = plt.cm.tab20.colors
        for k, B in enumerate(space.basis):
            u0s, u1s, v0s, v1s = B.support
            color = palette[k % len(palette)]
            rect = patches.Rectangle(
                (u0s, v0s), u1s - u0s, v1s - v0s,
                linewidth=1.2, edgecolor=color,
                facecolor=color, alpha=0.08)
            ax.add_patch(rect)

    # Mesh lines
    for ln in space.mesh.lines:
        lw = 0.8 + 0.5 * (ln.multiplicity - 1)
        col = '#1d3557' if ln.multiplicity == 1 else '#c1121f'
        if ln.axis == 0:
            ax.plot([ln.value, ln.value], [ln.start, ln.end],
                    color=col, lw=lw)
        else:
            ax.plot([ln.start, ln.end], [ln.value, ln.value],
                    color=col, lw=lw)

    u0, u1 = space.mesh.u_domain
    v0, v1 = space.mesh.v_domain
    ax.set_xlim(u0 - 0.05, u1 + 0.05)
    ax.set_ylim(v0 - 0.05, v1 + 0.05)
    ax.set_aspect('equal')
    ax.set_xlabel('$u$')
    ax.set_ylabel('$v$')
    ax.set_title(f'LR mesh  ({len(space.mesh.lines)} lines, '
                 f'{space.mesh.nelements} elements, '
                 f'{space.nfuncs} basis functions)')

    if created:
        return fig, ax


def plot_partition_of_unity(space: 'LRSplineSpace',
                            n_pts: int = 60):
    """
    Plot the sum of all LR basis functions and the pointwise deviation
    from 1 (should be machine-precision small).

    Returns
    -------
    fig : matplotlib Figure
    """
    u0, u1 = space.mesh.u_domain
    v0, v1 = space.mesh.v_domain
    u1d = np.linspace(u0, u1, n_pts)
    v1d = np.linspace(v0, v1, n_pts)
    U, V = np.meshgrid(u1d, v1d)
    pts = np.column_stack([U.ravel(), V.ravel()])

    B = space.evaluate(pts)
    S = B.sum(axis=1).reshape(n_pts, n_pts)
    err = np.abs(S - 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    cf0 = axes[0].contourf(U, V, S, levels=18, cmap='RdBu_r',
                            vmin=0.98, vmax=1.02)
    fig.colorbar(cf0, ax=axes[0], label=r'$\sum_i B_i$')
    axes[0].set_title('Sum of LR Basis Functions')
    axes[0].set_aspect('equal')

    cf1 = axes[1].contourf(U, V, err, levels=18, cmap='magma')
    fig.colorbar(cf1, ax=axes[1], label=r'$|\sum_i B_i - 1|$')
    axes[1].set_title('Pointwise Deviation from 1')
    axes[1].set_aspect('equal')

    for ax in axes:
        ax.set_xlabel('$u$')
        ax.set_ylabel('$v$')

    return fig
