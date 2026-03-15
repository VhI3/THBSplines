"""
generate_images.py
==================
Generates the key figures used in README.md and saves them to
THBSplines/images/.

Run with:
    .venv-thbsplines/bin/python generate_images.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import pathlib

# ── project root ──────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).resolve().parent
IMG_DIR  = ROOT / "THBSplines" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfbf8",
    "axes.edgecolor": "#262626",
    "axes.linewidth": 0.9,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.color": "#d9d4cb",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.45,
    "grid.linestyle": "--",
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

LINE  = ["#0f4c5c", "#e36414", "#6a994e", "#7b2cbf", "#c1121f", "#1d3557"]
CMAP  = "viridis"
ECMAP = "magma"
BCMAP = "RdBu_r"

def style(ax, *, square=False, xlabel=None, ylabel=None, title=None):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title, pad=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if square: ax.set_aspect("equal", adjustable="box")
    return ax

def save(name):
    path = IMG_DIR / name
    plt.savefig(path)
    plt.close("all")
    print(f"  saved → {path.relative_to(ROOT)}")

# ── THBSplines imports ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(ROOT))

from THBSplines.src.b_spline_numpy import BSpline, TensorProductBSpline
from THBSplines.src.hierarchical_space import HierarchicalSpace
from THBSplines.src.hierarchical_mesh import HierarchicalMesh
from THBSplines.src.refinement import refine
from THBSplines.src.assembly import hierarchical_mass_matrix, hierarchical_stiffness_matrix
from THBSplines.src.evaluation import evaluate_hierarchical_basis, check_partition_of_unity


# ══════════════════════════════════════════════════════════════════════════════
# 1.  bspline_basis.png  — quadratic basis + partition of unity
# ══════════════════════════════════════════════════════════════════════════════
print("1/5  B-spline basis …")

global_knots = np.array([0, 0, 0, 1, 2, 3, 4, 4, 4], dtype=float)
degree = 2
n_basis = len(global_knots) - degree - 1  # 6

x = np.linspace(0, 4, 400)
total = np.zeros_like(x)

fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8), constrained_layout=True)

for i in range(n_basis):
    kv = global_knots[i: i + degree + 2]
    Bi = BSpline(degree, kv, int(i == n_basis - 1))
    yi = Bi(x)
    total += yi
    axes[0].plot(x, yi, lw=2.0, color=LINE[i % len(LINE)], label=rf"$B_{{{i},{degree}}}$")

style(axes[0], xlabel="Knot vector $x$", ylabel="$B(x)$", title="Quadratic B-spline Basis Functions")
axes[0].set_xlim(0, 4); axes[0].set_ylim(-0.03, 1.08)
axes[0].legend(ncol=4, loc="best", bbox_to_anchor=(0.5, 1.20))

axes[1].plot(x, total, lw=2.6, color=LINE[0], label=r"$\sum_i B_i(x)$")
axes[1].axhline(1, color=LINE[4], ls="--", lw=1.3, label="$y = 1$")
style(axes[1], xlabel="Knot vector $x$", ylabel="Basis sum", title="Partition of Unity")
axes[1].set_xlim(0, 4); axes[1].set_ylim(0.93, 1.07)
axes[1].legend(loc="lower right")

save("bspline_basis.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  hierarchical_mesh.png  — adaptive refinement levels
# ══════════════════════════════════════════════════════════════════════════════
print("2/5  Hierarchical mesh …")

knots_m = [[0, 0, 0, 1, 2, 3, 3, 3],
           [0, 0, 0, 1, 2, 3, 3, 3]]
mesh = HierarchicalMesh(knots_m, dim=2)
mesh.refine({0: [0, 1, 3, 4]})
mesh.refine({1: [0, 1, 3, 4]})

fig, ax = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
mesh.plot_cells(ax=ax)
style(ax, square=True, xlabel="$x$", ylabel="$y$",
      title="Hierarchical Mesh — Two Local Refinements")
ax.set_facecolor("white")
save("hierarchical_mesh.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  thb_basis_functions.png  — grid of selected THB-spline functions
# ══════════════════════════════════════════════════════════════════════════════
print("3/5  THB-spline basis functions …")

T = HierarchicalSpace([[0,0,0,1,2,3,3,3],[0,0,0,1,2,3,3,3]], [2, 2], dim=2)
T = refine(T, {0: [0, 1, 3, 4]})
sub = np.array([[0.0, 1.5], [0.0, 1.5]])
T = refine(T, {1: T.refine_in_rectangle(sub, level=1)})

n = 55
x1d = np.linspace(0, 3, n)
X, Y = np.meshgrid(x1d, x1d)
pts = np.column_stack([X.ravel(), Y.ravel()])
B_thb = evaluate_hierarchical_basis(T, pts)

n_show = min(T.nfuncs, 12)
fig, axes = plt.subplots(3, 4, figsize=(13.0, 9.4), constrained_layout=True)

for k, ax in enumerate(axes.flat):
    if k >= n_show:
        ax.axis("off")
        continue
    Z = B_thb[:, k].reshape(n, n)
    ax.contourf(X, Y, Z, levels=14, cmap="cividis")
    ax.set_title(rf"$\tilde{{B}}_{{{k}}}$", fontsize=10, pad=3)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

fig.suptitle(f"Selected THB-spline Basis Functions ({n_show} of {T.nfuncs})",
             fontsize=15, y=1.01)
save("thb_basis_functions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  matrices_spy.png  — sparsity pattern of mass + stiffness
# ══════════════════════════════════════════════════════════════════════════════
print("4/5  Sparsity patterns …")

knots_a = [[0,0,0,1/3,2/3,1,1,1], [0,0,0,1/3,2/3,1,1,1]]
Ta = HierarchicalSpace(knots_a, [2, 2], dim=2)
rect = np.array([[0.0, 0.5], [0.0, 0.5]])
Ta = refine(Ta, {0: Ta.refine_in_rectangle(rect, 0)})

import io, contextlib
with contextlib.redirect_stderr(io.StringIO()):   # suppress tqdm bars
    M = hierarchical_mass_matrix(Ta)
    A = hierarchical_stiffness_matrix(Ta)

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
axes[0].spy(M, markersize=5, color="#1d3557")
style(axes[0], title=f"Mass Matrix $M$\n{M.shape[0]}×{M.shape[1]},  nnz={M.nnz}")
axes[0].set_facecolor("white")
axes[1].spy(A, markersize=5, color="#c1121f")
style(axes[1], title=f"Stiffness Matrix $A$\n{A.shape[0]}×{A.shape[1]},  nnz={A.nnz}")
axes[1].set_facecolor("white")

save("matrices_spy.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  poisson_solution.png  — exact / FEM / error for −Δu = f
# ══════════════════════════════════════════════════════════════════════════════
print("5/5  Poisson solution …")

import scipy.sparse.linalg as spla

# Space on [0,1]²
knots_p = [[0,0,0,0.5,1,1,1], [0,0,0,0.5,1,1,1]]
Tp = HierarchicalSpace(knots_p, [2, 2], dim=2)
rect_p = np.array([[0.25, 0.75], [0.25, 0.75]])
with contextlib.redirect_stderr(io.StringIO()):
    Tp = refine(Tp, {0: Tp.refine_in_rectangle(rect_p, 0)})
    Ap = hierarchical_stiffness_matrix(Tp)

# Load vector via quadrature
n_q = 12
x_q = np.linspace(0.01, 0.99, n_q)
Xq, Yq = np.meshgrid(x_q, x_q)
pts_q = np.column_stack([Xq.ravel(), Yq.ravel()])
B_q = evaluate_hierarchical_basis(Tp, pts_q)
f_vals_q = 2 * np.pi**2 * np.sin(np.pi * pts_q[:, 0]) * np.sin(np.pi * pts_q[:, 1])
f_h = (1.0 / n_q**2) * (B_q.T @ f_vals_q)

# Boundary DOFs: a basis function is a boundary DOF if it is nonzero
# on any of the four edges x=0, x=1, y=0, y=1.
# We sample each edge and check which columns of the basis matrix are nonzero.
n_edge = 30
t_edge = np.linspace(0, 1, n_edge)
edge_pts = np.vstack([
    np.column_stack([np.zeros(n_edge), t_edge]),   # x = 0
    np.column_stack([np.ones(n_edge),  t_edge]),   # x = 1
    np.column_stack([t_edge, np.zeros(n_edge)]),   # y = 0
    np.column_stack([t_edge, np.ones(n_edge)]),    # y = 1
])
B_edge = evaluate_hierarchical_basis(Tp, edge_pts)
on_boundary = np.any(B_edge > 1e-12, axis=0)      # shape (nfuncs,)
interior = np.where(~on_boundary)[0]

Ap_csr = Ap.tocsr()
A_int  = Ap_csr[np.ix_(interior, interior)]
f_int  = f_h[interior]
c_int  = spla.spsolve(A_int, f_int)
c_full = np.zeros(Tp.nfuncs)
c_full[interior] = c_int

# Evaluate on a fine grid
n_p = 80
x_p = np.linspace(0, 1, n_p)
Xp, Yp = np.meshgrid(x_p, x_p)
pts_p  = np.column_stack([Xp.ravel(), Yp.ravel()])
B_plot  = evaluate_hierarchical_basis(Tp, pts_p)
u_h     = (B_plot @ c_full).reshape(n_p, n_p)
u_exact = np.sin(np.pi * Xp) * np.sin(np.pi * Yp)
err     = np.abs(u_h - u_exact)

lvls_u = np.linspace(0, u_exact.max(), 18)
lvls_e = np.linspace(0, err.max() + 1e-12, 18)

print(max(u_exact.ravel()), max(u_h.ravel()), max(err.ravel()))

def cfill(fig, ax, X, Y, Z, levels, cmap, title, clabel):
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    style(ax, square=True, xlabel="$x$", ylabel="$y$", title=title)
    cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.048)
    cb.set_label(clabel)

fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), constrained_layout=True)
cfill(fig, axes[0], Xp, Yp, u_exact, lvls_u, CMAP,
      r"Exact $u(x,y)=\sin(\pi x)\sin(\pi y)$", "$u$")
cfill(fig, axes[1], Xp, Yp, u_h, lvls_u, CMAP,
      r"THB-spline FEM Solution $u_h$", "$u_h$")
cfill(fig, axes[2], Xp, Yp, err, lvls_e, ECMAP,
      r"Pointwise Error $|u_h - u|$", "error")

L2 = np.sqrt(np.mean((u_h - u_exact)**2))
fig.suptitle(f"Poisson Equation on $[0,1]^2$ — THB-spline FEM  (L² error = {L2:.2e})",
             fontsize=13, y=1.02)
save("poisson_solution.png")

print("\nAll images written to THBSplines/images/")
