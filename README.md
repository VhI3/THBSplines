# THBSplines

## What are Truncated Hierarchical B-Splines?

**Truncated Hierarchical B-Splines (THB-splines)** are an extension of classical **B-splines**
that enable **local refinement** in isogeometric analysis (IGA) while maintaining important
mathematical properties such as partition of unity, non-negativity, and linear independence.

- **B-splines** are piecewise polynomial functions commonly used in computer graphics and
  numerical analysis because they provide smooth and flexible representations of curves and surfaces.
- **Hierarchical B-splines (HB-splines)** allow local mesh refinement by organizing B-splines
  into levels of resolution, but they may introduce redundancy and lose partition of unity.
- **Truncated Hierarchical B-splines (THB-splines)** improve on this by introducing a
  **truncation mechanism**: basis functions from coarser levels are trimmed wherever
  finer-level functions are active.

This truncation guarantees that:

1. The basis functions remain **linearly independent**.
2. The representation is **sparse and efficient**.
3. The method supports **adaptive refinement**, crucial for finite element methods and isogeometric analysis.

THB-splines are particularly useful in solving PDEs with **adaptive isogeometric methods**,
where computational effort is concentrated in regions requiring higher accuracy
(e.g. around singularities or sharp gradients).

---

## Implementation

This repository contains a dimension-independent, **pure NumPy/SciPy** Python implementation
of truncated hierarchical B-splines together with methods for assembling stiffness and mass matrices.

The implementation is based on the article
[Algorithms for the implementation of adaptive isogeometric methods using hierarchical B-splines](https://doi.org/10.1016/j.apnum.2017.08.006)
and is heavily influenced by the
[GeoPDEs](http://rafavzqz.github.io/geopdes/) MATLAB/Octave package.

> **Note:** This project is mainly for my own **debugging and learning purposes**.

### Requirements

- Python >= 3.10
- NumPy, SciPy, Matplotlib, tqdm

**Quick setup** (creates an isolated virtual environment and runs a smoke test):

```bash
chmod +x setup_venv.sh && ./setup_venv.sh
source .venv-thbsplines/bin/activate
```

**Run the tests:**

```bash
pytest THBSplines/tests/ -v
```

**Interactive tutorial:**

```bash
jupyter lab notebooks/THBSplines_tutorial.ipynb
```

---

## Gallery

### 1.  B-spline basis functions

Six quadratic basis functions on `[0, 4]` with a clamped knot vector, and their sum (partition of unity).

![B-spline basis functions and partition of unity](THBSplines/images/bspline_basis.png)

---

### 2.  Adaptive hierarchical mesh

Two rounds of local dyadic refinement applied to the lower-left region of a biquadratic mesh.

![Hierarchical mesh after two local refinements](THBSplines/images/hierarchical_mesh.png)

---

### 3 -- THB-spline basis functions

Selected truncated hierarchical basis functions after two adaptive refinements.
Each function has compact support that is smaller than or equal to the corresponding HB-spline support.

![Selected THB-spline basis functions](THBSplines/images/thb_basis_functions.png)

---

### 4.  Mass and stiffness matrix sparsity

Sparsity patterns of the assembled mass matrix **M** and stiffness matrix **A**
on an adaptively refined biquadratic space.

![Sparsity patterns of the mass and stiffness matrices](THBSplines/images/matrices_spy.png)

---

### 5. Poisson equation solved with THB-splines

The Poisson problem $\Delta u = f$  on $[0,1]^2$ with exact solution $u = sin(\pi x)sin(\pi  y)$,
solved by a Galerkin THB-spline method on an adaptively refined mesh.

![Poisson solution: exact, FEM, and pointwise error](THBSplines/images/poisson_solution.png)

---

## Code example

```python
import numpy as np
import THBSplines as thb

# -- 1. Create a biquadratic hierarchical space --------------------------------
knots = [
    [0, 0, 0, 1/3, 2/3, 1, 1, 1],
    [0, 0, 0, 1/3, 2/3, 1, 1, 1],
]
degrees   = [2, 2]
dimension = 2
T = thb.HierarchicalSpace(knots, degrees, dimension)

# -- 2. Adaptive refinement ---------------------------------------------------
# Mark elements by index, or specify a rectangular sub-region.
T = thb.refine(T, {0: [0, 1, 2, 3, 4, 5, 6]})

rect = np.array([[0.0, 1/3], [0.0, 2/3]])
cells_l1 = T.refine_in_rectangle(rect, level=1)
T = thb.refine(T, {1: cells_l1})

T.mesh.plot_cells()          # visualise the refined mesh

# -- 3. Assemble FEM matrices -------------------------------------------------
# Exact Gauss quadrature order is chosen automatically.
M = thb.hierarchical_mass_matrix(T)
A = thb.hierarchical_stiffness_matrix(T)

print(f"DOFs: {T.nfuncs}  |  M shape: {M.shape}  |  A nnz: {A.nnz}")
```
