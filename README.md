# THBSplines

## What are Truncated Hierarchical B-Splines?

**Truncated Hierarchical B-Splines (THB-splines)** are an extension of classical **B-splines** that enable **local refinement** in isogeometric analysis (IGA) while maintaining important mathematical properties such as partition of unity, non-negativity, and linear independence.  

- **B-splines** are piecewise polynomial functions commonly used in computer graphics and numerical analysis because they provide smooth and flexible representations of curves and surfaces.  
- **Hierarchical B-splines (HB-splines)** allow local mesh refinement by organizing B-splines into levels of resolution, but they may introduce redundancy and lose partition of unity.  
- **Truncated Hierarchical B-splines (THB-splines)** improve on this by introducing a **truncation mechanism**: basis functions from coarser levels are “trimmed” where finer-level functions are active.  

This truncation ensures that:  
1. The basis functions remain **linearly independent**.  
2. The representation is **sparse and efficient**.  
3. The method supports **adaptive refinement**, crucial for finite element methods and isogeometric analysis.  

THB-splines are particularly useful in solving partial differential equations (PDEs) with **adaptive isogeometric methods**, where computational effort is concentrated in regions requiring higher accuracy (e.g., around singularities or sharp gradients).

---

## Truncated Hierarchical B-Splines in Python

This repository contains a dimension-independent Python-implementation of truncated hierarchical B-splines, and methods for the assembly of stiffness and mass matrices.  

The implementation is based on the article [Algorithms for the implementation of adaptive isogeometric methods using hierarchical B-splines](https://doi.org/10.1016/j.apnum.2017.08.006), and is heavily influenced by the [GeoPDEs](http://rafavzqz.github.io/geopdes/) Matlab/Octave package for isogeometric analysis developed by the authors.

⚠️ **Note:** This project is mainly for **my debugging and learning purposes**.
---

## Example - computing the mass and stiffness matrix

The computation of finite element matrices is fairly simple. Initialize the hierarchical space. Refine the space by choosing specific elements, or a rectangular region of refinement, and finally, assemble the matrices.

```python
import THBSplines as thb
import matplotlib.pyplot as plt

# Initialize a biquadraic space of Truncated Hierarchical B-Splines
knots = [
  [0, 0, 1/3, 2/3, 1, 1],
  [0, 0, 1/3, 2/3, 1, 1]
]
degrees = [2, 2]
dimension = 2
T = thb.HierarchicalSpace(knots, degrees, dimension)

# Select cells to refine at each level, either by explicitly marking the elements, or by choosing a rectangular region.
cells_to_refine = {}
cells_to_refine[0] = [0, 1, 2, 3, 4, 5, 6]
T = thb.refine(T, cells_to_refine)

rect = np.array([[0, 1 / 3], [0, 2 / 3]])
cells_to_refine[1] = T.refine_in_rectangle(rect, level = 1)
T = thb.refine(T, cells_to_refine)
T.mesh.plot_cells()
```

![](THBSplines/images/refined_mesh.png)

```python
# If no integration order is specified, exact gauss quadrature suitable for the given basis is used.
mass_matrix = thb.hierarchical_mass_matrix(T)
stiffness_matrix = thb.hierarchical_stiffness_matrix(T)

plt.spy(mass_matrix, markersize=1)
plt.show()
```

![](THBSplines/images/mass_matrix.png)

