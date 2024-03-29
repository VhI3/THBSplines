import numpy as np

from THBSplines.src.abstract_mesh import Mesh


class CartesianMesh(Mesh):

    def plot_cells(self) -> None:
        pass

    def get_gauss_points(self, cell_indices: np.ndarray) -> np.ndarray:
        pass

    def __init__(self, knots, parametric_dimension):
        """
        Represents a regular cartesian mesh in ``parametric_dimension`` dimensions.
        :param knots:
        :param parametric_dimension:
        """
        self.knots = np.array([np.unique(knot_v) for knot_v in knots])
        self.dim = parametric_dimension
        self.cells = self.compute_cells()
        self.nelems = len(self.cells)
        self.cell_area = np.prod(np.diff(self.cells[0][:]))

    def compute_cells(self) -> np.ndarray:
        """
        Computes an array of cells, represented as AABBs with each cell as [[min1, max1], [min2, max2], ..., ]
        :return: a list of N cells of shape (N, dim, 2).
        """
        knots_left = [k[:-1] for k in self.knots]
        knots_right = [k[1:] for k in self.knots]
        cells_bottom_left = np.stack(np.meshgrid(*knots_left), -1).reshape(-1, self.dim)
        cells_top_right = np.stack(np.meshgrid(*knots_right), -1).reshape(-1, self.dim)
        cells = np.concatenate((cells_bottom_left, cells_top_right), axis=1).reshape(-1, self.dim, 2)

        # TODO: Make sure this edge case is not needed. For univariate meshes, the axes does NOT have to be swapped.
        if self.dim != 1:
            cells = np.swapaxes(cells, 1, 2)

        return cells

    def refine(self) -> 'CartesianMesh':
        """
        Dyadic refinement of the mesh, by inserting midpoints in each knot vector.
        :return: a refined CartesianMesh object.
        """
        refined_knots = np.array([
            np.sort(np.concatenate((knot_v, (knot_v[1:] + knot_v[:-1]) / 2))) for knot_v in self.knots
        ])
        return CartesianMesh(refined_knots, self.dim)

    def get_sub_elements(self, box):
        """
        Returns the indices of the cells that are contained in the region delimited by `box`.
        :param box:
        :return:
        """

        indices = []
        for i, element in enumerate(self.cells):
            condition = (element[:, 0] >= box[:, 0]) & (element[:, 1] <= box[:, 1])
            if np.all(condition):
                indices.append(i)
        return indices

if __name__ == '__main__':
    knots = [
        [0, 1, 2],
        [0, 1, 2]
    ]
    C = CartesianMesh(knots, 2)
    C1 = C.refine().refine().refine()
