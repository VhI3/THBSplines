import numpy as np
import io
import contextlib
import scipy.sparse.linalg as spla
from THBSplines import create_subdivision_matrix
from THBSplines.src.assembly import hierarchical_stiffness_matrix
from THBSplines.src.evaluation import evaluate_hierarchical_basis
from THBSplines.src.hierarchical_space import HierarchicalSpace
from THBSplines.src.refinement import refine


def test_active_funcs_per_level():
    knots = [
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)

    np.testing.assert_equal(T.afunc_level, {0: [0, 1, 2, 3]})


def test_active_funcs_per_level_refine():
    knots = [
        [0, 0, 1, 2, 2],
        [0, 0, 1, 2, 2]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    cells = {0: [0]}
    T = refine(T, cells)

    np.testing.assert_equal(T.nfuncs_level, {0: 8, 1: 4})


def test_functions_to_deactivate_from_cells():
    knots = [
        [0, 0, 1, 2, 2],
        [0, 0, 1, 2, 2]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0]}
    new_cells = T.mesh.refine(marked_cells)
    marked_functions = T.functions_to_deactivate_from_cells(marked_cells)

    np.testing.assert_equal(marked_functions, {0: [0]})


def test_projection_matrix_linear():
    knots = [
        [0, 0, 1, 2, 3, 3]
    ]
    d = [1]
    dim = 1
    T = HierarchicalSpace(knots, d, dim)
    cells = {0: [1]}
    T = refine(T, cells)

    C = T.compute_full_projection_matrix(0)

    assert C.shape == (7, 4)
    np.testing.assert_allclose(C.toarray(), np.array([
        [1, 0, 0, 0],
        [0.5, 0.5, 0, 0],
        [0, 1, 0, 0],
        [0, 0.5, 0.5, 0],
        [0, 0, 1, 0],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1]]
    ))


def test_projection_matrix_bilinear():
    knots = [
        [0, 0, 1, 2, 3, 3],
        [0, 0, 1, 2, 3, 3]
    ]
    d = [1, 1]
    dim = 2
    T = HierarchicalSpace(knots, d, dim)
    cells = {0: [1]}
    T = refine(T, cells)
    C = T.compute_full_projection_matrix(0)
    assert C.shape == (49, 16)


def test_subdivision_matrix_linear():
    knots = [
        [0, 0, 1 / 3, 2 / 3, 1, 1]
    ]
    d = [1]
    dim = 1
    T = HierarchicalSpace(knots, d, dim)
    cells = {0: [1]}
    T = refine(T, cells)
    C = T.create_subdivision_matrix('full')

    np.testing.assert_allclose(C[0].toarray(), np.eye(4))
    np.testing.assert_allclose(C[1].toarray(), np.array([
        [1, 0, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0.5, 0.5, 0],
        [0, 0, 0, 1, 0]
    ]))


def test_change_of_basis_matrix_linear():
    knots = [
        [0, 0, 1 / 3, 2 / 3, 1, 1]
    ]
    d = [1]
    dim = 1
    T = HierarchicalSpace(knots, d, dim)
    cells = {0: [1]}
    T = refine(T, cells)
    C = T.get_basis_conversion_matrix(0)
    np.testing.assert_allclose(C.toarray(), np.array([
        [1, 0, 0, 0],
        [0.5, 0.5, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1]
    ]))


def test_full_mesh_refine():
    knots = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ], dtype=np.float64)
    deg = [1, 1]
    dim = 2
    T = HierarchicalSpace(knots, deg, dim)

    np.testing.assert_equal(T.nfuncs_level, {
        0: 4
    })

    cells = {}
    rectangle = np.array([[0, 1 + np.spacing(1)], [0, 1 + np.spacing(1)]], dtype=np.float64)
    cells[0] = T.refine_in_rectangle(rectangle, 0)
    T = refine(T, cells)

    np.testing.assert_equal(T.nfuncs_level, {
        0: 0,
        1: 9,
    })

    cells[1] = T.refine_in_rectangle(rectangle, 1)
    T = refine(T, cells)

    np.testing.assert_equal(T.nfuncs_level, {
        0: 0,
        1: 0,
        2: 25
    })

    cells[2] = T.refine_in_rectangle(rectangle, 2)
    T = refine(T, cells)

    np.testing.assert_equal(T.nfuncs_level, {
        0: 0,
        1: 0,
        2: 0,
        3: 81
    })


def test_partition_of_unity():
    knots = [
        [0, 0, 0, 1, 2, 3, 3, 3],
        [0, 0, 0, 1, 2, 3, 3, 3]
    ]
    d = 2
    degrees = [2, 2]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0, 1, 2, 3]}
    T = refine(T, marked_cells)
    marked_cells = {0: [0, 1, 2, 3], 1: [0, 1, 2]}
    T = refine(T, marked_cells)
    C = create_subdivision_matrix(T)
    N = 5
    x = np.linspace(0, 3, N)
    y = np.linspace(0, 3, N)
    z = np.zeros((N, N))


    c = C[T.nlevels - 1]
    c = c.toarray()
    for i in range(T.nfuncs):
        u = np.zeros(T.nfuncs)
        u[i] = 1
        u_fine = c @ u

        f = T.spaces[T.nlevels - 1].construct_function(u_fine)
        for i in range(N):
            for j in range(N):
                z[i, j] += f(np.array([x[i], y[j]]))

    np.testing.assert_allclose(z, 1)


def test_poisson_demo_setup_activates_fine_level_and_produces_bounded_solution():
    knots = [
        [0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1],
        [0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1],
    ]
    T = HierarchicalSpace(knots, [2, 2], dim=2)
    rect = np.array([[0.25, 0.75], [0.25, 0.75]])

    with contextlib.redirect_stderr(io.StringIO()):
        T = refine(T, {0: T.refine_in_rectangle(rect, 0)})
        A = hierarchical_stiffness_matrix(T)

    # Regression for the old demo setup: it refined the mesh but activated
    # no fine-level functions, so the "adaptive" solve stayed coarse.
    assert T.nlevels == 2
    assert T.nfuncs_level[1] > 0

    def f_rhs(pts):
        return 2 * np.pi**2 * np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])

    def thb_load_vector(hspace, order=5):
        pts_1d, w_1d = np.polynomial.legendre.leggauss(order)
        out = np.zeros(hspace.nfuncs)

        for level in range(hspace.nlevels):
            cells = hspace.mesh.meshes[level].cells[hspace.mesh.aelem_level[level]]
            for cell in cells:
                u0, u1 = cell[0, 0], cell[0, 1]
                v0, v1 = cell[1, 0], cell[1, 1]
                up = 0.5 * (u1 - u0) * pts_1d + 0.5 * (u0 + u1)
                vp = 0.5 * (v1 - v0) * pts_1d + 0.5 * (v0 + v1)
                U, V = np.meshgrid(up, vp)
                pts = np.column_stack([U.ravel(), V.ravel()])
                Wu, Wv = np.meshgrid(w_1d, w_1d)
                w = (Wu * Wv).ravel() * 0.25 * (u1 - u0) * (v1 - v0)
                B = evaluate_hierarchical_basis(hspace, pts)
                out += B.T @ (w * f_rhs(pts))

        return out

    is_interior = np.zeros(T.nfuncs, dtype=bool)
    cumulative = 0
    eps = 1e-12
    for level in range(T.nlevels):
        space = T.spaces[level]
        kx, ky = space.knots[0], space.knots[1]
        p = int(space.degrees[0])
        nx, ny = space.nfuncs_onedim
        gx = np.array([np.mean(kx[i + 1:i + p + 1]) for i in range(nx)])
        gy = np.array([np.mean(ky[j + 1:j + p + 1]) for j in range(ny)])
        active = T.afunc_level[level]

        for local_idx, k in enumerate(active):
            i, j = k // ny, k % ny
            if gx[i] > eps and gx[i] < 1 - eps and gy[j] > eps and gy[j] < 1 - eps:
                is_interior[cumulative + local_idx] = True

        cumulative += T.nfuncs_level[level]

    interior = np.where(is_interior)[0]
    assert interior.size > 0

    f_h = thb_load_vector(T)
    A_int = A.tocsr()[np.ix_(interior, interior)]
    c_int = spla.spsolve(A_int, f_h[interior])
    c_full = np.zeros(T.nfuncs)
    c_full[interior] = c_int

    n_plot = 60
    x1d = np.linspace(0, 1, n_plot)
    X, Y = np.meshgrid(x1d, x1d)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    B_plot = evaluate_hierarchical_basis(T, pts)
    u_h = (B_plot @ c_full).reshape(n_plot, n_plot)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    err = np.abs(u_h - u_exact)

    assert np.isfinite(u_h).all()
    assert float(u_h.max()) < 1.1
    assert float(err.max()) < 0.1
