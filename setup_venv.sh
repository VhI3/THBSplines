#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_venv.sh — Create a Python virtual environment for THBSplines,
#                 install all dependencies, and run a smoke test.
#
# Usage:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e # exit immediately on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv-thbsplines"

echo "──────────────────────────────────────────────"
echo " THBSplines — virtual environment setup"
echo " Project : $PROJECT_DIR"
echo " venv    : $VENV_DIR"
echo "──────────────────────────────────────────────"

# ── 1. Create .venv ────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
  echo "[1/4] .venv already exists — skipping creation."
else
  echo "[1/4] Creating .venv with $(python3 --version) ..."
  python3 -m venv "$VENV_DIR"
  echo "      Done."
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# ── 2. Upgrade pip + install dependencies ────────────────────────────────────
echo "[2/4] Upgrading pip ..."
"$PYTHON" -m pip install --upgrade pip --quiet

echo "[2/4] Installing dependencies ..."
"$PIP" install --quiet \
  "numpy>=1.24" \
  "scipy>=1.10" \
  "matplotlib>=3.7" \
  "tqdm>=4.65" \
  "jupyterlab>=4.0" \
  "ipympl>=0.9" \
  "pytest>=7.0" \
  "pytest-cov"

echo "      Installing THBSplines in editable mode ..."
"$PIP" install --quiet -e "$PROJECT_DIR"
echo "      Done."

# ── 3. Smoke test ─────────────────────────────────────────────────────────────
echo "[3/4] Running smoke test ..."
"$PYTHON" - <<'PYEOF'
import sys
print(f"Python {sys.version}")

import numpy as np
import scipy
import matplotlib
import tqdm
print(f"numpy   {np.__version__}")
print(f"scipy   {scipy.__version__}")
print(f"matplotlib {matplotlib.__version__}")
print(f"tqdm    {tqdm.__version__}")
print()

# ── B-spline evaluation ───────────────────────────────────────────────────────
from THBSplines.src.b_spline_numpy import BSpline, TensorProductBSpline

B = BSpline(2, np.array([0., 1., 2., 3.]))
x = np.linspace(0, 3, 50)
assert B(x).shape == (50,), "BSpline shape mismatch"

# Verify quadratic B-spline values at a few points
assert abs(B(np.array([1.5]))[0] - 0.75) < 1e-12, "BSpline value wrong"
print("✓  BSpline evaluation")

# Derivative via finite difference check
h = 1e-6
fd = (B(x + h) - B(x - h)) / (2 * h)
assert np.allclose(B.D(x, 1), fd, atol=1e-5), "BSpline derivative wrong"
print("✓  BSpline derivative")

# Tensor-product B-spline
degrees = np.array([2, 2])
knots   = np.array([[0., 1., 2., 3.], [0., 1., 2., 3.]])
B2 = TensorProductBSpline(degrees, knots)
pts = np.random.default_rng(0).uniform(0, 3, (20, 2))
vals = B2(pts)
grad = B2.grad(pts)
assert vals.shape == (20,),    "TP BSpline shape wrong"
assert grad.shape == (20, 2),  "TP BSpline grad shape wrong"
# Non-negativity
assert np.all(vals >= -1e-14), "TP BSpline non-negativity violated"
print("✓  TensorProductBSpline evaluation + gradient")

# ── Tensor-product space ──────────────────────────────────────────────────────
from THBSplines.src.tensor_product_space import TensorProductSpace2D

sp = TensorProductSpace2D(
    [[0,0,0,1,2,3,3,3], [0,0,0,1,2,3,3,3]], [2,2], dim=2
)
assert sp.nfuncs == 25, f"Expected 25 basis functions, got {sp.nfuncs}"
print("✓  TensorProductSpace2D (25 biquadratic basis functions)")

# ── Hierarchical space + refinement ──────────────────────────────────────────
from THBSplines.src.hierarchical_space import HierarchicalSpace
from THBSplines.src.refinement import refine
from THBSplines.src.evaluation import check_partition_of_unity

T = HierarchicalSpace(
    [[0,0,0,1,2,3,3,3], [0,0,0,1,2,3,3,3]], [2,2], dim=2
)
assert T.nfuncs == 25

T = refine(T, {0: [0, 1, 3, 4]})
print(f"✓  HierarchicalSpace  level 0 refine → {T.nfuncs} DOFs")

# Second refinement
sub = np.array([[0.0, 1.5], [0.0, 1.5]])
cells_l1 = T.refine_in_rectangle(sub, level=1)
T = refine(T, {1: cells_l1})
print(f"✓  Second refinement  → {T.nfuncs} DOFs over {T.nlevels} levels")

# ── Partition of unity ────────────────────────────────────────────────────────
ok = check_partition_of_unity(T, n_pts=10)
assert ok, "Partition of unity failed!"
print("✓  Partition of unity holds")

# ── Subdivision matrix ────────────────────────────────────────────────────────
from THBSplines.src.evaluation import create_subdivision_matrix
C = create_subdivision_matrix(T, mode='full')
assert len(C) == T.nlevels
print(f"✓  Subdivision matrix  ({T.nlevels} levels)")

# ── Assembly (small problem, no tqdm output in smoke test) ────────────────────
import io, contextlib
from THBSplines.src.assembly import hierarchical_mass_matrix, hierarchical_stiffness_matrix

# Use a tiny space for speed
Ts = HierarchicalSpace(
    [[0,0,1,1], [0,0,1,1]], [1,1], dim=2
)
with contextlib.redirect_stderr(io.StringIO()):  # suppress tqdm bars
    M = hierarchical_mass_matrix(Ts)
    A = hierarchical_stiffness_matrix(Ts)

assert M.shape == (4, 4)
assert np.allclose(M.toarray(), M.toarray().T), "Mass matrix not symmetric"
assert np.allclose(A.toarray(), A.toarray().T), "Stiffness matrix not symmetric"
print(f"✓  Mass matrix      {M.shape}  symmetric: {np.allclose(M.toarray(), M.toarray().T)}")
print(f"✓  Stiffness matrix {A.shape}  symmetric: {np.allclose(A.toarray(), A.toarray().T)}")

print()
print("══════════════════════════════════════════════")
print("  All smoke tests passed ✓")
print("══════════════════════════════════════════════")
PYEOF

# ── 4. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Setup complete."
echo ""
echo "  Activate the environment:"
echo "    source .venv-thbsplines/bin/activate"
echo ""
echo "  Run the tests:"
echo "    pytest THBSplines/tests/ -v"
echo ""
echo "  Launch the Jupyter notebook:"
echo "    jupyter lab notebooks/THBSplines_tutorial.ipynb"
echo ""
