"""
Benchmark problems for HB-spline adaptive FEM.

All problems solve:
    -m u''(x) + b u'(x) = f(x)   on [0, 1]
    u(0) = u0,  u(1) = uL

Ported from the MATLAB benchmark problems in arielsboiardi/adAHBsplineFEM.

References
----------
Höllig, *Finite Element Methods with B-Splines*, SIAM 2003.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Problem:
    """
    1D boundary-value problem container.

    Attributes
    ----------
    m    : diffusion coefficient  (>0)
    b    : advection coefficient  (can be 0)
    f    : forcing function  f(x: ndarray) -> ndarray
    u0   : Dirichlet BC at x=0
    uL   : Dirichlet BC at x=1
    name : human-readable label
    u_ex : exact solution (optional)  u_ex(x: ndarray) -> ndarray
    du_ex: exact derivative (optional)
    """
    m     : float
    b     : float
    f     : Callable[[np.ndarray], np.ndarray]
    u0    : float
    uL    : float
    name  : str = "unnamed"
    u_ex  : Callable[[np.ndarray], np.ndarray] | None = None
    du_ex : Callable[[np.ndarray], np.ndarray] | None = None


# ---------------------------------------------------------------------------
# Right boundary layer
# ---------------------------------------------------------------------------

def boundary_layer_right(m: float = 1.0, b: float = 10.0) -> Problem:
    """
    Right boundary layer:  exact solution  u(x) = (exp(b/m * x) - 1) / (exp(b/m) - 1).

    The solution is smooth everywhere except near x=1, where it rises
    steeply over a layer of width O(m/b).
    """
    lam = b / m
    denom = np.exp(lam) - 1.0

    def u_ex(x: np.ndarray) -> np.ndarray:
        return (np.exp(lam * x) - 1.0) / denom

    def du_ex(x: np.ndarray) -> np.ndarray:
        return lam * np.exp(lam * x) / denom

    # f = -m u'' + b u' = 0 (exact solution satisfies the homogeneous PDE)
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    return Problem(
        m=m, b=b, f=f,
        u0=float(u_ex(np.array([0.0]))[0]),
        uL=float(u_ex(np.array([1.0]))[0]),
        name=f"right_boundary_layer (m={m}, b={b})",
        u_ex=u_ex, du_ex=du_ex,
    )


# ---------------------------------------------------------------------------
# Left boundary layer
# ---------------------------------------------------------------------------

def boundary_layer_left(m: float = 1.0, b: float = -10.0) -> Problem:
    """
    Left boundary layer  (b < 0):  exact solution  u(x) = (exp(b/m * x) - 1) / (exp(b/m) - 1).

    The solution is steep near x=0.
    """
    lam = b / m
    denom = np.exp(lam) - 1.0

    def u_ex(x: np.ndarray) -> np.ndarray:
        return (np.exp(lam * x) - 1.0) / denom

    def du_ex(x: np.ndarray) -> np.ndarray:
        return lam * np.exp(lam * x) / denom

    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    return Problem(
        m=m, b=b, f=f,
        u0=float(u_ex(np.array([0.0]))[0]),
        uL=float(u_ex(np.array([1.0]))[0]),
        name=f"left_boundary_layer (m={m}, b={b})",
        u_ex=u_ex, du_ex=du_ex,
    )


# ---------------------------------------------------------------------------
# Interior layer (Runge-type)
# ---------------------------------------------------------------------------

def interior_layer(alpha: float = 1e5) -> Problem:
    """
    Interior layer:  exact solution  u(x) = 1/(1 + alpha*(x-0.5)^2) - 1/(1+alpha/4).

    Steep gradient at x=0.5; zero Dirichlet BCs (u vanishes at both ends
    because the second term subtracts the boundary value).

    This is a pure diffusion problem  (-u'' = f,  b=0)  with a sharp
    feature in the solution interior.
    """
    shift = 1.0 / (1.0 + alpha / 4.0)

    def u_ex(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + alpha * (x - 0.5) ** 2) - shift

    def du_ex(x: np.ndarray) -> np.ndarray:
        return -2.0 * alpha * (x - 0.5) / (1.0 + alpha * (x - 0.5) ** 2) ** 2

    # d²u/dx² = derivative of du_ex
    def ddu_ex(x: np.ndarray) -> np.ndarray:
        denom = (1.0 + alpha * (x - 0.5) ** 2)
        return (
            -2.0 * alpha / denom ** 2
            + 8.0 * alpha ** 2 * (x - 0.5) ** 2 / denom ** 3
        )

    def f(x: np.ndarray) -> np.ndarray:
        # -u'' + 0*u' = -u''
        return -ddu_ex(x)

    return Problem(
        m=1.0, b=0.0, f=f,
        u0=0.0, uL=0.0,
        name=f"interior_layer (alpha={alpha:.0e})",
        u_ex=u_ex, du_ex=du_ex,
    )


# ---------------------------------------------------------------------------
# Pure diffusion with smooth solution (easy baseline)
# ---------------------------------------------------------------------------

def smooth_diffusion() -> Problem:
    """
    Smooth solution  u(x) = sin(pi*x)  for  -u'' = pi^2 * sin(pi*x).

    Useful as a sanity-check problem: the exact solution is smooth, so
    uniform and adaptive refinement should give identical convergence rates.
    """
    def u_ex(x: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x)

    def du_ex(x: np.ndarray) -> np.ndarray:
        return np.pi * np.cos(np.pi * x)

    def f(x: np.ndarray) -> np.ndarray:
        return np.pi ** 2 * np.sin(np.pi * x)

    return Problem(
        m=1.0, b=0.0, f=f,
        u0=0.0, uL=0.0,
        name="smooth_diffusion (u=sin(pi*x))",
        u_ex=u_ex, du_ex=du_ex,
    )
