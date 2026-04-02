"""
HBSplines — Hierarchical B-Spline FEM for 1D adaptive problems.

Public API
----------
HBsplineSpace   : hierarchical space (active/deactivated B-splines per level)
adaptive_solve  : full AFEM loop (solve → estimate → mark → refine)
Problem         : dataclass describing a 1D BVP
"""

from HBSplines.src.hb_space import HBsplineSpace
from HBSplines.src.adaptive_solver import adaptive_solve, AdaptiveSolverSettings
from HBSplines.problems import Problem

__all__ = ["HBsplineSpace", "adaptive_solve", "AdaptiveSolverSettings", "Problem"]
