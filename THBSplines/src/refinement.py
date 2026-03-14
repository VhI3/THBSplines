"""
High-level refinement interface.

This module provides the single public function ``refine``, which is the
recommended entry point for performing one step of hierarchical refinement.

Refinement overview
-------------------
Given a ``HierarchicalSpace`` T and a dict ``marked_cells`` mapping each
level ℓ to a list of cell indices to refine, the algorithm:

1. Refines the hierarchical *mesh*: deactivates marked cells and activates
   their 2^d children on the next level.  See ``HierarchicalMesh.refine``.

2. Determines which basis *functions* must be deactivated as a result: a
   function is deactivated when all cells in its support are being refined
   away.  See ``HierarchicalSpace.functions_to_deactivate_from_cells``.

3. Refines the hierarchical *space*: deactivates the identified functions,
   promotes their children to the next level, and activates any additional
   functions at the new level whose support is fully covered.
   See ``HierarchicalSpace.refine``.

Usage example
-------------
::

    import THBSplines as thb

    knots   = [[0, 0, 0, 1, 2, 3, 3, 3], [0, 0, 0, 1, 2, 3, 3, 3]]
    degrees = [2, 2]
    T = thb.HierarchicalSpace(knots, degrees, dim=2)

    # Mark cells [0, 1, 2, 3] at level 0 for refinement
    T = thb.refine(T, {0: [0, 1, 2, 3]})
    T.mesh.plot_cells()
"""

from __future__ import annotations

import logging

from THBSplines.src.hierarchical_space import HierarchicalSpace

logger = logging.getLogger(__name__)


def refine(
    hspace: HierarchicalSpace,
    marked_entities: dict,
    entity_type: str = "cells",
) -> HierarchicalSpace:
    """
    Perform one step of hierarchical adaptive refinement.

    Parameters
    ----------
    hspace         : the current hierarchical space (modified in-place and returned)
    marked_entities: mapping  level → array-like of cell indices to refine
    entity_type    : currently only ``'cells'`` is supported (function marking
                     may be added in a future version)

    Returns
    -------
    HierarchicalSpace
        The updated space (same object, returned for convenience so that
        ``T = refine(T, cells)`` is a natural idiom).

    Notes
    -----
    The space is modified **in-place**.  The return value is the same object
    as ``hspace``, not a copy.
    """
    if entity_type != "cells":
        raise NotImplementedError(
            f"entity_type='{entity_type}' is not supported. Use 'cells'."
        )

    logger.info(
        "Refining hierarchical space  |  "
        "space levels: %d  |  mesh levels: %d",
        hspace.nlevels,
        hspace.mesh.nlevels,
    )

    # Step 1: refine the mesh (cell deactivation + child activation)
    new_cells = hspace.mesh.refine(marked_entities)

    # Step 2: identify which functions lose all their active cells
    marked_functions = hspace.functions_to_deactivate_from_cells(marked_entities)

    # Step 3: update the function sets in the hierarchical space
    hspace.refine(marked_functions, new_cells)

    logger.info(
        "After refinement  |  nfuncs: %d  |  nelems: %d",
        hspace.nfuncs,
        hspace.mesh.nel,
    )

    return hspace
