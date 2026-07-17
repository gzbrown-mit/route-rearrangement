"""Enumerate all valid orderings of one synthesis route's steps and materialize each
ordering's molecules by backward retro-template re-application.

The combinatorics (which orderings are chemically possible) come from
``synthesis_extraction.dependency`` — material atom-lineage edges, protection brackets,
counterfactual FG-exposure edges, and the downset-lattice enumeration of linear
extensions.  This package adds the missing recombination oracle: for a chosen ordering,
walk backward from the target and re-apply each step's extracted retro template to the
substrate *as it then exists*, pruning orderings whose required context is absent.
"""

from . import deps  # noqa: F401  (sys.path bootstrap must run before synthesis_extraction imports)
