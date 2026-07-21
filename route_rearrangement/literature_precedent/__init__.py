"""Literature-precedent statistics — is a rearrangement's step order precedented?

The route rearrangement engine produces orderings that are *materially* valid (atom
bookkeeping works) and pass the hard chemical vetoes in :mod:`..feasibility`.  This package
supplies the missing **extrinsic** signal: across a large corpus of literature routes, does
the chemistry community consistently run transformation A before transformation B, and is
that consistency statistically real or an artifact of how routes are shaped?

The unit of analysis is deliberately *not* a molecule or a single reaction — nearest-precedent
retrieval (KNN over fingerprints) answers "has a reaction like this been done", which says
nothing about **order** and carries no notion of significance.  Here the unit is the *ordered
pair of transformations*, which turns "is this rearrangement precedented" into a countable
event with a natural null (p = 0.5) and standard inference.

The mining half already exists upstream in
``synthesis_extraction.transformation`` (FrequenTree contextual-center extraction, the pair
order-evidence table, material-forcing confound control).  This package adds the two things
that stack lacks:

* :mod:`.ladder` — the abstraction hierarchy, exact template up to reaction class, so a pair
  too sparse to measure at template resolution degrades into a class-level statement instead
  of vanishing;
* :mod:`.aggregate` + :mod:`.significance` — per-pair inference with an exact binomial test,
  per-rung Benjamini-Hochberg FDR, **route-clustered** standard errors (observations from one
  route are not independent), and a stage-confound control that separates a genuine ordering
  constraint from "that transformation simply tends to happen early".

Nothing here writes into the rest of ``route_rearrangement``; the precedent signal is not yet
wired into any metric.
"""

from __future__ import annotations

__all__ = ["ladder", "aggregate", "significance"]
