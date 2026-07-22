"""How much of a published synthesis's step order is chemical necessity, and how much is convention?

This package answers that by transferring the architecture of **conserved gene cluster
statistics** onto synthesis routes.  Genomics faced the formally identical problem — genes that
sit together across many genomes may do so because selection holds them together, or because
they happened to land there — and solved it with an explicit cluster model, a null model of
random gene order, and analytic significance with FDR control.  The template here is the
*approximate common intervals* formulation of Jahn, Winter, Stoye and Böcker (2013), whose
δ-location distance tolerates variable gaps; that matters because the canonical rigid block in
synthesis, protect → react → deprotect, has a bracket of variable width.  See ``METHODS.md``
for the full provenance of every borrowed component.

The mapping::

    genome                  ->  one linear route, read as a string of transformation families
    gene                    ->  one step
    gene family (alphabet)  ->  the step's transformation, keyed by the bonds it changes
    reference cluster C     ->  a set of transformation families
    δ-location of C         ->  an approximate occurrence, D(C,C') = |C\\C'| + |C'\\C| <= δ
    quorum k'               ->  the cluster must recur in at least k' routes

**What synthesis has that genomics does not** is the reason this is more than a translation.
Gene cluster statistics can only test against random gene order, because no one knows which
gene orders are physically possible.  A synthesis route comes with a *known* partial order of
constraints — atom lineage, protection brackets, functional-group exposure — so there are two
nulls available, and the gap between them is the answer:

* **Null-0** (theirs): a random string with family probabilities matching observed frequency.
  Rejecting it means "this block is clustered".
* **Null-1** (ours): a uniform random linear extension of the route's own essential partial
  order.  Rejecting it means "this block is clustered *beyond what chemistry forces*".

So a cluster significant under Null-0 but not Null-1 is **necessity** — the partial order alone
explains it.  One significant under both is **convention** — chemists agree on an order that
chemistry leaves open.

One caveat governs how every number here must be read.  "Convention" is defined by what the
necessity model fails to explain, so it is an **upper bound**: it contains both real convention
and any chemistry the constraint model misses.  Measured on PaRoutes, adding protection
brackets and counterfactual exposure edges to bare atom lineage barely changes a route's
ordering freedom (median 12 valid orderings either way), so that residual is not small.  Every
report states the convention fraction as a bound and surfaces its top clusters for chemical
inspection, because a mechanism a chemist can name in that list is a finding about the model,
not a footnote.

Scope is **linear routes** (94.1% of PaRoutes), matching the linear-genome architecture of the
source papers.
"""

from __future__ import annotations

__all__ = ["corpus", "nulls", "clusters", "significance", "decompose"]

#: Necessity tiers, weakest to strongest — the sensitivity ladder every result is reported over.
#: Keys are the flags of ``dependency.analyze.dependency_graph_from_full_graph``, so the tiers
#: are the upstream constraint model rather than a re-derivation of it.
TIERS = {
    "material": dict(include_compatibility=False, include_counterfactual=False),
    "brackets": dict(include_compatibility=True, include_counterfactual=False),
    "exposure": dict(include_compatibility=True, include_counterfactual=True),
}
TIER_NAMES = list(TIERS)
