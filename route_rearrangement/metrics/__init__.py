"""Route-ranking metrics for comparing enumerated orderings of one literature route.

All order-sensitive (invariant descriptors like step count and convergence do not
discriminate between rearrangements of the *same* linear route, so they are omitted).
Every metric scores the route **as a whole** — its feasibility and in-lab applicability —
rather than the complexity of any one molecule in isolation.  Two families:

*Whole-route feasibility / in-lab applicability* — does the sequence hold together and can a
chemist run it at the bench?
1. ``exposure``      — functional-group exposure oracle (synthesis_extraction): protections
   the ordering forces onto bystander groups.
2. ``competing``     — competing reactivity sites (rxnutils): reactive groups present but not
   reacting (selectivity liabilities), incl. leaving groups exposed to a condensation.
3. ``isolability``   — bench-handleability of the isolated intermediates: unstable/hazardous
   groups (acyl halides, azides, peroxides, …) an ordering forces you to isolate and store.
4. ``carried_complexity`` — "build complexity late": mass installed early is carried (and
   risked) through every downstream step; rewards convergent, late-stage assembly.

*Learned whole-route likeness*
5. ``treelstm``      — Tree-LSTM literature-likeness (Mo et al.), learned whole-tree.
6. ``plausibility``  — template-relevance per-reaction plausibility (miniASKCOS), learned.

The per-molecule complexity metrics (SCScore ``complexity`` and SAscore ``accessibility``)
have been removed from the workflow: they score each intermediate in isolation, which is not
a statement about whole-route feasibility.  The modules remain on disk but are not wired into
the suite or :data:`METRIC_NAMES`.

Each metric exposes a ``score`` value where **higher is better**, so percentile ranks and
best/worst comparisons are uniform across metrics.
"""

METRIC_NAMES = ["treelstm", "plausibility", "exposure",
                "competing", "isolability", "carried_complexity"]
