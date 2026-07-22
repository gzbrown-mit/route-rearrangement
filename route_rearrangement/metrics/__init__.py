"""Route-ranking metrics for comparing enumerated orderings of one literature route.

All order-sensitive (invariant descriptors like step count and convergence do not
discriminate between rearrangements of the *same* linear route, so they are omitted).
Every metric scores the route **as a whole** — its feasibility and in-lab applicability —
rather than the complexity of any one molecule in isolation.  Two families:

*Whole-route feasibility / in-lab applicability* — does the sequence hold together and can a
chemist run it at the bench?
1. ``exposure``      — functional-group exposure oracle (synthesis_extraction): protections
   the ordering forces onto bystander groups.
2. ``selectivity``   — computed electronic structure (condensed Fukui indices from RDKit's
   extended-Hückel): is the site each step aims at the most reactive copy of that
   transformation on its substrate, or is an unmasked rival copy just as hot?  Feature-based
   throughout — the reacting site comes from the step's own template and the rivals from
   where that template also matches, so no functional-group vocabulary is consulted.
3. ``isolability``   — bench-handleability of the isolated intermediates: unstable/hazardous
   groups (acyl halides, azides, peroxides, …) an ordering forces you to isolate and store.
4. ``carried_complexity`` — "build complexity late": mass installed early is carried (and
   risked) through every downstream step; rewards convergent, late-stage assembly.

*Learned whole-route likeness*
5. ``treelstm``      — Tree-LSTM literature-likeness (Mo et al.), learned whole-tree.
6. ``plausibility``  — template-relevance per-reaction plausibility (miniASKCOS), learned.

The per-molecule complexity metrics (SCScore ``complexity`` and SAscore ``accessibility``)
have been removed from the workflow: they score each intermediate in isolation, which is not
a statement about whole-route feasibility.  ``competing`` — the SMARTS-library predecessor of
``selectivity`` — is likewise unwired: it asked whether a *named* sensitive group was present
rather than whether the intended site was electronically preferred, which made it an
extracted-rule metric in a tier that should be feature-based.  All three modules remain on
disk (``competing`` still backs a GUI label fallback) but are not in :data:`METRIC_NAMES`.

Each metric exposes a ``score`` value where **higher is better**, so percentile ranks and
best/worst comparisons are uniform across metrics.
"""

METRIC_NAMES = ["treelstm", "plausibility", "exposure",
                "selectivity", "isolability", "carried_complexity"]
