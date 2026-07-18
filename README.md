# route_rearrangement

Given **one literature synthesis route**, enumerate **every chemically valid ordering of its
reaction steps** and materialize each ordering — recompute all intermediates and starting
materials — by re-applying each step's retro reaction template backward from the target.

## How it works

1. **Partial order** (reused from `~/synthesis_extraction`): `build_route_graph` unifies atom
   maps across the route; `dependency_graph_from_full_graph` derives the *essential* precedence
   constraints — material edges (step B consumes atoms step A installed), protection brackets,
   counterfactual FG-exposure edges. `ScheduleLattice` enumerates the linear extensions =
   all candidate orderings.
2. **Backward materialization** (this package, the new "recombination oracle"): for each
   ordering, start at the target and undo the last-performed step first by applying its
   rdchiral retro template to the substrate *as it exists in the rearranged route*
   ([materialize.py](route_rearrangement/materialize.py)). The walk carries a **frontier** —
   the open intermediates not yet disconnected (one molecule for a linear route, several
   while convergent branches are open). Undoing a step deposits its precursors: building
   blocks are order-invariant, so they match the original step's side reactants exactly and
   leave the frontier; synthesized precursors stay open for a later-undone step
   ([chain.py](route_rearrangement/chain.py)). The new route's **tree topology is emergent**
   — a coupling undone late (performed early) hands the following steps the combined
   molecule, so functionalizations migrate onto it (*convergence-point migration*;
   `--no-migration` keeps every fragment fully assembled before its coupling). A template
   that no longer matches prunes the ordering — the built-in chemical veto; conservation
   invariants (the frontier can never exceed the steps left, and must end empty) prune
   structurally impossible walks.
3. **Calibration gate**: a route is only trusted if replaying the *original* order through the
   extracted templates reproduces the original intermediates **and the original tree** —
   products, synthesized-precursor sets, and child→parent edges (`replay_identity`). Routes
   that fail are skipped and counted.
4. **Filters** ([filters.py](route_rearrangement/filters.py)): hard gates — RDKit
   sanitization, connectivity (`propagate.disconnected_edges` on the rebuilt route tree),
   dedup. Soft flags — `fg_risk` (FG-survival verdicts on new intermediates against the
   steps on their path to the root — a parallel branch is a different flask),
   `inexact_side_match`, `migrated_steps` (steps running on a different substrate than the
   literature), `sm_mismatch` (starting-material multiset deviates from the original's).
5. **Engines**: `naive` materializes each lattice ordering independently;
   `dfs` (default, [search.py](route_rearrangement/search.py)) walks the trie of reversed
   orderings — shared backward suffixes are computed once, dead suffixes prune whole subtrees,
   and the original ordering is always the first leaf explored. Both produce the identical
   accepted set (tested); dfs measured 4.7–9.2× faster.

## Usage

Everything runs in the existing `trimmed-trees` conda env from this directory:

```bash
PY=~/anaconda3/envs/trimmed-trees/bin/python
CORPUS=~/synthesis_extraction/synthesis_extraction/data/slice_0-1000/trees.jsonl

# 0. run the WHOLE dataset in one job (enumerate → materialize → score for every route)
#    linear AND convergent trees (--linear-only restores the old skip; --no-migration keeps
#    fragments fully assembled before their couplings).
N1=~/synthesis_extraction/synthesis_extraction/data/paroutes/n1/trees.jsonl
$PY -m route_rearrangement.pipeline --corpus $N1 --out-dir results_n1/   # add --limit N to test
#    -> results_n1/{scored.jsonl, routes.jsonl, failures.jsonl, stats.jsonl, summary.json}
#    summary.json.counts reports linear vs convergent vs unmappable coverage + migration counts.

# ...or drive the stages individually:

# 1. rank routes by reordering modularity (most valid orderings / commutable pairs)
$PY -m route_rearrangement.select_examples --corpus $CORPUS --top 20 --out candidates.jsonl

# 2. enumerate + materialize the top candidates that pass the identity gate
$PY -m route_rearrangement.run --corpus $CORPUS --candidates candidates.jsonl \
    --take 3 --cap 500 --max-accepted 100 --out-dir results/

# or one specific route
$PY -m route_rearrangement.run --corpus $CORPUS --tree-id 262_38 --out-dir results/

# 3. score every enumeration with the eight metrics + cross-enumeration statistics
$PY -m route_rearrangement.score --corpus $CORPUS --routes results/routes.jsonl \
    --out-dir results/ [--plausibility]

# 4. browse a route's rearrangements next to the original (PyQt window)
$PY -m route_rearrangement.gui --routes results/scored.jsonl --tree-id 106_201 --sort exposure
#    ...or a display-free static HTML gallery:
$PY -m route_rearrangement.gui --routes results/scored.jsonl --tree-id 106_201 \
    --sort exposure --html results/gallery_106_201.html

# tests
$PY -m pytest route_rearrangement/tests/ -q
```

Outputs: `results/routes.jsonl` (accepted materialized routes — ordering, per-step new
reaction SMILES, starting materials, provenance, flags), `results/failures.jsonl` (pruned
orderings with the failing step/position/intermediate — the measure of template
over-specificity), `results/scored.jsonl` (each route + a `metrics` block), and
`results/stats.json` (per-original statistics).

If `synthesis_extraction` is not at `~/synthesis_extraction`, set `SYNTHESIS_EXTRACTION_PATH`.

## Route-ranking metrics

Rearrangements of the *same* linear route share target, building blocks and step count, so
invariant descriptors (step count, convergence, SA of the starting materials) do not
discriminate between them — every metric here is **order-sensitive**, and each exposes a
`score` where **higher is better** so percentiles and best/worst are uniform.

Every metric scores the route **as a whole** — its feasibility and in-lab applicability —
not the complexity of any one molecule in isolation. Most decision-relevant is **whole-route
feasibility / in-lab applicability** (`exposure`, `competing`, `isolability`,
`carried_complexity`): does the sequence hold together and can a chemist run it at the bench?
`treelstm`/`plausibility` add learned whole-route likeness.

| # | metric | family | source | what it measures |
|---|--------|--------|--------|------------------|
| 1 | `exposure` | route/lab | `synthesis_extraction.dependency.exposure` | bystander functional groups exposed to destroying conditions (protections the ordering forces) |
| 2 | `competing` | route/lab | rxnutils SMARTS | competing reactivity sites: reactive groups present but not reacting (selectivity liabilities), incl. leaving groups exposed to a condensation |
| 3 | `isolability` | route/lab | RDKit SMARTS (native) | **bench-handleability of the isolated intermediates**: unstable/hazardous groups (acyl halides, azides, peroxides, isocyanates, aldehydes, epoxides, boronic acids, …) that a given ordering forces you to isolate, purify and store |
| 4 | `carried_complexity` | route/lab | RDKit heavy-atom counts (native) | **"build complexity late"**: mass installed early is carried through — and risked by — every downstream step (`Σ max(Δheavy,0)·steps-remaining`); rewards convergent, late-stage assembly |
| 5 | `treelstm` | learned | [moyiming1 Tree-LSTM](https://github.com/moyiming1/Retrosynthesis-pathway-ranking) (cloned into `external/`) | learned whole-tree literature-likeness |
| 6 | `plausibility` | learned | miniASKCOS pistachio template-relevance (~900 MB model, opt-in `--plausibility`) | learned per-reaction plausibility: does a known template reproduce each rearranged reaction? |

`isolability` and `carried_complexity` are RDKit-only, so they are always available (no model
download) and never depend on the miniASKCOS/torch stack.

> **Removed:** the per-molecule complexity metrics `complexity` (SCScore) and `accessibility`
> (SAscore) scored each intermediate *in isolation* — a property of one structure, not of the
> route holding together — so they are no longer part of the workflow. Their modules remain in
> `route_rearrangement/metrics/` (unwired) should you want to re-enable them.

`score.py` reports, per original route and metric: the original ordering's value and its
**percentile among the rearrangements**, the best/worst rearrangement, how many
rearrangements beat the original, and the **Spearman agreement between metrics** across the
enumerations. Any metric whose model is missing is recorded `available: False` and skipped —
the pipeline never crashes on a missing model.

A metric's model location can be overridden by env var (`PATHWAY_RANKER_PATH`,
`SCSCORE_MODEL_PATH`, `TEMPLREL_MODEL_PATH`, `MINIASKCOS_PATH`).

## Route-to-route dissimilarity — the "most different" routes

Many rearrangements of one route are near-identical reorderings; to surface the genuinely
distinct alternatives, `score.py` also runs a **route-to-route dissimilarity** pass
([similarity.py](route_rearrangement/similarity.py)) built on MolecularAI
[`rxnutils`](https://github.com/MolecularAI/reaction_utils) tree-edit distance (TED, the
AiZynthFinder route metric), compared in `molecules` mode (structural overlap of the
intermediates — no atom mapping needed, since the rearranged routes are map-free). It picks
the **top-k routes that are far from the literature route *and* from each other** via greedy
farthest-first selection (seeded with the original), so the flagged set is a diverse spread,
not five variants of one idea. A cheap Jaccard prefilter over intermediate sets bounds the
exact-TED work to the most-promising pool (`--similarity-ted-cap`, default 60); if
`rxnutils`/`apted` are unavailable it falls back to that Jaccard distance and never crashes.

Each route gets a `similarity` block (`distance_to_original`, `rank_most_different`,
`diverse_rank`); `stats.json` gains a `most_different` list (the diverse top-k with distances
and metric scores). Flags: `--diverse-k` (default 5), `--similarity-ted-cap`, `--no-similarity`.

In the **GUI** this is the `distinct` sort key (the default when computed): the diverse top-k
appear first, badged **most-different #k** with their distance from the literature route, then
the remaining routes most-different-first. Show just the diverse set with:

```bash
$PY -m route_rearrangement.gui --routes results/scored.jsonl --tree-id 106_201 \
    --sort distinct --top 5 --html results/gallery_106_201.html
```

## GUI (borrowed from synthesis_extraction)

`route_rearrangement.gui` reuses `synthesis_extraction.gui`'s rendering core
(`pathway_renderer.render_pathway_png` → RDKit skeletal drawings laid out left-to-right by
Graphviz) to draw each ordering as a reaction scheme. The PyQt viewer stacks the **original
literature route above the current rearrangement**, with Prev/Next, a sort-by-metric
selector, and each route's metric scores + percentiles in the caption — so you can walk
the rearrangements best-first and eyeball each for chemical logic. `--html` writes the same
content as a self-contained static gallery (works without a display). The `dot` binary lives
in the conda env but off PATH; the viewer replicates the app's PATH shim automatically.

## Measured baselines (slice_0-1000, 2026-07)

- 1707 linear routes (3–10 steps) have ≥2 valid orderings; the most modular has 453,600
  orderings of 10 steps.
- Step-level retro-template identity: ~72% (rdchiral extraction on unified-map PaRoutes
  steps; failures mostly spuriously-mapped reagents/solvents).
- Route-level identity replay: ~62% of linear routes pass (long routes compound the
  per-step rate — 10-step routes rarely pass).
- Example run on the top three replay-passing modular routes (10 steps each):
  8 / 55 / 100 accepted rearranged routes; every accepted route preserves the original
  starting-material multiset, and the original ordering is always among the outputs.
- **Metric sanity check** (route 106_201, 7 rearrangements + original): the literature
  ordering ranks at the **86th percentile on plausibility, exposure, and accessibility** —
  i.e. the real route is genuinely near-best by chemistry-based metrics — but at the **0th
  percentile on Tree-LSTM**, which disagrees. Metrics partly anti-correlate
  (plausibility~accessibility Spearman ≈ −0.8), confirming they capture complementary axes;
  no single metric is sufficient, which is the point of computing all five.

### 50-route study (literature vs. rearrangements)

50 replay-passing literature routes, **1,519 total rearrangements** (`results_50/`).
Fraction of routes where at least one rearrangement outscores the literature ordering, and
how often the literature route is the single best of its enumerations:

| metric | routes with a better rearrangement | literature's mean percentile | literature is best |
|--------|-----------------------------------:|-----------------------------:|-------------------:|
| treelstm      | 44 / 50 | 0.62 |  6 / 50 |
| exposure      | 22 / 50 | 0.88 | 28 / 50 |
| complexity    |  9 / 50 | 0.95 | 41 / 50 |
| accessibility | 28 / 50 | 0.62 | 22 / 50 |

Reading: the literature ordering is genuinely strong on the **chemistry-based** metrics —
best-of-all on complexity trajectory 41/50 times (88th–95th percentile on exposure and
complexity), which validates both the enumeration and the metrics. The learned Tree-LSTM
literature-likeness disagrees: it prefers some rearrangement 44/50 times. On **9 of 50
routes a single rearrangement beats the literature on ≥3 of the 4 metrics at once** (e.g.
204_21, 106_192) — the strongest candidates for a genuinely better ordering; the rest of
the "wins" are single-metric and metric-dependent. Regenerate with:

```bash
$PY -m route_rearrangement.run   --corpus $CORPUS --candidates candidates.jsonl --take 50 \
    --cap 200 --max-accepted 40 --out-dir results_50/
$PY -m route_rearrangement.score --corpus $CORPUS --routes results_50/routes.jsonl --out-dir results_50/
$PY -m route_rearrangement.compare --stats results_50/stats.json
```

## Synthesis fragments (toward cross-pathway recombination)

`route_rearrangement.fragments` mines **cohesive fragments** — contiguous blocks of steps
that good orderings keep together in a fixed internal order — from a route's scored
enumeration. The motivation: a larger CASP search would harvest such fragments from many
targets' pathways and stitch the best of each into a new route better than any single source.

**Key point:** fragment cohesion is *not* implied by the dependency partial order. A material
edge A→B only forces A *before* B; it permits interleaving. So a "must-stay-together" block is
an **empirical** property mined from which orderings score well, not a structural one:

- for each ordered pair `(a,b)`, `good_freq` = how often `a` runs immediately before `b` among
  the top-scoring orderings, vs `all_freq` across all sampled orderings; the ratio is the
  adjacency **lift**;
- sticky adjacencies (high `good_freq`, lift > 1) are greedily chained into maximal fragments;
- a fragment is **hard** if every internal adjacency is also a material edge (the steps truly
  build on each other's atoms), else **soft** (good routes merely prefer them adjacent — the
  transferable heuristic). Each fragment carries its ordered **retro-template sequence** and
  example reactions — the unit you would graft into another synthesis.

```bash
$PY -m route_rearrangement.fragments --corpus $CORPUS --routes results_50/scored.jsonl \
    --out results_50/fragments.jsonl              # aggregate over all routes
$PY -m route_rearrangement.fragments --corpus $CORPUS --routes results_50/scored.jsonl \
    --tree-id 380_17                              # detail one route's fragments
```

**50-route result:** 45/50 routes yield ≥1 cohesive fragment; **44/50 have a soft (emergent,
transferable) multi-step fragment**; 55 soft vs only 4 hard fragments total (sizes 2–6 steps).
That soft ≫ hard ratio is the whole point — most cohesion is a learned tactical preference,
invisible to the partial order. Example (route 380_17, fragment `[6,4,5,3]`, lift 1.67): a
Sandmeyer chlorination → O-alkylation → acetyl deprotection → re-acylation that good orderings
keep intact though the partial order would let them split.

### Cross-pathway stitching — the next phase (designed, not yet built)

The single-route miner above is phase 1. The full vision needs:

1. **Multi-target corpus** of scored enumerations (run `run`+`score`+`fragments` over many
   trees) → a library of fragments keyed by their template sequence + FG entry/exit.
2. **Fragment abstraction** — represent a fragment by its retro-template sequence and the
   functional-group context it consumes/exposes (already extractable: templates + the
   `exposure`/FG machinery), so it is target-agnostic.
3. **Compatibility model** — a fragment B can follow fragment A iff B's entry FG matches A's
   exit and A's exit survives B's conditions (reuse `compat.fg_survives`).
4. **Assembly search** — beam/A* over fragment concatenations that build the target, scored by
   the same metrics, materialized and validated by the existing backward oracle.

Phase 1 supplies the fragments and their transferable representation; 2–4 are the recombination
engine. This is why within-route rearrangement rarely beats the literature (the human route is
near-optimal *for its own steps*) — the gains come from recombining fragments *across* routes.

## Scope and known limits (v1)

- **Linear routes only** — every step's non-chain reactants are purchasable building blocks.
  Convergent extension path: freeze side branches as fixed building blocks first, cross-branch
  reordering later.
- rdchiral templates are retro-specific — fidelity rests on the retro identity gate; forward
  checks would be unreliable and are not used as gates.
- Over-specific templates cause false pruning of valid reorderings (visible in
  `failures.jsonl`); a FrequenTree radius-0 re-extraction fallback is the planned mitigation.
- ≤20 steps (exact-DP lattice ceiling); practical target 3–10.
- `fg_risk` runs without a tolerance matrix by default (rules-only verdicts); pass a matrix
  to `process_route` for corpus-mined verdicts.
