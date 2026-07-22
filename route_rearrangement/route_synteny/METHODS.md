# Ordering patterns in synthesis routes — goals, sources, and what has been settled

A working document for anyone continuing this line. It records **what question is being asked**,
**where every borrowed method came from**, **what has already been measured**, and **which
mistakes produce plausible-looking wrong answers**. The last section is the most valuable part:
each trap listed cost a debugging cycle and none of them announced itself.

---

## 1. The question, and how it got there

The project enumerates every chemically valid ordering of a literature route's steps. That
raises an obvious question — *which of those orderings are any good?* — and three framings were
tried before one survived.

**Framing 1 — nearest-precedent retrieval (rejected).** The starting idea, from the KNN-style
reaction-feasibility literature ([10.1021/acs.jcim.9b00313](https://doi.org/10.1021/acs.jcim.9b00313),
[10.1021/acscentsci.7b00355](https://doi.org/10.1021/acscentsci.7b00355)), was to score a
rearrangement by its nearest literature neighbour. It fails on its own terms: those methods
score a *single reaction* as a point query in molecule space, whereas a rearrangement is a
*relation between two steps*. They also carry no notion of significance, which was the actual
stated gap.

**Framing 2 — pairwise precedent as a score (built, then set aside).** Implemented in
[`../literature_precedent/`](../literature_precedent/). Reduces "is this ordering precedented"
to a countable pairwise event with a natural null (p = 0.5). The machinery is sound and still
in use, but it could not carry a *score*: 76–81% of transformation pairs have ≤1 ordered
observation, and the motif audit does not pass. Its lasting contributions are the abstraction
ladder and the calibration numbers in §4.

**Transformation identity: one step, one transformation.** Steps were originally keyed by the
FrequenTree *contextual centre* they belonged to. That was removed (2026-07-22) for a chemical
reason that outranks the statistical one: a contextual centre is a *connected group of reaction
centres*, so any ordering pattern whose steps act on **different** centres cannot be represented
at all. The motif this project most wants — install a nitro to activate a ring, run the SNAr,
reduce the nitro — is three different reaction centres, and was therefore structurally
inexpressible. Identity now comes from [`step_identity.py`](step_identity.py): the bonds each
step actually changes.

**Framing 3 — necessity versus convention (current).** The framing that works, because the
corpus can actually answer it:

> **How much of a published synthesis's step order is forced by chemistry, and how much is
> convention?**

Nobody has measured this. It also sidesteps the validation problem that sinks framings 1 and 2:
it needs no ground truth about which route is *better*.

### Why this question is answerable here and not in genomics

Comparative genomics asks the structurally identical question about gene order and can only
ever reach "conserved beyond chance", because nobody knows which gene orders are *physically
possible*. Conservation therefore gets read as selection, and the field cannot separate
"selected for" from "physically forced".

A synthesis route comes with a **known partial order of physical constraints** (atom lineage,
protecting-group brackets, functional-group exposure). That permits a second null model, and
the gap between the two nulls *is* the necessity/convention split. This is the one methodological
contribution over the source literature.

---

## 2. Sources, and exactly what was taken from each

### 2.1 Primary architectural template

**Jahn K, Winter S, Stoye J, Böcker S. "Statistics for approximate gene clusters."**
*BMC Bioinformatics* 14 (Suppl 15), S14 (2013).
[BMC](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-S15-S14) ·
[PMC3908651](https://pmc.ncbi.nlm.nih.gov/articles/PMC3908651/) ·
[PubMed 24564620](https://pubmed.ncbi.nlm.nih.gov/24564620/)

> ⚠️ Earlier drafts of this codebase attributed this paper to "Wittler and Stoye". That is
> **wrong** — Wittler is not an author. Corrected 2026-07-22. If you find the old attribution
> anywhere, fix it.

Taken, essentially unchanged, into [`clusters.py`](clusters.py):

| Component | Definition | Where it lives |
|---|---|---|
| Reference cluster | a family set `C` with `\|C\| ≥ s`, anchored to an exact occurrence in some genome | `candidate_clusters()` |
| δ-location | occurrence within symmetric set distance `D(C,C′) = \|C\C′\| + \|C′\C\| ≤ δ` | `set_distance()`, `delta_locations()` |
| Quorum `k′` | the cluster must recur in ≥ k′ genomes | `Cluster.quorum` |
| Null model | random string per genome, character probabilities ∝ observed frequency | `nulls.frequency_null_string()` |
| FDR | `p_i^FDR = p_i · (Σ_j n_j(n_j+1)/2) / i` | `significance.fdr_reference()` |

Chosen over the r-window model specifically because **δ tolerates variable gaps**, and the
canonical rigid block in synthesis — protect → react → deprotect — has a bracket of variable
width by construction. A model demanding exact contiguity would miss every real instance of the
one block we are most confident about.

### 2.2 The gene-family-size correction

**Raghupathy N, Durand D. "Gene Cluster Statistics with Gene Families."**
*Mol Biol Evol* 26(5):957–968 (2009). [doi:10.1093/molbev/msp002](https://doi.org/10.1093/molbev/msp002) ·
[PMC2668827](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2668827/)

Source of the r-window model, but its load-bearing contribution here is the warning that
*"failure to incorporate gene family size … results in overestimation of significance"*. Our
transformation-family sizes are at least as skewed as a genome's, so the frequency-preserving
null is mandatory and **must not be simplified away**.

### 2.3 The formal frame for reordering

**Bäckström C. "Computational Aspects of Reordering Plans."**
*JAIR* 9:99–137 (1998). [JAIR](https://jair.org/index.php/jair/article/view/10210) ·
[arXiv:1105.5441](https://arxiv.org/abs/1105.5441)

Not an analogy — this *is* the problem, formally. Establishes that optimal **deordering** is
polynomial for planning languages built on **producers, consumers and threats**, while optimal
**reordering** is NP-hard and hard to approximate. Our dependency graph is exactly such a
language (material edges = producer/consumer, protecting-group brackets = threats), which is why
the tractable work sits where it does.

### 2.4 Chess — one borrowed idea, one trap

The useful concept is **transposition**: opening theory is stored as a DAG of positions rather
than a tree of move sequences, because different move orders reach the same position (Zobrist
hashing / transposition tables). The synthesis analogue is collapsing orderings that produce the
same intermediate multiset — the honest denominator when counting "distinct" rearrangements.

**The trap:** "move ordering" in the chess literature almost always means *the order in which a
search explores moves to maximise alpha-beta pruning* (killer heuristic, history heuristic,
MVV-LVA). That is search efficiency, not chemistry, and it will dominate any literature search.

---

## 3. The two nulls — the core construction

Both are drawn through the same `ScheduleLattice` code path so that **nothing but the constraint
set differs**. If they differed in any other respect the decomposition would confound two
changes at once and be uninterpretable.

- **Null-P** — uniform random permutation of *this route's own steps*.
- **Null-C** — uniform random **linear extension** of this route's essential partial order.

`ScheduleLattice.sample()` weights each choice by the number of completions of the resulting
downset, making it a genuinely uniform sampler over linear extensions (verified by chi-square in
`tests/test_nulls.py`). `precedence_probabilities()` gives the same distribution's pairwise
marginals **exactly**, so pair-level statistics need no sampling at all — which recovers the
analytic p-values the source papers prefer.

**The decomposition:**

| Null-P | Null-C | verdict |
|---|---|---|
| not significant | — | no ordering preference |
| significant | not significant *and* constraints explain ≥50% | **necessity** |
| significant | significant | **convention** |
| significant | neither explained nor rejected | **inconclusive** (usually too few routes) |

Read "convention" carefully: it means the *modelled* constraints do not explain the ordering,
which includes any real chemistry the constraint model omits. **It is an upper bound, always.**

---

## 4. What has been measured (PaRoutes, 457,166 routes)

Numbers a future study should not have to rediscover.

**Corpus shape**
- 430,011 linear routes (94.1%); 69,534 linear with ≥4 steps; ≥6 steps is only 13.6% of those.
- 99.2% of linear ≥4-step routes have more than one valid ordering (median 24, mean 4,414).

**The constraint model is nearly empty**
- Median **1** material edge per ≥4-step route.
- Protecting-group brackets add constraints to **6.2%** of routes.
- Counterfactual FG-exposure edges add constraints to **0.00%** — the tier is inert on this
  corpus. Its agreement with the tier below is *not* robustness; it is the same test twice.

**Headline result — and note how strongly it depends on segmentation**

| transformation identity | ordered pairs | necessity | convention | convention bound |
|---|---|---|---|---|
| FrequenTree centres | 657 | 31 | 587 | ≤89.3% |
| per-step bond changes | 3,330 | 925 | 2,161 | **≤64.9%** |

The 89.3% figure was **biased upward by coarse segmentation**. A contextual centre merges steps
whose reaction centres share atoms — which are exactly the most tightly coupled, materially
forced pairs — so those pairs were absorbed into single units and never counted, leaving only
the loosely coupled ones to be measured. At step resolution 5.1x more pairs are testable,
necessity rises 30-fold, and the bound falls 24 points. Quote the step-resolution number, and
treat any centre-resolution ordering statistic as an upper bound on convention twice over.
- On ≥6-step routes only: **≤80.2%**, with brackets doing 4× more work. The convention fraction
  *falls* with route length — the most informative trend found, and the reason longer routes
  matter more than more routes.
- Dominant pattern: *amide coupling last* (e.g. 1481/1690 routes, constraints explaining 0%).

**Validation: the motif that motivated dropping centres is now the top result**

The strongest convention pair in the corpus is **nitration before SNAr** — 621/621 routes
(100%), constrained null expecting 75%, chemistry explaining 49%:

| | key | reading |
|---|---|---|
| first | `NcR0>1//N(O.O.O).cR(cR.cR)` | an N bearing three oxygens bonds to an aromatic carbon — nitration |
| then | `NRcR0>1//NR(CR.CR).cR(F.cR.cR)` | a cyclic amine bonds to a fluorine-bearing aromatic carbon — SNAr |

Under centre-based identity this pair did not exist, because the two steps act on different
reaction centres. It is the single best evidence that the identity change was correct.

**Evidence is far less independent than it looks**
- Effective sample (distinct route skeletons) / nominal routes: **median 0.17, min 0.01**.
- One pair drew 35% of its 1,690 routes from a single skeleton.
- This is the phylogenetic non-independence problem, and it is **not yet corrected** (see §6).

**Confounding by position** (from `literature_precedent`)
- A depth-only null reaches McFadden pseudo-R² **0.55–0.60** at template rungs. Over half of
  raw ordering signal is *stage*, not constraint. Never report raw asymmetry.

**Segmentation — the binding limit on per-route application**
- FrequenTree contextual centres resolve a 6.9-step route into **2.27 transformations** (32.9%
  step coverage), because a centre deliberately merges steps sharing reaction-centre atoms.
- **Radius sweep (settled, do not repeat):** radius 0, 1 and 2 give **2,405 centres over the
  same 1,000 routes — bit-identical**. Segmentation is pinned at 32.9% throughout; only the
  family alphabet grows (193 → 470 keys). Mechanism: in `FC_mtemplate.py`, `radius` reaches only
  `get_fragments_for_changed_atoms` **after** centres are formed. Radius controls template
  specificity, never segmentation, and raising it would *thin* evidence.
- Fix: [`step_identity.py`](step_identity.py) keys each step by the bonds it changes. Full
  corpus, ≥6-step routes: **5.05 transformations/route, 75.8% coverage, 10.97 orderable
  pairs/route (5.1x more)**. It names *fewer* steps (76.7% vs 99.4%) — ~23% register no bond
  change (unmapped reagents, salt forms, stereochemistry) — but yields far more usable
  transformations, because centres merge and per-step identity does not.

**The blocker migrated rather than vanished** (rigidity survey, 2,500 routes, ≥6 steps)

| | FrequenTree centres | per-step |
|---|---|---|
| no transformation identity | 86.8% | **49.5%** |
| pair absent from table | 2.7% | 21.3% |
| below evidence floor | 3.3% | 16.5% |
| answered | 7.2% | **12.8%** |
| conventional verdicts | 2.7% | **8.8%** |
| routes with a convention-breaking candidate | 31% | **54%** |
| search-space pruning (median / mean) | 0% / 17% | **33% / 38%** |

Answered pairs nearly doubled and pruning finally works, but the `unknown` bucket only fell
72.7% → 68.3%: the freed pairs moved into *absent from table* and *below evidence floor*,
because a ~4x finer alphabet thins per-pair evidence. Together those are now **37.8%** of
movable pairs — pairs that *have* an identity but lack evidence at this rung. That makes
multi-rung backoff the top priority rather than a nicety (§6).

---

## 5. Traps — every one of these produces a confident wrong answer

1. **SMARTS-on-SMARTS matching fails silently.** Both halves of a template are themselves
   SMARTS. `[CX3]`/`[NX3;H1,H2]` never match a query molecule (and can raise), and FrequenTree
   writes charge as `[N;+1;…]`, which RDKit does *not* expose via `GetFormalCharge`. A nitro
   pattern matched a hand-written test and missed every real template — labelling nothing, which
   reads as "the corpus lacks that chemistry". Use graph predicates over the parsed query
   (element, aromaticity, degree, connectivity), never substructure matching.
2. **"Group present on the left, absent on the right" mislabels every spectator.** A product
   template shows only the reaction centre, so *any* reaction on a Boc-protected substrate looks
   like a Boc removal. 21 significant pairs were mislabelled this way. Require the atom to be
   atom-mapped, i.e. in the reaction centre.
3. **Monte Carlo p-values cannot meet an interval-scaled FDR correction.** An MC p-value floors
   at `1/(draws+1)` while the correction universe runs to millions of candidate intervals — every
   cluster comes out insignificant regardless of data. Simulate only the per-route probability,
   then take the exact **Poisson-binomial** tail across routes.
4. **Necessity must be earned, not won by default.** "Failed to reject the constrained null" ≠
   "explained by it" — a pair seen in 24 routes fails to reach significance under *either* null.
   Require an explained fraction ≥ 50%; report the rest as inconclusive. This moved
   material-tier necessity from 21 to 14.
5. **Anchor a multi-step transformation at its formation step.** Otherwise its family repeats,
   "does A precede B" becomes ambiguous, and 78% of routes are discarded (fixed: 22% → 72%).
6. **Width bounds go vacuous on short routes.** At `|C| + δ + max_extra = 5` on 4–7 step routes
   the window is the whole synthesis and the null expects 198/200 routes "tight". Also δ must
   stay **below** `|C|`, or a pair needs just one of its two members.
7. **A bond change counts if *one* endpoint survives, not both.** Requiring both discards the
   informative half: an amide coupling registers only "C–N formed" (collapsing onto
   N-alkylation), and a nitro reduction — whose every changed bond runs to a departing oxygen —
   registers **nothing at all**.
8. **Routes are not independent draws.** See §4. Count skeletons, not routes.
9. **Cross-rung joins cannot use key strings.** Rungs name the same chemistry differently; join
   on recorded lineage. Joining on keys silently matches nothing above the coarsest rung, making
   every pair appear to resolve there.
10. **Co-location is not order.** A protecting-group bracket forces protect *before* deprotect
    but leaves other steps free to sit between them — so it barely moves a compactness statistic
    while pinning a precedence one completely. The ordering question needs `precedence.py`; the
    cluster machinery answers a different (also valid) question.

---

## 6. Open questions, in priority order

1. **Divergence correction — the biggest methodological gap.** Genomics does not accept a
   cluster because it appears in many genomes; it requires appearance in *divergent lineages*.
   The synthesis analogue is quorum over **dissimilar scaffolds/targets**, not raw route count.
   Until this exists, the p-values and counts are soft. (The *explained fractions* are robust —
   they are ratios of expectations, not significance claims — so prefer them.)
2. **Does the convention fraction keep falling with route length?** 89.3% → 80.2% from ≥4 to ≥6
   steps. If the trend continues at ≥8, the whole picture is length-dependent and the honest
   headline must be conditioned on length. This argues for a corpus with *longer* routes
   (journal/Reaxys) over merely *more* patent routes — Pistachio brings more of the same short
   chemistry and would not move this.
3. **Multi-rung backoff for the rigidity map — now the top *implementation* priority.**
   Measured: 37.8% of movable pairs have an identity but insufficient evidence at
   `centre_env`. Build precedence tables at `centre` and `bond_changes` too (227 and 224
   families vs ~800) and back off fine → coarse per pair. This is the direct, quantified
   remedy for the migration documented in §4.
4. **The convention list as a constraint-model debugger.** Rank convention-classified orderings
   by conservation strength with explained fraction ≈ 0: each is a *hypothesis of missing
   chemistry* a chemist can adjudicate in seconds. This is the cheapest path to fixing the failed
   motif audit and it improves everything downstream.
5. **CASP validation gate.** Before building anything on top: can the score separate literature
   routes from CASP routes *for the same target*? If not, it is not measuring realism. Note that
   filtering CASP output *toward* convention selects for patent-typical chemistry and discards
   novelty — the valuable query is the complement (allowed, feasible, unprecedented).
6. **Telescoping candidates.** Pairs both always-adjacent and always-ordered are one-pot
   suggestions — a process-chemistry question nobody is answering computationally.

---

## 7. Reproducing

```bash
PY=~/anaconda3/envs/trimmed-trees/bin/python
R=route_rearrangement/route_synteny/results

$PY -m route_rearrangement.route_synteny.corpus \
    --corpus ~/Downloads/paroutes_all/trees.jsonl --out $R/genomes.jsonl
$PY -m route_rearrangement.route_synteny.precedence --genomes $R/genomes.jsonl --out $R/precedence.json
$PY -m route_rearrangement.route_synteny.report     --precedence $R/precedence.json --genomes $R/genomes.jsonl
$PY -m route_rearrangement.route_synteny.rigidity   --genomes $R/genomes.jsonl --precedence $R/precedence.json --survey 2500
```

Everything runs in **`trimmed-trees`** and needs only `trees.jsonl` — no centres cache, no
`frequentree` env, no `rdcanon`. Timings on the full corpus: genomes ~27 min, precedence ~25 s,
survey ~2 s. Superseded centre-based artifacts are in `results/legacy_centers/` with a README
explaining why they are not reproducible.

**Read the constraint-budget table before quoting any number**, and never quote a convention
fraction without the top-convention list beside it — that list is the falsification test.
