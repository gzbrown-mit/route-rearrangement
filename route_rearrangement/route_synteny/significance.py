"""Is a cluster tighter than chance — and than chemistry?

The statistic is deliberately about **compactness**, because that is what the permutation nulls
can move.  Null-P and Null-C permute a route's steps, so which families a route contains never
changes; only how spread out they are does.  So for each route we take the narrowest interval
containing enough of the cluster's families, and count the routes where that interval is tight:

    T(C) = #{routes whose narrowest window holding >= |C| - delta of C's families
             is no wider than |C| + delta + max_extra}

``T`` is exactly the quantity the reference-cluster model already uses for detection, so
detection and testing measure the same thing rather than two loosely related things.

**How the p-value is computed, and why it is not a plain Monte Carlo count.**  The source papers
compute p-values by dynamic programming; that derivation is specific to their i.i.d.
random-string null and has no analogue for "uniform over the linear extensions of a partial
order", which is the null carrying the scientific content here.  The obvious substitute — the
fraction of random draws reaching the observed statistic — cannot work, because a Monte Carlo
p-value is floored at ``1/(draws+1)`` while the multiple-testing universe runs to millions of
candidate intervals.  No affordable number of draws could ever clear that correction, and every
cluster would be declared insignificant regardless of the data.

So the two are combined, which also restores the papers' dynamic programming.  Simulation is
used only for what has no closed form: the per-route probability ``π_r`` that route *r* holds
the cluster tightly under the null.  The corpus statistic is then a sum of independent
Bernoulli trials across routes — routes are independent — so its null distribution is
**Poisson-binomial**, and its tail is computed exactly by DP over the routes
(:func:`poisson_binomial_sf`).  P-values reach 1e-300 rather than bottoming out at 1/201, and
the correction becomes meaningful again.

``π_r`` is Jeffreys-smoothed, ``(hits + 0.5) / (draws + 1)``: an unsmoothed ``π_r = 0`` from a
route that simply never came up tight in 200 draws would drive the tail to exactly zero and
manufacture certainty the simulation does not support.

The analytic route survives untouched where it was already available: for **pairs**,
``ScheduleLattice.precedence_probabilities`` gives exact Null-C precedence marginals with no
sampling at all (:func:`exact_pair_precedence`).

Multiple testing follows the source correction, ``p_i^FDR = p_i * (total candidate intervals) /
i`` for the *i*-th smallest p-value, with Benjamini-Hochberg reported alongside it — the two
differ only in the multiple-testing universe, and stating both makes that choice visible rather
than buried.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .. import deps  # noqa: F401
from .clusters import Cluster, FamilySet, required_members
from .corpus import UNKNOWN, Genome
from .nulls import lattice_for_genome, read_families

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------
def min_window(families: Sequence[str], c: FamilySet, need: int) -> Optional[int]:
    """Width of the narrowest interval containing at least *need* distinct members of *c*.

    Two-pointer sweep, linear in the route length.  This is the hot path — it runs once per
    cluster per route per Monte Carlo draw — so it avoids the nested window enumeration that
    :func:`clusters.delta_locations` uses when it also needs to report *where* the occurrence is.
    """
    if need <= 0:
        return 0
    n = len(families)
    have: Dict[str, int] = {}
    distinct = 0
    best: Optional[int] = None
    left = 0
    for right in range(n):
        f = families[right]
        if f in c and f != UNKNOWN:
            have[f] = have.get(f, 0) + 1
            if have[f] == 1:
                distinct += 1
        while distinct >= need:
            width = right - left + 1
            if best is None or width < best:
                best = width
            g = families[left]
            if g in c and g != UNKNOWN:
                have[g] -= 1
                if have[g] == 0:
                    distinct -= 1
            left += 1
    return best


def is_tight(families: Sequence[str], c: FamilySet, delta: int, max_extra: int) -> bool:
    """Does this route hold the cluster in a window narrow enough to count as a δ-location?

    The width bound has to stay well under a typical route length or the test is vacuous: at
    ``|C| + delta + max_extra = 5`` on 4–7 step routes, the window is the whole synthesis and
    every route passes, which is exactly what a first run produced (null expected 198 of 200
    routes tight).  Keep ``max_extra`` small.
    """
    w = min_window(families, c, required_members(c, delta))
    return w is not None and w <= len(c) + delta + max_extra


def corpus_statistic(c: FamilySet, strings: Sequence[Sequence[str]], delta: int,
                     max_extra: int) -> int:
    return sum(1 for s in strings if is_tight(s, c, delta, max_extra))


# ---------------------------------------------------------------------------
# Monte Carlo over the permutation nulls
# ---------------------------------------------------------------------------
def poisson_binomial_sf(pis: Sequence[float], k: int) -> float:
    """``P(X >= k)`` where ``X = sum of independent Bernoulli(pi_r)`` — exact, by DP.

    The corpus statistic counts routes that hold the cluster tightly, and routes are
    independent, so this is its exact null distribution once the per-route probabilities are
    known.  O(n^2) over the routes, which is nothing at n <= a few hundred.
    """
    if k <= 0:
        return 1.0
    n = len(pis)
    if k > n:
        return 0.0
    pmf = [1.0] + [0.0] * n
    for i, p in enumerate(pis):
        p = min(1.0, max(0.0, p))
        for j in range(i + 1, 0, -1):
            pmf[j] = pmf[j] * (1.0 - p) + pmf[j - 1] * p
        pmf[0] *= (1.0 - p)
    return float(sum(pmf[k:]))


@dataclass
class NullResult:
    null: str                 # "P" or a tier name for Null-C
    observed: int
    draws: int
    p_value: float            # Poisson-binomial tail at the observed statistic
    mean_null: float          # expected routes tight under the null, = sum of pi_r
    routes_tested: int
    routes_capped: int        # candidate routes dropped by the cap (0 = none)


def _candidate_routes(c: FamilySet, genomes: Sequence[Genome], delta: int,
                      max_routes: int) -> Tuple[List[Genome], int]:
    """Routes that could hold the cluster at all.

    Family *presence* is invariant under both permutation nulls, so this set is fixed across
    draws and is computed once.  Capping is a cost control; the number dropped is returned so
    the report can state it rather than silently presenting a truncated corpus as the whole one.
    """
    need = required_members(c, delta)
    hits = [g for g in genomes if len(c & set(g.families)) >= need]
    if max_routes and len(hits) > max_routes:
        # a deterministic *random* subsample, not a prefix: routes arrive in corpus order, so
        # taking the first N would systematically sample one region of the corpus
        rng = random.Random(len(hits) * 1_000_003 + len(c))
        return rng.sample(hits, max_routes), len(hits) - max_routes
    return hits, 0


def null_distribution(clusters: Sequence[Cluster], genomes: Sequence[Genome], *,
                      null: str, delta: int = 1, max_extra: int = 2, draws: int = 200,
                      seed: int = 0, max_routes: int = 200) -> Dict[FamilySet, NullResult]:
    """Monte Carlo p-values for every cluster under one null.

    The loop is **draw-major**: each draw permutes every route once and all clusters are scored
    against that same permuted corpus.  Cluster-major would re-permute the corpus per cluster
    and cost a factor of ``len(clusters)`` more for no statistical gain, since the draws are
    independent of which cluster is being tested.

    *null* is ``"P"`` for the free permutation or a necessity tier name for the constrained one.
    """
    tier = None if null == "P" else null
    cand: Dict[FamilySet, Tuple[List[Genome], int]] = {
        cl.families: _candidate_routes(cl.families, genomes, delta, max_routes)
        for cl in clusters}

    observed = {cl.families: corpus_statistic(
        cl.families, [g.families for g in cand[cl.families][0]], delta, max_extra)
        for cl in clusters}
    # per-route tightness counts: the only quantity simulation is used for
    hits: Dict[FamilySet, List[int]] = {cl.families: [0] * len(cand[cl.families][0])
                                        for cl in clusters}

    # every route that any cluster cares about, permuted once per draw
    involved = {g.route_id: g for _, (gs, _) in cand.items() for g in gs}
    lattices = {rid: lattice_for_genome(g, tier) for rid, g in involved.items()}
    # a stable per-route offset: str.__hash__ is salted per process, so using it here would make
    # every run irreproducible while looking deterministic
    offsets = {rid: i for i, rid in enumerate(sorted(involved))}

    for d in range(draws):
        permuted: Dict[str, List[str]] = {}
        for rid, g in involved.items():
            order = lattices[rid].sample(seed=seed * 7_919 + d * 1_000_003 + offsets[rid])
            permuted[rid] = read_families(g, order) if order else list(g.families)
        for cl in clusters:
            gs, _ = cand[cl.families]
            row = hits[cl.families]
            for i, g in enumerate(gs):
                if is_tight(permuted[g.route_id], cl.families, delta, max_extra):
                    row[i] += 1
        if draws >= 20 and (d + 1) % max(1, draws // 5) == 0:
            log.warning("null %s: draw %d/%d", null, d + 1, draws)

    out: Dict[FamilySet, NullResult] = {}
    for cl in clusters:
        fs = cl.families
        gs, capped = cand[fs]
        pis = [(h + 0.5) / (draws + 1.0) for h in hits[fs]]     # Jeffreys-smoothed
        out[fs] = NullResult(
            null=null, observed=observed[fs], draws=draws,
            p_value=poisson_binomial_sf(pis, observed[fs]),
            mean_null=float(sum(pis)),
            routes_tested=len(gs), routes_capped=capped)
    return out


def exact_pair_precedence(genome: Genome, tier: Optional[str]) -> Dict[Tuple[int, int], float]:
    """Exact ``P(a before b)`` under Null-C for one route — no sampling.

    The analytic path the source papers prefer, available here for pairwise questions.  Returns
    ``{}`` when the route's lattice exceeds upstream's exact-pairwise guard.
    """
    try:
        return lattice_for_genome(genome, tier).precedence_probabilities()
    except ValueError:
        return {}


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------
def n_candidate_intervals(genomes: Sequence[Genome]) -> int:
    """``sum_j n_j (n_j + 1) / 2`` — the source correction's multiple-testing universe."""
    return sum(len(g.families) * (len(g.families) + 1) // 2 for g in genomes)


def fdr_reference(pvals: Sequence[float], n_intervals: int) -> List[float]:
    """The source paper's correction: ``p_i^FDR = p_i * n_intervals / i``, made monotone."""
    n = len(pvals)
    if not n:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        val = min(1.0, pvals[i] * n_intervals / (rank + 1))
        running = min(running, val)
        adj[i] = running
    return adj


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """Standard BH, whose universe is the tests actually run rather than all intervals."""
    n = len(pvals)
    if not n:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        running = min(running, min(1.0, pvals[i] * n / (rank + 1)))
        adj[i] = running
    return adj
