"""Order, not co-location — and here the null is exact.

The cluster machinery in :mod:`.clusters` asks whether two transformations sit *together*.  That
is the genomics question and it is worth asking, but it is not quite the question the project
set out to answer, and a positive control exposed the gap: a protecting-group bracket forces
protect **before** deprotect, yet leaves other steps free to be scheduled between them.  So the
bracket barely moves a compactness statistic while pinning a precedence one completely.  Order
and adjacency are different properties, and the ordering constraints of chemistry act on order.

So this module runs the same necessity/convention decomposition on precedence:

    for each ordered family pair (A, B), across routes containing both exactly once,
    T(A, B) = number of routes in which A is performed before B

The gain is that **no simulation is needed anywhere**.  Under Null-P (free permutation) the
probability that A precedes B is exactly ``1/2``.  Under Null-C it is
``ScheduleLattice.precedence_probabilities()[a, b]`` — computed exactly by DP over the lattice
of downsets, per route.  The corpus statistic is again a sum of independent Bernoulli trials
with known, route-specific probabilities, so its null distribution is Poisson-binomial and its
tail is exact.  This is the analytic p-value the source papers use, recovered in full rather
than approximated by sampling.

The decomposition reads the same way as everywhere else in this package:

* significant against Null-P, explained by Null-C  -> **necessity**
* significant against both                          -> **convention**
* not significant against Null-P                    -> no ordering preference

Routes where either family occurs more than once are skipped rather than approximated: "does A
precede B" has no single answer when two A's straddle B.  The count of skipped routes is
reported, not hidden.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .. import deps  # noqa: F401
from . import TIER_NAMES
from .corpus import UNKNOWN, Genome
from .decompose import CONVENTION, NECESSITY, NOT_A_CLUSTER, classify, load_genomes
from .nulls import lattice_for_genome
from .significance import benjamini_hochberg, poisson_binomial_sf

log = logging.getLogger(__name__)

NO_PREFERENCE = "no_preference"
INCONCLUSIVE = "inconclusive"

#: How much of the ordering preference the constraints must account for to call it necessity.
MIN_EXPLAINED = 0.5


def unique_family_positions(genome: Genome) -> Dict[str, int]:
    """``{family: step_id}`` for transformations with a single unambiguous ordering event.

    Every identified step is an ordering event, so the only exclusion is genuine ambiguity: a
    family occurring twice in one route gives "does A precede B" no single answer when two A's
    straddle B, and such families are dropped rather than guessed at.
    """
    pairs = [(f, s) for f, s in zip(genome.families, genome.step_ids) if f != UNKNOWN]
    counts = Counter(f for f, _ in pairs)
    return {f: s for f, s in pairs if counts[f] == 1}


@dataclass
class PairResult:
    family_a: str
    family_b: str
    n_routes: int
    n_effective: int                   # distinct route skeletons — the independent evidence
    observed: int                      # routes where A ran before B
    p_free: float
    q_free: float
    expected_constrained: Dict[str, float]
    explained: Dict[str, float]        # fraction of the preference the constraints account for
    p_constrained: Dict[str, float]
    q_constrained: Dict[str, float]
    verdict: Dict[str, str]
    n_routes_skipped_repeats: int


def collect(genomes: Sequence[Genome], tiers: Sequence[str] = TIER_NAMES,
            min_routes: int = 20) -> Tuple[List[dict], Dict[str, int]]:
    """Per ordered family pair: the observed count and the exact null probabilities.

    One pass builds, for every route, the exact precedence marginals at each tier; those feed
    every pair the route contributes.  Routes whose lattice exceeds upstream's exact-pairwise
    guard fall back to the tier's structural certainty where it is known and are otherwise
    dropped, counted under ``lattice_too_large``.
    """
    obs: Dict[Tuple[str, str], int] = defaultdict(int)
    pis: Dict[str, Dict[Tuple[str, str], List[float]]] = {t: defaultdict(list) for t in tiers}
    n_pairs: Dict[Tuple[str, str], int] = defaultdict(int)
    # Distinct route *skeletons* behind each pair — the effective sample size.  Routes are not
    # independent draws: a med-chem campaign publishes the same synthesis skeleton dozens of
    # times, and counting each as fresh evidence is the synthesis analogue of treating two
    # strains of one bacterium as independent confirmation of a gene cluster.  Measured on
    # PaRoutes, the nominal count overstates the independent evidence by roughly 10x.
    skeletons: Dict[Tuple[str, str], set] = defaultdict(set)
    skel_id: Dict[Tuple[str, ...], int] = {}
    stats: Counter = Counter()

    for g in genomes:
        pos = unique_family_positions(g)
        if len(pos) < 2:
            stats["routes_no_usable_pair"] += 1
            continue
        stats["routes_used"] += 1
        repeats = len({f for f in g.families if f != UNKNOWN}) - len(pos)
        stats["families_dropped_as_repeats"] += max(0, repeats)

        prec: Dict[str, Dict[Tuple[int, int], float]] = {}
        ok = True
        for t in tiers:
            try:
                prec[t] = lattice_for_genome(g, t).precedence_probabilities()
            except ValueError:
                stats["lattice_too_large"] += 1
                ok = False
                break
        if not ok:
            continue

        order = {s: i for i, s in enumerate(g.step_ids)}   # literature order
        skel = tuple(f for f in g.families if f != UNKNOWN)
        sid = skel_id.setdefault(skel, len(skel_id))
        fams = sorted(pos)
        for i, fa in enumerate(fams):
            for fb in fams[i + 1:]:
                sa, sb = pos[fa], pos[fb]
                key = (fa, fb)
                n_pairs[key] += 1
                skeletons[key].add(sid)
                if order[sa] < order[sb]:
                    obs[key] += 1
                for t in tiers:
                    pis[t][key].append(prec[t].get((sa, sb), 0.5))

    rows: List[dict] = []
    for key, n in n_pairs.items():
        if n < min_routes:
            stats["pairs_below_min_routes"] += 1
            continue
        rows.append({
            "family_a": key[0], "family_b": key[1], "n_routes": n, "observed": obs[key],
            "n_effective": len(skeletons[key]),
            "pis": {t: pis[t][key] for t in tiers},
        })
    stats["pairs_tested"] = len(rows)
    return rows, dict(stats)


def _two_sided(pis: Sequence[float], observed: int) -> float:
    """Two-sided Poisson-binomial p-value: either direction of ordering preference counts.

    Clamped to [0, 1]: ``1 - sf`` loses all its precision when ``sf`` is within float epsilon of
    1, and the subtraction can land just below zero, which then propagates through the FDR
    correction as a *negative* q-value.
    """
    upper = poisson_binomial_sf(pis, observed)
    lower = 1.0 - poisson_binomial_sf(pis, observed + 1)
    p = 2.0 * min(max(0.0, upper), max(0.0, lower))
    return float(min(1.0, max(0.0, p)))


def explained_fraction(observed: int, n_routes: int, expected_constrained: float) -> float:
    """How much of the literature's ordering preference the constraints actually account for.

    ``0`` = the constrained null predicts a coin flip, exactly like the free null, so chemistry
    explains none of the observed preference.  ``1`` = it predicts the observation outright.

    This exists because "not significant under the constrained null" is *not* the same as
    "explained by the constrained null".  A pair seen in 24 routes can fail to reach
    significance under either null simply for want of data, and calling that necessity would
    credit the chemistry model for the analysis's own lack of power.  Requiring a real
    explained fraction separates the two, and pairs that satisfy neither are reported as
    inconclusive rather than silently assigned.
    """
    free = n_routes / 2.0
    denom = observed - free
    if abs(denom) < 1e-12:
        return 1.0                    # nothing to explain: the literature had no preference
    return max(0.0, min(1.0, (expected_constrained - free) / denom))


def analyze(rows: Sequence[dict], tiers: Sequence[str] = TIER_NAMES,
            alpha: float = 0.05, skipped: int = 0) -> List[PairResult]:
    p_free = [_two_sided([0.5] * r["n_routes"], r["observed"]) for r in rows]
    p_con = {t: [_two_sided(r["pis"][t], r["observed"]) for r in rows] for t in tiers}
    # the multiple-testing universe is the family pairs actually tested, which is the whole
    # search here — unlike the cluster stage, no interval scan happens
    q_free = benjamini_hochberg(p_free)
    q_con = {t: benjamini_hochberg(p_con[t]) for t in tiers}

    out: List[PairResult] = []
    for i, r in enumerate(rows):
        exp = {t: float(sum(r["pis"][t])) for t in tiers}
        expl = {t: explained_fraction(r["observed"], r["n_routes"], exp[t]) for t in tiers}
        out.append(PairResult(
            family_a=r["family_a"], family_b=r["family_b"], n_routes=r["n_routes"],
            n_effective=r.get("n_effective", r["n_routes"]),
            observed=r["observed"], p_free=p_free[i], q_free=q_free[i],
            expected_constrained=exp, explained=expl,
            p_constrained={t: p_con[t][i] for t in tiers},
            q_constrained={t: q_con[t][i] for t in tiers},
            verdict={t: _verdict(q_free[i], q_con[t][i], expl[t], alpha) for t in tiers},
            n_routes_skipped_repeats=skipped))
    return out


def _verdict(q_free: float, q_con: float, explained: float, alpha: float) -> str:
    """Necessity must be *earned* by an explained fraction, not won by default.

    Failing to reject the constrained null is only evidence of explanation when the constrained
    null actually moved toward the observation; otherwise it is a statement about sample size.
    """
    if classify(q_free, q_con, alpha) == NOT_A_CLUSTER:
        return NO_PREFERENCE
    if q_con <= alpha:
        return CONVENTION if explained < MIN_EXPLAINED else INCONCLUSIVE
    return NECESSITY if explained >= MIN_EXPLAINED else INCONCLUSIVE


def headline(results: Sequence[PairResult], tiers: Sequence[str] = TIER_NAMES) -> dict:
    out = {"n_pairs": len(results), "tiers": {}}
    for t in tiers:
        c = Counter(r.verdict[t] for r in results)
        real = c[NECESSITY] + c[CONVENTION] + c[INCONCLUSIVE]
        out["tiers"][t] = {
            "n_ordered_pairs": real, "necessity": c[NECESSITY], "convention": c[CONVENTION],
            "inconclusive": c[INCONCLUSIVE], "no_preference": c[NO_PREFERENCE],
            "convention_fraction_upper_bound": (c[CONVENTION] / real) if real else None,
        }
    return out


def format_headline(h: dict, tiers: Sequence[str] = TIER_NAMES) -> str:
    lines = [f"{h['n_pairs']:,} transformation pairs with an orderable relationship", "",
             f"{'necessity tier':>14}{'ordered':>10}{'necessity':>11}{'convention':>12}"
             f"{'inconcl.':>10}{'conv. frac':>12}", "-" * 69]
    for t in tiers:
        s = h["tiers"][t]
        frac = s["convention_fraction_upper_bound"]
        lines.append(f"{t:>14}{s['n_ordered_pairs']:>10,}{s['necessity']:>11,}"
                     f"{s['convention']:>12,}{s.get('inconclusive', 0):>10,}"
                     f"{(f'<= {frac:.1%}' if frac is not None else '-'):>12}")
    lines += [
        "",
        "'inconcl.' = the constrained null neither explained the ordering nor was rejected —",
        "usually too few routes to tell. Necessity requires the constraints to actually account",
        f"for >= {MIN_EXPLAINED:.0%} of the preference, not merely to escape rejection.",
        "",
        "Read 'conv. frac' as an UPPER BOUND. A pair is convention when the modelled",
        "constraints leave its order free yet the literature fixes it anyway — which includes",
        "every real constraint the dependency model does not represent. The constraint-budget",
        "table says how much room that leaves.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--genomes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-steps", type=int, default=4)
    ap.add_argument("--min-routes", type=int, default=20,
                    help="a pair must co-occur in this many routes to be tested")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    genomes = load_genomes(args.genomes, min_steps=args.min_steps, limit=args.limit)
    log.warning("loaded %d genomes", len(genomes))
    rows, stats = collect(genomes, min_routes=args.min_routes)
    results = analyze(rows, alpha=args.alpha)
    h = headline(results)
    h["n_genomes"] = len(genomes)
    h["collection_stats"] = stats

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"headline": h, "pairs": [asdict(r) for r in results]}, fh)

    print(format_headline(h))
    print()
    print("collection: " + ", ".join(f"{k}={v:,}" for k, v in sorted(stats.items())))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
