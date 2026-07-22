"""Necessity or convention — the classification, and the CLI that produces the headline.

For each cluster we hold two p-values that differ in exactly one respect: whether the route's
chemical ordering constraints were imposed on the null.  The verdict follows directly:

===========================  ==========================  ====================================
Null-P (free permutation)    Null-C (linear extensions)  verdict
===========================  ==========================  ====================================
not significant              --                          **not a cluster** — no tighter than chance
significant                  not significant             **necessity** — the partial order explains it
significant                  significant                 **convention** — tight beyond what chemistry forces
===========================  ==========================  ====================================

Read the convention row carefully.  It says the *modelled* constraints do not explain the
block, which is not the same as saying nothing does.  Any real constraint the dependency model
misses lands here, so the convention count is an **upper bound** and is reported as one.  On
PaRoutes this matters more than it might: adding protection brackets and counterfactual
exposure to bare atom lineage barely changes ordering freedom, so the unmodelled residual is
not a rounding error.  :func:`format_headline` therefore prints the bound with its caveat
attached, and ``report.py`` lists the strongest convention clusters for a chemist to read —
a block in that list with an obvious mechanism is evidence about the constraint model, and
saying so is the point of running the ladder at all.

Usage::

    python -m route_rearrangement.route_synteny.decompose \\
        --genomes results/genomes.jsonl --out results/decomposition.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .. import deps  # noqa: F401
from . import TIER_NAMES
from .clusters import Cluster, deduplicate, find_clusters
from .corpus import Genome
from .significance import (NullResult, fdr_reference, n_candidate_intervals,
                           null_distribution)

log = logging.getLogger(__name__)

NOT_A_CLUSTER = "not_a_cluster"
NECESSITY = "necessity"
CONVENTION = "convention"


@dataclass
class Verdict:
    families: List[str]
    size: int
    quorum: int
    reference_route: str
    observed: int
    p_free: float
    q_free: float
    p_constrained: Dict[str, float]      # tier -> p under Null-C
    q_constrained: Dict[str, float]
    verdict: Dict[str, str]              # tier -> necessity/convention/not_a_cluster
    mean_null_free: float
    mean_null_constrained: Dict[str, float]
    routes_tested: int
    routes_capped: int


def classify(q_free: float, q_constrained: float, alpha: float = 0.05) -> str:
    if q_free > alpha:
        return NOT_A_CLUSTER
    return CONVENTION if q_constrained <= alpha else NECESSITY


def decompose(clusters: Sequence[Cluster], genomes: Sequence[Genome], *,
              delta: int = 1, max_extra: int = 2, draws: int = 200, seed: int = 0,
              max_routes: int = 200, alpha: float = 0.05,
              tiers: Sequence[str] = TIER_NAMES) -> List[Verdict]:
    """Run both nulls over every cluster and classify."""
    free = null_distribution(clusters, genomes, null="P", delta=delta, max_extra=max_extra,
                             draws=draws, seed=seed, max_routes=max_routes)
    constrained: Dict[str, Dict] = {
        tier: null_distribution(clusters, genomes, null=tier, delta=delta, max_extra=max_extra,
                                draws=draws, seed=seed + 1, max_routes=max_routes)
        for tier in tiers}

    n_intervals = n_candidate_intervals(genomes)
    p_free = [free[c.families].p_value for c in clusters]
    q_free = fdr_reference(p_free, n_intervals)
    q_con = {tier: fdr_reference([constrained[tier][c.families].p_value for c in clusters],
                                 n_intervals)
             for tier in tiers}

    out: List[Verdict] = []
    for i, c in enumerate(clusters):
        r: NullResult = free[c.families]
        out.append(Verdict(
            families=sorted(c.families), size=c.size, quorum=c.quorum,
            reference_route=c.reference_route, observed=r.observed,
            p_free=r.p_value, q_free=q_free[i],
            p_constrained={t: constrained[t][c.families].p_value for t in tiers},
            q_constrained={t: q_con[t][i] for t in tiers},
            verdict={t: classify(q_free[i], q_con[t][i], alpha) for t in tiers},
            mean_null_free=r.mean_null,
            mean_null_constrained={t: constrained[t][c.families].mean_null for t in tiers},
            routes_tested=r.routes_tested, routes_capped=r.routes_capped))
    return out


def headline(verdicts: Sequence[Verdict], tiers: Sequence[str] = TIER_NAMES) -> dict:
    """The sensitivity ladder: the necessity/convention split at each tier."""
    out = {"n_clusters": len(verdicts), "tiers": {}}
    for t in tiers:
        counts = Counter(v.verdict[t] for v in verdicts)
        real = counts[NECESSITY] + counts[CONVENTION]
        out["tiers"][t] = {
            "n_significant_clusters": real,
            "necessity": counts[NECESSITY],
            "convention": counts[CONVENTION],
            "not_a_cluster": counts[NOT_A_CLUSTER],
            "convention_fraction_upper_bound": (counts[CONVENTION] / real) if real else None,
        }
    return out


def format_headline(h: dict, tiers: Sequence[str] = TIER_NAMES) -> str:
    lines = [f"{h['n_clusters']:,} candidate clusters tested", "",
             f"{'necessity tier':>14}{'clusters':>10}{'necessity':>11}{'convention':>12}"
             f"{'conv. frac':>12}",
             "-" * 59]
    for t in tiers:
        s = h["tiers"][t]
        frac = s["convention_fraction_upper_bound"]
        lines.append(f"{t:>14}{s['n_significant_clusters']:>10,}{s['necessity']:>11,}"
                     f"{s['convention']:>12,}"
                     f"{(f'<= {frac:.1%}' if frac is not None else '-'):>12}")
    lines += [
        "",
        "'conv. frac' is an UPPER BOUND on convention, not an estimate. A cluster counts as",
        "convention when the modelled constraints fail to explain its tightness, so any real",
        "chemistry the dependency model misses is counted here too. Read it together with the",
        "top convention clusters in the report: a block there with a nameable mechanism is",
        "evidence the constraint model is under-powered, not that chemists are arbitrary.",
    ]
    return "\n".join(lines)


def load_genomes(path: str, *, min_steps: int = 0, limit: int = 0) -> List[Genome]:
    out: List[Genome] = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            g = Genome.from_dict(json.loads(line))
            if g.n_steps < min_steps:
                continue
            out.append(g)
            if limit and len(out) >= limit:
                break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--genomes", required=True, help="genomes.jsonl from route_synteny.corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap routes loaded (smoke tests)")
    ap.add_argument("--min-steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=2, help="minimum cluster size s")
    ap.add_argument("--max-size", type=int, default=3)
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--max-extra", type=int, default=2)
    ap.add_argument("--quorum", type=int, default=5, help="k': routes a cluster must recur in")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--max-routes", type=int, default=200)
    ap.add_argument("--max-clusters", type=int, default=0,
                    help="test only the top-N clusters by quorum (0 = all); the number dropped "
                         "is reported, never silently discarded")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    genomes = load_genomes(args.genomes, min_steps=args.min_steps, limit=args.limit)
    log.warning("loaded %d genomes", len(genomes))

    clusters = deduplicate(find_clusters(
        genomes, s=args.size, max_size=args.max_size, delta=args.delta, quorum=args.quorum))
    dropped = 0
    if args.max_clusters and len(clusters) > args.max_clusters:
        clusters.sort(key=lambda c: -c.quorum)
        dropped = len(clusters) - args.max_clusters
        clusters = clusters[:args.max_clusters]
    log.warning("%d clusters after dedup (%d dropped by --max-clusters)", len(clusters), dropped)

    verdicts = decompose(clusters, genomes, delta=args.delta, max_extra=args.max_extra,
                         draws=args.draws, seed=args.seed, max_routes=args.max_routes,
                         alpha=args.alpha)
    h = headline(verdicts)
    h["clusters_dropped_by_cap"] = dropped
    h["n_genomes"] = len(genomes)
    h["params"] = {k: getattr(args, k) for k in
                   ("size", "max_size", "delta", "max_extra", "quorum", "draws",
                    "max_routes", "alpha", "min_steps", "seed")}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"headline": h, "verdicts": [asdict(v) for v in verdicts]}, fh)

    print(format_headline(h))
    if dropped:
        print(f"\nNOTE: {dropped:,} clusters were not tested (--max-clusters).")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
