"""Aggregate the per-route statistics: do any rearrangements beat the literature route?

Reads ``stats.json`` (written by :mod:`.score`) and answers, across all originals and for
each metric: on how many routes does at least one rearrangement outscore the literature
ordering, the distribution of the literature route's percentile, and the biggest single
improvement found.  Prints a table and (optionally) writes a JSON summary.

Usage::

    python -m route_rearrangement.compare --stats results_50/stats.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from .metrics import METRIC_NAMES


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def aggregate(stats: dict) -> dict:
    trees = stats.get("trees", [])
    per_metric: Dict[str, dict] = {}
    for name in METRIC_NAMES:
        rows = [t["metrics"][name] for t in trees
                if name in t.get("metrics", {}) and "original_value" in t["metrics"][name]]
        if not rows:
            continue
        n = len(rows)
        beaten = [r for r in rows if r.get("rearrangements_beating_original", 0) > 0]
        pctiles = [r["original_percentile"] for r in rows
                   if r.get("original_percentile") is not None]
        # biggest improvement: best rearrangement value minus original value
        improvements = []
        for r in rows:
            best = r.get("best", {}).get("value")
            if best is not None and r.get("original_value") is not None:
                improvements.append((best - r["original_value"], r))
        improvements.sort(key=lambda x: -x[0])
        top = improvements[0] if improvements else None
        per_metric[name] = {
            "n_routes": n,
            "routes_with_a_better_rearrangement": len(beaten),
            "frac_routes_beaten": round(len(beaten) / n, 3),
            "mean_original_percentile": round(_mean(pctiles), 3) if pctiles else None,
            "n_routes_original_is_best": sum(1 for r in rows if r.get("original_is_best")),
            "biggest_improvement": (
                {"delta": round(top[0], 4), "tree_id": top[1].get("tree_id"),
                 "original": top[1]["original_value"], "best": top[1]["best"]["value"],
                 "best_ordering": top[1]["best"]["ordering"]}
                if top else None),
        }
    return {"n_trees": len(trees), "per_metric": per_metric}


def _attach_tree_ids(stats: dict) -> None:
    for t in stats.get("trees", []):
        for m in t.get("metrics", {}).values():
            m["tree_id"] = t["tree_id"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stats", default="results_50/stats.json")
    ap.add_argument("--out", default="", help="write the aggregate JSON here")
    args = ap.parse_args(argv)

    with open(args.stats) as fh:
        stats = json.load(fh)
    _attach_tree_ids(stats)
    agg = aggregate(stats)

    print(f"Aggregate over {agg['n_trees']} literature routes "
          f"(availability: {stats.get('availability', {})})\n")
    header = (f"{'metric':<14}{'routes':>7}{'≥1 better':>11}{'frac':>7}"
              f"{'orig pctile':>13}{'orig=best':>11}")
    print(header)
    print("-" * len(header))
    for name in METRIC_NAMES:
        m = agg["per_metric"].get(name)
        if not m:
            continue
        pct = m["mean_original_percentile"]
        print(f"{name:<14}{m['n_routes']:>7}{m['routes_with_a_better_rearrangement']:>11}"
              f"{m['frac_routes_beaten']:>7}"
              f"{(f'{pct:.2f}' if pct is not None else 'n/a'):>13}"
              f"{m['n_routes_original_is_best']:>11}")

    print("\nBiggest single improvement found per metric (rearrangement vs literature):")
    for name in METRIC_NAMES:
        m = agg["per_metric"].get(name)
        if not m or not m["biggest_improvement"]:
            continue
        bi = m["biggest_improvement"]
        print(f"  {name:<14} +{bi['delta']:<8} on {bi['tree_id']}: "
              f"{bi['original']} -> {bi['best']}  (ordering {bi['best_ordering']})")

    print("\nReading: 'orig pctile' is the literature route's mean standing among its own "
          "rearrangements (1.0 = always best). 'frac' is the fraction of routes where at "
          "least one rearrangement outscores the literature ordering. Higher score = better "
          "for every metric.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(agg, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
