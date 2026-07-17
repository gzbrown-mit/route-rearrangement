"""Score enumerated routes with the order-sensitive metrics and summarize statistics per original.

Reads a ``routes.jsonl`` produced by :mod:`.run`, groups routes by their original
literature ``tree_id``, rebuilds each original's unified-map ``full_graph`` (needed by the
exposure oracle), scores every route, and writes:

* ``scored.jsonl`` — every route record with an added ``metrics`` block;
* ``stats.json``   — per original route: for each metric, the original ordering's value and
  its percentile among the rearrangements, the best/worst rearrangement, the fraction of
  rearrangements beating the original, and the cross-metric rank correlations (do the
  metrics agree on which orderings are good?).

Usage::

    python -m route_rearrangement.score --corpus <trees.jsonl> \
        --routes results/routes.jsonl --out-dir results/ [--plausibility]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import deps  # noqa: F401
from . import similarity
from .metrics import METRIC_NAMES
from .metrics.registry import MetricSuite, metric_score
from synthesis_extraction.load_trees import iter_trees
from synthesis_extraction.dependency.route_graph import build_route_graph


def _percentile_of(value: float, population: List[float]) -> float:
    """Fraction of *population* <= *value* (0..1); the original's standing among rearrangements."""
    if not population:
        return float("nan")
    return sum(1 for x in population if x <= value + 1e-12) / len(population)


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize_tree(tree_id: str, records: List[dict]) -> dict:
    """Per-metric statistics comparing the original ordering to its rearrangements."""
    orig_idx = next((i for i, r in enumerate(records) if r.get("is_original_order")), None)
    n = len(records)
    out: dict = {"tree_id": tree_id, "n_routes": n,
                 "n_rearrangements": n - (1 if orig_idx is not None else 0),
                 "metrics": {}, "metric_correlations": {}}

    def _r(x):
        return round(x, 4) if isinstance(x, float) else x

    per_metric_scores: Dict[str, List[Optional[float]]] = {}
    for name in METRIC_NAMES:
        scores = [metric_score(r.get("metrics", {}), name) for r in records]
        per_metric_scores[name] = scores
        avail = [s for s in scores if s is not None]
        if not avail:
            continue
        orig_val = scores[orig_idx] if orig_idx is not None else None
        rearr = [s for i, s in enumerate(scores) if i != orig_idx and s is not None]
        best_i = max((i for i in range(n) if scores[i] is not None), key=lambda i: scores[i])
        worst_i = min((i for i in range(n) if scores[i] is not None), key=lambda i: scores[i])
        stat = {
            "n_scored": len(avail),
            "mean": round(float(np.mean(avail)), 4),
            "std": round(float(np.std(avail)), 4),
            "best": {"ordering": records[best_i]["ordering"], "value": _r(scores[best_i])},
            "worst": {"ordering": records[worst_i]["ordering"], "value": _r(scores[worst_i])},
        }
        if orig_val is not None:
            stat["original_value"] = _r(orig_val)
            stat["original_percentile"] = round(_percentile_of(orig_val, rearr), 4) if rearr else None
            stat["rearrangements_beating_original"] = (
                sum(1 for s in rearr if s > orig_val + 1e-12) if rearr else 0)
            stat["original_is_best"] = best_i == orig_idx
        out["metrics"][name] = stat

    # cross-metric agreement across this tree's routes (Spearman over shared routes)
    for i, m1 in enumerate(METRIC_NAMES):
        for m2 in METRIC_NAMES[i + 1:]:
            pairs = [(x, y) for x, y in zip(per_metric_scores[m1], per_metric_scores[m2])
                     if x is not None and y is not None]
            if len(pairs) >= 3:
                rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
                if rho is not None:
                    out["metric_correlations"][f"{m1}~{m2}"] = round(rho, 3)

    # the diverse "most different" rearrangements (far from the literature route and each
    # other) — the routes the GUI and a reviewer should look at first
    diverse = [r for r in records if r.get("similarity", {}).get("diverse_rank") is not None]
    diverse.sort(key=lambda r: r["similarity"]["diverse_rank"])
    if diverse:
        out["similarity_method"] = diverse[0]["similarity"].get("method")
        out["most_different"] = [
            {"diverse_rank": r["similarity"]["diverse_rank"],
             "ordering": r["ordering"],
             "distance_to_original": r["similarity"]["distance_to_original"],
             "metrics": {m: metric_score(r.get("metrics", {}), m) for m in METRIC_NAMES
                         if metric_score(r.get("metrics", {}), m) is not None}}
            for r in diverse]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--routes", default="results/routes.jsonl")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--plausibility", action="store_true",
                    help="also run the template-relevance plausibility metric (~900MB model)")
    ap.add_argument("--no-treelstm", action="store_true")
    ap.add_argument("--no-similarity", action="store_true",
                    help="skip the route-to-route dissimilarity pass (rxnutils TED)")
    ap.add_argument("--diverse-k", type=int, default=5,
                    help="how many diverse 'most different' rearrangements to flag (default 5)")
    ap.add_argument("--similarity-ted-cap", type=int, default=60,
                    help="max routes per tree to score with exact TED (rest use the fast "
                         "Jaccard prefilter); bounds cost on large enumerations")
    args = ap.parse_args(argv)

    by_tree: Dict[str, List[dict]] = defaultdict(list)
    with open(args.routes) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_tree[rec["tree_id"]].append(rec)
    if not by_tree:
        print("no routes to score")
        return 0

    full_graphs: Dict[str, dict] = {}
    want = set(by_tree)
    for tid, tg in iter_trees(args.corpus):
        if tid in want:
            fg = build_route_graph(tg, tid)
            if fg is not None:
                full_graphs[tid] = fg
            want.discard(tid)
            if not want:
                break

    suite = MetricSuite(use_treelstm=not args.no_treelstm, use_plausibility=args.plausibility)
    avail = suite.availability()
    print("metric availability:", avail)
    do_similarity = not args.no_similarity
    if do_similarity:
        print("route dissimilarity: rxnutils TED"
              if similarity.available() else
              "route dissimilarity: rxnutils/apted unavailable — Jaccard fallback")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []
    n_scored = 0
    with (out_dir / "scored.jsonl").open("w") as sfh:
        for tid, records in by_tree.items():
            fg = full_graphs.get(tid)
            if fg is None:
                print(f"[{tid}] original full_graph unavailable — exposure metric skipped")
            blocks = suite.score_tree(tid, fg or {"nodes": [], "edges": []}, records)
            for rec, block in zip(records, blocks):
                rec["metrics"] = block
            if do_similarity and len(records) > 1:
                _, method = similarity.annotate_distinctness(
                    records, k=args.diverse_k, ted_cap=args.similarity_ted_cap)
                print(f"[{tid}] dissimilarity ({method}): flagged "
                      f"{min(args.diverse_k, len(records) - 1)} most-different rearrangements")
            for rec in records:
                sfh.write(json.dumps(rec) + "\n")
                n_scored += 1
            stats = summarize_tree(tid, records)
            all_stats.append(stats)
            _print_tree_summary(stats)

    with (out_dir / "stats.json").open("w") as fh:
        json.dump({"availability": avail, "trees": all_stats}, fh, indent=2)
    print(f"\nscored {n_scored} routes across {len(by_tree)} originals -> "
          f"{out_dir}/scored.jsonl, {out_dir}/stats.json")
    return 0


def _print_tree_summary(stats: dict) -> None:
    print(f"\n=== {stats['tree_id']}  ({stats['n_rearrangements']} rearrangements + original) ===")
    for name in METRIC_NAMES:
        m = stats["metrics"].get(name)
        if not m:
            continue
        line = f"  {name:<18} mean={m['mean']:>8} std={m['std']:>7}"
        if "original_value" in m:
            pct = m.get("original_percentile")
            pct_s = f"{pct:.0%}" if pct is not None else "n/a"
            line += (f"  original={m['original_value']:>8} (pctile {pct_s}, "
                     f"{m['rearrangements_beating_original']} beat it"
                     f"{', BEST' if m.get('original_is_best') else ''})")
        print(line)
    if stats["metric_correlations"]:
        corr = ", ".join(f"{k}={v:+.2f}" for k, v in stats["metric_correlations"].items())
        print(f"  metric agreement (Spearman): {corr}")
    if stats.get("most_different"):
        print(f"  most different (diverse, {stats.get('similarity_method', '')}): far from "
              f"the literature route and each other")
        for d in stats["most_different"]:
            order = "→".join(map(str, d["ordering"]))
            print(f"    #{d['diverse_rank']:<2} dist={d['distance_to_original']:>7}  {order}")


if __name__ == "__main__":
    raise SystemExit(main())
