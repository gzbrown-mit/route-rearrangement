"""Rank a corpus's routes by reordering modularity — the test-example selector.

The best test cases are routes with many *disconnections between steps that build on
each other only incidentally*: a large number of valid orderings
(``ScheduleLattice.count``) and many commutable pairs, i.e. modular rather than
strictly-coupled construction.  Convergent routes qualify too (``--linear-only``
restores the old filter); each candidate records its topology.

Usage::

    python -m route_rearrangement.select_examples \
        --corpus ~/synthesis_extraction/synthesis_extraction/data/slice_0-1000/trees.jsonl \
        --top 20 --out candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from . import deps  # noqa: F401
from .templates import is_linear
from synthesis_extraction.load_trees import iter_trees
from synthesis_extraction.dependency.route_graph import build_route_graph
from synthesis_extraction.dependency.analyze import dependency_graph_from_full_graph
from synthesis_extraction.dependency.schedule import MAX_STEPS, lattice_for


def rank_corpus(corpus: str, *, limit: int = 0, min_steps: int = 3, max_steps: int = 10,
                progress: bool = True, linear_only: bool = False):
    """Yield candidate dicts (unranked) for every qualifying route."""
    for i, (tid, tg) in enumerate(iter_trees(corpus)):
        if limit and i >= limit:
            break
        if progress and i and i % 200 == 0:
            print(f"  ...scanned {i} trees", file=sys.stderr)
        full = build_route_graph(tg, tid)
        if full is None or full["qc"]["disconnected"]:
            continue
        n_steps = full["qc"]["n_steps"]
        if not (min_steps <= n_steps <= min(max_steps, MAX_STEPS)):
            continue
        linear = is_linear(full)
        if linear_only and not linear:
            continue
        try:
            dep = dependency_graph_from_full_graph(full, tid)
            lat = lattice_for(dep)
            n_orders = lat.count()
        except Exception:
            continue
        if n_orders < 2:
            continue
        n_comm = len(dep.commutable_pairs())
        max_orders = math.factorial(n_steps)
        yield {
            "tree_id": tid,
            "n_steps": n_steps,
            "n_orders": n_orders,
            "n_commutable": n_comm,
            "linear": linear,
            "flexibility": round(math.log(n_orders) / math.log(max_orders), 4)
                           if max_orders > 1 else 0.0,
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--limit", type=int, default=0, help="scan at most N trees (0 = all)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-steps", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--linear-only", action="store_true", help="skip convergent trees")
    ap.add_argument("--out", default="", help="write ranked candidates JSONL here")
    args = ap.parse_args(argv)

    candidates = list(rank_corpus(args.corpus, limit=args.limit,
                                  min_steps=args.min_steps, max_steps=args.max_steps,
                                  linear_only=args.linear_only))
    candidates.sort(key=lambda c: (-c["n_orders"], -c["n_commutable"], c["tree_id"]))
    top = candidates[: args.top]

    n_conv = sum(1 for c in candidates if not c["linear"])
    print(f"{len(candidates)} qualifying routes ({n_conv} convergent); top {len(top)}:")
    print(f"{'tree_id':<14}{'steps':>6}{'orders':>8}{'commutable':>12}{'flexibility':>13}"
          f"{'topology':>12}")
    for c in top:
        print(f"{c['tree_id']:<14}{c['n_steps']:>6}{c['n_orders']:>8}"
              f"{c['n_commutable']:>12}{c['flexibility']:>13}"
              f"{'linear' if c['linear'] else 'convergent':>12}")

    if args.out:
        with open(args.out, "w") as fh:
            for c in candidates:
                fh.write(json.dumps(c) + "\n")
        print(f"wrote {len(candidates)} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
