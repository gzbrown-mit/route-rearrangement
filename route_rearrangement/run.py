"""End-to-end: enumerate all valid orderings of a route and materialize each.

Usage::

    python -m route_rearrangement.run \
        --corpus ~/synthesis_extraction/synthesis_extraction/data/slice_0-1000/trees.jsonl \
        --tree-id 0_6 --tree-id 1_42 --out-dir results/

    # or take the top-N routes from a select_examples candidates file
    python -m route_rearrangement.run --corpus ... --candidates candidates.jsonl --take 3 ...

Writes ``routes.jsonl`` (accepted materialized routes, incl. the original order) and
``failures.jsonl`` (pruned orderings with reasons) into ``--out-dir`` and prints a
per-route summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import deps  # noqa: F401
from .filters import dedup_key, evaluate
from .materialize import materialize_ordering, replay_identity
from .schema import failure_record, route_record, write_jsonl
from .search import materialize_all_dfs
from .templates import extract_step_templates, is_linear
from synthesis_extraction.load_trees import iter_trees
from synthesis_extraction.dependency.route_graph import build_route_graph
from synthesis_extraction.dependency.analyze import dependency_graph_from_full_graph
from synthesis_extraction.dependency.reorder import respects_constraints
from synthesis_extraction.dependency.schedule import lattice_for


def process_route(tree_id: str, tree_graph, *, engine: str = "dfs", cap: int = 500,
                  beam: int = 3, max_outcomes: int = 20, max_accepted: int = 200,
                  with_fg: bool = True, matrix=None, provenance=None):
    """Run the full pipeline on one route.  Returns ``(summary, records, failures)``."""
    summary = {"tree_id": tree_id, "status": "ok", "n_steps": 0, "n_orders": 0,
               "orderings_tried": 0, "accepted": 0, "duplicates": 0,
               "identity_roundtrip": False, "prune_reasons": Counter()}
    records, failures = [], []

    full = build_route_graph(tree_graph, tree_id)
    if full is None or full["qc"]["disconnected"]:
        summary["status"] = "unmappable_or_disconnected"
        return summary, records, failures
    summary["n_steps"] = full["qc"]["n_steps"]
    if not is_linear(full):
        summary["status"] = "not_linear"
        return summary, records, failures

    templates = extract_step_templates(full)
    dep = dependency_graph_from_full_graph(full, tree_id)
    incidental = dep.incidental_order()

    replay = replay_identity(full, templates, incidental, beam=beam)
    summary["identity_roundtrip"] = replay.ok
    if not replay.ok:
        summary["status"] = "identity_replay_failed"
        summary["detail"] = replay.detail
        return summary, records, failures

    lat = lattice_for(dep)
    summary["n_orders"] = lat.count()

    seen_routes = set()

    def handle_results(ordering_index, ordering, variants):
        accepted_any = False
        for variant_idx, route in enumerate(variants):
            if not route.ok:
                summary["prune_reasons"][route.status] += 1
                failures.append(failure_record(tree_id, route, ordering_index=ordering_index))
                continue
            flags = evaluate(route, templates, matrix=matrix, with_fg=with_fg)
            if flags is None:
                summary["prune_reasons"]["failed_hard_gate"] += 1
                continue
            key = dedup_key(route)
            if key in seen_routes:
                summary["duplicates"] += 1
                continue
            seen_routes.add(key)
            is_orig = list(ordering) == list(incidental)
            records.append(route_record(
                tree_id, route, ordering_index=ordering_index, variant=variant_idx,
                is_original_order=is_orig, identity_roundtrip=replay.ok, flags=flags,
                provenance=provenance))
            accepted_any = True
        return accepted_any

    if engine == "dfs":
        for ordering_index, (ordering, variants) in enumerate(
                materialize_all_dfs(full, templates, dep, cap=cap, beam=beam,
                                    max_outcomes=max_outcomes)):
            summary["orderings_tried"] += 1
            handle_results(ordering_index, ordering, variants)
            summary["accepted"] = len(records)
            if len(records) >= max_accepted:
                break
    else:
        for ordering_index, ordering in enumerate(lat.enumerate_orders(cap=cap)):
            ok, why = respects_constraints(dep, ordering)
            assert ok, f"lattice emitted an ordering violating constraints: {why}"
            summary["orderings_tried"] += 1
            variants = materialize_ordering(full, templates, ordering, beam=beam,
                                            max_outcomes=max_outcomes)
            handle_results(ordering_index, ordering, variants)
            summary["accepted"] = len(records)
            if len(records) >= max_accepted:
                break

    summary["accepted"] = len(records)
    summary["prune_reasons"] = dict(summary["prune_reasons"])
    return summary, records, failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tree-id", action="append", default=[],
                    help="route(s) to process (repeatable)")
    ap.add_argument("--candidates", default="",
                    help="candidates JSONL from select_examples (ranked)")
    ap.add_argument("--take", type=int, default=3,
                    help="with --candidates: process until N routes pass the identity"
                         " gate (candidates failing it are skipped and reported)")
    ap.add_argument("--engine", choices=["naive", "dfs"], default="dfs")
    ap.add_argument("--cap", type=int, default=500, help="max orderings per route")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--max-outcomes", type=int, default=20)
    ap.add_argument("--max-accepted", type=int, default=200)
    ap.add_argument("--no-fg", action="store_true", help="skip fg_risk soft flags")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)

    explicit = list(args.tree_id)
    ranked: list = []
    if args.candidates:
        with open(args.candidates) as fh:
            ranked = [json.loads(line)["tree_id"] for line in fh]
    if not explicit and not ranked:
        ap.error("give --tree-id or --candidates")
    wanted_set = set(explicit) | set(ranked)

    trees = {}
    for tid, tg in iter_trees(args.corpus):
        if tid in wanted_set:
            trees[tid] = tg
            if len(trees) == len(wanted_set):
                break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"corpus": str(args.corpus), "engine": args.engine,
                  "rdchiral_default_template": True}

    n_written = n_failures = n_passed = 0
    with (out_dir / "routes.jsonl").open("w") as routes_fh, \
         (out_dir / "failures.jsonl").open("w") as fail_fh:
        # explicit --tree-id routes always run; ranked candidates run until --take pass
        queue = [(tid, False) for tid in explicit] + \
                [(tid, True) for tid in ranked if tid not in set(explicit)]
        for tid, counted in queue:
            if counted and n_passed >= args.take:
                break
            tg = trees.get(tid)
            if tg is None:
                print(f"[{tid}] not found in corpus", file=sys.stderr)
                continue
            summary, records, failures = process_route(
                tid, tg, engine=args.engine, cap=args.cap, beam=args.beam,
                max_outcomes=args.max_outcomes, max_accepted=args.max_accepted,
                with_fg=not args.no_fg, provenance=provenance)
            for r in records:
                write_jsonl(routes_fh, r)
            for f in failures:
                write_jsonl(fail_fh, f)
            n_written += len(records)
            n_failures += len(failures)
            if counted and summary["status"] == "ok":
                n_passed += 1
            print(f"[{tid}] status={summary['status']} steps={summary['n_steps']} "
                  f"valid_orders={summary['n_orders']} tried={summary['orderings_tried']} "
                  f"accepted={summary['accepted']} dupes={summary['duplicates']} "
                  f"pruned={summary.get('prune_reasons', {})}")

    print(f"wrote {n_written} routes, {n_failures} failures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
