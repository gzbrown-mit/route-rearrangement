"""One-shot pipeline over a whole corpus: enumerate → materialize → score, in a single job.

Runs the full workflow on every route of a ``trees.jsonl`` (e.g. PaRoutes ``n1``/``n5``)
— **linear and convergent** trees alike (the frontier engine materializes branching
routes, including convergence-point migration; ``--no-migration`` keeps every fragment
fully assembled before its coupling).  For each route within the step bounds it
enumerates the valid orderings, materializes each backward, scores them with the metric
suite, and streams the results to disk so memory stays flat even on 10k-route corpora.

Outputs (one ``--out-dir``):
* ``scored.jsonl``   — every accepted rearrangement + its ``metrics`` block;
* ``routes.jsonl``   — same records without metrics (raw materialization);
* ``failures.jsonl`` — pruned orderings (template no-match etc.);
* ``stats.jsonl``    — one per-route metric-vs-rearrangement summary line;
* ``summary.json``   — corpus-level coverage counts (linear/convergent/skipped/…).

Usage::

    python -m route_rearrangement.pipeline \
        --corpus ~/synthesis_extraction/synthesis_extraction/data/paroutes/n1/trees.jsonl \
        --out-dir results_n1/ [--limit 0] [--plausibility]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import deps  # noqa: F401
from .metrics.registry import MetricSuite
from .run import process_route
from .score import summarize_tree
from .templates import is_linear
from synthesis_extraction.load_trees import iter_trees
from synthesis_extraction.dependency.route_graph import build_route_graph


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="trees.jsonl (e.g. paroutes n1/n5)")
    ap.add_argument("--out-dir", default="results_full")
    ap.add_argument("--limit", type=int, default=0, help="process at most N trees (0 = all)")
    ap.add_argument("--min-steps", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=10, help="<= 20 (lattice ceiling)")
    ap.add_argument("--cap", type=int, default=200, help="max orderings enumerated per route")
    ap.add_argument("--max-accepted", type=int, default=40, help="max accepted routes per tree")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--engine", choices=["naive", "dfs"], default="dfs")
    ap.add_argument("--no-migration", action="store_true",
                    help="topology-preserving mode: branches interleave but every "
                         "fragment is fully assembled before its coupling")
    ap.add_argument("--linear-only", action="store_true",
                    help="restore the old behavior: skip convergent trees")
    ap.add_argument("--strict", action="store_true",
                    help="reject routes with a Tier 1 'infeasible' finding "
                         "(unactivated SNAr, inverted protecting-group bracket)")
    ap.add_argument("--constraints", choices=["full", "material"], default="full",
                    help="'material' enumerates on the hard atom-based edges only and "
                         "lets the soft chemistry (protection brackets, FG exposure) be "
                         "scored rather than gated — many more orderings per route")
    ap.add_argument("--plausibility", action="store_true",
                    help="also run the ~900MB template-relevance metric (slow at scale)")
    ap.add_argument("--no-treelstm", action="store_true")
    ap.add_argument("--no-fg", action="store_true", help="skip fg_risk soft flags")
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args(argv)

    suite = MetricSuite(use_treelstm=not args.no_treelstm, use_plausibility=args.plausibility)
    avail = suite.availability()
    print("metric availability:", avail, file=sys.stderr)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"corpus": str(args.corpus), "engine": args.engine,
                  "rdchiral_default_template": True,
                  "migration": not args.no_migration,
                  "constraints": args.constraints,
                  "strict": args.strict}
    counts: Counter = Counter()

    scored_fh = (out / "scored.jsonl").open("w")
    routes_fh = (out / "routes.jsonl").open("w")
    fail_fh = (out / "failures.jsonl").open("w")
    stats_fh = (out / "stats.jsonl").open("w")

    def _progress():
        print(f"  [{counts['scanned']}] linear={counts['linear']} "
              f"convergent={counts['convergent']} "
              f"replay_ok={counts['materialized_ok']} "
              f"records={counts['scored_records']} "
              f"migrated={counts['migrated_records']}", file=sys.stderr)

    try:
        for i, (tid, tg) in enumerate(iter_trees(args.corpus)):
            if args.limit and i >= args.limit:
                break
            counts["scanned"] += 1
            if args.progress_every and counts["scanned"] % args.progress_every == 0:
                _progress()

            try:
                full = build_route_graph(tg, tid)
            except Exception:
                full = None
            if full is None:
                counts["unmappable"] += 1
                continue
            if full["qc"]["disconnected"]:
                counts["disconnected"] += 1
                continue
            n_steps = full["qc"]["n_steps"]
            if not (args.min_steps <= n_steps <= args.max_steps):
                counts["out_of_step_range"] += 1
                continue
            if is_linear(full):
                counts["linear"] += 1
            elif args.linear_only:
                counts["convergent_skipped"] += 1
                continue
            else:
                counts["convergent"] += 1

            try:
                summary, records, failures = process_route(
                    tid, tg, full_graph=full, engine=args.engine, cap=args.cap,
                    beam=args.beam, max_accepted=args.max_accepted,
                    with_fg=not args.no_fg, provenance=provenance,
                    migration=not args.no_migration, constraints=args.constraints,
                    strict=args.strict)
            except Exception as exc:  # noqa: BLE001
                counts["process_error"] += 1
                print(f"  [{tid}] error: {exc}", file=sys.stderr)
                continue

            for f in failures:
                fail_fh.write(json.dumps(f) + "\n")
            if summary["status"] != "ok":
                counts[f"status_{summary['status']}"] += 1
                continue
            counts["materialized_ok"] += 1

            try:
                blocks = suite.score_tree(tid, full, records)
            except Exception:  # noqa: BLE001
                blocks = [{} for _ in records]
            for rec, block in zip(records, blocks):
                rec["metrics"] = block
                routes_fh.write(json.dumps({k: v for k, v in rec.items()
                                            if k != "metrics"}) + "\n")
                scored_fh.write(json.dumps(rec) + "\n")
                counts["scored_records"] += 1
                if rec.get("flags", {}).get("migrated_steps"):
                    counts["migrated_records"] += 1
            try:
                stats_fh.write(json.dumps(summarize_tree(tid, records)) + "\n")
            except Exception:  # noqa: BLE001
                pass
    finally:
        for fh in (scored_fh, routes_fh, fail_fh, stats_fh):
            fh.close()

    summary_obj = {"corpus": str(args.corpus), "availability": avail, "counts": dict(counts)}
    with (out / "summary.json").open("w") as fh:
        json.dump(summary_obj, fh, indent=2)

    _progress()
    print(f"\nDONE. scanned {counts['scanned']} trees -> {out}/")
    print(f"  linear:             {counts['linear']}")
    print(f"  convergent:         {counts['convergent']}")
    if counts["convergent_skipped"]:
        print(f"  convergent skipped: {counts['convergent_skipped']}  (--linear-only)")
    print(f"  unmappable:         {counts['unmappable']}")
    print(f"  out of step range:  {counts['out_of_step_range']}")
    print(f"  replay-passed:      {counts['materialized_ok']}")
    print(f"  scored records:     {counts['scored_records']}")
    print(f"  with migration:     {counts['migrated_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
