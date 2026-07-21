"""Launch the route viewer.

    python -m route_rearrangement.gui --routes results/scored.jsonl --tree-id 106_201
    python -m route_rearrangement.gui --routes results/scored.jsonl --tree-id 106_201 --sort exposure

Use ``--ordering`` to jump straight to a specific rearrangement (repeatable; pinned ones
are shown first however the rest are sorted), and ``--feasibility`` to join in an audit
``feasibility.jsonl`` so each route's chemical findings are shown next to its metrics.

Use ``--html`` to write a static HTML gallery instead of opening the PyQt window (works
without a display).
"""

from __future__ import annotations

import argparse
import sys

from .gallery import build_gallery
from .model import load_groups, parse_ordering


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Browse enumerated rearrangements of a route")
    ap.add_argument("--routes", default="results/scored.jsonl",
                    help="scored.jsonl (preferred) or routes.jsonl")
    ap.add_argument("--tree-id", required=True)
    ap.add_argument("--sort", default="", help="sort key: a metric name, or 'distinct' "
                    "(default when computed) to show the most-different routes first")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--html", default="", help="write a static HTML gallery to this path "
                    "instead of opening the window")
    ap.add_argument("--top", type=int, default=25, help="HTML mode: max rearrangements shown "
                    "(e.g. --sort distinct --top 5 for the 5 most different)")
    ap.add_argument("--feasibility", default="",
                    help="audit feasibility.jsonl to join in; findings are displayed "
                         "alongside the metrics and never used to filter")
    ap.add_argument("--ordering", action="append", default=[],
                    help="pin this exact ordering to the front, e.g. "
                         "--ordering 6,3,5,2,4,1 (repeatable)")
    args = ap.parse_args(argv)

    groups = load_groups(args.routes, feasibility=args.feasibility or None,
                         pin=[parse_ordering(o) for o in args.ordering])
    if args.tree_id not in groups:
        print(f"{args.tree_id} not found in {args.routes}. Available: {sorted(groups)}",
              file=sys.stderr)
        return 2
    group = groups[args.tree_id]

    if args.html:
        import os
        work = os.path.join(os.path.dirname(args.html) or ".", f"_imgs_{args.tree_id}")
        build_gallery(group, args.html, work_dir=work, sort_metric=args.sort or None,
                      top=args.top, dpi=args.dpi)
        print(f"wrote {args.html}")
        return 0

    from .viewer import launch
    return launch(group, sort_metric=args.sort or None, dpi=args.dpi)


if __name__ == "__main__":
    raise SystemExit(main())
