"""Calibrate the abstraction ladder: how precise is each rung, and how much does it recur?

Every rung trades the same two things against each other.  Go finer and the key is chemically
faithful but the pair is never seen twice, so nothing can be measured.  Go coarser and the
counts arrive but the key may have pooled reactions that are not the same chemistry, so the
number measures nothing in particular.  The defaults inherited from upstream
(``--radius 0``, ``--ladder-level 3``) were never measured against that tradeoff — this module
measures it, so the backoff thresholds in :mod:`.ladder` are chosen rather than assumed.

Per rung it reports:

* **coverage** — fraction of contextual centers the rung can key at all;
* **recurrence** — the pair-support distribution, and how many pairs clear the testable floor.
  This is the axis ``transformation.pattern_key`` already flagged as fatal at template
  resolution (median pair seen ~twice);
* **collision** — within-key heterogeneity of the reactions pooled under one key, measured by
  the distinct ``synthon_env`` reacting substructures a key covers.  A key that pools many
  unrelated environments is over-abstracted, and this is where that starts to show;
* **route concentration** — the share of a rung's observations coming from its busiest routes,
  because a rung can look well-supported while resting on a handful of near-duplicates.

``radius`` is orthogonal to the rungs — it sets how far context extends from the reaction
centre when FrequenTree extracts, so changing it means re-running the expensive extraction.
Pass several centers caches built at different radii via ``--radius-dir NAME=PATH...`` to
profile the whole ladder at each; with one cache the sweep still profiles every rung.

Usage::

    python -m route_rearrangement.literature_precedent.sweep \\
        --centers ~/Downloads/paroutes_all/centers/centers_00*.jsonl \\
        --material-deps ~/Downloads/paroutes_all/material_deps/*.jsonl \\
        --out route_rearrangement/literature_precedent/results/sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .. import deps  # noqa: F401
from . import aggregate, ladder
from .aggregate import N_FS, N_ROUTES, N_SF, RungTable
from .ladder import RUNGS, Rung
from synthesis_extraction.transformation.extract_centers import iter_center_files
from synthesis_extraction.transformation.fc_adapter import ContextualCenter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collision audit
# ---------------------------------------------------------------------------
def collision_profile(route_centers: Iterable[Tuple[str, List[ContextualCenter]]],
                      rungs: Sequence[Rung] = RUNGS,
                      top_k: int = 200,
                      limit: int = 0) -> Dict[str, dict]:
    """How chemically heterogeneous are the reactions pooled under one key?

    The proxy is ``ContextualCenter.synthon_env`` — the real SMILES of the reacting
    substructure plus a shell, cached alongside every center.  Two centers under one key with
    the same env are the same transformation seen twice; many distinct envs under one key mean
    the key has abstracted away something chemical.  Reported for the *busiest* keys, since
    those are the ones carrying the pair statistics.
    """
    envs: Dict[str, Dict[str, set]] = {r.name: defaultdict(set) for r in rungs}
    counts: Dict[str, Dict[str, int]] = {r.name: defaultdict(int) for r in rungs}
    n = 0
    for _, centers in route_centers:
        if limit and n >= limit:
            break
        n += 1
        for c in centers:
            keys = ladder.keys_for_center(c, rungs)
            for r in rungs:
                k = keys[r.name]
                if not k:
                    continue
                counts[r.name][k] += 1
                if c.synthon_env:
                    envs[r.name][k].add(c.synthon_env)

    out: Dict[str, dict] = {}
    for r in rungs:
        busiest = sorted(counts[r.name].items(), key=lambda kv: -kv[1])[:top_k]
        widths = [len(envs[r.name].get(k, ())) for k, _ in busiest]
        occ = [c for _, c in busiest]
        out[r.name] = {
            "n_keys_profiled": len(busiest),
            "envs_per_key_median": statistics.median(widths) if widths else None,
            "envs_per_key_p90": _quantile(widths, 0.9),
            "envs_per_key_max": max(widths) if widths else None,
            # distinct environments per occurrence: 1.0 means every use of the key brings a
            # different reacting substructure (the key pools unrelated chemistry); near 0
            # means the key keeps naming the same transformation (specific, as intended)
            "env_diversity_ratio": (sum(widths) / sum(occ)) if sum(occ) else None,
        }
    return out


def _quantile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


# ---------------------------------------------------------------------------
# Rung profile
# ---------------------------------------------------------------------------
def profile_table(t: RungTable, *, min_n: int = 30, min_routes: int = 5) -> dict:
    """Coverage, recurrence and concentration for one rung's aggregate table."""
    supports = [r[N_FS] + r[N_SF] for r in t.pairs.values()]
    testable = [r for r in t.pairs.values()
                if (r[N_FS] + r[N_SF]) >= min_n and r[N_ROUTES] >= min_routes]
    ordered_total = sum(supports)
    return {
        "rung": t.rung.name,
        "rung_index": t.rung.index,
        "description": t.rung.description,
        "n_routes": t.n_routes,
        "coverage_centers": (t.n_centers_keyed / t.n_centers) if t.n_centers else None,
        "n_keys": t.n_keys,
        "n_pairs": len(t.pairs),
        "ordered_observations": ordered_total,
        "support_median": statistics.median(supports) if supports else None,
        "support_p90": _quantile(supports, 0.9),
        "support_max": max(supports) if supports else None,
        "frac_pairs_seen_once": (sum(1 for s in supports if s <= 1) / len(supports)) if supports else None,
        "n_pairs_testable": len(testable),
        "frac_pairs_testable": (len(testable) / len(supports)) if supports else None,
        # concentration: observations per contributing route, at the testable pairs
        "obs_per_route_testable": (
            sum(r[N_FS] + r[N_SF] for r in testable) / sum(r[N_ROUTES] for r in testable)
            if testable else None),
    }


def sweep(source, rungs: Sequence[Rung] = RUNGS,
          forced: Optional[Dict[str, Dict[int, set]]] = None,
          *, min_n: int = 30, min_routes: int = 5, limit: int = 0,
          collisions: bool = True, collision_limit: int = 20000) -> dict:
    """Profile every rung over one centers cache.

    *source* is a **callable returning a fresh iterator** of ``(route_id, centers)``, not an
    iterator: the collision audit needs a second pass, and materializing 457k routes' centers
    to replay them costs several GB.
    """
    make = source if callable(source) else (lambda _s=list(source): iter(_s))
    tables = aggregate.aggregate(make(), rungs=rungs, forced=forced, limit=limit)
    profiles = [profile_table(tables[r.name], min_n=min_n, min_routes=min_routes) for r in rungs]
    coll = collision_profile(make(), rungs, limit=collision_limit) if collisions else {}
    for p in profiles:
        p.update(coll.get(p["rung"], {}))
    return {"min_n": min_n, "min_routes": min_routes, "rungs": profiles}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_COLS = [
    ("rung", "rung", "{}", 20),
    ("keys", "n_keys", "{:,}", 9),
    ("cover", "coverage_centers", "{:.3f}", 7),
    ("pairs", "n_pairs", "{:,}", 9),
    ("med n", "support_median", "{:.0f}", 7),
    ("<=1 obs", "frac_pairs_seen_once", "{:.2f}", 8),
    ("testable", "n_pairs_testable", "{:,}", 9),
    ("obs/route", "obs_per_route_testable", "{:.2f}", 10),
    ("envs/key", "envs_per_key_median", "{:.0f}", 9),
    ("env div", "env_diversity_ratio", "{:.3f}", 8),
]


def format_table(result: dict) -> str:
    head = "".join(f"{h:>{w}}" for h, _, _, w in _COLS)
    lines = [head, "-" * len(head)]
    for p in result["rungs"]:
        row = ""
        for _, field, fmt, w in _COLS:
            v = p.get(field)
            s = "-" if v is None else (fmt.format(v) if not isinstance(v, str) else v)
            row += f"{s:>{w}}"
        lines.append(row)
    lines.append("")
    lines.append(f"testable = pairs with >= {result['min_n']} ordered observations from "
                 f">= {result['min_routes']} distinct routes")
    lines.append("med n    = median STRICTLY-ORDERED observations per pair; 0 means most pairs "
                 "are same-step or materially forced, not that the pair is unseen")
    lines.append("<=1 obs  = fraction of pairs with at most one ordered observation — the "
                 "recurrence problem, quantified")
    lines.append("obs/route= ordered observations per contributing route at the testable "
                 "pairs; ~1 means route clustering has almost nothing to correct")
    lines.append("env div  = distinct reacting environments per occurrence among the busiest "
                 "keys; HIGH means the key pools unrelated chemistry, low means it is specific")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--centers", nargs="+", required=True)
    ap.add_argument("--material-deps", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--min-routes", type=int, default=5)
    ap.add_argument("--no-collisions", action="store_true",
                    help="skip the collision audit (halves memory; keeps coverage/recurrence)")
    ap.add_argument("--radius-dir", action="append", default=None, metavar="NAME=GLOB",
                    help="additional centers cache built at another radius, e.g. r1=path/*.jsonl")
    args = ap.parse_args(argv)

    forced = aggregate.load_forced(args.material_deps)
    results = {}

    def _run(label: str, paths: Sequence[str]) -> None:
        log.warning("sweeping %s over %d file(s)", label, len(paths))
        results[label] = sweep(lambda _p=list(paths): iter_center_files(_p),
                               forced=forced, limit=args.limit,
                               min_n=args.min_n, min_routes=args.min_routes,
                               collisions=not args.no_collisions)
        print(f"\n=== {label} ===")
        print(format_table(results[label]))

    _run("radius0", args.centers)
    for spec in (args.radius_dir or []):
        if "=" not in spec:
            ap.error(f"--radius-dir needs NAME=GLOB, got {spec!r}")
        name, glob = spec.split("=", 1)
        paths = sorted(str(p) for p in Path(glob).parent.glob(Path(glob).name))
        if not paths:
            ap.error(f"--radius-dir {name}: no files matched {glob!r}")
        _run(name, paths)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
