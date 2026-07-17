"""Mine transferable synthesis *fragments* from a route's scored enumeration.

A fragment is a contiguous block of steps that good orderings keep together, in a fixed
internal order — a sub-sequence that behaves as one unit.  Crucially this is **not** a
property of the dependency partial order: a material edge A->B only forces A *before* B, it
does not forbid interleaving another step between them.  So fragments are mined empirically
from which orderings score well:

* for every ordered pair ``(a, b)`` we measure how often ``a`` runs *immediately before*
  ``b`` among the top-scoring orderings (``good_freq``) versus among all sampled orderings
  (``all_freq``); their ratio is the adjacency **lift**;
* sticky adjacencies (high ``good_freq`` and lift > 1) are chained, greedily and disjointly,
  into maximal fragments.

Each fragment is tagged **hard** when its internal adjacencies are all material-bonded
(the steps genuinely build on each other's atoms — the "cannot be separated" units) or
**soft** when good routes merely prefer them together (a transferable heuristic).  The
transferable representation is the ordered list of the steps' extracted retro-templates plus
the block's entry/exit molecules — the unit you would graft into a different synthesis.

This is the single-route half of the cross-pathway recombination idea: mine cohesive
fragments here; a later phase abstracts and stitches fragments drawn from many targets'
enumerations into new routes (see ``stitching`` design notes in the README).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import deps  # noqa: F401
from .metrics import METRIC_NAMES
from .metrics.registry import metric_score
from synthesis_extraction.load_trees import iter_trees
from synthesis_extraction.dependency.route_graph import build_route_graph
from synthesis_extraction.dependency.analyze import dependency_graph_from_full_graph
from synthesis_extraction.dependency.graph import MATERIAL

Pair = Tuple[int, int]


# ---------------------------------------------------------------------------
# goodness = mean per-metric percentile rank of an ordering among its siblings
# ---------------------------------------------------------------------------
def goodness_scores(records: List[dict]) -> List[Optional[float]]:
    n = len(records)
    ranks: List[List[Optional[float]]] = [[] for _ in range(n)]
    for name in METRIC_NAMES:
        vals = [metric_score(r.get("metrics", {}), name) for r in records]
        present = [(i, v) for i, v in enumerate(vals) if v is not None]
        if len(present) < 2:
            continue
        order = sorted(present, key=lambda iv: iv[1])
        for rank, (i, _v) in enumerate(order):
            ranks[i].append(rank / (len(present) - 1))     # 0 worst .. 1 best
    return [sum(rs) / len(rs) if rs else None for rs in ranks]


def _adjacencies(ordering: List[int]) -> List[Pair]:
    return list(zip(ordering[:-1], ordering[1:]))


@dataclass
class AdjStat:
    a: int
    b: int
    good_freq: float
    all_freq: float
    lift: float
    good_support: int          # # good orderings with this adjacency
    material: bool             # is a->b a material edge (hard)?


def adjacency_stats(records: List[dict], dep, good_frac: float = 0.34) -> Dict[Pair, AdjStat]:
    good = goodness_scores(records)
    scored = [(g, r) for g, r in zip(good, records) if g is not None]
    n_all = len(records)
    if scored:
        cutoff = sorted((g for g, _ in scored), reverse=True)[
            min(len(scored) - 1, max(0, int(round(good_frac * len(scored))) - 1))]
        good_set = [r for g, r in scored if g >= cutoff]
    else:
        good_set = records
    n_good = len(good_set) or 1

    all_count: Dict[Pair, int] = defaultdict(int)
    good_count: Dict[Pair, int] = defaultdict(int)
    for r in records:
        for p in _adjacencies(r["ordering"]):
            all_count[p] += 1
    for r in good_set:
        for p in _adjacencies(r["ordering"]):
            good_count[p] += 1

    material = {(e.earlier, e.later) for e in dep.edges if e.relation == MATERIAL}
    out: Dict[Pair, AdjStat] = {}
    for p, ac in all_count.items():
        af = ac / n_all
        gf = good_count.get(p, 0) / n_good
        out[p] = AdjStat(a=p[0], b=p[1], good_freq=round(gf, 3), all_freq=round(af, 3),
                         lift=round(gf / af, 3) if af else 0.0,
                         good_support=good_count.get(p, 0), material=p in material)
    return out


# ---------------------------------------------------------------------------
# fragments
# ---------------------------------------------------------------------------
@dataclass
class Fragment:
    steps: List[int]                       # step ids, internal (synthesis) order
    templates: List[str] = field(default_factory=list)   # retro SMARTS, aligned to steps
    reactions: List[str] = field(default_factory=list)    # example map-free reaction SMILES
    kind: str = "soft"                     # "hard" if every internal adjacency is material
    min_good_freq: float = 0.0             # weakest internal adjacency (cohesion)
    min_lift: float = 0.0

    def to_dict(self) -> dict:
        return {"steps": self.steps, "size": len(self.steps), "kind": self.kind,
                "cohesion_min_good_freq": round(self.min_good_freq, 3),
                "cohesion_min_lift": round(self.min_lift, 3),
                "templates": self.templates, "reactions": self.reactions}


def _step_lookup(records: List[dict]) -> Dict[int, dict]:
    """Map orig_step_id -> an example step dict (from any record containing it)."""
    out: Dict[int, dict] = {}
    for r in records:
        for s in r["steps"]:
            out.setdefault(s["orig_step_id"], s)
    return out


def extract_fragments(records: List[dict], dep, *, good_frac: float = 0.34,
                      min_good_freq: float = 0.6, min_lift: float = 1.03) -> List[Fragment]:
    """Greedy disjoint chaining of sticky adjacencies into maximal fragments."""
    stats = adjacency_stats(records, dep, good_frac=good_frac)
    sticky = [s for s in stats.values()
              if s.good_freq >= min_good_freq and s.lift >= min_lift]
    sticky.sort(key=lambda s: (-s.good_freq, -s.lift))

    succ: Dict[int, int] = {}
    pred: Dict[int, int] = {}
    used_edges: List[AdjStat] = []
    for s in sticky:
        if s.a in succ or s.b in pred:
            continue                       # keep chains disjoint (a linear path cover)
        succ[s.a] = s.b
        pred[s.b] = s.a
        used_edges.append(s)
    edge_of = {(s.a, s.b): s for s in used_edges}

    starts = [a for a in succ if a not in pred]
    steps_meta = _step_lookup(records)
    material_all = {(e.earlier, e.later) for e in dep.edges if e.relation == MATERIAL}

    fragments: List[Fragment] = []
    for start in starts:
        chain = [start]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        if len(chain) < 2:
            continue
        adjs = list(zip(chain[:-1], chain[1:]))
        kind = "hard" if all(p in material_all for p in adjs) else "soft"
        templates = [steps_meta.get(sid, {}).get("retro_smarts", "") for sid in chain]
        reactions = [steps_meta.get(sid, {}).get("new_rxn", "") for sid in chain]
        fragments.append(Fragment(
            steps=chain, templates=templates, reactions=reactions, kind=kind,
            min_good_freq=min(edge_of[p].good_freq for p in adjs),
            min_lift=min(edge_of[p].lift for p in adjs)))
    fragments.sort(key=lambda f: (-len(f.steps), -f.min_good_freq))
    return fragments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_by_tree(path: str) -> Dict[str, List[dict]]:
    by: Dict[str, List[dict]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by[rec["tree_id"]].append(rec)
    return by


def _deps_for(corpus: str, tree_ids) -> Dict[str, object]:
    want = set(tree_ids)
    out = {}
    for tid, tg in iter_trees(corpus):
        if tid in want:
            fg = build_route_graph(tg, tid)
            if fg is not None:
                out[tid] = dependency_graph_from_full_graph(fg, tid)
            want.discard(tid)
            if not want:
                break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--routes", default="results_50/scored.jsonl")
    ap.add_argument("--tree-id", default="", help="detail one route; else aggregate all")
    ap.add_argument("--good-frac", type=float, default=0.34)
    ap.add_argument("--min-good-freq", type=float, default=0.6)
    ap.add_argument("--min-lift", type=float, default=1.03)
    ap.add_argument("--out", default="", help="write per-tree fragments JSONL here")
    args = ap.parse_args(argv)

    by_tree = _load_by_tree(args.routes)
    tree_ids = [args.tree_id] if args.tree_id else list(by_tree)
    deps_by = _deps_for(args.corpus, tree_ids)

    kw = dict(good_frac=args.good_frac, min_good_freq=args.min_good_freq, min_lift=args.min_lift)
    all_frags: Dict[str, List[Fragment]] = {}
    for tid in tree_ids:
        recs = by_tree.get(tid, [])
        dep = deps_by.get(tid)
        if dep is None or len(recs) < 3:
            continue
        all_frags[tid] = extract_fragments(recs, dep, **kw)

    if args.tree_id:
        _print_one(args.tree_id, all_frags.get(args.tree_id, []), by_tree.get(args.tree_id, []))
    else:
        _print_aggregate(all_frags)

    if args.out:
        with open(args.out, "w") as fh:
            for tid, frags in all_frags.items():
                fh.write(json.dumps({"tree_id": tid,
                                     "fragments": [f.to_dict() for f in frags]}) + "\n")
        print(f"\nwrote {args.out}")
    return 0


def _print_one(tid: str, frags: List[Fragment], recs: List[dict]) -> None:
    n_steps = len(recs[0]["steps"]) if recs else 0
    print(f"=== {tid} ({n_steps} steps, {len(recs)} orderings) ===")
    if not frags:
        print("  no cohesive multi-step fragments above thresholds")
        return
    for i, f in enumerate(frags, 1):
        print(f"\n  fragment {i} [{f.kind}] steps {f.steps} "
              f"(cohesion good_freq>={f.min_good_freq:.2f}, lift>={f.min_lift:.2f})")
        for sid, rxn in zip(f.steps, f.reactions):
            print(f"     step {sid}: {rxn[:100]}")


def _print_aggregate(all_frags: Dict[str, List[Fragment]]) -> None:
    from collections import Counter
    n_trees = len(all_frags)
    sizes = Counter()
    kinds = Counter()
    trees_with_frag = 0
    trees_with_soft_multistep = 0
    for frags in all_frags.values():
        if frags:
            trees_with_frag += 1
        if any(f.kind == "soft" and len(f.steps) >= 2 for f in frags):
            trees_with_soft_multistep += 1
        for f in frags:
            sizes[len(f.steps)] += 1
            kinds[f.kind] += 1
    print(f"Fragment mining over {n_trees} routes")
    print(f"  routes with >=1 cohesive fragment: {trees_with_frag}/{n_trees}")
    print(f"  routes with a *soft* (emergent, transferable) multi-step fragment: "
          f"{trees_with_soft_multistep}/{n_trees}")
    print(f"  fragments by kind: {dict(kinds)}")
    print(f"  fragment sizes (steps -> count): {dict(sorted(sizes.items()))}")
    # a few illustrative soft fragments
    print("\n  example soft fragments (good routes keep these together though the "
          "partial order would allow splitting them):")
    shown = 0
    for tid, frags in all_frags.items():
        for f in frags:
            if f.kind == "soft" and len(f.steps) >= 2 and shown < 6:
                print(f"    {tid}: steps {f.steps}  cohesion good_freq>={f.min_good_freq:.2f}"
                      f" lift>={f.min_lift:.2f}")
                shown += 1


if __name__ == "__main__":
    raise SystemExit(main())
