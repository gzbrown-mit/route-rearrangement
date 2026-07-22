"""Turn linear literature routes into the genomics analogue: a family string plus a partial order.

One pass over the corpus produces, per route, everything the statistics need:

* **the family string** — the route read as a sequence of transformation-family symbols in
  synthesis order (deepest step first), which is the "genome";
* **the essential partial order**, at each necessity tier — the constraint set that defines
  which re-readings of that genome are chemically possible, and therefore Null-C.

Both are expensive to derive (atom-map unification, protecting-group detection) and cheap to
consume, so this follows an extract-once discipline: run it once, write a compact JSONL, and
every downstream stage reads that.

**One step, one transformation.**  Transformation identity comes from :mod:`.step_identity` —
the bonds each step actually changes.  An earlier version keyed steps by FrequenTree
*contextual centres* instead, and that was wrong for this question on two counts.

Chemically: a contextual centre is a connected group of reaction centres, so any pattern whose
steps act on *different* centres cannot be represented at all.  The canonical example is the
motif this project most wants to detect — install a nitro group to activate a ring, run the
SNAr, then reduce the nitro.  Three different reaction centres, one ordering constraint, and a
centre-based identity is structurally incapable of expressing it.

Statistically: centres merge exactly the most tightly coupled step pairs, so those pairs were
absorbed into single units and never counted.  Measured on PaRoutes that inflated the
convention fraction from ≤64.9% to ≤89.3% and hid a 30-fold difference in the necessity count.

**Step numbering.**  Getting this wrong would silently reverse every genome, so it is stated and
tested rather than assumed.  ``full_graph`` node ids run root = 1 down to the deepest step = N,
so synthesis order is **descending node id** — which is also the bit order ``ScheduleLattice``
uses.  :func:`synthesis_order` is the single place this is expressed.

Usage::

    python -m route_rearrangement.route_synteny.corpus \\
        --corpus ~/Downloads/paroutes_all/trees.jsonl \\
        --out route_rearrangement/route_synteny/results/genomes.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import RDLogger

from .. import deps  # noqa: F401
from . import TIERS
from .step_identity import RUNGS, families_for_steps  # noqa: F401  (re-exported for tests)
from synthesis_extraction.dependency.analyze import dependency_graph_from_full_graph
from synthesis_extraction.dependency.route_graph import build_route_graph
from synthesis_extraction.dependency.schedule import MAX_STEPS, lattice_for
from synthesis_extraction.load_trees import iter_trees

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

#: Symbol for a step whose transformation could not be identified — kept in the string rather
#: than dropped, so positions stay faithful to the real route, and excluded from every lookup.
UNKNOWN = "?"

SCHEMA_VERSION = 2


def synthesis_order(step_ids: Sequence[int]) -> List[int]:
    """Step ids in the order the chemist ran them: deepest precursor first.

    ``full_graph`` numbers the root product 1 and counts down into the precursors, so synthesis
    order is descending id.  This is the same convention ``ScheduleLattice`` uses for its bit
    order, which is what lets a sampled linear extension be compared to the literature order
    without any re-indexing.
    """
    return sorted((int(i) for i in step_ids), reverse=True)


def is_linear_tree(tree_graph) -> bool:
    """Every step feeds at most one other — the linear-genome precondition."""
    kids: Counter = Counter()
    for _c, p in tree_graph.edges():
        kids[p] += 1
    return all(v <= 1 for v in kids.values())


@dataclass
class Genome:
    """One linear route in the genomics analogue."""

    route_id: str
    n_steps: int
    step_ids: List[int]                       # synthesis order, deepest first
    families: List[str]                       # aligned to step_ids; UNKNOWN where unidentified
    constraints: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    n_orderings: Dict[str, int] = field(default_factory=dict)   # ordering freedom per tier

    @property
    def coverage(self) -> float:
        if not self.families:
            return 0.0
        return sum(1 for f in self.families if f != UNKNOWN) / len(self.families)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        # Schema 1 carried an ``anchors`` mask marking which steps were a contextual centre's
        # formation event.  With one transformation per step every identified step is an
        # ordering event, so the field is vestigial and is ignored on older caches.
        return cls(
            route_id=d["route_id"], n_steps=int(d["n_steps"]),
            step_ids=[int(i) for i in d["step_ids"]], families=list(d["families"]),
            constraints={k: [(int(a), int(b)) for a, b in v]
                         for k, v in (d.get("constraints") or {}).items()},
            n_orderings={k: int(v) for k, v in (d.get("n_orderings") or {}).items()},
        )


def build_genome(tree_id: str, tree_graph, *, rung: str = "centre_env",
                 count_orderings: bool = True) -> Optional[Genome]:
    """One route -> :class:`Genome`, or ``None`` if it is out of scope or unmappable."""
    full = build_route_graph(tree_graph, tree_id)
    if full is None or full.get("qc", {}).get("disconnected"):
        return None
    step_ids = synthesis_order(int(n["id"]) for n in full.get("nodes", []))
    n_steps = len(step_ids)
    if not n_steps or n_steps > MAX_STEPS:
        return None

    keyed = families_for_steps(full.get("nodes", []), rung)
    genome = Genome(
        route_id=tree_id, n_steps=n_steps, step_ids=step_ids,
        families=[keyed.get(sid) or UNKNOWN for sid in step_ids],
    )
    for tier, flags in TIERS.items():
        dep = dependency_graph_from_full_graph(full, tree_id, **flags)
        genome.constraints[tier] = sorted(
            {(int(e.earlier), int(e.later)) for e in dep.edges})
        if count_orderings:
            genome.n_orderings[tier] = lattice_for(dep).count()
    return genome


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rung", default="centre_env", choices=RUNGS,
                    help="transformation-identity abstraction: centre_env (finest) to "
                         "bond_changes (coarsest)")
    ap.add_argument("--min-steps", type=int, default=4,
                    help="routes shorter than this have too little ordering freedom to test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-count", action="store_true",
                    help="skip exact ordering counts (faster; loses the freedom distribution)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    with out.open("w") as fh:
        for tree_id, tree_graph in iter_trees(args.corpus):
            stats["seen"] += 1
            if args.limit and stats["written"] >= args.limit:
                break
            if tree_graph.number_of_nodes() < args.min_steps:
                stats["too_short"] += 1
                continue
            if not is_linear_tree(tree_graph):
                stats["convergent"] += 1
                continue
            try:
                g = build_genome(tree_id, tree_graph, rung=args.rung,
                                 count_orderings=not args.no_count)
            except Exception as exc:                      # noqa: BLE001 - keep the run going
                stats["error"] += 1
                log.debug("%s: %s", tree_id, exc)
                continue
            if g is None:
                stats["unmappable"] += 1
                continue
            stats["written"] += 1
            stats["fully_covered"] += int(g.coverage >= 0.999)
            stats["steps"] += g.n_steps
            stats["identified"] += sum(1 for f in g.families if f != UNKNOWN)
            fh.write(json.dumps(g.to_dict()) + "\n")
            if stats["written"] % 2000 == 0:
                log.warning("%d genomes written (%d seen)", stats["written"], stats["seen"])

    print(f"seen={stats['seen']:,} written={stats['written']:,} "
          f"(too_short={stats['too_short']:,} convergent={stats['convergent']:,} "
          f"unmappable={stats['unmappable']:,} error={stats['error']:,})")
    if stats["steps"]:
        print(f"steps with an identified transformation: "
              f"{stats['identified'] / stats['steps']:.1%}")
        print(f"routes with every step identified: "
              f"{stats['fully_covered'] / max(1, stats['written']):.1%}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
