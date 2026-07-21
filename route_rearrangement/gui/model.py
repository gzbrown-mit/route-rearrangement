"""Shared, Qt-free data model for the viewer and the HTML gallery.

Loads a ``routes.jsonl`` / ``scored.jsonl`` file, groups routes by their original
literature ``tree_id``, separates the original ordering from the rearrangements, and
supports sorting the rearrangements by any computed metric.

Optionally joins a ``feasibility.jsonl`` produced by :mod:`route_rearrangement.audit`, so
the post-hoc chemical findings can be displayed alongside the metrics.  The join is by
``(tree_id, ordering)``; findings are read-only annotations and never change the record.
Findings are classified against the route's **own literature ordering**: a check the
published route also trips is inherent to the chemistry, not damage the rearrangement did,
so only the *new* ones are chargeable to the rearrangement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..metrics import METRIC_NAMES
from ..metrics.registry import metric_score


# pseudo-metric sort key for route-to-route dissimilarity (see route_rearrangement.similarity)
DISTINCT_KEY = "distinct"


@dataclass
class RouteEntry:
    record: dict
    is_original: bool
    pin_rank: Optional[int] = None      # set when the user asked for this exact ordering

    @property
    def pinned(self) -> bool:
        return self.pin_rank is not None

    @property
    def ordering(self) -> List[int]:
        return self.record["ordering"]

    # -- post-hoc feasibility audit (joined from feasibility.jsonl, may be absent) --------
    def findings(self) -> List[dict]:
        return self.record.get("feasibility", {}).get("findings", []) or []

    def checks(self) -> set:
        return {f["check"] for f in self.findings()}

    def has_audit(self) -> bool:
        return "feasibility" in self.record

    def n_infeasible(self) -> int:
        return sum(1 for f in self.findings() if f.get("severity") == "infeasible")

    def n_risk(self) -> int:
        return sum(1 for f in self.findings() if f.get("severity") != "infeasible")

    def score(self, metric: str) -> Optional[float]:
        return metric_score(self.record.get("metrics", {}), metric)

    def metric_summary(self) -> Dict[str, Optional[float]]:
        return {m: self.score(m) for m in METRIC_NAMES}

    # -- route-to-route dissimilarity (populated by score.py via similarity.annotate) ------
    def distance_to_original(self) -> Optional[float]:
        d = self.record.get("similarity", {}).get("distance_to_original")
        return None if d is None else float(d)

    def diverse_rank(self) -> Optional[int]:
        return self.record.get("similarity", {}).get("diverse_rank")


@dataclass
class TreeGroup:
    tree_id: str
    original: Optional[RouteEntry]
    rearrangements: List[RouteEntry] = field(default_factory=list)

    def literature_checks(self) -> set:
        """Checks the published ordering itself trips — the baseline every rearrangement of
        this route inherits for free."""
        return self.original.checks() if self.original is not None else set()

    def new_checks(self, entry: RouteEntry) -> List[str]:
        """Checks *entry* trips that its own literature ordering does not."""
        return sorted(entry.checks() - self.literature_checks())

    def has_audit(self) -> bool:
        return any(e.has_audit() for e in
                   ([self.original] if self.original else []) + self.rearrangements)

    def available_metrics(self) -> List[str]:
        entries = ([self.original] if self.original else []) + self.rearrangements
        return [m for m in METRIC_NAMES
                if any(e.score(m) is not None for e in entries)]

    def has_distinctness(self) -> bool:
        return any(e.distance_to_original() is not None for e in self.rearrangements)

    def sort_keys(self) -> List[str]:
        """Selectable sort keys: the dissimilarity key first (if computed), then metrics."""
        keys = self.available_metrics()
        if self.has_distinctness():
            keys = [DISTINCT_KEY] + keys
        return keys

    def sorted_rearrangements(self, metric: Optional[str]) -> List[RouteEntry]:
        """Rearrangements best-first by *metric* (higher score = better); those without a
        score for it sink to the end.  ``None`` keeps file order.

        The special key :data:`DISTINCT_KEY` orders by route-to-route dissimilarity: the
        diverse top-k picks first (in farthest-first pick order), then every other route
        most-different-from-the-literature-route first.

        Pinned entries (``--ordering``) always come first, in the order they were asked
        for, whatever the sort key — the point of pinning is that you came to look at
        those specific ones."""
        rest = [e for e in self.rearrangements if not e.pinned]
        pins = sorted((e for e in self.rearrangements if e.pinned), key=lambda e: e.pin_rank)
        return pins + self._sorted(rest, metric)

    def _sorted(self, entries: List[RouteEntry], metric: Optional[str]) -> List[RouteEntry]:
        if not metric:
            return list(entries)
        if metric == DISTINCT_KEY:
            def key(e: RouteEntry):
                dr = e.diverse_rank()
                dist = e.distance_to_original() or 0.0
                # diverse picks first (group 0, by ascending pick order), then the rest by
                # descending distance-to-original (group 1, negate for ascending sort)
                return (0, dr) if dr is not None else (1, -dist)
            return sorted(entries, key=key)
        return sorted(
            entries,
            key=lambda e: (e.score(metric) is not None, e.score(metric) or 0.0),
            reverse=True,
        )

    def percentile(self, entry: RouteEntry, metric: str) -> Optional[float]:
        """Fraction of rearrangements with score <= *entry*'s score for *metric*."""
        v = entry.score(metric)
        if v is None:
            return None
        pop = [e.score(metric) for e in self.rearrangements if e.score(metric) is not None]
        if not pop:
            return None
        return sum(1 for x in pop if x <= v + 1e-12) / len(pop)


def parse_ordering(text: str) -> Tuple[int, ...]:
    """``"6,3,5,2,4,1"`` / ``"[6, 3, 5, 2, 4, 1]"`` / ``"6 3 5 2 4 1"`` -> tuple of ints."""
    cleaned = text.strip().strip("[]()").replace(",", " ")
    return tuple(int(tok) for tok in cleaned.split())


def load_feasibility(path: str) -> Dict[Tuple[str, Tuple[int, ...]], dict]:
    """Index an audit ``feasibility.jsonl`` by ``(tree_id, ordering)``."""
    index: Dict[Tuple[str, Tuple[int, ...]], dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                index[(rec["tree_id"], tuple(rec["ordering"]))] = rec
    return index


def load_groups(path: str, *, feasibility: Optional[str] = None,
                pin: Optional[Sequence[Sequence[int]]] = None) -> Dict[str, TreeGroup]:
    """Group the routes of *path* by literature tree.

    *feasibility* — path to an audit ``feasibility.jsonl`` to join in (findings are shown
    but never filter: the audit is a review instrument, not a gate).
    *pin* — orderings to surface first in every sorted view."""
    audit = load_feasibility(feasibility) if feasibility else {}
    pin_rank = {tuple(o): i for i, o in enumerate(pin or ())}

    by_tree: Dict[str, List[dict]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_tree.setdefault(rec["tree_id"], []).append(rec)
    groups: Dict[str, TreeGroup] = {}
    for tid, recs in by_tree.items():
        original = None
        rearr: List[RouteEntry] = []
        for rec in recs:
            key = (tid, tuple(rec["ordering"]))
            if key in audit:
                rec["feasibility"] = audit[key]
            entry = RouteEntry(record=rec, is_original=bool(rec.get("is_original_order")),
                               pin_rank=pin_rank.get(key[1]))
            if entry.is_original and original is None:
                original = entry
            else:
                rearr.append(entry)
        groups[tid] = TreeGroup(tree_id=tid, original=original, rearrangements=rearr)
    return groups
