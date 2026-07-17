"""Shared, Qt-free data model for the viewer and the HTML gallery.

Loads a ``routes.jsonl`` / ``scored.jsonl`` file, groups routes by their original
literature ``tree_id``, separates the original ordering from the rearrangements, and
supports sorting the rearrangements by any computed metric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..metrics import METRIC_NAMES
from ..metrics.registry import metric_score


# pseudo-metric sort key for route-to-route dissimilarity (see route_rearrangement.similarity)
DISTINCT_KEY = "distinct"


@dataclass
class RouteEntry:
    record: dict
    is_original: bool

    @property
    def ordering(self) -> List[int]:
        return self.record["ordering"]

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
        most-different-from-the-literature-route first."""
        if not metric:
            return list(self.rearrangements)
        if metric == DISTINCT_KEY:
            def key(e: RouteEntry):
                dr = e.diverse_rank()
                dist = e.distance_to_original() or 0.0
                # diverse picks first (group 0, by ascending pick order), then the rest by
                # descending distance-to-original (group 1, negate for ascending sort)
                return (0, dr) if dr is not None else (1, -dist)
            return sorted(self.rearrangements, key=key)
        return sorted(
            self.rearrangements,
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


def load_groups(path: str) -> Dict[str, TreeGroup]:
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
            entry = RouteEntry(record=rec, is_original=bool(rec.get("is_original_order")))
            if entry.is_original and original is None:
                original = entry
            else:
                rearr.append(entry)
        groups[tid] = TreeGroup(tree_id=tid, original=original, rearrangements=rearr)
    return groups
