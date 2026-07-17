"""Assemble the metrics and score a corpus of materialized routes.

A :class:`MetricSuite` holds the (lazily-loaded) scorers and computes, per route record,
a ``{metric_name: {..., "score": float}}`` block.  ``treelstm`` is batched across all of a
tree's routes; the rest are per-route.  Any metric whose model is unavailable is recorded
as ``{"available": False}`` and simply omitted from the statistics — the pipeline never
crashes because one model is missing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import METRIC_NAMES
from . import carried_complexity as _cc
from . import competing as _comp
from . import isolability as _iso
from .exposure import ExposureScorer
from .plausibility import PlausibilityScorer
from .treelstm import TreeLSTMRanker


class MetricSuite:
    def __init__(self, *, use_treelstm=True, use_plausibility=False, use_exposure=True,
                 use_competing=True, use_isolability=True, use_carried_complexity=True,
                 matrix=None):
        self.flags = {
            "treelstm": use_treelstm, "plausibility": use_plausibility,
            "exposure": use_exposure, "competing": use_competing,
            "isolability": use_isolability, "carried_complexity": use_carried_complexity,
        }
        self.treelstm = TreeLSTMRanker() if use_treelstm else None
        self.plausibility = PlausibilityScorer() if use_plausibility else None
        self.exposure = ExposureScorer(matrix=matrix) if use_exposure else None
        self._availability: Dict[str, bool] = {}

    def availability(self) -> Dict[str, bool]:
        if not self._availability:
            self._availability = {
                "treelstm": bool(self.treelstm and self.treelstm.available),
                "plausibility": bool(self.plausibility and self.plausibility.available),
                "exposure": bool(self.exposure),
                "competing": self.flags["competing"] and _comp.available(),
                "isolability": self.flags["isolability"] and _iso.available(),
                "carried_complexity": self.flags["carried_complexity"] and _cc.available(),
            }
        return self._availability

    def score_tree(self, tree_id: str, full_graph: dict,
                   records: List[dict]) -> List[Dict[str, dict]]:
        """Score every materialized route of one original tree.  Returns one metric block
        per record, aligned with *records*."""
        avail = self.availability()
        blocks: List[Dict[str, dict]] = [{} for _ in records]

        if avail["treelstm"]:
            scores = self.treelstm.score_records(records)
            for b, s in zip(blocks, scores):
                b["treelstm"] = {"score": s} if s is not None else {"available": False}

        for i, rec in enumerate(records):
            if avail["exposure"]:
                blocks[i]["exposure"] = self.exposure.score(
                    tree_id, full_graph, rec["ordering"]) or {"available": False}
            if avail["competing"]:
                blocks[i]["competing"] = _comp.competing_sites(rec) or {"available": False}
            if avail["isolability"]:
                blocks[i]["isolability"] = _iso.isolability(rec) or {"available": False}
            if avail["carried_complexity"]:
                blocks[i]["carried_complexity"] = _cc.carried_complexity(rec) or {"available": False}
            if avail["plausibility"]:
                blocks[i]["plausibility"] = self.plausibility.score(rec) or {"available": False}

        return blocks


def metric_score(block: Dict[str, dict], name: str) -> Optional[float]:
    """The comparable ``score`` for one metric of one route, or ``None`` if unavailable."""
    m = block.get(name)
    if not m or "score" not in m or m.get("available") is False:
        return None
    return m["score"]
