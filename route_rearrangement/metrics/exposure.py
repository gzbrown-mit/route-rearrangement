"""Metric 3 — functional-group exposure oracle (synthesis_extraction native).

This is the one metric that does *not* look at the rearranged molecules: it prices an
ordering directly on the original route's unified-atom-map ``full_graph``.  Walking the
candidate order, it tracks which reactive functional groups are free on the substrate at
each point and asks the interpretable verdict engine whether each bystander survives the
step's conditions.  ``n_destroyed`` = protections the ordering would force a chemist to
install; ``n_abstain`` = unknown-survival exposures (never silently treated as safe).

Cost is ``(n_destroyed, n_abstain)`` — lower is better; ``score`` negates the destroyed
count so higher is better.  An :class:`ExposureOracle` is built once per original route
(``full_graph``) and reused for every ordering of it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .. import deps  # noqa: F401
from synthesis_extraction.dependency.exposure import ExposureOracle

HIGHER_IS_BETTER = True


class ExposureScorer:
    """One :class:`ExposureOracle` per original ``full_graph``, cached by tree id."""

    def __init__(self, matrix=None):
        self.matrix = matrix
        self._oracles: Dict[str, ExposureOracle] = {}

    def oracle_for(self, tree_id: str, full_graph: dict) -> ExposureOracle:
        if tree_id not in self._oracles:
            self._oracles[tree_id] = ExposureOracle(full_graph, matrix=self.matrix)
        return self._oracles[tree_id]

    def score(self, tree_id: str, full_graph: dict, ordering: List[int]) -> dict:
        """``{n_destroyed, n_abstain, protections_needed, score}`` for one ordering."""
        try:
            oracle = self.oracle_for(tree_id, full_graph)
            rep = oracle.report(list(ordering))
        except Exception:
            return {}
        return {
            "n_destroyed": rep.n_destroyed,
            "n_abstain": rep.n_abstain,
            "protections_needed": len(rep.protections_needed),
            "score": round(-float(rep.n_destroyed) - 0.1 * rep.n_abstain, 4),
        }
