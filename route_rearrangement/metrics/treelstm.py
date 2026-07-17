"""Metric 1 — Tree-LSTM literature-likeness (Mo et al.).

Wraps the trained Tree-LSTM pathway ranker from
https://github.com/moyiming1/Retrosynthesis-pathway-ranking (cloned into
``external/Retrosynthesis-pathway-ranking``).  The model scores how much a route-tree
resembles the literature reference pathways it was trained on; a higher score = more
literature-like.  It is order-sensitive because the tree's per-reaction fingerprints and
structure change with the ordering of the steps.

Set ``PATHWAY_RANKER_PATH`` to override the checkout location.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from .base import retro_tree

PATHWAY_RANKER_PATH = os.environ.get(
    "PATHWAY_RANKER_PATH",
    str(Path(__file__).resolve().parents[2] / "external" / "Retrosynthesis-pathway-ranking"),
)

HIGHER_IS_BETTER = True


class TreeLSTMRanker:
    """Lazy-loading wrapper; ``score_records`` batches all trees through one forward pass."""

    def __init__(self, fp_size: int = 2048, lstm_size: int = 256,
                 model_path: Optional[str] = None):
        self.fp_size = fp_size
        self.lstm_size = lstm_size
        self.model_path = model_path or os.path.join(
            PATHWAY_RANKER_PATH, "trained_model", "treeLSTM256-fp2048.pt")
        self._model = None
        self._convert = None
        self._merge = None
        self._torch = None

    def _load(self):
        if self._model is not None:
            return
        if PATHWAY_RANKER_PATH not in sys.path:
            sys.path.insert(0, PATHWAY_RANKER_PATH)
        import torch
        # Import the model + feature converter directly rather than pathway_ranker, whose
        # module-level ``import hdbscan`` (optional clustering only) is not installed and
        # is not needed for scoring.
        from tree_lstm.treeLSTM_model import PathwayRankingModel
        from features.tree_to_treeLSTM_input import (
            convert_tree_to_singleinput, merge_into_batch)

        self._torch = torch
        self._convert = convert_tree_to_singleinput
        self._merge = merge_into_batch
        model = PathwayRankingModel(self.fp_size, self.lstm_size, encoder=True).to("cpu")
        state = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model.eval()
        self._model = model

    @property
    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def score_records(self, records: List[dict]) -> List[Optional[float]]:
        """One score per route record (``None`` for a route whose tree can't be built).

        The Tree-LSTM needs reaction depth >= 2 (at least two chained reactions); a
        1-step route is scored ``None``.
        """
        self._load()
        trees, idx = [], []
        for i, rec in enumerate(records):
            tree = retro_tree(rec)
            # the converter recurses only into children that are themselves reactions,
            # so a single-reaction tree yields an empty adjacency list -> skip
            if tree is None or not any(c["child"] for c in tree["child"]):
                continue
            trees.append(tree)
            idx.append(i)
        out: List[Optional[float]] = [None] * len(records)
        if not trees:
            return out
        try:
            batch = self._merge(
                [self._convert(t, fpsize=self.fp_size) for t in trees],
                to_tensor=True, device=self._torch.device("cpu"))
            with self._torch.no_grad():
                scores, _enc = self._model(
                    batch["pfp"], batch["rxnfp"], batch["adjacency_list"],
                    batch["node_order"], batch["edge_order"], batch["num_nodes"])
            for j, score in zip(idx, scores.view(-1).tolist()):
                out[j] = float(score)
        except Exception:
            pass
        return out
