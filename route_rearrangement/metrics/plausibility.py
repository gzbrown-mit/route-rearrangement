"""Metric 2 — learned reaction plausibility (miniASKCOS template relevance).

The rearranged reactions are novel (each step runs on a substrate it never saw in the
literature route), so an independent check that every step is a *real, known kind of
transformation* is exactly what "make sure they are logical" needs.  For each step's
product we run the pistachio template-relevance model, take its ranked retro templates,
and find the total model probability of the templates whose application reproduces the
step's actual reactants.  A step the model can explain with a high-probability known
template is plausible; one no template reproduces scores 0.

``mean`` and ``min`` over the route's steps are reported; ``score`` = ``mean`` (higher is
better).  This is the heaviest metric (a ~900 MB model); it is opt-in (``--plausibility``)
and loads lazily.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from rdkit import Chem

from .base import reactions
from .complexity import MINIASKCOS_PATH

MODEL_PATH = os.environ.get(
    "TEMPLREL_MODEL_PATH",
    str(Path(MINIASKCOS_PATH) / "askcos/data/models/template_relevance/pistachio/model_latest.pt"))
TEMPLATE_PATH = os.environ.get(
    "TEMPLREL_TEMPLATE_PATH",
    str(Path(MINIASKCOS_PATH) / "askcos/data/models/template_relevance/pistachio/templates.jsonl"))

HIGHER_IS_BETTER = True


def _canon_set(smi: str) -> frozenset:
    out = set()
    for frag in smi.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is not None:
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            out.add(Chem.MolToSmiles(m))
    return frozenset(out)


class PlausibilityScorer:
    """Lazy-loading template-relevance plausibility scorer."""

    def __init__(self, top_k: int = 50, model_path: Optional[str] = None,
                 template_path: Optional[str] = None):
        self.top_k = top_k
        self.model_path = model_path or MODEL_PATH
        self.template_path = template_path or TEMPLATE_PATH
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return
        import sys
        if MINIASKCOS_PATH not in sys.path:
            sys.path.insert(0, MINIASKCOS_PATH)
        from askcos.modules.template_relevance import TemplateRelevancePredictor
        self._predictor = TemplateRelevancePredictor(
            self.model_path, self.template_path, device="cpu")

    @property
    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    @lru_cache(maxsize=50_000)
    def _step_plausibility(self, product: str, reactants_key: frozenset) -> float:
        pred = self._predictor.predict([product], max_num_templates=self.top_k)[0]
        target = reactants_key
        total = 0.0
        for reac, score in zip(pred.get("reactants", []), pred.get("scores", [])):
            if reac and _canon_set(reac) == target:
                total += float(score)
        return total

    def score(self, record: dict) -> dict:
        """``{mean, min, per_step, score}`` for one route, or empty on failure."""
        self._load()
        per_step: List[float] = []
        for r in reactions(record):
            try:
                p = self._step_plausibility(r.product, _canon_set(".".join(r.reactants)))
            except Exception:
                continue
            per_step.append(round(p, 4))
        if not per_step:
            return {}
        return {
            "mean": round(sum(per_step) / len(per_step), 4),
            "min": round(min(per_step), 4),
            "per_step": per_step,
            "score": round(sum(per_step) / len(per_step), 4),
        }
