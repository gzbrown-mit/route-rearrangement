"""Metric 5 — synthetic-accessibility bottleneck (Ertl & Schuffenhauer SAscore).

The hardest-to-make intermediate a route must pass through.  Rearranging steps changes
which intermediates exist, so this discriminates orderings: a route that threads through
an awkward, low-accessibility intermediate is worse.  ``bottleneck`` is the max SAscore
over the route's intermediates (SAscore is 1 = easy … 10 = hard), and ``score`` is its
negation so higher is better.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Optional

from rdkit import Chem

from .base import intermediates

HIGHER_IS_BETTER = True


@lru_cache(maxsize=1)
def _sascorer():
    from rdkit.Chem import RDConfig
    sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_path not in sys.path:
        sys.path.append(sa_path)
    import sascorer
    return sascorer


@lru_cache(maxsize=200_000)
def _sa(smi: str) -> Optional[float]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return float(_sascorer().calculateScore(m))
    except Exception:
        return None


def available() -> bool:
    try:
        return _sa("CCO") is not None
    except Exception:
        return False


def accessibility(record: dict) -> dict:
    vals = [v for v in (_sa(s) for s in intermediates(record)) if v is not None]
    if not vals:
        return {}
    bottleneck = max(vals)
    return {
        "bottleneck": round(bottleneck, 3),
        "mean": round(sum(vals) / len(vals), 3),
        "score": round(-bottleneck, 4),
    }
