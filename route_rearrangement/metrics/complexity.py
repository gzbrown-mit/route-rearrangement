"""Metric 4 — SCScore complexity trajectory (Coley et al.).

A well-designed synthesis builds molecular complexity roughly monotonically toward the
target.  Using SCScore (miniASKCOS, numpy-only) on each step:

* ``peak`` — the maximum intermediate complexity the route passes through (lower is a
  gentler climb);
* ``inversions`` — the number of steps whose product is *less* complex than its main
  substrate, i.e. the route goes backward in complexity (deprotections / poorly-timed
  functional-group interconversions).  Fewer is better.

Both are order-sensitive.  ``score`` is ``-(inversions + 0.1 * peak)`` so higher is
better and inversions dominate.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .base import reactions

SCSCORE_MODEL_PATH = os.environ.get(
    "SCSCORE_MODEL_PATH",
    str(Path.home() / "miniASKCOS" / "askcos" / "data" / "models" / "scscore"
        / "model_1024bool.pickle"),
)
MINIASKCOS_PATH = os.environ.get("MINIASKCOS_PATH", str(Path.home() / "miniASKCOS"))

HIGHER_IS_BETTER = True


@lru_cache(maxsize=1)
def _scorer():
    import sys
    if MINIASKCOS_PATH not in sys.path:
        sys.path.insert(0, MINIASKCOS_PATH)
    from askcos.modules.scscore import SCScorer
    return SCScorer(SCSCORE_MODEL_PATH)


@lru_cache(maxsize=200_000)
def _sc(smi: str) -> Optional[float]:
    try:
        return float(_scorer().get_score(smi))
    except Exception:
        return None


def available() -> bool:
    try:
        return _sc("CCO") is not None
    except Exception:
        return False


def complexity_profile(record: dict) -> dict:
    """``{peak, inversions, trajectory, score}`` for one route, or empty on failure."""
    rxns = reactions(record)
    traj: List[float] = []
    inversions = 0
    ok = True
    for r in rxns:
        cp = _sc(r.product)
        cr = _sc(r.main_reactant)
        if cp is None:
            ok = False
            continue
        traj.append(round(cp, 3))
        if cr is not None and cp + 1e-9 < cr:
            inversions += 1
    if not traj:
        return {}
    peak = max(traj)
    return {
        "peak": round(peak, 3),
        "inversions": inversions,
        "trajectory": traj,
        "complete": ok,
        "score": round(-(inversions + 0.1 * peak), 4),
    }
