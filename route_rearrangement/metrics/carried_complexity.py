"""Metric 8 — carried complexity / "build complexity late" (route-level feasibility).

A cornerstone of practical route design (convergency, step/redox economy): **complexity and
mass installed early must survive every subsequent operation.**  Each step you run on a large,
highly-functionalised, valuable intermediate is another chance to degrade it, another
chromatography of a precious compound, another selectivity problem.  A good ordering keeps the
molecule small and cheap for as long as possible and assembles the bulk of it late.

This is exactly what a per-molecule complexity/accessibility score cannot see: it judges each
intermediate in isolation, whereas feasibility depends on **how long each piece of complexity
is carried through the route**.  For every step *k* (1-indexed in synthesis order) we take the
heavy-atom gain ``Δ_k = heavy(product) − heavy(main substrate)`` — the mass that step installs
— and weight it by the number of operations it must then survive, ``remaining = N − k``::

    carried = Σ_k  max(Δ_k, 0) · (N − k)

Installing a big fragment at step 1 of a 10-step route costs 9× what installing it at step 9
does; deprotections/fragmentations (Δ ≤ 0) cost nothing here.  ``score`` negates the sum so
**higher is better** (complexity built later, carried less far).  Heavy-atom counts only —
rdkit, always available, strongly order-sensitive.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from rdkit import Chem

from .base import reactions

HIGHER_IS_BETTER = True


@lru_cache(maxsize=200_000)
def _heavy(smi: str) -> Optional[int]:
    m = Chem.MolFromSmiles(smi)
    return m.GetNumHeavyAtoms() if m is not None else None


def available() -> bool:
    return _heavy("CCO") is not None


def carried_complexity(record: dict) -> dict:
    """``{score, carried, mean_carried_size, per_step}`` for one route, or ``{}`` on failure."""
    rxns = reactions(record)
    n = len(rxns)
    if n == 0:
        return {}
    carried = 0.0
    sizes: List[int] = []
    per_step: List[dict] = []
    ok = True
    for r in rxns:
        hp = _heavy(r.product)
        hs = _heavy(r.main_reactant)
        if hp is None:
            ok = False
            continue
        sizes.append(hp)
        remaining = n - r.position                 # operations this new mass must survive
        delta = (hp - hs) if hs is not None else 0
        contrib = max(delta, 0) * remaining
        carried += contrib
        per_step.append({"position": r.position, "product_size": hp,
                         "delta": delta, "remaining": remaining,
                         "carried": contrib})
    if not sizes:
        return {}
    return {
        "carried": round(float(carried), 3),
        "mean_carried_size": round(sum(sizes) / len(sizes), 3),
        "complete": ok,
        "per_step": per_step,
        "score": round(-float(carried), 4),
    }
