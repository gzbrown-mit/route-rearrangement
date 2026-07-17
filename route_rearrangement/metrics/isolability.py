"""Metric 7 — intermediate isolability / bench-handleability (route-level, in-lab).

The metrics above ask how *complex* a molecule is; this one asks a different, practical
question: **every ordering forces a specific set of compounds to be isolated, purified, and
often stored between steps**, and a rearrangement changes *which* molecules those are.  An
intermediate that carries an unstable or hazardous functional group is a bench liability no
matter how "simple" it looks to a complexity model — an acyl chloride hydrolyses on the
rotovap, an azide or peroxide is a safety hold, a boronic acid protodeboronates on silica.

For each *isolated intermediate* of the ordering (the product of every step except the fixed
final target — that molecule is identical across all orderings, so it cannot discriminate)
we sum weighted liability hits from a transparent SMARTS table.  ``score`` negates the total
so **higher is better** (fewer / milder liabilities to handle), uniform with the other
metrics.  rdkit-only, so always available.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from .base import intermediates

HIGHER_IS_BETTER = True

# (name, SMARTS, weight, why).  Weight ≈ how much a chemist dreads isolating/storing it.
# High (≈3): genuine safety / rapid decomposition.  Medium (≈2): moisture/thermally labile.
# Low (≈1): survivable but a real handling nuisance (oxidation, protodeboronation, alkylator).
_LIABILITIES: List[Tuple[str, str, float, str]] = [
    ("acyl_halide",   "[CX3](=[OX1])[F,Cl,Br,I]",              3.0, "hydrolyses; moisture-sensitive"),
    ("anhydride",     "[CX3](=[OX1])[OX2][CX3]=[OX1]",         2.0, "moisture-sensitive acylating agent"),
    ("azide",         "[$([NX1]=[NX2]=[NX1]),$(N=[N+]=[N-]),$([N-]=[N+]=[N-])]", 3.0, "shock/heat-sensitive; safety hold"),
    ("diazo",         "[$([CX3]=[NX2+]=[NX1-]),$([#6]=[N+]=[N-])]", 3.0, "explosive / highly reactive"),
    ("peroxide",      "[OX2][OX2]",                            3.0, "explosive; incompatible with heat"),
    ("isocyanate",    "[NX2]=[CX2]=[OX1]",                     2.5, "moisture-sensitive; toxic"),
    ("n_nitroso",     "[NX3][NX2]=[OX1]",                      2.5, "genotoxic; regulatory liability"),
    ("aldehyde",      "[CX3H1](=O)[#6]",                       1.0, "air-oxidation / polymerisation"),
    ("epoxide",       "[OX2r3]1[#6r3][#6r3]1",                 1.0, "ring-opens with trace nucleophile/acid"),
    ("boronic_acid",  "[BX3]([OX2H])[OX2H]",                   1.0, "protodeboronates; hard to purify"),
    ("alkyl_sulfonate", "[#6;!$([#6]=O)][OX2][SX4](=[OX1])(=[OX1])[#6]", 1.0, "alkylating / potential mutagen"),
    ("hydrolyzable_imine", "[CX3;!$(C=[!#6])]=[NX2][#6;!$([#6]=[#7,#8,#16])]", 1.0, "hydrolyses on silica / aqueous workup"),
]


@lru_cache(maxsize=1)
def _patterns() -> List[Tuple[str, "Chem.Mol", float]]:
    out = []
    for name, sma, w, _why in _LIABILITIES:
        p = Chem.MolFromSmarts(sma)
        if p is not None:
            out.append((name, p, w))
    return out


def available() -> bool:
    try:
        return bool(_patterns()) and Chem.MolFromSmiles("CCO") is not None
    except Exception:
        return False


@lru_cache(maxsize=200_000)
def _liabilities(smi: str) -> Tuple[Tuple[str, int, float], ...]:
    """Liability hits on one molecule: ``((name, count, weight), ...)``."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ()
    hits = []
    for name, patt, w in _patterns():
        n = len(m.GetSubstructMatches(patt, uniquify=True))
        if n:
            hits.append((name, n, w))
    return tuple(hits)


def isolability(record: dict) -> dict:
    """``{score, total_liability, peak_liability, per_intermediate, groups}`` for one route.

    Scores the transient isolated intermediates (every step's product except the final
    target).  Empty ``{}`` for a 1-step route (no transient intermediate to discriminate).
    """
    inter = intermediates(record)[:-1]          # drop the fixed final target
    if not inter:
        return {}
    per: List[dict] = []
    groups: Dict[str, int] = {}
    total = 0.0
    peak = 0.0
    for smi in inter:
        hits = _liabilities(smi)
        load = sum(w * c for _n, c, w in hits)
        total += load
        peak = max(peak, load)
        for name, c, _w in hits:
            groups[name] = groups.get(name, 0) + c
        per.append({"smiles": smi, "liability": round(load, 3),
                    "groups": [n for n, _c, _w in hits]})
    return {
        "total_liability": round(total, 3),
        "peak_liability": round(peak, 3),
        "groups": groups,
        "per_intermediate": per,
        "score": round(-total, 4),
    }
