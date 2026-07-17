"""Metric 6 — competing reactivity sites (MolecularAI reaction_utils / rxnutils).

A rearrangement changes *when* each group is installed and protected, so a step may end up
running on a substrate that still carries a reactive group the literature order had already
consumed or protected.  For every step this metric asks, via ``rxnutils``:

* ``match_smarts`` — every reactive/sensitive functional group present on the reactants;
* ``detect_reactive_functions`` — the group(s) that actually change in the reaction.

Anything present but **not** reacting is a **competing site** — a selectivity liability, or
a group that may need protecting.  Two things are singled out:

* **same-class competition** — the reacting group's class appears more than once (e.g. two
  free amines, only one should acylate);
* **leaving-group / ester exposed to a condensation** — a mesylate/sulfonate, alkyl halide,
  or ester sitting on the substrate while an amide/ester bond is being formed (exactly the
  "can the -OMs survive / does it need removing before the condensation?" question).

``score`` = ``-(number of competing-site exposures)`` (higher = fewer = better), so it ranks
uniformly with the other metrics.  ``per_step`` and ``leaving_group_exposures`` are recorded
for inspection.  Uses the ``rxnutils`` install in the env (falls back to unavailable if the
SMARTS library or package is missing — never crashes the pipeline).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

from .base import reactions

SMARTS_PATH = os.environ.get(
    "COMPETING_SMARTS_PATH",
    str(Path(__file__).resolve().parents[1] / "data" / "competing_smarts.json"))

HIGHER_IS_BETTER = True

# nucleophiles whose extra copies are a chemoselectivity concern (two amines, one should react)
_NUCLEOPHILES: Set[str] = {
    "amine_primary_aliphatic", "amine_secondary_aliphatic", "amine_aromatic",
    "alcohol_aliphatic", "phenol", "thiol", "carboxylic_acid",
}
# electrophilic leaving groups a nucleophile could displace — the -OMs worry.  (Plain esters
# are excluded: tBu/Et esters are stable to amide coupling and are common protecting groups.)
_LEAVING = {"sulfonate_ester", "alkyl_halide"}
# electrophilic carbonyls that could be attacked by an amine/alcohol present in the step
_ELECTROPHILES = {"aldehyde", "ketone"}
# groups whose *survival through a reacting step* counts as a competing site
_SENSITIVE: Set[str] = _NUCLEOPHILES | _LEAVING | _ELECTROPHILES
# a step deploys a nucleophile (amide/ester/ether bond formation) if one of these reacts
_ACYL_REACTIVE = {"carboxylic_acid", "amine_primary_aliphatic",
                  "amine_secondary_aliphatic", "amine_aromatic", "alcohol_aliphatic", "phenol"}


@lru_cache(maxsize=1)
def _library():
    from rxnutils.chem.smartslib import SmartsLibrary
    return SmartsLibrary(SMARTS_PATH, "competing")


def available() -> bool:
    try:
        _library().match_smarts("CCO")
        return True
    except Exception:
        return False


def _present_counts(smiles: str) -> Dict[str, int]:
    try:
        return {k: v.number_of_hits for k, v in _library().match_smarts(smiles).items()}
    except Exception:
        return {}


def _reactive_names(reactants_smi: str, product_smi: str) -> Set[str]:
    """Functional-group classes that change in ``reactants>>product``."""
    from rxnutils.chem.reaction import ChemicalReaction
    try:
        rxn = ChemicalReaction(f"{reactants_smi}>>{product_smi}", clean_smiles=False)
        names, _ = _library().detect_reactive_functions(
            rxn, sort=True, add_none=False, target_size=0)
        return {n for n in names if n and n != "None"}
    except Exception:
        return set()


def _step_competing(reactants: List[str], product: str) -> dict:
    """Competing sites for one step: sensitive groups that **survive into the product** while
    a reaction happens elsewhere.

    Counting survivors (product hits), not reactant hits, is what makes a ring-closing
    bis-alkylation (both leaving groups consumed -> 0 survive) *not* a competing site, while a
    genuinely untouched second amine or a mesylate that rides through *is* one.
    """
    reactants_smi = ".".join(reactants)
    reactive = _reactive_names(reactants_smi, product)
    # only a step that actually deploys a nucleophile can threaten a competing site
    nucleophile_active = bool(reactive & _ACYL_REACTIVE)
    survivors = _present_counts(product)

    competing: List[dict] = []
    if nucleophile_active:
        for group, count in survivors.items():
            if group not in _SENSITIVE or count <= 0:
                continue
            if group in _NUCLEOPHILES and group not in reactive:
                sev = "high"           # a bystander nucleophile survived a nucleophile-active step
            elif group in _NUCLEOPHILES:
                # the reacting nucleophile class, but an extra copy survived -> selectivity risk
                if count < 1:
                    continue
                sev = "high"
            elif group in _LEAVING:
                sev = "high"           # the -OMs worry: leaving group survives a nucleophilic step
            else:                       # aldehyde/ketone bystander electrophile
                sev = "medium"
            competing.append({"group": group, "count": count,
                              "same_class_as_reacting": group in reactive,
                              "severity": sev})
    return {"reactive": sorted(reactive), "nucleophile_active": nucleophile_active,
            "competing": competing,
            "n_competing": sum(c["count"] for c in competing)}


def competing_sites(record: dict) -> dict:
    """``{score, n_competing, per_step, leaving_group_exposures}`` for one route."""
    per_step: List[dict] = []
    total = 0
    leaving: List[dict] = []
    for r in reactions(record):
        step = _step_competing(r.reactants, r.product)
        total += step["n_competing"]
        entry = {"position": r.position, **step}
        per_step.append(entry)
        for c in step["competing"]:
            if c["group"] in _LEAVING and step["nucleophile_active"]:
                leaving.append({"position": r.position, "group": c["group"],
                                "count": c["count"]})
    if not per_step:
        return {}
    return {
        "n_competing": total,
        "n_high_severity": sum(1 for s in per_step for c in s["competing"]
                               if c["severity"] == "high"),
        "leaving_group_exposures": leaving,
        "per_step": per_step,
        "score": round(-float(total), 4),
    }
