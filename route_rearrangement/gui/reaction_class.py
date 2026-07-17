"""Lightweight reaction-class naming for the viewer's reaction boxes.

There is no licensed NameRXN model here, so the class is derived from what actually reacts:
``rxnutils``' ``detect_reactive_functions`` (the same engine the competing-sites metric uses)
plus a few heavy-atom / group-delta heuristics.  Common, unambiguous transformations get a
familiar name (amide coupling, O-/N-alkylation, esterification, sulfonylation, reduction,
deprotection, cross-coupling); everything else falls back to naming the reacting groups
(always honest — "amine + acid").  Returns ``""`` if rxnutils is unavailable.
"""

from __future__ import annotations

from typing import Dict, List, Set

from ..metrics import competing as _c

_PRETTY = {
    "amine_primary_aliphatic": "amine", "amine_secondary_aliphatic": "amine",
    "amine_aromatic": "aniline", "alcohol_aliphatic": "alcohol", "phenol": "phenol",
    "thiol": "thiol", "carboxylic_acid": "acid", "ester": "ester",
    "sulfonate_ester": "sulfonate", "alkyl_halide": "alkyl halide",
    "aryl_halide": "aryl halide", "aldehyde": "aldehyde", "ketone": "ketone",
}
_AMINES = {"amine_primary_aliphatic", "amine_secondary_aliphatic", "amine_aromatic"}
_OXY_NUC = {"alcohol_aliphatic", "phenol"}
_LEAVING = {"sulfonate_ester", "alkyl_halide"}


def available() -> bool:
    return _c.available()


def _pretty(groups: Set[str]) -> str:
    seen, out = set(), []
    for g in sorted(groups):
        p = _PRETTY.get(g, g)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return " + ".join(out)


def _heavy(smi: str) -> int:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return m.GetNumHeavyAtoms() if m is not None else 0


def _ring_count(smi: str) -> int:
    from rdkit import Chem
    total = 0
    for frag in smi.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is not None:
            total += m.GetRingInfo().NumRings()
    return total


def _has_nitro(smi: str) -> bool:
    from rdkit import Chem
    patt = Chem.MolFromSmarts("[NX3+](=O)[O-]")
    m = Chem.MolFromSmiles(smi)
    return bool(m is not None and patt is not None and m.HasSubstructMatch(patt))


def classify_reaction(reactants: List[str], product: str) -> str:
    """A short reaction-class name for ``reactants >> product`` (``""`` if unavailable)."""
    if not available() or not product:
        return ""
    reactants_smi = ".".join(r for r in reactants if r)
    try:
        reactive: Set[str] = _c._reactive_names(reactants_smi, product)
        present: Dict[str, int] = {}
        for r in reactants:
            for k, v in _c._present_counts(r).items():
                present[k] = present.get(k, 0) + v
        prod_present = _c._present_counts(product)
    except Exception:
        return ""

    has_amine = bool(reactive & _AMINES)
    has_oxy = bool(reactive & _OXY_NUC)
    has_acid = "carboxylic_acid" in reactive
    lg_present = bool(present.keys() & _LEAVING)
    main_heavy = max((_heavy(r) for r in reactants), default=0)
    prod_heavy = _heavy(product)

    # a sulfonate/halide newly appearing in the product = install of a leaving group
    installs_sulfonate = prod_present.get("sulfonate_ester", 0) > present.get("sulfonate_ester", 0)

    # ordered rules, most specific first
    if has_amine and has_acid:
        return "amide coupling"
    if has_oxy and has_acid:
        return "esterification"
    if installs_sulfonate:
        return "sulfonylation (–OMs/OTs install)"
    if has_amine and present.get("aryl_halide", 0):
        return "C–N coupling (SNAr/Buchwald)"
    if has_amine and lg_present:
        return "N-alkylation"
    if has_oxy and lg_present:
        return "O-alkylation (Williamson)"
    if has_amine and (present.get("aldehyde", 0) or present.get("ketone", 0)):
        return "reductive amination"
    # nitro -> amine: a nitro on a reactant is gone and an amine appears
    if any(_has_nitro(r) for r in reactants) and not _has_nitro(product) and \
            prod_present.get("amine_aromatic", 0) > present.get("amine_aromatic", 0):
        return "nitro reduction"
    if _ring_count(product) > _ring_count(reactants_smi):
        return "cyclization / ring formation"
    if prod_heavy <= main_heavy - 2 and (reactive & (_AMINES | _OXY_NUC | {"carboxylic_acid"})):
        return "deprotection / cleavage"
    if reactive:
        return f"reacts: {_pretty(reactive)}"
    return "transformation"
