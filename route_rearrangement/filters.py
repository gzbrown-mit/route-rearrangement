"""Acceptance gates and soft flags for materialized routes.

Hard gates (an ordering's variant is rejected):
* every molecule sanitizes (enforced upstream — :func:`chain.split_outcome` drops
  outcomes with uncanonicalizable fragments; asserted again here);
* chain connectivity — the rebuilt route passes
  :func:`synthesis_extraction.dependency.propagate.disconnected_edges` (catches
  chain-misidentification bugs);
* duplicates — distinct orderings/outcome branches that produce the identical chemistry
  collapse onto one record.

Soft flags (recorded, never filtered):
* ``fg_risk`` — a functional group on a *new* intermediate gets a ``survives=False``
  verdict from :func:`synthesis_extraction.compatibility.compat.fg_survives` against a
  later step's reaction profile; abstentions are listed separately;
* ``inexact_side_match`` — some step's side reactants were identified by similarity,
  not exact match.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from . import deps  # noqa: F401
from .materialize import MaterializedRoute
from .templates import StepTemplate
from synthesis_extraction.dependency.propagate import disconnected_edges


def rebuilt_full_graph(route: MaterializedRoute) -> dict:
    """A ``full_graph``-shaped dict (map-free) of a materialized linear route.

    Node ids follow the BFS convention: the last-performed step is the root (id 1),
    deeper steps have larger ids.
    """
    n = len(route.steps)
    nodes, edges = [], []
    for rec in route.steps:                       # rec.position 1 = deepest = id n
        nid = n - rec.position + 1
        nodes.append({"id": nid, "SMILES": rec.new_rxn, "rxn_index": rec.orig_rxn_index})
        if rec.position > 1:
            edges.append([nid + 1, nid])          # child (earlier step) -> parent
    return {"nodes": nodes, "edges": edges}


def passes_connectivity(route: MaterializedRoute) -> Tuple[bool, List]:
    bad = disconnected_edges(rebuilt_full_graph(route))
    return (not bad, bad)


def sanitizes(route: MaterializedRoute) -> bool:
    for rec in route.steps:
        for part in rec.new_rxn.split(">>"):
            for frag in part.split("."):
                if frag and Chem.MolFromSmiles(frag) is None:
                    return False
    return True


def dedup_key(route: MaterializedRoute) -> Tuple[str, ...]:
    return tuple(rec.new_rxn for rec in route.steps)


# ---------------------------------------------------------------------------
# fg_risk soft flag
# ---------------------------------------------------------------------------
def fg_risk_flags(route: MaterializedRoute, templates: Dict[int, StepTemplate],
                  matrix=None) -> Tuple[List[dict], List[dict]]:
    """``(risks, abstentions)`` for the route's new intermediates.

    The intermediate produced at position *k* is exposed to the conditions of every
    later step; each of its functional groups is judged with ``fg_survives`` against
    the later step's reaction profile (taken from the *original* step SMILES — the
    conditions travel with the step).  Never raises; returns empty lists on any
    library/profile failure.
    """
    try:
        from synthesis_extraction.compatibility.fg_library import enumerate_fgs, load_fg_library
        from synthesis_extraction.compatibility.compat import fg_survives, load_rules
        from synthesis_extraction.compatibility.reaction_profile import reaction_profile

        library = load_fg_library()
        rules = load_rules()
    except Exception:
        return [], []

    profiles = {}
    for rec in route.steps:
        tpl = templates.get(rec.orig_step_id)
        if tpl is None:
            continue
        try:
            profiles[rec.position] = reaction_profile(tpl.orig_rxn)
        except Exception:
            profiles[rec.position] = None

    risks: List[dict] = []
    abstains: List[dict] = []
    for rec in route.steps:
        mol = Chem.MolFromSmiles(rec.new_product) if rec.new_product else None
        if mol is None:
            continue
        try:
            fgs = {m.name for m in enumerate_fgs(mol, library)}
        except Exception:
            continue
        for later in route.steps:
            if later.position <= rec.position:
                continue
            profile = profiles.get(later.position)
            if profile is None:
                continue
            for fg in sorted(fgs):
                try:
                    v = fg_survives(fg, profile, matrix=matrix, rules=rules)
                except Exception:
                    continue
                entry = {"fg": fg, "intermediate_position": rec.position,
                         "at_position": later.position, "basis": v.basis}
                if v.survives is False:
                    risks.append(entry)
                elif v.survives is None:
                    abstains.append(entry)
    return risks, abstains


def evaluate(route: MaterializedRoute, templates: Dict[int, StepTemplate],
             matrix=None, with_fg: bool = True) -> Optional[dict]:
    """Apply hard gates; return the flags dict for an accepted route, else ``None``."""
    if not route.ok:
        return None
    if not sanitizes(route):
        return None
    ok, bad = passes_connectivity(route)
    if not ok:
        return None
    flags: dict = {
        "inexact_side_match": [r.position for r in route.steps if not r.exact_side_match],
    }
    if with_fg:
        risks, abstains = fg_risk_flags(route, templates, matrix=matrix)
        flags["fg_risk"] = risks
        flags["fg_abstain"] = abstains
    return flags
