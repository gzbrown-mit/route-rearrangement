"""Acceptance gates and soft flags for materialized routes.

Hard gates (an ordering's variant is rejected):
* every molecule sanitizes (enforced upstream — :func:`chain.route_outcomes` drops
  outcomes with uncanonicalizable fragments; asserted again here);
* connectivity — the rebuilt route (a tree for convergent routes) passes
  :func:`synthesis_extraction.dependency.propagate.disconnected_edges` (catches
  fragment-misrouting bugs);
* duplicates — distinct orderings/outcome branches that produce the identical chemistry
  collapse onto one record.

Soft flags (recorded, never filtered):
* ``fg_risk`` — a functional group on a *new* intermediate gets a ``survives=False``
  verdict from :func:`synthesis_extraction.compatibility.compat.fg_survives` against a
  downstream step's reaction profile.  Downstream = the steps on the intermediate's
  path to the root of the **new tree** — steps on a parallel branch never see it;
* ``inexact_side_match`` — some step's fragments were routed by similarity, not exact
  match;
* ``migrated_steps`` — steps whose parent in the new tree differs from the original
  (convergence-point migration happened);
* ``sm_mismatch`` — the route's starting-material multiset differs from the original's
  (order-invariance says they should agree; a mismatch means routing was heuristic).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem

from . import deps  # noqa: F401
from .materialize import MaterializedRoute, StepRecord
from .templates import StepTemplate, route_sm_budget
from synthesis_extraction.dependency.propagate import disconnected_edges


def _node_ids(route: MaterializedRoute) -> Dict[int, int]:
    """``{orig_step_id: rebuilt node id}`` — BFS convention: the last-performed step is
    the root (id 1), deeper steps have larger ids."""
    n = len(route.steps)
    return {rec.orig_step_id: n - rec.position + 1 for rec in route.steps}


def _parent_of(rec: StepRecord, by_position: Dict[int, StepRecord]) -> Optional[int]:
    """The step's parent (orig id) in the new route; linear-chain fallback for records
    predating the frontier engine (no parent pointers stored)."""
    if rec.parent_step_id is not None:
        return rec.parent_step_id
    nxt = by_position.get(rec.position + 1)
    return nxt.orig_step_id if nxt is not None else None


def rebuilt_full_graph(route: MaterializedRoute) -> dict:
    """A ``full_graph``-shaped dict (map-free) of a materialized route — a path for a
    linear route, a genuine tree when branches are open (child→parent edges follow the
    frontier walk's consumer pointers)."""
    ids = _node_ids(route)
    by_position = {rec.position: rec for rec in route.steps}
    nodes, edges = [], []
    for rec in route.steps:
        nid = ids[rec.orig_step_id]
        nodes.append({"id": nid, "SMILES": rec.new_rxn, "rxn_index": rec.orig_rxn_index})
        parent = _parent_of(rec, by_position)
        if parent is not None and parent in ids:
            edges.append([nid, ids[parent]])      # child (earlier step) -> parent
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


def dedup_key(route: MaterializedRoute) -> Tuple[Tuple[str, Optional[int]], ...]:
    by_position = {rec.position: rec for rec in route.steps}
    return tuple((rec.new_rxn, _parent_of(rec, by_position)) for rec in route.steps)


def _ancestor_positions(route: MaterializedRoute) -> Dict[int, List[int]]:
    """``{position: [ancestor positions]}`` — the steps each intermediate must survive,
    i.e. the path from the step to the root of the new tree (linear: all later steps)."""
    by_position = {rec.position: rec for rec in route.steps}
    by_step = {rec.orig_step_id: rec for rec in route.steps}
    out: Dict[int, List[int]] = {}
    for rec in route.steps:
        chain: List[int] = []
        cur, hops = rec, 0
        while hops <= len(route.steps):
            parent = _parent_of(cur, by_position)
            if parent is None or parent not in by_step:
                break
            cur = by_step[parent]
            chain.append(cur.position)
            hops += 1
        out[rec.position] = chain
    return out


# ---------------------------------------------------------------------------
# fg_risk soft flag
# ---------------------------------------------------------------------------
def _mapped(smiles: str):
    """Parse *smiles* with every atom carrying a map number.

    ``enumerate_fgs`` keys its matches by atom-map number and silently returns no
    matches on a map-free molecule — and materialized intermediates are map-free by
    construction, so they must be re-numbered here or the whole flag is inert."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    return mol


def fg_risk_flags(route: MaterializedRoute, templates: Dict[int, StepTemplate],
                  matrix=None) -> Tuple[List[dict], List[dict]]:
    """``(risks, abstentions)`` for the route's new intermediates.

    The intermediate produced at position *k* is exposed to the conditions of every
    step on its path to the root of the new tree (its ancestors — a parallel branch
    runs in a different flask and never sees it); each of its functional groups is
    judged with ``fg_survives`` against the downstream step's reaction profile (taken
    from the *original* step SMILES — the conditions travel with the step).  Never
    raises; returns empty lists on any library/profile failure.
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

    ancestors = _ancestor_positions(route)
    risks: List[dict] = []
    abstains: List[dict] = []
    for rec in route.steps:
        mol = _mapped(rec.new_product) if rec.new_product else None
        if mol is None:
            continue
        try:
            fgs = {m.name for m in enumerate_fgs(mol, library)}
        except Exception:
            continue
        for later_pos in ancestors.get(rec.position, []):
            profile = profiles.get(later_pos)
            if profile is None:
                continue
            for fg in sorted(fgs):
                try:
                    v = fg_survives(fg, profile, matrix=matrix, rules=rules)
                except Exception:
                    continue
                entry = {"fg": fg, "intermediate_position": rec.position,
                         "at_position": later_pos, "basis": v.basis}
                if v.survives is False:
                    risks.append(entry)
                elif v.survives is None:
                    abstains.append(entry)
    return risks, abstains


def evaluate(route: MaterializedRoute, templates: Dict[int, StepTemplate],
             matrix=None, with_fg: bool = True,
             orig_parents: Optional[Dict[int, Optional[int]]] = None) -> Optional[dict]:
    """Apply hard gates; return the flags dict for an accepted route, else ``None``.

    Structural gates only.  Chemical feasibility (:mod:`.feasibility`) is deliberately
    *not* applied here: the pipeline stays a neutral generator, and the chemistry audit
    runs over the results afterwards (:mod:`.audit`), so a check can be retuned without
    regenerating the corpus and can never silently discard a route."""
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
    if orig_parents is not None:
        by_position = {rec.position: rec for rec in route.steps}
        flags["migrated_steps"] = sorted(
            rec.orig_step_id for rec in route.steps
            if _parent_of(rec, by_position) != orig_parents.get(rec.orig_step_id))
    want = route_sm_budget(templates)
    got = Counter(route.starting_materials)
    if got != want:
        flags["sm_mismatch"] = {
            "missing": sorted((want - got).elements()),
            "extra": sorted((got - want).elements()),
        }
    if with_fg:
        risks, abstains = fg_risk_flags(route, templates, matrix=matrix)
        flags["fg_risk"] = risks
        flags["fg_abstain"] = abstains
    return flags
