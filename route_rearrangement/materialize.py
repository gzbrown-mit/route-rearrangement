"""Backward materialization of one ordering of a route's steps.

Given a unified-map ``full_graph``, per-step retro templates and a valid ordering
(earliest-performed first, a linear extension of the dependency partial order), walk
backward from the target: undo the last-performed step first by applying its retro
template to the substrate *as it exists in the rearranged route*.  The walk carries a
**frontier** — the multiset of open intermediates not yet disconnected (one molecule
for a linear route, several while convergent branches are open).  Undoing a step
removes the frontier molecule the template disconnected and deposits its precursor
fragments: starting materials leave the frontier for good, synthesized precursors stay
to be disconnected by a later-undone step (:mod:`.chain`).  The new route's tree
topology is *emergent*: each deposited fragment remembers which step consumed it, so a
coupling undone late (performed early) hands the following steps the combined molecule
— convergence-point migration.

An ordering whose required context is absent at some position (the template yields no
outcome on any frontier molecule) is pruned there — the built-in chemical veto.  Two
conservation invariants prune structurally impossible walks early: the frontier can
never hold more open molecules than there are steps left to undo, and the final step
must empty it.

Multiple template outcomes / ambiguous fragment routings make the walk a small beam
search; each surviving completed leaf is one materialized *variant* of the ordering.
:func:`replay_identity` runs the same walk on the chemist's original order and demands
the original intermediates *and the original tree* back — the calibration gate that
must pass before any rearrangement of that route is trusted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from .chain import route_outcomes
from .templates import (
    StepTemplate,
    apply_retro,
    canonicalize_smiles,
    original_parents,
    route_sm_budget,
)
from synthesis_extraction.step_classification.footprint import main_product_mol, split_reaction

Score = Tuple[int, float]   # (n_inexact_side_matches, -sum_similarity) — lower is better


@dataclass
class StepRecord:
    """One position of a materialized route (position 1 = first-performed step)."""

    position: int
    orig_step_id: int
    orig_rxn_index: int
    retro_smarts: str
    new_rxn: str                       # precursors>>product (canonical, map-free)
    chain_precursor: Optional[str]     # sole synthesized precursor, None otherwise (compat)
    side_reactants: List[str]          # starting-material fragments of this step
    new_product: str
    outcome_rank: int
    n_outcomes: int
    exact_side_match: bool
    sim_score: float
    parent_step_id: Optional[int] = None   # step consuming this step's product in the NEW route
    synth_precursors: List[str] = field(default_factory=list)  # open fragments this step deposits


@dataclass
class MaterializedRoute:
    ordering: List[int]                # step ids, earliest-performed first
    status: str                        # "ok" | failure reason
    target: str
    steps: List[StepRecord] = field(default_factory=list)   # position order
    starting_materials: List[str] = field(default_factory=list)
    score: Score = (0, 0.0)
    failed_position: Optional[int] = None
    failed_step_id: Optional[int] = None
    failure_intermediate: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class OpenMol:
    """One open frontier intermediate.  *consumer* is the step (orig id) that consumes
    it in the new route — the step whose undo deposited it; ``None`` for the target."""

    smiles: str
    consumer: Optional[int]


@dataclass
class _State:
    frontier: Tuple[OpenMol, ...]      # open intermediates still to be disconnected
    steps_rev: List[StepRecord]        # records collected last-position-first
    sm_budget: Counter                 # unconsumed purchasable blocks of the route
    n_inexact: int = 0
    sum_sim: float = 0.0

    @property
    def score(self) -> Score:
        return (self.n_inexact, -self.sum_sim)


def _frontier_key(frontier: Tuple[OpenMol, ...]) -> Tuple:
    return tuple(sorted((m.smiles, m.consumer if m.consumer is not None else -1)
                        for m in frontier))


def route_target(full_graph: dict) -> Optional[str]:
    """Canonical map-free final product of the route (main product of node 1)."""
    for n in full_graph.get("nodes", []):
        if int(n["id"]) == 1:
            _, pb = split_reaction(n.get("SMILES", ""))
            pm = main_product_mol(pb) if pb else None
            return canonicalize_smiles(Chem.MolToSmiles(pm)) if pm is not None else None
    return None


def advance_states(states: List[_State], tpl: StepTemplate, position: int, *,
                   remaining_after: int, beam: int,
                   max_outcomes: int) -> Tuple[List[_State], bool]:
    """Expand every beam state by one backward application of *tpl*.

    The template is tried on every frontier molecule; *remaining_after* steps are still
    to be undone afterwards, which drives the conservation pruning (the frontier can
    hold at most that many open molecules, and the last step must empty it).  Returns
    ``(new_states, any_outcome)`` — deterministic given the sequence of steps already
    undone, which is what lets the DFS engine share suffixes with the naive one.
    """
    nxt: List[_State] = []
    any_outcome = False
    for st in states:
        if remaining_after == 0 and len(st.frontier) != 1:
            continue                   # the final undo must empty the frontier
        for mi, mol in enumerate(st.frontier):
            outcomes = apply_retro(tpl.retro_smarts, mol.smiles, max_outcomes=max_outcomes)
            if outcomes:
                any_outcome = True
            rest = st.frontier[:mi] + st.frontier[mi + 1:]
            for rank, outcome in enumerate(outcomes):
                for routed in route_outcomes(outcome, tpl,
                                             last_step=remaining_after == 0,
                                             sm_budget=st.sm_budget):
                    n_open = len(rest) + len(routed.synth)
                    if remaining_after == 0:
                        if n_open:
                            continue
                    elif not (1 <= n_open <= remaining_after):
                        continue       # orphaned or unsatisfiable frontier
                    rec = StepRecord(
                        position=position,
                        orig_step_id=tpl.step_id,
                        orig_rxn_index=tpl.rxn_index,
                        retro_smarts=tpl.retro_smarts,
                        new_rxn=".".join(sorted(routed.sm + routed.synth))
                                + ">>" + mol.smiles,
                        chain_precursor=routed.synth[0] if len(routed.synth) == 1 else None,
                        side_reactants=list(routed.sm),
                        new_product=mol.smiles,
                        outcome_rank=rank,
                        n_outcomes=len(outcomes),
                        exact_side_match=routed.exact_side_match,
                        sim_score=routed.sim_score,
                        parent_step_id=mol.consumer,
                        synth_precursors=list(routed.synth),
                    )
                    budget = st.sm_budget.copy()
                    for s in routed.sm:
                        if budget.get(s, 0) > 0:
                            budget[s] -= 1
                    nxt.append(_State(
                        frontier=rest + tuple(OpenMol(s, tpl.step_id)
                                              for s in routed.synth),
                        steps_rev=st.steps_rev + [rec],
                        sm_budget=budget,
                        n_inexact=st.n_inexact + (0 if routed.exact_side_match else 1),
                        sum_sim=st.sum_sim + routed.sim_score,
                    ))
    # dedupe states on (frontier, last reaction) then keep the best `beam`
    nxt.sort(key=lambda s: s.score)
    seen, pruned = set(), []
    for st in nxt:
        key = (_frontier_key(st.frontier), st.steps_rev[-1].new_rxn)
        if key in seen:
            continue
        seen.add(key)
        pruned.append(st)
        if len(pruned) >= beam:
            break
    return pruned, any_outcome


def finalize_states(states: List[_State], ordering: List[int], target: str) -> List[MaterializedRoute]:
    """Turn completed beam states into :class:`MaterializedRoute` variants, best first."""
    out: List[MaterializedRoute] = []
    for st in sorted(states, key=lambda s: s.score):
        steps = list(reversed(st.steps_rev))          # position 1 first
        sms: List[str] = []
        for rec in steps:                             # every SM-routed fragment
            sms.extend(rec.side_reactants)
        out.append(MaterializedRoute(
            ordering=list(ordering), status="ok", target=target,
            steps=steps, starting_materials=sorted(sms), score=st.score))
    return out


def check_templates(templates: Dict[int, StepTemplate],
                    ordering: List[int]) -> Optional[Tuple[int, int]]:
    """``(step_id, position)`` of the first step lacking a template, or ``None``."""
    for i, sid in enumerate(ordering):
        tpl = templates.get(sid)
        if tpl is None or not tpl.retro_smarts:
            return sid, i + 1
    return None


def frontier_smiles(states: List[_State], fallback: str) -> str:
    """Best state's open intermediates as a ``.``-joined string (for failure records)."""
    if not states:
        return fallback
    return ".".join(sorted(m.smiles for m in states[0].frontier)) or fallback


def materialize_ordering(full_graph: dict, templates: Dict[int, StepTemplate],
                         ordering: List[int], *, beam: int = 3,
                         max_outcomes: int = 20) -> List[MaterializedRoute]:
    """Materialize one ordering.  Returns completed variants sorted best-first, or a
    single failure record (``status != "ok"``) if no variant completes."""
    target = route_target(full_graph)
    if target is None:
        return [MaterializedRoute(ordering=list(ordering), status="no_target", target="")]
    missing = check_templates(templates, ordering)
    if missing is not None:
        return [MaterializedRoute(
            ordering=list(ordering), status="template_extraction_failed", target=target,
            failed_step_id=missing[0], failed_position=missing[1])]

    states: List[_State] = [_State(frontier=(OpenMol(target, None),), steps_rev=[],
                                   sm_budget=route_sm_budget(templates))]
    n = len(ordering)
    for position in range(n, 0, -1):        # undo last-performed step first
        sid = ordering[position - 1]
        best_frontier = frontier_smiles(states, target)
        states, any_outcome = advance_states(
            states, templates[sid], position,
            remaining_after=position - 1, beam=beam, max_outcomes=max_outcomes)
        if not states:
            reason = "no_usable_outcome" if any_outcome else "template_no_match"
            return [MaterializedRoute(
                ordering=list(ordering), status=reason, target=target,
                failed_position=position, failed_step_id=sid,
                failure_intermediate=best_frontier)]
    return finalize_states(states, ordering, target)


# ---------------------------------------------------------------------------
# Identity replay — the calibration gate
# ---------------------------------------------------------------------------
def _no_stereo(smi: Optional[str]) -> Optional[str]:
    if smi is None:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


@dataclass
class ReplayResult:
    ok: bool
    stereo_loss: bool
    detail: str
    route: Optional[MaterializedRoute] = None


def replay_identity(full_graph: dict, templates: Dict[int, StepTemplate],
                    incidental_order: List[int], *, beam: int = 3) -> ReplayResult:
    """Materialize the chemist's own order and demand the original route back — whole
    tree included.

    A variant matches when, at every position, the reconstructed product equals the
    original step's product, the deposited synthesized-precursor multiset equals the
    original children's products, and the step's parent in the emergent tree is its
    original parent (so the reconstructed edge set is the original tree).  Canonical
    map-free compare; non-isomeric fallback → ``stereo_loss``.
    """
    variants = materialize_ordering(full_graph, templates, incidental_order, beam=beam)
    if not variants or not variants[0].ok:
        v = variants[0] if variants else None
        return ReplayResult(False, False,
                            f"replay failed: {v.status if v else 'no result'}"
                            f" at step {v.failed_step_id if v else '?'}")

    orig_parent = original_parents(full_graph)

    def matches(route: MaterializedRoute, strip) -> bool:
        for rec in route.steps:
            tpl = templates[rec.orig_step_id]
            if strip(rec.new_product) != strip(tpl.orig_product):
                return False
            want = sorted(strip(s) or "" for s in tpl.orig_synth_precursors)
            got = sorted(strip(s) or "" for s in rec.synth_precursors)
            if got != want:
                return False
            if rec.parent_step_id != orig_parent.get(rec.orig_step_id):
                return False
        return True

    for route in variants:
        if matches(route, lambda s: s):
            return ReplayResult(True, False, "exact", route=route)
    for route in variants:
        if matches(route, _no_stereo):
            return ReplayResult(True, True, "matched without stereochemistry", route=route)
    return ReplayResult(False, False, "no variant reproduced the original route",
                        route=variants[0])
