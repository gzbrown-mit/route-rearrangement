"""Backward materialization of one ordering of a route's steps.

Given a unified-map ``full_graph``, per-step retro templates and a valid ordering
(earliest-performed first, a linear extension of the dependency partial order), walk
backward from the target: undo the last-performed step first by applying its retro
template to the substrate *as it exists in the rearranged route*, split the outcome into
the new chain intermediate + side reactants (:mod:`.chain`), and continue with the chain.
An ordering whose required context is absent at some position (the template yields no
outcome) is pruned there — that is the built-in chemical veto.

Multiple template outcomes make the walk a small beam search; each surviving completed
leaf is one materialized *variant* of the ordering.  :func:`replay_identity` runs the
same walk on the chemist's original order and demands the original intermediates back —
the calibration gate that must pass before any rearrangement of that route is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from .chain import ChainSplit, split_outcome
from .templates import StepTemplate, apply_retro, canonicalize_smiles
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
    chain_precursor: Optional[str]     # None at position 1 (all fragments are SMs)
    side_reactants: List[str]
    new_product: str
    outcome_rank: int
    n_outcomes: int
    exact_side_match: bool
    sim_score: float


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


@dataclass
class _State:
    current: str                       # chain intermediate still to be disconnected
    steps_rev: List[StepRecord]        # records collected last-position-first
    n_inexact: int = 0
    sum_sim: float = 0.0

    @property
    def score(self) -> Score:
        return (self.n_inexact, -self.sum_sim)


def route_target(full_graph: dict) -> Optional[str]:
    """Canonical map-free final product of the route (main product of node 1)."""
    for n in full_graph.get("nodes", []):
        if int(n["id"]) == 1:
            _, pb = split_reaction(n.get("SMILES", ""))
            pm = main_product_mol(pb) if pb else None
            return canonicalize_smiles(Chem.MolToSmiles(pm)) if pm is not None else None
    return None


def advance_states(states: List[_State], tpl: StepTemplate, position: int, *,
                   terminal: bool, beam: int, max_outcomes: int) -> Tuple[List[_State], bool]:
    """Expand every beam state by one backward application of *tpl*.

    Returns ``(new_states, any_outcome)`` — deterministic given the sequence of steps
    already undone, which is what lets the DFS engine share suffixes with the naive one.
    """
    nxt: List[_State] = []
    any_outcome = False
    for st in states:
        outcomes = apply_retro(tpl.retro_smarts, st.current, max_outcomes=max_outcomes)
        if outcomes:
            any_outcome = True
        for rank, outcome in enumerate(outcomes):
            split = split_outcome(outcome, tpl, terminal=terminal)
            if split is None:
                continue
            rec = StepRecord(
                position=position,
                orig_step_id=tpl.step_id,
                orig_rxn_index=tpl.rxn_index,
                retro_smarts=tpl.retro_smarts,
                new_rxn=".".join(sorted(split.side + ([split.chain] if split.chain else [])))
                        + ">>" + st.current,
                chain_precursor=split.chain,
                side_reactants=list(split.side),
                new_product=st.current,
                outcome_rank=rank,
                n_outcomes=len(outcomes),
                exact_side_match=split.exact_side_match,
                sim_score=split.sim_score,
            )
            nxt.append(_State(
                current=split.chain if split.chain else "",
                steps_rev=st.steps_rev + [rec],
                n_inexact=st.n_inexact + (0 if split.exact_side_match else 1),
                sum_sim=st.sum_sim + split.sim_score,
            ))
    # dedupe states on (current, last reaction) then keep the best `beam`
    nxt.sort(key=lambda s: s.score)
    seen, pruned = set(), []
    for st in nxt:
        key = (st.current, st.steps_rev[-1].new_rxn)
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
        for rec in steps:                             # at position 1 every fragment is a SM
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

    states: List[_State] = [_State(current=target, steps_rev=[])]
    n = len(ordering)
    for position in range(n, 0, -1):        # undo last-performed step first
        sid = ordering[position - 1]
        best_current = states[0].current if states else target
        states, any_outcome = advance_states(
            states, templates[sid], position,
            terminal=position == 1, beam=beam, max_outcomes=max_outcomes)
        if not states:
            reason = "no_usable_outcome" if any_outcome else "template_no_match"
            return [MaterializedRoute(
                ordering=list(ordering), status=reason, target=target,
                failed_position=position, failed_step_id=sid,
                failure_intermediate=best_current)]
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
    """Materialize the chemist's own order and demand the original intermediates back.

    A variant matches when, at every position, the reconstructed product equals the
    original step's product and the reconstructed chain equals the original chain
    precursor (canonical map-free compare; non-isomeric fallback → ``stereo_loss``).
    """
    variants = materialize_ordering(full_graph, templates, incidental_order, beam=beam)
    if not variants or not variants[0].ok:
        v = variants[0] if variants else None
        return ReplayResult(False, False,
                            f"replay failed: {v.status if v else 'no result'}"
                            f" at step {v.failed_step_id if v else '?'}")

    def matches(route: MaterializedRoute, strip) -> bool:
        for rec in route.steps:
            tpl = templates[rec.orig_step_id]
            if strip(rec.new_product) != strip(tpl.orig_product):
                return False
            if rec.position > 1 and strip(rec.chain_precursor) != strip(tpl.orig_chain_precursor):
                return False
        return True

    for route in variants:
        if matches(route, lambda s: s):
            return ReplayResult(True, False, "exact", route=route)
    for route in variants:
        if matches(route, _no_stereo):
            return ReplayResult(True, True, "matched without stereochemistry", route=route)
    return ReplayResult(False, False, "no variant reproduced the original intermediates",
                        route=variants[0])
