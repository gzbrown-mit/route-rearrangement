"""Suffix-sharing DFS engine over all valid orderings.

Working backward from the target, orderings that share their final steps share every
frontier intermediate along that suffix.  The naive engine (:func:`materialize_ordering`
per ordering) recomputes each shared suffix once per ordering; this engine walks the
*trie* of reversed orderings instead: a DFS over "which not-yet-undone step was
performed last", where a step qualifies iff it is a **maximal** element of the remaining
partial order (no constraint requires it before a remaining step).  Each trie node's
beam states are computed once (via the same :func:`materialize.advance_states` the naive
engine uses, so the accepted-route set is identical), and a dead suffix prunes every
ordering under it in one stroke — such a pruned subtree is reported as a single
representative failure, not one failure per ordering.

*extra_constraints* narrows the explored orderings beyond the dependency poset — the
topology-preserving (no-migration) mode passes the original tree's child→parent pairs,
so every branch stays fully assembled before its coupling.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Iterator, List, Set, Tuple

from .materialize import (
    MaterializedRoute,
    OpenMol,
    _State,
    advance_states,
    check_templates,
    finalize_states,
    frontier_smiles,
    route_target,
)
from .templates import StepTemplate, route_sm_budget


def materialize_all_dfs(full_graph: dict, templates: Dict[int, StepTemplate], dep, *,
                        cap: int = 500, beam: int = 3, max_outcomes: int = 20,
                        extra_constraints: Iterable[Tuple[int, int]] = (),
                        ) -> Iterator[Tuple[List[int], List[MaterializedRoute]]]:
    """Yield ``(ordering, variants)`` for valid orderings of *dep*'s partial order.

    Equivalent accepted set to running :func:`materialize_ordering` on every ordering
    from ``lattice_for(dep).enumerate_orders()``; failures are per pruned *subtree*
    (one representative ordering) rather than per ordering.  Stops after *cap* yields.
    """
    target = route_target(full_graph)
    node_ids: List[int] = sorted(dep.nodes, reverse=True)
    if target is None:
        yield node_ids, [MaterializedRoute(ordering=node_ids, status="no_target", target="")]
        return
    missing = check_templates(templates, node_ids)
    if missing is not None:
        yield node_ids, [MaterializedRoute(
            ordering=node_ids, status="template_extraction_failed", target=target,
            failed_step_id=missing[0], failed_position=missing[1])]
        return

    out_edges: Dict[int, Set[int]] = {nid: set() for nid in node_ids}
    for e, l in dep.constraints():
        if e in out_edges:
            out_edges[e].add(l)
    for e, l in extra_constraints:
        if int(e) in out_edges:
            out_edges[int(e)].add(int(l))

    step_no = dep.step_no
    yielded = 0

    def representative(undone: List[int], rest: FrozenSet[int]) -> List[int]:
        """One earliest-first ordering consistent with this trie node, for reporting."""
        return sorted(rest, key=lambda s: step_no.get(s, s)) + list(reversed(undone))

    def walk(remaining: FrozenSet[int], undone: List[int],
             states: List[_State]) -> Iterator[Tuple[List[int], List[MaterializedRoute]]]:
        nonlocal yielded
        if not remaining:
            ordering = list(reversed(undone))
            yielded += 1
            yield ordering, finalize_states(states, ordering, target)
            return
        position = len(remaining)
        # ascending ids: node 1 (the chemist's last step) is tried first, so the first
        # leaf reached is the original ordering and exploration radiates outward from it
        for sid in sorted(remaining):
            if yielded >= cap:
                return
            if out_edges[sid] & (remaining - {sid}):
                continue                      # sid must precede a remaining step
            nxt, any_outcome = advance_states(
                states, templates[sid], position,
                remaining_after=position - 1, beam=beam, max_outcomes=max_outcomes)
            if not nxt:
                reason = "no_usable_outcome" if any_outcome else "template_no_match"
                yielded += 1
                yield representative(undone + [sid], remaining - {sid}), [MaterializedRoute(
                    ordering=representative(undone + [sid], remaining - {sid}),
                    status=reason, target=target, failed_position=position,
                    failed_step_id=sid,
                    failure_intermediate=frontier_smiles(states, target))]
                continue
            yield from walk(remaining - {sid}, undone + [sid], nxt)

    yield from walk(frozenset(node_ids), [],
                    [_State(frontier=(OpenMol(target, None),), steps_rev=[],
                            sm_budget=route_sm_budget(templates))])
