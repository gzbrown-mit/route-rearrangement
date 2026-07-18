"""Convergent-tree support: template metadata, frontier replay, end-to-end."""

from route_rearrangement.materialize import replay_identity
from route_rearrangement.templates import (
    extract_step_templates,
    is_linear,
    original_parents,
    route_sm_budget,
)

from .conftest import corpus_required

# a tiny convergent tree: node 2 makes the acid, node 3 makes the amine, node 1
# couples them.  Unmapped SMILES — retro extraction fails harmlessly; the roles
# (synth precursors vs side reactants) must still be identified correctly.
_FIXTURE = {
    "nodes": [
        {"id": 1, "rxn_index": -1,
         "SMILES": "CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1"},
        {"id": 2, "rxn_index": -1, "SMILES": "CCOC(C)=O>>CC(=O)O"},
        {"id": 3, "rxn_index": -1, "SMILES": "N#Cc1ccccc1>>NCc1ccccc1"},
    ],
    "edges": [[2, 1], [3, 1]],
    "qc": {"disconnected": [], "n_steps": 3},
}


def test_coupling_children_are_synth_precursors_not_side_reactants():
    """The metadata fix: both of a coupling's synthesized children are recognized as
    synth precursors — the old extractor dumped them into the side reactants."""
    assert not is_linear(_FIXTURE)
    templates = extract_step_templates(_FIXTURE)
    tpl = templates[1]
    assert sorted(tpl.orig_synth_precursors) == ["CC(=O)O", "NCc1ccccc1"]
    assert tpl.orig_side_reactants == []
    assert tpl.orig_chain_precursor is None       # not a single-chain step
    # leaf steps: no synth precursors, all reactants are starting materials
    assert templates[2].orig_synth_precursors == []
    assert templates[2].orig_side_reactants == ["CCOC(C)=O"]


def test_budget_and_parents_helpers():
    templates = extract_step_templates(_FIXTURE)
    assert dict(route_sm_budget(templates)) == {"CCOC(C)=O": 1, "N#Cc1ccccc1": 1}
    assert original_parents(_FIXTURE) == {1: None, 2: 1, 3: 1}


def _convergent_fulls(corpus_path, max_found=3, scan=600):
    from synthesis_extraction.load_trees import iter_trees
    from synthesis_extraction.dependency.route_graph import build_route_graph

    found = []
    for i, (tid, tg) in enumerate(iter_trees(corpus_path)):
        if i >= scan or len(found) >= max_found:
            break
        try:
            full = build_route_graph(tg, tid)
        except Exception:
            continue
        if full is None or full["qc"]["disconnected"]:
            continue
        if not (3 <= full["qc"]["n_steps"] <= 8) or is_linear(full):
            continue
        found.append((tid, tg, full))
    return found


@corpus_required
def test_convergent_replay_reproduces_original_tree(corpus_path):
    """Whole-tree calibration gate on real convergent corpus routes: replaying the
    chemist's own order must reproduce the original products AND the original
    child→parent edges (frontier walk, not the single-chain walk)."""
    import pytest
    from synthesis_extraction.dependency.analyze import dependency_graph_from_full_graph

    found = _convergent_fulls(corpus_path)
    if not found:
        pytest.skip("no convergent routes in the scanned corpus slice")

    passed = 0
    for tid, _tg, full in found:
        templates = extract_step_templates(full)
        dep = dependency_graph_from_full_graph(full, tid)
        res = replay_identity(full, templates, dep.incidental_order())
        if not res.ok:
            continue
        passed += 1
        orig_parent = original_parents(full)
        coupling_seen = False
        for rec in res.route.steps:
            assert rec.parent_step_id == orig_parent.get(rec.orig_step_id)
            if len(rec.synth_precursors) >= 2:
                coupling_seen = True
        assert coupling_seen, f"{tid}: convergent route replayed without a coupling step"
    assert passed >= 1, (
        f"no convergent route of {len(found)} passed identity replay: "
        f"{[t for t, _, _ in found]}")


@corpus_required
def test_convergent_end_to_end(corpus_path):
    """process_route on a convergent route: accepted records rebuild into a genuine
    tree that passes the (tree-safe) connectivity gate."""
    import pytest
    from route_rearrangement.filters import rebuilt_full_graph
    from route_rearrangement.run import process_route
    from route_rearrangement.schema import route_record, route_from_record

    found = _convergent_fulls(corpus_path)
    if not found:
        pytest.skip("no convergent routes in the scanned corpus slice")

    any_ok = False
    for tid, tg, full in found:
        summary, records, _failures = process_route(
            tid, tg, full_graph=full, cap=50, max_accepted=10, with_fg=False)
        if summary["status"] != "ok":
            continue
        any_ok = True
        assert summary["is_linear"] is False
        assert any(r["is_original_order"] for r in records)
        # the rebuilt graph of the original-order record is the original tree shape:
        # some node has two children (the coupling)
        rec = next(r for r in records if r["is_original_order"])
        graph = rebuilt_full_graph(route_from_record(rec))
        kids = {}
        for c, p in graph["edges"]:
            kids[p] = kids.get(p, 0) + 1
        assert max(kids.values()) >= 2
        break
    if not any_ok:
        pytest.skip("no scanned convergent route passed the identity gate")


def _roundtrip(rec_dict):
    from route_rearrangement.schema import route_from_record
    return route_from_record(rec_dict)


def test_route_from_record_backcompat_linear_parents():
    """Old (pre-frontier) records carry no parent pointers: the linear-chain fallback
    must reconstruct position k -> k+1 parentage."""
    record = {
        "ordering": [3, 2, 1], "status": "ok", "target": "CCO",
        "starting_materials": [],
        "steps": [
            {"position": p, "orig_step_id": 4 - p, "orig_rxn_index": -1,
             "retro_smarts": "", "new_rxn": "C>>C", "chain_precursor": "C",
             "side_reactants": [], "new_product": "C", "outcome_rank": 0,
             "n_outcomes": 1, "exact_side_match": True, "sim_score": 1.0}
            for p in (1, 2, 3)
        ],
    }
    route = _roundtrip(record)
    parents = {rec.position: rec.parent_step_id for rec in route.steps}
    assert parents == {1: 2, 2: 1, 3: None}
    assert all(rec.synth_precursors == ["C"] for rec in route.steps)
