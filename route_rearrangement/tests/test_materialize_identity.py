from route_rearrangement.materialize import replay_identity
from route_rearrangement.templates import extract_step_templates

from .conftest import corpus_required


@corpus_required
def test_identity_replay_reproduces_original_route(load_tree):
    """The calibration gate: replaying the chemist's own order through extracted retro
    templates must reproduce the original intermediates (tree 0_6 is a known-good case)."""
    from synthesis_extraction.dependency.route_graph import build_route_graph
    from synthesis_extraction.dependency.analyze import dependency_graph_from_full_graph

    tg = load_tree("0_6")
    assert tg is not None
    full = build_route_graph(tg, "0_6")
    assert full is not None and not full["qc"]["disconnected"]
    templates = extract_step_templates(full)
    dep = dependency_graph_from_full_graph(full, "0_6")
    res = replay_identity(full, templates, dep.incidental_order())
    assert res.ok, res.detail
    assert res.route is not None and res.route.ok
    # replayed products must equal the original step products exactly
    for rec in res.route.steps:
        assert rec.new_product == templates[rec.orig_step_id].orig_product
