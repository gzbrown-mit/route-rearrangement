from collections import Counter

from route_rearrangement.filters import dedup_key
from route_rearrangement.materialize import materialize_ordering
from route_rearrangement.run import process_route
from route_rearrangement.search import materialize_all_dfs
from route_rearrangement.templates import extract_step_templates

from .conftest import corpus_required

TREE = "262_38"   # 8 linear steps, 360 valid orderings, passes the identity gate


@corpus_required
def test_process_route_accepts_rearrangements(load_tree):
    tg = load_tree(TREE)
    assert tg is not None
    summary, records, failures = process_route(TREE, tg, engine="dfs", cap=1000,
                                               max_accepted=1000, with_fg=False)
    assert summary["status"] == "ok"
    assert summary["identity_roundtrip"]
    assert summary["accepted"] > 1                       # rearrangements beyond the original
    origs = [r for r in records if r["is_original_order"]]
    assert len(origs) == 1                               # the original order is present once
    sm_orig = Counter(origs[0]["starting_materials"])
    for r in records:                                    # building blocks are order-invariant
        assert Counter(r["starting_materials"]) == sm_orig
    assert len({tuple(r["ordering"]) for r in records}) == len(records)


@corpus_required
def test_naive_and_dfs_engines_agree(load_tree):
    from synthesis_extraction.dependency.route_graph import build_route_graph
    from synthesis_extraction.dependency.analyze import dependency_graph_from_full_graph
    from synthesis_extraction.dependency.schedule import lattice_for

    tg = load_tree(TREE)
    full = build_route_graph(tg, TREE)
    templates = extract_step_templates(full)
    dep = dependency_graph_from_full_graph(full, TREE)

    naive_keys = set()
    for ordering in lattice_for(dep).enumerate_orders(cap=1000):
        for v in materialize_ordering(full, templates, ordering):
            if v.ok:
                naive_keys.add(dedup_key(v))
    dfs_keys = set()
    for _ordering, variants in materialize_all_dfs(full, templates, dep, cap=1000):
        for v in variants:
            if v.ok:
                dfs_keys.add(dedup_key(v))
    assert naive_keys == dfs_keys
    assert naive_keys
