"""Route-to-route dissimilarity: distance-to-literature, diverse top-k, GUI sort."""

from route_rearrangement import similarity as S
from route_rearrangement.gui.model import DISTINCT_KEY, RouteEntry, TreeGroup


def _record(ordering, products, *, original=False):
    """A minimal materialized-route record whose transient intermediates are *products*
    (the final element is the shared target)."""
    steps = [
        {"position": i + 1, "new_product": p,
         "chain_precursor": products[i - 1] if i > 0 else None,
         "side_reactants": [], "orig_step_id": i + 1, "orig_rxn_index": -1,
         "retro_smarts": "a>>b", "new_rxn": "C>>C", "outcome_rank": 0,
         "n_outcomes": 1, "exact_side_match": True, "sim_score": 1.0}
        for i, p in enumerate(products)
    ]
    return {"tree_id": "toy", "ordering": ordering, "is_original_order": original,
            "target": products[-1], "steps": steps}


# same target "CCOC(C)=O"; transient intermediate sets are controlled to fix distances
_TARGET = "CCOC(C)=O"
_ORIG = _record([1, 2, 3], ["CCO", "CC(=O)O", _TARGET], original=True)
_SAME = _record([2, 1, 3], ["CCO", "CC(=O)O", _TARGET])          # identical intermediates → 0
_HALF = _record([1, 3, 2], ["CCO", "CCCCO", _TARGET])            # partial overlap
_DIFF = _record([3, 2, 1], ["CCCO", "CCC(=O)O", _TARGET])        # disjoint → maximal


def test_jaccard_fallback_ranks_and_diversifies(monkeypatch):
    monkeypatch.setattr(S, "available", lambda: False)          # force the map-free path
    recs = [dict(r) for r in (_ORIG, _SAME, _HALF, _DIFF)]
    diverse, method = S.annotate_distinctness(recs, k=5)
    assert method == "jaccard-intermediates"

    sim = {tuple(r["ordering"]): r["similarity"] for r in recs}
    # the original is the reference: distance 0, not itself a candidate
    assert sim[(1, 2, 3)]["distance_to_original"] == 0.0
    assert sim[(1, 2, 3)]["diverse_rank"] is None
    # disjoint intermediates = maximal distance = the single most different
    assert sim[(3, 2, 1)]["distance_to_original"] == 1.0
    assert sim[(3, 2, 1)]["rank_most_different"] == 1
    # 3 rearrangements → 3 diverse picks, disjoint one picked first
    assert len(diverse) == 3
    assert recs[diverse[0]]["ordering"] == [3, 2, 1]


def test_distinct_sort_puts_diverse_picks_first(monkeypatch):
    monkeypatch.setattr(S, "available", lambda: False)
    recs = [dict(r) for r in (_ORIG, _SAME, _HALF, _DIFF)]
    S.annotate_distinctness(recs, k=2)
    orig = RouteEntry(recs[0], is_original=True)
    rearr = [RouteEntry(r, is_original=False) for r in recs[1:]]
    g = TreeGroup("toy", orig, rearr)

    assert DISTINCT_KEY in g.sort_keys()
    ordered = g.sorted_rearrangements(DISTINCT_KEY)
    ranks = [e.diverse_rank() for e in ordered]
    # the two diverse picks (ranks 1,2) come before the unranked remainder
    assert ranks[:2] == [1, 2]
    assert ranks[2] is None


def test_ted_path_runs_when_available():
    if not S.available():
        return                                                   # rxnutils/apted not installed
    recs = [dict(r) for r in (_ORIG, _SAME, _HALF, _DIFF)]
    diverse, method = S.annotate_distinctness(recs, k=3)
    assert method == "ted-molecules"
    assert all("similarity" in r for r in recs)
    assert len(diverse) == 3
    assert all(r["similarity"]["distance_to_original"] >= 0 for r in recs)
