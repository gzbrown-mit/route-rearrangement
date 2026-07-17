from types import SimpleNamespace

from route_rearrangement.fragments import (
    adjacency_stats, extract_fragments, goodness_scores)


def _edge(earlier, later, relation="material"):
    return SimpleNamespace(earlier=earlier, later=later, relation=relation)


def _rec(ordering, exposure):
    return {
        "tree_id": "t", "ordering": ordering, "is_original_order": False,
        "steps": [{"orig_step_id": s, "retro_smarts": f"tmpl{s}",
                   "new_rxn": f"rxn{s}"} for s in ordering],
        "metrics": {"exposure": {"score": exposure}},
    }


# steps {1,2,3,4}; good (high-exposure) orderings keep 3 immediately before 2 while 1 and 4
# float around it; bad orderings never place 3 immediately before 2.
RECORDS = [
    _rec([1, 3, 2, 4], exposure=-1.0),   # good, has (3,2)
    _rec([4, 3, 2, 1], exposure=-1.0),   # good, has (3,2)
    _rec([3, 2, 1, 4], exposure=-1.0),   # good, has (3,2)
    _rec([3, 2, 4, 1], exposure=-1.0),   # good, has (3,2)
    _rec([1, 2, 3, 4], exposure=-5.0),   # bad, no (3,2)
    _rec([2, 1, 4, 3], exposure=-5.0),   # bad, no (3,2)
    _rec([4, 2, 1, 3], exposure=-5.0),   # bad, no (3,2)
]
DEP = SimpleNamespace(edges=[_edge(3, 2)], nodes={1: {}, 2: {}, 3: {}, 4: {}})


def test_goodness_ranks_high_exposure_first():
    g = goodness_scores(RECORDS)
    # the four exposure=-1.0 records should outrank the three -5.0 records
    assert min(g[:4]) > max(g[4:])


def test_adjacency_stats_surface_sticky_pair():
    stats = adjacency_stats(RECORDS, DEP, good_frac=0.6)
    assert (3, 2) in stats
    s = stats[(3, 2)]
    assert s.good_freq > s.all_freq        # more common among good orderings
    assert s.lift > 1.0
    assert s.material                      # 3->2 is a material edge here


def test_extract_hard_fragment():
    frags = extract_fragments(RECORDS, DEP, good_frac=0.6, min_good_freq=0.6, min_lift=1.0)
    assert frags, "expected a cohesive fragment"
    top = frags[0]
    assert top.steps == [3, 2]             # only 3->2 is sticky; 1 and 4 float
    assert top.kind == "hard"              # 3->2 is a material adjacency
    assert top.templates == ["tmpl3", "tmpl2"]
