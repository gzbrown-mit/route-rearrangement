"""The decomposition rule, on cases constructed so the right answer is known by hand.

These are the tests that would catch the classification being backwards — a failure mode that
produces a perfectly plausible-looking headline number with the meaning inverted.
"""

import pytest

from route_rearrangement.route_synteny import decompose
from route_rearrangement.route_synteny.clusters import Cluster, Occurrence
from route_rearrangement.route_synteny.corpus import Genome


def _genome(rid, families, constraints=()):
    ids = list(range(len(families), 0, -1))
    return Genome(route_id=rid, n_steps=len(families), step_ids=ids, families=list(families),
                  constraints={"material": list(constraints)})


def _cluster(fams, routes):
    return Cluster(families=frozenset(fams), reference_route=routes[0],
                   occurrences=[Occurrence(r, 0, len(fams), 0) for r in routes])


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_free,q_con,expected", [
    (0.5, 0.5, decompose.NOT_A_CLUSTER),     # not tight even against free permutation
    (0.5, 0.001, decompose.NOT_A_CLUSTER),   # free null rules; constrained cannot rescue it
    (0.001, 0.5, decompose.NECESSITY),       # the partial order explains the tightness
    (0.001, 0.001, decompose.CONVENTION),    # tight beyond what chemistry forces
])
def test_classification_rule(q_free, q_con, expected):
    assert decompose.classify(q_free, q_con) == expected


# ---------------------------------------------------------------------------
# End to end on constructed corpora
# ---------------------------------------------------------------------------
def _forced_corpus(n=40):
    """A block that is tight *because* the partial order forces the steps together.

    Steps 4 and 3 carry the block and are chained by constraints to sit at the front, so under
    Null-C they stay adjacent while under Null-P they scatter.
    """
    return [_genome(f"r{i}", ["A", "B", "X", "Y"], constraints=[(4, 3), (3, 2), (2, 1)])
            for i in range(n)]


def _free_corpus(n=40):
    """The same block, always written together, but with nothing forcing it."""
    return [_genome(f"r{i}", ["A", "B", "X", "Y"]) for i in range(n)]


def test_a_forced_block_is_classified_as_necessity():
    genomes = _forced_corpus()
    v = decompose.decompose([_cluster("AB", [g.route_id for g in genomes])], genomes,
                            delta=0, max_extra=0, draws=60, max_routes=50)[0]
    assert v.q_free <= 0.05, "a totally forced block must beat the free null"
    assert v.verdict["material"] == decompose.NECESSITY


def test_an_unforced_but_always_together_block_is_convention():
    genomes = _free_corpus()
    v = decompose.decompose([_cluster("AB", [g.route_id for g in genomes])], genomes,
                            delta=0, max_extra=0, draws=60, max_routes=50)[0]
    assert v.verdict["material"] == decompose.CONVENTION


def test_the_two_nulls_agree_when_there_are_no_constraints():
    """With an empty constraint set Null-C *is* Null-P, so nothing can be called necessity."""
    genomes = _free_corpus(30)
    v = decompose.decompose([_cluster("AB", [g.route_id for g in genomes])], genomes,
                            delta=0, max_extra=0, draws=60, max_routes=50)[0]
    assert v.p_free == pytest.approx(v.p_constrained["material"], abs=0.05)
    assert v.verdict["material"] != decompose.NECESSITY


def test_headline_reports_convention_as_a_fraction_of_real_clusters():
    verdicts = [
        decompose.Verdict(["A", "B"], 2, 9, "r", 9, 0.001, 0.001, {"material": 0.5},
                          {"material": 0.5}, {"material": decompose.NECESSITY}, 1.0,
                          {"material": 1.0}, 9, 0),
        decompose.Verdict(["C", "D"], 2, 9, "r", 9, 0.001, 0.001, {"material": 0.001},
                          {"material": 0.001}, {"material": decompose.CONVENTION}, 1.0,
                          {"material": 1.0}, 9, 0),
        decompose.Verdict(["E", "F"], 2, 9, "r", 9, 0.9, 0.9, {"material": 0.9},
                          {"material": 0.9}, {"material": decompose.NOT_A_CLUSTER}, 1.0,
                          {"material": 1.0}, 9, 0),
    ]
    h = decompose.headline(verdicts, tiers=["material"])
    s = h["tiers"]["material"]
    assert s["n_significant_clusters"] == 2 and s["not_a_cluster"] == 1
    assert s["convention_fraction_upper_bound"] == pytest.approx(0.5)


def test_headline_text_states_the_bound_not_an_estimate():
    """The caveat is load-bearing: without it the number reads as a measurement of convention."""
    h = decompose.headline([], tiers=["material"])
    text = decompose.format_headline(h, tiers=["material"])
    assert "UPPER BOUND" in text
    assert "under-powered" in text
