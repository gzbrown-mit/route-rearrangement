"""Genome construction — above all the step-numbering convention.

Two numbering schemes meet here (``full_graph`` node ids and ``ScheduleLattice``'s bit order).
A reversal between them would silently produce backwards genomes: every ordering statistic
would still run, still produce p-values, and mean the opposite of what it says. Nothing
downstream could detect it, so it is pinned here.
"""

import pytest

from route_rearrangement.route_synteny import corpus, precedence
from route_rearrangement.route_synteny.corpus import UNKNOWN, Genome
from route_rearrangement.route_synteny.tests.conftest import genomes_required

AMIDE = "[CH3:1][C:2](=[O:3])[OH:4].[NH2:5][CH3:6]>>[CH3:1][C:2](=[O:3])[NH:5][CH3:6]"
NITRO_RED = "[c:1][N+:2](=[O:3])[O-:4]>>[c:1][NH2:2]"


# ---------------------------------------------------------------------------
# Numbering
# ---------------------------------------------------------------------------
def test_synthesis_order_is_deepest_first():
    """full_graph numbers the root product 1, so the chemist's first step has the largest id."""
    assert corpus.synthesis_order([1, 4, 2, 3]) == [4, 3, 2, 1]


def test_synthesis_order_matches_the_schedule_lattice_bit_order():
    """If these disagreed, a sampled ordering could not be compared to the literature one."""
    from synthesis_extraction.dependency.schedule import ScheduleLattice
    ids = [2, 4, 1, 3]
    assert ScheduleLattice(ids, []).ids == corpus.synthesis_order(ids)


def test_linear_filter_rejects_a_convergent_tree():
    import networkx as nx
    linear = nx.DiGraph([(3, 2), (2, 1)])
    convergent = nx.DiGraph([(3, 1), (2, 1)])       # two steps feed the same parent
    assert corpus.is_linear_tree(linear)
    assert not corpus.is_linear_tree(convergent)


# ---------------------------------------------------------------------------
# One step, one transformation
# ---------------------------------------------------------------------------
def test_every_step_that_changes_a_bond_is_its_own_transformation():
    """The reason centres were dropped: nothing merges steps, so nothing needs anchoring."""
    nodes = [{"id": "2", "SMILES": AMIDE}, {"id": "1", "SMILES": NITRO_RED}]
    fam = corpus.families_for_steps(nodes, "centre")
    assert set(fam) == {1, 2}
    assert all(fam.values())
    assert fam[1] != fam[2]


def test_a_repeated_transformation_is_still_dropped_as_ambiguous():
    """Per-step identity fixes segmentation, not ambiguity: two amide couplings are still two."""
    g = Genome(route_id="r", n_steps=3, step_ids=[3, 2, 1], families=["A", "B", "A"])
    assert precedence.unique_family_positions(g) == {"B": 2}


def test_unidentified_steps_are_excluded_but_keep_their_position():
    g = Genome(route_id="r", n_steps=3, step_ids=[3, 2, 1],
               families=["A", UNKNOWN, "B"])
    assert precedence.unique_family_positions(g) == {"A": 3, "B": 1}
    assert len(g.families) == len(g.step_ids)      # position in the route stays faithful


def test_coverage_reports_identified_steps():
    g = Genome(route_id="r", n_steps=4, step_ids=[4, 3, 2, 1],
               families=["A", UNKNOWN, "B", "C"])
    assert g.coverage == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def test_genome_round_trips_through_json():
    g = Genome(route_id="r", n_steps=2, step_ids=[2, 1], families=["A", "B"],
               constraints={"material": [(2, 1)]}, n_orderings={"material": 1})
    assert Genome.from_dict(g.to_dict()) == g


def test_schema_1_caches_still_load_and_ignore_anchors():
    """Older genome files carry a vestigial ``anchors`` mask; it must not break loading."""
    old = {"route_id": "r", "n_steps": 2, "step_ids": [2, 1], "families": ["A", "B"],
           "anchors": [True, False], "constraints": {}, "n_orderings": {}, "schema": 1}
    g = Genome.from_dict(old)
    assert g.families == ["A", "B"] and not hasattr(g, "anchors")


# ---------------------------------------------------------------------------
# Against the real cache
# ---------------------------------------------------------------------------
@genomes_required
def test_real_genomes_are_ordered_deepest_first(corpus_genomes):
    for g in corpus_genomes[:500]:
        assert g.step_ids == sorted(g.step_ids, reverse=True)
        assert len(g.families) == len(g.step_ids)


@genomes_required
def test_real_constraints_point_forward_in_synthesis_order(corpus_genomes):
    """Every dependency edge must run earlier -> later, i.e. from a larger id to a smaller.

    This is the reversal check that matters: if the genome were built backwards, constraints
    would systematically point the wrong way and the constrained null would be nonsense.
    """
    checked = 0
    for g in corpus_genomes[:2000]:
        pos = {s: i for i, s in enumerate(g.step_ids)}
        for tier, edges in g.constraints.items():
            for earlier, later in edges:
                assert pos[earlier] < pos[later], (
                    f"{g.route_id} {tier}: edge {earlier}->{later} points backwards")
                checked += 1
    assert checked, "no constraints found to check"
