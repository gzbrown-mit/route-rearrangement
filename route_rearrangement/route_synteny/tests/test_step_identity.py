"""Per-step transformation identity, read off the bonds a step actually changes.

The identity exists to make every step lookup-able, so the tests that matter are the ones
checking it *discriminates* (different chemistry must not collapse to one key) and *recurs*
(the same chemistry on different scaffolds must give the same key).  Both failures are silent:
over-collapsing pools unrelated reactions into one statistic, over-splitting means nothing ever
repeats and no statistic can form.
"""

import pytest

from route_rearrangement.route_synteny import step_identity as SI

AMIDE = "[CH3:1][C:2](=[O:3])[OH:4].[NH2:5][CH3:6]>>[CH3:1][C:2](=[O:3])[NH:5][CH3:6]"
N_ALKYL = "[CH3:1][Cl:2].[NH2:5][CH3:6]>>[CH3:1][NH:5][CH3:6]"
NITRO_RED = "[c:1][N+:2](=[O:3])[O-:4]>>[c:1][NH2:2]"
ESTER_HYD = "[CH3:1][C:2](=[O:3])[O:4][CH3:7]>>[CH3:1][C:2](=[O:3])[OH:4]"
BOC_OFF = ("[NH:1]([CH3:2])[C:3](=[O:4])[O:5][C:6]([CH3:7])([CH3:8])[CH3:9]"
           ">>[NH2:1][CH3:2]")


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------
def test_bond_forming_reactions_are_told_apart_by_what_they_break():
    """Amide coupling and N-alkylation both form C-N; only the broken bond separates them.

    This is the case that motivated counting bonds to departing atoms at all.
    """
    a = SI.step_keys(AMIDE)["bond_changes"]
    n = SI.step_keys(N_ALKYL)["bond_changes"]
    assert "CN0>1" in a and "CN0>1" in n
    assert a != n
    assert "CO1>0" in a and "CCl1>0" in n


def test_a_reaction_whose_changed_bonds_all_leave_is_still_identified():
    """Every bond a nitro reduction changes runs to an oxygen that departs.

    Requiring both endpoints to survive returned ``None`` here — the reaction simply vanished.
    """
    k = SI.step_keys(NITRO_RED)
    assert k["bond_changes"] == "NO1>0|NO2>0"


def test_deprotection_and_hydrolysis_are_distinct():
    assert SI.step_keys(BOC_OFF)["bond_changes"] != SI.step_keys(ESTER_HYD)["bond_changes"]


@pytest.mark.parametrize("rung", SI.RUNGS)
def test_every_rung_produces_a_key_for_a_real_reaction(rung):
    assert SI.step_keys(AMIDE)[rung]


# ---------------------------------------------------------------------------
# Recurrence — the same chemistry on a different scaffold
# ---------------------------------------------------------------------------
def test_same_transformation_on_a_different_scaffold_gives_the_same_coarse_key():
    other = ("[CH3:1][CH2:10][C:2](=[O:3])[OH:4].[NH2:5][c:6]1[cH:11][cH:12]ccc1"
             ">>[CH3:1][CH2:10][C:2](=[O:3])[NH:5][c:6]1[cH:11][cH:12]ccc1")
    assert SI.step_keys(other)["bond_changes"] == SI.step_keys(AMIDE)["bond_changes"]


def test_rungs_are_ordered_from_specific_to_general():
    """Coarser rungs must not distinguish more than finer ones."""
    keys = SI.step_keys(AMIDE)
    assert len(keys["centre_env"]) >= len(keys["centre"]) >= len(keys["bond_changes"])
    assert SI.RUNGS == ("centre_env", "centre", "bond_changes")


def test_atom_map_renumbering_does_not_change_the_key():
    """Keys are built from elements and bond orders, never from map numbers themselves."""
    remapped = AMIDE.replace(":1]", ":91]").replace(":2]", ":92]").replace(":5]", ":95]")
    assert SI.step_keys(remapped)["centre"] == SI.step_keys(AMIDE)["centre"]


# ---------------------------------------------------------------------------
# Refusing to guess
# ---------------------------------------------------------------------------
def test_a_step_with_no_bond_change_yields_no_key():
    """A salt swap or pure stereochemical step must be excluded, not pooled into one key.

    Returning ``""`` would gather every such step under a single pseudo-transformation and
    manufacture a very popular, entirely meaningless family.
    """
    assert SI.step_keys("[CH3:1][OH:2]>>[CH3:1][OH:2]") == {r: None for r in SI.RUNGS}


@pytest.mark.parametrize("bad", ["", "not a reaction", "[CH3:1]", ">>"])
def test_unparseable_input_is_declined(bad):
    assert SI.step_keys(bad) == {r: None for r in SI.RUNGS}


def test_unmapped_reactions_yield_nothing_rather_than_a_wrong_answer():
    assert SI.step_keys("CC(=O)O.NC>>CC(=O)NC") == {r: None for r in SI.RUNGS}


def test_families_for_steps_covers_every_node():
    nodes = [{"id": "2", "SMILES": AMIDE}, {"id": "1", "SMILES": NITRO_RED}]
    fam = SI.families_for_steps(nodes, "centre")
    assert set(fam) == {1, 2} and all(fam.values())


def test_step_key_is_cached_and_consistent():
    assert SI.step_key(AMIDE, "centre") == SI.step_key(AMIDE, "centre")
    assert SI.step_key(AMIDE, "centre") == SI.step_keys(AMIDE)["centre"]
