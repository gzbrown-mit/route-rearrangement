import pytest

from route_rearrangement.metrics import competing


pytestmark = pytest.mark.skipif(not competing.available(),
                                reason="rxnutils / competing SMARTS library unavailable")


def test_bis_alkylation_not_flagged():
    """A ring-closing double O-alkylation consumes both C-Cl bonds — nothing survives, so it
    must not be flagged as a competing site (the false positive that motivated survivor-based
    counting)."""
    step = competing._step_competing(
        ["C=C(CCl)CCl", "CC1(C)CCc2cc(O)c(O)cc21"], "C=C1COc2cc3c(cc2OC1)C(C)(C)CC3")
    assert step["n_competing"] == 0


def test_mesylate_surviving_condensation_flagged():
    """A mesylate that rides through an amide coupling is the -OMs worry — high severity."""
    step = competing._step_competing(
        ["NCCCCOS(C)(=O)=O", "CC(=O)O"], "CC(=O)NCCCCOS(C)(=O)=O")
    groups = {c["group"]: c for c in step["competing"]}
    assert "sulfonate_ester" in groups
    assert groups["sulfonate_ester"]["severity"] == "high"


def test_second_amine_selectivity_flagged():
    """Two free amines, only one acylates — the surviving amine is a competing site."""
    step = competing._step_competing(["NCCCCN", "CC(=O)O"], "CC(=O)NCCCCN")
    assert any(c["group"].startswith("amine") for c in step["competing"])


def test_route_score_higher_is_fewer():
    record = {
        "steps": [
            {"position": 1, "new_product": "CC(=O)NCCCCOS(C)(=O)=O",
             "chain_precursor": None, "side_reactants": ["NCCCCOS(C)(=O)=O", "CC(=O)O"]},
        ],
    }
    res = competing.competing_sites(record)
    assert res["score"] == -res["n_competing"]
    assert res["leaving_group_exposures"]           # mesylate exposure recorded
