import pytest

from route_rearrangement.gui import reaction_class as rc
from route_rearrangement.gui.render import _box_labels


pytestmark = pytest.mark.skipif(not rc.available(),
                                reason="rxnutils unavailable for reaction classification")


def test_amide_coupling():
    assert rc.classify_reaction(["NCCCC", "CC(=O)O"], "CC(=O)NCCCC") == "amide coupling"


def test_sulfonylation_install():
    # install a mesylate onto an alcohol
    label = rc.classify_reaction(["OCCCC", "CS(=O)(=O)Cl"], "CCCCOS(C)(=O)=O")
    assert "sulfonylation" in label


def test_o_alkylation():
    label = rc.classify_reaction(["Oc1ccccc1", "CCBr"], "CCOc1ccccc1")
    assert "alkylation" in label.lower()


def test_fallback_is_never_empty_when_available():
    # a transformation with no library group still returns a non-empty honest label
    label = rc.classify_reaction(["C1CCCCC1"], "C1CCCCC1")
    assert isinstance(label, str) and label


def test_box_labels_use_original_step_numbers():
    """Box labels carry the ORIGINAL literature step number, stable across orderings."""
    record = {
        "tree_id": "t", "ordering": [3, 2, 1],
        "steps": [
            {"position": 1, "orig_step_id": 3, "new_product": "CC(=O)NCCCC",
             "chain_precursor": None, "side_reactants": ["NCCCC", "CC(=O)O"]},
            {"position": 2, "orig_step_id": 2, "new_product": "CCOC(C)=O",
             "chain_precursor": "CC(=O)NCCCC", "side_reactants": ["CCO"]},
            {"position": 3, "orig_step_id": 1, "new_product": "COC(C)=O",
             "chain_precursor": "CCOC(C)=O", "side_reactants": ["CO"]},
        ],
    }
    labels = _box_labels(record)
    # node id = n_steps - position + 1; orig step no = rank of orig_step_id (deepest = 1)
    assert labels[3].startswith("lit. step 1")   # position 1, orig_step_id 3 -> deepest
    assert labels[1].startswith("lit. step 3")   # position 3, orig_step_id 1 -> shallowest
    assert "amide coupling" in labels[3]
