from route_rearrangement.chain import split_outcome
from route_rearrangement.templates import StepTemplate


def _tpl(**kw):
    base = dict(step_id=2, rxn_index=-1, orig_rxn="", retro_smarts="x>>y",
                orig_product="CCOC(C)=O", orig_chain_precursor="CCO",
                orig_side_reactants=["CC(=O)O"], orig_reactants=["CCO", "CC(=O)O"])
    base.update(kw)
    return StepTemplate(**base)


def test_exact_side_match_identifies_chain():
    split = split_outcome("CC(=O)O.OCC", _tpl())
    assert split.exact_side_match
    assert split.chain == "CCO"
    assert split.side == ["CC(=O)O"]


def test_terminal_step_all_fragments_are_sms():
    split = split_outcome("CC(=O)O.OCC", _tpl(), terminal=True)
    assert split.chain is None
    assert sorted(split.side) == ["CC(=O)O", "CCO"]


def test_fallback_similarity_when_side_changed():
    # side reactant does not match the template's original -> similarity fallback;
    # the chain is the fragment closest to the original chain precursor (butanol vs ethanol)
    split = split_outcome("CCC(=O)O.OCCCC", _tpl())
    assert not split.exact_side_match
    assert split.chain == "CCCCO"


def test_unparsable_fragment_rejects_outcome():
    assert split_outcome("CC(=O)O.not_a_smiles", _tpl()) is None
