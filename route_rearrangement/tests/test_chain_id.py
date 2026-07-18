from collections import Counter

from route_rearrangement.chain import route_outcomes
from route_rearrangement.templates import StepTemplate


def _tpl(**kw):
    base = dict(step_id=2, rxn_index=-1, orig_rxn="", retro_smarts="x>>y",
                orig_product="CCOC(C)=O", orig_chain_precursor="CCO",
                orig_side_reactants=["CC(=O)O"], orig_reactants=["CCO", "CC(=O)O"],
                orig_synth_precursors=["CCO"])
    base.update(kw)
    return StepTemplate(**base)


def test_exact_side_match_identifies_chain():
    cands = route_outcomes("CC(=O)O.OCC", _tpl())
    assert cands and cands[0].exact_side_match
    assert cands[0].synth == ["CCO"]
    assert cands[0].sm == ["CC(=O)O"]


def test_last_step_all_fragments_are_sms():
    cands = route_outcomes("CC(=O)O.OCC", _tpl(), last_step=True)
    assert len(cands) == 1 and cands[0].synth == []
    assert sorted(cands[0].sm) == ["CC(=O)O", "CCO"]


def test_fallback_similarity_when_side_changed():
    # side reactant does not match the template's original -> similarity fallback keeps
    # the fragment closest to the original chain precursor (butanol vs ethanol) open
    cands = route_outcomes("CCC(=O)O.OCCCC", _tpl())
    assert any(not c.exact_side_match and c.synth == ["CCCCO"] for c in cands)


def test_unparsable_fragment_rejects_outcome():
    assert route_outcomes("CC(=O)O.not_a_smiles", _tpl()) == []


def test_coupling_step_keeps_both_synth_fragments():
    # a convergence step's retro outcome carries TWO synthesized fragments; both must
    # stay open on the frontier (the old engine mislabeled one a starting material)
    tpl = _tpl(orig_product="CC(=O)NCc1ccccc1", orig_chain_precursor=None,
               orig_synth_precursors=["CC(=O)O", "NCc1ccccc1"],
               orig_side_reactants=[], orig_reactants=["CC(=O)O", "NCc1ccccc1"])
    cands = route_outcomes("CC(=O)O.NCc1ccccc1", tpl)
    assert cands and cands[0].exact_side_match
    assert sorted(cands[0].synth) == ["CC(=O)O", "NCc1ccccc1"]
    assert cands[0].sm == []


def test_budget_absorbs_migrated_seed():
    # a fragment that matches an unconsumed purchasable of the whole route may be a
    # migrated seed -> an SM-absorbing candidate is offered alongside the open one
    tpl = _tpl(orig_synth_precursors=[], orig_chain_precursor=None,
               orig_side_reactants=["CC(=O)O"])
    budget = Counter({"CC(=O)O": 1, "CCO": 1})
    cands = route_outcomes("CC(=O)O.OCC", tpl, sm_budget=budget)
    routings = {(tuple(sorted(c.synth)), tuple(sorted(c.sm))) for c in cands}
    assert ((), ("CC(=O)O", "CCO")) in routings          # absorbed as migrated seed
    assert (("CCO",), ("CC(=O)O",)) in routings          # kept open on the frontier
