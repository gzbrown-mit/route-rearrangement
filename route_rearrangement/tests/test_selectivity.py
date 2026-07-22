"""The feature-based selectivity metric: no rule table, so the tests assert on chemistry
the descriptor has to reproduce on its own."""

import pytest

from route_rearrangement.metrics import electronic, selectivity

pytestmark = pytest.mark.skipif(not selectivity.available(),
                                reason="rdEHTTools / embedding unavailable")

# a generic amide-coupling retro template (product >> precursors)
AMIDE = "[C:1](=[O:2])-[N&H1&D2&+0:3]>>[C:1](=[O:2])-[O&H1&D1&+0].[N&H2&D1&+0:3]"
# SNAr: an amine displaces an aryl fluoride
SNAR = "[c:1]-[N&H0&D3&+0:2]>>[c:1]-[F].[N&H1&D2&+0:2]"


def test_template_centre_finds_the_changing_atoms():
    patt, centre = selectivity._template_centre(AMIDE)
    assert patt is not None
    # the carbonyl carbon and the nitrogen change bonding; the carbonyl oxygen does not
    assert centre == {1, 3}


def test_no_rival_no_penalty():
    """A single reacting site costs nothing — the metric only prices competition."""
    step = selectivity._step_selectivity(["CCCCN", "CC(=O)O"], AMIDE)
    assert step["n_rival_sites"] == 0
    assert step["penalty"] == 0.0
    assert step["margin"] == pytest.approx(1.0)


def test_second_identical_site_is_a_coin_flip():
    """Two equally reactive amines: half the frontier density is on the wrong one."""
    step = selectivity._step_selectivity(["NCCCCN", "CC(=O)O"], AMIDE)
    assert step["n_rival_sites"] >= 1
    assert step["penalty"] == pytest.approx(0.5, abs=0.05)


def test_protecting_the_rival_removes_the_penalty():
    """The protecting-group strategy falls out of the descriptor: mask the rival and the
    selectivity liability goes away, with no rule saying Boc protects an amine."""
    free = selectivity._step_selectivity(["NCCCCN", "CC(=O)O"], AMIDE)
    masked = selectivity._step_selectivity(["NCCCCNC(=O)OC(C)(C)C", "CC(=O)O"], AMIDE)
    assert masked["penalty"] < free["penalty"]
    assert masked["penalty"] == 0.0


def test_activation_collapses_when_the_activating_group_is_removed():
    """nitro_snar_reduction, from the wavefunction: reducing the nitro first empties the
    LUMO at the carbon the substitution is aimed at.  This is a feasibility statement, so
    it lands in `activation`, not in the selectivity score."""
    on = selectivity._step_selectivity(["O=[N+]([O-])c1ccc(F)cc1", "C1COCCN1"], SNAR)
    off = selectivity._step_selectivity(["Nc1ccc(F)cc1", "C1COCCN1"], SNAR)
    # read the electrophilic activation explicitly: after the reduction the *nucleophilic*
    # reading becomes the operative one, so the operative-mode number alone would not show it
    on_e = on["activation_by_mode"]["electrophile"]
    off_e = off["activation_by_mode"]["electrophile"]
    assert on_e > 100 * off_e
    assert off_e < 0.01


def test_a_rival_that_does_not_survive_the_step_is_not_a_liability():
    """Both amines of a symmetric diamine react: nothing reactive of that kind is left in
    the product, so there was no selectivity problem.  This is what stops every symmetric
    reagent in the corpus (Boc anhydride, thionyl chloride, Lawesson's) being charged 0.5
    for having two equivalent sites."""
    mono = selectivity._step_selectivity(["NCCCCN", "CC(=O)O"], AMIDE, product="CC(=O)NCCCCN")
    bis = selectivity._step_selectivity(["NCCCCN", "CC(=O)O"], AMIDE,
                                        product="CC(=O)NCCCCNC(C)=O")
    assert mono["penalty"] == pytest.approx(0.5, abs=0.05)   # free amine survives
    assert bis["penalty"] == 0.0                             # both consumed
    assert bis["n_rivals_consumed"] >= 1


def test_steric_factor_direction_and_neutrality():
    """A bulkier rival is discounted, a less bulky one is promoted, an equal one does
    nothing — and the adjustment is clamped so sterics can never fully override electronics."""
    assert selectivity._steric_factor(2.0, 1.0) < 1.0      # rival more crowded than the site
    assert selectivity._steric_factor(1.0, 2.0) > 1.0      # rival less crowded
    assert selectivity._steric_factor(1.5, 1.5) == pytest.approx(1.0)
    import math
    assert selectivity._steric_factor(99.0, 1.0) == pytest.approx(
        math.exp(-selectivity.STERIC_CLAMP))


def test_bulk_is_the_borrowed_descriptor():
    """The steric feature is synthesis_extraction's own heavy_atoms_decay, and the anchor's
    atom map is left as it was found (the routes it runs on are map-free)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CC(C)(N)CN")
    nitrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    bulks = [selectivity._bulk(mol, i) for i in nitrogens]
    assert bulks[0] != bulks[1]                            # the two amines are not equivalent
    assert max(bulks) > min(bulks)
    assert all(a.GetAtomMapNum() == 0 for a in mol.GetAtoms())   # restored


def test_sterics_move_the_margin_toward_the_open_site():
    """Acylating 2-methylpropane-1,2-diamine: the template anchors on the amine at the
    quaternary carbon, which is electronically favoured but crowded, so discounting the
    open primary amine's rival density narrows the margin.  The symmetric diamine, where
    both sites have identical bulk, is untouched."""
    hindered = selectivity._step_selectivity(["CC(C)(N)CN", "CC(=O)O"], AMIDE,
                                             product="CC(C)(N)CNC(C)=O")
    assert hindered["margin"] < hindered["margin_electronic"]
    assert hindered["bulk_site"] > hindered["bulk_rival"]

    symmetric = selectivity._step_selectivity(["NCCCCN", "CC(=O)O"], AMIDE,
                                              product="CC(=O)NCCCCN")
    assert symmetric["margin"] == pytest.approx(symmetric["margin_electronic"], abs=1e-6)


def test_abstains_rather_than_guessing():
    step = selectivity._step_selectivity(["CCCCN", "CC(=O)O"], "not a template")
    assert step["abstain"] == "no_template_centre"
    step = selectivity._step_selectivity(["CCO"], AMIDE)
    assert step["abstain"] == "template_did_not_match"


def test_route_score_sums_the_step_penalties():
    record = {
        "steps": [
            {"position": 1, "new_product": "CC(=O)NCCCCN", "chain_precursor": None,
             "side_reactants": ["NCCCCN", "CC(=O)O"], "retro_smarts": AMIDE},
            {"position": 2, "new_product": "CC(=O)NCCCCNC(C)=O",
             "chain_precursor": "CC(=O)NCCCCN",
             "side_reactants": ["CC(=O)O"], "retro_smarts": AMIDE},
        ]
    }
    out = selectivity.selectivity(record)
    assert out["n_steps"] == 2
    assert out["score"] == pytest.approx(-sum(s["penalty"] for s in out["per_step"]), abs=1e-4)
    assert out["score"] <= 0.0            # higher is better, 0 is a clean route


def test_frontier_is_cached_and_deterministic():
    a = electronic.frontier("O=[N+]([O-])c1ccc(F)cc1")
    b = electronic.frontier("Fc1ccc([N+](=O)[O-])cc1")     # same molecule, other spelling
    assert a is b                                          # canonicalized to one cache entry
    assert a.gap > 0 and len(a.f_plus) == 10
