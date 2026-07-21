"""Tier 1 deterministic feasibility checks.

The anchor case is the nitro/SNAr/reduction motif: the nitro group is both the
ring-activating EWG the SNAr needs and the masked aniline.  Atom bookkeeping alone
permits reducing it first (measured on PaRoutes n1-8: 3,360 of 6,720 materially-valid
orderings do), so only an electronic rule rejects it.
"""

import pytest

from route_rearrangement.feasibility import (
    Finding,
    _ring_activation,
    _mol,
    _patt,
    audit_route,
    summarize,
)
from route_rearrangement.materialize import MaterializedRoute, StepRecord
from route_rearrangement.motifs import BY_NAME, ENFORCED, MOTIFS
from route_rearrangement.templates import StepTemplate


def _step(position, step_id, product, side=(), synth=(), smarts="[c:1]>>[c:1]"):
    return StepRecord(
        position=position, orig_step_id=step_id, orig_rxn_index=-1,
        retro_smarts=smarts,
        new_rxn=".".join(list(side) + list(synth)) + ">>" + product,
        chain_precursor=synth[0] if len(synth) == 1 else None,
        side_reactants=list(side), new_product=product, outcome_rank=0, n_outcomes=1,
        exact_side_match=True, sim_score=1.0,
        parent_step_id=None, synth_precursors=list(synth))


def _route(steps):
    return MaterializedRoute(ordering=[s.orig_step_id for s in steps], status="ok",
                             target=steps[-1].new_product, steps=steps)


def _tpls(steps, ok=True):
    return {s.orig_step_id: StepTemplate(
        step_id=s.orig_step_id, rxn_index=-1, orig_rxn="", retro_smarts=s.retro_smarts,
        orig_product=s.new_product, orig_chain_precursor=None,
        retro_identity_ok=ok) for s in steps}


# ---------------------------------------------------------------- ring activation
@pytest.mark.parametrize("name,smiles,activated", [
    ("para-nitro", "O=[N+]([O-])c1ccc(F)cc1", True),
    ("ortho-nitro", "O=[N+]([O-])c1ccccc1F", True),
    ("meta-nitro (not activated)", "O=[N+]([O-])c1cccc(F)c1", False),
    ("bare fluorobenzene", "Fc1ccccc1", False),
    ("para-cyano", "N#Cc1ccc(F)cc1", True),
    ("2-chloropyrimidine (aza)", "Clc1ncccn1", True),
    ("chlorotriazine (aza)", "CSc1nc(C)nc(Cl)n1", True),
    ("para-amino (EDG, deactivated)", "Nc1ccc(F)cc1", False),
])
def test_ring_activation(name, smiles, activated):
    mol = _mol(smiles)
    ipso = mol.GetSubstructMatches(_patt("aryl_halide"), uniquify=True)[0][0]
    assert bool(_ring_activation(mol, ipso)) is activated, name


def test_known_limitation_meta_azine_treated_as_activated():
    """Documented false NEGATIVE, deliberately chosen.

    3-chloropyridine is *not* SNAr-activated (the halide is meta to the ring N), but
    the rule counts any aromatic N in the fused system.  Modelling the exact
    conjugation pattern across 5-rings and fused azines was not worth the false
    POSITIVES it cost on literature routes — and a false positive rejects a real
    synthesis, while this merely fails to reject a bad rearrangement.  Anything
    tightening this must keep the literature false-positive rate at zero.
    """
    mol = _mol("Clc1cccnc1")
    ipso = mol.GetSubstructMatches(_patt("aryl_halide"), uniquify=True)[0][0]
    assert _ring_activation(mol, ipso) is not None


# ------------------------------------------------- the canonical motif, both orders
_ANILINE_ETHER = "Nc1ccc(Oc2ccccc2)cc1"
_NITRO_ETHER = "O=[N+]([O-])c1ccc(Oc2ccccc2)cc1"


def test_snar_on_activated_nitroarene_is_feasible():
    """Literature order: SNAr while the nitro is still present -> no finding."""
    steps = [_step(1, 2, _NITRO_ETHER, side=["O=[N+]([O-])c1ccc(F)cc1", "Oc1ccccc1"]),
             _step(2, 1, _ANILINE_ETHER, synth=[_NITRO_ETHER])]
    findings = audit_route(_route(steps), _tpls(steps))
    assert not [f for f in findings if f.check == "snar_activation"]


def test_snar_after_reduction_is_infeasible():
    """Rearranged order: the aniline is unmasked first, so the SNAr runs on an
    electron-rich ring — atom-conserving, chemically dead, and it must be caught."""
    steps = [_step(1, 1, "Nc1ccc(F)cc1", side=["O=[N+]([O-])c1ccc(F)cc1"]),
             _step(2, 2, _ANILINE_ETHER, side=["Oc1ccccc1"], synth=["Nc1ccc(F)cc1"])]
    findings = audit_route(_route(steps), _tpls(steps))
    snar = [f for f in findings if f.check == "snar_activation"]
    assert snar, "unactivated SNAr was not flagged"
    assert snar[0].severity == "infeasible"
    assert snar[0].motif == "snar_requires_activation"
    assert summarize(findings)["n_infeasible"] >= 1


def test_aza_activated_snar_not_flagged():
    """Chloropyrimidines/triazines are activated by ring nitrogen alone — flagging
    them would drown the signal, since they are ubiquitous in these corpora."""
    steps = [_step(1, 1, "CSc1nc(C)nc(Nc2ccccc2)n1",
                   side=["CSc1nc(C)nc(Cl)n1", "Nc1ccccc1"])]
    findings = audit_route(_route(steps), _tpls(steps))
    assert not [f for f in findings if f.check == "snar_activation"]


# ------------------------------------------------------------ protecting groups
def test_pg_bracket_violation_flagged():
    steps = [_step(1, 3, "CCO"), _step(2, 1, "CCN"), _step(3, 2, "CCC")]
    # bracket protect=3 / deprotect=1; step 2 was inside it originally (3 > 2 > 1)
    findings = audit_route(_route(steps), _tpls(steps), brackets=[(3, 1)])
    pg = [f for f in findings if f.check == "pg_bracket"]
    assert pg and any("outside" in f.detail for f in pg)


def test_pg_bracket_inverted_is_infeasible():
    steps = [_step(1, 1, "CCO"), _step(2, 3, "CCN")]
    findings = audit_route(_route(steps), _tpls(steps), brackets=[(3, 1)])
    pg = [f for f in findings if f.check == "pg_bracket"]
    assert pg and pg[0].severity == "infeasible"


# --------------------------------------------------------------- other checks
def test_free_amine_in_cross_coupling_flagged():
    """The n1-8 case: Boc removed before the Suzuki, so Pd meets a free amine."""
    steps = [_step(1, 1, "c1ccc(-c2ccccc2)cc1",
                   side=["OB(O)c1ccccc1", "Ic1ccc(NCC)cc1"])]
    findings = audit_route(_route(steps), _tpls(steps))
    assert [f for f in findings if f.check == "metal_catalysis_donor"]


def test_redox_chemoselectivity_flagged():
    """Reducing a ketone while an aryl nitro (reduced more easily) survives."""
    steps = [_step(1, 1, "OCc1ccc([N+](=O)[O-])cc1",
                   side=["O=Cc1ccc([N+](=O)[O-])cc1"])]
    findings = audit_route(_route(steps), _tpls(steps))
    assert [f for f in findings if f.check == "redox_chemoselectivity"]


def test_template_self_consistency_surfaced():
    steps = [_step(1, 1, "CCO")]
    findings = audit_route(_route(steps), _tpls(steps, ok=False))
    assert [f for f in findings if f.check == "template_self_consistency"]


def test_fragment_balance_flagged():
    big = "CCCCCCCCCCCCCCCCCCCCc1ccccc1"
    steps = [_step(1, 1, big + "O", synth=[big, "CO"])]
    findings = audit_route(_route(steps), _tpls(steps))
    assert [f for f in findings if f.check == "fragment_balance"]


# ------------------------------------------------- audit is decoupled from the pipeline
def test_pipeline_does_not_apply_feasibility():
    """The generator must stay neutral: no feasibility rule may gate or annotate a
    route during materialization.  The audit is a separate pass over the results."""
    import inspect

    from route_rearrangement import filters, pipeline, run

    for mod in (filters, run, pipeline):
        src = inspect.getsource(mod)
        assert "audit_route" not in src, f"{mod.__name__} calls the feasibility audit"
        assert "summarize(" not in src, f"{mod.__name__} embeds feasibility findings"
    assert "strict" not in inspect.signature(run.process_route).parameters


def test_audit_record_round_trips_a_result_record():
    """audit_record works straight off a routes.jsonl record, with no corpus."""
    from route_rearrangement.feasibility import audit_record
    from route_rearrangement.schema import route_record

    steps = [_step(1, 1, "Nc1ccc(F)cc1", side=["O=[N+]([O-])c1ccc(F)cc1"]),
             _step(2, 2, _ANILINE_ETHER, side=["Oc1ccccc1"], synth=["Nc1ccc(F)cc1"])]
    rec = route_record("t1", _route(steps), ordering_index=0, variant=0,
                       is_original_order=False, identity_roundtrip=True, flags={})
    findings = audit_record(rec)
    assert [f for f in findings if f.check == "snar_activation"]


# ------------------------------------------------------------------- catalogue
def test_every_enforced_motif_has_a_live_check():
    """Each motif claiming a check must name one the audit can actually emit."""
    steps = [_step(1, 1, "CCO")]
    audit_route(_route(steps), _tpls(steps))          # smoke: import graph is sound
    known = {"snar_activation", "pg_bracket", "metal_catalysis_donor",
             "redox_chemoselectivity", "stereocontrol", "fragment_balance",
             "template_self_consistency"}
    for m in ENFORCED:
        assert m.check in known, f"{m.name} names unknown check {m.check}"
    assert BY_NAME["nitro_snar_reduction"].check == "snar_activation"
    assert len(MOTIFS) >= 15
