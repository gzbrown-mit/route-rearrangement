from route_rearrangement.metrics.base import intermediates, reactions, retro_tree

from .conftest import corpus_required

# a minimal 2-step linear route record (map-free), shaped like a routes.jsonl entry
RECORD = {
    "tree_id": "toy", "ordering": [2, 1], "is_original_order": True, "target": "CCOC(C)=O",
    "steps": [
        {"position": 1, "orig_step_id": 2, "orig_rxn_index": -1, "retro_smarts": "a>>b",
         "new_rxn": "CCO.CC(=O)Cl>>CCOC(C)=O", "chain_precursor": None,
         "side_reactants": ["CCO", "CC(=O)Cl"], "new_product": "CCOC(C)=O",
         "outcome_rank": 0, "n_outcomes": 1, "exact_side_match": True, "sim_score": 1.0},
        {"position": 2, "orig_step_id": 1, "orig_rxn_index": -1, "retro_smarts": "c>>d",
         "new_rxn": "CCOC(C)=O.CO>>COC(C)=O.CCO", "chain_precursor": "CCOC(C)=O",
         "side_reactants": ["CO"], "new_product": "COC(C)=O",
         "outcome_rank": 0, "n_outcomes": 1, "exact_side_match": True, "sim_score": 1.0},
    ],
}


def test_reactions_and_intermediates_in_order():
    rxns = reactions(RECORD)
    assert [r.position for r in rxns] == [1, 2]
    assert rxns[1].chain_precursor == "CCOC(C)=O"       # product of position 1
    assert intermediates(RECORD) == ["CCOC(C)=O", "COC(C)=O"]


def test_retro_tree_is_a_spine_with_leaves():
    tree = retro_tree(RECORD)
    assert tree["smiles"] == "COC(C)=O"                 # target = last product
    child_smis = {c["smiles"] for c in tree["child"]}
    assert "CCOC(C)=O" in child_smis                    # chain precursor recurses
    assert "CO" in child_smis                           # building block leaf
    chain = next(c for c in tree["child"] if c["smiles"] == "CCOC(C)=O")
    assert {c["smiles"] for c in chain["child"]} == {"CCO", "CC(=O)Cl"}


def test_complexity_and_accessibility_run():
    from route_rearrangement.metrics import accessibility, complexity
    if complexity.available():
        prof = complexity.complexity_profile(RECORD)
        assert "peak" in prof and "score" in prof
    if accessibility.available():
        acc = accessibility.accessibility(RECORD)
        assert "bottleneck" in acc and "score" in acc


def test_isolability_flags_unstable_intermediates():
    from route_rearrangement.metrics import isolability
    assert isolability.available()
    # a 2-step route whose transient intermediate is an acyl chloride (a bench liability),
    # and one whose intermediate is a benign ester — the former must score strictly worse.
    unstable = {
        "target": "CCOC(C)=O",
        "steps": [
            {"position": 1, "new_product": "CC(=O)Cl", "chain_precursor": None,
             "side_reactants": ["CC(=O)O"], "outcome_rank": 0, "n_outcomes": 1,
             "orig_step_id": 1, "orig_rxn_index": -1, "retro_smarts": "a>>b",
             "new_rxn": "CC(=O)O>>CC(=O)Cl", "exact_side_match": True, "sim_score": 1.0},
            {"position": 2, "new_product": "CCOC(C)=O", "chain_precursor": "CC(=O)Cl",
             "side_reactants": ["CCO"], "outcome_rank": 0, "n_outcomes": 1,
             "orig_step_id": 2, "orig_rxn_index": -1, "retro_smarts": "c>>d",
             "new_rxn": "CC(=O)Cl.CCO>>CCOC(C)=O", "exact_side_match": True, "sim_score": 1.0},
        ],
    }
    res = isolability.isolability(unstable)
    assert "acyl_halide" in res["groups"] and res["score"] < 0
    # the final target (ester) is excluded, so only the acyl chloride intermediate counts
    assert len(res["per_intermediate"]) == 1


def test_carried_complexity_rewards_building_late():
    from route_rearrangement.metrics import carried_complexity as cc
    assert cc.available()

    def route(sizes):
        # linear chain of the given intermediate heavy-atom sizes; sizes[0] is the starting
        # material, each later step grows the chain (SMILES = carbon strings, precursor = prev).
        steps = []
        for i, hp in enumerate(sizes, start=1):
            steps.append({
                "position": i, "new_product": "C" * hp,
                "chain_precursor": ("C" * sizes[i - 2]) if i > 1 else None,
                "side_reactants": [], "outcome_rank": 0, "n_outcomes": 1,
                "orig_step_id": i, "orig_rxn_index": -1, "retro_smarts": "a>>b",
                "new_rxn": "C>>C", "exact_side_match": True, "sim_score": 1.0})
        return {"target": "C" * sizes[-1], "steps": steps}

    # both reach a size-15 target; "early" installs the bulk at step 2 (carried through the
    # rest of the route), "late" installs it at the final step (carried nowhere).
    early = cc.carried_complexity(route([3, 13, 14, 15]))
    late = cc.carried_complexity(route([3, 4, 5, 15]))
    assert late["score"] > early["score"]        # building late is rewarded


@corpus_required
def test_exposure_scores_an_ordering(load_tree):
    from synthesis_extraction.dependency.route_graph import build_route_graph
    from synthesis_extraction.dependency.analyze import dependency_graph_from_full_graph
    from route_rearrangement.metrics.exposure import ExposureScorer

    tg = load_tree("106_201")
    full = build_route_graph(tg, "106_201")
    dep = dependency_graph_from_full_graph(full, "106_201")
    scorer = ExposureScorer()
    res = scorer.score("106_201", full, dep.incidental_order())
    assert "n_destroyed" in res and "score" in res
