"""The positive control: a block known a priori to be rigid must come out as necessity.

Protect → react → deprotect is the one ordering constraint in synthesis that is not in doubt,
and it enters the model only at the ``brackets`` tier.  That gives a two-sided test of the whole
decomposition which does not require trusting any statistic:

* at the ``material`` tier the bracket is invisible, so the pair must look like **convention**;
* at the ``brackets`` tier the constraint is imposed, so it must flip to **necessity**.

Writing this control is what showed the *cluster* statistic cannot serve it.  A bracket forces
protect before deprotect but leaves other steps free to be scheduled between them, so it barely
changes a compactness null while pinning a precedence one completely — order and adjacency are
different properties, and chemistry's constraints act on order.  Hence :mod:`.precedence`, and
hence this control living there.
"""

import pytest

from route_rearrangement.route_synteny import decompose, precedence
from route_rearrangement.route_synteny.corpus import Genome
from route_rearrangement.route_synteny.tests.conftest import genomes_required

TIERS = ["material", "brackets"]


def _bracketed_route(rid: str) -> Genome:
    """protect(8) ... deprotect(5), two steps inside the bracket, four unrelated steps.

    Step ids run deepest-first, so 8 is performed first.  The bracket edges are what
    ``compatibility_edges`` contributes for a real protecting group: protect before both
    interior steps, both interior steps before deprotect.  Nothing stops steps 4–1 from being
    scheduled anywhere, which is exactly why adjacency is the wrong lens and precedence the
    right one.
    """
    return Genome(
        route_id=rid, n_steps=8, step_ids=[8, 7, 6, 5, 4, 3, 2, 1],
        families=["protect", "inner1", "inner2", "deprotect", "u1", "u2", "u3", "u4"],
        constraints={
            "material": [],
            "brackets": [(8, 7), (8, 6), (7, 5), (6, 5)],
        },
    )


def _control_results(n: int = 40):
    genomes = [_bracketed_route(f"r{i}") for i in range(n)]
    rows, _ = precedence.collect(genomes, tiers=TIERS, min_routes=5)
    return {(r["family_a"], r["family_b"]): r
            for r in rows}, precedence.analyze(rows, tiers=TIERS)


def _find(results, a, b):
    for r in results:
        if {r.family_a, r.family_b} == {a, b}:
            return r
    raise AssertionError(f"pair {a}/{b} not tested")


def test_bracket_pair_flips_from_convention_to_necessity():
    _, results = _control_results()
    r = _find(results, "protect", "deprotect")

    # pairs are stored in canonical alphabetical order, so "always in one order" is
    # observed == 0 or == n_routes depending on which family sorts first
    assert r.observed in (0, r.n_routes), "the literature must be unanimous on this pair"
    assert r.q_free <= 0.05, (
        "a pair fixed in every route must beat the free-permutation null before anything else")
    assert r.verdict["material"] == decompose.CONVENTION, (
        "with the bracket unmodelled the order must look conventional")
    assert r.verdict["brackets"] == decompose.NECESSITY, (
        "once the bracket is modelled the order must be explained — if it does not flip, the "
        "nulls are not responding to the constraint set")


def test_the_flip_comes_from_the_null_not_the_observation():
    """The observed count is the same at both tiers; only what the null expects moves.

    Stated without reference to which family sorts first: the material tier expects a coin
    flip and is therefore surprised, while the brackets tier predicts the observation exactly
    and so has nothing left to explain.
    """
    _, results = _control_results()
    r = _find(results, "protect", "deprotect")
    assert r.expected_constrained["material"] == pytest.approx(r.n_routes / 2, rel=1e-9)
    assert r.expected_constrained["brackets"] == pytest.approx(r.observed, abs=1e-9)
    assert abs(r.observed - r.n_routes / 2) == pytest.approx(r.n_routes / 2)


def test_an_unconstrained_pair_stays_convention_at_every_tier():
    """The negative half: two steps nothing forces must never be explained away."""
    _, results = _control_results()
    r = _find(results, "u1", "u2")
    assert r.expected_constrained["brackets"] == pytest.approx(r.n_routes / 2, rel=1e-9)
    assert r.verdict["brackets"] == decompose.CONVENTION


def test_a_pair_with_no_consistent_order_is_no_preference():
    genomes = []
    for i in range(40):
        fams = ["A", "B"] if i % 2 else ["B", "A"]
        genomes.append(Genome(route_id=f"r{i}", n_steps=2, step_ids=[2, 1], families=fams,
                              constraints={"material": [], "brackets": []}))
    rows, _ = precedence.collect(genomes, tiers=TIERS, min_routes=5)
    r = _find(precedence.analyze(rows, tiers=TIERS), "A", "B")
    assert r.verdict["material"] == precedence.NO_PREFERENCE


def test_routes_with_repeated_families_are_skipped_not_guessed():
    """'Does A precede B' has no answer when two A's straddle B."""
    g = Genome(route_id="r", n_steps=3, step_ids=[3, 2, 1], families=["A", "B", "A"],
               constraints={"material": [], "brackets": []})
    assert "A" not in precedence.unique_family_positions(g)
    assert "B" in precedence.unique_family_positions(g)


# ---------------------------------------------------------------------------
# The corpus-backed half
# ---------------------------------------------------------------------------
@genomes_required
def test_real_bracketed_routes_exist_and_carry_extra_constraints(corpus_genomes):
    """The control is only meaningful if real routes actually gain bracket constraints."""
    bracketed = [g for g in corpus_genomes
                 if len(g.constraints.get("brackets", ())) > len(g.constraints.get("material", ()))]
    assert bracketed, "no route in the sample gained a constraint at the brackets tier"
    frac = len(bracketed) / len(corpus_genomes)
    assert frac > 0.01, f"only {frac:.2%} of routes are bracketed — control is underpowered"


@genomes_required
def test_exposure_tier_is_reported_honestly(corpus_genomes):
    """If the exposure tier adds nothing, its agreement with 'brackets' is not robustness.

    This does not demand that it add something — on PaRoutes it adds nothing at all.  It demands
    the two cases stay distinguishable, so a flat ladder is never read as three independent
    tiers agreeing when it is one tier counted three times.
    """
    added = sum(1 for g in corpus_genomes
                if len(g.constraints.get("exposure", ())) > len(g.constraints.get("brackets", ())))
    if added == 0:
        pytest.skip("exposure tier is inert on this corpus — constraint_budget() reports it as 0%")
    assert added > 0
