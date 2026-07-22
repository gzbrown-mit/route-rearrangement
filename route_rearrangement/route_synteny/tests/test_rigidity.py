"""The per-route rigidity map.

The map's job is to keep two very different claims apart — "chemistry forbids this swap" and
"chemists don't happen to do it" — so most of these tests are about the boundary between them.
Conflating the two is the failure mode that would make the output actively misleading rather
than merely wrong.
"""

import pytest

from route_rearrangement.route_synteny import rigidity
from route_rearrangement.route_synteny.corpus import UNKNOWN, Genome
from route_rearrangement.route_synteny.rigidity import (ANTI, CONVENTIONAL, FORCED, FREE,
                                                        UNKNOWN_EVIDENCE)


def _genome(families, constraints=(), n_steps=None):
    n = n_steps or len(families)
    ids = list(range(n, 0, -1))
    return Genome(route_id="r", n_steps=n, step_ids=ids, families=list(families),
                  constraints={"exposure": list(constraints)})


def _table(fa, fb, observed, n_routes, n_effective, explained=0.0):
    """One-row corpus table, keys canonically sorted as precedence.py writes them."""
    a, b = sorted((fa, fb))
    obs = observed if (a, b) == (fa, fb) else n_routes - observed
    return {(a, b): {"family_a": a, "family_b": b, "observed": obs, "n_routes": n_routes,
                     "n_effective": n_effective, "explained": {"exposure": explained}}}


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------
def test_lookup_flips_strength_for_the_routes_direction():
    """Table pairs are alphabetical; a route running them the other way needs the flip."""
    t = _table("A", "B", observed=90, n_routes=100, n_effective=50)
    _, fwd = rigidity._lookup(t, "A", "B")
    _, rev = rigidity._lookup(t, "B", "A")
    assert fwd == pytest.approx(0.9)
    assert rev == pytest.approx(0.1)


def test_lookup_misses_cleanly():
    assert rigidity._lookup({}, "A", "B") == (None, None)


# ---------------------------------------------------------------------------
# The verdict cross-tab
# ---------------------------------------------------------------------------
def test_forced_wins_over_any_precedent():
    """A swap chemistry forbids is not 'conventional', however strong the literature is."""
    g = _genome(["A", "B"], constraints=[(2, 1)])
    m = rigidity.rigidity_map(g, _table("A", "B", 100, 100, 99), tier="exposure")
    assert [p.verdict for p in m.pairs] == [FORCED]
    assert m.pairs[0].forced_prob == pytest.approx(1.0)


@pytest.mark.parametrize("observed,n_eff,expected", [
    (95, 50, CONVENTIONAL),      # literature runs it this way
    (5, 50, ANTI),               # literature runs it the other way; this route does not
    (50, 50, FREE),              # no preference either way
    (95, 3, UNKNOWN_EVIDENCE),   # strong-looking, but 3 skeletons is not evidence
])
def test_unforced_pairs_classify_on_strength_and_evidence(observed, n_eff, expected):
    g = _genome(["A", "B"])      # no constraints: the pair is free to move
    m = rigidity.rigidity_map(g, _table("A", "B", observed, 100, n_eff), tier="exposure")
    assert m.pairs[0].verdict == expected


def test_missing_from_the_table_is_unknown_not_free():
    """Absence of evidence must not read as evidence of freedom."""
    g = _genome(["A", "B"])
    assert rigidity.rigidity_map(g, {}, tier="exposure").pairs[0].verdict == UNKNOWN_EVIDENCE


def test_evidence_floor_counts_skeletons_not_routes():
    """1,690 routes from 3 skeletons is 3 pieces of evidence, and must not pass the floor."""
    g = _genome(["A", "B"])
    m = rigidity.rigidity_map(g, _table("A", "B", 1600, 1690, 3), tier="exposure",
                              min_effective=10)
    assert m.pairs[0].verdict == UNKNOWN_EVIDENCE
    assert m.pairs[0].lit_n_routes == 1690 and m.pairs[0].lit_n_effective == 3


# ---------------------------------------------------------------------------
# Coverage — the regression that mattered
# ---------------------------------------------------------------------------
def test_every_step_pair_is_mapped_even_when_families_repeat():
    """``forced`` is a fact about steps, so it must not need a nameable transformation.

    Restricting the map to nameable family pairs previously reduced a 7-step route to one
    mapped pair of 21 and silently dropped every forced verdict with it.
    """
    g = _genome(["A", "A", "A", "B"], constraints=[(4, 3), (3, 2)])
    m = rigidity.rigidity_map(g, {}, tier="exposure")
    assert len(m.pairs) == 6                      # all 4-choose-2 step pairs
    assert m.counts()[FORCED] == 3                # 4>3, 3>2 and the transitive 4>2
    ambiguous = [p for p in m.pairs if not p.earlier_named or not p.later_named]
    assert ambiguous, "repeated family should be flagged as unnameable, not silently labelled"


def test_repeated_families_are_reported_not_hidden():
    g = _genome(["A", "A", "B", UNKNOWN])
    m = rigidity.rigidity_map(g, {}, tier="exposure")
    assert m.skipped_repeat_families == 1


# ---------------------------------------------------------------------------
# Feeding the engine
# ---------------------------------------------------------------------------
def test_conventional_constraints_lock_only_conventions():
    g = _genome(["A", "B"])
    m = rigidity.rigidity_map(g, _table("A", "B", 95, 100, 50), tier="exposure")
    assert rigidity.conventional_constraints(m) == [(2, 1)]


def test_anti_conventional_pairs_are_not_locked():
    """This route already breaks that convention; freezing it would preserve the oddity."""
    g = _genome(["A", "B"])
    m = rigidity.rigidity_map(g, _table("A", "B", 5, 100, 50), tier="exposure")
    assert m.pairs[0].verdict == ANTI
    assert rigidity.conventional_constraints(m) == []


def test_locking_conventions_shrinks_the_search_but_keeps_the_real_route_valid():
    """The constraints come from the route's own direction, so it must survive them."""
    g = _genome(["A", "B", "C"])
    m = rigidity.rigidity_map(g, _table("A", "B", 99, 100, 40), tier="exposure")
    assert m.n_orderings == 6
    assert 1 <= m.n_orderings_conventional < m.n_orderings


def test_no_conventions_leaves_the_search_untouched():
    g = _genome(["A", "B", "C"])
    m = rigidity.rigidity_map(g, {}, tier="exposure")
    assert m.n_orderings_conventional == m.n_orderings


def test_convention_breaking_and_free_sets_are_disjoint():
    g = _genome(["A", "B", "C"])
    m = rigidity.rigidity_map(g, _table("A", "B", 99, 100, 40), tier="exposure")
    breaking = {id(p) for p in rigidity.convention_breaking_pairs(m)}
    free = {id(p) for p in rigidity.free_pairs(m)}
    assert breaking and not (breaking & free)


# ---------------------------------------------------------------------------
# Diversification
# ---------------------------------------------------------------------------
def test_can_be_last_separates_forbidden_from_merely_unprecedented():
    # 3 must precede 2 (chemistry); 3 before 1 is a convention only
    g = _genome(["A", "B", "C"], constraints=[(3, 2)])
    m = rigidity.rigidity_map(g, _table("A", "C", 99, 100, 40), tier="exposure")
    last = m.can_be_last()
    assert last[3] == "no"                 # chemistry forbids moving it later
    assert last[1] == "yes"                # already last, nothing after it
    assert last[2] in ("yes", "unprecedented")


def test_map_survives_a_route_with_no_usable_families():
    g = _genome([UNKNOWN, UNKNOWN])
    m = rigidity.rigidity_map(g, {}, tier="exposure")
    assert len(m.pairs) == 1 and m.pairs[0].verdict == UNKNOWN_EVIDENCE
