"""The nulls carry the entire scientific claim, so they are tested as such.

If Null-C is not uniform over linear extensions, "clustered beyond what chemistry forces" means
nothing; if Null-P and Null-C differ in any way other than the constraint set, the
necessity/convention contrast confounds two changes at once.
"""

import math
from collections import Counter

import pytest

from route_rearrangement.route_synteny import nulls
from route_rearrangement.route_synteny.corpus import Genome


def _genome(step_ids, families, constraints=()):
    return Genome(route_id="t", n_steps=len(step_ids), step_ids=list(step_ids),
                  families=list(families), constraints={"material": list(constraints)})


def test_free_null_is_the_constrained_null_with_no_constraints():
    """The two nulls must share a code path, so nothing but the constraint can differ."""
    g = _genome([4, 3, 2, 1], list("ABCD"), constraints=[(4, 3)])
    free = nulls.lattice_for_genome(g, None)
    con = nulls.lattice_for_genome(g, "material")
    assert free.count() == math.factorial(4)
    assert con.count() == math.factorial(4) // 2      # one pair fixed halves the extensions
    assert type(free) is type(con)


def test_constrained_null_never_violates_the_partial_order():
    g = _genome([4, 3, 2, 1], list("ABCD"), constraints=[(4, 2), (3, 1)])
    lat = nulls.lattice_for_genome(g, "material")
    for order in nulls.sample_orders(lat, 200, seed=7):
        assert order.index(4) < order.index(2)
        assert order.index(3) < order.index(1)


def test_constrained_null_is_uniform_over_linear_extensions():
    """A biased 'pick any available step' walk would fail this; the weighted sampler passes."""
    g = _genome([3, 2, 1], list("ABC"), constraints=[(3, 1)])
    lat = nulls.lattice_for_genome(g, "material")
    n_ext = lat.count()
    assert n_ext == 3                       # 3!/2 orderings respect a single precedence
    draws = 6000
    seen = Counter(tuple(o) for o in nulls.sample_orders(lat, draws, seed=11))
    assert len(seen) == n_ext
    expected = draws / n_ext
    chi2 = sum((c - expected) ** 2 / expected for c in seen.values())
    assert chi2 < 13.8, f"non-uniform sampler: {seen}"     # chi2(df=2) p=0.001


def test_sampled_precedence_matches_the_exact_marginals():
    """The analytic path and the Monte Carlo path must agree, or one of them is wrong."""
    g = _genome([4, 3, 2, 1], list("ABCD"), constraints=[(4, 2)])
    lat = nulls.lattice_for_genome(g, "material")
    exact = nulls.exact_precedence(lat)
    draws = 8000
    hits = Counter()
    for order in nulls.sample_orders(lat, draws, seed=3):
        pos = {s: i for i, s in enumerate(order)}
        for a in g.step_ids:
            for b in g.step_ids:
                if a != b and pos[a] < pos[b]:
                    hits[(a, b)] += 1
    for pair, p in exact.items():
        assert hits[pair] / draws == pytest.approx(p, abs=0.03), pair
    assert exact[(4, 2)] == 1.0             # the constrained pair is certain
    assert exact[(3, 1)] == pytest.approx(0.5, abs=1e-9)   # an unconstrained pair is free


def test_reading_families_under_a_permutation_preserves_the_multiset():
    g = _genome([3, 2, 1], ["X", "Y", "X"])
    for order in nulls.sample_orders(nulls.lattice_for_genome(g, None), 20, seed=1):
        assert Counter(nulls.read_families(g, order)) == Counter(g.families)


def test_frequency_null_matches_corpus_family_frequencies():
    genomes = [_genome([2, 1], ["A", "B"]), _genome([2, 1], ["A", "A"]),
               _genome([2, 1], ["A", "?"])]
    freqs = nulls.family_frequencies(genomes)
    assert freqs["A"] == pytest.approx(4 / 5)     # UNKNOWN excluded from the alphabet
    assert freqs["B"] == pytest.approx(1 / 5)
    assert "?" not in freqs

    import random
    draw = nulls.frequency_null_string(4000, freqs, random.Random(0))
    assert Counter(draw)["A"] / 4000 == pytest.approx(0.8, abs=0.03)
