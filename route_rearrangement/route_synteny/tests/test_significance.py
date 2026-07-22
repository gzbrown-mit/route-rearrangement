"""The tail computation and the multiple-testing correction.

The Poisson-binomial DP replaced a plain Monte Carlo count for a specific reason — an MC
p-value is floored at 1/(draws+1), which no correction over millions of candidate intervals can
ever clear — so it is now the load-bearing statistic and is checked against closed forms.
"""

import math

import pytest
from scipy.stats import binom

from route_rearrangement.route_synteny import significance as sig


# ---------------------------------------------------------------------------
# Poisson-binomial tail
# ---------------------------------------------------------------------------
def test_reduces_to_the_binomial_when_probabilities_are_equal():
    n, p = 12, 0.37
    pis = [p] * n
    for k in range(n + 1):
        assert sig.poisson_binomial_sf(pis, k) == pytest.approx(
            float(binom.sf(k - 1, n, p)), rel=1e-9, abs=1e-12)


def test_boundaries():
    pis = [0.5] * 10
    assert sig.poisson_binomial_sf(pis, 0) == 1.0
    assert sig.poisson_binomial_sf(pis, 11) == 0.0
    assert sig.poisson_binomial_sf(pis, 10) == pytest.approx(0.5 ** 10)
    assert sig.poisson_binomial_sf([1.0] * 5, 5) == pytest.approx(1.0)
    assert sig.poisson_binomial_sf([0.0] * 5, 1) == pytest.approx(0.0)


def test_handles_unequal_probabilities():
    pis = [0.1, 0.9, 0.5]
    # P(X >= 3) = 0.1*0.9*0.5
    assert sig.poisson_binomial_sf(pis, 3) == pytest.approx(0.045)
    # P(X >= 1) = 1 - P(none) = 1 - 0.9*0.1*0.5
    assert sig.poisson_binomial_sf(pis, 1) == pytest.approx(1 - 0.045)


def test_tail_reaches_far_below_any_monte_carlo_floor():
    """The whole reason this replaced a draw-counting p-value."""
    p = sig.poisson_binomial_sf([0.5] * 200, 200)
    assert 0 < p < 1e-50


def test_is_monotone_in_k():
    pis = [0.2, 0.4, 0.6, 0.8, 0.3]
    vals = [sig.poisson_binomial_sf(pis, k) for k in range(len(pis) + 1)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# The compactness statistic, on strings
# ---------------------------------------------------------------------------
def test_corpus_statistic_counts_routes_not_occurrences():
    strings = [list("ABAB"), list("AXXB"), list("ZZ")]
    c = frozenset("AB")
    assert sig.corpus_statistic(c, strings, delta=0, max_extra=0) == 1


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------
def test_reference_correction_scales_by_the_interval_universe():
    p = [0.001, 0.01]
    q = sig.fdr_reference(p, n_intervals=1000)
    assert q[0] == pytest.approx(0.001 * 1000 / 1)
    assert q[1] == pytest.approx(min(1.0, 0.01 * 1000 / 2))


def test_corrections_are_monotone_and_bounded():
    p = [0.5, 0.001, 0.2, 0.04, 0.9, 1e-9]
    for q in (sig.fdr_reference(p, 10_000), sig.benjamini_hochberg(p)):
        assert all(0.0 <= v <= 1.0 for v in q)
        pairs = sorted(zip(p, q))
        for (p1, q1), (p2, q2) in zip(pairs, pairs[1:]):
            assert q1 <= q2 + 1e-12


def test_bh_matches_hand_computation():
    assert sig.benjamini_hochberg([0.01, 0.02, 0.03, 0.9]) == pytest.approx(
        [0.04, 0.04, 0.04, 0.9], abs=1e-12)


def test_reference_correction_is_stricter_than_bh():
    """It corrects over every candidate interval, not just the clusters actually tested."""
    p = [1e-8, 1e-6, 0.01]
    ref = sig.fdr_reference(p, n_intervals=1_000_000)
    bh = sig.benjamini_hochberg(p)
    assert all(r >= b - 1e-15 for r, b in zip(ref, bh))


def test_interval_universe_counts_windows_per_route():
    from route_rearrangement.route_synteny.corpus import Genome
    g = Genome(route_id="r", n_steps=4, step_ids=[4, 3, 2, 1], families=list("ABCD"))
    assert sig.n_candidate_intervals([g, g]) == 2 * (4 * 5 // 2)


def test_empty_inputs():
    assert sig.fdr_reference([], 10) == []
    assert sig.benjamini_hochberg([]) == []
    assert sig.n_candidate_intervals([]) == 0
