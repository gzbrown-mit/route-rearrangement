"""The statistics must be right, and must refuse to be fooled by route clustering."""

import math

import pytest
from scipy.stats import binomtest

from route_rearrangement.literature_precedent import significance as sig


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------
def test_bh_matches_hand_computation():
    p = [0.01, 0.02, 0.03, 0.9]
    q = sig.benjamini_hochberg(p)
    # raw BH: p*n/rank = .04, .04, .04, .9 -> monotone from the top gives .04 for the first three
    assert q == pytest.approx([0.04, 0.04, 0.04, 0.9], abs=1e-12)


def test_bh_is_monotone_and_order_preserving():
    p = [0.5, 0.001, 0.2, 0.04, 0.9, 0.0001]
    q = sig.benjamini_hochberg(p)
    pairs = sorted(zip(p, q))
    for (p1, q1), (p2, q2) in zip(pairs, pairs[1:]):
        assert q1 <= q2 + 1e-12, "q must not decrease as p increases"
    assert all(0.0 <= v <= 1.0 for v in q)
    assert all(qi >= pi - 1e-12 for pi, qi in zip(p, q)), "q must never be below p"


def test_bh_empty():
    assert sig.benjamini_hochberg([]) == []


# ---------------------------------------------------------------------------
# Clustered ratio estimator
# ---------------------------------------------------------------------------
def _from_routes(routes):
    """Sufficient statistics from explicit per-route ``(a_r, n_r)`` pairs."""
    routes = [(a, n) for a, n in routes if n > 0]
    return dict(
        sum_a=sum(a for a, _ in routes), sum_n=sum(n for _, n in routes),
        sum_a2=sum(a * a for a, _ in routes),
        sum_an=sum(a * n for a, n in routes),
        sum_n2=sum(n * n for _, n in routes),
        n_routes=len(routes),
    )


def test_sufficient_statistics_reproduce_the_direct_variance():
    """The expanded form must equal the textbook Σ(a_r - p̂·n_r)² computed directly."""
    routes = [(7, 9), (2, 8), (5, 5), (0, 6), (3, 11)]
    stats = _from_routes(routes)
    fit = sig.cluster_fit(**stats)
    p = stats["sum_a"] / stats["sum_n"]
    direct_ss = sum((a - p * n) ** 2 for a, n in routes)
    direct_var = (len(routes) / (len(routes) - 1)) * direct_ss / stats["sum_n"] ** 2
    assert fit.se_cluster == pytest.approx(math.sqrt(direct_var), rel=1e-12)


def test_clustering_kills_a_pattern_that_lives_in_few_routes():
    """The headline case: 100/150 looks overwhelming until you see it came from 3 routes.

    Two routes run the pair one way every time, a third runs it the other way every time.
    The naive binomial sees 150 independent coin flips; the clustered estimator sees three
    facts that disagree, and correctly declines to call it.
    """
    routes = [(50, 50), (50, 50), (0, 50)]
    stats = _from_routes(routes)
    fit = sig.cluster_fit(**stats)

    naive_p = binomtest(stats["sum_a"], stats["sum_n"], 0.5).pvalue
    assert naive_p < 1e-4, "the naive test should be very confident here"

    assert fit.p_hat == pytest.approx(2 / 3)
    assert fit.p_value_cluster > 0.2, "clustered inference must not call this significant"
    assert fit.ci_lo < 0.5 < fit.ci_hi, "clustered CI must span the null"
    assert fit.design_effect > 10, "design effect should expose the inflation"


def test_many_independent_routes_stay_significant():
    """The control for the test above: the same proportion spread over many routes is real."""
    routes = [(2, 3)] * 60
    fit = sig.cluster_fit(**_from_routes(routes))
    assert fit.p_hat == pytest.approx(2 / 3)
    assert fit.p_value_cluster < 0.01
    assert fit.ci_lo > 0.5
    assert fit.design_effect == pytest.approx(1.0, abs=0.35)


def test_design_effect_is_floored_at_one():
    """Homogeneous routes make the between-cluster residuals vanish; the variance must fall
    back to binomial rather than claiming impossible precision."""
    fit = sig.cluster_fit(**_from_routes([(2, 3)] * 60))
    assert fit.design_effect == pytest.approx(1.0, abs=1e-9)
    expected_se = math.sqrt((2 / 3) * (1 / 3) / 180)
    assert fit.se_cluster == pytest.approx(expected_se, rel=1e-9)


def test_single_route_gets_no_interval():
    fit = sig.cluster_fit(**_from_routes([(40, 40)]))
    assert fit.n_routes == 1
    assert fit.se_cluster is None and fit.p_value_cluster is None


def test_perfect_agreement_reports_no_spurious_certainty():
    """Every route unanimous: between-route variance is zero, so a normal interval is
    undefined — report the point estimate, not an infinite z."""
    fit = sig.cluster_fit(**_from_routes([(5, 5), (5, 5), (5, 5)]))
    assert fit.p_hat == 1.0
    assert fit.se_cluster == 0.0
    assert fit.p_value_cluster is None


def test_empty_pair():
    fit = sig.cluster_fit(0, 0, 0, 0, 0, 0)
    assert math.isnan(fit.p_hat) and fit.ci_lo is None


# ---------------------------------------------------------------------------
# Stage/depth confound
# ---------------------------------------------------------------------------
def test_depth_null_absorbs_purely_positional_ordering():
    """Pairs whose order is exactly what depth predicts must show ~zero excess."""
    beta = 4.0
    rows, depths = [], []
    for i in range(40):
        da = 0.02 * i
        db = 1.0 - da
        p = 1.0 / (1.0 + math.exp(-beta * (db - da)))
        n = 200
        rows.append((da, db, int(round(p * n)), n))
        depths.append((da, db))
    null = sig.fit_depth_null(rows)
    assert null is not None
    assert null.beta == pytest.approx(beta, rel=0.1)

    for (da, db), (_, _, a, n) in zip(depths, rows):
        excess = (sig._logit(a / n) - sig._logit(null.predict(da, db))) / math.log(2)
        assert abs(excess) < 0.3, "depth-explained ordering should leave little excess"


def test_depth_null_leaves_a_genuine_constraint_visible():
    """A pair locked in one order despite identical depths keeps its full excess."""
    rows = [(0.1 * i, 0.1 * i + 0.5, 150, 200) for i in range(9)]
    rows.append((0.5, 0.5, 199, 200))       # same depth, still locked
    null = sig.fit_depth_null(rows)
    assert null is not None
    excess = (sig._logit(199 / 200) - sig._logit(null.predict(0.5, 0.5))) / math.log(2)
    assert excess > 3.0


def test_depth_null_needs_data():
    assert sig.fit_depth_null([(0.1, 0.2, 5, 10)]) is None


# ---------------------------------------------------------------------------
# End-to-end over one rung table
# ---------------------------------------------------------------------------
def _table(pairs, keys=("A", "B", "C"), depth=None):
    return {
        "rung": "synthon_shell0", "rung_index": 6,
        "counts": {"n_keys_total": len(keys)},
        "keys": list(keys),
        "density": [10] * len(keys),
        "depth": depth if depth is not None else [0.5] * len(keys),
        "pair_columns": ["a", "b", "n_first_second", "n_second_first", "n_same_step",
                         "n_material_forced", "n_routes", "sum_a", "sum_n",
                         "sum_a2", "sum_an", "sum_n2"],
        "pairs": pairs,
    }


def test_analyze_rung_filters_and_reports():
    # a real pair: 40 routes, 2-of-3 each way; and a starved pair below the support floor
    a_r, n_r, R = 2, 3, 40
    strong = [0, 1, a_r * R, (n_r - a_r) * R, 0, 0, R,
              a_r * R, n_r * R, a_r * a_r * R, a_r * n_r * R, n_r * n_r * R]
    weak = [1, 2, 3, 1, 0, 0, 2, 3, 4, 5, 6, 8]
    res = sig.analyze_rung(_table([strong, weak]), min_n=30, min_routes=5)
    assert res["counts"]["n_pairs_tested"] == 1
    assert res["counts"]["n_pairs_skipped_low_support"] == 1
    row = res["pairs"][0]
    assert row["p_first"] == pytest.approx(2 / 3)
    assert row["q_cluster"] is not None and row["q_cluster"] < 0.05
    assert sig.significant(res["pairs"])


def test_significant_requires_the_ci_to_exclude_the_null():
    rows = [{"q_cluster": 0.001, "ci_lo": 0.4, "ci_hi": 0.9, "excess_log2_odds": 1.0}]
    assert sig.significant(rows) == []
    rows[0]["ci_lo"] = 0.6
    assert len(sig.significant(rows)) == 1
