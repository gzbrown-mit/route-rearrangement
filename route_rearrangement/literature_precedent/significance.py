"""Per-pair inference: is this ordering pattern real, or is it counting noise and confounds?

Three things separate a claim like *"SNAr precedes nitro reduction"* from an artifact, and the
upstream stack tests none of them — its verdicts come from bare constants
(``MIN_LOCKED_SUPPORT = 5``, a raw ``order_bias`` fraction), so 4/5 and 400/500 are treated
identically.

**1. Is the asymmetry more than chance?**  An exact two-sided binomial test against p = 0.5 on
the strictly-ordered observations, then Benjamini-Hochberg FDR *within each rung separately*.
Pooling rungs would be wrong: the same chemistry appears at several rungs, so those tests are
nested rather than independent and one BH pass over all of them distorts q.

**2. Are the observations independent?**  They are not.  Observations from one route are
correlated, and PaRoutes carries many near-duplicate routes, so a pattern seen 300 times across
4 routes is roughly 4 facts, not 300.  We use a clustered ratio estimator over routes: from the
sufficient statistics :mod:`.aggregate` carries, the cluster-robust variance is exact and needs
no bootstrap.  The **design effect** it yields — how much the naive test overstates precision —
is reported per pair, because it is usually the single most informative number in the row.

**3. Is the asymmetry about ordering at all?**  Much of it is not: some transformations simply
happen early in routes and others late, which produces asymmetry with no ordering *constraint*
behind it.  We fit a one-parameter logistic on the difference in mean normalized depth between
the two keys, corpus-wide, and report each pair's **excess** log-odds over that depth-only
prediction.  Pairs that survive are ordering constraints; pairs that do not are stage
artifacts.  The fitted slope and its pseudo-R² say how much of the corpus's ordering is *just*
stage, which is worth knowing on its own.

Usage::

    python -m route_rearrangement.literature_precedent.significance \\
        --pairs results/agg/pairs_*.json --out results/order_significance.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, t as student_t

from . import ladder

Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """BH-adjusted q-values, order-preserving and monotone (step-up, cumulative-min)."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(np.asarray(pvals, dtype=float))
    ranked = np.asarray(pvals, dtype=float)[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]   # enforce monotonicity from the largest down
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.tolist()


# ---------------------------------------------------------------------------
# Clustered ratio estimator
# ---------------------------------------------------------------------------
@dataclass
class ClusterFit:
    """Route-clustered estimate of p = P(key_a's transformation forms first)."""

    p_hat: float
    se_cluster: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    p_value_cluster: Optional[float]
    design_effect: Optional[float]   # cluster variance / binomial variance; >1 means inflated
    n_routes: int
    n_obs: int


def cluster_fit(sum_a: int, sum_n: int, sum_a2: int, sum_an: int, sum_n2: int,
                n_routes: int) -> ClusterFit:
    """Cluster-robust inference on the direction proportion, from route sufficient statistics.

    The estimator is the ratio ``p̂ = Σa_r / Σn_r`` over routes *r*, whose linearized variance
    is ``[R/(R-1)] · Σ_r (a_r - p̂ n_r)² / (Σ n_r)²``.  Expanding the square is what lets it be
    computed from ``Σa²``, ``Σan`` and ``Σn²`` alone — no per-route storage, no resampling.

    That between-cluster estimator is degenerate when the routes happen to agree closely: if
    every route reports the same ratio its residuals vanish and the formula claims a standard
    error of zero, which is obviously false — the within-route sampling uncertainty is still
    there.  So the variance is floored at the independent-observations binomial value.
    Chemically, intra-route correlation is non-negative (steps in one route share a substrate,
    a chemist and a patent), so a design effect below 1 is sampling noise rather than evidence
    that clustering *helps*, and refusing to go below iid is the conservative reading.
    """
    if sum_n <= 0:
        return ClusterFit(float("nan"), None, None, None, None, None, n_routes, 0)
    p = sum_a / sum_n
    if n_routes < 2:
        # one cluster: the data cannot speak to between-route variability at all
        return ClusterFit(p, None, None, None, None, None, n_routes, sum_n)
    ss = sum_a2 - 2.0 * p * sum_an + p * p * sum_n2
    ss = max(ss, 0.0)                                  # guard float cancellation
    var_cluster = (n_routes / (n_routes - 1.0)) * ss / (sum_n ** 2)
    var_binom = p * (1.0 - p) / sum_n
    var = max(var_cluster, var_binom)
    se = math.sqrt(var)
    if se <= 0.0:
        # p̂ is exactly 0 or 1 *and* every route agreed: no normal-theory interval exists.
        # Report the point estimate rather than a spuriously infinite z.
        return ClusterFit(p, 0.0, p, p, None, None, n_routes, sum_n)
    df = n_routes - 1
    tcrit = float(student_t.ppf(0.975, df))
    lo, hi = p - tcrit * se, p + tcrit * se
    tstat = (p - 0.5) / se
    pval = float(2 * student_t.sf(abs(tstat), df))
    # reported against the floored variance, so it reads as "the naive test overstates
    # precision by this factor" and is never misleadingly below 1
    deff = (var / var_binom) if var_binom > 0 else None
    return ClusterFit(p, se, max(0.0, lo), min(1.0, hi), pval, deff, n_routes, sum_n)


# ---------------------------------------------------------------------------
# Stage (depth) confound
# ---------------------------------------------------------------------------
@dataclass
class DepthNull:
    """Corpus-wide depth-only model of ordering: ``logit p = beta · (depth_b - depth_a)``."""

    beta: float
    pseudo_r2: float
    n_pairs_fit: int

    def predict(self, depth_a: float, depth_b: float) -> float:
        z = self.beta * (depth_b - depth_a)
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))


def fit_depth_null(rows: Sequence[Tuple[float, float, int, int]]) -> Optional[DepthNull]:
    """Fit the one-parameter depth model on ``(depth_a, depth_b, n_first_second, n_total)``.

    A single slope, deliberately: the point is not to predict well but to quantify how much
    ordering is explained by *position alone*, and one interpretable number does that.  Fit by
    weighted Newton steps on the binomial log-likelihood; ``pseudo_r2`` is McFadden's against
    the intercept-free null p = 0.5.
    """
    data = [(da, db, a, n) for da, db, a, n in rows
            if n > 0 and da is not None and db is not None]
    if len(data) < 10:
        return None
    x = np.array([db - da for da, db, _, _ in data], dtype=float)
    a = np.array([r[2] for r in data], dtype=float)
    n = np.array([r[3] for r in data], dtype=float)

    beta = 0.0
    for _ in range(50):
        z = np.clip(beta * x, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = float(np.sum(x * (a - n * p)))
        hess = float(np.sum(x * x * n * p * (1.0 - p)))
        if hess <= 1e-12:
            break
        step = grad / hess
        beta += step
        if abs(step) < 1e-9:
            break

    def _ll(b: float) -> float:
        z = np.clip(b * x, -40.0, 40.0)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-12, 1 - 1e-12)
        return float(np.sum(a * np.log(p) + (n - a) * np.log(1.0 - p)))

    ll_null = float(np.sum(n) * math.log(0.5))
    ll_fit = _ll(beta)
    pseudo = 1.0 - (ll_fit / ll_null) if ll_null != 0 else 0.0
    return DepthNull(beta=beta, pseudo_r2=pseudo, n_pairs_fit=len(data))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Per-rung analysis
# ---------------------------------------------------------------------------
def analyze_rung(table: dict, *, min_n: int = 30, min_routes: int = 5) -> dict:
    """Test every pair in one rung's aggregate table."""
    keys: List[str] = table["keys"]
    depth: List[Optional[float]] = table.get("depth") or [None] * len(keys)
    density: List[int] = table.get("density") or [0] * len(keys)
    cols = table["pair_columns"]
    idx = {c: i for i, c in enumerate(cols)}

    tested: List[dict] = []
    skipped_low_support = 0
    for row in table["pairs"]:
        ia, ib = row[idx["a"]], row[idx["b"]]
        n_fs, n_sf = row[idx["n_first_second"]], row[idx["n_second_first"]]
        n = n_fs + n_sf
        n_routes = row[idx["n_routes"]]
        if n < min_n or n_routes < min_routes:
            skipped_low_support += 1
            continue
        fit = cluster_fit(row[idx["sum_a"]], row[idx["sum_n"]], row[idx["sum_a2"]],
                          row[idx["sum_an"]], row[idx["sum_n2"]], n_routes)
        naive_p = float(binomtest(n_fs, n, 0.5).pvalue)
        tested.append({
            "a": ia, "b": ib,
            "n_first_second": n_fs, "n_second_first": n_sf,
            "n_same_step": row[idx["n_same_step"]],
            "n_material_forced": row[idx["n_material_forced"]],
            "n_obs": n, "n_routes": n_routes,
            "p_first": fit.p_hat,
            "log2_odds": _logit(fit.p_hat) / math.log(2.0),
            "p_naive": naive_p,
            "p_cluster": fit.p_value_cluster,
            "se_cluster": fit.se_cluster,
            "ci_lo": fit.ci_lo, "ci_hi": fit.ci_hi,
            "design_effect": fit.design_effect,
            "depth_a": depth[ia] if ia < len(depth) else None,
            "depth_b": depth[ib] if ib < len(depth) else None,
            "density_a": density[ia] if ia < len(density) else 0,
            "density_b": density[ib] if ib < len(density) else 0,
        })

    null = fit_depth_null([(r["depth_a"], r["depth_b"], r["n_first_second"], r["n_obs"])
                           for r in tested])
    for r in tested:
        if null is not None and r["depth_a"] is not None and r["depth_b"] is not None:
            p_null = null.predict(r["depth_a"], r["depth_b"])
            r["p_depth_null"] = p_null
            r["excess_log2_odds"] = (_logit(r["p_first"]) - _logit(p_null)) / math.log(2.0)
        else:
            r["p_depth_null"] = None
            r["excess_log2_odds"] = None

    for name, src in (("q_naive", "p_naive"), ("q_cluster", "p_cluster")):
        vals = [r[src] for r in tested]
        if any(v is None for v in vals):
            # cluster p is undefined for perfectly-agreeing pairs; treat as untested for FDR
            usable = [i for i, v in enumerate(vals) if v is not None]
            qs = benjamini_hochberg([vals[i] for i in usable])
            for r in tested:
                r[name] = None
            for i, q in zip(usable, qs):
                tested[i][name] = q
        else:
            for r, q in zip(tested, benjamini_hochberg(vals)):
                r[name] = q

    return {
        "rung": table["rung"],
        "rung_index": table["rung_index"],
        "counts": dict(table.get("counts", {}),
                       n_pairs_tested=len(tested),
                       n_pairs_skipped_low_support=skipped_low_support,
                       min_n=min_n, min_routes=min_routes),
        "depth_null": asdict(null) if null is not None else None,
        "keys": keys,
        # carried through unchanged: resolve_pairs joins rungs on this lineage, not on keys
        "parent_key": table.get("parent_key"),
        "env_samples": table.get("env_samples"),
        "pairs": tested,
    }


def significant(rows: Sequence[dict], *, q: float = 0.05, use_cluster: bool = True,
                require_excess: bool = False) -> List[dict]:
    """Rows passing the significance bar — clustered by default, because the naive one lies."""
    out = []
    for r in rows:
        qv = r.get("q_cluster") if use_cluster else r.get("q_naive")
        if qv is None or qv > q:
            continue
        if use_cluster and r.get("ci_lo") is not None:
            if r["ci_lo"] <= 0.5 <= r["ci_hi"]:
                continue
        if require_excess and not (r.get("excess_log2_odds") or 0.0):
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Backoff across rungs
# ---------------------------------------------------------------------------
def resolve_pairs(per_rung: Dict[str, dict], *, min_n: int = 30,
                  min_routes: int = 5) -> List[dict]:
    """For every pair of transformations, the finest rung that can actually support a verdict.

    Rungs name the same chemistry with different key strings, so the cross-rung join cannot be
    done on the key — it is done on the **lineage** each rung's table carries: every key
    records its own key at the coarsest rung, so a fine pair and a coarse pair are the same
    chemistry when both their keys share a parent.  (Joining on key strings instead silently
    matches nothing above the coarsest rung, which makes every pair appear to resolve there.)

    Many fine pairs collapse into one coarse pair.  For a given coarse pair we take, at each
    rung, the **best-supported** fine pair beneath it: the ladder exists to make the most
    specific defensible statement, and that is the most specific one with evidence behind it.
    """
    coarse = ladder.RUNGS[-1].name
    if coarse not in per_rung:
        return []

    # {rung: {(parent_a, parent_b): best-supported row}}
    by_rung_lookup: Dict[str, Dict[Tuple[str, str], dict]] = {}
    for name, res in per_rung.items():
        keys = res["keys"]
        parents = res.get("parent_key") or [None] * len(keys)
        best: Dict[Tuple[str, str], dict] = {}
        for r in res["pairs"]:
            pa = parents[r["a"]] if r["a"] < len(parents) else None
            pb = parents[r["b"]] if r["b"] < len(parents) else None
            if not pa or not pb:
                continue
            # the pair's canonical direction is set by its own key order; re-canonicalize on
            # the parents and flip the proportion when the parent order disagrees
            flip = (keys[r["a"]] <= keys[r["b"]]) != (pa <= pb)
            key = (pa, pb) if pa <= pb else (pb, pa)
            row = dict(r, p_first=(1.0 - r["p_first"]) if flip else r["p_first"])
            prev = best.get(key)
            if prev is None or row["n_obs"] > prev["n_obs"]:
                best[key] = row
        by_rung_lookup[name] = best

    out = []
    for pair_keys in by_rung_lookup[coarse]:
        support = {}
        routes = {}
        verdicts = {}
        for name in ladder.RUNG_NAMES:
            r = by_rung_lookup.get(name, {}).get(pair_keys)
            if r is None:
                continue
            support[name] = r["n_obs"]
            routes[name] = r["n_routes"]
            verdicts[name] = r
        res = ladder.resolve(support, routes, min_n=min_n, min_routes=min_routes)
        chosen = verdicts.get(res.rung) if res.rung else None
        out.append({
            "key_a": pair_keys[0], "key_b": pair_keys[1],
            "rung_fired": res.rung,
            "backed_off": res.backed_off,
            "starved_rungs": res.starved,
            "resolved": chosen,
            "per_rung": {k: {"p_first": v["p_first"], "n_obs": v["n_obs"],
                             "n_routes": v["n_routes"], "q_cluster": v.get("q_cluster")}
                         for k, v in verdicts.items()},
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="pairs_<rung>.json file(s) from literature_precedent.aggregate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--min-routes", type=int, default=5)
    args = ap.parse_args(argv)

    per_rung: Dict[str, dict] = {}
    for p in args.pairs:
        with open(p) as fh:
            table = json.load(fh)
        res = analyze_rung(table, min_n=args.min_n, min_routes=args.min_routes)
        per_rung[res["rung"]] = res
        sig = significant(res["pairs"])
        c = res["counts"]
        print(f"{res['rung']:20s} tested={c['n_pairs_tested']:7d} "
              f"significant={len(sig):6d} keys={c.get('n_keys_total', 0):7d}")

    resolved = resolve_pairs(per_rung, min_n=args.min_n, min_routes=args.min_routes)
    fired = sum(1 for r in resolved if r["rung_fired"])
    print(f"backoff: {fired}/{len(resolved)} coarse pairs resolved at some rung")

    out = {
        "min_n": args.min_n, "min_routes": args.min_routes,
        "rungs": per_rung,
        "resolved": resolved,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
