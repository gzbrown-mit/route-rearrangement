"""Readable output — and specifically, output that cannot be quoted out of context.

The headline number here is a fraction of ordering constraints attributable to convention, and
it is an upper bound whose slack depends on how good the chemistry model is.  A table that
printed it as a bare percentage would be quoted as a measurement, so the report is built to
make the bound and its escape hatch inseparable from the number:

* the **sensitivity ladder** shows the split at every necessity tier, so a reader sees how much
  the answer moves when more chemistry is modelled — on PaRoutes, barely, which is itself the
  most important thing to know about it;
* the **convention list** names the strongest convention-classified blocks in readable
  chemistry, because that list is the falsification test.  A chemist who reads it and names the
  mechanism for the top entries has shown the constraint model is under-powered, which is a
  finding about the method rather than about chemists;
* **caps and drops** are printed whenever a run bounded its own coverage.  A truncated corpus
  presented as a whole one is the easiest way to overstate a result by accident.

Family keys are bond-change signatures from :mod:`.step_identity` (``CN0>1|CO1>0`` — a C-N bond
formed and a C-O broken, i.e. an amide coupling), which are legible on their own and need no
external rendering table.

Usage::

    python -m route_rearrangement.route_synteny.report \\
        --precedence results/precedence.json --genomes results/genomes.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from typing import Dict, List, Optional, Sequence

from . import TIER_NAMES
from .decompose import CONVENTION, NECESSITY, format_headline, load_genomes
from .precedence import format_headline as prec_headline


def _render(family: str, envs: Dict[str, str], width: int = 38) -> str:
    env = envs.get(family)
    if env:
        return env if len(env) <= width else env[: width - 1] + "…"
    return family if len(family) <= width else family[: width - 1] + "…"


def freedom_table(genomes) -> str:
    """The ordering-freedom distribution — how much choice a published route actually had."""
    lines = ["ordering freedom (valid orderings per route)", "",
             f"{'tier':>12}{'median':>9}{'mean':>14}{'max':>14}{'frac > 1':>10}",
             "-" * 59]
    for t in TIER_NAMES:
        vals = [g.n_orderings.get(t, 0) for g in genomes if g.n_orderings.get(t)]
        if not vals:
            continue
        lines.append(f"{t:>12}{st.median(vals):>9,.0f}{st.mean(vals):>14,.0f}"
                     f"{max(vals):>14,}{sum(1 for v in vals if v > 1) / len(vals):>10.1%}")
    lines += ["", "A route with 1 ordering had no choice; the rest is the space this whole "
                  "question lives in."]
    return "\n".join(lines)


def constraint_budget(genomes) -> str:
    """How much ordering each tier actually imposes — the context a flat ladder needs.

    A sensitivity ladder that barely moves can mean two very different things: that the extra
    chemistry is genuinely irrelevant, or that it was never present in the data to begin with.
    This table separates them by showing what each tier *added*, so a tier that turns out to be
    inert is visible as inert rather than quietly reported as agreement.
    """
    n = len(genomes)
    lines = ["constraint budget (what each necessity tier imposes)", "",
             f"{'tier':>12}{'median edges':>14}{'mean edges':>12}{'routes affected':>17}",
             "-" * 55]
    prev = None
    for t in TIER_NAMES:
        counts = [len(g.constraints.get(t, ())) for g in genomes]
        if not counts:
            continue
        added = 0 if prev is None else sum(
            1 for g in genomes if len(g.constraints.get(t, ())) > len(g.constraints.get(prev, ())))
        label = "-" if prev is None else f"{added:,} ({added / n:.2%})"
        lines.append(f"{t:>12}{st.median(counts):>14,.0f}{st.mean(counts):>12,.2f}{label:>17}")
        prev = t
    lines += ["", "'routes affected' counts routes where this tier added a constraint the "
                  "previous tier lacked.",
              "A tier at 0% adds nothing on this corpus, and its agreement with the tier below",
              "is not evidence of robustness — it is the same test run twice."]
    return "\n".join(lines)


def separation_table(genomes) -> str:
    """How far apart are the transformations being compared?

    Makes the scope of the analysis explicit, because "correlation between steps" is easy to
    read as "correlation between *neighbouring* steps".  The precedence statistic imposes no
    distance limit at all — every pair of transformations in a route is compared, and the
    constrained null covers a constraint acting across six intervening steps exactly as it
    covers an adjacent one.  The co-location statistic is the local one, bounded by its window.
    """
    import itertools

    gaps: List[int] = []
    for g in genomes:
        at = [i for i, f in enumerate(g.families) if f != "?"]
        gaps.extend(abs(b - a) for a, b in itertools.combinations(at, 2))
    if not gaps:
        return "separation: no comparable pairs"
    n = len(gaps)
    adj = sum(1 for x in gaps if x == 1)
    far = sum(1 for x in gaps if x >= 3)
    return "\n".join([
        "separation between the transformations compared (ordering analysis)", "",
        f"  {n:,} pairs | median gap {st.median(gaps):.0f} | mean {st.mean(gaps):.2f} | "
        f"max {max(gaps)}",
        f"  adjacent (gap 1): {adj / n:.1%}   |   non-adjacent: {1 - adj / n:.1%}   |   "
        f"3+ steps apart: {far / n:.1%}",
        "",
        "The ordering statistic applies no distance limit — these are simply the separations",
        "that occur. The co-location statistic is the local one, bounded by its δ-window.",
    ])


def cluster_list(verdicts: Sequence[dict], tier: str, verdict_kind: str,
                 envs: Dict[str, str], top: int = 20) -> str:
    rows = [v for v in verdicts if v["verdict"].get(tier) == verdict_kind]
    rows.sort(key=lambda v: (v["q_constrained"].get(tier, 1.0), -v["quorum"]))
    head = (f"top {min(top, len(rows))} of {len(rows)} {verdict_kind} blocks at tier "
            f"'{tier}'")
    out = [head, ""]
    for v in rows[:top]:
        fams = "  +  ".join(_render(f, envs) for f in v["families"])
        out.append(f"  {fams}")
        out.append(f"      tight in {v['observed']}/{v['routes_tested']} routes "
                   f"(null expects {v['mean_null_constrained'].get(tier, float('nan')):.1f}) | "
                   f"q_free={v['q_free']:.2e} | "
                   f"q_{tier}={v['q_constrained'].get(tier, float('nan')):.2e}")
    if not rows:
        out.append("  (none)")
    return "\n".join(out)


def pair_list(pairs: Sequence[dict], tier: str, verdict_kind: str, envs: Dict[str, str],
              top: int = 20) -> str:
    """Ordered family pairs at one verdict — the precedence analysis's readable output."""
    rows = [p for p in pairs if p["verdict"].get(tier) == verdict_kind]
    rows.sort(key=lambda p: (p["q_constrained"].get(tier, 1.0), -p["n_routes"]))
    out = [f"top {min(top, len(rows))} of {len(rows)} {verdict_kind} orderings at tier "
           f"'{tier}'", ""]
    for p in rows[:top]:
        # the pair is stored in canonical alphabetical order; display it in the direction the
        # literature actually prefers, and flip the null expectation to match — quoting the
        # observed share in one orientation and the expectation in the other reads as a
        # contradiction ("80% observed, null expects 32%") when in fact they agree
        flip = p["observed"] * 2 < p["n_routes"]
        first, second = ((p["family_b"], p["family_a"]) if flip
                         else (p["family_a"], p["family_b"]))
        n = p["n_routes"]
        shown = n - p["observed"] if flip else p["observed"]
        exp = p["expected_constrained"].get(tier, float("nan"))
        exp_shown = n - exp if flip else exp
        out.append(f"  {_render(first, envs)}")
        out.append(f"    BEFORE {_render(second, envs)}")
        out.append(f"      {shown}/{n} routes ({shown / n:.0%}) | "
                   f"constrained null expects {exp_shown:.1f} ({exp_shown / n:.0%}) | "
                   f"explains {p.get('explained', {}).get(tier, float('nan')):.0%} | "
                   f"q_free={p['q_free']:.1e} q_{tier}={p['q_constrained'].get(tier, 1.0):.1e}")
    if not rows:
        out.append("  (none)")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decomposition", default=None, help="cluster decomposition JSON")
    ap.add_argument("--precedence", default=None,
                    help="precedence JSON — the primary ordering analysis")
    ap.add_argument("--genomes", default=None)
    ap.add_argument("--tier", default="exposure", choices=TIER_NAMES)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)

    if not args.decomposition and not args.precedence:
        ap.error("give --precedence and/or --decomposition")
    envs: Dict[str, str] = {}

    if args.genomes:
        genomes = load_genomes(args.genomes)
        print(f"corpus: {len(genomes):,} linear routes")
        print()
        print(constraint_budget(genomes))
        print()
        print(freedom_table(genomes))
        print()
        print(separation_table(genomes))
        print()

    if args.precedence:
        with open(args.precedence) as fh:
            pdata = json.load(fh)
        print("=" * 72)
        print("ORDERING  —  is a transformation pair's order forced, or chosen?")
        print("=" * 72)
        print(prec_headline(pdata["headline"]))
        print()
        print(pair_list(pdata["pairs"], args.tier, CONVENTION, envs, args.top))
        print()
        print("^ THIS LIST IS THE FALSIFICATION TEST. If a chemist can name the mechanism for")
        print("  these orderings, the constraint model is missing chemistry and the convention")
        print("  bound is loose by that much. Read it before quoting the number above.")
        print()
        print(pair_list(pdata["pairs"], args.tier, NECESSITY, envs, args.top))
        print()

    if args.decomposition:
        with open(args.decomposition) as fh:
            data = json.load(fh)
        h, verdicts = data["headline"], data["verdicts"]
        print("=" * 72)
        print("CO-LOCATION  —  which transformations stay together (the gene-cluster analogue)")
        print("=" * 72)
        print(f"params: {h.get('params')}")
        print()
        print(format_headline(h))
        print()
        print(cluster_list(verdicts, args.tier, CONVENTION, envs, args.top))
        dropped = h.get("clusters_dropped_by_cap") or 0
        capped = sum(1 for v in verdicts if v.get("routes_capped"))
        if dropped or capped:
            print()
            print(f"COVERAGE CAPS: {dropped:,} clusters untested (--max-clusters); "
                  f"{capped:,} clusters had their route set truncated (--max-routes). "
                  f"These results describe a bounded search, not the whole corpus.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
