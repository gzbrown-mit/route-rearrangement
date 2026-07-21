"""Check a finished run against the chemical-ordering motifs — a *post-hoc* audit.

This is deliberately **not** part of the rearrangement pipeline.  The pipeline is a
neutral generator: it enumerates orderings, materializes them and scores them, and it
must not quietly drop routes on the strength of a heuristic chemistry rule.  Keeping
the audit separate means the checks can be retuned and re-run over an existing corpus
without regenerating a single route, and a bug in a check costs a wrong report rather
than a lost result.

Two things are reported:

* **the audit** — every Tier 1 finding (see :mod:`.feasibility`), aggregated by check
  and by the motif it enforces (see :mod:`.motifs`);
* **the control** — the same checks applied to the *literature* ordering of each route.
  Those syntheses were published and worked, so an ``infeasible`` finding there is
  never a fact about the route.  It is either a bug in the rule, or a record the
  engine mis-materialized — the control splits the two by cross-referencing
  ``sm_mismatch``.  In practice the second case dominates: an unactivated-SNAr verdict
  on a literature ordering has so far always meant the retro template disconnected
  that bond backwards, making the electron-rich ring the electrophile.  So this column
  is both how the rules get calibrated and an independent detector of bad
  disconnections.

Usage::

    python -m route_rearrangement.audit --results results_n1/
    python -m route_rearrangement.audit --results results_n1/ --corpus .../n1/trees.jsonl
    python -m route_rearrangement.audit --results results_n1/ --check snar_activation --show 5

``--corpus`` is optional and enables the two checks that need the original route
(``template_self_consistency``, ``pg_bracket``).  ``--out`` writes one findings record
per audited route to JSONL for downstream filtering.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from . import deps  # noqa: F401
from .feasibility import audit_record, detect_brackets
from .motifs import BY_NAME, MOTIFS


def _load_corpus_context(corpus: str, tree_ids) -> Dict[str, tuple]:
    """``{tree_id: (templates, brackets)}`` for the routes present in the results."""
    from .templates import extract_step_templates
    from synthesis_extraction.load_trees import iter_trees
    from synthesis_extraction.dependency.route_graph import build_route_graph

    wanted = set(tree_ids)
    out: Dict[str, tuple] = {}
    for tid, tg in iter_trees(corpus):
        if tid not in wanted:
            continue
        try:
            full = build_route_graph(tg, tid)
            if full is None:
                continue
            out[tid] = (extract_step_templates(full), detect_brackets(full))
        except Exception:
            continue
        if len(out) == len(wanted):
            break
    return out


def _pct(a: int, b: int) -> str:
    return f"{a/b:.1%}" if b else "n/a"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True,
                    help="a pipeline --out-dir (reads scored.jsonl, else routes.jsonl)")
    ap.add_argument("--corpus", default="",
                    help="original trees.jsonl; enables the template-consistency and "
                         "protecting-group-bracket checks")
    ap.add_argument("--check", default="", help="show examples of this check only")
    ap.add_argument("--show", type=int, default=0, help="print N example findings")
    ap.add_argument("--out", default="", help="write per-route findings JSONL here")
    args = ap.parse_args(argv)

    res = Path(args.results)
    src = res / "scored.jsonl"
    if not src.exists():
        src = res / "routes.jsonl"
    if not src.exists():
        print(f"no scored.jsonl or routes.jsonl in {res}", file=sys.stderr)
        return 1

    records = [json.loads(line) for line in src.open()]
    if not records:
        print("no records to audit", file=sys.stderr)
        return 1

    context: Dict[str, tuple] = {}
    if args.corpus:
        print("loading corpus context (templates + protecting-group brackets)...",
              file=sys.stderr)
        context = _load_corpus_context(args.corpus, {r["tree_id"] for r in records})

    lit_fired: Counter = Counter()
    lit_infeasible: Counter = Counter()
    lit_infeasible_by_quality: Counter = Counter()
    re_fired: Counter = Counter()
    re_infeasible: Counter = Counter()
    motif_hits: Counter = Counter()
    n_lit = n_re = 0
    clean_re = 0
    routes_touched: Dict[str, set] = defaultdict(set)
    examples: List[tuple] = []
    out_rows: List[dict] = []

    for rec in records:
        templates, brackets = context.get(rec["tree_id"], (None, ()))
        try:
            findings = audit_record(rec, templates, brackets)
        except Exception:
            continue
        checks = {f.check for f in findings}
        infeas = {f.check for f in findings if f.severity == "infeasible"}
        is_lit = bool(rec.get("is_original_order"))
        mis = bool(rec.get("flags", {}).get("sm_mismatch"))
        if is_lit:
            n_lit += 1
            for c in checks:
                lit_fired[c] += 1
            for c in infeas:
                lit_infeasible[c] += 1
            if infeas:
                lit_infeasible_by_quality["mis-materialized" if mis else "clean"] += 1
        else:
            n_re += 1
            if not findings:
                clean_re += 1
            for c in checks:
                re_fired[c] += 1
                routes_touched[c].add(rec["tree_id"])
            for c in infeas:
                re_infeasible[c] += 1
            for f in findings:
                if f.motif:
                    motif_hits[f.motif] += 1
                if args.check and f.check == args.check and len(examples) < args.show:
                    examples.append((rec["tree_id"], rec["ordering"], f))
        if args.out:
            out_rows.append({
                "tree_id": rec["tree_id"], "ordering": rec["ordering"],
                "is_original_order": is_lit,
                "n_infeasible": sum(1 for f in findings if f.severity == "infeasible"),
                "n_risk": sum(1 for f in findings if f.severity == "risk"),
                "findings": [asdict(f) for f in findings],
            })

    print(f"\n=== audit of {src} ===")
    print(f"{len(records)} records: {n_lit} literature orderings, {n_re} rearrangements")
    if not args.corpus:
        print("(no --corpus: template_self_consistency and pg_bracket were skipped)")
    print(f"\nrearrangements with NO finding: {clean_re} ({_pct(clean_re, n_re)})")

    print(f"\n{'check':<30} | {'rearrangements':>16} {'infeasible':>10} | "
          f"{'literature (control)':>20}")
    print("-" * 84)
    for check in sorted(set(re_fired) | set(lit_fired)):
        lit = f"{lit_fired[check]} ({_pct(lit_fired[check], n_lit)})"
        if lit_infeasible[check]:
            lit += f"  !! {lit_infeasible[check]} INFEASIBLE"
        rearr = f"{re_fired[check]} ({_pct(re_fired[check], n_re)})"
        print(f"{check:<30} | {rearr:>16} {re_infeasible[check]:>10} | {lit:>20}")

    clean_fp = lit_infeasible_by_quality["clean"]
    mis_fp = lit_infeasible_by_quality["mis-materialized"]
    print("\ncontrol — an 'infeasible' verdict on a published route means one of two "
          "things:")
    if mis_fp:
        print(f"  {mis_fp} on records already flagged sm_mismatch — the retro template "
              f"disconnected\n     that bond the wrong way round (electrophile and "
              f"nucleophile swapped), so the\n     record is not faithful literature "
              f"chemistry. The check corroborates sm_mismatch.")
    if clean_fp:
        print(f"  {clean_fp} on CLEAN records — genuine false positives. A rule is "
              f"rejecting chemistry\n     that demonstrably worked and must be fixed.")
    if not (mis_fp or clean_fp):
        print("  none — no check rejects any published route.")

    print(f"\n{'motif':<32}{'flagged rearrangements':>24}")
    for m in MOTIFS:
        if motif_hits.get(m.name):
            print(f"{m.name:<32}{motif_hits[m.name]:>24}")
    silent = [m.name for m in MOTIFS if m.check and not motif_hits.get(m.name)]
    if silent:
        print(f"\nenforced but never fired here: {', '.join(silent)}")
    unenforced = [m.name for m in MOTIFS if not m.check]
    if unenforced:
        print(f"documented, not yet enforced: {', '.join(unenforced)}")

    if examples:
        print(f"\nexamples of {args.check}:")
        for tid, ordering, f in examples:
            print(f"  {tid} {ordering} step {f.step_id} [{f.severity}]")
            print(f"    {f.detail}")
            motif = BY_NAME.get(f.motif)
            if motif:
                print(f"    motif: {motif.name} — {motif.rule}")

    if args.out:
        with open(args.out, "w") as fh:
            for row in out_rows:
                fh.write(json.dumps(row) + "\n")
        print(f"\nwrote {len(out_rows)} audited routes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
