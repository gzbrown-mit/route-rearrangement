"""Mine a corpus for routes exhibiting each ordering-dependent motif, and report how
the engine performs on them.

The literature route always satisfies its own motifs, which makes them ground truth:
for every mined instance we materialize the rearrangements and ask two questions —

* **false positives** — does a Tier 1 check fire on the *literature* ordering?  It
  should not; every hit is a bug in the check or an over-strict rule.
* **catches** — do the *rearranged* orderings that violate the motif get flagged?
  These are the orderings that are atom-valid but chemically dead.

Usage::

    python -m route_rearrangement.find_motifs --corpus .../n1/trees.jsonl \\
        --limit 400 [--motif snar_requires_activation] [--out motif_cases.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List

from rdkit import Chem, RDLogger

from . import deps  # noqa: F401
from .feasibility import _mol, _patt, audit_route, detect_brackets
from .motifs import MOTIFS
from .run import process_route
from .schema import route_from_record
from .templates import extract_step_templates
from synthesis_extraction.load_trees import iter_trees
from synthesis_extraction.dependency.route_graph import build_route_graph

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Structural miners: does this ORIGINAL route exhibit the motif at all?
# ---------------------------------------------------------------------------
def _rxn_parts(node_smiles: str):
    left = node_smiles.split(">")[0]
    right = node_smiles.split(">")[-1]
    return [f for f in left.split(".") if f], [f for f in right.split(".") if f]


def _strip(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def _has_p(smi: str, key: str) -> bool:
    m, p = _mol(smi), _patt(key)
    return bool(m is not None and p is not None and m.HasSubstructMatch(p))


def _route_mols(full: dict):
    """(reactants, products) per node, map-free."""
    out = []
    for n in full.get("nodes", []):
        r, p = _rxn_parts(n.get("SMILES", ""))
        out.append(([_strip(x) for x in r], [_strip(x) for x in p]))
    return out


def mine(full: dict) -> List[str]:
    """Names of the motifs this original route exhibits."""
    hits: List[str] = []
    mols = _route_mols(full)
    n_nitro_red = 0
    n_snar = 0
    n_coupling = 0
    n_reduction_multi = 0
    n_stereo_gain = 0
    n_coupling_multi = 0
    for reactants, products in mols:
        if not reactants or not products:
            continue
        prod = max(products, key=lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms()
                   if Chem.MolFromSmiles(s) else 0)
        joined = ".".join(reactants)
        # nitro -> aniline reduction
        if any(_has_p(r, "aryl_nitro") for r in reactants) and _has_p(prod, "aniline") \
                and not _has_p(prod, "aryl_nitro"):
            n_nitro_red += 1
        # SNAr-like: aryl halide consumed, aryl-heteroatom formed
        if any(_has_p(r, "aryl_halide") for r in reactants) and _has_p(prod, "aryl_hetero"):
            n_snar += 1
        # cross-coupling
        if (_has_p(joined, "boron") or _has_p(joined, "stannane")) \
                and _has_p(joined, "aryl_halide"):
            n_coupling += 1
            if sum(_has_p(r, "aryl_halide") for r in reactants) > 1:
                n_coupling_multi += 1
        # reduction with a second reducible group surviving
        for key in ("ketone", "aldehyde", "nitrile", "alkene"):
            if any(_has_p(r, key) for r in reactants) and not _has_p(prod, key) \
                    and _has_p(prod, "aryl_nitro"):
                n_reduction_multi += 1
                break
        try:
            pm = Chem.MolFromSmiles(prod)
            if pm is not None and Chem.FindMolChiralCenters(
                    pm, includeUnassigned=False, useLegacyImplementation=False):
                n_stereo_gain += 1
        except Exception:
            pass

    if n_snar:
        hits.append("snar_requires_activation")
    if n_snar and n_nitro_red:
        hits.append("nitro_snar_reduction")
    if n_coupling:
        hits.append("amine_free_before_pd")
    if n_coupling_multi:
        hits.append("halide_chemoselectivity")
    if n_reduction_multi:
        hits.append("reduction_chemoselectivity")
    if n_stereo_gain:
        hits.append("stereocontrol_support")
    if detect_brackets(full):
        hits.append("pg_bracket_intact")
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--motif", default="", help="only report this motif")
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--max-accepted", type=int, default=12)
    ap.add_argument("--max-routes", type=int, default=60,
                    help="materialize at most N mined routes")
    ap.add_argument("--out", default="", help="write per-route cases JSONL here")
    args = ap.parse_args(argv)

    exhibits: Counter = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)
    mined = []
    for i, (tid, tg) in enumerate(iter_trees(args.corpus)):
        if i >= args.limit:
            break
        try:
            full = build_route_graph(tg, tid)
        except Exception:
            continue
        if full is None or full["qc"]["disconnected"]:
            continue
        if not (3 <= full["qc"]["n_steps"] <= 10):
            continue
        hits = mine(full)
        if args.motif and args.motif not in hits:
            continue
        for h in hits:
            exhibits[h] += 1
            if len(examples[h]) < 5:
                examples[h].append(tid)
        if hits:
            mined.append((tid, tg, full, hits))

    print(f"scanned {min(i+1, args.limit)} trees; motif instances found:\n")
    print(f"{'motif':<32}{'routes':>8}   examples")
    for m in MOTIFS:
        if exhibits.get(m.name):
            print(f"{m.name:<32}{exhibits[m.name]:>8}   {', '.join(examples[m.name][:4])}")
    unmined = [m.name for m in MOTIFS if not exhibits.get(m.name)]
    if unmined:
        print(f"\nno instances mined (miner not implemented or absent from corpus): "
              f"{', '.join(unmined)}")

    # ---- behaviour on the mined routes
    print(f"\nmaterializing up to {args.max_routes} mined routes...\n")
    fp = Counter()          # check fired on the LITERATURE ordering
    caught = Counter()      # check fired on a rearrangement
    n_lit = n_rearr = 0
    cases = []
    for tid, tg, full, hits in mined[: args.max_routes]:
        try:
            summary, records, _f = process_route(
                tid, tg, full_graph=full, cap=args.cap,
                max_accepted=args.max_accepted, with_fg=False)
        except Exception:
            continue
        if summary["status"] != "ok":
            continue
        templates = extract_step_templates(full)
        brackets = detect_brackets(full)
        for rec in records:
            route = route_from_record(rec)
            findings = audit_route(route, templates, brackets=brackets)
            checks = {f.check for f in findings}
            infeas = {f.check for f in findings if f.severity == "infeasible"}
            if rec["is_original_order"]:
                n_lit += 1
                for c in checks:
                    fp[c] += 1
                for c in infeas:
                    fp[f"{c}::INFEASIBLE"] += 1
            else:
                n_rearr += 1
                for c in checks:
                    caught[c] += 1
                for c in infeas:
                    caught[f"{c}::INFEASIBLE"] += 1
        cases.append({"tree_id": tid, "motifs": hits,
                      "n_records": len(records)})

    print(f"literature orderings audited: {n_lit};  rearrangements audited: {n_rearr}\n")
    print(f"{'check':<32}{'on literature':>15}{'on rearrangements':>20}")
    allchecks = sorted(set(fp) | set(caught))
    for c in allchecks:
        lit = f"{fp[c]} ({fp[c]/max(n_lit,1):.0%})"
        re_ = f"{caught[c]} ({caught[c]/max(n_rearr,1):.0%})"
        print(f"{c:<32}{lit:>15}{re_:>20}")
    print("\nNOTE: a check firing on the LITERATURE ordering is a false positive — the\n"
          "published route worked. INFEASIBLE rows there must be zero.")

    if args.out:
        with open(args.out, "w") as fh:
            for c in cases:
                fh.write(json.dumps(c) + "\n")
        print(f"\nwrote {len(cases)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
