"""Readable output, and the audit that decides whether any of it can be trusted.

The numbers :mod:`.significance` produces are only worth having if they reproduce ordering
constraints we already know to be true.  :mod:`..motifs` catalogues those — pairs of steps
whose relative order is fixed by deterministic chemistry, not taste — and the literature route
always satisfies them.  So they are ground truth: the mined statistics **must** recover them in
the correct direction, and must not report a significant pair that contradicts one.  A table
that fails this is measuring something other than chemistry, no matter how small its q-values.

Only some motifs are expressible as an ordered pair of transformations, which is what this
table can represent at all.  ``halide_chemoselectivity``, for instance, is a statement about
*one* step's substrate, not about the order of two.  Those are listed as not-auditable rather
than quietly omitted, so the pass rate is not read as broader than it is.

**On the SMARTS in this module**: they label an already-extracted key for the audit — "does
this key mean nitro reduction?" — and play no part in defining transformation identity, which
comes entirely from FrequenTree.  Labelling reads the key's own template halves (reactant side
``>>`` product side), so direction is available; the coarse synthon rungs have no parseable
template and are audited by association to the template rungs instead.

Usage::

    python -m route_rearrangement.literature_precedent.report \\
        --significance results/order_significance.json [--top 25]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

from .. import deps  # noqa: F401
from . import ladder, significance as sig
from ..motifs import MOTIFS

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Audit labelling (reporting only — never transformation identity)
# ---------------------------------------------------------------------------
# Audit predicates are *graph* tests over the parsed template half, not SMARTS matches.
#
# Both halves of a template key are themselves SMARTS, and matching a SMARTS pattern against a
# query molecule is unreliable in ways that fail silently: ``[CX3]``/``[NX3;H1,H2]`` never
# match (a query atom has no computed degree or H-count) and can raise, and FrequenTree writes
# charge as ``[N;+1;...]``, which RDKit does not surface via ``GetFormalCharge`` on the parsed
# query — so ``[c][N+](=O)[O-]`` matches a hand-written template and silently misses every real
# one.  What *does* survive parsing is element symbol, aromaticity, degree and connectivity, so
# the predicates below use only those.  Nitro is therefore identified topologically (an N with
# two terminal oxygens) rather than by charge.
def _side_mol(smarts_side: str):
    """Parse one half of a template key, stripping FrequenTree's fragment-grouping parens.

    Reuses upstream's ``_strip_group_parens`` rather than a local guess: those parens group
    disconnected fragments (``(frag1.frag2).frag3``) and are illegal at SMARTS top level, and
    telling them apart from branch parens is fiddly enough that upstream already got it wrong
    once.
    """
    from synthesis_extraction.transformation.pattern_key import _strip_group_parens
    if not smarts_side:
        return None
    try:
        return Chem.MolFromSmarts(_strip_group_parens(smarts_side.strip()))
    except Exception:
        return None


def _nbrs(atom):
    return list(atom.GetNeighbors())


def _is_nitro_n(atom) -> bool:
    """N carrying two terminal oxygens — a nitro group, identified without charge."""
    if atom.GetSymbol() != "N":
        return False
    terminal_o = [n for n in _nbrs(atom) if n.GetSymbol() == "O" and n.GetDegree() == 1]
    return len(terminal_o) >= 2


def _in_reaction_centre(atom) -> bool:
    """Is this atom part of the transformation, rather than spectator context?

    A template's reactant side carries context atoms the product side never repeats — the
    product is written only around the centre.  So "group present on the left, absent on the
    right" is *not* evidence the group was consumed: it is the default for every spectator.
    Requiring the atom to be atom-mapped is what distinguishes a Boc **removal** from any
    reaction that merely happens on a Boc-protected substrate, which otherwise all label as
    deprotections (observed: 21 significant pairs mislabelled this way, e.g. a sulfonylation
    whose substrate carries an untouched Boc).
    """
    return atom.GetAtomMapNum() > 0


def _has(mol, key: str) -> bool:
    if mol is None:
        return False
    try:
        atoms = list(mol.GetAtoms())
        if key == "aryl_nitro":
            return any(_is_nitro_n(a) and _in_reaction_centre(a)
                       and any(n.GetIsAromatic() for n in _nbrs(a)) for a in atoms)
        if key == "aryl_n":
            return any(a.GetIsAromatic() and any(n.GetSymbol() == "N" and not _is_nitro_n(n)
                                                 for n in _nbrs(a)) for a in atoms)
        if key == "aryl_halide":
            return any(a.GetIsAromatic() and any(n.GetSymbol() in ("F", "Cl", "Br", "I")
                                                 for n in _nbrs(a)) for a in atoms)
        if key == "aryl_hetero":
            return any(a.GetIsAromatic() and any(n.GetSymbol() in ("N", "O", "S")
                                                 and not _is_nitro_n(n)
                                                 for n in _nbrs(a)) for a in atoms)
        if key == "boron":
            return any(a.GetSymbol() == "B" for a in atoms)
        if key == "carbamate":
            # N-C(=O)-O : an sp2 carbon bearing a terminal O and a two-connected O, bonded to N
            for a in atoms:
                if a.GetSymbol() != "C" or not _in_reaction_centre(a):
                    continue
                ns = _nbrs(a)
                if not any(n.GetSymbol() == "N" for n in ns):
                    continue
                if any(n.GetSymbol() == "O" and n.GetDegree() == 1 for n in ns) and \
                        any(n.GetSymbol() == "O" and n.GetDegree() >= 2 for n in ns):
                    return True
            return False
    except Exception:
        return False
    return False


def label_key(key: str) -> List[str]:
    """Audit labels for a template-rung key (``reactants>>products`` SMARTS).

    Each label is a *transformation*, read as a change between the two halves — appearance or
    disappearance of a group — never as the mere presence of one, so that direction is
    genuinely determined by the key rather than assumed.
    """
    if ">>" not in key:
        return []
    lhs, rhs = (_side_mol(s) for s in key.split(">>", 1))
    labels = []
    if _has(lhs, "aryl_nitro") and not _has(rhs, "aryl_nitro") and _has(rhs, "aryl_n"):
        labels.append("nitro_reduction")
    if _has(lhs, "aryl_halide") and _has(rhs, "aryl_hetero") and not _has(rhs, "aryl_halide"):
        labels.append("snar_like")
    if _has(lhs, "boron") and _has(lhs, "aryl_halide"):
        labels.append("cross_coupling")
    if _has(rhs, "carbamate") and not _has(lhs, "carbamate"):
        labels.append("protection_carbamate")
    if _has(lhs, "carbamate") and not _has(rhs, "carbamate"):
        labels.append("deprotection_carbamate")
    return labels


# motif -> (label that must come FIRST, label that must come SECOND)
AUDITABLE_MOTIFS: Dict[str, Tuple[str, str]] = {
    "nitro_snar_reduction": ("snar_like", "nitro_reduction"),
    "pg_bracket_intact": ("protection_carbamate", "deprotection_carbamate"),
    "amine_free_before_pd": ("cross_coupling", "deprotection_carbamate"),
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _short(key: str, width: int = 46) -> str:
    return key if len(key) <= width else key[: width - 1] + "…"


def env_index(data: dict) -> Dict[str, str]:
    """``{key: a real reacting-substructure SMILES}`` across all rungs.

    The coarse synthon keys are bags of atom tokens (``C[];N[];O[]>>…``) — correct as
    identifiers, unreadable as chemistry.  Every key carries a few cached ``synthon_env``
    SMILES of the substructures it actually covers, which is what a chemist can read.
    """
    out: Dict[str, str] = {}
    for res in data["rungs"].values():
        envs = res.get("env_samples") or []
        for i, k in enumerate(res["keys"]):
            if k not in out and i < len(envs) and envs[i]:
                out[k] = ", ".join(envs[i][:2])
    return out


def _render(key: str, envs: Dict[str, str], width: int = 46) -> str:
    env = envs.get(key)
    return f"{_short(key, width)}   [{_short(env, 40)}]" if env else _short(key, width)


def rung_table(data: dict) -> str:
    """Per rung: how much it keys, how much it can test, and what it uniquely rescues."""
    rungs = data["rungs"]
    resolved = data.get("resolved") or []
    fired = defaultdict(int)
    for r in resolved:
        if r["rung_fired"]:
            fired[r["rung_fired"]] += 1

    head = (f"{'rung':>20}{'keys':>9}{'tested':>9}{'signif':>8}{'med DEFF':>10}"
            f"{'resolved':>10}{'depth R2':>10}")
    lines = [head, "-" * len(head)]
    for name in ladder.RUNG_NAMES:
        res = rungs.get(name)
        if res is None:
            continue
        pairs = res["pairs"]
        s = sig.significant(pairs)
        deffs = sorted(p["design_effect"] for p in pairs if p.get("design_effect"))
        med = deffs[len(deffs) // 2] if deffs else None
        null = res.get("depth_null")
        med_s = f"{med:.1f}" if med else "-"
        r2_s = f"{null['pseudo_r2']:.3f}" if null else "-"
        lines.append(
            f"{name:>20}{res['counts'].get('n_keys_total', 0):>9,}"
            f"{res['counts']['n_pairs_tested']:>9,}{len(s):>8,}"
            f"{med_s:>10}{fired.get(name, 0):>10,}{r2_s:>10}")
    lines.append("")
    lines.append("signif   = q_cluster <= 0.05 AND the route-clustered CI excludes 0.5")
    lines.append("med DEFF = median design effect; how much the naive binomial overstates "
                 "precision at this rung")
    lines.append("resolved = coarse pairs whose backoff landed on this rung (finest rung with "
                 "enough support)")
    lines.append("depth R2 = McFadden pseudo-R^2 of the depth-only null; how much of this "
                 "rung's ordering is just position in the route")
    return "\n".join(lines)


def top_pairs(data: dict, rung: str, top: int = 25) -> str:
    res = data["rungs"].get(rung)
    if res is None:
        return f"(no rung {rung})"
    rows = sig.significant(res["pairs"])
    rows.sort(key=lambda r: -abs(r.get("excess_log2_odds") or r["log2_odds"]))
    keys = res["keys"]
    envs = env_index(data)
    out = [f"top {min(top, len(rows))} of {len(rows)} significant pairs at {rung}", ""]
    for r in rows[:top]:
        first, second = (r["a"], r["b"]) if r["p_first"] >= 0.5 else (r["b"], r["a"])
        frac = max(r["p_first"], 1 - r["p_first"])
        out.append(
            f"  {_render(keys[first], envs)}\n"
            f"    BEFORE {_render(keys[second], envs)}\n"
            f"    {frac:.1%} of {r['n_obs']} obs across {r['n_routes']} routes | "
            f"CI [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | q={r['q_cluster']:.2e} | "
            f"DEFF={r['design_effect']:.1f} | "
            f"excess={r['excess_log2_odds']:+.2f} log2-odds over depth")
    return "\n".join(out)


def cross_rung_divergence(data: dict, top: int = 15, q: float = 0.05) -> str:
    """Pairs whose verdict changes with abstraction — substrate-specific vs class-general.

    Both sides must be individually significant.  Without that filter the list fills with fine
    pairs sitting at p = 1.00 on a dozen observations, which is not a disagreement with the
    class-level statistic so much as an absence of evidence.
    """
    def _sig(row) -> bool:
        return bool(row.get("q_cluster") is not None and row["q_cluster"] <= q)

    rows = []
    for r in data.get("resolved") or []:
        per = r["per_rung"]
        fine = [per[n] for n in ladder.RUNG_NAMES[:4] if n in per and _sig(per[n])]
        coarse = [per[n] for n in ladder.RUNG_NAMES[4:] if n in per and _sig(per[n])]
        if not fine or not coarse:
            continue
        pf = sum(x["p_first"] for x in fine) / len(fine)
        pc = sum(x["p_first"] for x in coarse) / len(coarse)
        if abs(pf - pc) >= 0.25:
            rows.append((abs(pf - pc), pf, pc, r))
    rows.sort(key=lambda t: -t[0])
    envs = env_index(data)
    out = [f"{len(rows)} pairs whose ordering verdict changes with abstraction "
           f"(|Δp| >= 0.25 between fine and coarse rungs, both individually significant)", ""]
    for _, pf, pc, r in rows[:top]:
        verdict = ("locked at template resolution, free at class resolution — the constraint "
                   "is substrate-specific" if pf > pc else
                   "free at template resolution, locked at class resolution — the constraint "
                   "is general, the exact template is just sparse")
        out.append(f"  {_render(r['key_a'], envs, 40)}")
        out.append(f"  / {_render(r['key_b'], envs, 40)}")
        out.append(f"    fine p={pf:.2f}  coarse p={pc:.2f}  -> {verdict}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The acceptance test
# ---------------------------------------------------------------------------
def motif_audit(data: dict, *, q: float = 0.05) -> dict:
    """Do the mined statistics recover the known ordering constraints, and contradict none?"""
    by_motif: Dict[str, dict] = {}
    contradictions: List[dict] = []

    for motif_name, (first_label, second_label) in AUDITABLE_MOTIFS.items():
        hits: List[dict] = []
        wrong: List[dict] = []
        for rung in ladder.RUNG_NAMES:
            res = data["rungs"].get(rung)
            if res is None:
                continue
            keys = res["keys"]
            labels = [label_key(k) for k in keys]
            for r in res["pairs"]:
                la, lb = labels[r["a"]], labels[r["b"]]
                if first_label in la and second_label in lb:
                    expect_first = True          # key_a should come first
                elif first_label in lb and second_label in la:
                    expect_first = False
                else:
                    continue
                observed_first = r["p_first"] >= 0.5
                agrees = (observed_first == expect_first)
                rec = {"rung": rung, "n_obs": r["n_obs"], "n_routes": r["n_routes"],
                       "p_first": r["p_first"], "q_cluster": r.get("q_cluster"),
                       "agrees": agrees,
                       "significant": bool(r.get("q_cluster") is not None
                                           and r["q_cluster"] <= q
                                           and r.get("ci_lo") is not None
                                           and not (r["ci_lo"] <= 0.5 <= r["ci_hi"]))}
                (hits if agrees else wrong).append(rec)
                if rec["significant"] and not agrees:
                    contradictions.append(dict(rec, motif=motif_name,
                                               key_a=keys[r["a"]], key_b=keys[r["b"]]))
        recovered = [h for h in hits if h["significant"]]
        by_motif[motif_name] = {
            "n_pair_instances": len(hits) + len(wrong),
            "n_agreeing": len(hits),
            "n_recovered_significant": len(recovered),
            "finest_rung_recovered": min((h["rung"] for h in recovered),
                                         key=lambda n: ladder.RUNG_BY_NAME[n].index,
                                         default=None),
            "contradicting_significant": len([w for w in wrong if w["significant"]]),
        }

    not_auditable = [m.name for m in MOTIFS if m.name not in AUDITABLE_MOTIFS]
    return {"motifs": by_motif, "contradictions": contradictions,
            "not_pairwise_expressible": not_auditable}


def format_motif_audit(audit: dict) -> str:
    out = ["motif recovery — the acceptance test", ""]
    for name, r in audit["motifs"].items():
        status = ("RECOVERED" if r["n_recovered_significant"] else
                  ("present but not significant" if r["n_agreeing"] else "NOT FOUND"))
        out.append(f"  {name:<28} {status}")
        out.append(f"      {r['n_pair_instances']} labelled pair instance(s); "
                   f"{r['n_agreeing']} in the literature direction; "
                   f"{r['n_recovered_significant']} significant; "
                   f"finest rung = {r['finest_rung_recovered'] or '-'}")
        if r["contradicting_significant"]:
            out.append(f"      !! {r['contradicting_significant']} SIGNIFICANT pair(s) "
                       f"contradict this motif — inspect before trusting the table")
    out.append("")
    out.append(f"  not expressible as an ordered pair of transformations, so not audited here: "
               f"{', '.join(audit['not_pairwise_expressible'])}")
    if audit["contradictions"]:
        out.append("")
        out.append(f"  {len(audit['contradictions'])} contradiction(s) require manual review")
    return "\n".join(out)


def hard_rule_overlap(data: dict) -> str:
    """Pairs already enforced by :mod:`..feasibility` — a precedent score must exclude these
    or it double-counts what the hard checks already veto."""
    enforced = {"snar_like": "snar_activation", "deprotection_carbamate": "pg_bracket"}
    n = 0
    for rung in ladder.RUNG_NAMES:
        res = data["rungs"].get(rung)
        if res is None:
            continue
        labels = [label_key(k) for k in res["keys"]]
        for r in res["pairs"]:
            if set(labels[r["a"]] + labels[r["b"]]) & set(enforced):
                n += 1
    return (f"{n} tested pair(s) involve a transformation already covered by a "
            f"feasibility.py Tier 1 check ({', '.join(sorted(set(enforced.values())))}). "
            f"A precedent metric must exclude them so 'unprecedented' keeps meaning "
            f"'unusual but allowed'.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--significance", required=True)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--rung", default=None, help="rung for the top-pairs listing")
    args = ap.parse_args(argv)

    with open(args.significance) as fh:
        data = json.load(fh)

    print(rung_table(data))
    print()
    rung = args.rung or _busiest_rung(data)
    print(top_pairs(data, rung, args.top))
    print()
    print(cross_rung_divergence(data))
    print()
    audit = motif_audit(data)
    print(format_motif_audit(audit))
    print()
    print(hard_rule_overlap(data))
    return 0


def _busiest_rung(data: dict) -> str:
    best, best_n = ladder.RUNG_NAMES[-1], -1
    for name, res in data["rungs"].items():
        n = len(sig.significant(res["pairs"]))
        if n > best_n:
            best, best_n = name, n
    return best


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
