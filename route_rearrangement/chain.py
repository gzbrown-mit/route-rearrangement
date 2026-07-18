"""Route one retro-application outcome's fragments: synthesized precursors vs SMs.

When a step's retro template is applied to a (possibly rearranged) intermediate, the
outcome is a ``.``-joined set of precursor fragments.  Each fragment is either a
**starting material** (a purchasable block, order-invariant: the same molecule is
installed no matter when the step runs) or a **synthesized precursor** (an open
intermediate that a later-undone step must disconnect — one for a linear step, two or
more for a convergent coupling, zero for a branch-tip step).

Routing is ambiguous in the corners (template context bleeding into a block; a
migrated seed occupying a slot a purchasable held in the original order), so
:func:`route_outcomes` returns a small set of *candidate* routings — the frontier
walk's beam search keeps the ones that stay chemically consistent:

* **peel** — the step's own original side reactants are matched exactly (multiset) and
  removed; the remainder are synthesized precursors.  The exact, unambiguous case.
* **budget absorb** — a remainder fragment that equals a still-unconsumed purchasable
  of the *whole route* is a starting material after all (a seed that shifted to this
  step because the step that originally consumed it now runs on the combined
  molecule — the migration case).
* **expected-k** — when the peel is inexact, keep the ``len(orig_synth_precursors)``
  fragments most similar (Morgan-Tanimoto) to the original synthesized precursors and
  treat the rest as SMs (the legacy similarity fallback, generalized from k=1).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .templates import StepTemplate, canonicalize_smiles

_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


@lru_cache(maxsize=8192)
def _fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    return _fpgen.GetFingerprint(m) if m is not None else None


def tanimoto(smi_a: Optional[str], smi_b: Optional[str]) -> float:
    if not smi_a or not smi_b:
        return 0.0
    fa, fb = _fp(smi_a), _fp(smi_b)
    if fa is None or fb is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fa, fb)


@dataclass
class RoutedOutcome:
    """One candidate routing of an outcome's fragments."""

    synth: List[str] = field(default_factory=list)   # stay open on the frontier
    sm: List[str] = field(default_factory=list)      # starting materials (leave)
    exact_side_match: bool = True
    sim_score: float = 1.0                # mean Tanimoto(synth, orig synth precursors)


def _heavy_atoms(smi: str) -> int:
    m = Chem.MolFromSmiles(smi)
    return m.GetNumHeavyAtoms() if m is not None else 0


def _expected_synth(tpl: StepTemplate) -> List[str]:
    """The step's original synthesized precursors (compat: fall back to the single
    chain precursor for templates built before the plural field existed)."""
    if tpl.orig_synth_precursors:
        return list(tpl.orig_synth_precursors)
    return [tpl.orig_chain_precursor] if tpl.orig_chain_precursor else []


def _greedy_sim(synth: Sequence[str], refs: Sequence[str], fallback_ref: str) -> float:
    """Mean Tanimoto of a greedy synth↔reference assignment (k=1 reduces to the
    legacy chain-vs-orig-chain similarity)."""
    if not synth:
        return 1.0
    pool = list(refs)
    total = 0.0
    for f in synth:
        if pool:
            best = max(pool, key=lambda r: tanimoto(f, r))
            total += tanimoto(f, best)
            pool.remove(best)
        else:
            total += tanimoto(f, fallback_ref)
    return total / len(synth)


def route_outcomes(outcome: str, tpl: StepTemplate, *, last_step: bool = False,
                   sm_budget: Optional[Counter] = None) -> List[RoutedOutcome]:
    """Candidate routings of a ``.``-joined retro outcome (deduped, exact first).

    *last_step* — this is the final step being undone: every fragment must be a
    starting material (the frontier ends empty).  Returns ``[]`` when a fragment does
    not canonicalize (unusable outcome).
    """
    frags: List[str] = []
    for f in (outcome or "").split("."):
        if not f:
            continue
        c = canonicalize_smiles(f)
        if not c:
            return []
        frags.append(c)
    if not frags:
        return []

    if last_step:
        return [RoutedOutcome(synth=[], sm=frags, exact_side_match=True, sim_score=1.0)]

    expected = _expected_synth(tpl)
    ref_product = tpl.orig_product or ""

    # 1) peel the step's own side reactants (multiset, exact canonical match)
    remaining = list(frags)
    peeled: List[str] = []
    matched_all = True
    for s in tpl.orig_side_reactants:
        if s in remaining:
            remaining.remove(s)
            peeled.append(s)
        else:
            matched_all = False

    candidates: List[RoutedOutcome] = []

    def _add(synth: List[str], sm: List[str], exact: bool) -> None:
        key = (tuple(sorted(synth)), tuple(sorted(sm)))
        for c in candidates:
            if (tuple(sorted(c.synth)), tuple(sorted(c.sm))) == key:
                return
        candidates.append(RoutedOutcome(
            synth=synth, sm=sm, exact_side_match=exact,
            sim_score=_greedy_sim(synth, expected, ref_product)))

    # A) peel-based: the unmatched remainder stays open on the frontier
    _add(list(remaining), list(peeled),
         exact=matched_all and len(remaining) == len(expected))

    # B) budget absorb: remainder fragments equal to a still-unconsumed purchasable of
    #    the whole route are SMs (migrated seed).  Budget is corrected for this peel.
    if sm_budget is not None and remaining:
        local = sm_budget.copy()
        for s in peeled:
            if local.get(s, 0) > 0:
                local[s] -= 1
        absorbed, kept = [], []
        for f in remaining:
            if local.get(f, 0) > 0:
                local[f] -= 1
                absorbed.append(f)
            else:
                kept.append(f)
        if absorbed:
            _add(kept, peeled + absorbed, exact=False)

    # C) expected-k similarity fallback (legacy behavior generalized from k=1): keep
    #    the k fragments most similar to the original synth precursors, rest are SMs
    if not (matched_all and len(remaining) == len(expected)) and expected:
        k = min(len(expected), len(frags))
        ranked = sorted(frags,
                        key=lambda f: (_greedy_sim([f], expected, ref_product),
                                       _heavy_atoms(f)),
                        reverse=True)
        synth_c = ranked[:k]
        sm_c = list(frags)
        for f in synth_c:
            sm_c.remove(f)
        _add(synth_c, sm_c, exact=False)

    return candidates
