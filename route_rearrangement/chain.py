"""Split one retro-application outcome into the chain intermediate vs side reactants.

When a step's retro template is applied to a (possibly rearranged) intermediate, the
outcome is a ``.``-joined set of precursor fragments.  The step's building blocks are
order-invariant — the same side reactants get installed no matter when the step runs —
so they are identified by exact canonical match against the original step's side
reactants; the fragment left over is the new chain intermediate (the substrate as it
exists at that point of the rearranged route).  When exact matching fails (template
context bled into a side reactant, or symmetric ambiguity) the chain is the fragment
most similar to the original chain precursor by Morgan-fingerprint Tanimoto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

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
class ChainSplit:
    """One outcome split into roles.  ``chain`` is ``None`` for a terminal step (all
    fragments are starting materials)."""

    chain: Optional[str]
    side: List[str] = field(default_factory=list)
    exact_side_match: bool = True
    sim_score: float = 1.0                # Tanimoto(chain, original chain precursor)
    heavy_atoms: int = 0                  # of the chain fragment (tie-break)


def _heavy_atoms(smi: str) -> int:
    m = Chem.MolFromSmiles(smi)
    return m.GetNumHeavyAtoms() if m is not None else 0


def split_outcome(outcome: str, tpl: StepTemplate, terminal: bool = False) -> Optional[ChainSplit]:
    """Split a ``.``-joined retro outcome into chain + side fragments.

    *terminal* — this is the first-performed step of the ordering being undone last:
    every fragment is a starting material, no chain is carried further.  Returns ``None``
    when a fragment does not canonicalize (unusable outcome).
    """
    frags: List[str] = []
    for f in (outcome or "").split("."):
        if not f:
            continue
        c = canonicalize_smiles(f)
        if not c:
            return None
        frags.append(c)
    if not frags:
        return None

    if terminal:
        return ChainSplit(chain=None, side=frags, exact_side_match=True, sim_score=1.0)

    # 1) peel off the original side reactants (multiset, exact canonical match)
    remaining = list(frags)
    matched_all = True
    for s in tpl.orig_side_reactants:
        if s in remaining:
            remaining.remove(s)
        else:
            matched_all = False
    if matched_all and len(remaining) == 1:
        chain = remaining[0]
        side = list(frags)
        side.remove(chain)
        return ChainSplit(
            chain=chain,
            side=side,
            exact_side_match=True,
            sim_score=tanimoto(chain, tpl.orig_chain_precursor),
            heavy_atoms=_heavy_atoms(chain),
        )

    # 2) fallback: most-similar-to-original-chain fragment (tie-break: heavier)
    ref = tpl.orig_chain_precursor or tpl.orig_product
    best, best_key = None, (-1.0, -1)
    for f in frags:
        key = (tanimoto(f, ref), _heavy_atoms(f))
        if key > best_key:
            best, best_key = f, key
    side = list(frags)
    side.remove(best)
    return ChainSplit(
        chain=best,
        side=side,
        exact_side_match=False,
        sim_score=best_key[0],
        heavy_atoms=best_key[1],
    )
