"""Per-step retro template extraction and application.

Each step of a unified-map ``full_graph`` gets a :class:`StepTemplate`: its rdchiral
retro SMARTS plus the map-free canonical roles of its molecules — the product, the
*chain precursor* (the reactant that is the previous step's product, i.e. the growing
scaffold) and the *side reactants* (the building blocks the step installs).  Side
reactants are order-invariant: the same building block gets installed no matter when the
step runs, so after re-applying the template to a rearranged intermediate they reappear
exactly and identify the new chain fragment by elimination (see :mod:`.chain`).

The rdchiral wrappers (``apply_retro``, ``canonicalize_smiles``, ``canonicalize_smarts``,
``canon_set``) are adapted from miniASKCOS (``askcos/modules/template_relevance.py::
_apply_template`` and ``scripts/model_pipeline/extract_templates.py``) rather than
imported: that repo pulls in torch/pebble at module level.  rdchiral's extracted SMARTS
are retro-specific — forward application is unreliable by design, so fidelity is judged
retro-only (``retro_identity_ok``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set

from rdkit import Chem, RDLogger
from rdchiral.template_extractor import extract_from_reaction
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun

from . import deps  # noqa: F401
from synthesis_extraction.step_classification.footprint import (  # noqa: E402
    main_product_mol,
    split_reaction,
)

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Vendored rdchiral/canonicalization helpers (adapted from miniASKCOS)
# ---------------------------------------------------------------------------
def canonicalize_smiles(smiles: str, remove_atom_number: bool = True) -> str:
    """Canonical isomeric SMILES, atom maps stripped; ``""`` on failure.

    Double-canonicalizes for stereochemistry edge cases (miniASKCOS convention).
    """
    smiles = "".join(smiles.split())
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    if remove_atom_number:
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
    cano = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    mol2 = Chem.MolFromSmiles(cano)
    if mol2 is not None:
        cano = Chem.MolToSmiles(mol2, isomericSmiles=True, canonical=True)
    return cano


def canonicalize_smarts(smarts: str) -> str:
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        return smarts
    canon = Chem.MolToSmarts(mol)
    if "[[se]]" in canon:
        return smarts
    return canon


def canon_set(smi: str) -> Set[str]:
    """Set of canonical map-free component SMILES of a ``.``-joined string."""
    out: Set[str] = set()
    for frag in (smi or "").split("."):
        if not frag:
            continue
        c = canonicalize_smiles(frag)
        if c:
            out.add(c)
    return out


@lru_cache(maxsize=4096)
def _cached_reaction(retro_smarts_one: str):
    return rdchiralReaction(retro_smarts_one)


@lru_cache(maxsize=100_000)
def apply_retro(retro_smarts: str, product_smi: str,
                max_outcomes: int = 20) -> tuple:
    """Run a retro template (``product_side>>reactant_side``) on a map-free product
    SMILES; return up to *max_outcomes* ``.``-joined precursor-set SMILES.

    Cached on ``(template, product)`` — orderings sharing a backward suffix reuse each
    other's rdchiral runs, in both the naive and the DFS engine.
    """
    retro_smarts_one = "(" + retro_smarts.replace(">>", ")>>(") + ")"
    try:
        rxn = _cached_reaction(retro_smarts_one)
        prod = rdchiralReactants(product_smi)
        outcomes = rdchiralRun(rxn, prod, return_mapped=False)
    except Exception:
        return ()
    if not outcomes:
        return ()
    # dedupe while keeping rdchiral's order
    seen: Set[str] = set()
    out: List[str] = []
    for o in outcomes:
        if o not in seen:
            seen.add(o)
            out.append(o)
        if len(out) >= max_outcomes:
            break
    return tuple(out)


# ---------------------------------------------------------------------------
# StepTemplate
# ---------------------------------------------------------------------------
@dataclass
class StepTemplate:
    """One route step's retro template plus the map-free roles of its molecules."""

    step_id: int                       # full_graph node id (root product = 1)
    rxn_index: int
    orig_rxn: str                      # unified-map reactants>reagents>products
    retro_smarts: Optional[str]        # "product_side>>reactant_side"; None if extraction failed
    orig_product: Optional[str]        # canonical map-free main product
    orig_chain_precursor: Optional[str]  # child step's main product among reactants; None for deepest step
    orig_side_reactants: List[str] = field(default_factory=list)  # canonical map-free (multiset)
    orig_reactants: List[str] = field(default_factory=list)       # every reactant fragment, canonical map-free
    retro_identity_ok: bool = False    # retro template reproduces original reactants on original product
    extract_error: str = ""


def _extract_retro_smarts(rxn_smiles: str, step_id: int) -> Optional[str]:
    """rdchiral retro SMARTS for one atom-mapped step, or ``None``."""
    rb, pb = split_reaction(rxn_smiles)
    if not rb or not pb:
        return None
    try:
        template = extract_from_reaction({"_id": step_id, "reactants": rb, "products": pb})
    except Exception:
        return None
    if not template or "products" not in template or "reactants" not in template:
        return None
    p_canon = canonicalize_smarts(template["products"])
    r_canon = canonicalize_smarts(template["reactants"])
    return f"{p_canon}>>{r_canon}"


def _retro_identity_ok(retro_smarts: str, product_smi: str,
                       reactant_frags: List[str]) -> bool:
    """Does the retro template on the original product reproduce the original reactants?"""
    expected = set(reactant_frags)
    if not expected:
        return False
    for outcome in apply_retro(retro_smarts, product_smi, max_outcomes=50):
        if canon_set(outcome) == expected:
            return True
    return False


def extract_step_templates(full_graph: dict) -> Dict[int, StepTemplate]:
    """A :class:`StepTemplate` for every node of a unified-map ``full_graph``.

    The chain precursor of a step is the reactant equal (map-free) to its child's main
    product; a step with no child (the deepest step of a linear route) has none — all of
    its reactants are starting materials.
    """
    nodes = {int(n["id"]): n for n in full_graph.get("nodes", [])}
    children: Dict[int, List[int]] = {}
    for c, p in full_graph.get("edges", []):
        children.setdefault(int(p), []).append(int(c))

    # map-free main product per node, for chain-precursor identification
    product_of: Dict[int, Optional[str]] = {}
    for nid, n in nodes.items():
        _, pb = split_reaction(n.get("SMILES", ""))
        pm = main_product_mol(pb) if pb else None
        product_of[nid] = canonicalize_smiles(Chem.MolToSmiles(pm)) if pm is not None else None

    out: Dict[int, StepTemplate] = {}
    for nid, n in nodes.items():
        rxn = n.get("SMILES", "")
        rb, pb = split_reaction(rxn)
        reactant_frags = [canonicalize_smiles(f) for f in (rb or "").split(".") if f]
        reactant_frags = [f for f in reactant_frags if f]

        chain = None
        kids = children.get(nid, [])
        if len(kids) == 1:
            child_prod = product_of.get(kids[0])
            if child_prod is not None and child_prod in reactant_frags:
                chain = child_prod
        side = list(reactant_frags)
        if chain is not None:
            side.remove(chain)  # one occurrence — side is a multiset

        retro = _extract_retro_smarts(rxn, nid)
        tpl = StepTemplate(
            step_id=nid,
            rxn_index=int(n.get("rxn_index", -1)),
            orig_rxn=rxn,
            retro_smarts=retro,
            orig_product=product_of.get(nid),
            orig_chain_precursor=chain,
            orig_side_reactants=side,
            orig_reactants=reactant_frags,
            extract_error="" if retro else "template_extraction_failed",
        )
        if retro and tpl.orig_product:
            tpl.retro_identity_ok = _retro_identity_ok(retro, tpl.orig_product, reactant_frags)
        out[nid] = tpl
    return out


def is_linear(full_graph: dict) -> bool:
    """True iff every node has at most one child (each step's non-chain reactants are
    leaves/purchasable) — the v1 topology gate."""
    n_children: Dict[int, int] = {}
    for c, p in full_graph.get("edges", []):
        n_children[int(p)] = n_children.get(int(p), 0) + 1
    return all(v <= 1 for v in n_children.values())
