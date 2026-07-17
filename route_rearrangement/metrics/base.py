"""Shared representations every metric consumes.

A scored route is read from a ``routes.jsonl`` record (see :mod:`..schema`).  Two views
are derived:

* :func:`reactions` — the per-step ``(reactants, main_reactant, product)`` list, in
  synthesis order (position 1 first), all canonical map-free SMILES;
* :func:`retro_tree` — the nested ``{"smiles", "child"}`` retro tree the Tree-LSTM ranker
  expects (root = target, children = precursors, building blocks = leaves).

For a linear route the chain precursor at position *k* is the product of position *k-1*,
so the tree is a single spine with building blocks hanging off each node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from rdkit import Chem


@dataclass
class Reaction:
    position: int
    product: str
    reactants: List[str]
    main_reactant: str          # largest reactant by heavy-atom count (the substrate)
    chain_precursor: Optional[str]
    side_reactants: List[str]


def _heavy(smi: str) -> int:
    m = Chem.MolFromSmiles(smi)
    return m.GetNumHeavyAtoms() if m is not None else 0


def reactions(record: dict) -> List[Reaction]:
    """Per-step reactions of a materialized route, position 1 (first-performed) first."""
    out: List[Reaction] = []
    for s in sorted(record["steps"], key=lambda s: s["position"]):
        reactants = list(s["side_reactants"])
        if s.get("chain_precursor"):
            reactants = reactants + [s["chain_precursor"]]
        main = max(reactants, key=_heavy) if reactants else s["new_product"]
        out.append(Reaction(
            position=s["position"], product=s["new_product"], reactants=reactants,
            main_reactant=main, chain_precursor=s.get("chain_precursor"),
            side_reactants=list(s["side_reactants"]),
        ))
    return out


def intermediates(record: dict) -> List[str]:
    """The growing-scaffold products at each step (position order) — the target last."""
    return [r.product for r in reactions(record)]


def retro_tree(record: dict) -> Optional[dict]:
    """Nested ``{"smiles", "child"}`` retro tree for the Tree-LSTM ranker, or ``None``.

    Built from the top step (forms the target) downward: each reaction node's children
    are its precursors; the chain precursor recurses into the step that made it, every
    other precursor is a purchasable leaf.
    """
    rxns = reactions(record)
    if not rxns:
        return None
    by_product: Dict[str, Reaction] = {r.product: r for r in rxns}

    def node(smiles: str, seen: frozenset) -> dict:
        r = by_product.get(smiles)
        # treat as a leaf if not a product, or if expanding it would revisit a molecule
        # already on this path (a product that recurs — e.g. a redox returning to an
        # earlier structure — would otherwise recurse forever)
        if r is None or smiles in seen:
            return {"smiles": smiles, "child": []}
        seen = seen | {smiles}
        return {"smiles": smiles, "child": [node(p, seen) for p in r.reactants]}

    top = max(rxns, key=lambda r: r.position)
    return node(top.product, frozenset())
