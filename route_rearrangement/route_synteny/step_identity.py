"""A transformation identity for **every** step, from the bonds the step actually changes.

FrequenTree contextual centres deliberately merge steps whose reaction centres share atoms, so
a 6.9-step route resolves into ~2.3 transformations and two thirds of its step pairs have no
identity to look up.  Measured on PaRoutes that is the binding limit on the rigidity map — not
corpus size, and not (as the radius sweep proved) any parameter of centre extraction, since
``radius`` is applied after the centres are already formed and leaves their count bit-identical.

This module supplies the alternative: key a step by *what it does to the molecular graph*.
Every step gets exactly one identity, so segmentation coverage goes to 100% by construction.

The identity is read straight off the atom-mapped reaction — for each pair of mapped atoms,
compare their bond order in the reactants with the products, and keep the ones that changed.
Nothing is extracted, matched or inferred: no SMARTS, no rdchiral, no template heuristics.
That makes it fast enough to run over the whole corpus inline (RDKit graph operations only),
deterministic, and stable across RDKit releases — the SMARTS-parsing drift that silently
dropped a third of FrequenTree templates on one RDKit version cannot happen here.

Three rungs, finest to coarsest, for the usual reason: a fine key is chemically faithful but
never recurs, a coarse one recurs but pools unrelated chemistry.

===============  ==========================================================================
``centre_env``   changed bonds + the reacting atoms + one shell of neighbours
``centre``       changed bonds + the reacting atoms
``bond_changes`` the changed bonds alone — a reaction-class fingerprint
===============  ==========================================================================

An amide coupling reads at ``bond_changes`` as "a C-N bond formed and a C-O bond broken",
which is both recognisable and common enough to accumulate statistics.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

#: Finest to coarsest — the same ordering convention as the FrequenTree ladder.
RUNGS: Tuple[str, ...] = ("centre_env", "centre", "bond_changes")


def _atom_token(atom: Chem.Atom) -> str:
    """Element, aromaticity and ring membership — deliberately not degree or H count.

    Degree and hydrogen count shift with tautomers, explicit-H handling and how a depositor
    drew the molecule, none of which is a difference in the transformation.
    """
    sym = atom.GetSymbol() or "*"
    if atom.GetIsAromatic():
        sym = sym.lower()
    return sym + ("R" if atom.IsInRing() else "")


def _bond_order(bond: Optional[Chem.Bond]) -> float:
    if bond is None:
        return 0.0
    return 1.5 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble()


def _mapped_atoms(mol: Chem.Mol) -> Dict[int, Chem.Atom]:
    return {a.GetAtomMapNum(): a for a in mol.GetAtoms() if a.GetAtomMapNum()}


def _side(smiles: str) -> Optional[Chem.Mol]:
    """Parse one side of a reaction, sanitizing leniently.

    Route SMILES routinely carry fragments RDKit will not fully sanitize; a step that cannot be
    parsed yields no identity rather than a wrong one.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                         ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        return None
    return mol


def _bond_map(mol: Chem.Mol) -> Dict[Tuple[int, int], float]:
    """``{(mapnum_lo, mapnum_hi): order}`` for bonds between mapped atoms."""
    out: Dict[Tuple[int, int], float] = {}
    for b in mol.GetBonds():
        m1 = b.GetBeginAtom().GetAtomMapNum()
        m2 = b.GetEndAtom().GetAtomMapNum()
        if m1 and m2:
            out[(min(m1, m2), max(m1, m2))] = _bond_order(b)
    return out


def changed_bonds(rxn_smiles: str) -> Optional[Tuple[List[Tuple[int, int, float, float]],
                                                    Chem.Mol, Chem.Mol]]:
    """Bonds whose order differs between reactants and products.

    Returns ``(changes, reactant_mol, product_mol)`` where each change is
    ``(mapnum_lo, mapnum_hi, order_before, order_after)``, or ``None`` if the step is unusable.
    """
    parts = rxn_smiles.split(">")
    if len(parts) < 2:
        return None
    r_mol, p_mol = _side(parts[0]), _side(parts[-1])
    if r_mol is None or p_mol is None:
        return None
    rb, pb = _bond_map(r_mol), _bond_map(p_mol)
    # A bond counts as changed when at least one endpoint survives into the products.  Requiring
    # *both* endpoints to survive looks safer but discards exactly the informative half of most
    # reactions: an amide coupling would register only "C-N formed" and lose "C-O broken",
    # collapsing it onto N-alkylation and reductive amination, and a nitro reduction — whose
    # every changed bond runs to an oxygen that leaves — would register nothing at all.
    # Anchoring on a surviving endpoint keeps departures while still refusing to invent changes
    # among atoms the mapping never tracked.
    shared = set(_mapped_atoms(r_mol)) & set(_mapped_atoms(p_mol))
    changes = []
    for key in set(rb) | set(pb):
        if key[0] not in shared and key[1] not in shared:
            continue
        before, after = rb.get(key, 0.0), pb.get(key, 0.0)
        if before != after:
            changes.append((key[0], key[1], before, after))
    return changes, r_mol, p_mol


def _fmt(order: float) -> str:
    return "a" if order == 1.5 else str(int(order)) if order == int(order) else str(order)


def step_keys(rxn_smiles: str) -> Dict[str, Optional[str]]:
    """``{rung: key}`` for one atom-mapped step; values are ``None`` where underivable.

    A step with no detectable bond change — a purely stereochemical or salt-form step, or one
    whose mapping is too incomplete to compare — gets ``None`` at every rung rather than an
    empty key, so that such steps are excluded rather than silently pooled into one giant
    pseudo-transformation.
    """
    empty: Dict[str, Optional[str]] = {r: None for r in RUNGS}
    got = changed_bonds(rxn_smiles)
    if got is None:
        return empty
    changes, r_mol, p_mol = got
    if not changes:
        return empty

    r_atoms = _mapped_atoms(r_mol)
    centre = sorted({m for c in changes for m in c[:2]})

    bond_toks = sorted(
        "".join(sorted((_atom_token(r_atoms[m1]), _atom_token(r_atoms[m2]))))
        + f"{_fmt(b)}>{_fmt(a)}"
        for m1, m2, b, a in changes if m1 in r_atoms and m2 in r_atoms)
    if not bond_toks:
        return empty
    k_bonds = "|".join(bond_toks)

    centre_toks = sorted(_atom_token(r_atoms[m]) for m in centre if m in r_atoms)
    k_centre = k_bonds + "//" + ".".join(centre_toks)

    shells = []
    for m in centre:
        atom = r_atoms.get(m)
        if atom is None:
            continue
        nbrs = sorted(_atom_token(n) for n in atom.GetNeighbors())
        shells.append(_atom_token(atom) + "(" + ".".join(nbrs) + ")")
    k_env = k_bonds + "//" + ".".join(sorted(shells))

    return {"centre_env": k_env, "centre": k_centre, "bond_changes": k_bonds}


@lru_cache(maxsize=200_000)
def step_key(rxn_smiles: str, rung: str = "centre") -> Optional[str]:
    """One rung's key, cached — the corpus repeats identical steps constantly."""
    return step_keys(rxn_smiles).get(rung)


def families_for_steps(nodes: Sequence[dict], rung: str = "centre") -> Dict[int, Optional[str]]:
    """``{step_id: key}`` for every node of a ``full_graph``.

    Every step is its own transformation event, which is the entire point: there is nothing to
    anchor or de-duplicate, so every step that changes a bond is available to the statistics.
    """
    return {int(n["id"]): step_key(str(n.get("SMILES", "")), rung) for n in nodes}
