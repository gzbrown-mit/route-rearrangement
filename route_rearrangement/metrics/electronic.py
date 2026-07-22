"""Condensed Fukui indices from RDKit's extended-Hückel — the feature layer under
:mod:`.selectivity`.

The point of this module is that **no chemistry is written down anywhere in it**.  It takes
a molecule, computes a wavefunction, and reports how much of the frontier orbital sits on
each atom.  Whether that constitutes a selectivity problem is decided downstream, by
comparing numbers — never by consulting a table of functional groups.

Fukui's frontier-electron theory gives the condensed indices directly from the frontier
molecular orbitals:

* ``f_plus``  — LUMO density on the atom: susceptibility to **nucleophilic** attack, i.e.
  how electrophilic that atom is;
* ``f_minus`` — HOMO density on the atom: susceptibility to **electrophilic** attack, i.e.
  how nucleophilic that atom is.

``rdEHTTools`` supplies the reduced charge matrix (atoms × orbitals) of an extended-Hückel
calculation, so both fall out of two rows of that matrix.  This is deliberately the cheap
backend: it needs no QM install, is always available wherever RDKit is (like
``isolability``), and costs 0.04–0.25 s for a 20–40 heavy-atom intermediate.  It is also
crude — gas phase, one ETKDG conformer, Hückel-level electronics — which is why the metric
built on it only ever makes **within-molecule** comparisons, where the shared approximations
largely cancel, and never compares an absolute f value between molecules.

``QM_BACKEND`` is the hook for a better wavefunction (xTB/DFT finite-difference Fukui):
set it to a callable with the same signature as :func:`_eht_frontier` and every consumer
picks it up.  The global softness ``1/gap`` is reported alongside, so a downstream metric can
scale local reactivity if it wants to.

Results are cached on canonical SMILES; a rearrangement enumeration re-walks the same
intermediates many times, so the cache carries most of the cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

EMBED_SEED = 0xF00D          # fixed: the same molecule must always give the same numbers
MMFF_ITERS = 500
CACHE_SIZE = 4096


@dataclass(frozen=True)
class Frontier:
    """Per-atom frontier densities for one molecule, indexed by the atom indices of
    ``Chem.MolFromSmiles(smiles)`` for the *canonical* smiles this was computed from."""

    smiles: str
    f_plus: Tuple[float, ...]      # LUMO density  -> electrophilicity of the atom
    f_minus: Tuple[float, ...]     # HOMO density  -> nucleophilicity of the atom
    gap: float                     # HOMO-LUMO gap (eV)

    @property
    def softness(self) -> float:
        """Global softness 1/gap — a soft molecule is reactive everywhere."""
        return 1.0 / self.gap if self.gap else float("inf")


def available() -> bool:
    try:
        from rdkit.Chem import rdEHTTools  # noqa: F401
    except Exception:
        return False
    return frontier("CCO") is not None


def _eht_frontier(mol) -> Optional[Tuple[List[float], List[float], float]]:
    """``(f_plus, f_minus, gap)`` per atom of *mol* (which must carry explicit Hs and a
    conformer), or ``None`` if the calculation does not converge."""
    from rdkit.Chem import rdEHTTools

    ok, res = rdEHTTools.RunMol(mol)
    if not ok:
        return None
    rcm = res.GetReducedChargeMatrix()          # (n_atoms, n_orbitals)
    energies = res.GetOrbitalEnergies()
    homo = res.numElectrons // 2 - 1
    lumo = homo + 1
    if homo < 0 or lumo >= len(energies):
        return None
    n = mol.GetNumAtoms()
    f_minus = [float(rcm[i][homo]) for i in range(n)]
    f_plus = [float(rcm[i][lumo]) for i in range(n)]
    return f_plus, f_minus, float(energies[lumo] - energies[homo])


#: Swap in a richer wavefunction (xTB / DFT finite-difference Fukui) without touching
#: any consumer.  Same signature and units as :func:`_eht_frontier`.
QM_BACKEND: Callable = _eht_frontier


@lru_cache(maxsize=CACHE_SIZE)
def frontier(smiles: str) -> Optional[Frontier]:
    """Condensed Fukui indices for *smiles*, or ``None`` if the molecule cannot be
    embedded or the calculation fails.  Indices align with ``Chem.MolFromSmiles(canonical)``
    where ``canonical`` is :attr:`Frontier.smiles` — use :func:`canonical_index_map` to get
    there from an arbitrarily-ordered mol."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    if canonical != smiles:                      # one cache entry per molecule, not per string
        return frontier(canonical)

    heavy = mol.GetNumAtoms()
    molh = Chem.AddHs(mol)                       # AddHs appends: heavy indices are preserved
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    if AllChem.EmbedMolecule(molh, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(molh, maxIters=MMFF_ITERS)
    except Exception:
        pass                                     # an unoptimized ETKDG geometry still runs
    try:
        out = QM_BACKEND(molh)
    except Exception:
        return None
    if out is None:
        return None
    f_plus, f_minus, gap = out
    return Frontier(smiles=canonical,
                    f_plus=tuple(f_plus[:heavy]), f_minus=tuple(f_minus[:heavy]), gap=gap)


def canonical_index_map(mol) -> Tuple[str, Dict[int, int]]:
    """``(canonical_smiles, {atom index in mol -> atom index in the canonical mol})``.

    Needed because :func:`frontier` is keyed and indexed on canonical SMILES, while callers
    hold a mol whose atom order came from wherever it was built.
    """
    smiles = Chem.MolToSmiles(mol)
    order = mol.GetPropsAsDict(includePrivate=True, includeComputed=True).get(
        "_smilesAtomOutputOrder")
    if order is None:
        return smiles, {}
    return smiles, {orig: pos for pos, orig in enumerate(order)}


def frontier_for(mol) -> Tuple[Optional[Frontier], Dict[int, int]]:
    """:func:`frontier` for an already-built *mol*, plus the index map into its result."""
    smiles, idx_map = canonical_index_map(mol)
    return frontier(smiles), idx_map
