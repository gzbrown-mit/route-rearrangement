"""Metric — selectivity from computed electronic structure, not from a rule table.

The question a bench chemist actually asks of a rearranged step is *"is the site I am aiming
at still the most reactive one in this molecule?"*  A rearrangement changes which groups are
present and unmasked when a step runs, so the answer changes with the ordering — that is
exactly the signal a route-ranking metric should carry, and exactly what a protecting group
exists to fix.

This metric answers it by **comparing numbers on atoms**.  For each step:

1. the **reacting atoms** come from the step's own retro template — the map numbers whose
   bonding changes between the two halves.  Nothing is named or classified;
2. the **rival sites** come from two sources, neither of them a group vocabulary: other
   places the step's own template matches, and other atoms sharing the reacting atom's
   radius-1 environment (element, aromaticity, hybridization, charge, bonded neighbours).
   The second source carries the weight — rdchiral templates are specific enough that they
   rarely match twice — and rivals are labelled ``symmetry_equivalent`` (reacting there
   gives the same product: an over-reaction risk, e.g. bis-acylation of a symmetric
   diamine) or ``distinct`` (a different product: a regio-/chemoselectivity risk).  This is
   exactly the set a protecting group exists to mask;
3. both **attack modes** (LUMO density for an electrophilic site, HOMO density for a
   nucleophilic one) are read for every reacting fragment and the **operative** one is kept —
   the mode in which the site actually carries frontier density.  A site with density in
   neither mode abstains rather than scoring, because the ratio of two near-zero numbers is
   noise.  Two alternatives were tried and measured away on PaRoutes n1: assigning roles by
   comparing f across the two reactants (compares absolute densities between molecules, which
   this backend cannot support) and keeping the *worse* reading (picks the mode the site is
   absent from 59% of the time);
4. **sterics** discount a crowded rival before the comparison.  Fukui indices are purely
   electronic, so without this a primary and a tertiary amine of equal HOMO density look
   equally reactive — the one place the descriptor most obviously parts company with a bench
   chemist.  The bulk feature is ``heavy_atoms_decay`` borrowed from
   ``synthesis_extraction.step_classification.descriptors`` (the same number that package's
   step classifier uses, not a re-implementation): every heavy atom contributes
   ``exp(-BULK_LAMBDA * d)`` for its bond distance *d* from the site.  Only the site-to-rival
   *difference* enters, so this stays a within-molecule comparison, and the adjustment is
   clamped so sterics can never override the electronics outright;
5. the **margin** is ``(f_site - f_rival) / (f_site + f_rival)`` ∈ [-1, 1]: 1 when the site
   is the only copy, 0 when two copies are equally reactive, negative when a rival copy is
   hotter than the one the step is aimed at.  ``margin_electronic`` records the same number
   before the steric discount, so the two contributions stay separable.

``score = -Σ (1 - margin)/2`` over steps (higher = better) — the frontier density landing on
a rival copy rather than the intended one, summed over the route.  A step with a single
reacting site costs nothing; the ordering that leaves a second, equally reactive copy
unmasked pays 0.5 for it, and one that aims at the colder of two copies pays more.  Steps
whose template does not match, or whose electronic structure will not converge, **abstain**
and are counted rather than scored.

Each reading also records **activation** — the site's density as a fraction of the hottest
atom in the molecule for that mode.  That is a feasibility statement rather than a
selectivity one, so it stays out of the score, but it is the number that collapses when an
ordering strips a site's activating group (SNAr on a ring whose nitro has already been
reduced: 0.17 -> 0.0004), which makes it the natural feature-based successor to the Tier 1
activation checks.

Comparisons are always *within one molecule*, where the crudeness of the extended-Hückel
backend (:mod:`.electronic`) largely cancels; absolute f values are never compared between
molecules.  Diagnostics per step (mode, site, rival, margin, activation, gap, local softness)
are kept so the metric can be calibrated against the ordering motifs in :mod:`..motifs`.

Supersedes the SMARTS-driven ``competing`` metric, which asked whether a *named* sensitive
group was present rather than whether the site was electronically preferred.

**Known scope limit.**  A rival must share the reacting atom's element, so a phenol competing
with an amine for an acylation is not currently seen — cross-element rivalry is precisely
what Fukui indices are good at, but it needs a generalized reaction pattern to say which
foreign atoms are candidate partners at all.  The FrequenTree template ladder already used by
:mod:`..literature_precedent` supplies exactly that generalization and is the intended next
source of rivals; inventing a heteroatom rule here would put an extracted rule back into a
tier that is meant to be feature-based.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rdkit import Chem, RDLogger

from . import electronic as _el
from .. import deps  # noqa: F401  (puts synthesis_extraction on sys.path)
from .base import reactions
from synthesis_extraction.step_classification import descriptors as _D

RDLogger.DisableLog("rdApp.*")

HIGHER_IS_BETTER = True

EPS = 1e-9
#: a margin at or below this counts as a reported liability (not just a rounding wobble)
LIABILITY_MARGIN = 0.0
#: how many copies of the transformation to look for on one substrate
MAX_MATCHES = 8
#: a rival this far ahead of the intended site is called out separately
SEVERE_MARGIN = -0.5
#: below this share of the molecule's frontier density the site is not engaged in that mode
#: at all, so comparing it with a rival divides noise by noise — the step abstains instead
ACTIVATION_FLOOR = 0.01

# --- steric term ---------------------------------------------------------------------
#: locality of the bulk descriptor, passed to ``heavy_atoms_decay``: an atom *d* bonds from
#: the site contributes ``exp(-BULK_LAMBDA * d)``, so the reaction site's own substituents
#: dominate and remote scaffold mass fades out
BULK_LAMBDA = 1.0
#: how hard bulk discounts frontier density.  Calibrated so that the textbook case —
#: acylating 2-methylpropane-1,2-diamine, where the primary CH2-NH2 wins over the amine on
#: the quaternary carbon — comes out at margin ~0.3: a clear preference, not an absolute one.
#: The measured bulk difference there is 0.171, and ln((1+0.3)/(1-0.3)) / 0.171 = 3.6.
STERIC_LAMBDA = 3.6
#: ceiling on the steric log-adjustment, so a rival buried in a large scaffold can never
#: override the electronics outright (exp(1.5) = 4.5x, i.e. margin +-0.64 from sterics alone)
STERIC_CLAMP = 1.5
#: scratch atom-map number used to point the borrowed bulk descriptor at one atom
_BULK_MAP = 9999


def available() -> bool:
    return _el.available()


# ---------------------------------------------------------------------------
# The reaction site, read off the step's own template
# ---------------------------------------------------------------------------
def _neighbourhood(patt) -> Dict[int, Tuple]:
    """``{atom map -> its bonding signature}`` for one half of a template."""
    out: Dict[int, Tuple] = {}
    for atom in patt.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap <= 0:
            continue
        bonds = []
        unmapped = 0
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomMapNum() > 0:
                bonds.append((other.GetAtomMapNum(), str(bond.GetBondType())))
            else:
                unmapped += 1
        out[amap] = (tuple(sorted(bonds)), unmapped)
    return out


@lru_cache(maxsize=2048)
def _template_centre(retro_smarts: str) -> Tuple[Optional[object], frozenset]:
    """``(precursor pattern, map numbers whose bonding changes)`` for a retro template.

    The template is written retro (``product >> precursors``), so the *right* half is what
    the forward step consumes and is the pattern to locate in the substrate.
    """
    if not retro_smarts or ">>" not in retro_smarts:
        return None, frozenset()
    try:
        prod_side, prec_side = retro_smarts.split(">>")
        prod_patt = Chem.MolFromSmarts(prod_side)
        prec_patt = Chem.MolFromSmarts(prec_side)
    except Exception:
        return None, frozenset()
    if prod_patt is None or prec_patt is None:
        return None, frozenset()

    before, after = _neighbourhood(prec_patt), _neighbourhood(prod_patt)
    centre = {m for m in set(before) | set(after) if before.get(m) != after.get(m)}
    return prec_patt, frozenset(centre)


def _site_matches(mol, prec_patt, centre_maps: frozenset) -> List[Set[int]]:
    """One set of changing atom indices per place the template matches *mol*.

    The first entry is the site the step is aimed at; the rest are **rival sites** — other
    copies of the very same transformation on the same molecule.  Defining rivals this way
    means the rival set comes from the step's own template, so no functional-group
    vocabulary is consulted and no cross-reaction guess is made: a second free amine is a
    rival for an amide coupling because the coupling template matches it too.
    """
    if prec_patt is None or not centre_maps:
        return []
    try:
        matches = mol.GetSubstructMatches(prec_patt, uniquify=True, maxMatches=MAX_MATCHES)
    except Exception:
        return []
    if not matches:
        return []
    centre_positions = [i for i, a in enumerate(prec_patt.GetAtoms())
                        if a.GetAtomMapNum() in centre_maps]
    out: List[Set[int]] = []
    for match in matches:
        atoms = {match[p] for p in centre_positions if p < len(match)}
        if atoms and atoms not in out:
            out.append(atoms)
    return out


# ---------------------------------------------------------------------------
# Fukui comparison
# ---------------------------------------------------------------------------
def _densities(frontier, mode: str):
    return frontier.f_plus if mode == "electrophile" else frontier.f_minus


def _env_class(atom) -> Tuple:
    """An atom's radius-1 environment: what a reagent arriving at it would encounter.

    Two atoms sharing this signature are the same kind of site, so their frontier densities
    are commensurable.  This is the second source of rivals, and the one that carries the
    weight in practice: rdchiral templates are specific enough (full degree, H-count and
    charge specification) that they rarely match a second time on the same substrate, so
    template re-matching alone finds almost no rivals.
    """
    return (atom.GetAtomicNum(), atom.GetIsAromatic(), str(atom.GetHybridization()),
            atom.GetFormalCharge(),
            tuple(sorted((nb.GetAtomicNum(), str(bond.GetBondType()))
                         for nb, bond in zip(atom.GetNeighbors(), atom.GetBonds()))))


def _bulk(mol, idx: int) -> float:
    """Distance-decayed heavy-atom bulk around one atom — how crowded that site is.

    This is ``synthesis_extraction.step_classification.descriptors.heavy_atoms_decay``
    itself, not a re-implementation, so the steric feature here is the same number that
    package's step classifier uses.  It keys on atom-map numbers, and rearranged routes are
    map-free, so the anchor is temporarily stamped with a map number and restored.
    """
    atom = mol.GetAtomWithIdx(idx)
    previous = atom.GetAtomMapNum()
    atom.SetAtomMapNum(_BULK_MAP)
    try:
        return _D.heavy_atoms_decay(mol, [_BULK_MAP], BULK_LAMBDA)
    except Exception:
        return 0.0
    finally:
        atom.SetAtomMapNum(previous)


def _steric_factor(bulk_rival: float, bulk_site: float) -> float:
    """How much the rival's frontier density is discounted for being the more crowded site.

    Only the *difference* in bulk matters — the site's own weight cancels — which keeps this
    a within-molecule comparison like everything else in the metric.  Fukui indices are
    purely electronic, so without this a primary and a tertiary amine of equal HOMO density
    look equally reactive, which is the one place the descriptor most obviously disagrees
    with a bench chemist.
    """
    delta = STERIC_LAMBDA * (bulk_rival - bulk_site)
    return math.exp(-max(-STERIC_CLAMP, min(STERIC_CLAMP, delta)))


def _surviving_classes(product: str) -> frozenset:
    """The environment classes still present in the step's product.

    A rival only matters if it is still reactive *after* the step.  Both sulfur atoms of
    Lawesson's reagent, both carbonyls of Boc anhydride and both chlorines of thionyl
    chloride are symmetry-equivalent, so without this test every Boc protection in the
    corpus is charged 0.5 for a selectivity problem it does not have — the second site is
    consumed or leaves, and nothing of that kind survives into the product.  A symmetric
    diamine that is only mono-acylated *does* leave a free amine behind, and is still
    charged.  (The same reasoning retired the reactant-counting version of the old
    ``competing`` metric, where a ring-closing bis-alkylation looked like a liability.)
    """
    mol = Chem.MolFromSmiles(product) if product else None
    if mol is None:
        return frozenset()
    return frozenset(_env_class(a) for a in mol.GetAtoms())


def _env_rivals(mol, anchor: int, claimed: Set[int]) -> List[Tuple[int, str]]:
    """``[(atom index, kind)]`` for atoms of the anchor's own environment class.

    ``kind`` separates the two liabilities a chemist protects against: a
    ``symmetry_equivalent`` rival reacts to give the *same* product, so it is an
    over-reaction risk (bis-acylation of a symmetric diamine), while a ``distinct`` rival
    gives a different product and is a regio-/chemoselectivity risk.
    """
    want = _env_class(mol.GetAtomWithIdx(anchor))
    try:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:
        ranks = list(range(mol.GetNumAtoms()))
    out = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in claimed or _env_class(atom) != want:
            continue
        kind = "symmetry_equivalent" if ranks[idx] == ranks[anchor] else "distinct"
        out.append((idx, kind))
    return out


def _hottest(values, idx_map: Dict[int, int], atoms: Set[int]) -> Optional[Tuple[int, float]]:
    """``(atom index, f)`` for the atom of *atoms* carrying the most frontier density."""
    best = None
    for idx in atoms:
        pos = idx_map.get(idx)
        if pos is None or pos >= len(values):
            continue
        if best is None or values[pos] > best[1]:
            best = (idx, values[pos])
    return best


def _evaluate(mol, frontier, idx_map, matches: List[Set[int]], mode: str,
              surviving: frozenset = frozenset()) -> Optional[dict]:
    """Compare the intended site against every rival copy of the same transformation."""
    values = _densities(frontier, mode)
    site = _hottest(values, idx_map, matches[0])
    if site is None:
        return None
    anchor, f_site = site

    # rivals from two sources: other places this same template matches, and other atoms of
    # the anchor's own environment class.  Either way a rival must survive the step to be a
    # liability — see :func:`_surviving_classes`.
    def survives(idx: int) -> bool:
        return not surviving or _env_class(mol.GetAtomWithIdx(idx)) in surviving

    rivals: List[Tuple[int, float, str]] = []
    consumed = 0
    for other in matches[1:]:
        got = _hottest(values, idx_map, other)
        if got is None:
            continue
        if survives(got[0]):
            rivals.append((got[0], got[1], "template_match"))
        else:
            consumed += 1
    claimed = {i for m in matches for i in m}
    for idx, kind in _env_rivals(mol, anchor, claimed):
        pos = idx_map.get(idx)
        if pos is None or pos >= len(values):
            continue
        if survives(idx):
            rivals.append((idx, values[pos], kind))
        else:
            consumed += 1

    # Sterics: a crowded rival is discounted before the comparison, so the winner is the site
    # that is both electronically hot and reachable.  Skipped entirely when there is nothing
    # to compare against — most steps have no rival, and the bulk descriptor walks a distance
    # matrix, so computing it unconditionally more than doubled the metric's cost.
    bulk_site = _bulk(mol, anchor) if rivals else None
    weighted = []
    for idx, f, kind in rivals:
        bulk_rival = _bulk(mol, idx)
        weighted.append((idx, f * _steric_factor(bulk_rival, bulk_site), kind, bulk_rival))

    best_rival = max(weighted, key=lambda r: r[1]) if weighted else None
    f_rival = best_rival[1] if best_rival else 0.0
    f_rival_raw = max((f for _, f, _ in rivals), default=0.0)

    total = abs(f_site) + abs(f_rival)
    margin = (f_site - f_rival) / total if total > EPS else 0.0
    total_raw = abs(f_site) + abs(f_rival_raw)
    margin_electronic = (f_site - f_rival_raw) / total_raw if total_raw > EPS else 0.0

    # activation: how the intended site compares with the hottest atom in the whole molecule
    # for this mode.  Not part of the score — it answers "is this site reactive at all",
    # which is a feasibility question, not a selectivity one — but it is the number that
    # moves when an ordering removes a site's activating group, so it is recorded.
    mol_max = max(values) if values else 0.0
    activation = round(f_site / mol_max, 4) if mol_max > EPS else 0.0

    return {
        "mode": mode,
        "site_atom": mol.GetAtomWithIdx(anchor).GetSymbol(),
        "f_site": round(f_site, 5),
        "f_rival": round(f_rival, 5),
        "f_rival_electronic": round(f_rival_raw, 5),
        "bulk_site": round(bulk_site, 3) if bulk_site is not None else None,
        "bulk_rival": round(best_rival[3], 3) if best_rival else None,
        "margin_electronic": round(margin_electronic, 4),
        "n_rival_sites": len(rivals),
        "n_rivals_consumed": consumed,
        "rival_atom": mol.GetAtomWithIdx(best_rival[0]).GetSymbol() if best_rival else None,
        "rival_kind": best_rival[2] if best_rival else None,
        "margin": round(margin, 4),
        "activation": activation,
        "gap": round(frontier.gap, 3),
        "local_softness": round(f_site * frontier.softness, 5),
        # frontier density sitting on a rival copy rather than the intended site: 0 when the
        # site is the only one, 0.5 for two equally reactive copies, -> 1 when a rival wins
        "penalty": round((1.0 - margin) / 2.0 if rivals else 0.0, 4),
    }


def _fragments_with_sites(reactants: Sequence[str], prec_patt, centre_maps):
    """``[(fragment mol, site indices)]`` — the reacting fragments of one step."""
    combined = Chem.MolFromSmiles(".".join(reactants)) if reactants else None
    if combined is None:
        return []
    matches = _site_matches(combined, prec_patt, centre_maps)
    if not matches:
        return []
    # GetMolFrags gives each fragment's original indices in ascending order and, with
    # asMols, mols whose atoms follow that same order — so position within the group tuple
    # is the atom index in the fragment mol.
    groups = Chem.GetMolFrags(combined)
    try:
        frags = Chem.GetMolFrags(combined, asMols=True, sanitizeFrags=True)
    except Exception:
        return []
    out = []
    for group, frag in zip(groups, frags):
        if frag.GetNumAtoms() != len(group):
            continue
        local = [{group.index(i) for i in m if i in group} for m in matches]
        local = [m for m in local if m]
        if local:
            out.append((frag, local))
    return out


def _step_selectivity(reactants: Sequence[str], retro_smarts: str,
                      product: str = "") -> dict:
    """Selectivity assessment for one step, or ``{"abstain": reason}``."""
    prec_patt, centre_maps = _template_centre(retro_smarts)
    if prec_patt is None or not centre_maps:
        return {"abstain": "no_template_centre"}

    frags = _fragments_with_sites(reactants, prec_patt, centre_maps)
    if not frags:
        return {"abstain": "template_did_not_match"}

    # Each reacting fragment is read in both modes and the **operative** one is kept: the mode
    # in which the site actually carries frontier density (highest ``activation``).  Two other
    # rules were tried and measured on PaRoutes n1 and are wrong:
    #   * assigning roles by comparing f between the two reactants compares absolute densities
    #     across molecules, which this backend cannot support (a deactivated arene plus a good
    #     amine comes out backwards);
    #   * keeping the *worse* of the two readings looks conservative but picks the mode with
    #     the lower activation 59% of the time (median 0.19 vs 0.52) — i.e. it mostly scores
    #     the orbital the site is absent from, where f_site ~ 1e-4 against a rival at 1e-1
    #     yields margin ~ -1 out of pure numerical noise.
    surviving = _surviving_classes(product)
    readings: List[dict] = []
    for frag, matches in frags:
        frontier, idx_map = _el.frontier_for(frag)
        if frontier is None or not idx_map:
            continue
        modes = [_evaluate(frag, frontier, idx_map, matches, mode, surviving)
                 for mode in ("electrophile", "nucleophile")]
        modes = [m for m in modes if m is not None]
        if modes:
            kept = max(modes, key=lambda p: p["activation"])
            # both activations travel with the kept reading.  Reporting only the operative
            # mode's would hide the feasibility signal exactly when it matters: reducing the
            # nitro of an SNAr substrate does not merely lower the electrophilic activation,
            # it makes the *nucleophilic* reading operative, so a single number would flip
            # from 0.17 to 1.0 and look like an improvement.
            kept = dict(kept, activation_by_mode={m["mode"]: m["activation"] for m in modes})
            readings.append(kept)
    if not readings:
        return {"abstain": "no_wavefunction"}

    # A site with no frontier density in either mode cannot be compared with anything: the
    # ratio of two numbers that are both ~zero is noise, not selectivity.  That is a
    # feasibility observation, so it abstains from the score and reports the activation.
    live = [r for r in readings if r["activation"] >= ACTIVATION_FLOOR]
    if not live:
        best = max(readings, key=lambda p: p["activation"])
        return {"abstain": "site_not_frontier_active",
                "activation": best["activation"], "mode": best["mode"]}

    # the step is only as selective as its worst reacting partner
    worst = max(live, key=lambda p: p["penalty"])
    # only the governing reading is kept: the discarded ones roughly double the size of a
    # scored corpus and their information survives in ``activation_by_mode``
    return {"n_readings": len(readings), **worst}


def selectivity(record: dict) -> dict:
    """``{score, n_liabilities, n_abstain, per_step, worst_step}`` for one route."""
    by_position = {s["position"]: s for s in record.get("steps", [])}
    per_step: List[dict] = []
    penalty_total = 0.0
    n_abstain = 0

    for r in reactions(record):
        step = by_position.get(r.position, {})
        result = _step_selectivity(r.reactants, step.get("retro_smarts", ""), r.product)
        entry = {"position": r.position, **result}
        per_step.append(entry)
        if "abstain" in result:
            n_abstain += 1
        else:
            penalty_total += result["penalty"]

    if not per_step:
        return {}

    scored = [s for s in per_step if "abstain" not in s]
    worst = min(scored, key=lambda s: s["margin"]) if scored else None
    return {
        "n_steps": len(per_step),
        "n_scored": len(scored),
        "n_abstain": n_abstain,
        "n_liabilities": sum(1 for s in scored if s["margin"] < LIABILITY_MARGIN),
        "n_severe": sum(1 for s in scored if s["margin"] <= SEVERE_MARGIN),
        "worst_step": {k: worst[k] for k in ("position", "mode", "site_atom", "rival_atom",
                                             "margin")} if worst else None,
        "per_step": per_step,
        "score": round(-penalty_total, 4),
    }
