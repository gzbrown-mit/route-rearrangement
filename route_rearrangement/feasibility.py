"""Tier 1 — deterministic chemical invariants on a materialized route.

Applied **after the fact**, by :mod:`.audit`, over a finished run.  The rearrangement
pipeline does not call anything here: it stays a neutral generator, and these rules are
how you *review* what it produced.  A check can therefore be corrected and the whole
corpus re-audited in seconds, and a bug costs a wrong report rather than a route that
was quietly thrown away.

Tier 0 (:mod:`.filters`) asks whether a route is *structurally* coherent: the retro
template matched, molecules parse, the tree connects, atoms are conserved.  Those are
graph and valence facts — a route can pass every one of them and still be chemistry
that cannot happen, because template application is a **substructure** test and says
nothing about electronics, oxidation state, protection or stereocontrol.

This module supplies the missing rejecting layer: checks that are *deterministic*
(no learned model, no score) and whose violation means the step will not work, or
will not work selectively.  Each check implements a motif from :mod:`.motifs`.

Findings carry a severity:

* ``infeasible`` — the step cannot proceed as written (e.g. SNAr on an unactivated
  ring).  ``--strict`` rejects these routes outright.
* ``risk``       — the step can proceed but selectivity/chemoselectivity is not
  established (e.g. a second reducible group is present).

Everything is intentionally conservative: a check abstains rather than guessing, and
severity is downgraded wherever an alternative mechanism could rescue the step (an
unactivated Ar-Br + amine is likely Buchwald-Hartwig, not a failed SNAr, so it is a
``risk``, while Ar-F essentially requires SNAr activation and stays ``infeasible``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

from .materialize import MaterializedRoute, StepRecord
from .templates import StepTemplate

RDLogger.DisableLog("rdApp.*")


@dataclass
class Finding:
    check: str
    motif: str
    severity: str          # "infeasible" | "risk"
    position: int
    step_id: int
    detail: str


# ---------------------------------------------------------------------------
# SMARTS vocabulary
# ---------------------------------------------------------------------------
_SMARTS = {
    "aryl_halide": "[c][F,Cl,Br,I]",
    "aryl_f": "[c][F]",
    "aryl_hetero": "[c][N,O,S;!$([N+](=O)[O-])]",
    "nitro": "[N+](=O)[O-]",
    "aryl_nitro": "[c][N+](=O)[O-]",
    "aniline": "[c][NX3;H1,H2;!$(NC=O);!$(N[N+](=O)[O-])]",
    "boron": "[BX3]",
    "stannane": "[Sn]",
    "azide": "[NX1]~[NX2]~[NX2,NX1]",
    "nitrile": "[NX1]#[CX2]",
    "alkene": "[CX3;!$(C=[O,N,S])]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[CX3](=O)[NX3]",
    "benzyl_ether": "[cX3]:[cX3]-[CH2]-[OX2]-[#6]",
    "free_nh_amine": "[NX3;H1,H2;!$(NC=O);!$(NS(=O)=O);!$([N+])]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "thiol": "[SX2H]",
    "alcohol": "[OX2H][CX4]",
    # electron-withdrawing substituents that activate a ring toward SNAr
    "ewg_nitro": "[N+](=O)[O-]",
    "ewg_cyano": "C#N",
    "ewg_carbonyl": "[CX3]=[OX1]",
    "ewg_sulfonyl": "[SX4](=O)(=O)",
    "ewg_cf3": "C(F)(F)F",
}


@lru_cache(maxsize=None)
def _patt(key: str):
    return Chem.MolFromSmarts(_SMARTS[key])


@lru_cache(maxsize=100_000)
def _mol(smi: str):
    return Chem.MolFromSmiles(smi) if smi else None


def _has(smi: str, key: str) -> bool:
    m, p = _mol(smi), _patt(key)
    return bool(m is not None and p is not None and m.HasSubstructMatch(p))


def _count(smi: str, key: str) -> int:
    m, p = _mol(smi), _patt(key)
    if m is None or p is None:
        return 0
    return len(m.GetSubstructMatches(p, uniquify=True))


def _heavy(smi: str) -> int:
    m = _mol(smi)
    return m.GetNumHeavyAtoms() if m is not None else 0


_EWG_KEYS = ("ewg_nitro", "ewg_cyano", "ewg_carbonyl", "ewg_sulfonyl", "ewg_cf3")


# ---------------------------------------------------------------------------
# SNAr electronic activation  (motifs: nitro_snar_reduction, snar_requires_activation)
# ---------------------------------------------------------------------------
def _aromatic_system(mol, ipso: int) -> set:
    """Atoms of the fused aromatic ring system containing *ipso*."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    containing = [r for r in rings if ipso in r]
    if not containing:
        return set()
    merged = set().union(*containing)
    changed = True
    while changed:                       # absorb rings fused to the growing system
        changed = False
        for r in rings:
            if r & merged and not r <= merged:
                merged |= r
                changed = True
    return merged


def _ring_activation(mol, ipso: int) -> Optional[str]:
    """Why the ring position *ipso* is activated toward SNAr, or ``None``.

    Activation is delocalised through the ring, so a substituent counts at the ortho
    and para positions (ring-graph offsets 1, 3, 5 in a six-membered ring).  An
    aromatic ring nitrogen does the same job electronically with no substituent at
    all — chloropyrimidines, chlorotriazines and every 5-ring/fused azine (purine,
    benzothiazole, quinazoline) are activated this way.  Because those systems are
    ubiquitous in medicinal-chemistry corpora and their exact conjugation pattern is
    not worth modelling atom-by-atom, **any** aromatic nitrogen in the fused system
    counts: the check exists to reject the clear-cut unactivated carbocycle, not to
    adjudicate borderline heteroaromatics.
    """
    system = _aromatic_system(mol, ipso)
    for idx in system:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetIsAromatic() and atom.GetSymbol() == "N":
            return "aza-activated (aromatic N in the ring system)"

    for ring in mol.GetRingInfo().AtomRings():
        if ipso not in ring or len(ring) != 6:
            continue
        n = len(ring)
        pos = ring.index(ipso)
        score = 0.0
        reasons: List[str] = []
        ewg_hits = {k: mol.GetSubstructMatches(_patt(k), uniquify=True)
                    for k in _EWG_KEYS if _patt(k) is not None}
        for offset in range(1, n):
            idx = ring[(pos + offset) % n]
            resonance = offset in (1, 3, 5)           # ortho / para
            where = "ortho" if offset in (1, 5) else "para" if offset == 3 else "meta"
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetIdx() in ring:
                    continue
                sym = nbr.GetSymbol()
                if sym in ("F", "Cl", "Br", "I"):
                    # halogens withdraw inductively: several of them activate a ring
                    # for SNAr even with no resonance acceptor (perfluoroarenes)
                    score += 1.0 if resonance else 0.5
                    reasons.append(f"{sym} {where}")
                    continue
                for key, matches in ewg_hits.items():
                    if any(nbr.GetIdx() in m for m in matches):
                        # ortho/para acceptors stabilise the Meisenheimer complex by
                        # resonance and suffice alone; meta ones only withdraw
                        # inductively and must accumulate
                        score += 2.0 if resonance else 1.0
                        reasons.append(f"{key.replace('ewg_', '')} {where}")
                        break
        if score >= 2.0:
            return "activated by " + ", ".join(reasons[:3])
    return None


def _is_carbocyclic_benzene(mol, ipso: int) -> bool:
    """True iff *ipso* sits on an isolated all-carbon six-membered aromatic ring —
    the only situation where 'no activation' is a confident verdict."""
    system = _aromatic_system(mol, ipso)
    if len(system) != 6:
        return False                      # fused or non-six-membered: stay cautious
    return all(mol.GetAtomWithIdx(i).GetSymbol() == "C"
               and mol.GetAtomWithIdx(i).GetIsAromatic() for i in system)


def _snar_findings(rec: StepRecord, reactants: Sequence[str]) -> List[Finding]:
    """Flag a nucleophilic aromatic substitution whose ring is not activated.

    Detection must be specific: a cross-coupling also consumes an aryl halide, and
    most products contain an aryl-heteroatom bond *somewhere*, so the test is that
    the number of aryl-heteroatom bonds actually **increases** while an aryl halide
    is consumed — and that no boron/tin partner is present (that is a coupling).
    """
    product = rec.new_product
    n_halide_in = sum(_count(r, "aryl_halide") for r in reactants)
    if n_halide_in == 0 or _count(product, "aryl_halide") >= n_halide_in:
        return []                                     # no aryl halide was consumed
    joined = ".".join(reactants)
    if _has(joined, "boron") or _has(joined, "stannane"):
        return []                                     # Suzuki/Stille, not SNAr
    if _count(product, "aryl_hetero") <= sum(_count(r, "aryl_hetero") for r in reactants):
        return []                                     # no new aryl-heteroatom bond

    # The retro template is applied without atom maps, so which halide position
    # actually reacted is unknown.  Be conservative: a substrate is only reported
    # when *every* halide-bearing position on it is unactivated — otherwise the
    # reaction may well have occurred at the activated one.
    candidates: List[Finding] = []
    for smi in reactants:
        mol, p = _mol(smi), _patt("aryl_halide")
        if mol is None or p is None:
            continue
        sites = mol.GetSubstructMatches(p, uniquify=True)
        if not sites:
            continue
        if any(_ring_activation(mol, ipso) is not None for ipso, _h in sites):
            return []                                 # an activated site exists
        for ipso, hal in sites:
            halogen = mol.GetAtomWithIdx(hal).GetSymbol()
            # Confident only on an isolated carbocyclic benzene: Ar-F there
            # essentially requires SNAr activation.  Ar-Br/I + amine is more likely a
            # Pd-catalysed amination, which needs no activating group at all.
            plain = _is_carbocyclic_benzene(mol, ipso)
            hard = plain and halogen == "F"
            if not plain and halogen != "F":
                continue                              # too uncertain to report
            candidates.append(Finding(
                check="snar_activation",
                motif="snar_requires_activation",
                severity="infeasible" if hard else "risk",
                position=rec.position, step_id=rec.orig_step_id,
                detail=(f"aryl-{halogen} displaced with no EWG ortho/para and no ring "
                        f"nitrogen: SNAr is not activated on this substrate"
                        + ("" if hard else "; feasible only if Pd-catalysed"))))
            break
    return candidates[:1]


# ---------------------------------------------------------------------------
# Protecting-group bracket audit  (motif: pg_bracket_intact)
# ---------------------------------------------------------------------------
def _pg_findings(route: MaterializedRoute, brackets: Sequence[Tuple[int, int]],
                 ) -> List[Finding]:
    """Every step that ran inside a (protect, deprotect) bracket must still do so."""
    pos_of = {rec.orig_step_id: rec.position for rec in route.steps}
    out: List[Finding] = []
    for protect, deprotect in brackets:
        if protect not in pos_of or deprotect not in pos_of:
            continue
        p_new, d_new = pos_of[protect], pos_of[deprotect]
        if p_new > d_new:
            out.append(Finding(
                check="pg_bracket", motif="pg_bracket_intact", severity="infeasible",
                position=d_new, step_id=deprotect,
                detail=f"deprotection (step {deprotect}) now precedes its protection "
                       f"(step {protect})"))
            continue
        # steps originally inside the bracket, by original step id ordering
        lo, hi = min(protect, deprotect), max(protect, deprotect)
        for rec in route.steps:
            sid = rec.orig_step_id
            if sid in (protect, deprotect) or not (lo < sid < hi):
                continue
            if not (p_new < rec.position < d_new):
                out.append(Finding(
                    check="pg_bracket", motif="pg_bracket_intact", severity="risk",
                    position=rec.position, step_id=sid,
                    detail=f"step {sid} ran inside the protect({protect})/"
                           f"deprotect({deprotect}) bracket in the literature route "
                           f"but now runs outside it — the group it protected is "
                           f"exposed to this step"))
    return out


# ---------------------------------------------------------------------------
# Metal catalysis / organometallic donors  (motifs: amine_free_before_pd,
# halide_chemoselectivity, organometallic_needs_aprotic)
# ---------------------------------------------------------------------------
_DONORS = (("free_nh_amine", "free N-H amine"), ("carboxylic_acid", "carboxylic acid"),
           ("thiol", "thiol"))


def _coupling_findings(rec: StepRecord, reactants: Sequence[str]) -> List[Finding]:
    joined = ".".join(reactants)
    is_coupling = ((_has(joined, "boron") or _has(joined, "stannane"))
                   and _has(joined, "aryl_halide"))
    if not is_coupling:
        return []
    out: List[Finding] = []
    for key, label in _DONORS:
        if _has(joined, key):
            out.append(Finding(
                check="metal_catalysis_donor", motif="amine_free_before_pd",
                severity="risk", position=rec.position, step_id=rec.orig_step_id,
                detail=f"cross-coupling run with a {label} exposed: Pd coordination / "
                       f"competing amination or protodemetalation not excluded"))
    halides = {rec_atom for smi in reactants for rec_atom in _halogens(smi)}
    if len(halides) > 1:
        out.append(Finding(
            check="metal_catalysis_donor", motif="halide_chemoselectivity",
            severity="risk", position=rec.position, step_id=rec.orig_step_id,
            detail=f"more than one aryl halide type present ({', '.join(sorted(halides))}): "
                   f"oxidative addition selectivity (I > Br > OTf >> Cl) decides the site"))
    return out


def _halogens(smi: str) -> List[str]:
    m, p = _mol(smi), _patt("aryl_halide")
    if m is None or p is None:
        return []
    return [m.GetAtomWithIdx(h).GetSymbol() for _c, h in m.GetSubstructMatches(p, uniquify=True)]


# ---------------------------------------------------------------------------
# Redox chemoselectivity  (motif: reduction_chemoselectivity)
# ---------------------------------------------------------------------------
_REDUCIBLE = ("aryl_nitro", "azide", "nitrile", "alkene", "alkyne", "ketone",
              "aldehyde", "benzyl_ether")
_EASE = {"azide": 1, "aryl_nitro": 2, "alkyne": 3, "alkene": 3, "benzyl_ether": 3,
         "aldehyde": 4, "ketone": 5, "nitrile": 6}


def _redox_findings(rec: StepRecord, reactants: Sequence[str]) -> List[Finding]:
    """A reduction that leaves an equally- or more-easily-reduced group untouched."""
    substrate = max(reactants, key=_heavy) if reactants else ""
    if not substrate:
        return []
    reduced = [k for k in _REDUCIBLE
               if _count(substrate, k) > _count(rec.new_product, k)]
    if not reduced:
        return []
    target = min(reduced, key=lambda k: _EASE.get(k, 9))
    survivors = [k for k in _REDUCIBLE
                 if k not in reduced and _count(rec.new_product, k) > 0
                 and _EASE.get(k, 9) <= _EASE.get(target, 9)]
    if not survivors:
        return []
    return [Finding(
        check="redox_chemoselectivity", motif="reduction_chemoselectivity",
        severity="risk", position=rec.position, step_id=rec.orig_step_id,
        detail=f"reduces {target} while {', '.join(sorted(survivors))} survives — "
               f"these are reduced under comparable or milder conditions")]


# ---------------------------------------------------------------------------
# Stereocontrol support  (motif: stereocontrol_support)
# ---------------------------------------------------------------------------
def _stereocentres(smi: str) -> int:
    m = _mol(smi)
    if m is None:
        return 0
    try:
        return len(Chem.FindMolChiralCenters(m, includeUnassigned=False,
                                             useLegacyImplementation=False))
    except Exception:
        return 0


def _stereo_findings(rec: StepRecord, reactants: Sequence[str]) -> List[Finding]:
    gained = _stereocentres(rec.new_product) - sum(_stereocentres(r) for r in reactants)
    if gained <= 0:
        return []
    if "@" in (rec.retro_smarts or ""):
        return []                       # template itself carries the stereochemistry
    if any(_stereocentres(r) for r in reactants):
        return []                       # substrate control is at least available
    return [Finding(
        check="stereocontrol", motif="stereocontrol_support", severity="risk",
        position=rec.position, step_id=rec.orig_step_id,
        detail=f"sets {gained} stereocentre(s) from an achiral substrate with a "
               f"stereochemistry-free template: the configuration is inherited from "
               f"the literature product, not established by this step")]


# ---------------------------------------------------------------------------
# Convergency quality  (motif: convergent_fragment_balance)
# ---------------------------------------------------------------------------
def _balance_findings(rec: StepRecord) -> List[Finding]:
    if len(rec.synth_precursors) < 2:
        return []
    sizes = sorted(_heavy(s) for s in rec.synth_precursors)
    if sizes[0] == 0:
        return []
    ratio = sizes[-1] / sizes[0]
    if ratio < 4:
        return []
    return [Finding(
        check="fragment_balance", motif="convergent_fragment_balance", severity="risk",
        position=rec.position, step_id=rec.orig_step_id,
        detail=f"coupling joins fragments of {sizes[-1]} and {sizes[0]} heavy atoms "
               f"({ratio:.1f}x): counted as convergent but confers little "
               f"longest-linear-sequence benefit")]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _reactants_of(rec: StepRecord) -> List[str]:
    return list(rec.side_reactants) + list(rec.synth_precursors)


def audit_route(route: MaterializedRoute, templates: Dict[int, StepTemplate], *,
                brackets: Sequence[Tuple[int, int]] = ()) -> List[Finding]:
    """Every Tier 1 finding for *route*, most severe first.

    *brackets* — ``(protect_step_id, deprotect_step_id)`` pairs from
    :func:`synthesis_extraction.compatibility.pg_pairs.detect_pg_pairs`; pass ``()``
    to skip the protecting-group audit.
    """
    out: List[Finding] = []
    for rec in route.steps:
        # independent of the step's reactants — a template that cannot reproduce its
        # own literature reaction undermines every rearrangement built on it
        tpl = templates.get(rec.orig_step_id)
        if tpl is not None and not tpl.retro_identity_ok:
            out.append(Finding(
                check="template_self_consistency", motif="",
                severity="risk", position=rec.position, step_id=rec.orig_step_id,
                detail="this step's retro template does not reproduce its own "
                       "literature reactants — any rearrangement built on it is "
                       "less trustworthy"))
        reactants = _reactants_of(rec)
        if not reactants:
            continue
        out.extend(_snar_findings(rec, reactants))
        out.extend(_coupling_findings(rec, reactants))
        out.extend(_redox_findings(rec, reactants))
        out.extend(_stereo_findings(rec, reactants))
        out.extend(_balance_findings(rec))
    out.extend(_pg_findings(route, brackets))
    out.sort(key=lambda f: (f.severity != "infeasible", f.position))
    return out


def audit_record(record: dict, templates: Optional[Dict[int, StepTemplate]] = None,
                 brackets: Sequence[Tuple[int, int]] = ()) -> List[Finding]:
    """Audit one ``routes.jsonl``/``scored.jsonl`` record.

    Most checks need only the record's own reactants and products.  *templates* and
    *brackets* come from the original corpus route and enable the two checks that
    cannot be answered from the record alone (template self-consistency and the
    protecting-group bracket); omit them to audit results without corpus access.
    """
    from .schema import route_from_record

    return audit_route(route_from_record(record), templates or {}, brackets=brackets)


def summarize(findings: Sequence[Finding]) -> dict:
    """Compact, JSON-serialisable feasibility block for a route record."""
    infeasible = [f for f in findings if f.severity == "infeasible"]
    return {
        "n_infeasible": len(infeasible),
        "n_risk": len(findings) - len(infeasible),
        "checks_fired": sorted({f.check for f in findings}),
        "findings": [asdict(f) for f in findings],
    }


def detect_brackets(full_graph: dict) -> List[Tuple[int, int]]:
    """``(protect, deprotect)`` step-id pairs of the original route; ``[]`` on failure."""
    try:
        from synthesis_extraction.compatibility.pg_pairs import detect_pg_pairs
        return [(int(p.as_bracket()[0]), int(p.as_bracket()[1]))
                for p in detect_pg_pairs(full_graph) if p.in_scope]
    except Exception:
        return []
