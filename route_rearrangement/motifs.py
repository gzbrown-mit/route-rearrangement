"""Catalogue of *ordering-dependent* chemical motifs — the patterns a rearrangement
can break, and the ground truth for judging whether the engine is performing.

This catalogue is a **review instrument, not a filter**.  Nothing here runs inside the
rearrangement pipeline: the pipeline enumerates, materializes and scores without
consulting these rules, and :mod:`.audit` applies them to a finished run afterwards.
Keeping the two apart means the generator is never biased by a heuristic, the rules can
be retuned and re-run over an existing corpus without regenerating a route, and a wrong
rule produces a wrong report instead of a silently discarded result.

Each motif is a rule a bench chemist applies without thinking: a pair (or triple) of
steps whose **relative order is not free**, for a reason that is deterministic
chemistry rather than taste.  They are the right validation set for route
rearrangement because the literature route always satisfies them, so any rearrangement
that violates one is wrong *regardless* of how it scores.

The canonical case is ``nitro_snar_reduction``:

    install NO2  ->  SNAr  ->  reduce NO2 to NH2

The nitro group is doing double duty — it is the ring-activating EWG that makes the
SNAr go at all, and it is the masked form of the aniline.  Reduce it first and the
ring is electron-rich: the SNAr dies, and the freshly minted aniline competes as a
nucleophile.  Nothing about atom bookkeeping forbids that order (measured on PaRoutes
``n1-8``: 3,360 of 6,720 materially-valid orderings put the reduction first), so it
must be caught by an electronic rule.

``check`` names the :mod:`.feasibility` check that enforces the motif, or ``None``
where the motif is currently documented-but-unenforced (the honest to-do list).
``mined_by`` is the SMARTS/predicate the :mod:`.find_motifs` CLI uses to pull real
instances out of a corpus, so each rule can be regression-tested against literature
routes that actually exhibit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Motif:
    name: str
    family: str
    rule: str                       # the ordering constraint, in one line
    why: str                        # the chemistry that makes it non-negotiable
    check: Optional[str] = None     # feasibility check enforcing it, if any
    mined_by: str = ""              # how find_motifs locates instances
    examples: List[str] = field(default_factory=list)   # known corpus tree ids


MOTIFS: List[Motif] = [
    # ---------------------------------------------------------------- electronics
    Motif(
        name="nitro_snar_reduction",
        family="electronic activation",
        rule="an SNAr on a nitroarene must precede reduction of that nitro group",
        why="NO2 is the ring-activating EWG the SNAr depends on; the aniline it "
            "becomes is electron-donating (ring deactivated toward SNAr) and is "
            "itself a competing nucleophile",
        check="snar_activation",
        mined_by="route contains an SNAr step and a nitro->amine reduction on the "
                 "same ring system",
        examples=["n1-8"],
    ),
    Motif(
        name="snar_requires_activation",
        family="electronic activation",
        rule="any SNAr needs an EWG ortho/para to the leaving group, or an "
             "alpha/gamma ring nitrogen (aza-activation)",
        why="SNAr proceeds through a Meisenheimer complex; without an acceptor to "
            "delocalise the negative charge the barrier is prohibitive. Chloro-"
            "pyrimidines/triazines are activated by ring N, not by substituents",
        check="snar_activation",
        mined_by="aryl halide + N/O/S nucleophile -> aryl heteroatom bond",
        examples=["n1-13"],
    ),
    Motif(
        name="friedel_crafts_before_deactivation",
        family="electronic activation",
        rule="Friedel-Crafts acylation/alkylation must precede installation of a "
             "strong EWG (nitro, sulfonyl) on the same ring",
        why="FC fails on strongly deactivated arenes; the Lewis acid also complexes "
            "basic nitrogen",
        check=None,
        mined_by="FC acylation step and a nitration/sulfonylation step on one ring",
    ),
    Motif(
        name="eas_regiochemistry",
        family="electronic activation",
        rule="an electrophilic aromatic substitution must run when the directing "
             "groups present give the observed isomer",
        why="o/p- vs m-direction is set by the substituents present at that moment; "
            "reordering silently changes which isomer forms",
        check=None,
        mined_by="nitration/halogenation/sulfonylation on a substituted arene",
    ),
    Motif(
        name="directed_metalation_needs_dmg",
        family="electronic activation",
        rule="directed ortho-metalation must run while the directing group is intact "
             "and unprotected",
        why="DoM regiochemistry is entirely set by the DMG coordinating the base",
        check=None,
        mined_by="organolithium + substituted arene -> ortho-functionalised arene",
    ),

    # ------------------------------------------------------------ protecting groups
    Motif(
        name="pg_bracket_intact",
        family="protecting groups",
        rule="every step that ran inside a protect/deprotect bracket must still run "
             "inside it",
        why="the protecting group was installed because that functionality does not "
            "survive the intervening chemistry; moving a step outside the bracket "
            "exposes it",
        check="pg_bracket",
        mined_by="detect_pg_pairs finds a (protect, deprotect) pair",
    ),
    Motif(
        name="amine_free_before_pd",
        family="protecting groups",
        rule="do not unmask a basic N-H amine before a Pd-catalysed coupling",
        why="free amines coordinate Pd (catalyst inhibition) and compete as "
            "nucleophiles in Buchwald-Hartwig amination of the aryl halide present",
        check="metal_catalysis_donor",
        mined_by="Boc/Cbz/Fmoc removal and a cross-coupling in the same route",
        examples=["n1-8"],
    ),
    Motif(
        name="acid_labile_orthogonality",
        family="protecting groups",
        rule="an acidic deprotection must not run while another acid-labile group "
             "(acetal, THP, trityl, tBu ester) must survive",
        why="TFA/HCl conditions are not orthogonal; the other group is lost",
        check=None,
        mined_by="two acid-labile PGs present in one route",
    ),

    # ------------------------------------------------------------- chemoselectivity
    Motif(
        name="reduction_chemoselectivity",
        family="chemoselectivity",
        rule="a reduction must not run while a more-easily-reduced group is present "
             "and expected to survive",
        why="H2/Pd-C reduces nitro, alkene, azide, benzyl ether and dehalogenates; "
            "hydrides discriminate ketone vs ester only within a window",
        check="redox_chemoselectivity",
        mined_by="a reduction step whose substrate carries a second reducible group",
    ),
    Motif(
        name="halide_chemoselectivity",
        family="chemoselectivity",
        rule="with two different halides present, the coupling occurs at the more "
             "reactive one (I > Br > OTf >> Cl)",
        why="oxidative addition order is fixed; a rearrangement that exposes a "
            "second halide changes which site couples",
        check="metal_catalysis_donor",
        mined_by="cross-coupling step whose substrate holds >1 distinct halide",
    ),
    Motif(
        name="organometallic_needs_aprotic",
        family="chemoselectivity",
        rule="Grignard/organolithium addition requires no free O-H, N-H or CO2H, and "
             "no competing electrophile",
        why="the reagent is quenched by protic groups and adds to the most "
            "electrophilic carbonyl, not necessarily the intended one",
        check="metal_catalysis_donor",
        mined_by="organometallic addition step",
    ),
    Motif(
        name="strong_base_vs_ester",
        family="chemoselectivity",
        rule="strong base (LDA/NaH/NaOR) must not run while an enolisable ester and "
            "an epimerisable stereocentre are both exposed",
        why="Claisen condensation and alpha-epimerisation",
        check=None,
        mined_by="enolate alkylation step with an ester elsewhere in the substrate",
    ),

    # ------------------------------------------------------------------ stereochem
    Motif(
        name="stereocontrol_support",
        family="stereochemistry",
        rule="a step that sets a stereocentre needs a stereo-directing element "
             "(existing centre, auxiliary or chiral reagent) present at that point",
        why="substrate-controlled diastereoselection is impossible on an achiral "
            "substrate; copying the literature configuration then asserts an "
            "outcome the chemistry cannot deliver",
        check="stereocontrol",
        mined_by="step whose product gains a stereocentre",
    ),

    # ---------------------------------------------------------------- assembly/ring
    Motif(
        name="macrocyclisation_late",
        family="ring formation",
        rule="macrocyclisation must be intramolecular and late (high dilution)",
        why="a rearrangement that makes the ring-closing partners separate molecules "
            "turns it into an oligomerisation",
        check=None,
        mined_by="ring of >=8 atoms formed in one step",
    ),
    Motif(
        name="convergent_fragment_balance",
        family="assembly",
        rule="a coupling only buys convergency if the two fragments are of "
             "comparable size",
        why="joining a 40-atom fragment to a 5-atom one is a linear step wearing a "
            "convergent hat; the longest-linear-sequence benefit is not real",
        check="fragment_balance",
        mined_by="step with >=2 synthesized precursors",
    ),
]

BY_NAME = {m.name: m for m in MOTIFS}
ENFORCED = [m for m in MOTIFS if m.check]
UNENFORCED = [m for m in MOTIFS if not m.check]
