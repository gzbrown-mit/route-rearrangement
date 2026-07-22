"""Per-route rigidity map: which orderings in *this* synthesis are locked, and by what.

The corpus statistics say what chemists do in general.  This turns that into a statement about
one route: for every pair of transformations in it, is their order fixed by chemistry, fixed by
habit, or genuinely free?  Two independent facts are combined, and keeping them separate is the
whole point:

1. **Does this route's own partial order force it?**  Exact, from
   ``ScheduleLattice.precedence_probabilities()`` on the route's dependency graph.  This is
   physics — atom lineage, protecting-group brackets — and it is route-specific.
2. **Does the literature fix it?**  From the corpus precedence table.  This is precedent, and
   it says nothing about whether the swap is *possible*.

Crossing them gives five verdicts:

===================  ==========================================================================
``forced``           the partial order requires it — cannot be reordered at all
``conventional``     free to move, but the literature almost always runs it this way
``anti``             free to move, and this route already runs it *against* the literature
``free``             free to move, and the literature has no preference either way
``unknown``          free to move, but there is too little independent evidence to say
===================  ==========================================================================

Two design choices matter for trusting the output.

**Evidence is counted in route skeletons, not routes.**  A pair backed by 1,690 routes may rest
on 170 distinct synthesis skeletons — a med-chem campaign republishes one route shape many
times, and counting each as fresh support is the synthesis analogue of treating two strains of
one bacterium as independent confirmation of a gene cluster.  Across the corpus the nominal
count overstates independent evidence by ~6x at the median and 100x at worst, so the gate here
is ``n_effective``.

**The verdicts rest on the robust statistics.**  ``forced`` comes from an exact per-route
computation, and ``conventional`` from a directional *frequency* plus an evidence floor — not
from the corpus p-values, which the same non-independence inflates.  So the map degrades
gracefully where the significance counts would not.

The practical payoff is pruning.  A route in this corpus has a median of 24 valid orderings and
a mean of 4,414; enumerating them is not useful, but knowing which handful of swaps are
*interesting* is.  :func:`conventional_constraints` feeds straight into
``lattice_for(dep, extra_constraints=...)``, so the existing rearrangement engine can be told
to explore only where the literature is silent — or, inverted, to go looking specifically for
the swaps that break a convention nothing forbids.

Usage::

    python -m route_rearrangement.route_synteny.rigidity \\
        --genomes results/genomes.jsonl --precedence results/precedence.json \\
        --route all-1234 [--corpus trees.jsonl]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .. import deps  # noqa: F401
from .corpus import UNKNOWN, Genome
from .nulls import lattice_for_genome
from .precedence import unique_family_positions
from synthesis_extraction.dependency.schedule import ScheduleLattice

log = logging.getLogger(__name__)

FORCED = "forced"
CONVENTIONAL = "conventional"
ANTI = "anti"
FREE = "free"
UNKNOWN_EVIDENCE = "unknown"

#: A pair counts as forced only when the partial order leaves no alternative at all.
FORCED_EPS = 1e-9
#: Directional frequency in the literature above which an ordering counts as a convention.
MIN_STRENGTH = 0.8
#: Independent route skeletons required before the literature is allowed to speak.
MIN_EFFECTIVE = 10


# ---------------------------------------------------------------------------
# The corpus table
# ---------------------------------------------------------------------------
def load_table(path: str) -> Dict[Tuple[str, str], dict]:
    """``{(family_a, family_b): row}`` from a ``precedence.json``, keys canonically sorted."""
    with open(path) as fh:
        data = json.load(fh)
    return {(p["family_a"], p["family_b"]): p for p in data["pairs"]}


def _lookup(table: Dict[Tuple[str, str], dict], fam_earlier: str, fam_later: str
            ) -> Tuple[Optional[dict], Optional[float]]:
    """The table row for this pair, and the literature's support **for this route's direction**.

    Table pairs are stored in canonical alphabetical order, so a route running them the other
    way needs the frequency flipped.  Reporting the unflipped number against a flipped
    observation is the kind of error that reads as a contradiction rather than a bug.
    """
    key = (fam_earlier, fam_later)
    row = table.get(key)
    if row is not None:
        return row, row["observed"] / row["n_routes"]
    row = table.get((fam_later, fam_earlier))
    if row is not None:
        return row, 1.0 - row["observed"] / row["n_routes"]
    return None, None


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------
@dataclass
class PairRigidity:
    """One ordered pair of transformations, as this route runs them."""

    step_earlier: int
    step_later: int
    family_earlier: str
    family_later: str
    verdict: str
    forced_prob: float                    # P(earlier before later) under this route's own order
    lit_strength: Optional[float] = None  # fraction of literature routes running it this way
    lit_n_routes: Optional[int] = None
    lit_n_effective: Optional[int] = None
    lit_explained: Optional[float] = None  # how much of the literature preference chemistry explains
    # Whether each step's transformation could be named unambiguously. A step whose family
    # repeats in the route has no single ordering event, so the literature cannot be consulted
    # for it — and its family label must not be printed as if it were known.
    earlier_named: bool = True
    later_named: bool = True

    @property
    def movable(self) -> bool:
        return self.verdict != FORCED


@dataclass
class RigidityMap:
    route_id: str
    tier: str
    n_steps: int
    pairs: List[PairRigidity] = field(default_factory=list)
    n_orderings: int = 0                  # valid orderings under chemistry alone
    n_orderings_conventional: int = 0     # ...and additionally preserving every convention
    skipped_repeat_families: int = 0

    def counts(self) -> Dict[str, int]:
        out = {v: 0 for v in (FORCED, CONVENTIONAL, ANTI, FREE, UNKNOWN_EVIDENCE)}
        for p in self.pairs:
            out[p.verdict] += 1
        return out

    def can_be_last(self) -> Dict[int, str]:
        """Per step: could it be moved to the end of the synthesis?

        The diversification question — which transformation could become the final step, so a
        library can be branched there.  ``no`` means the partial order forbids it; ``unprecedented``
        means nothing forbids it but the literature consistently does not; ``yes`` means it is
        free on both counts.
        """
        blocked_hard: set = set()
        blocked_soft: set = set()
        for p in self.pairs:
            if p.verdict == FORCED:
                blocked_hard.add(p.step_earlier)
            elif p.verdict in (CONVENTIONAL, ANTI):
                blocked_soft.add(p.step_earlier)
        out: Dict[int, str] = {}
        for s in {p.step_earlier for p in self.pairs} | {p.step_later for p in self.pairs}:
            if s in blocked_hard:
                out[s] = "no"
            elif s in blocked_soft:
                out[s] = "unprecedented"
            else:
                out[s] = "yes"
        return out


def rigidity_map(genome: Genome, table: Dict[Tuple[str, str], dict], *,
                 tier: str = "exposure", min_strength: float = MIN_STRENGTH,
                 min_effective: int = MIN_EFFECTIVE) -> RigidityMap:
    """Annotate every orderable pair of transformations in one route."""
    # Every *step* pair is mapped, not every family pair.  Whether the partial order forces an
    # ordering is a fact about steps and needs no transformation identity at all, so the physics
    # half is answerable for all of them; only the literature half needs an unambiguous family
    # on both sides.  Restricting the whole map to nameable pairs threw away most of a route —
    # a 7-step route yielded 1 mapped pair of 21 — and with it every ``forced`` verdict.
    pos = unique_family_positions(genome)
    nameable = {s: f for f, s in pos.items()}
    fam_of = dict(zip(genome.step_ids, genome.families))
    order = {s: i for i, s in enumerate(genome.step_ids)}
    named = len({f for f in genome.families if f != UNKNOWN})

    try:
        prec = lattice_for_genome(genome, tier).precedence_probabilities()
    except ValueError:                      # lattice beyond the exact-pairwise guard
        prec = {}

    out = RigidityMap(route_id=genome.route_id, tier=tier, n_steps=genome.n_steps,
                      skipped_repeat_families=max(0, named - len(pos)))

    for s1, s2 in itertools.combinations(genome.step_ids, 2):
        # orient by what this route actually does
        se, sl = (s1, s2) if order[s1] < order[s2] else (s2, s1)
        fe = nameable.get(se) or fam_of.get(se, UNKNOWN)
        fl = nameable.get(sl) or fam_of.get(sl, UNKNOWN)

        p_forced = prec.get((se, sl), 0.5)
        row = strength = n_eff = None
        if se in nameable and sl in nameable:
            row, strength = _lookup(table, fe, fl)
            n_eff = row.get("n_effective") if row else None

        if p_forced >= 1.0 - FORCED_EPS:
            verdict = FORCED
        elif row is None or (n_eff or 0) < min_effective or strength is None:
            verdict = UNKNOWN_EVIDENCE
        elif strength >= min_strength:
            verdict = CONVENTIONAL
        elif strength <= 1.0 - min_strength:
            verdict = ANTI
        else:
            verdict = FREE

        out.pairs.append(PairRigidity(
            step_earlier=se, step_later=sl, family_earlier=fe, family_later=fl,
            verdict=verdict, forced_prob=p_forced, lit_strength=strength,
            lit_n_routes=row["n_routes"] if row else None, lit_n_effective=n_eff,
            lit_explained=(row["explained"].get(tier) if row and row.get("explained") else None),
            earlier_named=se in nameable, later_named=sl in nameable,
        ))

    constraints = list(genome.constraints.get(tier, ()))
    out.n_orderings = ScheduleLattice(genome.step_ids, constraints).count()
    out.n_orderings_conventional = ScheduleLattice(
        genome.step_ids, constraints + conventional_constraints(out)).count()
    return out


# ---------------------------------------------------------------------------
# Feeding the rearrangement engine
# ---------------------------------------------------------------------------
def conventional_constraints(m: RigidityMap) -> List[Tuple[int, int]]:
    """Extra ordering constraints that hold every convention this route already follows.

    Drops straight into ``lattice_for(dep, extra_constraints=...)``, so the existing engine
    enumerates only rearrangements that keep the literature's conventions intact and vary the
    genuinely free dimensions.  ``anti`` pairs are deliberately *not* locked: this route already
    breaks that convention, so freezing it would preserve an unusual choice as if it were a rule.
    """
    return [(p.step_earlier, p.step_later) for p in m.pairs if p.verdict == CONVENTIONAL]


def free_pairs(m: RigidityMap) -> List[PairRigidity]:
    """Pairs worth reordering: movable, and with no convention arguing either way."""
    return [p for p in m.pairs if p.verdict in (FREE, UNKNOWN_EVIDENCE)]


def convention_breaking_pairs(m: RigidityMap) -> List[PairRigidity]:
    """Swaps that would break a real convention while remaining chemically allowed.

    The novelty candidates — "permitted, but nobody does it" — which is the class the
    rearrangement engine exists to surface.
    """
    return [p for p in m.pairs if p.verdict == CONVENTIONAL]


# ---------------------------------------------------------------------------
# Corpus survey — what the map actually yields, and where it runs out
# ---------------------------------------------------------------------------
def survey(genomes: Sequence[Genome], table: Dict[Tuple[str, str], dict], *,
           tier: str = "exposure", min_strength: float = MIN_STRENGTH,
           min_effective: int = MIN_EFFECTIVE) -> dict:
    """Run the map over many routes and report yield *and* why it runs out.

    Two different denominators matter and quoting only one is misleading.  Against **step**
    pairs the map looks sparse, but most step pairs are not transformation pairs at all: a
    step has an identifiable transformation — a step with no detectable bond change (unmapped
    reagents, salt forms, pure stereochemistry) has nothing to look up.  Against
    **transformation** pairs — the ones that do have an identity — the coverage is far higher.
    Both are reported, along with the split of what stopped the rest.
    """
    verd: Dict[str, int] = {}
    reasons = {"no transformation identity": 0, "answered": 0,
               "below evidence floor": 0, "pair absent from table": 0}
    prunes: List[float] = []
    routes = identified_tot = steps_tot = with_candidate = with_anti = 0

    for g in genomes:
        m = rigidity_map(g, table, tier=tier, min_strength=min_strength,
                         min_effective=min_effective)
        routes += 1
        steps_tot += g.n_steps
        identified_tot += sum(1 for f in g.families if f != UNKNOWN)
        c = m.counts()
        for k, v in c.items():
            verd[k] = verd.get(k, 0) + v
        if c[CONVENTIONAL]:
            with_candidate += 1
        if c[ANTI]:
            with_anti += 1
        if m.n_orderings:
            prunes.append(1.0 - m.n_orderings_conventional / m.n_orderings)
        for p in m.pairs:
            if p.verdict == FORCED:
                continue
            if not (p.earlier_named and p.later_named):
                reasons["no transformation identity"] += 1
            elif p.lit_strength is None:
                reasons["pair absent from table"] += 1
            elif (p.lit_n_effective or 0) < min_effective:
                reasons["below evidence floor"] += 1
            else:
                reasons["answered"] += 1

    prunes.sort()
    nameable = reasons["answered"] + reasons["below evidence floor"] + \
        reasons["pair absent from table"]
    return {
        "routes": routes, "tier": tier,
        "steps_per_route": steps_tot / routes if routes else 0,
        "transformations_per_route": identified_tot / routes if routes else 0,
        "verdicts": verd,
        "movable_pair_reasons": reasons,
        "answered_of_transformation_pairs": (reasons["answered"] / nameable) if nameable else None,
        "routes_with_convention_candidate": with_candidate,
        "routes_already_anti_conventional": with_anti,
        "prune_median": prunes[len(prunes) // 2] if prunes else None,
        "prune_mean": (sum(prunes) / len(prunes)) if prunes else None,
        "routes_pruned_none": sum(1 for p in prunes if p < 1e-9),
    }


def format_survey(s: dict) -> str:
    v = s["verdicts"]
    tot = sum(v.values()) or 1
    spr, tpr = s["steps_per_route"], s["transformations_per_route"]
    lines = [f"rigidity survey — {s['routes']:,} routes, tier '{s['tier']}'", "",
             f"  {spr:.1f} steps per route, {tpr:.1f} identified transformations "
             f"({tpr / spr:.0%} of steps)" if spr else ""]
    # Only blame the segmentation when it is actually the cause: with per-step identity the
    # shortfall is unmapped or unchanged steps, not centres merging several steps into one.
    lines += ["  (the shortfall is steps with no detectable bond change — unmapped reagents,",
              "   salt forms, pure stereochemistry)"]
    lines += ["", "  step-pair verdicts:"]
    for k, n in sorted(v.items(), key=lambda kv: -kv[1]):
        lines.append(f"      {k:<14}{n:>8,}  ({n / tot:.1%})")
    lines += ["", "  of movable pairs, what decided them:"]
    r = s["movable_pair_reasons"]
    rt = sum(r.values()) or 1
    for k, n in sorted(r.items(), key=lambda kv: -kv[1]):
        lines.append(f"      {k:<28}{n:>8,}  ({n / rt:.1%})")
    if s["answered_of_transformation_pairs"] is not None:
        lines += ["", f"  -> where a transformation pair exists, the literature answers "
                      f"{s['answered_of_transformation_pairs']:.0%} of the time.",
                  "     The sparse-looking numbers above are a segmentation limit, not a",
                  "     shortage of corpus evidence."]
    lines += ["",
              f"  routes with >=1 convention-breaking candidate: {s['routes_with_convention_candidate']:,}",
              f"  routes already breaking a convention:          {s['routes_already_anti_conventional']:,}"]
    if s["prune_median"] is not None:
        lines += ["",
                  f"  search-space pruning from locking conventions: median "
                  f"{s['prune_median']:.0%}, mean {s['prune_mean']:.0%} "
                  f"({s['routes_pruned_none']:,} routes pruned by nothing)"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render(family: str, envs: Dict[str, str], width: int = 34) -> str:
    s = envs.get(family) or family
    return s if len(s) <= width else s[: width - 1] + "…"


_EXPLAIN = {
    FORCED: "chemistry requires it",
    CONVENTIONAL: "allowed to move; literature does not",
    ANTI: "this route already breaks the literature's preference",
    FREE: "movable, literature has no preference",
    UNKNOWN_EVIDENCE: "movable, too little independent evidence to say",
}


def format_map(m: RigidityMap, envs: Optional[Dict[str, str]] = None,
               step_labels: Optional[Dict[int, str]] = None) -> str:
    envs = envs or {}
    step_labels = step_labels or {}
    c = m.counts()
    lines = [
        f"rigidity map — route {m.route_id} ({m.n_steps} steps, tier '{m.tier}')", "",
        f"  valid orderings under chemistry alone:        {m.n_orderings:,}",
        f"  ...also preserving every literature convention: {m.n_orderings_conventional:,}",
    ]
    if m.n_orderings:
        lines.append(f"  -> conventions prune the search space by "
                     f"{1 - m.n_orderings_conventional / m.n_orderings:.1%}")
    lines += ["", "  pair verdicts: " + ", ".join(f"{k}={v}" for k, v in c.items() if v), ""]

    for verdict in (FORCED, CONVENTIONAL, ANTI, FREE, UNKNOWN_EVIDENCE):
        rows = [p for p in m.pairs if p.verdict == verdict]
        if not rows:
            continue
        lines.append(f"  [{verdict}] — {_EXPLAIN[verdict]}")
        for p in rows:
            e = step_labels.get(p.step_earlier) or (
                _render(p.family_earlier, envs) if p.earlier_named else "(family ambiguous)")
            l = step_labels.get(p.step_later) or (
                _render(p.family_later, envs) if p.later_named else "(family ambiguous)")
            detail = ""
            if p.lit_strength is not None:
                detail = (f"lit {p.lit_strength:.0%} this way "
                          f"({p.lit_n_effective} independent skeletons"
                          f" of {p.lit_n_routes} routes)")
                if p.lit_explained is not None:
                    detail += f", chemistry explains {p.lit_explained:.0%}"
            elif verdict == FORCED:
                detail = "partial order: P = 1.00"
            lines.append(f"      step {p.step_earlier} -> {p.step_later}   {e}  BEFORE  {l}")
            if detail:
                lines.append(f"          {detail}")
        lines.append("")

    last = m.can_be_last()
    yes = [s for s, v in last.items() if v == "yes"]
    unp = [s for s, v in last.items() if v == "unprecedented"]
    lines += [
        "  which step could become the final (diversification) step?",
        f"      freely: {sorted(yes, reverse=True) or 'none'}",
        f"      allowed but unprecedented: {sorted(unp, reverse=True) or 'none'}",
    ]
    if m.skipped_repeat_families:
        lines.append(f"\n  note: {m.skipped_repeat_families} transformation(s) occur more than "
                     f"once and have no single ordering event, so they are not mapped.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _step_labels(corpus: Optional[str], route_id: str, genome: Genome) -> Dict[int, str]:
    """``{step_id: product SMILES}`` by streaming the corpus for one route.

    Atom maps are stripped before truncation.  A mapped route SMILES begins with a long
    ``[O:10001]=[C:10002](...`` prefix that is identical for every step, so truncating the raw
    string labels all nine steps the same way — informative-looking and completely useless.
    """
    if not corpus:
        return {}
    from rdkit import Chem, RDLogger
    from synthesis_extraction.load_trees import iter_trees
    from synthesis_extraction.dependency.route_graph import build_route_graph
    RDLogger.DisableLog("rdApp.*")

    def _clean(smi: str) -> str:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi[:40]
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        s = Chem.MolToSmiles(m)
        return s if len(s) <= 40 else s[:39] + "…"

    for tid, tg in iter_trees(corpus):
        if tid != route_id:
            continue
        full = build_route_graph(tg, tid)
        if full is None:
            return {}
        return {int(n["id"]): _clean(str(n.get("SMILES", "")).split(">")[-1])
                for n in full.get("nodes", [])}
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--genomes", required=True)
    ap.add_argument("--precedence", required=True)
    ap.add_argument("--route", default=None, help="tree id; default = the first long route")
    ap.add_argument("--tier", default="exposure")
    ap.add_argument("--corpus", default=None, help="trees.jsonl, to label steps with SMILES")
    ap.add_argument("--min-strength", type=float, default=MIN_STRENGTH)
    ap.add_argument("--min-effective", type=int, default=MIN_EFFECTIVE)
    ap.add_argument("--min-steps", type=int, default=6, help="when picking a route automatically")
    ap.add_argument("--survey", type=int, default=0, metavar="N",
                    help="instead of one route, survey N routes and report yield and limits")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.survey:
        table = load_table(args.precedence)
        picked = []
        with open(args.genomes) as fh:
            for line in fh:
                if not line.strip():
                    continue
                g = Genome.from_dict(json.loads(line))
                if g.n_steps >= args.min_steps:
                    picked.append(g)
                if len(picked) >= args.survey:
                    break
        s = survey(picked, table, tier=args.tier, min_strength=args.min_strength,
                   min_effective=args.min_effective)
        print(format_survey(s))
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(s, fh, indent=1)
            print(f"\n-> {args.out}")
        return 0

    genome = None
    with open(args.genomes) as fh:
        for line in fh:
            if not line.strip():
                continue
            g = Genome.from_dict(json.loads(line))
            if args.route:
                if g.route_id == args.route:
                    genome = g
                    break
            elif g.n_steps >= args.min_steps:
                genome = g
                break
    if genome is None:
        ap.error(f"route {args.route!r} not found in {args.genomes}")

    table = load_table(args.precedence)
    m = rigidity_map(genome, table, tier=args.tier, min_strength=args.min_strength,
                     min_effective=args.min_effective)

    print(format_map(m, {}, _step_labels(args.corpus, genome.route_id, genome)))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(asdict(m), fh, indent=1)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
