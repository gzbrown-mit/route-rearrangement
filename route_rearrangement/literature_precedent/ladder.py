"""The abstraction ladder: exact template (finest) up to reaction class (coarsest).

A pair statistic needs two things that pull in opposite directions.  **Precision**: the key
identifying a transformation must not pool chemically different reactions, or the order
statistic averages unrelated things.  **Recurrence**: the same key pair must be seen often
enough for "A before B" to be measurable at all — and at exact-template resolution it is not.
``synthesis_extraction.transformation.pattern_key`` records the number: the FrequenTree ladder
gives ~21k distinct keys over 457k routes, so the median pattern *pair* is seen about twice.

So we do not choose one abstraction.  We keep the whole ladder, compute the statistic at every
rung, and let each pair resolve at the finest rung that can actually support it
(:func:`resolve`).  A pair with 400 observations answers at exact-template resolution; a pair
with 3 backs off to reaction class and says something weaker but true.  Every reported number
carries the rung it came from.

The rungs are **FrequenTree's own**, not invented here: ``TEMPLATE_LADDER`` in
``transformation.fc_adapter`` supplies the five template granularities in its own
specific-to-general order, and ``transformation.pattern_key.synthon_key`` supplies two coarser
reaction-class rungs above them.  ``radius`` is *not* a rung — it controls how far context
extends from the reaction centre at extraction time and is orthogonal to specificity, which is
why it is swept separately (:mod:`.sweep`).

Cross-rung disagreement is signal, not noise.  A pair locked in one order at ``template_exact``
but freely commutable at ``synthon_shell0`` means the constraint is *substrate-specific* rather
than general to the reaction class — a statement worth making, and one nearest-precedent
retrieval cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .. import deps  # noqa: F401
from synthesis_extraction.transformation.fc_adapter import TEMPLATE_LADDER, ContextualCenter


@dataclass(frozen=True)
class Rung:
    """One level of abstraction.  ``index`` 0 is the finest; larger is coarser."""

    index: int
    name: str
    source: str          # "template" | "synthon"
    param: int           # TEMPLATE_LADDER index, or synthon shell
    description: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


# Finest -> coarsest.  The five template rungs are TEMPLATE_LADDER in its own order, so the
# ordering claim is FrequenTree's, not ours; the two synthon rungs sit above the most general
# template rung because ``synthon_key`` is derived *from* it by dropping further detail.
RUNGS: Sequence[Rung] = (
    Rung(0, "template_exact", "template", 0,
         description="specific/smarts — full degree, H-count and charge specification"),
    Rung(1, "template_abs", "template", 1,
         description="specific/abs_smarts — curated functional groups abstracted"),
    Rung(2, "template_noDH", "template", 2,
         description="specific/noDHsmarts — literal atoms, degree/H constraints dropped"),
    Rung(3, "template_abs_noDH", "template", 3,
         description="specific/abs_noDHsmarts — abstracted FGs and no degree/H"),
    Rung(4, "roots_abs_noDH", "template", 4,
         description="roots/abs_noDHsmarts — most general FrequenTree rung"),
    Rung(5, "synthon_shell1", "synthon", 1,
         description="reaction class plus one shell of functional context"),
    Rung(6, "synthon_shell0", "synthon", 0,
         description="reaction class — the reacting synthon alone"),
)

RUNG_BY_NAME: Dict[str, Rung] = {r.name: r for r in RUNGS}
RUNG_NAMES: List[str] = [r.name for r in RUNGS]

# Guard against an upstream ladder change silently remapping our rungs onto the wrong
# granularity: every template rung must still point at the tuple we documented above.
_EXPECTED_TEMPLATE_RUNGS = (("specific", "smarts"), ("specific", "abs_smarts"),
                            ("specific", "noDHsmarts"), ("specific", "abs_noDHsmarts"),
                            ("roots", "abs_noDHsmarts"))
if tuple(TEMPLATE_LADDER) != _EXPECTED_TEMPLATE_RUNGS:  # pragma: no cover - upstream drift
    raise RuntimeError(
        "synthesis_extraction TEMPLATE_LADDER changed: expected "
        f"{_EXPECTED_TEMPLATE_RUNGS}, got {tuple(TEMPLATE_LADDER)}. "
        "literature_precedent.ladder.RUNGS must be re-derived before any statistic is trusted.")


def key_at(center: ContextualCenter, rung: Rung) -> Optional[str]:
    """This center's transformation key at *rung*, or ``None`` if underivable.

    Template rungs go through :meth:`ContextualCenter.template_key`, which already rejects
    degenerate (empty ``>>``) templates and falls back along the ladder rather than dropping
    the center — so a center missing one granularity still gets a key, at a neighbouring
    specificity.  That fallback is why key counts are non-increasing *in aggregate* down the
    ladder but not guaranteed monotone for an individual center.
    """
    if rung.source == "template":
        return center.template_key(rung.param)
    from synthesis_extraction.transformation.pattern_key import synthon_key
    return synthon_key(center, shell=rung.param)


def key_fn_for(rung: Rung) -> Callable[[ContextualCenter], Optional[str]]:
    """A ``key_fn`` for ``OrderEvidenceTable.observe_route`` at *rung*."""
    return lambda c, _r=rung: key_at(c, _r)


def keys_for_center(center: ContextualCenter,
                    rungs: Sequence[Rung] = RUNGS) -> Dict[str, Optional[str]]:
    """``{rung_name: key}`` for every rung — one pass, so a corpus sweep costs one read.

    The synthon rungs are derived from the same roots-rung SMARTS, so computing all of them
    together is barely more expensive than computing one.
    """
    return {r.name: key_at(center, r) for r in rungs}


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------
@dataclass
class Resolved:
    """Which rung a pair's statistic was answered at, and why."""

    rung: Optional[str]          # None if no rung had support
    support: int                 # ordered observations at that rung
    n_routes: int                # distinct routes contributing
    backed_off: int              # how many finer rungs were skipped for lack of support
    starved: List[str]           # the skipped rung names, finest first


def resolve(support_by_rung: Dict[str, int],
            routes_by_rung: Optional[Dict[str, int]] = None,
            *, min_n: int = 30, min_routes: int = 5) -> Resolved:
    """Pick the finest rung with real support; report what was skipped.

    A rung qualifies when it has at least *min_n* strictly-ordered observations **and** at
    least *min_routes* distinct routes behind them.  The route floor is not redundant: one
    prolific route (or a cluster of near-duplicates, which PaRoutes has many of) can supply
    dozens of observations that are nowhere near dozens of independent facts.
    """
    routes_by_rung = routes_by_rung or {}
    starved: List[str] = []
    for r in RUNGS:
        n = int(support_by_rung.get(r.name, 0) or 0)
        n_routes = int(routes_by_rung.get(r.name, 0) or 0)
        if n >= min_n and (not routes_by_rung or n_routes >= min_routes):
            return Resolved(rung=r.name, support=n, n_routes=n_routes,
                            backed_off=len(starved), starved=starved)
        starved.append(r.name)
    return Resolved(rung=None, support=0, n_routes=0,
                    backed_off=len(starved), starved=starved)
