"""The cluster model, transferred from approximate common intervals.

Jahn, Winter, Stoye and Böcker (2013) define a **reference cluster** as a set of gene families ``C`` with
``|C| >= s`` that has an exact occurrence in one genome and *δ-locations* in at least ``k'-1``
others.  A δ-location is an interval of some genome whose family content ``C'`` is within
symmetric set distance ``D(C, C') = |C\\C'| + |C'\\C| <= δ`` of the reference — insertions plus
deletions.  Tolerating δ is what lets a conserved block be recognised when a genome has an
extra gene wedged into it or is missing one.

Everything in that definition transfers without strain, and the tolerance is not a nicety here
but the point: the canonical rigid block in synthesis is protect → react → deprotect, and the
whole reason to protect is that *other steps happen inside the bracket*.  A model demanding
exact contiguity would miss every real instance of the one block we are most confident about.

Two properties of the reference-cluster formulation matter downstream:

* The reference set is anchored to an **exact occurrence**, so a cluster is always a set of
  families that really did appear together somewhere, not a synthetic combination.
* Occurrences are **intervals** — contiguous windows of the route.  This is what makes the
  statistic sensitive to *order*, and therefore what the two permutation nulls can speak to: a
  block that must stay together will keep producing tight intervals under Null-C, while one
  held together only by convention will scatter.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .corpus import UNKNOWN, Genome

FamilySet = FrozenSet[str]


def set_distance(c: FamilySet, other: Iterable[str]) -> int:
    """``D(C, C') = |C\\C'| + |C'\\C|`` — insertions plus deletions, the paper's distance."""
    o = frozenset(other)
    return len(c - o) + len(o - c)


def required_members(c: FamilySet, delta: int) -> int:
    """How many of *C*'s families a window must contain to be a δ-location.

    ``|C| - delta`` is the naive answer and it degenerates badly at the small cluster sizes
    synthesis works with: a pair at δ=1 would need just *one* of its two families, so the test
    becomes "does this route contain either A or B", which every route passes.  The floor of two
    keeps a cluster a statement about co-location rather than presence.
    """
    if len(c) < 2:
        return 1
    return max(2, len(c) - delta)


@dataclass(frozen=True)
class Occurrence:
    """A δ-location: an interval of one route and how far its content is from the reference."""

    route_id: str
    start: int          # index into the route's family string (synthesis order)
    end: int            # exclusive
    distance: int       # D(C, C')

    @property
    def width(self) -> int:
        return self.end - self.start


def delta_locations(families: Sequence[str], c: FamilySet, delta: int,
                    route_id: str = "", max_extra: int = 2) -> List[Occurrence]:
    """Best δ-location of *c* in one route, per starting position.

    Intervals are searched up to ``|C| + max_extra + delta`` wide: a δ-location may contain
    foreign steps (that is the point of tolerating insertions), but an unbounded window would
    eventually swallow the whole route and match everything.  Only the *best* (smallest
    distance, then narrowest) interval at each start is kept, so one occurrence is not counted
    many times at nested widths.
    """
    n = len(families)
    if not c or n == 0:
        return []
    # A δ-location may be *narrower* than |C| when the tolerance is spent on deletions: with
    # C = {A,B,Z} and δ=1, the two-wide window "AB" is a valid location (one deletion). Starting
    # the search at |C| would miss exactly the approximate matches the model exists to catch.
    need = required_members(c, delta)
    lo, hi = need, len(c) + max_extra + delta
    out: List[Occurrence] = []
    for start in range(n):
        best: Optional[Occurrence] = None
        for width in range(lo, min(hi, n - start) + 1):
            window = families[start:start + width]
            present = {f for f in window if f != UNKNOWN}
            if len(c & present) < need:
                continue
            d = set_distance(c, present)
            if d > delta:
                continue
            if best is None or d < best.distance or (d == best.distance and width < best.width):
                best = Occurrence(route_id, start, start + width, d)
        if best is not None:
            out.append(best)
    return out


def _dominant(occurrences: Sequence[Occurrence]) -> List[Occurrence]:
    """One occurrence per route — the best — so quorum counts routes, not windows."""
    best: Dict[str, Occurrence] = {}
    for o in occurrences:
        cur = best.get(o.route_id)
        if cur is None or o.distance < cur.distance or (
                o.distance == cur.distance and o.width < cur.width):
            best[o.route_id] = o
    return list(best.values())


@dataclass
class Cluster:
    """A reference cluster and where it occurs."""

    families: FamilySet
    reference_route: str
    occurrences: List[Occurrence] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.families)

    @property
    def quorum(self) -> int:
        """Number of distinct routes with a δ-location — the paper's ``k'``."""
        return len({o.route_id for o in self.occurrences})

    @property
    def exact_quorum(self) -> int:
        return len({o.route_id for o in self.occurrences if o.distance == 0})

    def key(self) -> Tuple[str, ...]:
        return tuple(sorted(self.families))


def candidate_clusters(genomes: Sequence[Genome], *, s: int = 2, max_size: int = 4,
                       min_route_support: int = 2) -> Dict[FamilySet, str]:
    """Reference sets anchored to exact occurrences: ``{family set: reference route}``.

    Candidates are the family sets of contiguous windows actually present in some route — the
    "exact occurrence in one of the genomes" the reference-cluster model requires.  Sets that
    appear as a window in fewer than *min_route_support* routes are dropped up front: they can
    never reach quorum, and enumerating their δ-locations across the corpus is where the cost
    would otherwise go.
    """
    seen: Dict[FamilySet, Set[str]] = defaultdict(set)
    for g in genomes:
        fam = g.families
        n = len(fam)
        for start in range(n):
            for width in range(s, min(max_size, n - start) + 1):
                window = [f for f in fam[start:start + width] if f != UNKNOWN]
                fs = frozenset(window)
                if len(fs) < s:                 # repeated symbols do not make a bigger set
                    continue
                seen[fs].add(g.route_id)
    return {fs: sorted(routes)[0] for fs, routes in seen.items()
            if len(routes) >= min_route_support}


def find_clusters(genomes: Sequence[Genome], *, s: int = 2, max_size: int = 4,
                  delta: int = 1, quorum: int = 3,
                  min_route_support: int = 2) -> List[Cluster]:
    """Reference clusters meeting the quorum, with their δ-locations.

    Mirrors the source pipeline's ordering: enumerate reference sets, locate them approximately
    across the corpus, then keep those recurring in at least ``quorum`` routes.  Significance is
    a separate stage (:mod:`.significance`) — detection here is deliberately permissive, since
    a detector that pre-filters on strength would bias the p-values it is later handed.
    """
    candidates = candidate_clusters(genomes, s=s, max_size=max_size,
                                    min_route_support=min_route_support)
    # index routes by family so a candidate only visits routes that could contain it
    by_family: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(genomes):
        for f in set(g.families):
            if f != UNKNOWN:
                by_family[f].append(i)

    out: List[Cluster] = []
    for fs, ref in candidates.items():
        need = required_members(fs, delta)
        counts: Counter = Counter()
        for f in fs:
            for i in by_family.get(f, ()):
                counts[i] += 1
        occ: List[Occurrence] = []
        for i, have in counts.items():
            if have < need:
                continue
            g = genomes[i]
            occ.extend(delta_locations(g.families, fs, delta, route_id=g.route_id))
        occ = _dominant(occ)
        if len({o.route_id for o in occ}) >= quorum:
            out.append(Cluster(families=fs, reference_route=ref, occurrences=occ))
    return out


def deduplicate(clusters: Sequence[Cluster]) -> List[Cluster]:
    """Drop a cluster whose family set is contained in a larger one with the same quorum.

    The source pipeline merges duplicates and reports only the strongest; without this, every
    subset of a real block is reported alongside it and the cluster count is meaningless.
    """
    by_size = sorted(clusters, key=lambda c: (-c.size, -c.quorum))
    kept: List[Cluster] = []
    for c in by_size:
        if any(c.families < k.families and c.quorum <= k.quorum for k in kept):
            continue
        kept.append(c)
    return kept
