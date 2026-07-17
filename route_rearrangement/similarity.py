"""Route-to-route similarity — surface the genuinely *distinct* rearrangements.

Every rearrangement of one literature route shares the same target, building blocks and set
of reactions; what differs is the **order**, and therefore the actual intermediate molecules
threaded through.  To surface distinct alternatives (rather than a page of near-identical
reorderings) we want the top-k routes that are **both far from the original literature route
and far from each other** — a diverse spread, not five variants of the same idea.

Distance is the **tree-edit distance (TED)** between two synthesis trees, via MolecularAI
``rxnutils`` (`rxnutils.routes.comparison`, the AiZynthFinder route metric).  We compare in
``molecules`` mode — structural overlap of the intermediates — which needs no atom mapping
(our rearranged routes are map-free) and is ~20 ms per pair.

Selection is greedy farthest-first (max-min): seed the "chosen" set with the original, then
repeatedly add the rearrangement whose minimum distance to everything already chosen (the
original + picks so far) is largest.  The first pick is the most different from literature;
each subsequent pick is pushed away from both literature and the earlier picks.  Only the
distances greedy actually needs are computed (≈ k·pool, not the full O(n²) matrix), over a
pool capped to the routes most different from the original so cost stays bounded.

If ``rxnutils``/``apted`` are unavailable the module transparently falls back to a Jaccard
distance over the canonical intermediate SMILES sets, so the pipeline never crashes; the
method used is recorded on each route.  Higher ``distance_to_original`` = more different.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem

from .metrics.base import intermediates, reactions

TED_CONTENT = "molecules"


def available() -> bool:
    try:
        import apted  # noqa: F401
        from rxnutils.routes.comparison import ted_distances_calculator  # noqa: F401
        return True
    except Exception:
        return False


# -- rxnutils tree construction --------------------------------------------------------

def build_synthesis_route(record: dict):
    """An ``rxnutils.routes.base.SynthesisRoute`` (AiZynthFinder ``mol``/``reaction`` tree)
    for one materialized route, built from its per-step reactions."""
    from rxnutils.routes.base import SynthesisRoute

    rxns = reactions(record)
    by_product = {r.product: r for r in rxns}

    def mol(smiles: str, seen: frozenset) -> dict:
        r = by_product.get(smiles)
        if r is None or smiles in seen:               # purchasable / already-on-path leaf
            return {"type": "mol", "smiles": smiles, "in_stock": True}
        seen = seen | {smiles}
        rxn = {"type": "reaction",
               "smiles": ".".join(r.reactants) + ">>" + r.product,
               "metadata": {},
               "children": [mol(p, seen) for p in r.reactants]}
        return {"type": "mol", "smiles": smiles, "in_stock": False, "children": [rxn]}

    if not rxns:
        return SynthesisRoute({"type": "mol", "smiles": record.get("target", ""),
                               "in_stock": True})
    top = max(rxns, key=lambda r: r.position)
    return SynthesisRoute(mol(top.product, frozenset()))


# -- fallback distance (no rxnutils) ---------------------------------------------------

@lru_cache(maxsize=200_000)
def _canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else smi


def _intermediate_set(record: dict) -> frozenset:
    # the transient intermediates (drop the shared final target) as canonical SMILES
    return frozenset(_canon(s) for s in intermediates(record)[:-1])


def _jaccard_distance(a: dict, b: dict) -> float:
    sa, sb = _intermediate_set(a), _intermediate_set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return 1.0 - (inter / union if union else 1.0)


# -- distances from the reference (original) route -------------------------------------

class _DistanceProvider:
    """Lazily-memoized pairwise route distance over a fixed record list.

    rxnutils TED is used for the pair only when **both** endpoints are in ``ted_indices`` (the
    bounded pool); every other pair uses the instant Jaccard distance over intermediate sets.
    This keeps the expensive TED work to ~O(k·pool) regardless of how many routes a tree has.
    Tree wrappers are built lazily and cached.  ``method`` names the scheme in force.
    """

    def __init__(self, records: Sequence[dict], ted_indices: Optional[Sequence[int]] = None):
        self.records = records
        self._cache: Dict[Tuple[int, int], float] = {}
        self._wrappers: Dict[int, object] = {}
        self._ted = set(ted_indices) if ted_indices is not None else set(range(len(records)))
        self._use_ted = available()
        self.method = f"ted-{TED_CONTENT}" if self._use_ted else "jaccard-intermediates"

    def _wrapper(self, i: int):
        if i not in self._wrappers:
            from rxnutils.routes.ted.reactiontree import ReactionTreeWrapper
            self._wrappers[i] = ReactionTreeWrapper(build_synthesis_route(self.records[i]),
                                                    TED_CONTENT)
        return self._wrappers[i]

    def distance(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        key = (i, j) if i < j else (j, i)
        if key not in self._cache:
            self._cache[key] = self._compute(key[0], key[1])
        return self._cache[key]

    def _compute(self, i: int, j: int) -> float:
        if self._use_ted and i in self._ted and j in self._ted:
            try:
                return float(self._wrapper(i).distance_to(self._wrapper(j)))
            except Exception:
                pass
        return _jaccard_distance(self.records[i], self.records[j])


def _reference_index(records: Sequence[dict]) -> Optional[int]:
    """Index of the original literature ordering; falls back to the lexicographically
    smallest ordering if no route is flagged original."""
    for i, r in enumerate(records):
        if r.get("is_original_order"):
            return i
    if not records:
        return None
    return min(range(len(records)), key=lambda i: records[i].get("ordering", []))


def _greedy_farthest_first(pool: List[int], ref: int, k: int,
                           prov: _DistanceProvider) -> List[int]:
    """Pick ``k`` indices from ``pool`` maximising the minimum distance to the already-chosen
    set (seeded with ``ref``): far from literature and from each other."""
    pool = list(pool)
    selected: List[int] = []
    anchors = [ref]
    while pool and len(selected) < k:
        best = max(pool, key=lambda c: min(prov.distance(c, a) for a in anchors))
        selected.append(best)
        anchors.append(best)
        pool.remove(best)
    return selected


def annotate_distinctness(records: Sequence[dict], k: int = 5,
                          ted_cap: int = 60) -> Tuple[List[int], str]:
    """Attach a ``similarity`` block to every record in place and return the diverse top-``k``.

    Per record::

        {"distance_to_original": float, "method": str,
         "rank_most_different": int|None,   # 1 = farthest from literature (by distance)
         "diverse_rank": int|None}          # 1..k = greedy farthest-first pick order, else None

    ``diverse_rank`` is what the GUI and stats prioritise: the top-``k`` routes that are far
    from the literature route *and* spread apart from one another.  A cheap Jaccard prefilter
    ranks all routes, then rxnutils TED is applied only to the ``ted_cap`` most-different pool
    (plus the original), so cost is bounded regardless of the enumeration size.  No-op-safe
    for a single route.  Returns ``(diverse_top_k_indices, method)``.
    """
    ref = _reference_index(records)
    if ref is None:
        return [], "none"

    # instant Jaccard prefilter over intermediate sets → the pool worth spending TED on
    others = [i for i in range(len(records)) if i != ref]
    jac = {i: _jaccard_distance(records[ref], records[i]) for i in others}
    pool = sorted(others, key=lambda i: jac[i], reverse=True)[:max(k, ted_cap)]

    prov = _DistanceProvider(records, ted_indices=[ref] + pool)
    dists = [0.0 if i == ref else (prov.distance(ref, i) if i in set(pool) else jac[i])
             for i in range(len(records))]

    raw_order = sorted(others, key=lambda i: dists[i], reverse=True)
    raw_rank = {idx: r for r, idx in enumerate(raw_order, start=1)}

    diverse = _greedy_farthest_first(pool, ref, k, prov)
    diverse_rank = {idx: r for r, idx in enumerate(diverse, start=1)}

    for i, rec in enumerate(records):
        rec["similarity"] = {
            "distance_to_original": round(dists[i], 4),
            "method": prov.method,
            "rank_most_different": None if i == ref else raw_rank.get(i),
            "diverse_rank": diverse_rank.get(i),
        }
    return diverse, prov.method


def most_different(records: Sequence[dict], k: int = 5) -> List[int]:
    """The diverse top-``k`` indices (far from literature and from each other).  Reads the
    stored ``diverse_rank`` annotation if already present, else computes it."""
    if not records:
        return []
    ranked = [(r["similarity"]["diverse_rank"], i) for i, r in enumerate(records)
              if r.get("similarity", {}).get("diverse_rank") is not None]
    if not ranked:
        diverse, _ = annotate_distinctness(records, k)
        return diverse
    return [i for _, i in sorted(ranked)][:k]
