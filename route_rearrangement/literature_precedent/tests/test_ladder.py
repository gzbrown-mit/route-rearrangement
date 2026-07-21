"""The ladder must actually be a ladder, and the backoff must prefer the finest usable rung."""

import pytest

from route_rearrangement.literature_precedent import aggregate, ladder
from route_rearrangement.literature_precedent.tests.conftest import centers_required


def test_rungs_are_ordered_and_unique():
    assert [r.index for r in ladder.RUNGS] == list(range(len(ladder.RUNGS)))
    assert len({r.name for r in ladder.RUNGS}) == len(ladder.RUNGS)
    # the coarsest rung is the one backoff ends at; it must be a reaction-class rung
    assert ladder.RUNGS[-1].source == "synthon"
    assert ladder.RUNGS[0].name == "template_exact"


@centers_required
def test_key_count_is_non_increasing_down_the_ladder(centers_slice):
    """Coarser rungs must not distinguish *more* transformations than finer ones.

    Checked in aggregate over a real slice rather than per center: ``template_key`` falls back
    along the ladder when a granularity is missing, so an individual center can get its key
    from a neighbouring rung, and only the corpus-level counts are meaningfully monotone.
    """
    tables = aggregate.aggregate(iter(centers_slice))
    counts = [tables[r.name].n_keys for r in ladder.RUNGS]
    assert counts[0] > 0
    # allow the fallback to perturb adjacent template rungs slightly, but the ladder as a whole
    # must contract: the coarsest rung is strictly coarser than the finest.
    assert counts[-1] < counts[0]
    for finer, coarser in zip(counts, counts[1:]):
        assert coarser <= finer * 1.05, f"key count grew going coarser: {counts}"


@centers_required
def test_keys_are_stable_across_calls(centers_slice):
    _, centers = next((r for r in centers_slice if r[1]), (None, None))
    assert centers, "expected at least one route with centers"
    first = ladder.keys_for_center(centers[0])
    second = ladder.keys_for_center(centers[0])
    assert first == second
    assert any(v for v in first.values()), "no rung produced a key for a real center"


def test_resolve_picks_finest_rung_with_support():
    names = ladder.RUNG_NAMES
    support = {n: 0 for n in names}
    routes = {n: 0 for n in names}
    # only the two coarsest rungs are populated
    support[names[-1]] = 500
    routes[names[-1]] = 60
    support[names[-2]] = 40
    routes[names[-2]] = 12
    res = ladder.resolve(support, routes, min_n=30, min_routes=5)
    assert res.rung == names[-2]              # finest rung that clears the bar, not the biggest
    assert res.backed_off == len(names) - 2
    assert res.starved == names[:-2]


def test_resolve_route_floor_rejects_one_prolific_route():
    """Many observations from few routes are few facts — the route floor must catch that."""
    names = ladder.RUNG_NAMES
    support = {n: 0 for n in names}
    routes = {n: 0 for n in names}
    support[names[0]] = 400          # plenty of observations...
    routes[names[0]] = 2             # ...from two routes
    support[names[-1]] = 60
    routes[names[-1]] = 30
    res = ladder.resolve(support, routes, min_n=30, min_routes=5)
    assert res.rung == names[-1]
    assert names[0] in res.starved


def test_resolve_returns_none_when_nothing_qualifies():
    res = ladder.resolve({n: 1 for n in ladder.RUNG_NAMES},
                         {n: 1 for n in ladder.RUNG_NAMES}, min_n=30)
    assert res.rung is None
    assert res.starved == ladder.RUNG_NAMES
