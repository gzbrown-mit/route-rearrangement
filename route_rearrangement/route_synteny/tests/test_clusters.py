"""The cluster model and the compactness statistic, against the paper's definitions."""

import pytest

from route_rearrangement.route_synteny import clusters, significance
from route_rearrangement.route_synteny.corpus import UNKNOWN, Genome


def _genome(rid, families):
    ids = list(range(len(families), 0, -1))
    return Genome(route_id=rid, n_steps=len(families), step_ids=ids, families=list(families))


# ---------------------------------------------------------------------------
# D(C, C') — the symmetric set distance
# ---------------------------------------------------------------------------
def test_set_distance_is_insertions_plus_deletions():
    c = frozenset("ABC")
    assert clusters.set_distance(c, "ABC") == 0
    assert clusters.set_distance(c, "AB") == 1            # one deletion
    assert clusters.set_distance(c, "ABCD") == 1          # one insertion
    assert clusters.set_distance(c, "ABD") == 2           # one of each


def test_a_missing_member_is_a_delta_location_at_delta_1_but_not_0():
    fams = list("XABY")                                    # C = {A,B,Z}, Z absent
    c = frozenset("ABZ")
    assert clusters.delta_locations(fams, c, delta=0, route_id="r") == []
    assert clusters.delta_locations(fams, c, delta=1, route_id="r")


def test_delta_location_tolerates_a_step_inside_the_bracket():
    """The protect -> react -> deprotect case: a foreign step sits inside the block."""
    fams = ["protect", "coupling", "deprotect"]
    c = frozenset({"protect", "deprotect"})
    got = clusters.delta_locations(fams, c, delta=1, route_id="r")
    assert got and min(o.distance for o in got) <= 1


def test_unknown_steps_do_not_count_as_cluster_members():
    fams = ["A", UNKNOWN, "B"]
    c = frozenset("AB")
    best = min(clusters.delta_locations(fams, c, delta=1, route_id="r"),
               key=lambda o: (o.distance, o.width))
    assert best.distance == 0        # the UNKNOWN is an insertion in position, not in content


# ---------------------------------------------------------------------------
# The compactness statistic
# ---------------------------------------------------------------------------
def test_min_window_finds_the_narrowest_interval():
    fams = list("AXXBAB")
    assert significance.min_window(fams, frozenset("AB"), 2) == 2      # the trailing "AB"
    assert significance.min_window(fams, frozenset("AZ"), 2) is None   # Z never occurs
    assert significance.min_window(fams, frozenset("AZ"), 1) == 1      # one member suffices


def test_min_window_agrees_with_the_delta_location_scan():
    """The fast statistic and the slower locating scan must not disagree."""
    fams = list("CABXXAB")
    c = frozenset("AB")
    w = significance.min_window(fams, c, 2)
    best = min(clusters.delta_locations(fams, c, delta=0, route_id="r"),
               key=lambda o: o.width)
    assert w == best.width


def test_tightness_respects_the_width_bound():
    scattered = list("A" + "X" * 6 + "B")
    tight = list("AB" + "X" * 6)
    c = frozenset("AB")
    assert not significance.is_tight(scattered, c, delta=0, max_extra=2)
    assert significance.is_tight(tight, c, delta=0, max_extra=2)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_candidates_are_anchored_to_real_windows():
    """A family set that never appears contiguously must not become a reference cluster."""
    gs = [_genome("r1", "AXB"), _genome("r2", "AXB")]
    cand = clusters.candidate_clusters(gs, s=2, max_size=2, min_route_support=2)
    assert frozenset("AB") not in cand           # A and B are never adjacent
    assert frozenset("AX") in cand


def test_quorum_counts_routes_not_windows():
    gs = [_genome("r1", "ABAB"), _genome("r2", "AB"), _genome("r3", "ZZ")]
    found = clusters.find_clusters(gs, s=2, max_size=2, delta=0, quorum=2,
                                   min_route_support=2)
    ab = [c for c in found if c.families == frozenset("AB")]
    assert ab and ab[0].quorum == 2              # r1 contributes once despite two windows


def test_deduplicate_drops_subsumed_subsets():
    big = clusters.Cluster(families=frozenset("ABC"), reference_route="r",
                           occurrences=[clusters.Occurrence("r", 0, 3, 0),
                                        clusters.Occurrence("s", 0, 3, 0)])
    small = clusters.Cluster(families=frozenset("AB"), reference_route="r",
                             occurrences=[clusters.Occurrence("r", 0, 2, 0),
                                          clusters.Occurrence("s", 0, 2, 0)])
    kept = clusters.deduplicate([big, small])
    assert [c.families for c in kept] == [frozenset("ABC")]


def test_a_subset_with_wider_support_survives():
    """Only *subsumed* subsets go: one that recurs more widely is its own finding."""
    big = clusters.Cluster(families=frozenset("ABC"), reference_route="r",
                           occurrences=[clusters.Occurrence("r", 0, 3, 0)])
    small = clusters.Cluster(families=frozenset("AB"), reference_route="r",
                             occurrences=[clusters.Occurrence(x, 0, 2, 0) for x in "rst"])
    kept = {c.families for c in clusters.deduplicate([big, small])}
    assert kept == {frozenset("ABC"), frozenset("AB")}
