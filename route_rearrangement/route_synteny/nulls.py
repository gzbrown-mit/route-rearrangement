"""The three null models — and why the decomposition needs exactly this set.

Gene cluster statistics have one null: random gene order, with character frequencies matched to
the genome so that common families do not manufacture significance.  That null answers *"is
this family set found together more than chance?"* and it is kept here unchanged as
:func:`frequency_null_string` (**Null-F**), because it is what cross-route cluster *detection*
needs and because our family frequencies are at least as skewed as a genome's.

The necessity/convention split needs a second and third null, and the crucial design point is
that they differ from each other in **exactly one respect**:

* **Null-P** — a uniform random permutation of *this route's own steps*.
* **Null-C** — a uniform random **linear extension** of this route's essential partial order.

Both permute the same multiset of steps; the only difference is whether the chemistry's
ordering constraints are imposed.  That is what makes "significant under Null-P but not under
Null-C" mean *the partial order explains it* and nothing else.  Had Null-P instead resampled
characters from corpus frequencies (as Null-F does), the contrast would confound the constraint
with a change of composition, and the decomposition would not be interpretable.

They are implemented through the same code path to guarantee that property: Null-P is simply
Null-C with an empty constraint set, so both come from ``ScheduleLattice`` and any bias in the
sampler affects both identically.  ``ScheduleLattice.sample`` weights each choice by the number
of completions of the resulting downset, which makes it a uniform sampler over linear
extensions rather than the biased "pick a random available step" walk; ``precedence_probabilities``
gives the same distribution's pairwise marginals *exactly*, so pair-level statistics need no
sampling at all.

One adaptation from the source papers is unavoidable and is called out where it is made: they
draw character probabilities from the frequencies within a single genome, which works when a
genome has thousands of genes.  A route has 4–20 steps, so per-route frequencies are degenerate
and Null-F draws from **corpus-wide** family frequencies instead.
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .. import deps  # noqa: F401
from .corpus import Genome
from synthesis_extraction.dependency.schedule import ScheduleLattice


def lattice_for_genome(genome: Genome, tier: Optional[str]) -> ScheduleLattice:
    """The route's ordering lattice.

    *tier* selects the necessity tier; ``None`` gives the **unconstrained** lattice (Null-P).
    Both go through ``ScheduleLattice`` so the constrained and free nulls are drawn by identical
    machinery and differ only in the constraint set.
    """
    constraints = () if tier is None else genome.constraints.get(tier, ())
    return ScheduleLattice(genome.step_ids, constraints)


def sample_orders(lattice: ScheduleLattice, n: int, seed: int = 0) -> Iterator[List[int]]:
    """*n* independent uniform draws of a valid ordering (step ids, earliest first)."""
    for i in range(n):
        order = lattice.sample(seed=seed * 1_000_003 + i)
        if order is None:                       # unsatisfiable constraints
            return
        yield order


def exact_precedence(lattice: ScheduleLattice) -> Dict[Tuple[int, int], float]:
    """``{(a, b): P(a before b)}`` exactly, over all valid orderings.

    Preferred over sampling wherever the statistic is pairwise: it is the analytic p-value the
    source papers reach for, with no Monte Carlo error. Raises for routes whose lattice is too
    large (upstream guard), so callers should fall back to :func:`sample_orders`.
    """
    return lattice.precedence_probabilities()


def family_frequencies(genomes: Sequence[Genome], *, exclude: str = "?") -> Dict[str, float]:
    """Corpus-wide family probabilities — the alphabet distribution Null-F draws from.

    This is the gene-family-size correction the genomics literature insists on: a family that
    appears in a third of all routes will land near another common family by chance constantly,
    and a null that ignores that will call it a cluster.
    """
    counts: Dict[str, int] = {}
    total = 0
    for g in genomes:
        for f in g.families:
            if f == exclude:
                continue
            counts[f] = counts.get(f, 0) + 1
            total += 1
    if not total:
        return {}
    return {k: v / total for k, v in counts.items()}


def frequency_null_string(length: int, freqs: Dict[str, float], rng: random.Random) -> List[str]:
    """Null-F: a random family string of *length*, drawn i.i.d. from corpus frequencies."""
    if not freqs:
        return []
    families = list(freqs)
    weights = [freqs[f] for f in families]
    return rng.choices(families, weights=weights, k=length)


def read_families(genome: Genome, order: Sequence[int]) -> List[str]:
    """Re-read the genome under a different step order — the string a null draw produces.

    The families travel with their steps, so a permuted route is the same multiset of symbols in
    a new arrangement.  This is the operation both Null-P and Null-C apply.
    """
    by_step = dict(zip(genome.step_ids, genome.families))
    return [by_step[s] for s in order]
