"""Aggregate a centers cache into per-rung pair tables carrying route-cluster statistics.

This exists rather than reusing ``transformation.build_evidence`` for one concrete reason:
``OrderEvidenceTable`` caps ``PairEvidence.witnesses`` at 25 entries per pair, so the route
identities behind a 400-observation pair are truncated and cannot support route-clustered
inference.  Clustering matters here more than usual — PaRoutes contains many near-duplicate
routes, and treating their observations as independent inflates significance without bound.

So instead of storing route ids, we accumulate the **sufficient statistics** for a clustered
ratio estimator.  For each pair, per route *r*, let ``a_r`` be the observations in the
first-second direction and ``n_r = a_r + b_r`` the strictly-ordered observations.  Carrying
only ``R``, ``Σa_r``, ``Σn_r``, ``Σa_r²``, ``Σa_r·n_r`` and ``Σn_r²`` — six integers — is
enough for :mod:`.significance` to compute a cluster-robust standard error exactly, with no
per-route storage and no bootstrap resampling pass.  Memory stays flat in corpus size.

Everything chemical is delegated: contextual centers come from the FrequenTree cache written
by ``transformation.extract_centers``, transformation identity from :mod:`.ladder`, and the
material-forcing confound control from ``transformation.order_evidence._is_forced`` (a pair
whose observed order is forced because one center consumes an atom the other creates is not
evidence about chemists' preferences — it is bookkeeping, and it is excluded).

Usage::

    python -m route_rearrangement.literature_precedent.aggregate \\
        --centers ~/Downloads/paroutes_all/centers/centers_00*.jsonl \\
        --material-deps ~/Downloads/paroutes_all/material_deps/*.jsonl \\
        --out-dir route_rearrangement/literature_precedent/results/agg/ [--rung template_exact]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .. import deps  # noqa: F401
from . import ladder
from .ladder import RUNGS, Rung
from synthesis_extraction.transformation.extract_centers import iter_center_files
from synthesis_extraction.transformation.fc_adapter import ContextualCenter
from synthesis_extraction.transformation.order_evidence import _is_forced

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Per-pair accumulator
# ---------------------------------------------------------------------------
# Slot layout for the flat list stored per pair.  A list of ints is ~3x lighter than a
# dataclass here and the tables reach millions of pairs at the finest rungs.
(N_FS, N_SF, N_SAME, N_FORCED, N_ROUTES, S_A, S_N, S_AA, S_AN, S_NN) = range(10)
_WIDTH = 10


def _new_row() -> List[int]:
    return [0] * _WIDTH


class RungTable:
    """Pair statistics at one rung, with string keys interned to ints to save memory."""

    def __init__(self, rung: Rung):
        self.rung = rung
        self._key_id: Dict[str, int] = {}
        self._keys: List[str] = []
        self.pairs: Dict[Tuple[int, int], List[int]] = {}
        self.density: Dict[int, int] = defaultdict(int)      # routes containing the key
        self.depth_sum: Dict[int, float] = defaultdict(float)  # Σ normalized formation depth
        self.depth_n: Dict[int, int] = defaultdict(int)
        # A few real reacting-substructure SMILES per key.  Downstream auditing (does this key
        # mean "nitro reduction"?) is far more reliable on real SMILES than on SMARTS-matching
        # the template string against itself, and the collision audit needs them too.
        self.env_samples: Dict[int, List[str]] = defaultdict(list)
        self.max_env_samples = 4
        # This key's key at the *coarsest* rung.  Rungs use different key strings for the same
        # chemistry, so cross-rung backoff cannot join on the key itself; it joins on this
        # lineage instead.  Recorded here because only the aggregation pass sees a center's
        # keys at every rung at once.
        self.parent: Dict[int, str] = {}
        self.n_routes = 0
        self.n_routes_with_keys = 0
        self.n_centers = 0
        self.n_centers_keyed = 0

    def intern(self, key: str) -> int:
        i = self._key_id.get(key)
        if i is None:
            i = len(self._keys)
            self._key_id[key] = i
            self._keys.append(key)
        return i

    def key_of(self, key_id: int) -> str:
        return self._keys[key_id]

    @property
    def n_keys(self) -> int:
        return len(self._keys)

    # -- accumulation --------------------------------------------------------
    def observe_route(self, keyed: Sequence[Tuple[ContextualCenter, str]],
                      forced_index: Optional[Dict[int, set]] = None,
                      parents: Optional[Sequence[Optional[str]]] = None) -> None:
        """Fold one route's observations in, clustering by this route.

        *keyed* is the route's centers paired with their key at this rung (already filtered to
        centers that have one).  Depth is the center's rank among the route's distinct
        formation steps, normalized to [0, 1] — a rank, not a step number, because the centers
        cache does not record route length and a rank needs no such normalization.
        """
        self.n_routes += 1
        if not keyed:
            return
        self.n_routes_with_keys += 1

        ids = [self.intern(k) for _, k in keyed]
        for kid in set(ids):
            self.density[kid] += 1
        for i, ((c, _), kid) in enumerate(zip(keyed, ids)):
            bucket = self.env_samples[kid]
            if c.synthon_env and len(bucket) < self.max_env_samples and c.synthon_env not in bucket:
                bucket.append(c.synthon_env)
            if parents is not None and kid not in self.parent and parents[i]:
                self.parent[kid] = parents[i]

        # normalized formation depth (rank-based)
        steps = sorted({c.formation_step for c, _ in keyed})
        if len(steps) > 1:
            rank_of = {s: i / (len(steps) - 1) for i, s in enumerate(steps)}
            for (c, _), kid in zip(keyed, ids):
                self.depth_sum[kid] += rank_of[c.formation_step]
                self.depth_n[kid] += 1

        # per-route pair tallies, so the cluster statistics see one route at a time
        local: Dict[Tuple[int, int], List[int]] = {}
        n = len(keyed)
        for i in range(n):
            for j in range(i + 1, n):
                (ca, ka), (cb, kb) = keyed[i], keyed[j]
                if ka == kb:
                    continue
                # canonical direction: sort by key string, exactly as upstream does, so a pair
                # means the same thing here and in transformation.order_evidence
                if ka <= kb:
                    (ka_s, ca_s), (kb_s, cb_s) = (ka, ca), (kb, cb)
                    pair = (ids[i], ids[j])
                else:
                    (ka_s, ca_s), (kb_s, cb_s) = (kb, cb), (ka, ca)
                    pair = (ids[j], ids[i])
                row = local.get(pair)
                if row is None:
                    row = local[pair] = _new_row()
                if forced_index is not None and _is_forced(ca_s, cb_s, forced_index):
                    row[N_FORCED] += 1
                    continue
                sa, sb = ca_s.formation_step, cb_s.formation_step
                if sa < sb:
                    row[N_FS] += 1
                elif sb < sa:
                    row[N_SF] += 1
                else:
                    row[N_SAME] += 1

        for pair, row in local.items():
            a, b = row[N_FS], row[N_SF]
            tot = a + b
            g = self.pairs.get(pair)
            if g is None:
                g = self.pairs[pair] = _new_row()
            g[N_FS] += a
            g[N_SF] += b
            g[N_SAME] += row[N_SAME]
            g[N_FORCED] += row[N_FORCED]
            if tot:
                g[N_ROUTES] += 1
                g[S_A] += a
                g[S_N] += tot
                g[S_AA] += a * a
                g[S_AN] += a * tot
                g[S_NN] += tot * tot

    # -- serialization -------------------------------------------------------
    def to_dict(self, *, min_n: int = 1) -> dict:
        """JSON-ready. *min_n* drops pairs below a support floor — the tail is the bulk of the
        table and nothing downstream can test a pair with two observations anyway."""
        rows = []
        used: Dict[int, int] = {}
        for (ia, ib), r in self.pairs.items():
            if r[N_FS] + r[N_SF] < min_n:
                continue
            used.setdefault(ia, len(used))
            used.setdefault(ib, len(used))
            rows.append([used[ia], used[ib]] + r)
        keys = [None] * len(used)
        for kid, new in used.items():
            keys[new] = self._keys[kid]
        return {
            "schema": SCHEMA_VERSION,
            "rung": self.rung.name,
            "rung_index": self.rung.index,
            "rung_description": self.rung.description,
            "min_n_written": min_n,
            "counts": {
                "n_routes": self.n_routes,
                "n_routes_with_keys": self.n_routes_with_keys,
                "n_centers": self.n_centers,
                "n_centers_keyed": self.n_centers_keyed,
                "n_keys_total": self.n_keys,
                "n_pairs_total": len(self.pairs),
                "n_pairs_written": len(rows),
            },
            "keys": keys,
            "density": [self.density.get(kid, 0) for kid, _ in
                        sorted(used.items(), key=lambda t: t[1])],
            "depth": [
                (self.depth_sum.get(kid, 0.0) / self.depth_n[kid]) if self.depth_n.get(kid) else None
                for kid, _ in sorted(used.items(), key=lambda t: t[1])
            ],
            "env_samples": [self.env_samples.get(kid, []) for kid, _ in
                            sorted(used.items(), key=lambda t: t[1])],
            "parent_key": [self.parent.get(kid) for kid, _ in
                           sorted(used.items(), key=lambda t: t[1])],
            "pair_columns": ["a", "b", "n_first_second", "n_second_first", "n_same_step",
                             "n_material_forced", "n_routes", "sum_a", "sum_n",
                             "sum_a2", "sum_an", "sum_n2"],
            "pairs": rows,
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_forced(paths: Optional[Sequence[str]]) -> Dict[str, Dict[int, set]]:
    if not paths:
        return {}
    from synthesis_extraction.dependency.extract_material_deps import iter_material_dep_files
    out = dict(iter_material_dep_files(paths))
    log.warning("loaded material lineage for %d routes", len(out))
    return out


def aggregate(route_centers: Iterable[Tuple[str, List[ContextualCenter]]],
              rungs: Sequence[Rung] = RUNGS,
              forced: Optional[Dict[str, Dict[int, set]]] = None,
              limit: int = 0,
              progress_every: int = 5000) -> Dict[str, RungTable]:
    """Stream ``(route_id, centers)`` into one :class:`RungTable` per rung."""
    tables = {r.name: RungTable(r) for r in rungs}
    forced = forced or {}
    n = 0
    for route_id, centers in route_centers:
        if limit and n >= limit:
            break
        n += 1
        fidx = forced.get(route_id)
        # one key computation per center per rung, reused across the pair loop
        keys_per_center = [ladder.keys_for_center(c, rungs) for c in centers]
        coarsest = rungs[-1].name
        for r in rungs:
            t = tables[r.name]
            t.n_centers += len(centers)
            pairs_ = [(c, ks[r.name], ks[coarsest])
                      for c, ks in zip(centers, keys_per_center) if ks[r.name]]
            keyed = [(c, k) for c, k, _ in pairs_]
            t.n_centers_keyed += len(keyed)
            t.observe_route(keyed, forced_index=fidx, parents=[p for _, _, p in pairs_])
        if progress_every and n % progress_every == 0:
            log.warning("%d routes; pairs at %s: %d", n, rungs[0].name,
                        len(tables[rungs[0].name].pairs))
    return tables


def _iter_centers(paths: Sequence[str]) -> Iterator[Tuple[str, List[ContextualCenter]]]:
    return iter_center_files(paths)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--centers", nargs="+", required=True, help="centers-cache JSONL file(s)")
    ap.add_argument("--material-deps", nargs="+", default=None,
                    help="material-deps JSONL file(s); materially-forced pairs are excluded")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rung", action="append", default=None,
                    help="rung name (repeatable); default all. Fewer rungs = less memory.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-n-write", type=int, default=2,
                    help="drop pairs below this ordered-observation count when writing")
    args = ap.parse_args(argv)

    if args.rung:
        unknown = [r for r in args.rung if r not in ladder.RUNG_BY_NAME]
        if unknown:
            ap.error(f"unknown rung(s) {unknown}; choose from {ladder.RUNG_NAMES}")
        rungs = [ladder.RUNG_BY_NAME[r] for r in args.rung]
    else:
        rungs = list(RUNGS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = aggregate(_iter_centers(args.centers), rungs=rungs,
                       forced=load_forced(args.material_deps), limit=args.limit)

    for name, t in tables.items():
        path = out_dir / f"pairs_{name}.json"
        with path.open("w") as fh:
            json.dump(t.to_dict(min_n=args.min_n_write), fh)
        print(f"{name:20s} keys={t.n_keys:8d} pairs={len(t.pairs):9d} "
              f"routes={t.n_routes_with_keys}/{t.n_routes} -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
