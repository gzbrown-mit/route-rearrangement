"""JSONL (de)serialization of materialized-route and failure records."""

from __future__ import annotations

import json
from typing import IO, Optional

from .materialize import MaterializedRoute, StepRecord


def route_record(tree_id: str, route: MaterializedRoute, *, ordering_index: int,
                 variant: int, is_original_order: bool, identity_roundtrip: bool,
                 flags: dict, provenance: Optional[dict] = None) -> dict:
    return {
        "tree_id": tree_id,
        "ordering": route.ordering,
        "ordering_index": ordering_index,
        "variant": variant,
        "status": route.status,
        "target": route.target,
        "steps": [
            {
                "position": r.position,
                "orig_step_id": r.orig_step_id,
                "orig_rxn_index": r.orig_rxn_index,
                "retro_smarts": r.retro_smarts,
                "new_rxn": r.new_rxn,
                "chain_precursor": r.chain_precursor,
                "side_reactants": r.side_reactants,
                "new_product": r.new_product,
                "outcome_rank": r.outcome_rank,
                "n_outcomes": r.n_outcomes,
                "exact_side_match": r.exact_side_match,
                "sim_score": round(r.sim_score, 4),
            }
            for r in route.steps
        ],
        "starting_materials": route.starting_materials,
        "is_original_order": is_original_order,
        "identity_roundtrip": identity_roundtrip,
        "flags": flags,
        "provenance": provenance or {},
    }


def failure_record(tree_id: str, route: MaterializedRoute, *, ordering_index: int) -> dict:
    return {
        "tree_id": tree_id,
        "ordering": route.ordering,
        "ordering_index": ordering_index,
        "reason": route.status,
        "failed_position": route.failed_position,
        "failed_step_id": route.failed_step_id,
        "intermediate_smiles": route.failure_intermediate,
    }


def route_from_record(record: dict) -> MaterializedRoute:
    """Reconstruct a :class:`MaterializedRoute` from a ``routes.jsonl`` record (for the
    GUI renderer, which needs the route's step SMILES)."""
    steps = [
        StepRecord(
            position=s["position"], orig_step_id=s["orig_step_id"],
            orig_rxn_index=s["orig_rxn_index"], retro_smarts=s["retro_smarts"],
            new_rxn=s["new_rxn"], chain_precursor=s.get("chain_precursor"),
            side_reactants=list(s["side_reactants"]), new_product=s["new_product"],
            outcome_rank=s["outcome_rank"], n_outcomes=s["n_outcomes"],
            exact_side_match=s["exact_side_match"], sim_score=s["sim_score"],
        )
        for s in sorted(record["steps"], key=lambda s: s["position"])
    ]
    return MaterializedRoute(
        ordering=record["ordering"], status=record.get("status", "ok"),
        target=record.get("target", ""), steps=steps,
        starting_materials=record.get("starting_materials", []))


def write_jsonl(fh: IO[str], record: dict) -> None:
    fh.write(json.dumps(record) + "\n")
