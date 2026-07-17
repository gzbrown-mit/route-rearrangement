"""Route-record -> PNG bridge, reusing the synthesis_extraction rendering core verbatim.

``synthesis_extraction.gui.pathway_renderer.render_pathway_png`` already turns a
``full_graph``-shaped dict into a left-to-right reaction scheme (RDKit skeletal drawings
joined by Graphviz arrows).  Our materialized routes rebuild into exactly that shape
(:func:`..filters.rebuilt_full_graph`), so rendering is a thin wrapper.  The molecules are
map-free, so the reaction-centre highlighting simply draws nothing extra.
"""

from __future__ import annotations

import csv
import os
import sys
from functools import lru_cache
from typing import Dict, List

from .. import deps  # noqa: F401
from ..filters import rebuilt_full_graph
from ..schema import route_from_record
from .reaction_class import classify_reaction


def ensure_graphviz_on_path() -> None:
    """Prepend the interpreter's bin dir so the ``dot`` binary is found (the gui app's
    shim — in the conda env ``dot`` sits next to python but is not on PATH)."""
    bindir = os.path.dirname(sys.executable)
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")


@lru_cache(maxsize=1)
def _render_core():
    """The three reusable synthesis_extraction rendering functions (dot on PATH first)."""
    ensure_graphviz_on_path()
    from synthesis_extraction.gui.pathway_renderer import build_reaction_graph
    from synthesis_extraction.gui.reaction_sequence_plot import (
        build_augmented_graph, visualize_augmented_graph)
    return build_reaction_graph, build_augmented_graph, visualize_augmented_graph


def _reactants_of(step: dict) -> List[str]:
    reactants = list(step.get("side_reactants", []))
    if step.get("chain_precursor"):
        reactants.append(step["chain_precursor"])
    return reactants


@lru_cache(maxsize=8)
def _rxn_class_lookup(corpus_path: str) -> Dict[int, str]:
    """Authoritative ``{rxn_index: reaction-class definition}`` from the corpus's
    ``rxn_class_lookup.csv`` (the same NameRXN-style labels the source route carries).

    Located next to the corpus file, or via ``ROUTE_RXN_CLASS_LOOKUP``.  Empty if absent.
    """
    path = os.environ.get("ROUTE_RXN_CLASS_LOOKUP", "")
    if not path and corpus_path:
        cand = os.path.join(os.path.dirname(corpus_path), "rxn_class_lookup.csv")
        if os.path.exists(cand):
            path = cand
    out: Dict[int, str] = {}
    if path and os.path.exists(path):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                try:
                    out[int(row["rxn_index"])] = row["definition"]
                except (ValueError, KeyError):
                    continue
    return out


def _reaction_class(step: dict, lookup: Dict[int, str]) -> str:
    """The step's reaction class.

    The dataset's ``rxn_class_lookup.csv`` label is authoritative and used verbatim whenever
    present — including its own "Unrecognized" verdict, which is not overridden with a guess.
    The functional-group heuristic is used only when the corpus ships no class at all for that
    reaction (no lookup file, or the rxn_index is absent from it)."""
    defn = lookup.get(int(step.get("orig_rxn_index", -1)))
    if defn:
        return defn
    return classify_reaction(_reactants_of(step), step.get("new_product", ""))


def _box_labels(record: dict) -> Dict[int, str]:
    """``{rebuilt_node_id: "lit. step N: <reaction class>"}`` for every step.

    The rebuilt-graph node id is ``n_steps - position + 1`` (matching
    :func:`..filters.rebuilt_full_graph`); the *original* literature step number is the
    step's rank among the original step ids (deepest = step 1), so it is stable across all
    rearrangements of the same route.  Reaction classes come from the corpus's
    ``rxn_class_lookup.csv`` (authoritative), falling back to a heuristic only when the
    dataset has no class for that reaction.
    """
    steps = record.get("steps", [])
    n = len(steps)
    orig_ids = sorted({s["orig_step_id"] for s in steps}, reverse=True)
    orig_step_no = {sid: i + 1 for i, sid in enumerate(orig_ids)}
    lookup = _rxn_class_lookup((record.get("provenance") or {}).get("corpus", ""))
    labels: Dict[int, str] = {}
    for s in steps:
        nid = n - s["position"] + 1
        cls = _reaction_class(s, lookup)
        base = f"lit. step {orig_step_no[s['orig_step_id']]}"
        labels[nid] = f"{base}: {cls}" if cls else base
    return labels


def render_record_png(record: dict, work_dir: str, dpi: int = 130) -> str:
    """Render one materialized-route record to a PNG under *work_dir*; returns its path.

    Reaction boxes are labelled with the reaction-class name and the original literature
    step number (not the rearranged position), so a box reads e.g. "lit. step 5: amide
    coupling".  Each route renders into its own subdirectory keyed by ``tree_id`` + ordering
    so distinct routes get distinct, stable PNG paths."""
    key = f"{record.get('tree_id', 'route')}_" + "-".join(map(str, record["ordering"]))
    sub = os.path.join(work_dir, key)
    os.makedirs(sub, exist_ok=True)

    build_reaction_graph, build_augmented_graph, visualize_augmented_graph = _render_core()
    full_graph = rebuilt_full_graph(route_from_record(record))
    reaction_graph = build_reaction_graph(full_graph, rxn_classes=None)
    # overwrite the default "Step N" box text with our reaction-class + original-step label
    labels = _box_labels(record)
    for nid in reaction_graph.nodes:
        if nid in labels:
            reaction_graph.nodes[nid]["NameRXN"] = labels[nid]
    augmented = build_augmented_graph(reaction_graph)
    return visualize_augmented_graph(
        augmented, filename=os.path.join(sub, "pathway"), dpi=dpi,
        drop_atom_maps=True, all_expanded_ccenter=None,
        image_dir=os.path.join(sub, "molecule_images"))
