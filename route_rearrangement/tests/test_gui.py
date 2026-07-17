import json
import os

from route_rearrangement.gui.model import TreeGroup, load_groups

from .conftest import corpus_required

# two toy records for the same tree: an original and one rearrangement, pre-scored
_RECS = [
    {"tree_id": "toy", "ordering": [2, 1], "is_original_order": True,
     "target": "CCOC(C)=O", "steps": [], "flags": {},
     "metrics": {"exposure": {"score": -1.0}, "complexity": {"score": -0.5}}},
    {"tree_id": "toy", "ordering": [1, 2], "is_original_order": False,
     "target": "CCOC(C)=O", "steps": [], "flags": {"fg_risk": [{"fg": "x"}]},
     "metrics": {"exposure": {"score": -2.0}, "complexity": {"score": -0.2}}},
]


def _group_from(recs) -> TreeGroup:
    from route_rearrangement.gui.model import RouteEntry
    original = RouteEntry(recs[0], is_original=True)
    rearr = [RouteEntry(r, is_original=False) for r in recs[1:]]
    return TreeGroup("toy", original, rearr)


def test_load_groups_separates_original(tmp_path):
    p = tmp_path / "scored.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _RECS))
    groups = load_groups(str(p))
    g = groups["toy"]
    assert g.original is not None and g.original.is_original
    assert len(g.rearrangements) == 1


def test_sort_and_percentile():
    g = _group_from(_RECS)
    assert "exposure" in g.available_metrics()
    # original beats the rearrangement on exposure (-1.0 > -2.0), loses on complexity
    assert g.percentile(g.original, "exposure") == 1.0
    assert g.percentile(g.original, "complexity") == 0.0


@corpus_required
def test_html_gallery_end_to_end(tmp_path):
    """Render a real scored route to an HTML gallery with embedded PNGs (needs graphviz)."""
    import pytest

    scored = "results/scored.jsonl"
    if not os.path.exists(scored):
        pytest.skip("run route_rearrangement.score first to produce results/scored.jsonl")
    from route_rearrangement.gui.gallery import build_gallery
    groups = load_groups(scored)
    tid = next(iter(groups))
    out = tmp_path / "gallery.html"
    build_gallery(groups[tid], str(out), work_dir=str(tmp_path / "imgs"), top=3)
    html = out.read_text()
    assert "data:image/png;base64," in html
