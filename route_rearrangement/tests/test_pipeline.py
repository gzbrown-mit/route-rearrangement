import json

from route_rearrangement.pipeline import main as pipeline_main

from .conftest import CORPUS, corpus_required


@corpus_required
def test_pipeline_runs_end_to_end(tmp_path):
    """The one-shot pipeline enumerates, materializes and scores over a corpus slice, and
    reports linear/convergent coverage rather than silently dropping branching trees."""
    out = tmp_path / "out"
    rc = pipeline_main([
        "--corpus", CORPUS, "--out-dir", str(out),
        "--limit", "40", "--cap", "50", "--max-accepted", "10",
        "--no-treelstm", "--no-fg",
    ])
    assert rc == 0
    for name in ("scored.jsonl", "routes.jsonl", "failures.jsonl", "summary.json"):
        assert (out / name).exists(), f"missing {name}"

    summary = json.loads((out / "summary.json").read_text())
    counts = summary["counts"]
    assert counts["scanned"] == 40
    # coverage is fully accounted for: every scanned tree lands in exactly one bucket
    # (convergent trees are now processed, not skipped)
    buckets = ("linear", "convergent", "unmappable", "disconnected",
               "out_of_step_range")
    assert sum(counts.get(b, 0) for b in buckets) == counts["scanned"]

    # scored records carry a metrics block and the original ordering is present per tree
    scored = [json.loads(l) for l in (out / "scored.jsonl").read_text().splitlines()]
    if scored:
        assert all("metrics" in r for r in scored)
        assert any(r.get("is_original_order") for r in scored)
