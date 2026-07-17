import os
from pathlib import Path

import pytest

CORPUS = os.environ.get(
    "ROUTE_REARRANGEMENT_TEST_CORPUS",
    str(Path.home() / "synthesis_extraction/synthesis_extraction/data/slice_0-1000/trees.jsonl"),
)

corpus_required = pytest.mark.skipif(
    not Path(CORPUS).exists(), reason=f"test corpus not found at {CORPUS}"
)


@pytest.fixture(scope="session")
def corpus_path():
    return CORPUS


@pytest.fixture(scope="session")
def load_tree(corpus_path):
    """Fetch one tree by id from the corpus (cached per session)."""
    from synthesis_extraction.load_trees import iter_trees

    cache = {}

    def _load(tree_id: str):
        if tree_id not in cache:
            for tid, tg in iter_trees(corpus_path):
                cache[tid] = tg
                if tid == tree_id:
                    break
        return cache.get(tree_id)

    return _load
