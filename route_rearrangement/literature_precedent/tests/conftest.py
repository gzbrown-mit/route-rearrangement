import os
from pathlib import Path

import pytest

CENTERS = os.environ.get(
    "LITERATURE_PRECEDENT_TEST_CENTERS",
    str(Path.home() / "Downloads/paroutes_all/centers/centers_000.jsonl"),
)

centers_required = pytest.mark.skipif(
    not Path(CENTERS).exists(), reason=f"centers cache not found at {CENTERS}"
)


@pytest.fixture(scope="session")
def centers_path():
    return CENTERS


@pytest.fixture(scope="session")
def centers_slice(centers_path):
    """The first 200 routes of the centers cache as ``[(route_id, [ContextualCenter])]``."""
    from synthesis_extraction.transformation.extract_centers import iter_center_files

    out = []
    for i, item in enumerate(iter_center_files([centers_path])):
        if i >= 200:
            break
        out.append(item)
    return out
