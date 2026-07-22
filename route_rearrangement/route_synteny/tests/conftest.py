import os
from pathlib import Path

import pytest

GENOMES = os.environ.get(
    "ROUTE_SYNTENY_TEST_GENOMES",
    str(Path(__file__).resolve().parents[1] / "results" / "genomes.jsonl"),
)

genomes_required = pytest.mark.skipif(
    not Path(GENOMES).exists(),
    reason=f"genomes cache not found at {GENOMES}; run route_synteny.corpus first",
)


@pytest.fixture(scope="session")
def corpus_genomes():
    """A slice of the real genome cache, for the corpus-backed controls."""
    from route_rearrangement.route_synteny.decompose import load_genomes

    return load_genomes(GENOMES, limit=4000)
