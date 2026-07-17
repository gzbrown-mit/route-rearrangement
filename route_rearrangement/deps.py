"""Path bootstrap for the ``synthesis_extraction`` checkout (which has no packaging
metadata), following the convention of its own ``transformation/fc_adapter.py``:
an env-var-overridable ``sys.path`` insert.  Set ``SYNTHESIS_EXTRACTION_PATH`` if the
clone is not at ``~/synthesis_extraction``.

Import the pieces of synthesis_extraction through :func:`bootstrap` (or just import
this module before any ``synthesis_extraction.*`` import).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SYNTHESIS_EXTRACTION_PATH = os.environ.get(
    "SYNTHESIS_EXTRACTION_PATH", str(Path.home() / "synthesis_extraction")
)


def bootstrap() -> None:
    if SYNTHESIS_EXTRACTION_PATH not in sys.path:
        sys.path.insert(0, SYNTHESIS_EXTRACTION_PATH)


bootstrap()
