"""Pytest configuration.

Adds ``src/`` to ``sys.path`` so the test-suite runs even when the package
has not been installed with ``pip install -e .``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
