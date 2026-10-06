"""
Pytest setup for the game-free unit tests.

The package is imported as ``rainworld_rl.*``; putting the repository root on
``sys.path`` makes that work without ``pip install -e .``.
"""

import sys
from pathlib import Path

_REPO_PARENT = Path(__file__).resolve().parents[2]  # repo root
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))
