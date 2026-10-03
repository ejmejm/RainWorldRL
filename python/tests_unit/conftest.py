"""
Pytest setup for the game-free unit tests.

The package is imported as ``rainworld_rl.python.*`` (the repository directory
is the top-level package), so the repository's *parent* must be importable.
"""

import sys
from pathlib import Path

_REPO_PARENT = Path(__file__).resolve().parents[3]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))
