"""The prebuilt mod DLL shipped in the package must match the mod sources."""

from __future__ import annotations

import pytest

from rainworld_rl import launcher


def test_packaged_dll_matches_sources():
    current = launcher.mod_source_hash()
    if current is None:
        pytest.skip("not a source checkout")
    assert launcher.PACKAGED_DLL.is_file(), "no packaged DLL; run `python -m rainworld_rl.launcher --build-only`"
    assert launcher.PACKAGED_HASH.read_text().strip() == current, (
        "the mod sources changed since rainworld_rl/mod/RainWorldRL.dll was built; "
        "run `python -m rainworld_rl.launcher --build-only` and commit rainworld_rl/mod/"
    )
