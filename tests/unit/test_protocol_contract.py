"""The protocol constants in the mod (mod/SharedMemoryBridge.cs) must match the Python client's."""

from __future__ import annotations

import re

import pytest

from rainworld_rl import launcher, shared_memory


def test_protocol_constants_match_the_mod():
    bridge = launcher.MOD_SOURCE_DIR / "SharedMemoryBridge.cs"
    if not bridge.is_file():
        pytest.skip("not a source checkout")
    cs = {}
    for name, expr in re.findall(r"\bconst \w+ (\w+) = ([^;]+);", bridge.read_text()):
        cs[name] = eval("(" + re.sub(r"(?<=\d)u\b", "", expr) + ")", {}, cs)
    names = [n for n in cs if re.match(r"PROTOCOL_VERSION$|(OFFSET|KEY|STATUS|GAME_FLAG|CMD)_", n)]
    py_names = {"KEY_COUNT": "NUM_KEYS"}
    expected = {n: cs[n] for n in names}
    actual = {n: getattr(shared_memory, py_names.get(n, re.sub("^CMD_", "COMMAND_", n)), None) for n in names}
    assert actual == expected
