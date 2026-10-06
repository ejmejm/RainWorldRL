"""Connection-level checks: status bits and heartbeat."""

from __future__ import annotations

from tests.e2e.helpers import assert_info_contract, step_n, wait_for

# PROTOCOL.md: the mod may still be entering the game after CONNECTED rises.
READY_TIMEOUT_S = 60.0
HEARTBEAT_TIMEOUT_S = 2.0


def test_connected_and_alive(env):
    """MOD_ALIVE, CONNECTED, READY and IN_GAME are set, and `heartbeat` advances every Unity Update."""
    read_state = env._client.read_state
    state = read_state()
    assert state.mod_alive, "MOD_ALIVE is clear: the mod's Awake has not run (or OnDestroy ran)"
    assert state.connected, "CONNECTED is clear: the client did not assert its bit"
    assert wait_for(lambda: read_state().ready, timeout=READY_TIMEOUT_S), (
        f"READY did not become set within {READY_TIMEOUT_S}s (status={read_state().status:#x})"
    )
    assert read_state().in_game, "IN_GAME bit clear right after connect"
    assert wait_for(lambda: read_state().heartbeat != state.heartbeat, timeout=HEARTBEAT_TIMEOUT_S), (
        f"heartbeat stuck at {state.heartbeat} for {HEARTBEAT_TIMEOUT_S}s - game frozen or mod dead"
    )

    (_obs, info), = step_n(env, 1)
    assert_info_contract(info)
    assert info["in_game"] is True
    assert info["human_override"] is False, "human override active - press F10 in game to return control"
