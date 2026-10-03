"""Connection-level checks: READY, heartbeat, in-game status."""

from __future__ import annotations

from tests.harness import (
    assert_info_contract,
    read_state,
    state_field,
    status_bit,
    step_n,
    wait_for,
)

# PROTOCOL.md: the mod may still be entering the game after CONNECTED rises.
READY_TIMEOUT_S = 60.0
HEARTBEAT_TIMEOUT_S = 2.0


def test_ready_within_timeout(env):
    """The game fixture waits for READY; it must still hold (or re-arrive) within the timeout."""
    state = read_state(env)
    assert status_bit(state, "mod_alive"), "MOD_ALIVE is clear: the mod's Awake has not run (or OnDestroy ran)"
    assert status_bit(state, "connected"), "CONNECTED is clear: the client did not assert its bit"

    ready = wait_for(lambda: status_bit(read_state(env), "ready"), timeout=READY_TIMEOUT_S)
    assert ready, f"READY did not become set within {READY_TIMEOUT_S}s (status={state_field(read_state(env), 'status', '?')})"


def test_heartbeat_advances(env):
    """`heartbeat` is incremented every Unity Update, even while paused between steps."""
    first = int(state_field(read_state(env), "heartbeat"))
    advanced = wait_for(
        lambda: int(state_field(read_state(env), "heartbeat")) != first,
        timeout=HEARTBEAT_TIMEOUT_S,
    )
    second = int(state_field(read_state(env), "heartbeat"))
    assert advanced, f"heartbeat stuck at {first} for {HEARTBEAT_TIMEOUT_S}s - game frozen or mod dead"
    assert second != first


def test_in_game_after_connect(env):
    """IN_GAME is set once connected, both in the raw status and in the info dict."""
    assert status_bit(read_state(env), "in_game"), "IN_GAME bit clear right after connect"

    (_obs, info), = step_n(env, 1)
    assert_info_contract(info)
    assert bool(info["in_game"]) is True
    assert info["human_override"] in (False, 0), "human override active - press F10 in game to return control"
