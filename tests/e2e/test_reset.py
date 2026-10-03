"""reset(): wipes the save, starts a fresh game, and is repeatable."""

from __future__ import annotations

import numpy as np

from tests.harness import FRESH_SAVE_CYCLE, assert_info_contract, step_n

# PROTOCOL.md does not say whether RESET zeroes `step_counter`; it only says the
# counter increments once per completed step. Accept either a counter that was
# reset (small) or one that simply continued from before the reset.
SMALL_STEP_COUNTER = 50
RESET_STEP_SLACK = 5


def test_reset_starts_fresh_save(env):
    (_obs, before), = step_n(env, 1)
    obs, info = env.reset()

    assert_info_contract(info)
    assert isinstance(obs, np.ndarray) and obs.shape == tuple(env.observation_space.shape)
    # A wiped save starts a brand-new story game: cycle 0 (PROTOCOL.md: -1 only if unavailable).
    assert int(info["cycle_number"]) == FRESH_SAVE_CYCLE, f"cycle_number after reset = {info['cycle_number']}"
    assert int(info["room_index"]) >= 0, "room_index unavailable after reset"
    assert bool(info["in_game"])

    counter = int(info["step_counter"])
    assert counter < SMALL_STEP_COUNTER or counter - int(before["step_counter"]) <= RESET_STEP_SLACK, (
        f"step_counter={counter} after reset (was {before['step_counter']} before)"
    )


def test_reset_returns_to_same_room(env):
    _obs, first = env.reset()
    room_first = int(first["room_index"])
    assert room_first >= 0

    results = step_n(env, 30)
    assert len(results) == 30

    _obs, second = env.reset()
    assert int(second["room_index"]) == room_first, (
        f"second reset landed in room {second['room_index']}, first reset was room {room_first}"
    )
    assert int(second["cycle_number"]) == FRESH_SAVE_CYCLE
