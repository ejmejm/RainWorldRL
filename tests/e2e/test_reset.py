"""reset(): wipes the save, starts a fresh game, and is repeatable."""

from __future__ import annotations

import numpy as np

from tests.e2e.helpers import assert_info_contract, step_n

# A wiped save starts a brand-new story game: cycle 0 (PROTOCOL.md: -1 only if unavailable).
FRESH_SAVE_CYCLE = 0

# PROTOCOL.md does not say whether RESET zeroes `step_counter`; it only says the
# counter increments once per completed step. Accept either a counter that was
# reset (small) or one that simply continued from before the reset.
SMALL_STEP_COUNTER = 50
RESET_STEP_SLACK = 5

# Fade skipping: the mod collapses the game's fade-from-black (2 s fade + 3 s black on a
# new game) so the agent sees the room right after reset(). A pixel counts as non-black
# when any channel exceeds BLACK_LEVEL (the game's darkest palette colours are not 0).
BLACK_LEVEL = 10
MIN_NONBLACK_FRACTION = 0.05
FRAMES_ALLOWED_TO_SETTLE = 3  # the obs returned by reset(), or at worst the 3rd frame
FRAMES_CHECKED_FOR_BLACK = 20


def test_reset_starts_fresh_save_in_the_same_room(env):
    (_obs, before), = step_n(env, 1)
    obs, first = env.reset()

    assert_info_contract(first)
    assert isinstance(obs, np.ndarray) and obs.shape == tuple(env.observation_space.shape)
    assert int(first["cycle_number"]) == FRESH_SAVE_CYCLE, f"cycle_number after reset = {first['cycle_number']}"
    room_first = int(first["room_index"])
    assert room_first >= 0, "room_index unavailable after reset"
    assert bool(first["in_game"])

    counter = int(first["step_counter"])
    assert counter < SMALL_STEP_COUNTER or counter - int(before["step_counter"]) <= RESET_STEP_SLACK, (
        f"step_counter={counter} after reset (was {before['step_counter']} before)"
    )

    step_n(env, 30)
    _obs, second = env.reset()
    assert int(second["room_index"]) == room_first, (
        f"second reset landed in room {second['room_index']}, first reset was room {room_first}"
    )
    assert int(second["cycle_number"]) == FRESH_SAVE_CYCLE


def test_reset_obs_shows_the_room(env):
    """
    The fade-from-black is skipped: the reset obs (at worst the 3rd frame) has >= 5% lit pixels,
    and none of the first 20 frames after reset() is entirely black.
    """
    obs, _info = env.reset()
    frames = [np.array(obs, copy=True)]
    frames += [np.array(o, copy=True) for o, _info in step_n(env, FRAMES_CHECKED_FOR_BLACK - 1)]

    # Fraction of pixels with at least one channel above BLACK_LEVEL.
    fractions = [float(np.mean(np.max(f, axis=2) > BLACK_LEVEL)) for f in frames[:FRAMES_ALLOWED_TO_SETTLE]]
    assert max(fractions) >= MIN_NONBLACK_FRACTION, (
        f"first {FRAMES_ALLOWED_TO_SETTLE} frames after reset() are black "
        f"(non-black fractions {[f'{x:.3f}' for x in fractions]}); is the fade skip working?"
    )
    black = [i for i, f in enumerate(frames) if not np.any(f > BLACK_LEVEL)]
    assert not black, (
        f"frames {black} of the first {FRAMES_CHECKED_FOR_BLACK} after reset() are entirely black"
    )
    print(f"non-black fractions after reset: {[f'{x:.3f}' for x in fractions]}")
