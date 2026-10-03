"""Stepping: throughput, step_counter, frame shape, frames and position react to actions."""

from __future__ import annotations

import time

import numpy as np
import pytest

from tests.harness import (
    LEFT,
    MAP,
    NOOP,
    RIGHT,
    assert_info_contract,
    bits_to_vector,
    infos,
    step_n,
)

# 200 steps at ticks_per_step=1 should take well under this even on a slow machine;
# a step that silently times out and returns a stale frame is caught by the counter check.
MAX_SECONDS_FOR_200_STEPS = 120.0


def _assert_consecutive(counters: list[int]) -> None:
    deltas = [b - a for a, b in zip(counters, counters[1:])]
    bad = [(i, d) for i, d in enumerate(deltas) if d != 1]
    assert not bad, f"step_counter did not advance by exactly 1 at (index, delta): {bad[:10]}"


def test_200_noop_steps_complete(env):
    start = time.monotonic()
    results = step_n(env, 200)
    elapsed = time.monotonic() - start

    assert len(results) == 200
    assert elapsed < MAX_SECONDS_FOR_200_STEPS, f"200 steps took {elapsed:.1f}s"
    _assert_consecutive(infos(results, "step_counter"))
    for _obs, info in results[:3]:
        assert_info_contract(info)


def test_step_counter_increments_by_one(env):
    (_obs, baseline), = step_n(env, 1)
    results = step_n(env, 25)
    counters = [int(baseline["step_counter"])] + [int(c) for c in infos(results, "step_counter")]
    _assert_consecutive(counters)


def test_frame_shape_and_dtype(env):
    (obs, _info), = step_n(env, 1)
    expected = (env.frame_height, env.frame_width, 3)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == expected, f"frame shape {obs.shape} != requested {expected}"
    assert obs.shape == tuple(env.observation_space.shape)
    assert obs.dtype == np.uint8
    assert np.any(obs != 0), "frame is all zeros (capture not working?)"


def test_frames_change_with_alternating_actions(env):
    (first, _info), = step_n(env, 1, lambda i: NOOP)
    first = np.array(first, copy=True)  # the env may reuse its buffer
    results = step_n(env, 50, lambda i: LEFT if i % 2 == 0 else RIGHT)
    differing = [i for i, (obs, _info) in enumerate(results) if np.any(obs != first)]
    assert differing, "none of 50 frames differed from the first - is the game actually advancing?"


def test_step_accepts_multibinary_vectors_and_sampled_actions(env):
    """The native action space is MultiBinary; vectors, numpy samples and int bitmasks all work."""
    import gymnasium.spaces as spaces

    assert isinstance(env.action_space, spaces.MultiBinary)
    assert env.action_space.shape == (len(bits_to_vector(0)),)

    results = step_n(env, 3, lambda i: bits_to_vector(RIGHT))        # python list
    results += step_n(env, 3, lambda i: np.array(bits_to_vector(RIGHT), dtype=np.int8))  # ndarray
    results += step_n(env, 5, lambda i: env.action_space.sample())  # random key combos
    results += step_n(env, 2, lambda i: RIGHT)                       # int bitmask
    _assert_consecutive([int(c) for c in infos(results, "step_counter")])
    for _obs, info in results:
        assert_info_contract(info)


def test_map_key_does_not_break_stepping(env):
    """Holding MAP opens the in-game map overlay; steps must keep being serviced and the game must stay up."""
    results = step_n(env, 30, lambda i: MAP)
    results += step_n(env, 10)  # release
    _assert_consecutive([int(c) for c in infos(results, "step_counter")])
    assert all(bool(v) for v in infos(results, "in_game"))
    assert "dialog_open" in results[-1][1]
    assert not results[-1][1]["dialog_open"], "no prompt should be open after releasing the map key"


@pytest.mark.xfail(
    strict=False,
    reason="start room geometry may block horizontal movement (wall, pipe, sleeping); best effort",
)
def test_player_pos_responds_to_direction(env):
    """Holding RIGHT should move +x and holding LEFT should move -x (with tolerance)."""
    hold = 20
    tolerance = 1.0  # room units; slugcat moves several units per tick when free

    (_obs, settled), = step_n(env, 5)[-1:]
    x0 = float(settled["player_pos"][0])
    room0 = settled["room_index"]

    right = step_n(env, hold, lambda i: RIGHT)
    x_right = float(right[-1][1]["player_pos"][0])
    left = step_n(env, hold, lambda i: LEFT)
    x_left = float(left[-1][1]["player_pos"][0])

    rooms = {room0, right[-1][1]["room_index"], left[-1][1]["room_index"]}
    if len(rooms) > 1:
        pytest.xfail(f"room changed during the test ({rooms}); positions are not comparable")

    dx_right = x_right - x0
    dx_left = x_left - x_right
    assert dx_right > tolerance, f"holding RIGHT for {hold} steps moved x by {dx_right:+.1f}"
    assert dx_left < -tolerance, f"holding LEFT for {hold} steps moved x by {dx_left:+.1f}"
