"""Stepping: throughput, step_counter, frames react to actions, the map key."""

from __future__ import annotations

import time

import numpy as np

from rainworld_rl import KEY_LEFT, KEY_MAP, KEY_RIGHT

from tests.e2e.helpers import assert_info_contract, infos, step_n

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


def test_frames_change_with_alternating_actions(env):
    (first, _info), = step_n(env, 1)
    first = np.array(first, copy=True)  # the env may reuse its buffer
    results = step_n(env, 50, lambda i: KEY_LEFT if i % 2 == 0 else KEY_RIGHT)
    differing = [i for i, (obs, _info) in enumerate(results) if np.any(obs != first)]
    assert differing, "none of 50 frames differed from the first - is the game actually advancing?"


def test_map_key_does_not_break_stepping(env):
    """Holding MAP opens the in-game map overlay; steps must keep being serviced and the game must stay up."""
    results = step_n(env, 30, lambda i: KEY_MAP)
    results += step_n(env, 10)  # release
    _assert_consecutive([int(c) for c in infos(results, "step_counter")])
    assert all(bool(v) for v in infos(results, "in_game"))
    assert "dialog_open" in results[-1][1]
    assert not results[-1][1]["dialog_open"], "no prompt should be open after releasing the map key"
