"""Automatic restart: step() survives the game being killed (SIGKILL) or hanging (SIGSTOP)."""

from __future__ import annotations

import os
import signal
import time

from tests.harness import step_n

# Short so a dead or frozen game is noticed quickly (the default, 60 s, tolerates long stalls).
FRAME_TIMEOUT_S = 5.0


def test_step_restarts_killed_and_hung_game(env):
    from rainworld_rl.shared_memory import default_shm_path

    pidfile = default_shm_path(env.instance) + ".pid"  # holds the game's process group id
    (_obs, before), = step_n(env, 1)
    assert before["ready"]
    old_timeout, env.frame_timeout = env.frame_timeout, FRAME_TIMEOUT_S
    try:
        for sig in (signal.SIGKILL, signal.SIGSTOP):
            with open(pidfile) as f:
                os.killpg(int(f.read()), sig)
            start = time.monotonic()
            _obs, _r, terminated, truncated, info = env.step(0)
            print(f"[restart] {sig.name}: step() returned after {time.monotonic() - start:.0f}s")
            assert info["game_restarted"] and info["ready"] and not (terminated or truncated)
            assert info["step_counter"] == 1, "the mod's step counter restarts with the game"
            # Continues from the RL save: same cycle and region (a crash does not advance the cycle).
            assert (info["cycle_number"], info["region"]) == (before["cycle_number"], before["region"])
            assert not any(i["game_restarted"] for _o, i in step_n(env, 5))
    finally:
        env.frame_timeout = old_timeout
