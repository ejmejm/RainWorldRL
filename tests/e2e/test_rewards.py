"""Live smoke test of the default reward (DriveReward) on the real game: 300 random steps."""

from __future__ import annotations

import numpy as np

from tests.harness import KEY_ALL_MASK, assert_info_contract

RANDOM_STEPS = 300


def test_drive_reward_over_random_steps(env):
    """
    Wrap the shared env with make_default_reward_env, take 300 random key combinations and print the
    per-term totals. NewRoom should fire at least once if the slugcat happens to change rooms; otherwise
    it is 0 - the test only checks the plumbing (breakdown present, finite, Death/Malnourished <= 0,
    Food/NewRoom/Sleep >= 0, and the sum of the breakdown equals the returned reward).
    """
    from rainworld_rl.rewards import DriveReward, make_default_reward_env

    rng = np.random.default_rng(0)
    renv = make_default_reward_env(env)
    assert isinstance(renv, DriveReward)
    assert renv.ticks_per_step == env.ticks_per_step
    renv.reset(options = {"wipe": False})       # attach to the game in progress; no save wipe

    totals = {t.name: 0.0 for t in renv.terms}
    rooms = set()
    for _ in range(RANDOM_STEPS):
        action = int(rng.integers(0, KEY_ALL_MASK + 1)) & ~0x80     # any combination, but no map key
        _obs, reward, terminated, truncated, info = renv.step(action)
        assert (terminated, truncated) == (False, False)
        assert_info_contract(info)
        breakdown = info["reward_terms"]
        assert set(breakdown) == set(totals)
        assert all(np.isfinite(v) for v in breakdown.values())
        assert abs(sum(breakdown.values()) - reward) < 1e-9
        assert breakdown["Death"] <= 0 and breakdown["Malnourished"] <= 0
        assert breakdown["Food"] >= 0 and breakdown["NewRoom"] >= 0 and breakdown["Sleep"] >= 0
        for k, v in breakdown.items():
            totals[k] += v
        if info["ready"]:
            rooms.add((info["region"], info["room_index"]))

    print(f"[rewards] {RANDOM_STEPS} random steps: reward_terms totals = "
          + ", ".join(f"{k}={v:.4f}" for k, v in totals.items())
          + f"; rooms seen = {sorted(rooms)}")
    if len(rooms) > 1:
        assert totals["NewRoom"] >= 1.0, totals
    else:
        assert totals["NewRoom"] == 0.0, totals
