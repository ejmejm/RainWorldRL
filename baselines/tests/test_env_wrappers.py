"""Fake env, grayscale and frame-stack wrappers, metrics tracker (no game)."""

import numpy as np

from baselines.common.env import FakeRainWorldEnv, FrameStack, GrayscaleObs, make_env, obs_shape_for
from baselines.common.metrics import RolloutMetrics


def test_fake_env_spaces_and_info_contract():
    env = FakeRainWorldEnv(96, 54, 4, seed = 0)
    obs, info = env.reset()
    assert obs.shape == (54, 96, 3) and obs.dtype == np.uint8
    for key in ("player_dead", "karma", "food", "food_max", "room_index", "cycle_number", "cycle_survived",
                "in_game", "ready", "step_counter", "malnourished", "dialog_open"):
        assert key in info
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    assert reward == 0.0 and term is False and trunc is False


def test_make_env_fake_obs_shape_and_reward_terms():
    env = make_env(frame_width = 96, frame_height = 54, frame_stack = 4, grayscale = True, fake = True, seed = 1)
    assert env.observation_space.shape == obs_shape_for(96, 54, 4, True) == (4, 54, 96)
    obs, info = env.reset()
    assert obs.shape == (4, 54, 96) and obs.dtype == np.uint8
    obs, reward, _, _, info = env.step(np.zeros(9, dtype = np.int8))
    assert "reward_terms" in info and isinstance(info["reward_terms"], dict)
    assert isinstance(reward, float)

    rgb = make_env(frame_stack = 2, grayscale = False, fake = True)
    assert rgb.observation_space.shape == (6, 54, 96)


def test_grayscale_luma():
    env = GrayscaleObs(FakeRainWorldEnv(8, 4))
    rgb = np.zeros((4, 8, 3), dtype = np.uint8)
    rgb[..., 0] = 255  # pure red -> 0.299 * 255 = 76.2
    out = env.observation(rgb)
    assert out.shape == (4, 8, 1) and out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 76
    white = env.observation(np.full((4, 8, 3), 255, dtype = np.uint8))
    assert int(white[0, 0, 0]) == 255


def test_frame_stack_orders_oldest_first_and_fills_on_reset():
    base = FakeRainWorldEnv(8, 4, seed = 0)
    env = FrameStack(GrayscaleObs(base), k = 3)
    obs, _ = env.reset()
    assert obs.shape == (3, 4, 8)
    assert np.array_equal(obs[0], obs[1]) and np.array_equal(obs[1], obs[2])
    first = obs[0].copy()
    obs2, *_ = env.step(np.zeros(9, dtype = np.int8))
    assert np.array_equal(obs2[0], first) and np.array_equal(obs2[1], first)
    obs3, *_ = env.step(np.zeros(9, dtype = np.int8))
    assert np.array_equal(obs3[0], first) and np.array_equal(obs3[1], obs2[2])


def test_rollout_metrics_counts_edges_rooms_and_food():
    m = RolloutMetrics()
    base = dict(ready = True, in_game = True, room_index = 0, cycle_number = 0, food = 0, karma = 1,
                player_dead = False, cycle_survived = False, reward_terms = {"CycleSurvived": 0.0})
    m.start(base)
    m.update({**base, "room_index": 5, "food": 2}, 0.0)
    m.update({**base, "room_index": 5, "food": 1}, 0.0)                      # losing food does not count
    m.update({**base, "player_dead": True, "food": 0}, 0.0)
    m.update({**base, "ready": False, "room_index": -1, "cycle_number": -1}, 0.0)
    m.update({**base, "cycle_survived": True, "karma": 2, "reward_terms": {"CycleSurvived": 1.0}}, 1.0)
    s = m.summary()
    assert s["env/rooms_discovered"] == 2
    assert s["env/food_eaten"] == 2
    assert s["env/deaths"] == 1
    assert s["env/cycles_survived"] == 1
    assert s["env/karma"] == 2
    assert s["env/total_steps"] == 5
    assert s["reward_terms/CycleSurvived"] == 1.0
    assert abs(s["env/reward_mean"] - 0.2) < 1e-9
    m.end_window()
    assert m.summary()["env/reward_sum"] == 0.0 and m.summary()["env/rooms_discovered"] == 2
