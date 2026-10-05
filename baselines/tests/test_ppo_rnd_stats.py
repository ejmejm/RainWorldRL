"""RND running statistics: obs whitening/clipping, forward-filter intrinsic normalisation, warm-up (no game)."""

import time

import jax.numpy as jnp
import numpy as np

from baselines.common.env import make_env
from baselines.common.metrics import RolloutMetrics
from baselines.ppo_rnd.rnd import RewardForwardFilter, RNDStats, RunningMeanStd, latest_frame, whiten
from baselines.ppo_rnd.train_ppo_rnd import init_obs_norm


def _gym_rms(batches):
    """Reference: gym's RunningMeanStd recursion written out from scratch."""
    mean, var, count = 0.0, 1.0, 1e-4
    for b in batches:
        bm, bv, bc = np.mean(b), np.var(b), len(b)
        delta = bm - mean
        tot = count + bc
        m2 = var * count + bv * bc + delta ** 2 * count * bc / tot
        mean, var, count = mean + delta * bc / tot, m2 / tot, tot
    return mean, var, count


def test_running_mean_std_matches_numpy_over_chunks():
    data = np.random.default_rng(0).normal(3.0, 2.0, size = (500, 2, 3))
    rms = RunningMeanStd(shape = (2, 3))
    for chunk in np.array_split(data, 7):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, data.mean(0), rtol = 1e-5)
    np.testing.assert_allclose(rms.var, data.var(0), rtol = 1e-4)
    assert abs(rms.count - (500 + 1e-4)) < 1e-9


def test_reward_forward_filter_runs_along_time_and_is_never_reset():
    rff = RewardForwardFilter(0.5)
    assert [rff.update(r) for r in (1.0, 0.0, 2.0)] == [1.0, 0.5, 2.25]
    assert rff.update(1.0) == 2.125           # a later rollout continues the same discounted sum


def test_intrinsic_normalisation_divides_by_running_std_of_discounted_return():
    stats = RNDStats((1, 2, 2), gamma_int = 0.9)
    raw1 = np.array([1.0, 2.0, 3.0, 4.0])
    norm1, rets1 = stats.normalise_intrinsic(raw1)
    exp_rets1 = [1.0, 0.9 * 1.0 + 2.0, 0.9 * 2.9 + 3.0, 0.9 * 5.61 + 4.0]
    np.testing.assert_allclose(rets1, exp_rets1, rtol = 1e-12)
    _, var, _ = _gym_rms([rets1])
    np.testing.assert_allclose(norm1, raw1 / np.sqrt(var), rtol = 1e-12)   # divide only: no mean subtraction

    raw2 = np.array([0.5, 0.5])
    norm2, rets2 = stats.normalise_intrinsic(raw2)
    np.testing.assert_allclose(rets2, [0.9 * exp_rets1[-1] + 0.5, 0.9 * (0.9 * exp_rets1[-1] + 0.5) + 0.5])
    _, var, count = _gym_rms([rets1, rets2])
    np.testing.assert_allclose(norm2, raw2 / np.sqrt(var), rtol = 1e-12)
    assert abs(stats.reward_rms.count - count) < 1e-9 and abs(count - (6 + 1e-4)) < 1e-9
    np.testing.assert_allclose(stats.int_return_std, np.sqrt(var))


def test_intrinsic_normalisation_is_scale_free_and_unclipped():
    raw = np.random.default_rng(0).exponential(size = 256)
    a, _ = RNDStats((1, 1, 1), 0.99).normalise_intrinsic(raw)
    b, _ = RNDStats((1, 1, 1), 0.99).normalise_intrinsic(1000.0 * raw)
    np.testing.assert_allclose(a, b, rtol = 1e-3)              # only the initial count of 1e-4 breaks exactness
    np.testing.assert_allclose(a / a[0], raw / raw[0], rtol = 1e-9)

    stats = RNDStats((1, 1, 1), 0.99)
    stats.normalise_intrinsic(np.full(1000, 0.01))             # a long, boring stretch ...
    spike, _ = stats.normalise_intrinsic(np.array([100.0]))    # ... then something novel
    assert spike[0] > 5.0                                      # no clipping of the bonus
    np.testing.assert_allclose(spike[0], 100.0 / stats.int_return_std, rtol = 1e-12)


def test_whiten_is_per_pixel_and_clipped_to_five():
    frames = np.array([[[[0, 10], [100, 255]]], [[[50, 60], [70, 80]]]], dtype = np.uint8)   # (2, 1, 2, 2)
    mean = np.array([[[0.0, 200.0], [50.0, 100.0]]], dtype = np.float32)
    std = np.array([[[1.0, 2.0], [5.0, 10.0]]], dtype = np.float32)
    got = np.asarray(whiten(jnp.asarray(frames), jnp.asarray(mean), jnp.asarray(std)))
    expected = np.clip((frames.astype(np.float32) - mean) / std, -5.0, 5.0)
    np.testing.assert_allclose(got, expected, rtol = 1e-6)
    assert got.max() == 5.0 and got.min() == -5.0
    assert got[0, 0, 0, 0] == 0.0 and got[0, 0, 1, 0] == 5.0 and got[1, 0, 0, 1] == -5.0 and got[1, 0, 1, 1] == -2.0


def test_latest_frame_takes_only_the_newest_frame_of_the_stack():
    gray = np.arange(4 * 3 * 5).reshape(4, 3, 5)
    np.testing.assert_array_equal(latest_frame(gray, 1), gray[3:4])
    batch = np.stack([gray, gray + 1000])
    np.testing.assert_array_equal(latest_frame(batch, 1), batch[:, 3:4])
    rgb = np.arange(6 * 3 * 5).reshape(6, 3, 5)               # 2 RGB frames, oldest first
    np.testing.assert_array_equal(latest_frame(rgb, 3), rgb[3:6])


def test_obs_norm_warmup_folds_random_policy_frames_into_the_pixel_stats():
    env = make_env(frame_width = 48, frame_height = 36, fake = True, seed = 0)
    obs, info = env.reset()
    metrics = RolloutMetrics()
    metrics.start(info)
    stats = RNDStats((1, 36, 48), 0.99)
    obs, taken = init_obs_norm(env, obs, stats, 50, 16, 1, metrics, np.random.default_rng(0))
    assert taken == 50 and metrics.total_steps == 50 and obs.shape == (4, 36, 48)
    assert abs(stats.obs_rms.count - (50 + 1e-4)) < 1e-9
    assert stats.obs_rms.mean.shape == (1, 36, 48)
    assert 90.0 < stats.obs_rms.mean.mean() < 165.0            # fake frames are uniform noise
    assert stats.reward_rms.count == 1e-4 and stats.rff.rewems is None   # warm-up touches obs stats only

    stats2 = RNDStats((1, 36, 48), 0.99)
    _, taken = init_obs_norm(env, obs, stats2, 50, 16, 1, metrics, np.random.default_rng(0),
                             deadline = time.perf_counter() - 1.0)
    assert taken == 0 and stats2.obs_rms.count == 1e-4


def test_stats_roundtrip(tmp_path):
    stats = RNDStats((1, 3, 4), 0.99)
    stats.update_obs(np.random.default_rng(0).integers(0, 256, size = (10, 1, 3, 4)))
    stats.normalise_intrinsic(np.array([1.0, 2.0, 3.0]))
    stats.save(tmp_path / "s.npz")
    loaded = RNDStats.load(tmp_path / "s.npz")
    np.testing.assert_array_equal(loaded.obs_rms.mean, stats.obs_rms.mean)
    np.testing.assert_array_equal(loaded.obs_rms.var, stats.obs_rms.var)
    assert loaded.obs_rms.count == stats.obs_rms.count and loaded.reward_rms.count == stats.reward_rms.count
    assert loaded.rff.rewems == stats.rff.rewems and loaded.rff.gamma == 0.99
    assert RNDStats.load(_save_fresh(tmp_path)).rff.rewems is None


def _save_fresh(tmp_path):
    path = tmp_path / "fresh.npz"
    RNDStats((1, 2, 2), 0.9).save(path)
    return path
