"""PPO+RND update on FakeRainWorldEnv data (in-process) and a short --fake CLI run end to end."""

import math
import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from baselines.common.env import make_env
from baselines.common.metrics import RolloutMetrics
from baselines.ppo_rnd.rnd import RNDStats, intrinsic_reward, latest_frame
from baselines.ppo_rnd.train_ppo_rnd import (
    collect_rollout,
    init_experiment,
    init_obs_norm,
    load_checkpoint,
    parse_args,
    prepare_batch,
    train_step,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL = ["--frame_width", "48", "--frame_height", "36", "--rollout_steps", "64", "--minibatches", "2",
         "--epochs", "2", "--seed", "0"]


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def test_train_step_is_finite_keeps_target_fixed_and_shrinks_the_bonus():
    args = parse_args(["--fake", "--total_steps", "64", "--rnd_lr", "1e-3", *SMALL])
    ts = init_experiment(args)
    env = make_env(frame_width = 48, frame_height = 36, fake = True, seed = 0)
    obs, info = env.reset()
    metrics = RolloutMetrics()
    metrics.start(info)
    stats = RNDStats((1, 36, 48), args.int_gamma)
    obs, _ = init_obs_norm(env, obs, stats, 64, 64, 1, metrics, np.random.default_rng(0))

    rollout, obs, _, _ = collect_rollout(env, ts.agent, obs, jr.PRNGKey(1), 64, metrics, 1.0)
    assert rollout["obs"].shape == (64, 4, 36, 48) and rollout["obs"].dtype == np.uint8
    assert rollout["deaths"].shape == (64,) and set(np.unique(rollout["deaths"])) <= {0.0, 1.0}
    batch, rnd_info = prepare_batch(ts, stats, rollout, obs)
    assert all(math.isfinite(v) for v in rnd_info.values())
    assert rnd_info["rnd/int_reward_raw_mean"] > 0 and abs(stats.obs_rms.count - (128 + 1e-4)) < 1e-9

    new_ts, m = train_step(ts, batch)
    for name in ("policy_loss", "ext_value_loss", "int_value_loss", "entropy", "predictor_loss", "approx_kl",
                 "grad_norm", "total_loss", "ext_explained_variance", "int_explained_variance"):
        assert math.isfinite(float(getattr(m, name))), name
    assert 0.0 < float(m.predictor_keep_frac) <= 1.0
    assert int(new_ts.step) == 64
    for a, b in zip(_leaves(ts.target), _leaves(new_ts.target)):
        np.testing.assert_array_equal(a, b)                       # the random target is never trained
    assert any(not np.array_equal(a, b) for a, b in zip(_leaves(ts.predictor), _leaves(new_ts.predictor)))
    assert any(not np.array_equal(a, b) for a, b in zip(_leaves(ts.agent), _leaves(new_ts.agent)))

    # repeated updates on the same frames drive the predictor error on them down
    frames = jnp.asarray(latest_frame(rollout["obs"], 1))
    mean, std = (jnp.asarray(x) for x in stats.obs_mean_std())
    before = float(intrinsic_reward(ts.predictor, ts.target, frames, mean, std).mean())
    for _ in range(5):
        new_ts, m = train_step(new_ts, batch)
    after = float(intrinsic_reward(new_ts.predictor, new_ts.target, frames, mean, std).mean())
    assert after < 0.8 * before, (before, after)


def test_fake_cli_run_writes_checkpoint_and_finite_metrics(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    mlflow_uri = "sqlite:///" + (tmp_path / "mlruns" / "mlflow.db").as_posix()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-m", "baselines.ppo_rnd.train_ppo_rnd", "--fake", "--total_steps", "128",
           "--obs_norm_init_steps", "32", "--ckpt_interval", "64", "--ckpt_dir", str(ckpt_dir),
           "--mlflow_uri", mlflow_uri, *SMALL]
    proc = subprocess.run(cmd, cwd = str(REPO_ROOT), capture_output = True, text = True, encoding = "utf-8",
                          errors = "replace", timeout = 600, env = env)
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-6000:]}"

    for name in ("model_config.json", "latest.eqx", "model_000000064.eqx", "model_000000128.eqx",
                 "latest_rnd_stats.npz", "rnd_stats_000000128.npz"):
        assert (ckpt_dir / name).is_file(), name
    (agent, predictor, target), config = load_checkpoint(ckpt_dir / "latest.eqx")
    logits, v_ext, v_int = agent(jnp.zeros((4, 36, 48), dtype = jnp.uint8))
    assert logits.shape == (9,) and config["algo"] == "ppo_rnd"
    stats = RNDStats.load(ckpt_dir / "latest_rnd_stats.npz")
    assert abs(stats.obs_rms.count - (32 + 128 + 1e-4)) < 1e-6   # warm-up + both rollouts, never reset
    assert abs(stats.reward_rms.count - (128 + 1e-4)) < 1e-6

    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri = mlflow_uri)
    exp = client.get_experiment_by_name("rainworld-ppo-rnd")
    (run,) = client.search_runs([exp.experiment_id])
    metrics, params = run.data.metrics, run.data.params
    for key in ("ppo/policy_loss", "ppo/ext_value_loss", "ppo/int_value_loss", "ppo/entropy", "rnd/predictor_loss",
                "rnd/int_reward_raw_mean", "rnd/int_reward_raw_std", "rnd/int_reward_norm_mean",
                "rnd/int_reward_norm_std", "rnd/int_return_std", "rollout/deaths", "env/rooms_discovered",
                "final/env_rooms_discovered", "time/update_seconds"):
        assert key in metrics and math.isfinite(metrics[key]), key
    assert params["novelty"] == "False" and float(params["predictor_batch_effective"]) == 8.0
