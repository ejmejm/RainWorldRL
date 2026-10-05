"""DCEO update step and an end-to-end --fake training run that writes a checkpoint (no game)."""

import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from baselines.dceo import train_dceo as T
from baselines.dceo.replay import FrameReplay

REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL = ["--fake", "--seed", "0", "--frame_width", "48", "--frame_height", "36", "--hidden", "32", "--d", "4",
         "--batch_size", "8", "--rep_batch", "16"]


def filled_replay(args, n = 200, seed = 0):
    rng = np.random.default_rng(seed)
    rb = FrameReplay(n + 1, (args.frame_height, args.frame_width), args.frame_stack, seed = seed)
    for i in range(n):
        rb.add(rng.integers(0, 256, (args.frame_height, args.frame_width), dtype = np.uint8),
               int(rng.integers(512)), float(rng.normal()), i % 50 == 49, i % 50 == 49)
    return rb


def leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def changed(a, b):
    return any(not np.allclose(x, y) for x, y in zip(leaves(a), leaves(b)))


def test_train_step_updates_every_network_and_keeps_targets():
    args = T.parse_args(SMALL)
    ts = T.init_experiment(args)
    assert len(ts.option_dims) == 2 * (args.d - 1)
    batch = T.sample_batch(filled_replay(args), args)
    new, m = T.train_step(ts, batch)
    for name in ("main_loss", "skill_loss", "rep_loss", "graph_loss", "inv_loss", "orth_error"):
        assert math.isfinite(float(getattr(m, name))), name
    assert np.isfinite(np.asarray(m.intr_mean)).all() and np.asarray(m.intr_mean).shape == (len(ts.option_dims),)
    assert changed(ts.main, new.main) and changed(ts.skill, new.skill) and changed(ts.rep, new.rep)
    assert not np.allclose(np.asarray(ts.duals), np.asarray(new.duals))
    assert not changed(ts.main_target, new.main_target) and not changed(ts.skill_target, new.skill_target)
    assert int(new.step) == 1
    synced = T.sync_targets(new)
    assert not changed(synced.main, synced.main_target) and not changed(synced.skill, synced.skill_target)


def test_parallel_update_is_the_same_maths():
    args = T.parse_args(SMALL)
    ts = T.init_experiment(args)
    batch = T.sample_batch(filled_replay(args, seed = 1), args)
    seq_state, seq_m = T.train_step(ts, batch)
    with ThreadPoolExecutor(3) as pool:
        par_state, par_m = T.train_step(ts, batch, pool)
    for a, b in zip(leaves((seq_state, seq_m)), leaves((par_state, par_m))):
        np.testing.assert_allclose(a, b, rtol = 1e-5, atol = 1e-6)


def _run(args, timeout = 900):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-m", *args], cwd = str(REPO_ROOT), capture_output = True, text = True,
        encoding = "utf-8", errors = "replace", timeout = timeout, env = env,
    )
    assert proc.returncode == 0, f"{' '.join(args)} failed:\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-6000:]}"
    return proc


def _logged(stdout: str, key: str) -> float:
    match = re.search(rf"^\s+{re.escape(key)}\s+(\S+)\s*$", stdout, flags = re.M)
    assert match, f"{key} not in output"
    return float(match.group(1))


def test_fake_training_run_end_to_end(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    mlflow_uri = "sqlite:///" + (tmp_path / "mlruns" / "mlflow.db").as_posix()
    common = SMALL + ["--min_replay", "150", "--replay_capacity", "1000", "--log_interval", "200",
                      "--target_update_period", "100", "--mlflow_uri", mlflow_uri]
    proc = _run(["baselines.dceo.train_dceo", *common, "--total_steps", "600", "--ckpt_interval", "300",
                 "--ckpt_dir", str(ckpt_dir)])
    for key in ("dceo/main_loss", "dceo/skill_loss", "rep/graph_loss", "rep/inv_loss", "rep/eig_est_1",
                "intrinsic/abs_mean", "explore/frac_option"):
        assert math.isfinite(_logged(proc.stdout, key)), key
    assert _logged(proc.stdout, "env/total_steps") == 600.0
    assert _logged(proc.stdout, "explore/epsilon") < 1.0
    assert (ckpt_dir / "latest.eqx").is_file() and (ckpt_dir / "model_000000300.eqx").is_file()
    assert (ckpt_dir / "model_000000600.eqx").is_file() and (tmp_path / "mlruns" / "mlflow.db").is_file()

    models, config = T.load_checkpoint(ckpt_dir / "latest.eqx")
    assert config["d"] == 4 and models.duals.shape == (4, 4)
    obs = np.zeros((4, 36, 48), np.uint8)
    assert models.rep(obs).shape == (4,) and models.skill(obs).shape == (6, 512) and models.main(obs).shape == (1, 512)

    # the reference-style variants: synchronous single-thread updates, separate constraint batches, NaP,
    # one update per env step, one-directional options, death as terminal, masking off
    proc = _run(["baselines.dceo.train_dceo", *common, "--total_steps", "250", "--ckpt_dir", str(tmp_path / "ckpt2"),
                 "--no_async_updates", "--no_parallel_updates", "--separate_constraint_batches", "--nap",
                 "--updates_per_step", "1.0", "--option_directions", "positive", "--death_terminal",
                 "--no_mask_death", "--no_mask_not_ready"])
    assert _logged(proc.stdout, "time/updates") > 0
    assert math.isfinite(_logged(proc.stdout, "rep/total_loss"))
