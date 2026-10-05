"""End-to-end on FakeRainWorldEnv: tiny PPO run writes a checkpoint; play.py and random_agent load/run it."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(args, timeout = 600):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-m", *args], cwd = str(REPO_ROOT), capture_output = True, text = True,
        encoding = "utf-8", errors = "replace", timeout = timeout, env = env,
    )
    assert proc.returncode == 0, f"{' '.join(args)} failed:\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-6000:]}"
    return proc


def test_train_play_random_on_fake_env(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    mlflow_uri = "sqlite:///" + (tmp_path / "mlruns" / "mlflow.db").as_posix()

    _run([
        "baselines.ppo.train_ppo", "--fake", "--total_steps", "64", "--rollout_steps", "32",
        "--log_interval", "32", "--ckpt_interval", "32", "--minibatches", "4", "--epochs", "2",
        "--frame_width", "32", "--frame_height", "18", "--seed", "0",
        "--ckpt_dir", str(ckpt_dir), "--mlflow_uri", mlflow_uri,
    ])
    assert (ckpt_dir / "model_config.json").is_file()
    assert (ckpt_dir / "latest.eqx").is_file()
    assert (ckpt_dir / "model_000000064.eqx").is_file()
    # the mlflow run landed in the temporary store, not in the repo
    assert (tmp_path / "mlruns" / "mlflow.db").is_file()

    proc = _run([
        "baselines.ppo.play", "--ckpt", str(ckpt_dir / "latest.eqx"), "--fake", "--steps", "40",
        "--print_every", "20", "--seed", "0",
    ])
    assert "env/total_steps" in proc.stdout and "40.0000" in proc.stdout

    proc = _run([
        "baselines.ppo.play", "--ckpt", str(ckpt_dir / "latest.eqx"), "--fake", "--steps", "10", "--sample",
    ])
    assert "env/rooms_discovered" in proc.stdout

    proc = _run([
        "baselines.random_agent", "--fake", "--steps", "30", "--print_every", "30",
        "--frame_width", "32", "--frame_height", "18",
    ])
    assert "env/total_steps" in proc.stdout
