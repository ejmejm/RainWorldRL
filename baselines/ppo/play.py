"""
Run a trained PPO checkpoint in the environment and print the same metrics
training logs (reward, reward terms, rooms discovered, deaths, ...).

    python -m baselines.ppo.play --ckpt baselines/ppo/checkpoints/<run_id>/latest.eqx --steps 2000
    python -m baselines.ppo.play --ckpt ... --sample          # sample keys instead of greedy
    python -m baselines.ppo.play --ckpt ... --fake --steps 50 # no game

The model architecture / frame settings are read from ``model_config.json``
next to the checkpoint, so the env is built to match.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import jax.random as jr
import mlflow
import numpy as np

import rainworld_rl

from baselines.common.env import attach, make_env
from baselines.common.game_lock import game_lock
from baselines.common.metrics import RolloutMetrics
from baselines.common.runner import add_common_env_args, run_steps
from baselines.ppo.train_ppo import DEFAULT_MLFLOW_URI, greedy_action, load_checkpoint, sample_action, setup_mlflow

logger = logging.getLogger("baselines.play")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog = "python -m baselines.ppo.play",
        description = "Play a PPO checkpoint in Rain World and print metrics.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type = str, required = True, help = "Path to a model_*.eqx / latest.eqx checkpoint")
    p.add_argument("--steps", type = int, default = 1000)
    p.add_argument("--sample", action = "store_true", help = "Sample keys from the policy instead of greedy (p > 0.5)")
    p.add_argument("--mlflow", action = "store_true", help = "Also log the metrics to an MLflow run")
    p.add_argument("--mlflow_uri", type = str, default = DEFAULT_MLFLOW_URI)
    p.add_argument("--experiment", type = str, default = "rainworld-ppo-eval")
    add_common_env_args(p, with_frame_args = False)
    args = p.parse_args(argv)
    if args.seed is None:
        args.seed = random.randrange(2**31)
    return args


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("rainworld_rl imported from %s", rainworld_rl.__file__)

    model, config = load_checkpoint(Path(args.ckpt))
    logger.info("loaded %s (config: %s)", args.ckpt, config)
    rng = jr.PRNGKey(args.seed)

    def act(obs: np.ndarray) -> np.ndarray:
        nonlocal rng
        if args.sample:
            rng, key = jr.split(rng)
            action, _, _ = sample_action(model, jnp.asarray(obs), key)
        else:
            action, _, _ = greedy_action(model, jnp.asarray(obs))
        return np.asarray(action).astype(np.int8)

    if args.mlflow:
        setup_mlflow(args.mlflow_uri, args.experiment)
        mlflow.start_run(run_name = f"play-{Path(args.ckpt).parent.name}")
        mlflow.log_params({**vars(args), **{f"model_{k}": v for k, v in config.items()}})

    with game_lock(args.game_lock):
        env = make_env(
            frame_width = config["frame_width"], frame_height = config["frame_height"],
            ticks_per_step = config["ticks_per_step"], frame_stack = config["frame_stack"],
            grayscale = config["grayscale"], fake = args.fake, seed = args.seed,
        )
        try:
            attach(env, launch = args.launch)
            metrics = RolloutMetrics()
            obs, info = env.reset(options = {"wipe": not args.no_wipe})
            metrics.start(info)
            run_steps(env, obs, act, args.steps, metrics, print_every = args.print_every, log_mlflow = args.mlflow)
        finally:
            if args.mlflow:
                mlflow.end_run()
            try:
                env.close()
            finally:
                if args.kill_game and not args.fake:
                    from rainworld_rl.launcher import kill_game
                    kill_game()
    return 0


if __name__ == "__main__":
    sys.exit(main())
