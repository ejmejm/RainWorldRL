"""
Uniform-random MultiBinary agent through the same env stack and metrics as
PPO, for a comparison baseline.

    python -m baselines.random_agent --steps 1000                # attach to a running game
    python -m baselines.random_agent --steps 1000 --launch --kill_game
    python -m baselines.random_agent --steps 200 --fake          # no game
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from typing import Optional

import mlflow
import numpy as np

import rainworld_rl
from rainworld_rl.shared_memory import NUM_KEYS

from baselines.common.env import attach, make_env
from baselines.common.game_lock import game_lock
from baselines.common.metrics import RolloutMetrics
from baselines.common.runner import add_common_env_args, run_steps
from baselines.ppo.train_ppo import DEFAULT_MLFLOW_URI, setup_mlflow

logger = logging.getLogger("baselines.random_agent")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog = "python -m baselines.random_agent",
        description = "Random MultiBinary(9) agent for Rain World RL (comparison baseline).",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--steps", type = int, default = 1000)
    p.add_argument("--mlflow", action = "store_true", help = "Also log the metrics to an MLflow run")
    p.add_argument("--mlflow_uri", type = str, default = DEFAULT_MLFLOW_URI)
    p.add_argument("--experiment", type = str, default = "rainworld-random")
    add_common_env_args(p, with_frame_args = True)
    args = p.parse_args(argv)
    if args.seed is None:
        args.seed = random.randrange(2**31)
    return args


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("rainworld_rl imported from %s", rainworld_rl.__file__)
    rng = np.random.default_rng(args.seed)

    def act(_obs: np.ndarray) -> np.ndarray:
        return rng.integers(0, 2, size = NUM_KEYS, dtype = np.int8)

    if args.mlflow:
        setup_mlflow(args.mlflow_uri, args.experiment)
        mlflow.start_run(run_name = "random-agent")
        mlflow.log_params(vars(args))

    with game_lock(args.game_lock):
        env = make_env(
            frame_width = args.frame_width, frame_height = args.frame_height, ticks_per_step = args.ticks_per_step,
            frame_stack = args.frame_stack, grayscale = not args.rgb, fake = args.fake, seed = args.seed,
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
