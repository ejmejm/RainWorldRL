"""Shared evaluation loop for ``play.py`` and ``random_agent.py``."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import mlflow
import numpy as np

from .metrics import RolloutMetrics, format_summary


def run_steps(
    env,
    obs: np.ndarray,
    act_fn: Callable[[np.ndarray], np.ndarray],
    steps: int,
    metrics: RolloutMetrics,
    *,
    print_every: int = 500,
    log_mlflow: bool = False,
) -> Dict[str, float]:
    """
    Run ``act_fn`` in ``env`` for ``steps`` steps, feeding ``metrics``. Prints a
    summary every ``print_every`` steps (and at the end), optionally logging
    the same dict to the active MLflow run. Returns the final cumulative summary.
    """
    summary: Dict[str, float] = {}
    for t in range(1, steps + 1):
        action = act_fn(obs)
        obs, reward, _terminated, _truncated, info = env.step(action)
        metrics.update(info, float(reward))
        if t % print_every == 0 or t == steps:
            summary = metrics.summary()
            print(f"\n[{t}/{steps} steps]\n{format_summary(summary)}")
            if log_mlflow:
                mlflow.log_metrics(summary, step = metrics.total_steps)
            metrics.end_window()
    return summary


def add_common_env_args(p, *, with_frame_args: bool = True) -> None:
    """Flags every runner shares (env attach/launch, lock, fake)."""
    p.add_argument("--fake", action = "store_true", help = "Use FakeRainWorldEnv (no game)")
    p.add_argument("--launch", action = "store_true", help = "(Re)start Rain World with the deployed DLL first")
    p.add_argument("--kill_game", action = "store_true", help = "Close the game when done")
    p.add_argument("--no_wipe", action = "store_true", help = "Attach to the game in progress instead of wiping the RL save")
    p.add_argument("--game_lock", type = str, default = None, help = "mkdir-lock directory to hold while the game is in use")
    p.add_argument("--print_every", type = int, default = 500, help = "Steps between metric printouts")
    p.add_argument("--seed", type = int, default = None, help = "Random if omitted")
    if with_frame_args:
        p.add_argument("--frame_width", type = int, default = 96)
        p.add_argument("--frame_height", type = int, default = 54)
        p.add_argument("--ticks_per_step", type = int, default = 4)
        p.add_argument("--frame_stack", type = int, default = 4)
        p.add_argument("--rgb", action = "store_true", help = "Keep RGB frames instead of grayscale")
