"""The frame returned by step() shows the game after that step, not an earlier one.

Two capture bugs returned stale frames while every other test passed: frames one step behind (the back
buffer read before the game's canvas composite) and, with several steps per Unity frame, frames frozen at
the frame-start sprite geometry. In the fresh-save start room the slugcat is the only white object in the
lower left of the frame, so its column should follow player_pos x at lag 0, not lag 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.harness import LEFT, NOOP, RIGHT, step_n

TICKS = 10  # enough movement per step to show up in a 160x90 frame
CYCLE = [NOOP, NOOP, NOOP, RIGHT, NOOP, NOOP, NOOP, LEFT]  # each move followed by still steps
STEPS = 64


def _slugcat_column(obs: np.ndarray) -> float:
    """Mean column of the white pixels in the lower-left part of the frame (NaN if none)."""
    h, w, _ = obs.shape
    white = obs.min(axis = 2) > 140
    white[: h * 4 // 9] = False  # tutorial text appears above
    white[:, w * 3 // 8 :] = False  # an unrelated white object sits to the right
    cols = np.nonzero(white)[1]
    return float(cols.mean()) if len(cols) else float("nan")


def _fit_error(cols: np.ndarray, xs: np.ndarray) -> float:
    a = np.stack([xs, np.ones_like(xs)], axis = 1)
    coef, *_ = np.linalg.lstsq(a, cols, rcond = None)
    return float(np.abs(a @ coef - cols).mean())


def test_frame_shows_this_steps_position(fresh_env):
    env = fresh_env
    if (env.frame_width, env.frame_height) != (160, 90):
        pytest.skip("the slugcat detection is tuned for 160x90 frames")
    ticks = env.ticks_per_step
    env.set_ticks_per_step(TICKS)
    try:
        results = step_n(env, STEPS, lambda i: CYCLE[i % len(CYCLE)])
    finally:
        env.set_ticks_per_step(ticks)

    cols = np.array([_slugcat_column(obs) for obs, _ in results])
    xs = np.array([float(info["player_pos"][0]) for _, info in results])
    i = np.nonzero(~np.isnan(cols))[0]
    i = i[i > 0]
    if len(i) < STEPS // 2 or np.ptp(xs[i]) < 30:
        pytest.skip(f"slugcat found in {len(i)} frames, x range {np.ptp(xs[i]) if len(i) else 0:.0f}: start room not as expected")

    err_now = _fit_error(cols[i], xs[i])
    err_previous = _fit_error(cols[i], xs[i - 1])
    assert err_now < 0.6 and err_now < err_previous / 2, (
        f"frame column follows player x at lag 0 with {err_now:.2f}px error and at lag 1 with "
        f"{err_previous:.2f}px: frames are not showing the step just taken"
    )
