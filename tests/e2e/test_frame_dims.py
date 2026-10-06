"""Frame dimensions are honoured for each env instance."""

from __future__ import annotations

import numpy as np
import pytest

from rainworld_rl import RainWorldEnv

DIMS = [(320, 180), (84, 84)]


def _check(obs: np.ndarray, width: int, height: int, what: str) -> None:
    assert isinstance(obs, np.ndarray), f"{what}: not an ndarray"
    assert obs.shape == (height, width, 3), f"{what}: shape {obs.shape} != {(height, width, 3)}"
    assert obs.dtype == np.uint8
    assert np.any(obs != 0), f"{what}: frame is all zeros"


@pytest.mark.parametrize("width,height", DIMS, ids=[f"{w}x{h}" for w, h in DIMS])
def test_frame_dims(width, height, reattach_shared_env):
    """
    A second env with its own dims attaches to the running game (no save wipe) and
    receives frames of the requested size. `reattach_shared_env` restores the
    session env afterwards.
    """
    env = RainWorldEnv(frame_width=width, frame_height=height, ticks_per_step=1)
    try:
        env.connect()
        obs, info = env.reset(options={"wipe": False})
        _check(obs, width, height, "reset(wipe=False) obs")
        assert tuple(env.observation_space.shape) == (height, width, 3)

        obs2, _r, _t, _tr, info2 = env.step(0)
        _check(obs2, width, height, "step obs")
        assert int(info2["step_counter"]) >= int(info["step_counter"])
    finally:
        env.close()
