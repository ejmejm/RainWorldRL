"""
Environment construction for the baselines.

``make_env`` builds a ``RainWorldEnv`` (or ``FakeRainWorldEnv``), wraps it with
the project's default reward (``rainworld_rl.rewards.make_default_reward_env``,
which also writes ``info["reward_terms"]``), then applies the observation
wrappers used by every baseline:

* ``GrayscaleObs`` - RGB -> luma, uint8, shape ``(H, W, 1)``.
* ``FrameStack``   - keeps the last ``frame_stack`` frames and returns them
  channel-first as uint8 ``(frame_stack * C, H, W)``.

Observations stay ``uint8`` all the way into the rollout buffers; the ``/255``
float conversion happens inside the model.

Nothing here contacts the game unless ``attach()`` / ``make_env(launch=True)``
is used. The whole module is game-free when ``fake=True``.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Any, Deque, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rainworld_rl.rewards import make_default_reward_env
from rainworld_rl.shared_memory import NUM_KEYS

logger = logging.getLogger(__name__)

DEFAULT_FRAME_WIDTH = 96
DEFAULT_FRAME_HEIGHT = 54
DEFAULT_TICKS_PER_STEP = 4
DEFAULT_FRAME_STACK = 4


# ---------------------------------------------------------------------------
# Fake environment (tests, --fake runs)
# ---------------------------------------------------------------------------

class FakeRainWorldEnv(gym.Env):
    """
    Game-free stand-in for ``RainWorldEnv`` with the same spaces and ``info``
    contract: random uint8 RGB frames, occasional room changes, food pickups,
    deaths and the rare ``cycle_survived`` edge (so the default reward is
    sometimes non-zero). ``terminated``/``truncated`` are always False.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        ticks_per_step: int = DEFAULT_TICKS_PER_STEP,
        *,
        seed: Optional[int] = None,
        p_new_room: float = 0.02,
        p_eat: float = 0.03,
        p_death: float = 0.005,
        p_survive: float = 0.01,
        **_ignored: Any,
    ):
        super().__init__()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ticks_per_step = ticks_per_step
        self.observation_space = spaces.Box(0, 255, (frame_height, frame_width, 3), dtype = np.uint8)
        self.action_space = spaces.MultiBinary(NUM_KEYS)
        self._rng = np.random.default_rng(seed)
        self._p = (p_new_room, p_eat, p_death, p_survive)
        self.connected = False
        self._state: Dict[str, Any] = {}
        self._step_counter = 0
        self._reset_state()

    # -- lifecycle no-ops so callers can treat it like the real env ----------
    def launch(self, **_kw: Any) -> None:
        self.connected = True

    def connect(self, **_kw: Any) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def close(self) -> None:
        self.disconnect()

    def debug_kill(self, timeout: float = 10.0) -> None:
        self._state["pending_death"] = True

    # -- internals -----------------------------------------------------------
    def _reset_state(self) -> None:
        self._state = {
            "karma": 1, "karma_cap": 5, "food": 0, "food_max": 7, "food_to_hibernate": 4,
            "room_index": 0, "cycle_number": 0, "cycle_progress": 0.0, "pending_death": False,
            "malnourished": False, "next_room": 1,
        }

    def _frame(self) -> np.ndarray:
        return self._rng.integers(0, 256, size = self.observation_space.shape, dtype = np.uint8)

    def _info(self, *, player_dead: bool = False, cycle_survived: bool = False, ready: bool = True) -> Dict[str, Any]:
        s = self._state
        return {
            "player_dead": player_dead,
            "karma": s["karma"] if ready else 0,
            "karma_cap": s["karma_cap"] if ready else 0,
            "food": s["food"] if ready else 0,
            "food_max": s["food_max"] if ready else 0,
            "food_to_hibernate": s["food_to_hibernate"] if ready else 0,
            "malnourished": s["malnourished"],
            "player_pos": (float(self._rng.uniform(0, 1000)), float(self._rng.uniform(0, 600))),
            "room_index": s["room_index"] if ready else -1,
            "cycle_number": s["cycle_number"] if ready else -1,
            "cycle_progress": s["cycle_progress"] if ready else 0.0,
            "rain": s["cycle_progress"] >= 1.0,
            "in_shelter": False,
            "cycle_survived": cycle_survived,
            "dialog_open": False,
            "step_counter": self._step_counter,
            "in_game": True,
            "ready": ready,
            "human_override": False,
        }

    # -- gym API -------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed = seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.connected = True
        wipe = True if options is None else bool(options.get("wipe", True))
        if wipe:
            self._reset_state()
        self._step_counter += 1
        return self._frame(), self._info()

    def step(self, action):
        if not self.connected:
            raise RuntimeError("FakeRainWorldEnv is not connected; call reset()/connect() first")
        s = self._state
        p_new_room, p_eat, p_death, p_survive = self._p
        u = self._rng.random(4)
        self._step_counter += 1
        s["cycle_progress"] = min(1.5, s["cycle_progress"] + 1.0 / 2000.0)

        if s["pending_death"] or u[2] < p_death:
            s["pending_death"] = False
            s["karma"] = max(0, s["karma"] - 1)
            s["food"] = 0
            s["cycle_progress"] = 0.0
            s["room_index"] = 0
            return self._frame(), 0.0, False, False, self._info(player_dead = True)
        if u[3] < p_survive and s["food"] >= s["food_to_hibernate"]:
            s["cycle_number"] += 1
            s["karma"] = min(s["karma_cap"], s["karma"] + 1)
            s["food"] -= s["food_to_hibernate"]
            s["cycle_progress"] = 0.0
            return self._frame(), 0.0, False, False, self._info(cycle_survived = True)
        if u[0] < p_new_room:
            s["room_index"] = s["next_room"]
            s["next_room"] += 1
        if u[1] < p_eat and s["food"] < s["food_max"]:
            s["food"] += 1
        return self._frame(), 0.0, False, False, self._info()


# ---------------------------------------------------------------------------
# Observation wrappers
# ---------------------------------------------------------------------------

class GrayscaleObs(gym.ObservationWrapper):
    """RGB uint8 ``(H, W, 3)`` -> luma uint8 ``(H, W, 1)`` (Rec. 601 weights)."""

    _WEIGHTS = np.array([0.299, 0.587, 0.114], dtype = np.float32)

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, _ = env.observation_space.shape
        self.observation_space = spaces.Box(0, 255, (h, w, 1), dtype = np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        luma = obs.astype(np.float32) @ self._WEIGHTS
        return np.clip(luma + 0.5, 0, 255).astype(np.uint8)[..., None]


class FrameStack(gym.Wrapper):
    """
    Keep the last ``k`` frames; observations become channel-first uint8
    ``(k * C, H, W)`` with the oldest frame first. ``reset()`` fills the stack
    with copies of the first frame.
    """

    def __init__(self, env: gym.Env, k: int = DEFAULT_FRAME_STACK):
        super().__init__(env)
        if k < 1:
            raise ValueError("frame_stack must be >= 1")
        self.k = k
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(0, 255, (k * c, h, w), dtype = np.uint8)
        self._frames: Deque[np.ndarray] = collections.deque(maxlen = k)

    def _stacked(self) -> np.ndarray:
        # (k, H, W, C) -> (k, C, H, W) -> (k*C, H, W)
        arr = np.stack(self._frames, axis = 0).transpose(0, 3, 1, 2)
        return np.ascontiguousarray(arr.reshape(-1, *arr.shape[2:]))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(obs)
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._stacked(), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Construction / attachment
# ---------------------------------------------------------------------------

def obs_shape_for(frame_width: int, frame_height: int, frame_stack: int, grayscale: bool) -> Tuple[int, int, int]:
    """Channel-first observation shape produced by ``make_env`` for these settings."""
    channels = 1 if grayscale else 3
    return (frame_stack * channels, frame_height, frame_width)


def make_env(
    frame_width: int = DEFAULT_FRAME_WIDTH,
    frame_height: int = DEFAULT_FRAME_HEIGHT,
    ticks_per_step: int = DEFAULT_TICKS_PER_STEP,
    frame_stack: int = DEFAULT_FRAME_STACK,
    grayscale: bool = True,
    launch: bool = False,
    *,
    fake: bool = False,
    seed: Optional[int] = None,
    **env_kwargs: Any,
) -> gym.Env:
    """
    Build the baseline environment stack.

    ``RainWorldEnv`` (or ``FakeRainWorldEnv`` when ``fake``) -> default reward
    wrapper (``reward_terms`` in info) -> optional ``GrayscaleObs`` ->
    ``FrameStack``. With ``launch=True`` the real game is (re)started
    (``build=False``) and connected before returning; see ``attach``.
    """
    if fake:
        base: gym.Env = FakeRainWorldEnv(frame_width, frame_height, ticks_per_step, seed = seed, **env_kwargs)
    else:
        from rainworld_rl import RainWorldEnv
        base = RainWorldEnv(frame_width, frame_height, ticks_per_step, **env_kwargs)

    env: gym.Env = make_default_reward_env(base)
    if grayscale:
        env = GrayscaleObs(env)
    env = FrameStack(env, frame_stack)
    if launch:
        attach(env, launch = True)
    return env


def attach(env: gym.Env, launch: bool, retries: int = 3, retry_sleep: float = 5.0) -> None:
    """
    Connect ``env`` to the game. ``launch=True`` (re)starts Rain World with the
    already-deployed DLL (``build=False``); otherwise attaches to a running game.

    Works around the known flake where ``connect()`` right after a launch
    raises ``GameNotRunningError`` ("heartbeat did not advance") during the
    game's synchronous initial load: the connect is retried ``retries`` times
    with ``retry_sleep`` seconds in between.
    """
    base = env.unwrapped
    if isinstance(base, FakeRainWorldEnv):
        base.connect()
        return

    from rainworld_rl import GameNotRunningError

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            if launch and attempt == 1:
                base.launch(build = False)
            else:
                base.connect()
            return
        except GameNotRunningError as e:
            last_error = e
            logger.warning("connect attempt %d/%d failed: %s", attempt, retries, e)
            time.sleep(retry_sleep)
    assert last_error is not None
    raise last_error
