"""
Gymnasium environment wrapper for Rain World RL.

Lifecycle
---------
``RainWorldEnv(...)`` is cheap and never touches the game. Attach with one of:

* ``env.launch()``  - build the mod, (re)start ``RainWorld.exe``, wait for the
  mod, then connect. Heavy; Steam must be running.
* ``env.connect()`` - attach to a game that is already running.
* ``env.reset()``   - connects if needed (raising ``GameNotRunningError`` with
  a hint to call ``launch()`` if nothing is running), sends the ``RESET``
  command (wipe RL save, fresh game) and returns the first frame.
  ``reset(options = {"wipe": False})`` skips the RESET command and just
  returns the current frame of whatever game is in progress.

This is a continual environment: a death is **not** a reset. ``terminated``
and ``truncated`` are always False; ``info["player_dead"]`` is a one-step
edge taken straight from the mod's status bit.

Observation = RGB frame only. All other state is in ``info``:
``player_dead``, ``karma``, ``karma_cap``, ``food``, ``player_pos`` (x, y),
``room_index``, ``cycle_number``, ``step_counter``, ``in_game``, ``ready``,
``human_override``.

While the human override (F10 in game) is active, ``step()`` blocks until it is
released instead of timing out, and logs once via ``logging``.

Importing / registration
------------------------
The repository directory is the package (``rainworld_rl``) and this module
lives in its ``python`` subpackage, so the gym entry point is
``rainworld_rl.rainworld_env:RainWorldEnv``. That requires the *parent*
of the repository to be on ``sys.path`` (e.g. ``E:/projects`` for
``E:/projects/rainworld_rl``), which is also what makes
``from rainworld_rl import RainWorldEnv`` work. ``gym.make("RainWorld-v0")``
only works after this module has been imported once (the ``gym.register``
call is at the bottom of this file).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import Config
from .shared_memory import (
    NUM_DISCRETE_ACTIONS,
    GameNotRunningError,
    ModState,
    SharedMemoryClient,
)


logger = logging.getLogger(__name__)


class RainWorldEnv(gym.Env):
    """
    Gymnasium environment for Rain World.

    Observation Space:
        Box(0, 255, (height, width, 3), uint8) - RGB frame from the game.

    Action Space:
        Discrete(18) - see ``shared_memory.DISCRETE_ACTION_NAMES``.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        frame_width: int = 160,
        frame_height: int = 90,
        ticks_per_step: int = 1,
        *,
        ready_timeout: float = 60.0,
        frame_timeout: float = 10.0,
        reset_timeout: float = 90.0,
        render_mode: Optional[str] = None,
        debug_timing: bool = False,
        config: Optional[Config] = None,
        client: Optional[SharedMemoryClient] = None,
    ):
        """
        Args:
            frame_width: Observation width in pixels (1..1920).
            frame_height: Observation height in pixels (1..1080).
            ticks_per_step: Physics ticks per ``step()`` (1..255).
            ready_timeout: Seconds ``connect()`` waits for the mod's READY bit.
            frame_timeout: Seconds ``step()`` waits for a frame (not counted
                while the human override is active).
            reset_timeout: Seconds ``reset()`` waits for the RESET command ack.
            render_mode: Only "rgb_array" is supported.
            debug_timing: Print per-phase step timing every ~2 s.
            config: Launch configuration (game_dir etc.). Loaded lazily from
                ``rainworld_rl.toml`` / defaults by ``launch()`` if None.
            client: Pre-built ``SharedMemoryClient`` (used by unit tests to
                inject a fake mapping). Normally None.
        """
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ticks_per_step = max(1, min(255, int(ticks_per_step)))
        self.ready_timeout = ready_timeout
        self.frame_timeout = frame_timeout
        self.reset_timeout = reset_timeout
        self.render_mode = render_mode
        self.debug_timing = debug_timing
        self.config = config

        self.observation_space = spaces.Box(
            low = 0, high = 255, shape = (frame_height, frame_width, 3), dtype = np.uint8
        )
        self.action_space = spaces.Discrete(NUM_DISCRETE_ACTIONS)

        self._client: SharedMemoryClient = client or SharedMemoryClient(
            frame_width, frame_height, debug_timing = debug_timing
        )
        self._last_frame: Optional[np.ndarray] = None
        self._last_state: Optional[ModState] = None
        self._game_process = None

    # -- connection management ---------------------------------------------

    @property
    def connected(self) -> bool:
        return self._client.is_connected()

    @property
    def last_state(self) -> Optional[ModState]:
        """Header snapshot from the most recent step/reset, or None."""
        return self._last_state

    def connect(self, wait_ready: bool = True) -> None:
        """
        Attach to an already-running game.

        Raises:
            GameNotRunningError: no live mod detected. Start the game yourself
                or call ``launch()``.
            ReadyTimeoutError: ``wait_ready`` and the mod never became READY.
        """
        if self.connected:
            return
        try:
            self._client.connect(wait_ready = wait_ready, ready_timeout = self.ready_timeout)
        except GameNotRunningError as e:
            raise GameNotRunningError(
                f"{e} Start Rain World with the RainWorldRL mod (Steam running), "
                "or call env.launch() to build + start it automatically."
            ) from e

    def launch(self, *, build: bool = True, restart: bool = True, wait_ready: bool = True) -> None:
        """
        Build the mod, (re)start Rain World, wait for the mod and connect.

        This is the one heavy call. Steam must already be running. Uses
        ``self.config`` (or loads ``rainworld_rl.toml`` / defaults).

        Args:
            build: Run ``dotnet build`` and deploy the DLL first.
            restart: Kill a running game before starting a new one.
            wait_ready: Wait for READY as part of the connect.

        Raises:
            LaunchError: build/start failed (message includes the BepInEx log tail).
        """
        from . import launcher  # heavy-ish imports kept out of construction

        if self.config is None:
            from .config import load_config
            self.config = load_config()

        if self.connected:
            self.disconnect()
        self._game_process = launcher.launch(self.config, build = build, restart = restart)
        self.connect(wait_ready = wait_ready)

    def disconnect(self) -> None:
        """Clear CONNECTED (the game returns to normal play). Does not close the game."""
        self._client.disconnect()

    def close(self) -> None:
        """Gymnasium close: same as ``disconnect()``. The game keeps running."""
        self.disconnect()

    # -- gym API -----------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Connect if needed, start a fresh game, return the first frame.

        Args:
            seed: Stored by gymnasium; the game itself is not seeded.
            options: ``{"wipe": bool}`` (default True). With ``wipe=True`` the
                RESET command is sent: the mod deletes the RL save, starts a
                fresh story game and acks once READY. With ``wipe=False`` no
                command is sent; the env just attaches to the game in
                progress and returns its current frame.

        The first observation is obtained with a single no-op step of
        ``ticks_per_step`` ticks.

        Raises:
            GameNotRunningError: nothing to connect to (see ``launch()``).
            CommandError: the mod rejected or did not ack the RESET.
        """
        super().reset(seed = seed)
        wipe = True if options is None else bool(options.get("wipe", True))

        # Always wait for READY before issuing commands: the mod rejects RESET
        # (command_result = ERROR) while it is still entering RL mode.
        if not self.connected:
            self.connect(wait_ready = True)
        else:
            self._client.wait_for_ready(timeout = self.ready_timeout)

        if wipe:
            logger.info("Sending RESET (wipe save, fresh game)...")
            self._client.reset_game(timeout = self.reset_timeout)

        frame, state = self._client.step(0, self.ticks_per_step, timeout = self.frame_timeout)
        self._last_frame = frame
        self._last_state = state
        return frame, state.to_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance the game by ``ticks_per_step`` ticks with the given action.

        Returns:
            observation: RGB frame.
            reward: Always 0.0 (reward shaping is left to the user).
            terminated / truncated: Always False (continual environment).
            info: See module docstring.

        Raises:
            StepTimeoutError / GameNotRunningError: the mod stopped answering.
        """
        if not self.connected:
            raise GameNotRunningError("Environment is not connected; call reset(), connect() or launch() first")

        frame, state = self._client.step(int(action), self.ticks_per_step, timeout = self.frame_timeout)
        self._last_frame = frame
        self._last_state = state
        return frame, 0.0, False, False, state.to_info()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._last_frame
        return None

    # -- knobs -------------------------------------------------------------

    def set_ticks_per_step(self, ticks: int) -> None:
        """Change the number of physics ticks per step (1..255)."""
        self.ticks_per_step = max(1, min(255, int(ticks)))

    def set_frame_dimensions(self, width: int, height: int) -> None:
        """Change the observation size; takes effect on the next step."""
        self._client.set_frame_dimensions(width, height)
        self.frame_width = width
        self.frame_height = height
        self.observation_space = spaces.Box(
            low = 0, high = 255, shape = (height, width, 3), dtype = np.uint8
        )


# Register with Gymnasium. See the module docstring for the import-path caveat.
_ENV_ID = "RainWorld-v0"
if _ENV_ID not in gym.registry:
    gym.register(id = _ENV_ID, entry_point = "rainworld_rl.rainworld_env:RainWorldEnv")
