"""
Gymnasium environment wrapper for Rain World RL.

Lifecycle
---------
``RainWorldEnv(...)`` is cheap and never touches the game. Attach with one of:

* ``env.launch()``  - build the mod, (re)start ``RainWorld.exe``, wait for the
  mod, then connect. Heavy.
* ``env.connect()`` - attach to a game that is already running.
* ``env.reset()``   - connects if needed (raising ``GameNotRunningError`` with
  a hint to call ``launch()`` if nothing is running), sends the ``RESET``
  command (wipe RL save, fresh game) and returns the first frame.
  ``reset(options = {"wipe": False})`` skips the RESET command and just
  returns the current frame of whatever game is in progress.

This is a continual environment: a death is **not** a reset. ``terminated``
and ``truncated`` are always False; ``info["player_dead"]`` is a one-step
edge taken straight from the mod's status bit.

Automatic restart (``auto_restart = True``, the default): when ``step()`` finds
the game dead or hung (``GameNotRunningError`` / ``StepTimeoutError``), or
``ready`` has been False for more than ``stuck_timeout`` (120 s wall time, F10
override excluded; deaths, sleeps and region loads drop it for a few seconds),
it kills and relaunches the game (``build = False``), waits for READY and
returns the reloaded game's first frame with ``info["game_restarted"] = True``.
The RL save is kept, so play continues from the last save (the game books an
unfinished cycle as a death, so karma may drop), and the mod's
``step_counter`` starts again from 0. After ``restart_attempts`` (3) failed
relaunches in a row ``step()`` raises ``LaunchError``. ``reset()`` and
``debug_kill()`` never restart.

``env.debug_kill()`` is a **debug/testing** hook that kills the slugcat on
demand so the death -> respawn flow can be exercised deterministically. It is
not part of the RL interface and is not something an agent should call.
``env.debug_enter_shelter(food)`` is its counterpart for the sleep flow: it
sends the slugcat into its den shelter so a real hibernation can be tested.

Observation = RGB frame only. All other state is in ``info``:
``player_dead``, ``karma``, ``karma_cap``, ``food``, ``food_max``,
``player_pos`` (x, y), ``room_index``, ``cycle_number``, ``step_counter``,
``in_game``, ``ready``, ``human_override``, ``in_shelter``, ``cycle_survived``,
``rain``, ``dialog_open``, ``cycle_progress``, ``game_restarted``.

Actions
-------
``action_space = MultiBinary(NUM_KEYS)``: one bit per key the player can hold
(``shared_memory.KEY_NAMES`` gives the bit order: left, right, up, down, jump,
grab, throw, map, special). Any combination may be held in one step.
``step()`` also accepts a plain ``int`` bitmask of ``KEY_*`` values.
The classic Discrete(18) set is available as ``rainworld_rl.wrappers.DiscreteActions``.

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
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import Config
from .shared_memory import (
    KEY_NAMES,
    NUM_KEYS,
    GameNotRunningError,
    ModState,
    SharedMemoryClient,
    SharedMemoryError,
    StepTimeoutError,
    action_to_bits,
    default_shm_path,
)


logger = logging.getLogger(__name__)


class RainWorldEnv(gym.Env):
    """
    Gymnasium environment for Rain World.

    Observation Space:
        Box(0, 255, (height, width, 3), uint8) - RGB frame from the game.

    Action Space:
        MultiBinary(NUM_KEYS) - one entry per key in ``shared_memory.KEY_NAMES``
        (left, right, up, down, jump, grab, throw, map, special); any
        combination at once. ``step()`` also takes an ``int`` bitmask of
        ``shared_memory.KEY_*``. Wrap with ``wrappers.DiscreteActions`` for Discrete(18).
    """

    key_names = KEY_NAMES

    metadata = {"render_modes": ["rgb_array"]}

    # auto_restart: seconds ``ready`` may stay False (wall time, F10 override excluded)
    # before step() treats the game as stuck and restarts it.
    stuck_timeout: float = 120.0
    # auto_restart: failed relaunches in a row before step() gives up and raises.
    restart_attempts: int = 3

    def __init__(
        self,
        frame_width: int = 160,
        frame_height: int = 90,
        ticks_per_step: int = 4,
        *,
        ready_timeout: float = 60.0,
        frame_timeout: float = 60.0,
        reset_timeout: float = 90.0,
        render_mode: Optional[str] = None,
        debug_timing: bool = False,
        config: Optional[Config] = None,
        client: Optional[SharedMemoryClient] = None,
        instance: int = 0,
        auto_restart: bool = True,
    ):
        """
        Args:
            frame_width: Observation width in pixels (1..1920).
            frame_height: Observation height in pixels (1..1080).
            ticks_per_step: Physics ticks per ``step()`` (1..255; the game runs 40
                ticks/s, so the default 4 is 100 ms of game time per step).
            ready_timeout: Seconds ``connect()`` waits for the mod's READY bit.
            frame_timeout: Seconds ``step()`` waits for a frame (not counted
                while the human override is active) before it restarts the
                game (or raises, see ``auto_restart``).
            reset_timeout: Seconds ``reset()`` waits for the RESET command ack.
            render_mode: Only "rgb_array" is supported.
            debug_timing: Print per-phase step timing every ~2 s.
            config: Launch configuration (game_dir etc.). Loaded lazily from
                ``rainworld_rl.toml`` / defaults by ``launch()`` if None.
            client: Pre-built ``SharedMemoryClient`` (used by unit tests to
                inject a fake mapping). Normally None.
            instance: Which game instance to drive (Linux only; see
                ``launcher``). Each has its own shared memory.
            auto_restart: Let ``step()`` restart a dead, hung or stuck game
                (see the module docstring). False: ``step()`` raises instead.
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
        self.instance = instance
        self.auto_restart = auto_restart

        self.observation_space = spaces.Box(
            low = 0, high = 255, shape = (frame_height, frame_width, 3), dtype = np.uint8
        )
        self.action_space = spaces.MultiBinary(NUM_KEYS)

        self._client: SharedMemoryClient = client or SharedMemoryClient(
            frame_width, frame_height, debug_timing = debug_timing, shm_path = default_shm_path(instance)
        )
        self._last_frame: Optional[np.ndarray] = None
        self._last_state: Optional[ModState] = None
        self._not_ready_since: Optional[float] = None  # monotonic time READY was first seen down
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
                f"{e} Start Rain World with the RainWorldRL mod, "
                "or call env.launch() to build + start it automatically."
            ) from e

    def launch(self, *, build: Optional[bool] = None, restart: bool = True, wait_ready: bool = True) -> None:
        """
        Build the mod, (re)start Rain World, wait for the mod and connect.

        This is the one heavy call. Uses
        ``self.config`` (or loads ``rainworld_rl.toml`` / defaults).

        Args:
            build: ``dotnet build`` and deploy the DLL first. None (default):
                build when possible, else deploy the packaged prebuilt DLL;
                False: leave the deployed DLL alone (see ``launcher.launch``).
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
        self._game_process = launcher.launch(self.config, build = build, restart = restart, instance = self.instance)
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
        return frame, self._observe(frame, state)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance the game by ``ticks_per_step`` ticks holding the given keys.

        Args:
            action: A ``MultiBinary(NUM_KEYS)`` vector (0/1 per key, ``KEY_NAMES``
                order) or an ``int`` bitmask of ``shared_memory.KEY_*`` values.

        Returns:
            observation: RGB frame.
            reward: Always 0.0 (reward shaping is left to the user).
            terminated / truncated: Always False (continual environment).
            info: See module docstring. ``info["game_restarted"]`` is True on
                the step that restarted the game, False otherwise.

        Raises:
            StepTimeoutError / GameNotRunningError: the mod stopped answering
                (only with ``auto_restart = False``).
            LaunchError: ``restart_attempts`` relaunches in a row failed.
        """
        if not self.connected:
            raise GameNotRunningError("Environment is not connected; call reset(), connect() or launch() first")

        bits = action_to_bits(action)
        try:
            frame, state = self._client.step(bits, self.ticks_per_step, timeout = self.frame_timeout)
        except (StepTimeoutError, GameNotRunningError) as e:
            if not self.auto_restart:
                raise
            return self._restart_game(f"{type(e).__name__}: {e}")
        info = self._observe(frame, state)
        if self.auto_restart and self._not_ready_since is not None:
            down = time.monotonic() - max(self._not_ready_since, self._client.last_override_time)
            if down > self.stuck_timeout:
                return self._restart_game(f"ready has been False for {down:.0f}s (stuck in a screen?)")
        return frame, 0.0, False, False, info

    def _observe(self, frame: np.ndarray, state: ModState, restarted: bool = False) -> Dict[str, Any]:
        """Remember the frame and state, track since when READY is down, build ``info``."""
        self._last_frame = frame
        self._last_state = state
        if state.ready or state.human_override:
            self._not_ready_since = None
        elif self._not_ready_since is None:
            self._not_ready_since = time.monotonic()
        info = state.to_info()
        info["game_restarted"] = restarted
        return info

    def _restart_game(self, cause: str) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Relaunch the game (RL save kept), wait for READY and return a no-op step as ``step()`` would."""
        from .launcher import LaunchError

        logger.warning("Restarting the game: %s", cause)
        start = time.monotonic()
        for attempt in range(1, self.restart_attempts + 1):
            try:
                self.launch(build = False)  # restart = True kills the old game first
                frame, state = self._client.step(0, self.ticks_per_step, timeout = self.frame_timeout)
            except (LaunchError, SharedMemoryError) as e:
                error = e
                logger.warning("Restart attempt %d/%d failed: %s", attempt, self.restart_attempts, e)
                continue
            logger.info("Game restarted in %.0fs", time.monotonic() - start)
            return frame, 0.0, False, False, self._observe(frame, state, restarted = True)
        raise LaunchError(
            f"Could not restart the game ({cause}): {self.restart_attempts} attempts failed, last: {error}"
        ) from error

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._last_frame
        return None

    # -- debugging / testing -----------------------------------------------

    def debug_kill(self, timeout: float = 10.0) -> None:
        """
        **Debug/testing only**: kill the slugcat right now (``KILL_PLAYER``).

        Intended for tests and tooling that need a deterministic death, e.g.
        to exercise the ``player_dead`` edge and the respawn flow. Not part of
        the RL interface; an agent has no business calling it.

        Returns as soon as the mod reports the slugcat dead - the respawn is
        observed through subsequent ``step()`` calls: the next step has
        ``info["player_dead"] == True`` (one step only), then ``ready`` /
        ``in_game`` may drop while the mod skips the death screen and reloads
        the cycle, and the slugcat reappears in the start-of-cycle shelter.

        Raises:
            GameNotRunningError: not connected.
            CommandError: the mod rejected the kill (not in a game, no live
                player, already dead) or did not ack within ``timeout``.
        """
        if not self.connected:
            raise GameNotRunningError("Environment is not connected; call reset(), connect() or launch() first")
        logger.info("Sending KILL_PLAYER (debug)...")
        self._client.kill_player(timeout = timeout)

    def debug_enter_shelter(self, food: int, timeout: float = 10.0) -> None:
        """
        **Debug/testing only**: give the slugcat ``food`` pips and send it into
        its den shelter through the entrance pipe (``ENTER_SHELTER``).

        Intended for tests that need a real hibernation: from there the game's
        own shelter logic decides. With ``info["food_to_hibernate"]`` pips the
        slugcat hibernates once it stands still away from the entrance
        (``cycle_survived`` edge, then the cycle reloads in the shelter); with
        fewer (but at least one) it sleeps starving if DOWN is held for 260
        ticks. Not part of the RL interface.

        Raises:
            GameNotRunningError: not connected.
            CommandError: the mod rejected the command (not in a game, no live
                player, den shelter not in the current region) or did not ack
                within ``timeout``.
        """
        if not self.connected:
            raise GameNotRunningError("Environment is not connected; call reset(), connect() or launch() first")
        logger.info("Sending ENTER_SHELTER (debug, food=%d)...", food)
        self._client.enter_shelter(food, timeout = timeout)

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
