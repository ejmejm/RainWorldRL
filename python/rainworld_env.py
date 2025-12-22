"""
Gymnasium environment wrapper for Rain World RL.

Provides a standard Gymnasium interface for training RL agents on Rain World.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .shared_memory import SharedMemoryClient


class RainWorldEnv(gym.Env):
    """
    Gymnasium environment for Rain World.

    This environment connects to a running Rain World game with the RainWorldRL mod
    installed. The game simulation is entirely driven by calls to step().

    Observation Space:
        Box(0, 255, (height, width, 3), uint8) - RGB frame from the game

    Action Space:
        Discrete(18) - See action mapping in shared_memory.py

    Note:
        This is a continual learning environment with no episode boundaries.
        The terminated and truncated flags are always False.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        frame_width: int = 160,
        frame_height: int = 90,
        ticks_per_step: int = 1,
        connection_timeout: float = 30.0,
        frame_timeout: float = 5.0,
        render_mode: Optional[str] = None,
        debug_timing: bool = False,
    ):
        """
        Initialize the Rain World environment.

        Args:
            frame_width: Width of observation frames in pixels.
            frame_height: Height of observation frames in pixels.
            ticks_per_step: Number of physics ticks per step() call.
            connection_timeout: Timeout for connecting to the game in seconds.
            frame_timeout: Timeout for waiting for frames in seconds.
            render_mode: Render mode (only "rgb_array" is supported).
            debug_timing: If True, print detailed timing info to console.
        """
        super().__init__()

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ticks_per_step = ticks_per_step
        self.connection_timeout = connection_timeout
        self.frame_timeout = frame_timeout
        self.render_mode = render_mode
        self.debug_timing = debug_timing

        # Define spaces
        self.observation_space = spaces.Box(
            low = 0,
            high = 255,
            shape = (frame_height, frame_width, 3),
            dtype = np.uint8,
        )

        # 18 discrete actions (see shared_memory.py for mapping)
        self.action_space = spaces.Discrete(18)

        # Shared memory client
        self._client: Optional[SharedMemoryClient] = None
        self._last_frame: Optional[np.ndarray] = None
        self._connected = False

    def _ensure_connected(self):
        """Ensure connection to the game."""
        if self._connected:
            return

        self._client = SharedMemoryClient(
            self.frame_width, self.frame_height, debug_timing = self.debug_timing
        )

        if not self._client.connect(timeout = self.connection_timeout):
            raise RuntimeError(
                f"Failed to connect to Rain World within {self.connection_timeout}s. "
                "Make sure the game is running with the RainWorldRL mod."
            )

        self._connected = True

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Run one step of the environment.

        Args:
            action: Discrete action to take (0-17).

        Returns:
            observation: RGB frame from the game.
            reward: Always 0.0 (reward shaping is left to the user).
            terminated: Always False (continual environment).
            truncated: Always False (continual environment).
            info: Additional information (currently empty).
        """
        self._ensure_connected()

        # Send action to the game
        self._client.send_action_discrete(action, self.ticks_per_step)

        # Wait for frame
        frame = self._client.wait_for_frame(timeout = self.frame_timeout)

        if frame is None:
            # Timeout - return last frame or zeros
            if self._last_frame is not None:
                frame = self._last_frame
            else:
                frame = np.zeros(
                    (self.frame_height, self.frame_width, 3), dtype = np.uint8
                )
        else:
            self._last_frame = frame

        # Get status for info
        player_dead, _ = self._client.get_status()
        info = {"player_dead": player_dead}

        # Continual environment - never terminates
        return frame, 0.0, False, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Since this is a continual learning environment, reset() simply returns
        the current observation without actually resetting the game state.

        Args:
            seed: Random seed (unused).
            options: Additional options (unused).

        Returns:
            observation: Current RGB frame from the game.
            info: Additional information.
        """
        super().reset(seed = seed)

        self._ensure_connected()

        # Send no-op action to get initial frame
        self._client.send_action_discrete(0, self.ticks_per_step)
        frame = self._client.wait_for_frame(timeout = self.frame_timeout)

        if frame is None:
            frame = np.zeros(
                (self.frame_height, self.frame_width, 3), dtype = np.uint8
            )

        self._last_frame = frame
        return frame, {}

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            RGB frame if render_mode is "rgb_array", None otherwise.
        """
        if self.render_mode == "rgb_array":
            return self._last_frame
        return None

    def close(self):
        """Close the environment and disconnect from the game."""
        if self._client is not None:
            self._client.disconnect()
            self._client = None
            self._connected = False

    def set_ticks_per_step(self, ticks: int):
        """
        Change the number of physics ticks per step.

        Args:
            ticks: New ticks per step value (1-255).
        """
        self.ticks_per_step = max(1, min(255, ticks))

    def set_frame_dimensions(self, width: int, height: int):
        """
        Change the frame dimensions.

        Args:
            width: New frame width.
            height: New frame height.
        """
        self.frame_width = width
        self.frame_height = height

        # Update observation space
        self.observation_space = spaces.Box(
            low = 0,
            high = 255,
            shape = (height, width, 3),
            dtype = np.uint8,
        )

        # Update client if connected
        if self._client is not None:
            self._client.set_frame_dimensions(width, height)


# Register the environment with Gymnasium
gym.register(
    id = "RainWorld-v0",
    entry_point = "rainworld_rl.python.rainworld_env:RainWorldEnv",
)

