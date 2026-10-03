"""
Rain World RL - Python client for the Rain World reinforcement learning environment.

Install with ``pip install -e .`` from the repo root, then::

    from rainworld_rl import RainWorldEnv

    env = RainWorldEnv(160, 90, ticks_per_step = 4)   # cheap, no game contact
    env.launch()                                      # or env.connect() if the game is up
    obs, info = env.reset()                           # fresh game
"""

from .config import Config, ConfigError, load_config
from .rainworld_env import RainWorldEnv
from .shared_memory import (
    CommandError,
    GameNotRunningError,
    ModState,
    NotConnectedError,
    ReadyTimeoutError,
    SharedMemoryClient,
    SharedMemoryError,
    StepTimeoutError,
)

__all__ = [
    "RainWorldEnv",
    "SharedMemoryClient",
    "ModState",
    "Config",
    "ConfigError",
    "load_config",
    "SharedMemoryError",
    "GameNotRunningError",
    "NotConnectedError",
    "StepTimeoutError",
    "ReadyTimeoutError",
    "CommandError",
]
__version__ = "0.2.0"
