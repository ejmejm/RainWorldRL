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
    KEY_BITS,
    KEY_DOWN,
    KEY_GRAB,
    KEY_JUMP,
    KEY_LEFT,
    KEY_MAP,
    KEY_NAMES,
    KEY_RIGHT,
    KEY_SPECIAL,
    KEY_THROW,
    KEY_UP,
    NUM_KEYS,
    CommandError,
    GameNotRunningError,
    ModState,
    NotConnectedError,
    ReadyTimeoutError,
    SharedMemoryClient,
    SharedMemoryError,
    StepTimeoutError,
    action_to_bits,
    bits_to_keys,
    encode_keys,
    keys_to_bits,
    pressed_key_names,
)
from .wrappers import ACTION_NAMES, DISCRETE_ACTIONS, DiscreteActions

__all__ = [
    "RainWorldEnv",
    "SharedMemoryClient",
    "ModState",
    "Config",
    "ConfigError",
    "load_config",
    # raw-key action space
    "KEY_NAMES", "NUM_KEYS", "KEY_BITS",
    "KEY_LEFT", "KEY_RIGHT", "KEY_UP", "KEY_DOWN",
    "KEY_JUMP", "KEY_GRAB", "KEY_THROW", "KEY_MAP", "KEY_SPECIAL",
    "encode_keys", "keys_to_bits", "bits_to_keys", "action_to_bits", "pressed_key_names",
    # optional discrete wrapper
    "DiscreteActions", "ACTION_NAMES", "DISCRETE_ACTIONS",
    "SharedMemoryError",
    "GameNotRunningError",
    "NotConnectedError",
    "StepTimeoutError",
    "ReadyTimeoutError",
    "CommandError",
]
__version__ = "0.2.0"
