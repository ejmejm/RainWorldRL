"""
Rain World RL - Python client for the Rain World reinforcement learning environment.
"""

from .rainworld_env import RainWorldEnv
from .shared_memory import SharedMemoryClient

__all__ = ["RainWorldEnv", "SharedMemoryClient"]
__version__ = "0.1.0"

