"""
Optional action wrappers for ``RainWorldEnv``.

The environment's native action space is ``MultiBinary(NUM_KEYS)``: one bit
per key the player can hold (see ``shared_memory.KEY_NAMES``). This module
provides the classic 18-action discrete set as a ``gymnasium.ActionWrapper``
for agents that prefer a small ``Discrete`` space::

    from rainworld_rl import RainWorldEnv, DiscreteActions

    env = DiscreteActions(RainWorldEnv(160, 90, ticks_per_step = 4))
    env.action_space            # Discrete(18)
    env.step(9)                 # Right + Jump

It is purely a convenience: the mod never sees discrete indices, only the key
bits. Note that the discrete set has no ``map`` or ``special`` key, so a
discrete agent cannot dismiss the game-over "press MAP to restart" prompt or
use slugcat-specific abilities; use the native key space for that.
"""

from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .shared_memory import KEY_NAMES, NUM_KEYS, bits_to_keys, encode_keys


# Keys held for each discrete action index.
DISCRETE_ACTIONS: Tuple[Tuple[str, ...], ...] = (
    (),                      # 0  No-op
    ("left",),               # 1  Left
    ("right",),              # 2  Right
    ("up",),                 # 3  Up
    ("down",),               # 4  Down
    ("jump",),               # 5  Jump
    ("grab",),               # 6  Grab
    ("throw",),              # 7  Throw
    ("left", "jump"),        # 8  Left + Jump
    ("right", "jump"),       # 9  Right + Jump
    ("up", "jump"),          # 10 Up + Jump
    ("down", "jump"),        # 11 Down + Jump
    ("left", "grab"),        # 12 Left + Grab
    ("right", "grab"),       # 13 Right + Grab
    ("up", "grab"),          # 14 Up + Grab
    ("down", "grab"),        # 15 Down + Grab
    ("left", "down"),        # 16 Crawl Left (Left + Down)
    ("right", "down"),       # 17 Crawl Right (Right + Down)
)
NUM_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)
assert NUM_DISCRETE_ACTIONS == 18

ACTION_NAMES: Tuple[str, ...] = (
    "No-op", "Left", "Right", "Up", "Down",
    "Jump", "Grab", "Throw",
    "Left+Jump", "Right+Jump", "Up+Jump", "Down+Jump",
    "Left+Grab", "Right+Grab", "Up+Grab", "Down+Grab",
    "Crawl Left", "Crawl Right",
)
assert len(ACTION_NAMES) == NUM_DISCRETE_ACTIONS

# Precomputed bitmask per discrete action.
DISCRETE_ACTION_BITS: Tuple[int, ...] = tuple(encode_keys(*keys) for keys in DISCRETE_ACTIONS)


def discrete_to_bits(action: int) -> int:
    """Map a discrete action index (0-17) to an ``action_bits`` mask."""
    action = int(action)
    if not 0 <= action < NUM_DISCRETE_ACTIONS:
        raise ValueError(f"Discrete action must be in [0, {NUM_DISCRETE_ACTIONS}), got {action}")
    return DISCRETE_ACTION_BITS[action]


def discrete_to_keys(action: int) -> np.ndarray:
    """Map a discrete action index (0-17) to a ``MultiBinary(NUM_KEYS)`` key vector."""
    return bits_to_keys(discrete_to_bits(action))


class DiscreteActions(gym.ActionWrapper):
    """
    Expose ``Discrete(18)`` on top of the native ``MultiBinary(NUM_KEYS)`` space.

    ``ACTION_NAMES[i]`` names action ``i``; ``DISCRETE_ACTIONS[i]`` lists the
    keys it holds. The wrapped env receives the equivalent key vector.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        inner = env.action_space
        if not (isinstance(inner, spaces.MultiBinary) and int(np.prod(inner.shape)) == NUM_KEYS):
            raise TypeError(
                f"DiscreteActions expects an env with MultiBinary({NUM_KEYS}) actions "
                f"({KEY_NAMES}), got {inner}"
            )
        self.action_space = spaces.Discrete(NUM_DISCRETE_ACTIONS)

    def action(self, action) -> np.ndarray:  # type: ignore[override]
        return discrete_to_keys(int(action))

    def reverse_action(self, action) -> int:  # type: ignore[override]
        """Key vector -> discrete index (``ValueError`` if the combination is not in the table)."""
        from .shared_memory import action_to_bits
        bits = action_to_bits(action)
        try:
            return DISCRETE_ACTION_BITS.index(bits)
        except ValueError:
            raise ValueError(f"key combination {bits:#x} has no discrete action") from None


__all__ = [
    "DiscreteActions",
    "ACTION_NAMES",
    "DISCRETE_ACTIONS",
    "DISCRETE_ACTION_BITS",
    "NUM_DISCRETE_ACTIONS",
    "discrete_to_bits",
    "discrete_to_keys",
]
