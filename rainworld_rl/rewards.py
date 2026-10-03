"""
Reward functions for the Rain World RL environment.

Philosophy
----------
``RainWorldEnv`` returns reward ``0.0`` and puts every piece of game state in
``info``; rewards are Python wrappers over ``info``. The environment exists to
test exploration / intrinsic-motivation algorithms, so the **default reward is
deliberately a single sparse signal and nothing else is shaped**:

    ``SurviveCycleReward`` = ``CycleSurvived(+1.0)`` only.

Surviving a Rain World cycle (``info["cycle_survived"]``) means the slugcat
hibernated in a shelter *with enough food*. To get there an agent has to find
food (``food_to_hibernate`` pips; edible things respawn slowly, so a spot that
fed you once is empty the next cycle and you must range further), find a
shelter, read the clock (``cycle_progress`` -> rain) and not get eaten on the
way. The game already couples all of that into one event; we do not add a
step bonus, a death penalty, a distance term or anything else that would tell
the agent *how*. A starving sleep (``Win(malnourished = true)``) does not set
the flag, so camping in the start shelter earns nothing. Whether current
algorithms can reach the signal at all is exactly the question the environment
is meant to ask; it is fine if they fail.

Composable terms
----------------
``InfoReward(env, [terms...])`` is a ``gym.Wrapper`` that replaces the
environment's reward by the sum of its ``RewardTerm``s. Each term is called as
``term(prev_info, info) -> float`` once per step with the previous and current
``info`` dicts (so it can detect edges and deltas), gets ``term.reset(info)`` on
``env.reset()``, and owns its own ``weight``. The per-term breakdown is written
to ``info["reward_terms"]``.

Built-in terms (all weights are signed multipliers, defaults shown):

* ``CycleSurvived(weight = 1.0)``  - ``weight`` on the step ``cycle_survived`` is True.
* ``Death(weight = -1.0)``         - ``weight`` on the step ``player_dead`` is True.
* ``NewRoom(weight = 1.0)``        - ``weight`` the first time each ``room_index`` is
  entered. The room the episode starts in counts as visited. The visited set
  is cleared on ``reset()``.
* ``Ate(weight = 1.0)``            - ``weight`` per food pip gained within a cycle.
* ``Alive(weight = 0.01)``         - ``weight`` on every step the player is in the
  game and not dying.
* ``FunctionTerm(fn, name, weight)`` - wrap any ``fn(prev_info, info) -> float``.

Example::

    from rainworld_rl import RainWorldEnv
    from rainworld_rl.rewards import InfoReward, CycleSurvived, Death, NewRoom, make_default_reward_env

    env = make_default_reward_env(RainWorldEnv(ticks_per_step = 4))      # default: survive a cycle
    env = InfoReward(RainWorldEnv(), [CycleSurvived(1.0), Death(-1.0), NewRoom(0.1)])  # alternative

See ``docs/REWARDS.md`` for the meaning of the ``info`` fields the terms use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import gymnasium as gym


Info = Mapping[str, Any]


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------

def _ready(info: Info) -> bool:
    """True when the mod reported real game state for this step (zeros / -1 otherwise)."""
    return bool(info.get("ready", True)) and bool(info.get("in_game", True))


class RewardTerm(ABC):
    """
    One additive component of a reward. Subclasses implement
    ``__call__(prev_info, info) -> float`` and may override ``reset(info)``.

    ``weight`` is a signed multiplier applied by the term itself; ``name`` is
    the key used in the ``info["reward_terms"]`` breakdown (defaults to the
    class name).
    """

    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        self.weight = float(weight)
        self.name = name or type(self).__name__

    def reset(self, info: Info) -> None:  # noqa: B027 - optional hook
        """Called with the first ``info`` of an episode (``env.reset()``)."""

    @abstractmethod
    def __call__(self, prev_info: Info, info: Info) -> float:
        """Reward contributed by the transition ``prev_info -> info``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(weight = {self.weight!r})"


class CycleSurvived(RewardTerm):
    """``weight`` on the step where ``cycle_survived`` is True (hibernated with enough food)."""

    def __call__(self, prev_info: Info, info: Info) -> float:
        return self.weight if info.get("cycle_survived", False) else 0.0


class Death(RewardTerm):
    """``weight`` (negative by default) on the step where ``player_dead`` is True."""

    def __init__(self, weight: float = -1.0, name: Optional[str] = None):
        super().__init__(weight, name)

    def __call__(self, prev_info: Info, info: Info) -> float:
        return self.weight if info.get("player_dead", False) else 0.0


class NewRoom(RewardTerm):
    """
    ``weight`` the first time each ``room_index`` is entered during the episode.

    The start room (from ``reset()``) counts as already visited. Steps with no
    real game state (``ready`` False, ``room_index`` < 0) are ignored.
    """

    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        super().__init__(weight, name)
        self.visited: Set[int] = set()

    def reset(self, info: Info) -> None:
        self.visited = set()
        self._visit(info)

    def _visit(self, info: Info) -> bool:
        """Record the current room; True if it was not seen before."""
        if not _ready(info):
            return False
        room = int(info.get("room_index", -1))
        if room < 0 or room in self.visited:
            return False
        self.visited.add(room)
        return True

    def __call__(self, prev_info: Info, info: Info) -> float:
        return self.weight if self._visit(info) else 0.0


class Ate(RewardTerm):
    """
    ``weight`` per food pip gained between consecutive steps.

    Only counted when both steps carry real game state and belong to the same
    cycle (food is reset by hibernation and death, and reads 0 while the mod is
    not READY, so those transitions are ignored). Losing food never counts.
    """

    def __call__(self, prev_info: Info, info: Info) -> float:
        if not (_ready(prev_info) and _ready(info)):
            return 0.0
        if prev_info.get("cycle_number", -1) != info.get("cycle_number", -1):
            return 0.0
        gained = int(info.get("food", 0)) - int(prev_info.get("food", 0))
        return self.weight * gained if gained > 0 else 0.0


class Alive(RewardTerm):
    """``weight`` on every step where the player is in the game and did not die this step."""

    def __init__(self, weight: float = 0.01, name: Optional[str] = None):
        super().__init__(weight, name)

    def __call__(self, prev_info: Info, info: Info) -> float:
        if not _ready(info) or info.get("player_dead", False):
            return 0.0
        return self.weight


class FunctionTerm(RewardTerm):
    """Wrap a plain ``fn(prev_info, info) -> float`` (multiplied by ``weight``)."""

    def __init__(self, fn: Callable[[Info, Info], float], name: Optional[str] = None, weight: float = 1.0):
        super().__init__(weight, name or getattr(fn, "__name__", "FunctionTerm"))
        self.fn = fn

    def __call__(self, prev_info: Info, info: Info) -> float:
        return self.weight * float(self.fn(prev_info, info))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class InfoReward(gym.Wrapper):
    """
    Replace the environment's reward by the sum of ``terms`` evaluated on ``info``.

    Args:
        env: the ``RainWorldEnv`` (or any env whose ``info`` has the same keys).
        terms: ``RewardTerm`` instances; each is called once per step and reset
            on ``reset()``. The wrapper owns them, do not share instances between
            wrappers.
        breakdown_key: ``info`` key receiving ``{term.name: value}`` per step
            (``None`` to disable).
        add_env_reward: also add the wrapped env's own reward (always 0.0 for
            ``RainWorldEnv``; off by default so the sum is exactly the terms).
    """

    def __init__(
        self,
        env: gym.Env,
        terms: Sequence[RewardTerm],
        *,
        breakdown_key: Optional[str] = "reward_terms",
        add_env_reward: bool = False,
    ):
        super().__init__(env)
        self.terms: List[RewardTerm] = list(terms)
        if not self.terms:
            raise ValueError("InfoReward needs at least one RewardTerm")
        names = [t.name for t in self.terms]
        if len(set(names)) != len(names):
            raise ValueError(f"RewardTerm names must be unique, got {names}; pass name=... to disambiguate")
        self.breakdown_key = breakdown_key
        self.add_env_reward = add_env_reward
        self._prev_info: Optional[Dict[str, Any]] = None

    @property
    def prev_info(self) -> Optional[Dict[str, Any]]:
        """The ``info`` of the previous step (None before the first ``reset()``)."""
        return self._prev_info

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        for term in self.terms:
            term.reset(info)
        self._prev_info = dict(info)
        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        if self._prev_info is None:
            # step() before reset(): treat the current info as the baseline.
            for term in self.terms:
                term.reset(info)
            self._prev_info = dict(info)

        total = float(env_reward) if self.add_env_reward else 0.0
        breakdown: Dict[str, float] = {}
        for term in self.terms:
            value = float(term(self._prev_info, info))
            breakdown[term.name] = value
            total += value
        self._prev_info = dict(info)

        if self.breakdown_key is not None:
            info[self.breakdown_key] = breakdown
        return obs, total, terminated, truncated, info


# ---------------------------------------------------------------------------
# Default
# ---------------------------------------------------------------------------

def default_terms() -> List[RewardTerm]:
    """Fresh instances of the default reward's terms: ``[CycleSurvived(1.0)]``."""
    return [CycleSurvived(1.0)]


class SurviveCycleReward(InfoReward):
    """
    The default reward: ``+1`` on the step the slugcat survives a cycle
    (hibernates with enough food), ``0`` otherwise. See the module docstring.
    """

    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env, default_terms(), **kwargs)


def make_default_reward_env(env: gym.Env) -> SurviveCycleReward:
    """Wrap ``env`` with the default (sparse, unshaped) ``SurviveCycleReward``."""
    return SurviveCycleReward(env)


__all__ = [
    "RewardTerm",
    "CycleSurvived",
    "Death",
    "NewRoom",
    "Ate",
    "Alive",
    "FunctionTerm",
    "InfoReward",
    "SurviveCycleReward",
    "default_terms",
    "make_default_reward_env",
]
