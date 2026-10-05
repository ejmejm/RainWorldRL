"""
Reward functions for the Rain World RL environment.

Philosophy
----------
``RainWorldEnv`` returns reward ``0.0`` and puts every piece of game state in
``info``; rewards are Python wrappers over ``info``. The environment exists to
test exploration / intrinsic-motivation algorithms, so the default reward is
**not a task** - it mimics the drives of an animal (hunger, tiredness,
satisfaction, pain) and leaves *how* to satisfy them entirely to the agent.

Why not just "+1 per survived cycle"? Because the game has a camping optimum:
a slugcat with ``food_to_hibernate`` pips standing still in a shelter for ~40
ticks hibernates immediately (``Player.cs`` ~5734-5776: ``readyForWin`` ->
``shelterDoor.Close()``), and batflies respawn every cycle
(``FliesWorldAI.cs`` :60-80, ``fliesToSpawn`` is refilled for every new
``World``). "Camp next to a hive, eat 4, step into the shelter next door,
sleep at once" therefore survives a cycle in a minute of game time with no
exploration at all. The default here (``DriveReward``) makes that strictly
suboptimal with two mechanisms:

* **Per-room satiety** (``Food``): every pip eaten in a room multiplies that
  room's food value by ``satiety_decay`` (0.5); the value recovers linearly
  over ``recovery_steps`` *real env steps* (one nominal cycle of 24000 ticks
  by default). Sleeping early does not refresh a patch, so the camper's hive is
  worth a few percent of a fresh room after the first cycle.
* **Tiredness-scaled sleep** (``Sleep``): a survived cycle pays
  ``weight * min(1, steps_awake / nominal_cycle_steps)``. Sleeping after 1 % of
  a cycle pays 1 %; the only way to collect a full sleep is to stay out a full
  cycle - and then the food you need is not where you slept.

Plus a bonus for sleeping with a surplus (pips above ``food_to_hibernate`` are
capped by ``food_max``, Player.AddFood, so this is bounded), a death penalty,
a small per-step malnourishment penalty, and an optional novelty bonus
(``NewRoom``) that can be switched off for algorithms that bring their own
intrinsic motivation.

Composable terms
----------------
``InfoReward(env, [terms...])`` is a ``gym.Wrapper`` that replaces the
environment's reward by the sum of its ``RewardTerm``s. Each term is called as
``term(prev_info, info) -> float`` once per step with the previous and current
``info`` dicts (so it can detect edges and deltas), gets ``term.reset(info)`` on
``env.reset()``, and owns its own ``weight``. The per-term breakdown is written
to ``info["reward_terms"]``.

Drive terms (the default; weights are signed multipliers, defaults shown):

* ``NewRoom(weight = 1.0)``            - first entry into each ``(region, room_index)``;
  the visited set persists across deaths and clears only on ``reset()``.
* ``Food(below = 0.3, above = 0.1, satiety_decay = 0.5, recovery_steps = ...)``
  - per pip gained: ``below`` while the pip count is <= ``food_to_hibernate``,
  ``above`` beyond it, times the room's satiety factor.
* ``Sleep(weight = 1.0, full_belly_per_pip = 0.25)`` - on ``cycle_survived``:
  ``weight * tiredness + full_belly_per_pip * max(0, food - food_to_hibernate)``.
* ``Death(weight = -3.0)``             - on the ``player_dead`` edge.
* ``Malnourished(per_step = -0.0003)`` - every ready step while ``malnourished``.

Building blocks (not in the default): ``CycleSurvived``, ``Ate``, ``Alive``,
``FunctionTerm``. ``SurviveCycleReward`` (``CycleSurvived(1.0)`` only) is kept
as the sparse alternative.

Example::

    from rainworld_rl import RainWorldEnv
    from rainworld_rl.rewards import DriveReward, InfoReward, CycleSurvived, Death, make_default_reward_env

    env = make_default_reward_env(RainWorldEnv(ticks_per_step = 4))          # DriveReward, novelty on
    env = DriveReward(RainWorldEnv(ticks_per_step = 4), novelty = False)     # agent brings its own curiosity
    env = InfoReward(RainWorldEnv(), [CycleSurvived(1.0), Death(-1.0)])      # hand-rolled alternative

See ``docs/REWARDS.md`` for the analysis and the meaning of the ``info`` fields.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import gymnasium as gym


Info = Mapping[str, Any]
RoomKey = Tuple[str, int]

# A Rain World cycle is drawn per cycle from 400..800 s of game time at 40 ticks/s
# (RainCycle.cs:245 ``cycleLength = minutes * 40 * 60``), i.e. 16000..32000 ticks.
# 24000 ticks (10 min) is the nominal cycle the drive terms measure time against.
NOMINAL_CYCLE_TICKS = 24000


def nominal_cycle_steps(ticks_per_step: int = 1) -> int:
    """Env steps in one nominal cycle (``NOMINAL_CYCLE_TICKS / ticks_per_step``, at least 1)."""
    return max(1, round(NOMINAL_CYCLE_TICKS / max(1, int(ticks_per_step))))


_nominal_cycle_steps = nominal_cycle_steps  # alias for use where a parameter shadows the name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ready(info: Info) -> bool:
    """True when the mod reported real game state for this step (zeros / -1 otherwise)."""
    return bool(info.get("ready", True)) and bool(info.get("in_game", True))


def room_key(info: Info) -> Optional[RoomKey]:
    """``(region, room_index)`` of a ready info with a valid room, else None.

    Room indices are only unique within a region, so the region acronym
    (``info["region"]``, "" when the mod does not report it) is part of the key.
    """
    if not _ready(info):
        return None
    room = int(info.get("room_index", -1))
    if room < 0:
        return None
    return (str(info.get("region", "") or ""), room)


class RewardTerm(ABC):
    """
    One additive component of a reward. Subclasses implement
    ``__call__(prev_info, info) -> float`` and may override ``reset(info)``.

    ``weight`` is a signed multiplier applied by the term itself; ``name`` is
    the key used in the ``info["reward_terms"]`` breakdown (defaults to the
    class name). Every term must cope with non-ready steps (fields zero / -1)
    and with ``reset(info)`` being called with the first info of an episode.
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


# ---------------------------------------------------------------------------
# Drive terms (the default)
# ---------------------------------------------------------------------------

class NewRoom(RewardTerm):
    """
    ``weight`` the first time each room is entered, keyed by ``(region, room_index)``.

    The start room (from ``reset()``) counts as already visited. The visited set
    persists across deaths (the environment is continual) and is cleared only
    on ``reset()``. Steps with no real game state (``ready`` False,
    ``room_index`` < 0) are ignored.
    """

    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        super().__init__(weight, name)
        self.visited: Set[RoomKey] = set()

    def reset(self, info: Info) -> None:
        self.visited = set()
        self._visit(info)

    def _visit(self, info: Info) -> bool:
        """Record the current room; True if it was not seen before."""
        key = room_key(info)
        if key is None or key in self.visited:
            return False
        self.visited.add(key)
        return True

    def __call__(self, prev_info: Info, info: Info) -> float:
        return self.weight if self._visit(info) else 0.0


class Food(RewardTerm):
    """
    Hunger: reward per food pip gained, discounted by the room's satiety.

    For each pip gained in a step (``food`` increased, same ``cycle_number``,
    both steps ready) the base value is ``below`` if the pip's index (the food
    count after that pip) is <= ``food_to_hibernate`` and ``above`` otherwise,
    multiplied by the **satiety factor** of the room the pip was eaten in.

    Every room starts at factor 1.0. Each pip eaten there multiplies the factor
    by ``satiety_decay``. The factor recovers **linearly** toward 1.0 by
    ``1 / recovery_steps`` per real env step (every ``__call__``, ready or not;
    never per cycle, so sleeping early does not refresh a patch). A room eaten
    down to ``f`` is fresh again after ``(1 - f) * recovery_steps`` steps.
    ``recovery_steps`` defaults to one nominal cycle at ``ticks_per_step`` 1
    (``NOMINAL_CYCLE_TICKS``); ``DriveReward`` derives it from the env.

    Food decreases (hibernation subtracts ``food_to_hibernate``, death resets)
    never give negative reward. ``weight`` scales the whole term.
    """

    def __init__(
        self,
        below: float = 0.3,
        above: float = 0.1,
        satiety_decay: float = 0.5,
        recovery_steps: Optional[int] = None,
        weight: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(weight, name)
        if not 0.0 <= satiety_decay <= 1.0:
            raise ValueError(f"satiety_decay must be in [0, 1], got {satiety_decay}")
        self.below = float(below)
        self.above = float(above)
        self.satiety_decay = float(satiety_decay)
        self.recovery_steps = int(recovery_steps if recovery_steps is not None else nominal_cycle_steps(1))
        if self.recovery_steps < 1:
            raise ValueError("recovery_steps must be >= 1")
        self._step = 0
        # room -> (factor at last update, step index of that update)
        self._satiety: Dict[RoomKey, Tuple[float, int]] = {}

    def reset(self, info: Info) -> None:
        self._step = 0
        self._satiety = {}

    def satiety(self, key: RoomKey) -> float:
        """Current satiety factor of a room (1.0 = fresh), including recovery since the last pip."""
        if key not in self._satiety:
            return 1.0
        factor, at = self._satiety[key]
        return min(1.0, factor + (self._step - at) / self.recovery_steps)

    def __call__(self, prev_info: Info, info: Info) -> float:
        self._step += 1
        if not (_ready(prev_info) and _ready(info)):
            return 0.0
        if prev_info.get("cycle_number", -1) != info.get("cycle_number", -1):
            return 0.0
        before, after = int(prev_info.get("food", 0)), int(info.get("food", 0))
        if after <= before:
            return 0.0
        key = room_key(info)
        if key is None:
            return 0.0
        threshold = int(info.get("food_to_hibernate", 0))
        factor = self.satiety(key)
        total = 0.0
        for pip in range(before + 1, after + 1):
            base = self.below if pip <= threshold else self.above
            total += base * factor
            factor *= self.satiety_decay
        self._satiety[key] = (factor, self._step)
        return self.weight * total


class Sleep(RewardTerm):
    """
    Tiredness / satisfaction: reward for a survived cycle, scaled by how long
    the slugcat has been awake.

    On the ``cycle_survived`` edge::

        weight * tiredness + full_belly_per_pip * max(0, food_before_sleep - food_to_hibernate)

    where ``tiredness = min(1, steps_awake / nominal_cycle_steps)``,
    ``steps_awake`` counts real env steps since the last sleep (or since
    ``reset()``), and ``food_before_sleep`` is the last ``food`` value seen
    before the edge (``prev_info["food"]`` - the save already subtracted the
    hibernation cost by the time the edge is reported).

    A starving sleep (``cycle_number`` goes up without ``cycle_survived``, or
    ``malnourished`` turns True) earns nothing. ``steps_awake`` is reset after
    *any* sleep. ``nominal_cycle_steps`` defaults to one nominal cycle at
    ``ticks_per_step`` 1; ``DriveReward`` derives it from the env.
    """

    def __init__(
        self,
        weight: float = 1.0,
        full_belly_per_pip: float = 0.25,
        nominal_cycle_steps: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(weight, name)
        self.full_belly_per_pip = float(full_belly_per_pip)
        self.nominal_cycle_steps = int(
            nominal_cycle_steps if nominal_cycle_steps is not None else _nominal_cycle_steps(1)
        )
        if self.nominal_cycle_steps < 1:
            raise ValueError("nominal_cycle_steps must be >= 1")
        self.steps_awake = 0
        self._last_food = 0
        self._last_threshold = 0
        self._last_cycle = -1
        self._last_malnourished = False

    def reset(self, info: Info) -> None:
        self.steps_awake = 0
        self._last_food = 0
        self._last_threshold = 0
        self._last_cycle = -1
        self._last_malnourished = False
        self._track(info)

    def _track(self, info: Info) -> None:
        if not _ready(info):
            return
        self._last_food = int(info.get("food", 0))
        self._last_threshold = int(info.get("food_to_hibernate", 0))
        self._last_cycle = int(info.get("cycle_number", -1))
        self._last_malnourished = bool(info.get("malnourished", False))

    @property
    def tiredness(self) -> float:
        return min(1.0, self.steps_awake / self.nominal_cycle_steps)

    def __call__(self, prev_info: Info, info: Info) -> float:
        self.steps_awake += 1
        reward = 0.0
        slept = False

        if info.get("cycle_survived", False):
            # Fed sleep. food/threshold come from the last ready info before the edge
            # (normally prev_info; the edge itself may land on a non-ready frame).
            surplus = max(0, self._last_food - self._last_threshold)
            reward = self.weight * self.tiredness + self.full_belly_per_pip * surplus
            slept = True

        if _ready(info):
            cycle = int(info.get("cycle_number", -1))
            if self._last_cycle >= 0 and cycle > self._last_cycle:
                slept = True                               # any sleep (fed or starving) advances the cycle
            if bool(info.get("malnourished", False)) and not self._last_malnourished:
                slept = True                               # starving sleep just happened
            self._track(info)

        if slept:
            self.steps_awake = 0
        return reward


class Death(RewardTerm):
    """Pain: ``weight`` (negative by default) on the step where ``player_dead`` is True (one-step edge)."""

    def __init__(self, weight: float = -3.0, name: Optional[str] = None):
        super().__init__(weight, name)

    def __call__(self, prev_info: Info, info: Info) -> float:
        return self.weight if info.get("player_dead", False) else 0.0


class Malnourished(RewardTerm):
    """Hunger pain: ``per_step`` (negative by default) on every ready step while ``malnourished`` is True."""

    def __init__(self, per_step: float = -0.0003, name: Optional[str] = None):
        super().__init__(per_step, name)

    @property
    def per_step(self) -> float:
        return self.weight

    def __call__(self, prev_info: Info, info: Info) -> float:
        if _ready(info) and bool(info.get("malnourished", False)):
            return self.weight
        return 0.0


# ---------------------------------------------------------------------------
# Building blocks (not part of the default)
# ---------------------------------------------------------------------------

class CycleSurvived(RewardTerm):
    """Building block: ``weight`` on the step where ``cycle_survived`` is True (hibernated with enough food)."""

    def __call__(self, prev_info: Info, info: Info) -> float:
        return self.weight if info.get("cycle_survived", False) else 0.0


class Ate(RewardTerm):
    """
    Building block: flat ``weight`` per food pip gained between consecutive steps
    (no satiety, no threshold; see ``Food`` for the drive version).

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
    """Building block: ``weight`` on every step where the player is in the game and did not die this step."""

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

    def term(self, name: str) -> RewardTerm:
        """Look up a term by its breakdown name."""
        for t in self.terms:
            if t.name == name:
                return t
        raise KeyError(name)

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
# Default: DriveReward
# ---------------------------------------------------------------------------

def _env_ticks_per_step(env: gym.Env) -> int:
    """``ticks_per_step`` of the innermost env (1 if it has none)."""
    base = getattr(env, "unwrapped", env)
    return max(1, int(getattr(base, "ticks_per_step", 1)))


def drive_terms(
    novelty: bool = True,
    *,
    ticks_per_step: int = 1,
    new_room: float = 1.0,
    food_below: float = 0.3,
    food_above: float = 0.1,
    satiety_decay: float = 0.5,
    recovery_steps: Optional[int] = None,
    sleep: float = 1.0,
    full_belly_per_pip: float = 0.25,
    death: float = -3.0,
    malnourished: float = -0.0003,
) -> List[RewardTerm]:
    """
    Fresh instances of the drive terms: ``[NewRoom (if novelty), Food, Sleep, Death, Malnourished]``.

    ``recovery_steps`` (Food) and the Sleep term's nominal cycle both default to
    ``nominal_cycle_steps(ticks_per_step)``.
    """
    cycle_steps = nominal_cycle_steps(ticks_per_step)
    terms: List[RewardTerm] = []
    if novelty:
        terms.append(NewRoom(new_room))
    terms += [
        Food(food_below, food_above, satiety_decay,
             recovery_steps if recovery_steps is not None else cycle_steps),
        Sleep(sleep, full_belly_per_pip, cycle_steps),
        Death(death),
        Malnourished(malnourished),
    ]
    return terms


class DriveReward(InfoReward):
    """
    The default reward: animal drives (see the module docstring).

    ``DriveReward(env, novelty = True, **weights)`` = ``InfoReward(env,
    drive_terms(novelty, ticks_per_step = env.ticks_per_step, **weights))``.
    ``weights`` are the keyword arguments of ``drive_terms``; ``ticks_per_step``
    may be passed to override what is read from the env.
    """

    def __init__(self, env: gym.Env, novelty: bool = True, *, ticks_per_step: Optional[int] = None,
                 breakdown_key: Optional[str] = "reward_terms", add_env_reward: bool = False, **weights):
        tps = _env_ticks_per_step(env) if ticks_per_step is None else int(ticks_per_step)
        self.novelty = bool(novelty)
        self.ticks_per_step = tps
        super().__init__(env, drive_terms(novelty, ticks_per_step = tps, **weights),
                         breakdown_key = breakdown_key, add_env_reward = add_env_reward)


def default_terms(novelty: bool = True, ticks_per_step: int = 1, **weights) -> List[RewardTerm]:
    """Fresh instances of the default reward's terms (``drive_terms``)."""
    return drive_terms(novelty, ticks_per_step = ticks_per_step, **weights)


def make_default_reward_env(env: gym.Env, novelty: bool = True, **weights) -> DriveReward:
    """Wrap ``env`` with the default ``DriveReward`` (``novelty`` toggles the ``NewRoom`` term)."""
    return DriveReward(env, novelty, **weights)


# ---------------------------------------------------------------------------
# Named alternative: sparse survival
# ---------------------------------------------------------------------------

def survive_cycle_terms() -> List[RewardTerm]:
    """Fresh instances of the sparse alternative's terms: ``[CycleSurvived(1.0)]``."""
    return [CycleSurvived(1.0)]


class SurviveCycleReward(InfoReward):
    """
    Sparse alternative: ``+1`` on the step the slugcat survives a cycle
    (hibernates with enough food), ``0`` otherwise. Note the camping optimum
    described in the module docstring; this is a baseline, not the default.
    """

    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env, survive_cycle_terms(), **kwargs)


__all__ = [
    "NOMINAL_CYCLE_TICKS",
    "nominal_cycle_steps",
    "room_key",
    "RewardTerm",
    # drive terms
    "NewRoom",
    "Food",
    "Sleep",
    "Death",
    "Malnourished",
    # building blocks
    "CycleSurvived",
    "Ate",
    "Alive",
    "FunctionTerm",
    # wrappers
    "InfoReward",
    "DriveReward",
    "SurviveCycleReward",
    "drive_terms",
    "default_terms",
    "survive_cycle_terms",
    "make_default_reward_env",
]
