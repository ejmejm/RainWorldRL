"""
Shared helpers for the Rain World RL test harness.

Everything in here is importable WITHOUT the game and without the
``rainworld_rl.python`` package, so unit tests can exercise it with a fake env.
Constants mirror docs/PROTOCOL.md; keep them in sync with that document.
"""

from __future__ import annotations

import numbers
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Protocol constants (docs/PROTOCOL.md, header offset 3 "Status bits")
# ---------------------------------------------------------------------------
STATUS_PLAYER_DEAD = 1 << 0
STATUS_CONNECTED = 1 << 1
STATUS_READY = 1 << 2
STATUS_HUMAN_OVERRIDE = 1 << 3
STATUS_IN_GAME = 1 << 4
STATUS_MOD_ALIVE = 1 << 5

# Name of a decoded bool on a state object/dict -> raw status mask fallback.
STATUS_BIT_NAMES: Dict[str, int] = {
    "player_dead": STATUS_PLAYER_DEAD,
    "connected": STATUS_CONNECTED,
    "ready": STATUS_READY,
    "human_override": STATUS_HUMAN_OVERRIDE,
    "in_game": STATUS_IN_GAME,
    "mod_alive": STATUS_MOD_ALIVE,
}

# Discrete(18) action indices (see SharedMemoryClient.send_action_discrete).
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_JUMP = 5
ACTION_GRAB = 6
ACTION_THROW = 7
NUM_ACTIONS = 18

# A fresh (wiped) save starts at cycle 0; PROTOCOL.md reports -1 when unavailable.
FRESH_SAVE_CYCLE = 0

# info-dict contract returned by RainWorldEnv.reset()/step().
# value: tuple of acceptable python types (numpy scalars are accepted via numbers.*).
INFO_CONTRACT: Dict[str, Tuple[type, ...]] = {
    "player_dead": (bool, numbers.Integral),  # bool or np.bool_
    "karma": (numbers.Integral,),
    "karma_cap": (numbers.Integral,),
    "food": (numbers.Integral,),
    "player_pos": (tuple, list),  # (x, y) floats; ndarray handled separately
    "room_index": (numbers.Integral,),
    "cycle_number": (numbers.Integral,),
    "step_counter": (numbers.Integral,),
    "in_game": (bool, numbers.Integral),
    "human_override": (bool, numbers.Integral),
}

StepResult = Tuple[Any, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Stepping helpers
# ---------------------------------------------------------------------------
def step_n(
    env: Any,
    n: int,
    action_fn: Optional[Callable[[int], int]] = None,
) -> List[StepResult]:
    """
    Step ``env`` ``n`` times and return ``[(obs, info), ...]``.

    ``action_fn(i)`` picks the action for step ``i`` (0-based); when omitted every
    step is a no-op (action 0). Reward/terminated/truncated are dropped because the
    environment is continual (they are always 0.0/False/False).
    """
    results: List[StepResult] = []
    for i in range(n):
        action = ACTION_NOOP if action_fn is None else int(action_fn(i))
        obs, _reward, _terminated, _truncated, info = env.step(action)
        results.append((obs, info))
    return results


def infos(results: List[StepResult], key: str) -> list:
    """Extract ``info[key]`` from every entry of a ``step_n`` result list."""
    return [info[key] for _obs, info in results]


def assert_info_contract(info: Dict[str, Any]) -> None:
    """Assert ``info`` has every key of the API contract with a sane type."""
    missing = sorted(set(INFO_CONTRACT) - set(info))
    assert not missing, f"info is missing contract keys: {missing}"
    for key, types in INFO_CONTRACT.items():
        value = info[key]
        if key == "player_pos":
            # tuple/list/ndarray of two numbers
            assert len(value) == 2, f"player_pos should be (x, y), got {value!r}"
            for coord in value:
                assert isinstance(coord, numbers.Real), f"player_pos coord not numeric: {coord!r}"
        else:
            assert isinstance(value, types), (
                f"info[{key!r}]={value!r} is {type(value).__name__}, expected one of {types}"
            )


# ---------------------------------------------------------------------------
# Low-level state access (via the env's SharedMemoryClient)
# ---------------------------------------------------------------------------
def get_client(env: Any) -> Any:
    """Return the SharedMemoryClient owned by ``env`` (``env.client`` or ``env._client``)."""
    for name in ("client", "_client"):
        client = getattr(env, name, None)
        if client is not None:
            return client
    raise AttributeError(
        "env exposes no shared-memory client (expected attribute 'client' or '_client')"
    )


def read_state(env: Any) -> Any:
    """Read the raw header state from the env's client (``client.read_state()``)."""
    return get_client(env).read_state()


_MISSING = object()


def state_field(state: Any, name: str, default: Any = _MISSING) -> Any:
    """Fetch ``name`` from a state that may be a dict, dataclass or namespace."""
    if isinstance(state, dict):
        if name in state:
            return state[name]
    elif hasattr(state, name):
        return getattr(state, name)
    if default is _MISSING:
        raise KeyError(f"state has no field {name!r}: {state!r}")
    return default


def status_bit(state: Any, name: str) -> bool:
    """
    True when the named status bit (``ready``, ``in_game``, ``mod_alive``, ...) is set.

    Prefers a decoded boolean field of that name on ``state``; otherwise masks the
    raw ``status`` byte with the PROTOCOL.md bit.
    """
    if name not in STATUS_BIT_NAMES:
        raise KeyError(f"unknown status bit {name!r}; known: {sorted(STATUS_BIT_NAMES)}")
    decoded = state_field(state, name, None)
    if decoded is not None:
        return bool(decoded)
    status = int(state_field(state, "status"))
    return bool(status & STATUS_BIT_NAMES[name])


def wait_for(
    predicate: Callable[[], bool],
    timeout: float,
    interval: float = 0.05,
) -> bool:
    """Poll ``predicate`` until it is truthy or ``timeout`` seconds pass."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
