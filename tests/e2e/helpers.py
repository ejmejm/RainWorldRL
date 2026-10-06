"""Stepping and info helpers for the end-to-end tests (key constants come from ``rainworld_rl``)."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from rainworld_rl import ModState

# The info dict of reset()/step() has the keys of ModState.to_info(), with values of the same types.
INFO_TYPES = {key: type(value) for key, value in ModState().to_info().items()}

StepResult = Tuple[Any, Dict[str, Any]]


def step_n(
    env: Any,
    n: int,
    action_fn: Optional[Callable[[int], Any]] = None,
) -> List[StepResult]:
    """
    Step ``env`` ``n`` times and return ``[(obs, info), ...]``.

    ``action_fn(i)`` picks the action for step ``i`` (0-based): an int key bitmask
    (``KEY_RIGHT | KEY_JUMP``) or a MultiBinary vector; when omitted every step is a no-op
    (no keys held). Reward/terminated/truncated are dropped because the environment
    is continual (they are always 0.0/False/False).
    """
    results: List[StepResult] = []
    for i in range(n):
        action = 0 if action_fn is None else action_fn(i)
        obs, _reward, _terminated, _truncated, info = env.step(action)
        results.append((obs, info))
    return results


def infos(results: List[StepResult], key: str) -> list:
    """Extract ``info[key]`` from every entry of a ``step_n`` result list."""
    return [info[key] for _obs, info in results]


def assert_info_contract(info: Dict[str, Any]) -> None:
    """Assert ``info`` has every ``ModState.to_info()`` key, each with the same type."""
    missing = sorted(set(INFO_TYPES) - set(info))
    assert not missing, f"info is missing keys: {missing}"
    wrong = {key: type(info[key]).__name__ for key, t in INFO_TYPES.items() if not isinstance(info[key], t)}
    assert not wrong, f"info values of the wrong type: {wrong}"


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
