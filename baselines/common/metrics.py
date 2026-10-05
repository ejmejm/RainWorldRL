"""
Per-step environment bookkeeping shared by PPO training, ``play.py`` and the
random agent.

``RolloutMetrics.update(info, reward)`` once per env step; ``summary()``
returns a flat ``{name: float}`` dict (MLflow-ready) covering the current
window (mean reward, per-term reward sums, steps/s) plus cumulative counters
(rooms discovered, deaths, cycles survived, food eaten, karma, total steps).
``end_window()`` resets the per-window accumulators.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Set, Tuple


def _ready(info: Mapping[str, Any]) -> bool:
    return bool(info.get("ready", True)) and bool(info.get("in_game", True))


class RolloutMetrics:
    def __init__(self) -> None:
        # cumulative
        self.total_steps = 0
        self.deaths = 0
        self.cycles_survived = 0
        self.food_eaten = 0
        self.karma = 0
        self.rooms: Set[Tuple[str, int]] = set()
        self.total_reward = 0.0
        self.term_totals: Dict[str, float] = {}
        # window
        self.window_steps = 0
        self.window_reward = 0.0
        self.window_terms: Dict[str, float] = {}
        self.window_deaths = 0
        self.window_cycles = 0
        self.window_food = 0
        self.window_new_rooms = 0
        self._window_start = time.perf_counter()
        self._prev_info: Optional[Mapping[str, Any]] = None

    # -- feeding -------------------------------------------------------------
    def start(self, info: Mapping[str, Any]) -> None:
        """Record the ``reset()`` info (start room counts as discovered)."""
        self._prev_info = info
        self._visit(info)
        if _ready(info):
            self.karma = int(info.get("karma", 0))
        self._window_start = time.perf_counter()

    def _visit(self, info: Mapping[str, Any]) -> None:
        if not _ready(info):
            return
        room = int(info.get("room_index", -1))
        if room < 0:
            return
        key = (str(info.get("region", "")), room)
        if key not in self.rooms:
            self.rooms.add(key)
            self.window_new_rooms += 1

    def update(self, info: Mapping[str, Any], reward: float) -> None:
        self.total_steps += 1
        self.window_steps += 1
        self.total_reward += reward
        self.window_reward += reward

        for name, value in (info.get("reward_terms") or {}).items():
            self.window_terms[name] = self.window_terms.get(name, 0.0) + float(value)
            self.term_totals[name] = self.term_totals.get(name, 0.0) + float(value)

        if info.get("player_dead", False):
            self.deaths += 1
            self.window_deaths += 1
        if info.get("cycle_survived", False):
            self.cycles_survived += 1
            self.window_cycles += 1

        prev = self._prev_info
        if prev is not None and _ready(prev) and _ready(info) \
                and prev.get("cycle_number", -1) == info.get("cycle_number", -1):
            gained = int(info.get("food", 0)) - int(prev.get("food", 0))
            if gained > 0:
                self.food_eaten += gained
                self.window_food += gained

        if _ready(info):
            self.karma = int(info.get("karma", 0))
        self._visit(info)
        self._prev_info = info

    # -- reporting -----------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        elapsed = max(1e-9, time.perf_counter() - self._window_start)
        steps = max(1, self.window_steps)
        out: Dict[str, float] = {
            "env/reward_mean": self.window_reward / steps,
            "env/reward_sum": self.window_reward,
            "env/reward_total": self.total_reward,
            "env/rooms_discovered": float(len(self.rooms)),
            "env/new_rooms": float(self.window_new_rooms),
            "env/deaths": float(self.deaths),
            "env/deaths_window": float(self.window_deaths),
            "env/cycles_survived": float(self.cycles_survived),
            "env/cycles_survived_window": float(self.window_cycles),
            "env/food_eaten": float(self.food_eaten),
            "env/food_eaten_window": float(self.window_food),
            "env/karma": float(self.karma),
            "env/steps_per_second": self.window_steps / elapsed,
            "env/total_steps": float(self.total_steps),
        }
        for name, value in self.window_terms.items():
            out[f"reward_terms/{name}"] = value
        for name, value in self.term_totals.items():
            out[f"reward_terms_total/{name}"] = value
        return out

    def end_window(self) -> None:
        self.window_steps = 0
        self.window_reward = 0.0
        self.window_terms = {}
        self.window_deaths = 0
        self.window_cycles = 0
        self.window_food = 0
        self.window_new_rooms = 0
        self._window_start = time.perf_counter()


def format_summary(summary: Mapping[str, float]) -> str:
    """Multi-line, aligned rendering of a ``summary()`` dict for the console."""
    width = max(len(k) for k in summary) if summary else 0
    return "\n".join(f"  {k:<{width}}  {v: .4f}" for k, v in sorted(summary.items()))
