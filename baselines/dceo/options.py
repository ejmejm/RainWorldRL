"""
DCEO option execution (Klissarov & Machado, ICML 2023, Algorithm 1) and the
epsilon schedule.

Every env step::

    if an option is running: it terminates with probability P_term
    if no option is running:
        with prob. epsilon (exploratory step):
            with prob. mu: launch a uniformly random option -> act greedily w.r.t. its Q-head
            else:          uniformly random primitive action
        else: greedy w.r.t. the main (task) Q-network
    else: the running option acts greedily w.r.t. its Q-head

Options are never chosen greedily and have no task values (that is Wayfarer's
phase-two change). Termination is memoryless, so the expected option duration
is ``1 / P_term`` steps. The official code (``full_rainbow_dceo.py``
``select_action``) draws the two coins slightly differently (the primitive
branch is the main network's own epsilon-greedy) and also lets options act
epsilon-greedily; here Algorithm 1 is followed and ``option_epsilon`` (default
0) exposes the latter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

GREEDY = 0
RANDOM = 1
OPTION = 2


def linearly_decaying_epsilon(decay_period: int, step: int, warmup_steps: int, epsilon: float) -> float:
    """Dopamine's schedule: 1.0 during warm-up, then linear decay to ``epsilon`` over ``decay_period`` steps."""
    if decay_period <= 0:
        return epsilon if step >= warmup_steps else 1.0
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = float(np.clip(bonus, 0.0, 1.0 - epsilon))
    return epsilon + bonus


@dataclass
class Decision:
    kind: int          # GREEDY / RANDOM / OPTION
    option: int = -1   # option index when kind == OPTION
    launched: bool = False


@dataclass
class ExplorerStats:
    steps: np.ndarray = field(default_factory = lambda: np.zeros(3, dtype = np.int64))   # per decision kind
    durations: List[int] = field(default_factory = list)                                   # finished option lengths
    launches: Dict[int, int] = field(default_factory = dict)


class DCEOExplorer:
    """Stateful option controller; network calls stay in the trainer (see ``Decision``)."""

    def __init__(self, num_options: int, mu: float, p_term: float, seed: int | None = None):
        if num_options < 1:
            raise ValueError("need at least one option")
        self.num_options = num_options
        self.mu = mu
        self.p_term = p_term
        self.rng = np.random.default_rng(seed)
        self.current = -1
        self.duration = 0
        self.stats = ExplorerStats()

    def _end_option(self) -> None:
        if self.current >= 0:
            self.stats.durations.append(self.duration)
        self.current = -1
        self.duration = 0

    def terminate(self) -> None:
        """Force the running option to end (used at trajectory cuts such as a death)."""
        self._end_option()

    def decide(self, epsilon: float) -> Decision:
        launched = False
        if self.current >= 0 and self.rng.random() < self.p_term:
            self._end_option()
        if self.current < 0:
            if self.rng.random() < epsilon:
                if self.rng.random() < self.mu:
                    self.current = int(self.rng.integers(self.num_options))
                    self.duration = 0
                    launched = True
                    self.stats.launches[self.current] = self.stats.launches.get(self.current, 0) + 1
                else:
                    self.stats.steps[RANDOM] += 1
                    return Decision(RANDOM)
            else:
                self.stats.steps[GREEDY] += 1
                return Decision(GREEDY)
        self.duration += 1
        self.stats.steps[OPTION] += 1
        return Decision(OPTION, self.current, launched)

    def pop_stats(self) -> ExplorerStats:
        """Return and reset the window statistics (the running option keeps its state)."""
        stats, self.stats = self.stats, ExplorerStats()
        return stats
