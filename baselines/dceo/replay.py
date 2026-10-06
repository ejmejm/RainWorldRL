"""
Single-stream circular replay of single frames, with frame stacks rebuilt at
sampling time and trajectory "cuts" that stacks, Laplacian / inverse-dynamics
pairs and option n-step windows never cross.

Storage (index ``i`` = the ``i``-th env step of the lifetime, modulo capacity):

* ``frames[i]``  - the (grayscale uint8) frame the agent acted on at step ``i``;
* ``actions[i]`` - joint action index (0..511) taken at that frame;
* ``rewards[i]`` - extrinsic reward of that step;
* ``died[i]``    - the step ended in a death (``info["player_dead"]`` returned
                   with the NEXT frame); only used when the task head treats death
                   as terminal (``death_terminal``);
* ``cut[i]``     - frame ``i + 1`` is NOT a continuation of frame ``i`` for
                   representation purposes (set by the trainer from its masking
                   flags: after a death frame, and around steps where the agent is
                   not in control). With masking off it is always False.

The newest frame (returned by the latest ``env.step``) is not stored until the
next ``add``; transitions are only sampled where every frame they read exists.

Frame stacks: ``stack(i)`` = frames ``i-k+1 .. i``; positions before a cut (or
before the oldest stored frame) are replaced by the first frame of the segment,
exactly what ``baselines.common.env.FrameStack`` does on ``reset()`` and what
the trainer's acting-side ``StackBuilder`` does at a cut.

This mirrors Dopamine's ``OutOfGraphReplayBuffer`` (single frames + stacking
at sample time, n-step returns truncated at terminals) without episodes: the
only boundaries are the cuts and (optionally) deaths.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .networks import ACTION_BITS


def discounted_sampling(ranges: np.ndarray, discount: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample ``i`` in ``0 .. n-1`` with ``P(i) prop. to discount**i`` for each ``n`` in
    ``ranges`` (inverse-CDF sampling; port of ALLO's ``discounted_sampling``).
    """
    ranges = np.asarray(ranges)
    assert np.min(ranges) >= 1
    seeds = rng.uniform(size = ranges.shape)
    if discount <= 0.0:
        return np.zeros(ranges.shape, dtype = np.int64)
    if discount >= 1.0:
        return np.floor(seeds * ranges).astype(np.int64)
    samples = np.log(1.0 - (1.0 - np.power(discount, ranges)) * seeds) / np.log(discount)
    return np.minimum(np.floor(samples).astype(np.int64), ranges - 1)


class StackBuilder:
    """Acting-side frame stack that restarts (repeats the new frame) at a cut, matching ``FrameReplay.stack``."""

    def __init__(self, k: int):
        self.k = k
        self._frames: Optional[np.ndarray] = None

    def reset(self, frame: np.ndarray) -> None:
        self._frames = np.repeat(frame[None], self.k, axis = 0)

    def push(self, frame: np.ndarray, cut: bool) -> None:
        if cut or self._frames is None:
            self.reset(frame)
        else:
            self._frames = np.concatenate([self._frames[1:], frame[None]], axis = 0)

    def get(self) -> np.ndarray:
        assert self._frames is not None, "call reset() first"
        return self._frames


class FrameReplay:
    def __init__(self, capacity: int, frame_shape: Tuple[int, int], frame_stack: int, seed: Optional[int] = None):
        if capacity < 2:
            raise ValueError("replay capacity must be >= 2")
        self.capacity = int(capacity)
        self.frame_stack = int(frame_stack)
        self.frames = np.zeros((self.capacity,) + tuple(frame_shape), dtype = np.uint8)
        self.actions = np.zeros(self.capacity, dtype = np.int16)
        self.rewards = np.zeros(self.capacity, dtype = np.float32)
        self.died = np.zeros(self.capacity, dtype = bool)
        self.cut = np.zeros(self.capacity, dtype = bool)
        self.add_count = 0
        self.rng = np.random.default_rng(seed)

    # -- writing -------------------------------------------------------------
    def add(self, frame: np.ndarray, action: int, reward: float, died: bool, cut: bool) -> None:
        i = self.add_count % self.capacity
        self.frames[i] = frame
        self.actions[i] = action
        self.rewards[i] = reward
        self.died[i] = died
        self.cut[i] = cut
        self.add_count += 1

    # -- bookkeeping ---------------------------------------------------------
    @property
    def size(self) -> int:
        return min(self.add_count, self.capacity)

    @property
    def oldest(self) -> int:
        """Logical index of the oldest stored frame."""
        return self.add_count - self.size

    def _p(self, logical: np.ndarray) -> np.ndarray:
        return np.mod(logical, self.capacity)

    def _cut_eff(self, logical: np.ndarray) -> np.ndarray:
        """Cut after ``logical``, also True for the newest stored frame (its successor is not stored yet)."""
        return self.cut[self._p(logical)] | (logical >= self.add_count - 1)

    # -- stacks --------------------------------------------------------------
    def stack_indices(self, t: np.ndarray) -> np.ndarray:
        """Logical frame indices ``(B, k)`` of the stacks ending at ``t`` (oldest first), segment-aware."""
        t = np.asarray(t, dtype = np.int64)
        k = self.frame_stack
        idx = np.empty(t.shape + (k,), dtype = np.int64)
        idx[..., k - 1] = t
        cur = t.copy()
        alive = np.ones(t.shape, dtype = bool)
        oldest = self.oldest
        for o in range(1, k):
            prev = t - o
            ok = alive & (prev >= oldest)
            ok &= ~self.cut[self._p(np.maximum(prev, 0))]
            cur = np.where(ok, prev, cur)
            alive = ok
            idx[..., k - 1 - o] = cur
        return idx

    def stack(self, t: np.ndarray) -> np.ndarray:
        """uint8 frame stacks ``(B, k, H, W)`` ending at logical indices ``t``."""
        return self.frames[self._p(self.stack_indices(t))]

    # -- index sampling ------------------------------------------------------
    def _sample_starts(self, batch: int, max_offset: int, require_no_cut: bool) -> np.ndarray:
        """Uniform logical starts ``t`` with frame ``t + max_offset`` stored (and ``cut[t]`` False if asked)."""
        lo, hi = self.oldest, self.add_count - 1 - max_offset
        if hi < lo:
            raise ValueError(f"replay has {self.size} frames; not enough to sample (offset {max_offset})")
        out = np.empty(0, dtype = np.int64)
        for _ in range(100):
            cand = self.rng.integers(lo, hi + 1, size = 2 * batch + 8)
            if require_no_cut:
                cand = cand[~self.cut[self._p(cand)]]
            out = np.concatenate([out, cand])
            if out.size >= batch:
                return out[:batch]
        raise RuntimeError("could not sample transitions that do not start at a cut (is masking cutting every step?)")

    # -- batches -------------------------------------------------------------
    def sample_task(self, batch: int, n_step: int, gamma: float, death_terminal: bool) -> Dict[str, np.ndarray]:
        """
        n-step transitions for the main (task) head. Continuing: the window
        crosses cuts and deaths; with ``death_terminal`` it is truncated after a
        death step (reward included, no bootstrap).
        """
        t = self._sample_starts(batch, n_step, require_no_cut = False)
        offs = t[:, None] + np.arange(n_step)[None, :]
        rewards = self.rewards[self._p(offs)]
        powers = gamma ** np.arange(n_step, dtype = np.float64)
        if death_terminal:
            died = self.died[self._p(offs)]
            # step j counts iff no death strictly before j
            before = np.concatenate([np.zeros((batch, 1), dtype = bool), np.cumsum(died, axis = 1)[:, :-1] > 0], axis = 1)
            alive = ~before
            ret = np.sum(rewards * powers[None, :] * alive, axis = 1)
            terminal = died.any(axis = 1)
        else:
            ret = rewards @ powers
            terminal = np.zeros(batch, dtype = bool)
        discount = np.where(terminal, 0.0, gamma ** n_step)
        return {
            "obs": self.stack(t),
            "action": self.actions[self._p(t)].astype(np.int32),
            "ret": ret.astype(np.float32),
            "next_obs": self.stack(t + n_step),
            "discount": discount.astype(np.float32),
        }

    def sample_option(self, batch: int, n_step: int, gamma: float) -> Dict[str, np.ndarray]:
        """
        Transitions for the option heads: start ``t`` (not at a cut) and end
        ``t + m`` where ``m = n_step`` or, if a cut falls inside the window, the
        last frame before it (then ``discount = 0``: no intrinsic bootstrap across
        the boundary). The intrinsic reward ``u(x_{t+m}) - u(x_t)`` is computed in
        the update with the current representation.
        """
        t = self._sample_starts(batch, n_step, require_no_cut = True)
        offs = t[:, None] + np.arange(n_step)[None, :]
        window = self.cut[self._p(offs)]
        has_cut = window.any(axis = 1)
        m = np.where(has_cut, np.argmax(window, axis = 1), n_step)
        discount = np.where(has_cut, 0.0, gamma ** n_step)
        return {
            "obs": self.stack(t),
            "action": self.actions[self._p(t)].astype(np.int32),
            "next_obs": self.stack(t + m),
            "discount": discount.astype(np.float32),
            "steps": m.astype(np.int32),
        }

    def pair_starts_and_deltas(self, batch: int, gamma_rep: float, max_delta: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pair indices ``(t, t + Delta)`` with ``P(Delta) prop. to gamma_rep**(Delta - 1)``,
        ``1 <= Delta``, truncated at the segment end (next cut / newest frame) and
        at ``max_delta`` (Wayfarer / ALLO pair sampling without episodes).
        """
        t = self._sample_starts(batch, 1, require_no_cut = True)
        offs = t[:, None] + np.arange(max_delta)[None, :]
        window = self._cut_eff(offs)
        has_cut = window.any(axis = 1)
        delta_max = np.where(has_cut, np.argmax(window, axis = 1), max_delta)
        delta_max = np.maximum(delta_max, 1)   # cut[t] is False and t+1 is stored, so this is already >= 1
        delta = 1 + discounted_sampling(delta_max, gamma_rep, self.rng)
        return t, delta

    def sample_pairs(self, batch: int, gamma_rep: float, max_delta: int, separate_constraints: bool) -> Dict[str, np.ndarray]:
        """Laplacian + inverse-dynamics batch: ``(x_t, a_t, x_{t+Delta})`` (+ two independent state batches)."""
        t, delta = self.pair_starts_and_deltas(batch, gamma_rep, max_delta)
        out = {
            "start": self.stack(t),
            "end": self.stack(t + delta),
            "bits": ACTION_BITS[self.actions[self._p(t)].astype(np.int64)].astype(np.float32),
            "delta": delta.astype(np.int32),
        }
        if separate_constraints:
            out["c1"] = self.stack(self._sample_starts(batch, 1, require_no_cut = True))
            out["c2"] = self.stack(self._sample_starts(batch, 1, require_no_cut = True))
        return out
