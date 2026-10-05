"""Replay: segment-aware frame stacking, death/not-ready masking of pairs and n-step windows (no game)."""

import numpy as np
import pytest

from baselines.dceo.replay import FrameReplay, StackBuilder, discounted_sampling
from baselines.dceo.train_dceo import compute_cut


def make_replay(n, capacity = None, cuts = (), died = (), rewards = None, stack = 4, seed = 0):
    """Frames carry their own logical index (pixel 0 = i % 256, pixel 1 = i // 256) so stacks can be decoded."""
    rb = FrameReplay(capacity or n + 1, (2, 3), stack, seed = seed)
    for i in range(n):
        r = 0.0 if rewards is None else rewards[i]
        rb.add(frame_for(i), i % 512, r, i in died, i in cuts)
    return rb


def frame_for(i):
    f = np.zeros((2, 3), np.uint8)
    f[0, 0], f[0, 1] = i % 256, i // 256
    return f


def decode(stacks):
    return stacks[:, :, 0, 0].astype(int) + 256 * stacks[:, :, 0, 1].astype(int)


def cuts_between(rb, a, b):
    """Is there a cut after any frame in [a, b) (i.e. does a -> b cross a boundary)?"""
    return np.array([rb.cut[np.arange(x, y) % rb.capacity].any() for x, y in zip(a, b)])


def test_stacks_are_consecutive_and_pad_at_start():
    rb = make_replay(10)
    np.testing.assert_array_equal(decode(rb.stack(np.array([6]))), [[3, 4, 5, 6]])
    np.testing.assert_array_equal(decode(rb.stack(np.array([1]))), [[0, 0, 0, 1]])   # like FrameStack.reset


def test_stacks_never_cross_a_cut():
    rb = make_replay(12, cuts = {4})          # boundary between frame 4 and 5
    np.testing.assert_array_equal(decode(rb.stack(np.array([4]))), [[1, 2, 3, 4]])
    np.testing.assert_array_equal(decode(rb.stack(np.array([5]))), [[5, 5, 5, 5]])
    np.testing.assert_array_equal(decode(rb.stack(np.array([7]))), [[5, 5, 6, 7]])
    np.testing.assert_array_equal(decode(rb.stack(np.array([9]))), [[6, 7, 8, 9]])


def test_stacks_do_not_read_overwritten_frames_after_wraparound():
    rb = make_replay(20, capacity = 8)        # frames 12..19 stored
    assert rb.oldest == 12 and rb.size == 8
    np.testing.assert_array_equal(decode(rb.stack(np.array([13]))), [[12, 12, 12, 13]])
    np.testing.assert_array_equal(decode(rb.stack(np.array([19]))), [[16, 17, 18, 19]])


def test_acting_stack_builder_matches_replay_stacks():
    rng = np.random.default_rng(0)
    n = 60
    cuts = set(np.flatnonzero(rng.random(n) < 0.1).tolist())
    rb = make_replay(n, cuts = cuts)
    sb = StackBuilder(4)
    sb.reset(frame_for(0))
    for i in range(1, n):
        assert decode(sb.get()[None]).tolist() == decode(rb.stack(np.array([i - 1]))).tolist()
        sb.push(frame_for(i), cut = (i - 1) in cuts)


def test_discounted_sampling_range_and_mean():
    rng = np.random.default_rng(0)
    s = discounted_sampling(np.full(200_000, 1000), 0.9, rng)
    assert s.min() == 0 and s.max() < 1000
    assert abs((s + 1).mean() - 10.0) < 0.1          # 1 + Geom: mean 1 / (1 - 0.9)
    short = discounted_sampling(np.array([1, 2, 3] * 1000), 0.9, rng)
    assert (short < np.array([1, 2, 3] * 1000)).all()
    assert (discounted_sampling(np.full(10, 5), 0.0, rng) == 0).all()   # gamma_rep = 0 -> consecutive pairs


def test_pairs_never_cross_a_death_cut_and_respect_the_newest_frame():
    rng = np.random.default_rng(1)
    n = 3000
    cuts = set(np.flatnonzero(rng.random(n) < 0.02).tolist())
    rb = make_replay(n, cuts = cuts, seed = 2)
    t, delta = rb.pair_starts_and_deltas(4000, 0.9, 100)
    assert (delta >= 1).all() and (delta <= 100).all()
    assert (t + delta <= rb.add_count - 1).all()
    assert not rb.cut[t].any()                              # never starts at a death frame / not-ready step
    assert not cuts_between(rb, t, t + delta).any()          # never spans a boundary
    # without masking (no cuts) Delta is the plain truncated geometric
    rb2 = make_replay(n, seed = 3)
    _, d2 = rb2.pair_starts_and_deltas(20_000, 0.9, 100)
    assert abs(d2.mean() - 10.0) < 0.3
    batch = rb.sample_pairs(32, 0.9, 100, separate_constraints = True)
    assert batch["start"].shape == (32, 4, 2, 3) and batch["bits"].shape == (32, 9)
    assert batch["c1"].shape == batch["c2"].shape == (32, 4, 2, 3)
    np.testing.assert_array_equal(decode(batch["end"])[:, -1] - decode(batch["start"])[:, -1], batch["delta"])


def test_option_windows_are_truncated_at_cuts_without_bootstrap():
    rng = np.random.default_rng(4)
    n = 2000
    cuts = set(np.flatnonzero(rng.random(n) < 0.05).tolist())
    rb = make_replay(n, cuts = cuts, seed = 5)
    for _ in range(20):
        b = rb.sample_option(64, 3, 0.99)
        start = decode(b["obs"])[:, -1]
        end = decode(b["next_obs"])[:, -1]
        steps = end - start
        np.testing.assert_array_equal(steps, b["steps"])
        assert (b["steps"] >= 1).all() and (b["steps"] <= 3).all()
        crossed = cuts_between(rb, start, start + b["steps"])
        assert not crossed.any()
        truncated = b["steps"] < 3
        np.testing.assert_allclose(b["discount"][truncated], 0.0)
        np.testing.assert_allclose(b["discount"][~truncated], 0.99 ** 3, rtol = 1e-6)
        # a full window ends at t + 3 only if no cut lies inside it
        assert not cuts_between(rb, start[~truncated], start[~truncated] + 3).any()
        # next-state stacks never contain frames from before a cut
        nxt = decode(b["next_obs"])
        for row, e in zip(nxt, end):
            assert not cuts_between(rb, row, np.full(len(row), e)).any()


def test_task_n_step_returns_continuing_and_death_terminal():
    rewards = np.arange(1, 31, dtype = np.float32)    # reward of step i = i + 1
    rb = make_replay(30, cuts = {10}, died = {10}, rewards = rewards, seed = 6)
    b = rb.sample_task(500, 3, 0.5, death_terminal = False)
    t = decode(b["obs"])[:, -1]
    expect = (t + 1) + 0.5 * (t + 2) + 0.25 * (t + 3)
    np.testing.assert_allclose(b["ret"], expect, rtol = 1e-6)
    np.testing.assert_allclose(b["discount"], 0.125)                      # crosses deaths/cuts: continuing
    assert (decode(b["next_obs"])[:, -1] == t + 3).all()

    b = rb.sample_task(2000, 3, 0.5, death_terminal = True)
    t = decode(b["obs"])[:, -1]
    for ti, ret, disc in zip(t, b["ret"], b["discount"]):
        if ti in (8, 9, 10):                                                # death at step 10 inside the window
            k = 10 - ti                                                     # steps after which it stops
            assert disc == 0.0
            np.testing.assert_allclose(ret, sum(0.5 ** j * (ti + 1 + j) for j in range(k + 1)), rtol = 1e-6)
        else:
            assert disc == pytest.approx(0.125)


def test_compute_cut_rules():
    alive = {"player_dead": False, "ready": True, "in_game": True}
    dead = {"player_dead": True, "ready": True, "in_game": True}
    loading = {"player_dead": False, "ready": False, "in_game": True}
    assert not compute_cut(alive, alive, True, True)
    assert not compute_cut(alive, dead, True, True)        # the step INTO the death frame is a real transition
    assert compute_cut(dead, alive, True, True)            # nothing continues past the death frame
    assert compute_cut(alive, loading, False, True) and compute_cut(loading, alive, False, True)
    assert not compute_cut(dead, loading, False, False)    # masking off: never cut
    assert not compute_cut(dead, alive, False, True)
