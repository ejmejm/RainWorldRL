"""DCEO option execution (Algorithm 1) and the continuing epsilon schedule (no game)."""

import numpy as np
import pytest

from baselines.dceo.options import GREEDY, OPTION, RANDOM, DCEOExplorer, linearly_decaying_epsilon


def test_epsilon_schedule_warmup_decay_and_floor():
    f = lambda step: linearly_decaying_epsilon(1000, step, 200, 0.05)
    assert f(0) == 1.0 and f(200) == 1.0                    # epsilon = 1 until learning starts
    assert f(700) == pytest.approx(1.0 - 0.95 * 0.5)         # linear in between
    assert f(1200) == pytest.approx(0.05) and f(10**9) == pytest.approx(0.05)   # never below the floor
    assert linearly_decaying_epsilon(0, 5, 10, 0.1) == 1.0 and linearly_decaying_epsilon(0, 50, 10, 0.1) == 0.1


def test_exploit_and_random_branches():
    ex = DCEOExplorer(num_options = 4, mu = 0.8, p_term = 0.1, seed = 0)
    assert all(ex.decide(0.0).kind == GREEDY for _ in range(100))
    ex = DCEOExplorer(num_options = 4, mu = 0.0, p_term = 0.1, seed = 0)
    assert all(ex.decide(1.0).kind == RANDOM for _ in range(100))
    assert ex.current == -1


def test_option_runs_until_memoryless_termination():
    ex = DCEOExplorer(num_options = 6, mu = 1.0, p_term = 0.2, seed = 1)
    d = ex.decide(1.0)
    assert d.kind == OPTION and d.launched and 0 <= d.option < 6
    # while it runs, the same option keeps control even with epsilon = 0 (no greedy interruption)
    seq = [d]
    while True:
        nd = ex.decide(0.0)
        if nd.kind != OPTION:
            break
        assert nd.option == d.option and not nd.launched
        seq.append(nd)
    assert nd.kind == GREEDY and ex.stats.durations == [len(seq)]


def test_mean_option_duration_is_one_over_p_term_and_options_are_uniform():
    ex = DCEOExplorer(num_options = 5, mu = 1.0, p_term = 0.1, seed = 2)
    for _ in range(200_000):
        ex.decide(1.0)
    stats = ex.pop_stats()
    assert np.mean(stats.durations) == pytest.approx(10.0, rel = 0.05)
    counts = np.array([stats.launches[k] for k in range(5)])
    assert counts.min() > 0.8 * counts.mean()
    assert stats.steps[OPTION] == 200_000
    assert ex.pop_stats().steps.sum() == 0     # window stats reset


def test_mu_splits_exploratory_steps_between_options_and_primitives():
    ex = DCEOExplorer(num_options = 3, mu = 0.8, p_term = 1.0, seed = 3)   # p_term = 1: one-step options
    kinds = np.array([ex.decide(0.5).kind for _ in range(100_000)])
    assert (kinds == GREEDY).mean() == pytest.approx(0.5, abs = 0.01)
    assert (kinds == OPTION).mean() == pytest.approx(0.5 * 0.8, abs = 0.01)
    assert (kinds == RANDOM).mean() == pytest.approx(0.5 * 0.2, abs = 0.01)


def test_terminate_ends_the_running_option():
    ex = DCEOExplorer(num_options = 2, mu = 1.0, p_term = 0.0, seed = 4)   # would never end on its own
    assert ex.decide(1.0).kind == OPTION
    assert ex.decide(0.0).kind == OPTION
    ex.terminate()
    assert ex.current == -1 and ex.stats.durations == [2]
    assert ex.decide(0.0).kind == GREEDY
