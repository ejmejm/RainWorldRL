"""Two-head GAE: intrinsic non-episodic, extrinsic continuing or death-terminal (no game)."""

import jax.numpy as jnp
import numpy as np

from baselines.ppo.train_ppo import compute_gae as ppo_compute_gae
from baselines.ppo_rnd.train_ppo_rnd import compute_dual_gae, compute_gae


def _j(x):
    return jnp.asarray(np.asarray(x, dtype = np.float32))


def test_gae_without_terminals_equals_the_ppo_baseline():
    rng = np.random.default_rng(0)
    rewards, values = rng.normal(size = 16), rng.normal(size = 16)
    last = np.float32(0.3)
    adv, ret = compute_gae(_j(rewards), _j(values), _j(last), _j(np.ones(16)), 0.97, 0.9)
    adv0, ret0 = ppo_compute_gae(_j(rewards), _j(values), _j(last), 0.97, 0.9)
    np.testing.assert_allclose(np.asarray(adv), np.asarray(adv0), rtol = 1e-5, atol = 1e-6)
    np.testing.assert_allclose(np.asarray(ret), np.asarray(ret0), rtol = 1e-5, atol = 1e-6)


def test_terminal_cuts_bootstrap_and_trace_at_that_transition():
    g, lam = 0.9, 0.5
    rewards, values, last = [1.0, 0.0, 0.5], [0.2, 0.4, 0.6], 0.8
    adv, ret = compute_gae(_j(rewards), _j(values), _j(last), _j([1.0, 0.0, 1.0]), g, lam)
    d0 = 1.0 + g * 0.4 - 0.2
    d1 = 0.0 - 0.4                    # transition 1 -> 2 is terminal: no V(s_2) bootstrap
    d2 = 0.5 + g * 0.8 - 0.6
    a2 = d2
    a1 = d1                           # and the trace from step 2 does not leak back
    a0 = d0 + g * lam * a1
    np.testing.assert_allclose(np.asarray(adv), [a0, a1, a2], rtol = 1e-5, atol = 1e-6)
    np.testing.assert_allclose(np.asarray(ret), np.asarray([a0, a1, a2]) + np.asarray(values), rtol = 1e-5)


def test_dual_gae_intrinsic_ignores_death_and_extrinsic_death_is_optional():
    rng = np.random.default_rng(1)
    n = 12
    deaths = np.zeros(n)
    deaths[4] = 1.0
    batch = {
        "ext_rewards": _j(rng.normal(size = n)), "ext_values": _j(rng.normal(size = n)), "last_ext_value": _j(0.5),
        "int_rewards": _j(rng.uniform(size = n)), "int_values": _j(rng.normal(size = n)), "last_int_value": _j(1.5),
        "deaths": _j(deaths),
    }
    g_e, g_i, lam = 0.99, 0.9, 0.95
    cont = compute_dual_gae(batch, g_e, g_i, lam, ext_death_terminal = False)
    term = compute_dual_gae(batch, g_e, g_i, lam, ext_death_terminal = True)

    ones = _j(np.ones(n))
    ext_cont = compute_gae(batch["ext_rewards"], batch["ext_values"], batch["last_ext_value"], ones, g_e, lam)
    ext_term = compute_gae(batch["ext_rewards"], batch["ext_values"], batch["last_ext_value"], 1.0 - batch["deaths"], g_e, lam)
    int_nonep = compute_gae(batch["int_rewards"], batch["int_values"], batch["last_int_value"], ones, g_i, lam)
    int_masked = compute_gae(batch["int_rewards"], batch["int_values"], batch["last_int_value"], 1.0 - batch["deaths"], g_i, lam)

    np.testing.assert_allclose(np.asarray(cont[0]), np.asarray(ext_cont[0]), rtol = 1e-6)
    np.testing.assert_allclose(np.asarray(term[0]), np.asarray(ext_term[0]), rtol = 1e-6)
    assert not np.allclose(np.asarray(cont[0])[:5], np.asarray(term[0])[:5])   # the flag matters up to the death
    np.testing.assert_allclose(np.asarray(cont[0])[5:], np.asarray(term[0])[5:], rtol = 1e-6)  # and not after it
    # intrinsic: identical in both modes, never cut at the death step, uses gamma_I
    for mode in (cont, term):
        np.testing.assert_allclose(np.asarray(mode[2]), np.asarray(int_nonep[0]), rtol = 1e-6)
        np.testing.assert_allclose(np.asarray(mode[3]), np.asarray(int_nonep[1]), rtol = 1e-6)
    assert not np.allclose(np.asarray(int_nonep[0])[:5], np.asarray(int_masked[0])[:5])
