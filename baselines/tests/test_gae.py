"""GAE for a continuing environment (no dones, bootstrap from the last value)."""

import jax.numpy as jnp
import numpy as np

from baselines.ppo.train_ppo import compute_gae


def test_gae_matches_hand_computation():
    rewards = np.array([1.0, 0.0, 0.5], dtype = np.float32)
    values = np.array([0.2, 0.4, 0.6], dtype = np.float32)
    last_value = np.float32(0.8)
    gamma, lam = 0.9, 0.5

    # deltas: r_t + gamma * V_{t+1} - V_t with V_3 = last_value
    d0 = 1.0 + 0.9 * 0.4 - 0.2
    d1 = 0.0 + 0.9 * 0.6 - 0.4
    d2 = 0.5 + 0.9 * 0.8 - 0.6
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    expected_adv = np.array([a0, a1, a2], dtype = np.float32)

    adv, ret = compute_gae(jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(last_value), gamma, lam)
    np.testing.assert_allclose(np.asarray(adv), expected_adv, rtol = 1e-5, atol = 1e-6)
    np.testing.assert_allclose(np.asarray(ret), expected_adv + values, rtol = 1e-5, atol = 1e-6)


def test_gae_with_lambda_one_is_discounted_return_minus_value():
    rewards = np.array([0.0, 0.0, 1.0, 0.0], dtype = np.float32)
    values = np.zeros(4, dtype = np.float32)
    last_value = np.float32(2.0)
    gamma = 0.5
    adv, ret = compute_gae(jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(last_value), gamma, 1.0)
    # R_3 = 0 + 0.5*2 = 1; R_2 = 1 + 0.5*1 = 1.5; R_1 = 0.75; R_0 = 0.375
    np.testing.assert_allclose(np.asarray(ret), [0.375, 0.75, 1.5, 1.0], rtol = 1e-6)
    np.testing.assert_allclose(np.asarray(adv), np.asarray(ret), rtol = 1e-6)
