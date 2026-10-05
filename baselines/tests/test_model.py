"""Model forward shapes and Bernoulli policy maths (no game)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from baselines.ppo.train_ppo import (
    ActorCritic,
    bernoulli_entropy,
    bernoulli_log_prob,
    greedy_action,
    sample_action,
)


def test_forward_shapes_single_and_batched():
    obs_shape = (4, 54, 96)
    model = ActorCritic(obs_shape, 9, 256, key = jr.PRNGKey(0))
    obs = jnp.zeros(obs_shape, dtype = jnp.uint8)
    logits, value = model(obs)
    assert logits.shape == (9,)
    assert value.shape == ()

    batch = jnp.zeros((5,) + obs_shape, dtype = jnp.uint8)
    logits_b, values_b = jax.vmap(model)(batch)
    assert logits_b.shape == (5, 9)
    assert values_b.shape == (5,)


def test_forward_works_for_rgb_and_other_sizes():
    model = ActorCritic((12, 72, 128), 9, 64, key = jr.PRNGKey(1))
    logits, value = model(jnp.full((12, 72, 128), 255, dtype = jnp.uint8))
    assert logits.shape == (9,) and jnp.isfinite(logits).all() and jnp.isfinite(value)


def test_sample_and_greedy_actions_are_binary():
    model = ActorCritic((4, 54, 96), 9, 32, key = jr.PRNGKey(2))
    obs = jnp.zeros((4, 54, 96), dtype = jnp.uint8)
    action, logp, value = sample_action(model, obs, jr.PRNGKey(3))
    assert action.shape == (9,) and set(np.unique(np.asarray(action))) <= {0.0, 1.0}
    assert logp.shape == () and value.shape == ()
    g_action, g_logp, _ = greedy_action(model, obs)
    logits, _ = model(obs)
    np.testing.assert_array_equal(np.asarray(g_action), (np.asarray(logits) > 0).astype(np.float32))
    # greedy picks the per-key mode, so it is at least as likely as any sample
    assert float(g_logp) >= float(logp) - 1e-5


def test_bernoulli_log_prob_and_entropy_match_hand_computation():
    logits = np.array([[0.0, 2.0, -1.5, 0.3]], dtype = np.float32)
    actions = np.array([[1.0, 0.0, 1.0, 1.0]], dtype = np.float32)
    p = 1.0 / (1.0 + np.exp(-logits))
    expected_logp = np.sum(actions * np.log(p) + (1 - actions) * np.log(1 - p), axis = -1)
    expected_entropy = np.sum(-p * np.log(p) - (1 - p) * np.log(1 - p), axis = -1)

    got_logp = np.asarray(bernoulli_log_prob(jnp.asarray(logits), jnp.asarray(actions)))
    got_entropy = np.asarray(bernoulli_entropy(jnp.asarray(logits)))
    np.testing.assert_allclose(got_logp, expected_logp, rtol = 1e-5, atol = 1e-6)
    np.testing.assert_allclose(got_entropy, expected_entropy, rtol = 1e-5, atol = 1e-6)

    # zero logits: every key is a fair coin -> log(0.5) per key, ln 2 entropy per key
    zeros = jnp.zeros((4,))
    np.testing.assert_allclose(float(bernoulli_log_prob(zeros, jnp.ones((4,)))), 4 * np.log(0.5), rtol = 1e-6)
    np.testing.assert_allclose(float(bernoulli_entropy(zeros)), 4 * np.log(2.0), rtol = 1e-6)
