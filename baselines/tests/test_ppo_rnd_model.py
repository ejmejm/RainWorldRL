"""PPO+RND networks: forward shapes, torch-style orthogonal init, the raw bonus definition (no game)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from baselines.ppo_rnd.rnd import RNDPredictor, RNDTarget, intrinsic_reward, orthogonal_init
from baselines.ppo.train_ppo import bernoulli_log_prob
from baselines.ppo_rnd.train_ppo_rnd import RNDActorCritic, act, parse_args, unpack_step, values_of


def test_actor_critic_forward_shapes_single_and_batched():
    obs_shape = (4, 54, 96)
    agent = RNDActorCritic(obs_shape, 9, 256, 448, key = jr.PRNGKey(0))
    logits, v_ext, v_int = agent(jnp.zeros(obs_shape, dtype = jnp.uint8))
    assert logits.shape == (9,) and v_ext.shape == () and v_int.shape == ()

    logits_b, v_ext_b, v_int_b = jax.vmap(agent)(jnp.zeros((5,) + obs_shape, dtype = jnp.uint8))
    assert logits_b.shape == (5, 9) and v_ext_b.shape == (5,) and v_int_b.shape == (5,)

    # RGB stack, other frame size
    agent = RNDActorCritic((12, 36, 48), 9, 32, 64, key = jr.PRNGKey(1))
    logits, v_ext, v_int = agent(jnp.full((12, 36, 48), 255, dtype = jnp.uint8))
    assert jnp.isfinite(logits).all() and jnp.isfinite(v_ext) and jnp.isfinite(v_int)


def test_act_samples_binary_keys_and_two_values():
    agent = RNDActorCritic((4, 36, 48), 9, 32, 64, key = jr.PRNGKey(2))
    obs = jnp.zeros((4, 36, 48), dtype = jnp.uint8)
    rng = jr.PRNGKey(3)
    packed, next_rng = act(agent, obs, rng)
    assert packed.shape == (12,) and packed.dtype == jnp.float32
    assert not np.array_equal(np.asarray(next_rng), np.asarray(rng))        # the rng advances inside the call
    action, logp, v_ext, v_int = unpack_step(np.asarray(packed))
    assert action.shape == (9,) and set(np.unique(action)) <= {0.0, 1.0}
    logits, ve, vi = agent(obs)
    np.testing.assert_allclose(logp, float(bernoulli_log_prob(logits, jnp.asarray(action))), rtol = 1e-5)
    np.testing.assert_allclose(v_ext, float(ve), rtol = 1e-5)
    np.testing.assert_allclose(v_int, float(vi), rtol = 1e-5)
    ve2, vi2 = values_of(agent, obs)
    np.testing.assert_allclose(float(ve2), v_ext, rtol = 1e-5)
    np.testing.assert_allclose(float(vi2), v_int, rtol = 1e-5)


def test_rnd_target_and_predictor_shapes_and_predictor_is_deeper():
    frame_shape = (1, 54, 96)
    target = RNDTarget(frame_shape, 512, key = jr.PRNGKey(0))
    predictor = RNDPredictor(frame_shape, 512, 512, key = jr.PRNGKey(1))
    x = jnp.zeros(frame_shape)
    assert target(x).shape == (512,) and predictor(x).shape == (512,)

    def n_linear(m):
        return sum(isinstance(l, eqx.nn.Linear) for l in jax.tree.leaves(m, is_leaf = lambda l: isinstance(l, eqx.nn.Linear)))

    assert n_linear(target) == 1 and n_linear(predictor) == 3


def test_orthogonal_init_has_torch_semantics_for_conv_and_linear():
    k = jr.split(jr.PRNGKey(0), 4)
    conv = orthogonal_init(eqx.nn.Conv2d(1, 32, 8, 4, key = k[0]), float(np.sqrt(2.0)), k[1])
    w = np.asarray(conv.weight).reshape(32, -1)            # torch views the kernel as (out, in*kh*kw)
    np.testing.assert_allclose(w @ w.T, 2.0 * np.eye(32), atol = 1e-5)
    assert float(jnp.abs(conv.bias).max()) == 0.0

    lin = orthogonal_init(eqx.nn.Linear(448, 9, key = k[2]), 0.01, k[3])
    w = np.asarray(lin.weight)
    np.testing.assert_allclose(w @ w.T, 1e-4 * np.eye(9), atol = 1e-8)


def test_intrinsic_reward_is_half_squared_error_on_whitened_clipped_frame():
    k = jr.split(jr.PRNGKey(4), 3)
    frame_shape = (1, 36, 48)
    target = RNDTarget(frame_shape, 16, key = k[0])
    predictor = RNDPredictor(frame_shape, 16, 32, key = k[1])
    frames = jr.randint(k[2], (3,) + frame_shape, 0, 256).astype(jnp.uint8)
    mean = jnp.full(frame_shape, 100.0)
    std = jnp.full(frame_shape, 10.0)            # most pixels land outside [-5, 5] and get clipped

    got = intrinsic_reward(predictor, target, frames, mean, std)
    x = jnp.clip((frames.astype(jnp.float32) - mean) / std, -5.0, 5.0)
    expected = jnp.sum((jax.vmap(target)(x) - jax.vmap(predictor)(x)) ** 2, axis = -1) / 2.0
    assert got.shape == (3,)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol = 1e-5)


def test_defaults_follow_the_reference_and_the_brief():
    a = parse_args(["--fake"])
    assert a.novelty is False                       # intrinsic method: no hand-written room bonus by default
    assert (a.gamma, a.int_gamma, a.gae_lambda) == (0.99, 0.99, 0.95)
    assert (a.lr, a.rnd_lr, a.ent_coef, a.clip) == (1e-4, 1e-4, 0.001, 0.1)
    assert (a.ext_coef, a.int_coef, a.update_proportion) == (2.0, 1.0, 0.25)
    assert a.clip_vloss and a.norm_adv and not a.anneal_lr and not a.ext_death_terminal
    assert a.log_interval == a.rollout_steps
    b = parse_args(["--novelty", "--rnd_lr", "3e-5", "--ext_death_terminal", "--no_clip_vloss"])
    assert b.novelty is True and b.rnd_lr == 3e-5 and b.ext_death_terminal and not b.clip_vloss
