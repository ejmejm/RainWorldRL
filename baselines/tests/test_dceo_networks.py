"""DCEO networks: forward shapes, action codec, option layout, intrinsic rewards, NaP (no game)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from baselines.dceo.networks import (
    ACTION_BITS,
    NUM_JOINT_ACTIONS,
    QNetwork,
    RepresentationNet,
    bits_to_index,
    index_to_bits,
    nap_project,
    nap_reference_norms,
)
from baselines.dceo.train_dceo import intrinsic_rewards, option_layout

OBS = (4, 54, 96)


def test_action_codec_round_trip_and_key_order():
    assert NUM_JOINT_ACTIONS == 512 and ACTION_BITS.shape == (512, 9)
    idx = np.arange(512)
    np.testing.assert_array_equal(bits_to_index(index_to_bits(idx)), idx)
    # every key combination appears exactly once
    assert len({tuple(r) for r in ACTION_BITS}) == 512
    # bit k <-> key k of KEY_NAMES (left, right, up, down, jump, grab, throw, map, special)
    np.testing.assert_array_equal(index_to_bits(0), np.zeros(9))
    np.testing.assert_array_equal(index_to_bits(1), [1, 0, 0, 0, 0, 0, 0, 0, 0])      # left
    np.testing.assert_array_equal(index_to_bits(2 + 16), [0, 1, 0, 0, 1, 0, 0, 0, 0])  # right + jump
    assert int(bits_to_index(np.array([0, 0, 0, 1, 0, 0, 0, 0, 1]))) == 8 + 256
    assert ACTION_BITS.dtype == np.int8   # what env.step expects (MultiBinary)


def test_representation_shapes_and_symlog():
    rep = RepresentationNet(OBS, d = 6, hidden = 64, layer_norm = True, use_symlog = True, key = jr.PRNGKey(0))
    x = jnp.zeros((3,) + OBS, jnp.uint8).at[1].set(255)
    z = jax.vmap(rep.features)(x)
    u = jax.vmap(rep)(x)
    logits = jax.vmap(rep.inverse_logits)(z, z[::-1])
    assert z.shape == (3, 64) and u.shape == (3, 6) and logits.shape == (3, 9)
    assert jnp.isfinite(u).all() and jnp.isfinite(logits).all()
    raw = jax.vmap(lambda zz: rep.lap_head(zz))(z)
    np.testing.assert_allclose(np.asarray(u), np.sign(raw) * np.log1p(np.abs(raw)), rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("layer_norm,dueling", [(True, True), (False, False)])
def test_q_network_shapes(layer_norm, dueling):
    main = QNetwork(OBS, 1, NUM_JOINT_ACTIONS, 64, layer_norm, dueling, key = jr.PRNGKey(1))
    skill = QNetwork(OBS, 14, NUM_JOINT_ACTIONS, 64, layer_norm, dueling, key = jr.PRNGKey(2))
    x = jnp.zeros((2,) + OBS, jnp.uint8)
    assert main(x[0]).shape == (1, 512)
    q = jax.vmap(skill)(x)
    assert q.shape == (2, 14, 512) and jnp.isfinite(q).all()


def test_encoder_rejects_too_small_frames():
    with pytest.raises(ValueError):
        QNetwork((4, 18, 32), 1, 512, 32, True, True, key = jr.PRNGKey(0))


def test_option_layout_both_directions_and_positive():
    dims, signs = option_layout(8, "both")
    assert len(dims) == 2 * (8 - 1) == 14
    assert dims[:4] == (1, 1, 2, 2) and signs[:4] == (1.0, -1.0, 1.0, -1.0)
    assert 0 not in dims   # the constant eigenfunction has no option
    dims_p, signs_p = option_layout(8, "positive")
    assert dims_p == tuple(range(1, 8)) and set(signs_p) == {1.0}


def test_intrinsic_rewards_are_signed_eigenfunction_changes():
    u_start = jnp.array([[0.5, 0.1, -0.2], [0.5, 0.3, 0.3]])
    u_end = jnp.array([[0.5, 0.4, -0.5], [0.5, 0.3, 1.3]])
    dims, signs = option_layout(3, "both")   # (+u1, -u1, +u2, -u2)
    r = np.asarray(intrinsic_rewards(u_start, u_end, dims, signs))
    np.testing.assert_allclose(r, [[0.3, -0.3, -0.3, 0.3], [0.0, 0.0, 1.0, -1.0]], atol = 1e-6)
    # the n-step reward telescopes: sum of one-step changes == end - start
    u_mid = jnp.array([[0.5, 0.2, 0.0], [0.5, 0.0, 0.0]])
    two_step = intrinsic_rewards(u_start, u_mid, dims, signs) + intrinsic_rewards(u_mid, u_end, dims, signs)
    np.testing.assert_allclose(np.asarray(two_step), r, atol = 1e-6)


def test_nap_projection_restores_reference_norms():
    rep = RepresentationNet(OBS, 4, 32, True, True, key = jr.PRNGKey(3))
    ref = nap_reference_norms(rep)
    grown = jax.tree.map(lambda a: a * 3.0 if a.ndim >= 2 else a, rep)
    projected = nap_project(grown, ref)
    np.testing.assert_allclose(np.asarray(nap_reference_norms(projected)), np.asarray(ref), rtol = 1e-5)
    # the Laplacian head is not LayerNorm-ed, so it is not projected
    np.testing.assert_allclose(np.asarray(projected.lap_head.weight), np.asarray(grown.lap_head.weight))
