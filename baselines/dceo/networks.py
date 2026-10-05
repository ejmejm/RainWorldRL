"""
Networks for DCEO with Wayfarer's representation upgrades.

Three disjoint networks, each with its own conv encoder (DCEO and Wayfarer both
keep them separate):

* ``RepresentationNet`` - encoder psi + Laplacian head phi (d eigenfunctions,
  optional symmetric-log output) + multi-step inverse-dynamics head chi
  (9 independent key logits, i.e. the MultiBinary(9) version of Wayfarer's
  softmax over actions).
* ``QNetwork(num_heads = K)`` - the skill (option) network: one shared encoder,
  K dueling Q-heads over the 512 joint actions.
* ``QNetwork(num_heads = 1)`` - the main (task) network: dueling Q over the 512
  joint actions.

Encoder: Nature-DQN conv stack (32x8/4, 64x4/2, 64x3/1, VALID padding as in the
PPO baseline) -> Linear(hidden). LayerNorm before every ReLU when
``layer_norm`` (plasticity; Wayfarer uses LayerNorm, Appendix I), which is also
what Normalize-and-Project needs (``nap_project``). Weights use Xavier-uniform
init with zero bias, as Dopamine's ``NatureDQNNetwork`` / ``FullRainbowNetwork``.

Actions: the MultiBinary(9) key vector is flattened to a joint categorical over
``2**9 = 512`` actions; bit ``k`` of the index is key ``k`` in
``rainworld_rl.KEY_NAMES`` order (``index_to_bits`` / ``bits_to_index``).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from rainworld_rl.shared_memory import NUM_KEYS

from .allo import symlog

NUM_JOINT_ACTIONS = 2 ** NUM_KEYS
# (512, 9) table: row i = key bits of joint action i (bit k <-> key k).
ACTION_BITS = ((np.arange(NUM_JOINT_ACTIONS)[:, None] >> np.arange(NUM_KEYS)[None, :]) & 1).astype(np.int8)
_BIT_WEIGHTS = (1 << np.arange(NUM_KEYS)).astype(np.int64)


# ---------------------------------------------------------------------------
# Action codec
# ---------------------------------------------------------------------------

def index_to_bits(index) -> np.ndarray:
    """Joint action index (scalar or array) -> int8 key bits ``(..., 9)``."""
    return ACTION_BITS[np.asarray(index, dtype = np.int64)]


def bits_to_index(bits) -> np.ndarray:
    """Key bits ``(..., 9)`` in {0, 1} -> joint action index (int64, shape ``(...)``)."""
    b = (np.asarray(bits) > 0).astype(np.int64)
    return (b * _BIT_WEIGHTS).sum(axis = -1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _xavier(layer, key: jax.Array):
    """Xavier-uniform weight + zero bias for a Conv2d / Linear layer (Dopamine's init)."""
    w_shape = layer.weight.shape
    if len(w_shape) == 2:
        fan_out, fan_in = w_shape
    else:  # conv (out, in, kh, kw)
        receptive = int(np.prod(w_shape[2:]))
        fan_in, fan_out = w_shape[1] * receptive, w_shape[0] * receptive
    limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
    w = jr.uniform(key, w_shape, jnp.float32, -limit, limit)
    layer = eqx.tree_at(lambda l: l.weight, layer, w)
    if layer.bias is not None:
        layer = eqx.tree_at(lambda l: l.bias, layer, jnp.zeros_like(layer.bias))
    return layer


class ConvEncoder(eqx.Module):
    """uint8 ``(C, H, W)`` frame stack -> ``(hidden,)`` features (``/255`` inside)."""

    convs: List[eqx.nn.Conv2d]
    conv_norms: List[Optional[eqx.nn.LayerNorm]]
    fc: eqx.nn.Linear
    fc_norm: Optional[eqx.nn.LayerNorm]
    layer_norm: bool = eqx.field(static = True)

    def __init__(self, obs_shape: Tuple[int, int, int], hidden: int, layer_norm: bool, *, key: jax.Array):
        c, h, w = obs_shape
        keys = jr.split(key, 8)
        specs = ((32, 8, 4), (64, 4, 2), (64, 3, 1))
        convs, norms = [], []
        shape = (c, h, w)
        for i, (out_c, k, s) in enumerate(specs):
            conv = _xavier(eqx.nn.Conv2d(shape[0], out_c, k, s, key = keys[i]), keys[i + 3])
            shape = (out_c, (shape[1] - k) // s + 1, (shape[2] - k) // s + 1)
            if shape[1] < 1 or shape[2] < 1:
                raise ValueError(f"observation {obs_shape} is too small for the conv encoder")
            convs.append(conv)
            norms.append(eqx.nn.LayerNorm(shape) if layer_norm else None)
        self.convs = convs
        self.conv_norms = norms
        flat = int(np.prod(shape))
        self.fc = _xavier(eqx.nn.Linear(flat, hidden, key = keys[6]), keys[7])
        self.fc_norm = eqx.nn.LayerNorm(hidden) if layer_norm else None
        self.layer_norm = layer_norm

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs.astype(jnp.float32) / 255.0
        for conv, norm in zip(self.convs, self.conv_norms):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = jax.nn.relu(x)
        x = self.fc(x.reshape(-1))
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        return jax.nn.relu(x)


# ---------------------------------------------------------------------------
# Representation network (Laplacian + inverse dynamics)
# ---------------------------------------------------------------------------

class RepresentationNet(eqx.Module):
    """Encoder psi, Laplacian head phi (``d`` outputs), inverse-dynamics head chi (9 key logits)."""

    encoder: ConvEncoder
    lap_head: eqx.nn.Linear
    inv_fc: eqx.nn.Linear
    inv_norm: Optional[eqx.nn.LayerNorm]
    inv_out: eqx.nn.Linear
    d: int = eqx.field(static = True)
    use_symlog: bool = eqx.field(static = True)

    def __init__(self, obs_shape, d: int, hidden: int, layer_norm: bool, use_symlog: bool, *, key: jax.Array):
        keys = jr.split(key, 7)
        self.encoder = ConvEncoder(obs_shape, hidden, layer_norm, key = keys[0])
        self.lap_head = _xavier(eqx.nn.Linear(hidden, d, key = keys[1]), keys[2])
        self.inv_fc = _xavier(eqx.nn.Linear(2 * hidden, hidden, key = keys[3]), keys[4])
        self.inv_norm = eqx.nn.LayerNorm(hidden) if layer_norm else None
        self.inv_out = _xavier(eqx.nn.Linear(hidden, NUM_KEYS, key = keys[5]), keys[6])
        self.d = d
        self.use_symlog = use_symlog

    def features(self, obs: jax.Array) -> jax.Array:
        return self.encoder(obs)

    def laplacian(self, z: jax.Array) -> jax.Array:
        u = self.lap_head(z)
        return symlog(u) if self.use_symlog else u

    def inverse_logits(self, z_start: jax.Array, z_end: jax.Array) -> jax.Array:
        x = self.inv_fc(jnp.concatenate([z_start, z_end], axis = -1))
        if self.inv_norm is not None:
            x = self.inv_norm(x)
        return self.inv_out(jax.nn.relu(x))

    def __call__(self, obs: jax.Array) -> jax.Array:
        """uint8 frame stack -> eigenfunction values ``(d,)``."""
        return self.laplacian(self.features(obs))


# ---------------------------------------------------------------------------
# Q networks (main: 1 head; skills: K heads on a shared encoder)
# ---------------------------------------------------------------------------

class QNetwork(eqx.Module):
    """Encoder -> ``num_heads`` (dueling) Q-heads over ``num_actions``; output ``(num_heads, num_actions)``."""

    encoder: ConvEncoder
    value: Optional[eqx.nn.Linear]
    advantage: eqx.nn.Linear
    num_heads: int = eqx.field(static = True)
    num_actions: int = eqx.field(static = True)
    dueling: bool = eqx.field(static = True)

    def __init__(self, obs_shape, num_heads: int, num_actions: int, hidden: int, layer_norm: bool,
                 dueling: bool, *, key: jax.Array):
        keys = jr.split(key, 5)
        self.encoder = ConvEncoder(obs_shape, hidden, layer_norm, key = keys[0])
        self.advantage = _xavier(eqx.nn.Linear(hidden, num_heads * num_actions, key = keys[1]), keys[2])
        self.value = _xavier(eqx.nn.Linear(hidden, num_heads, key = keys[3]), keys[4]) if dueling else None
        self.num_heads = num_heads
        self.num_actions = num_actions
        self.dueling = dueling

    def __call__(self, obs: jax.Array) -> jax.Array:
        z = self.encoder(obs)
        adv = self.advantage(z).reshape(self.num_heads, self.num_actions)
        if self.value is None:
            return adv
        value = self.value(z)[:, None]
        return value + adv - jnp.mean(adv, axis = -1, keepdims = True)


# ---------------------------------------------------------------------------
# Normalize-and-Project (Lyle et al., 2024), optional
# ---------------------------------------------------------------------------

def _normalized_layer_getters(model) -> List[Callable]:
    """Getters for the weights of every layer that is followed by a LayerNorm."""
    getters: List[Callable] = []

    def encoder_getters(prefix: Callable) -> List[Callable]:
        out = [lambda m, i = i: prefix(m).convs[i].weight for i in range(3)]
        out.append(lambda m: prefix(m).fc.weight)
        return out

    if isinstance(model, RepresentationNet):
        getters += encoder_getters(lambda m: m.encoder)
        if model.inv_norm is not None:
            getters.append(lambda m: m.inv_fc.weight)
    elif isinstance(model, QNetwork):
        getters += encoder_getters(lambda m: m.encoder)
    else:
        raise TypeError(f"unsupported model type {type(model)}")
    return getters


def nap_reference_norms(model) -> Tuple[jax.Array, ...]:
    """Frobenius norms of the normalised layers' weights at init (the NaP projection targets)."""
    return tuple(jnp.linalg.norm(g(model)) for g in _normalized_layer_getters(model))


def nap_project(model, ref_norms: Sequence[jax.Array]):
    """Rescale every LayerNorm-followed weight back to its reference norm (NaP's projection step)."""
    getters = _normalized_layer_getters(model)
    new_weights = [g(model) * (n / (jnp.linalg.norm(g(model)) + 1e-12)) for g, n in zip(getters, ref_norms)]
    return eqx.tree_at(lambda m: [g(m) for g in getters], model, new_weights)
