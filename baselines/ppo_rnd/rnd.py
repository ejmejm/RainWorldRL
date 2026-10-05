"""
Random Network Distillation components (Burda, Edwards, Storkey, Klimov,
"Exploration by Random Network Distillation", ICLR 2019, arXiv:1810.12894).

Ported from CleanRL ``cleanrl/ppo_rnd_envpool.py`` (vwxyzjn/cleanrl master
@ fe8d8a03c41a7ef5b523e2e354bd01c363e786bb; the file itself last changed in
35896b1fefa9898b904f7e09bcbe6e168e15d2a9), with the official
openai/random-network-distillation (@ f75c0f1e) as the tie-breaker.

* ``RNDTarget``     fixed random CNN: 3 LeakyReLU convs -> Linear(512).
* ``RNDPredictor``  trained CNN: same convs -> ReLU(512) -> ReLU(512) -> Linear(512)
                    (the "extra layers" of the paper / CleanRL).
* ``RunningMeanStd``      gym's parallel-moments running mean/var (float64,
                          initial count 1e-4, var 1), used per pixel for the
                          observation whitening and as a scalar for the
                          intrinsic-return std.
* ``RewardForwardFilter`` discounted running sum of intrinsic rewards over
                          time (gamma_I); its outputs feed the scalar
                          ``RunningMeanStd`` whose std divides the bonus.
* ``RNDStats``            the three objects above for one lifetime; never reset.

The RND input is the single latest frame of the policy's frame stack, as raw
pixel values (0..255, no /255), whitened per pixel with the running mean/std
and clipped to [-5, 5]. The policy itself never sees the whitened frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

OBS_CLIP = 5.0


# ---------------------------------------------------------------------------
# Init (torch semantics)
# ---------------------------------------------------------------------------

def orthogonal_init(layer, gain: float, key: jax.Array, bias: float = 0.0):
    """
    ``torch.nn.init.orthogonal_(w, gain)`` + constant bias, as CleanRL's
    ``layer_init``. Torch views a conv kernel ``(out, in, kh, kw)`` as the
    matrix ``(out, in*kh*kw)``; doing the same here keeps the random target's
    feature scale identical to the reference (jax's initializer on the raw 4-D
    shape would orthogonalise over the kernel-width axis instead).
    """
    shape = layer.weight.shape
    flat = (shape[0], int(np.prod(shape[1:])))
    w = jax.nn.initializers.orthogonal(gain)(key, flat, jnp.float32).reshape(shape)
    layer = eqx.tree_at(lambda l: l.weight, layer, w)
    if layer.bias is not None:
        layer = eqx.tree_at(lambda l: l.bias, layer, jnp.full_like(layer.bias, bias))
    return layer


def nature_convs(in_channels: int, keys) -> Tuple[eqx.nn.Conv2d, eqx.nn.Conv2d, eqx.nn.Conv2d]:
    """The Nature-DQN conv stack (32x8s4, 64x4s2, 64x3s1), orthogonal gain sqrt(2), zero bias."""
    g = float(np.sqrt(2.0))
    c1 = orthogonal_init(eqx.nn.Conv2d(in_channels, 32, 8, 4, key = keys[0]), g, keys[1])
    c2 = orthogonal_init(eqx.nn.Conv2d(32, 64, 4, 2, key = keys[2]), g, keys[3])
    c3 = orthogonal_init(eqx.nn.Conv2d(64, 64, 3, 1, key = keys[4]), g, keys[5])
    return c1, c2, c3


def conv_flat_dim(in_shape: Tuple[int, int, int]) -> int:
    """Flattened size after ``nature_convs`` for a channel-first input of ``in_shape``."""
    convs = nature_convs(in_shape[0], jr.split(jr.PRNGKey(0), 6))

    def f(x):
        for c in convs:
            x = c(x)
        return x.reshape(-1)

    return int(jax.eval_shape(f, jax.ShapeDtypeStruct(in_shape, jnp.float32)).shape[0])


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class RNDTarget(eqx.Module):
    """Fixed random network f(o): convs (LeakyReLU) -> Linear(rep_size). Never trained."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    fc: eqx.nn.Linear

    def __init__(self, in_shape: Tuple[int, int, int], rep_size: int = 512, *, key: jax.Array):
        k = jr.split(key, 8)
        self.conv1, self.conv2, self.conv3 = nature_convs(in_shape[0], k[:6])
        flat = conv_flat_dim(in_shape)
        self.fc = orthogonal_init(eqx.nn.Linear(flat, rep_size, key = k[6]), float(np.sqrt(2.0)), k[7])

    def __call__(self, x: jax.Array) -> jax.Array:
        """``x``: whitened float32 ``(c, H, W)`` -> features ``(rep_size,)``."""
        x = jax.nn.leaky_relu(self.conv1(x))
        x = jax.nn.leaky_relu(self.conv2(x))
        x = jax.nn.leaky_relu(self.conv3(x))
        return self.fc(x.reshape(-1))


class RNDPredictor(eqx.Module):
    """Trained network f_hat(o): convs (LeakyReLU) -> ReLU(hidden) -> ReLU(hidden) -> Linear(rep_size)."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear

    def __init__(self, in_shape: Tuple[int, int, int], rep_size: int = 512, hidden: int = 512, *, key: jax.Array):
        k = jr.split(key, 12)
        g = float(np.sqrt(2.0))
        self.conv1, self.conv2, self.conv3 = nature_convs(in_shape[0], k[:6])
        flat = conv_flat_dim(in_shape)
        self.fc1 = orthogonal_init(eqx.nn.Linear(flat, hidden, key = k[6]), g, k[7])
        self.fc2 = orthogonal_init(eqx.nn.Linear(hidden, hidden, key = k[8]), g, k[9])
        self.fc3 = orthogonal_init(eqx.nn.Linear(hidden, rep_size, key = k[10]), g, k[11])

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.leaky_relu(self.conv1(x))
        x = jax.nn.leaky_relu(self.conv2(x))
        x = jax.nn.leaky_relu(self.conv3(x))
        x = jax.nn.relu(self.fc1(x.reshape(-1)))
        x = jax.nn.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Whitening and the bonus
# ---------------------------------------------------------------------------

def latest_frame(obs: Union[np.ndarray, jax.Array], frame_channels: int):
    """Last frame of a channel-first stack ``(..., k*c, H, W)`` -> ``(..., c, H, W)``."""
    return obs[..., -frame_channels:, :, :]


def whiten(frames: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    """Per-pixel ``(x - mean) / std`` on raw pixel values, clipped to [-5, 5] (RND Sec. 2.4)."""
    return jnp.clip((frames.astype(jnp.float32) - mean) / std, -OBS_CLIP, OBS_CLIP)


def prediction_error(predictor: RNDPredictor, target: RNDTarget, x: jax.Array) -> jax.Array:
    """Per-sample bonus on whitened inputs ``x`` ``(N, c, H, W)``: ``sum((f(x) - f_hat(x))^2) / 2`` (CleanRL)."""
    tgt = jax.lax.stop_gradient(jax.vmap(target)(x))
    pred = jax.vmap(predictor)(x)
    return jnp.sum((tgt - pred) ** 2, axis = -1) / 2.0


@eqx.filter_jit
def intrinsic_reward(predictor: RNDPredictor, target: RNDTarget, frames: jax.Array,
                     mean: jax.Array, std: jax.Array) -> jax.Array:
    """Raw (unnormalised) RND bonus for raw uint8 frames ``(N, c, H, W)`` given the obs statistics."""
    return prediction_error(predictor, target, whiten(frames, mean, std))


# ---------------------------------------------------------------------------
# Running statistics (numpy float64, never reset during a lifetime)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """``gym.wrappers.normalize.RunningMeanStd``: parallel-moments mean/var, initial count ``epsilon``."""

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype = np.float64)
        self.var = np.ones(shape, dtype = np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype = np.float64)
        self.update_from_moments(x.mean(axis = 0), x.var(axis = 0), x.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = np.asarray(new_mean, dtype = np.float64)
        self.var = np.asarray(m2 / tot_count, dtype = np.float64)
        self.count = float(tot_count)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


class RewardForwardFilter:
    """``rewems <- rewems * gamma + r`` (first call: ``rewems = r``); the running discounted intrinsic return."""

    def __init__(self, gamma: float):
        self.rewems: Optional[np.ndarray] = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class RNDStats:
    """
    Lifetime statistics of one RND agent: per-pixel ``obs_rms`` over the latest
    frame, the intrinsic ``RewardForwardFilter`` and the scalar ``reward_rms``
    of its outputs. Nothing here is ever reset (continuing env).
    """

    def __init__(self, frame_shape: Tuple[int, int, int], gamma_int: float):
        self.obs_rms = RunningMeanStd(shape = tuple(frame_shape))
        self.reward_rms = RunningMeanStd(shape = ())
        self.rff = RewardForwardFilter(gamma_int)

    # -- intrinsic reward normalisation ---------------------------------------
    def normalise_intrinsic(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ``raw``: one env's bonuses in time order ``(T,)``. Runs the forward
        filter along time (as the official code does; CleanRL's transpose walks
        the env axis instead), updates ``reward_rms`` with the moments of the
        filtered returns, and returns (``raw / sqrt(reward_rms.var)``, filtered
        returns). No mean subtraction, no clipping.
        """
        raw = np.asarray(raw, dtype = np.float64)
        rets = np.array([self.rff.update(r) for r in raw], dtype = np.float64)
        self.reward_rms.update_from_moments(float(np.mean(rets)), float(np.var(rets)), len(rets))
        return raw / np.sqrt(self.reward_rms.var), rets

    @property
    def int_return_std(self) -> float:
        return float(np.sqrt(self.reward_rms.var))

    # -- observation statistics -------------------------------------------------
    def update_obs(self, frames: np.ndarray) -> None:
        """Fold a batch of raw latest frames ``(N, c, H, W)`` into the per-pixel statistics."""
        self.obs_rms.update(frames)

    def obs_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """Current whitening statistics as float32 arrays ``(c, H, W)``."""
        return self.obs_rms.mean.astype(np.float32), np.sqrt(self.obs_rms.var).astype(np.float32)

    # -- persistence ------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        rewems = np.nan if self.rff.rewems is None else float(self.rff.rewems)
        np.savez(
            path, obs_mean = self.obs_rms.mean, obs_var = self.obs_rms.var, obs_count = self.obs_rms.count,
            rew_mean = self.reward_rms.mean, rew_var = self.reward_rms.var, rew_count = self.reward_rms.count,
            rff_rewems = rewems, rff_gamma = self.rff.gamma,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RNDStats":
        d = np.load(path)
        stats = cls(tuple(d["obs_mean"].shape), float(d["rff_gamma"]))
        stats.obs_rms.mean, stats.obs_rms.var, stats.obs_rms.count = d["obs_mean"], d["obs_var"], float(d["obs_count"])
        stats.reward_rms.mean, stats.reward_rms.var = d["rew_mean"], d["rew_var"]
        stats.reward_rms.count = float(d["rew_count"])
        rewems = float(d["rff_rewems"])
        stats.rff.rewems = None if np.isnan(rewems) else rewems
        return stats
