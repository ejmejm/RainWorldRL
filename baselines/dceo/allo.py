"""
Augmented Lagrangian Laplacian Objective (ALLO) with Wayfarer's online-stability choices.

Ported from the official ALLO code (Gomez, Bowling, Machado, "Proper Laplacian
Representation Learning", ICLR 2024),
https://github.com/tarod13/laplacian_dual_dynamics @ b8156180133ab1d77e3ca03d7cefd96a4463ba6e,
``src/trainer/generalized_augmented.py`` (graph loss, orthogonality error matrices,
dual / barrier losses, dual ascent) and ``src/trainer/al.py`` (scalar barrier).

Objective (inside ALLO's max over duals ``beta``)::

    sum_i E[(u_i(x) - u_i(x'))^2]                                   graph-drawing (smoothness) term
  + sum_{j>=k} beta_jk (<u_j, [[u_k]]> - delta_jk)                   asymmetric dual term
  + b * sum_{j>=k} (<u_j, [[u_k]]> - delta_jk)_1 (<u_j, [[u_k]]> - delta_jk)_2   barrier term

``[[.]]`` is a stop-gradient; inner products are batch averages over two
independent batches of states (subscripts 1, 2), so the barrier is an unbiased
estimate of the squared constraint error. ``beta`` is updated by plain gradient
ascent on the batch error estimate (``duals += lr * tril(errors)``) and clipped,
as in the reference. Wayfarer (Lintunen & Machado, arXiv:2610.03604, Sec. 3.3 and
Table 1) keeps the barrier coefficient ``b`` FIXED (the reference grows it) and
applies a symmetric log transform to the representation outputs (done in the
network, see ``networks.RepresentationNet``); duals start at ``dual_init``
(-2.0) on the diagonal.

Equilibrium health signal: at a stationary point ``-beta_ii / 2`` equals the
graph-drawing norm ``E[(u_i(x) - u_i(x'))^2]`` of eigenfunction ``i`` (it is
``2 lambda_i`` for the eigenvalue ``lambda_i`` of the pair-transition Laplacian),
so the two should agree and be (weakly) increasing in ``i``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class AlloAux(NamedTuple):
    graph_loss: jax.Array        # scalar
    dual_loss: jax.Array         # scalar
    barrier_loss: jax.Array      # scalar
    graph_norms: jax.Array       # (d,) per-eigenfunction E[(u_i(x) - u_i(x'))^2]
    errors: jax.Array            # (d, d) lower-triangular constraint errors (stop-gradient), used for dual ascent
    inner: jax.Array             # (d, d) <u_j, u_k> estimate from batch 1 (stop-gradient)


def symlog(x: jax.Array) -> jax.Array:
    """Symmetric log transform ``sign(x) * log(1 + |x|)`` (Wayfarer's output compression)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def init_duals(d: int, dual_init: float) -> jax.Array:
    """Dual variables: lower-triangular ``(d, d)``; diagonal at ``dual_init``, off-diagonal 0."""
    return jnp.eye(d, dtype = jnp.float32) * jnp.float32(dual_init)


def allo_loss(
    u_start: jax.Array, u_end: jax.Array, u_c1: jax.Array, u_c2: jax.Array,
    duals: jax.Array, barrier: float,
) -> tuple[jax.Array, AlloAux]:
    """
    ALLO Lagrangian for one batch.

    ``u_start``/``u_end``: ``(B, d)`` representations of the pair endpoints
    ``(x_t, x_{t+Delta})``; ``u_c1``/``u_c2``: ``(B', d)`` representations of two
    independent batches of states for the orthonormality constraints.
    Gradients flow into the representations only (duals and barrier are
    constants here; ``update_duals`` does the ascent step).
    """
    d = u_start.shape[-1]
    eye = jnp.eye(d, dtype = u_start.dtype)

    graph_norms = jnp.mean((u_start - u_end) ** 2, axis = 0)
    graph_loss = jnp.sum(graph_norms)

    n1 = u_c1.shape[0]
    n2 = u_c2.shape[0]
    inner_1 = jnp.einsum("ij,ik->jk", u_c1, jax.lax.stop_gradient(u_c1)) / n1
    inner_2 = jnp.einsum("ij,ik->jk", u_c2, jax.lax.stop_gradient(u_c2)) / n2
    error_1 = jnp.tril(inner_1 - eye)
    error_2 = jnp.tril(inner_2 - eye)
    errors = 0.5 * (error_1 + error_2)
    quadratic_errors = error_1 * error_2

    dual_loss = jnp.sum(jax.lax.stop_gradient(duals) * errors)
    barrier_loss = jax.lax.stop_gradient(jnp.asarray(barrier, dtype = u_start.dtype)) * jnp.sum(quadratic_errors)

    loss = graph_loss + dual_loss + barrier_loss
    aux = AlloAux(
        graph_loss = graph_loss, dual_loss = dual_loss, barrier_loss = barrier_loss,
        graph_norms = graph_norms, errors = jax.lax.stop_gradient(errors),
        inner = jax.lax.stop_gradient(inner_1),
    )
    return loss, aux


def update_duals(duals: jax.Array, errors: jax.Array, lr: jax.Array | float, clip: float) -> jax.Array:
    """Gradient ascent on the duals: ``tril(clip(duals + lr * tril(errors)))`` (reference ``update_duals``)."""
    updated = jnp.clip(duals + lr * jnp.tril(errors), -clip, clip)
    return jnp.tril(updated)


def eigenvalue_estimates(duals: jax.Array) -> jax.Array:
    """``-beta_ii / 2``: the equilibrium graph-drawing norm of each eigenfunction (see module docstring)."""
    return -0.5 * jnp.diagonal(duals)
