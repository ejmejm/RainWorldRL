"""ALLO loss on a known graph: stationarity at the true eigenvectors and learning them from samples (no game)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from baselines.dceo.allo import allo_loss, eigenvalue_estimates, init_duals, symlog, update_duals


def chain_eigenvectors(n: int, d: int):
    """
    Random walk on a path of ``n`` states with reflecting ends (P = 1/2 left / 1/2 right,
    staying put at the walls) is symmetric, so with uniform state weights its Laplacian
    eigenvectors are the DCT-II basis cos(pi k (i + 1/2) / n) with eigenvalues
    1 - cos(pi k / n). Normalised so that E_uniform[v_k^2] = 1.
    """
    i = np.arange(n)
    v = np.stack([np.cos(np.pi * k * (i + 0.5) / n) for k in range(d)], axis = 1)
    v /= np.sqrt((v ** 2).mean(axis = 0))
    lam = 1.0 - np.cos(np.pi * np.arange(d) / n)
    return v.astype(np.float32), lam.astype(np.float32)


def test_symlog_and_dual_update():
    np.testing.assert_allclose(np.asarray(symlog(jnp.array([-3.0, 0.0, 2.0]))), [-np.log(4.0), 0.0, np.log(3.0)], rtol = 1e-6)
    duals = init_duals(3, -2.0)
    np.testing.assert_array_equal(np.asarray(duals), -2.0 * np.eye(3))
    errors = jnp.array([[1.0, 5.0, 5.0], [2.0, 1.0, 5.0], [3.0, 4.0, 1.0]])
    new = np.asarray(update_duals(duals, errors, 0.5, 100.0))
    assert np.allclose(np.triu(new, 1), 0.0)                       # stays lower triangular
    np.testing.assert_allclose(np.tril(new), np.tril(-2.0 * np.eye(3) + 0.5 * np.asarray(errors)))
    assert np.abs(np.asarray(update_duals(duals, errors * 1e6, 1.0, 10.0))).max() <= 10.0


def test_allo_is_stationary_at_true_eigenvectors_with_equilibrium_duals():
    n, d = 10, 4
    v, lam = chain_eigenvectors(n, d)
    states = np.arange(n)
    # exact expectation over (s ~ uniform, s' ~ P(s, .)): every state with both moves (clipped at the walls)
    starts = np.concatenate([states, states])
    ends = np.concatenate([np.clip(states - 1, 0, n - 1), np.clip(states + 1, 0, n - 1)])
    graph_norms = ((v[starts] - v[ends]) ** 2).mean(axis = 0)
    np.testing.assert_allclose(graph_norms, 2.0 * lam, atol = 1e-5)          # E[(u(s)-u(s'))^2] = 2 lambda
    duals = jnp.diag(jnp.asarray(-2.0 * graph_norms))                        # equilibrium: beta_ii = -2 g_i = -4 lambda_i
    np.testing.assert_allclose(np.asarray(eigenvalue_estimates(duals)), graph_norms, rtol = 1e-6)

    def lagrangian(u):
        return allo_loss(u[starts], u[ends], u, u, duals, barrier = 0.5)[0]

    grad = np.asarray(jax.grad(lagrangian)(jnp.asarray(v)))
    assert np.abs(grad).max() < 1e-5
    # ...and it is not stationary for a rotation of the eigenvectors (the point of ALLO's asymmetry)
    c, s = np.cos(0.3), np.sin(0.3)
    rot = np.eye(d, dtype = np.float32)
    rot[1:3, 1:3] = [[c, -s], [s, c]]
    assert np.abs(np.asarray(jax.grad(lagrangian)(jnp.asarray(v @ rot)))).max() > 1e-3


def test_allo_learns_ordered_chain_eigenvectors_up_to_sign():
    """Tabular encoder + symlog outputs + fixed barrier 0.5 + duals initialised at -2 (Wayfarer settings)."""
    n, d, batch, steps = 16, 4, 256, 24_000
    v, lam = chain_eigenvectors(n, d)
    optimizer = optax.adam(3e-3)

    def rep(w, s):
        return symlog(w[s])

    def step(carry, key):
        w, opt_state, duals = carry
        k1, k2, k3, k4 = jr.split(key, 4)
        s = jr.randint(k1, (batch,), 0, n)
        s_next = jnp.clip(s + 2 * jr.bernoulli(k2, 0.5, (batch,)).astype(jnp.int32) - 1, 0, n - 1)
        c1 = jr.randint(k3, (batch,), 0, n)
        c2 = jr.randint(k4, (batch,), 0, n)
        (_, aux), grads = jax.value_and_grad(
            lambda w_: allo_loss(rep(w_, s), rep(w_, s_next), rep(w_, c1), rep(w_, c2), duals, 0.5), has_aux = True)(w)
        updates, opt_state = optimizer.update(grads, opt_state)
        w = optax.apply_updates(w, updates)
        duals = update_duals(duals, aux.errors, 3e-4, 100.0)
        return (w, opt_state, duals), aux.graph_norms

    w0 = 0.1 * jr.normal(jr.PRNGKey(0), (n, d))
    carry = (w0, optimizer.init(w0), init_duals(d, -2.0))
    (w, _, duals), _ = jax.jit(lambda c, ks: jax.lax.scan(step, c, ks))(carry, jr.split(jr.PRNGKey(1), steps))

    u = np.asarray(rep(w, jnp.arange(n)))
    cos = np.abs((u * v).sum(0)) / (np.linalg.norm(u, axis = 0) * np.linalg.norm(v, axis = 0))
    assert (cos > 0.95).all(), cos                                        # eigenvector k in output k (ordered)
    np.testing.assert_allclose((u ** 2).mean(axis = 0), 1.0, atol = 0.1)  # orthonormal under uniform weights
    eig = np.asarray(eigenvalue_estimates(duals))
    np.testing.assert_allclose(eig, 2.0 * lam, atol = 0.03)                # duals recover the eigenvalues
    assert np.all(np.diff(eig) > 0)
