"""
DCEO (Deep Covering Eigenoptions; Klissarov & Machado, ICML 2023) with the
representation upgrades of Wayfarer (Lintunen & Machado, arXiv:2610.03604):
an exploration baseline for the continuing Rain World env.

Three networks, each with its own encoder, all trained off-policy from one
replay buffer, one gradient step each per update; ``--updates_per_step`` (default 0.25,
i.e. every 4 env steps as in DCEO / Wayfarer) sets the replay ratio:

1. Representation: ALLO Laplacian head (``--d`` eigenfunctions, fixed barrier,
   diagonal dual init -2, symmetric-log outputs, slower dual ascent) trained on
   ``(x_t, x_{t+Delta})`` pairs with ``Delta ~ Geometric(--gamma_rep)``, plus a
   multi-step inverse-dynamics head (9-key multi-label BCE, weight ``--inv_weight``).
2. Skills: ``K = 2(d-1)`` option Q-heads on a shared encoder; option ``2(i-1)``
   / ``2(i-1)+1`` is rewarded by ``+/-(u_i(x_{t+n}) - u_i(x_t))`` (telescoped
   n-step eigenfunction change) and learns double-DQN values over the 512
   joint actions.
3. Main: double DQN + n-step + dueling on the extrinsic (drive) reward over the
   512 joint actions; DCEO exploration (epsilon -> option with prob ``--mu``,
   memoryless termination ``--p_term``; see ``options.py``).

Continuing-env adaptations (flags, default ON): no episode truncation (pairs and
windows only stop at cuts); cuts after a death frame (``--mask_death``) and around
steps where the agent is not in control (``--mask_not_ready``) remove those
transitions from the Laplacian / inverse-dynamics pairs, from frame stacks, and
from the option heads' intrinsic bootstrapping; the task head treats death as
non-terminal unless ``--death_terminal``.

Run from the repository root::

    python -m baselines.dceo.train_dceo --fake --total_steps 3000 --min_replay 500
    python -m baselines.dceo.train_dceo --launch --kill_game --game_lock E:/projects/rainworld_rl-wt/.game-lock
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mlflow
import numpy as np
import optax
from tqdm import tqdm

import rainworld_rl
from rainworld_rl.shared_memory import KEY_NAMES

from baselines.common.env import attach, make_env
from baselines.common.game_lock import game_lock
from baselines.common.metrics import RolloutMetrics, format_summary
from baselines.ppo.train_ppo import DEFAULT_MLFLOW_URI, setup_mlflow

from .allo import allo_loss, eigenvalue_estimates, init_duals, update_duals
from .networks import (
    ACTION_BITS,
    NUM_JOINT_ACTIONS,
    QNetwork,
    RepresentationNet,
    nap_project,
    nap_reference_norms,
)
from .options import GREEDY, OPTION, RANDOM, DCEOExplorer, linearly_decaying_epsilon
from .replay import FrameReplay, StackBuilder

logger = logging.getLogger("baselines.dceo")

DEFAULT_CKPT_ROOT = Path(__file__).resolve().parent / "checkpoints"
MODEL_CONFIG_FILENAME = "model_config.json"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def add_bool(group, name: str, default: bool, help: str = "") -> None:
    """``--name`` / ``--no_name`` pair (underscore style, like ``--no_wipe``)."""
    group.add_argument(f"--{name}", dest = name, action = "store_true", help = f"{help} [--no_{name} turns it off]".strip())
    group.add_argument(f"--no_{name}", dest = name, action = "store_false", help = argparse.SUPPRESS)
    group.set_defaults(**{name: default})


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog = "python -m baselines.dceo.train_dceo",
        description = "DCEO + Wayfarer representation (ALLO + inverse dynamics) for Rain World RL "
                      "(continuing env, 512-way joint action over the 9 keys).",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    # env
    g = p.add_argument_group("env")
    g.add_argument("--fake", action = "store_true", help = "Use FakeRainWorldEnv (no game)")
    g.add_argument("--launch", action = "store_true", help = "(Re)start Rain World with the deployed DLL before training")
    g.add_argument("--kill_game", action = "store_true", help = "Close the game when training ends")
    g.add_argument("--no_wipe", action = "store_true", help = "Attach to the game in progress instead of wiping the RL save")
    g.add_argument("--game_lock", type = str, default = None, help = "mkdir-lock directory held for launch + run + kill")
    g.add_argument("--frame_width", type = int, default = 96)
    g.add_argument("--frame_height", type = int, default = 54)
    g.add_argument("--ticks_per_step", type = int, default = 4)
    g.add_argument("--frame_stack", type = int, default = 4, help = "Frames per state (stacked from replay at sample time)")
    add_bool(g, "novelty", False,
                   help = "Include the hand-written NewRoom bonus in the extrinsic reward. Off by default: an "
                          "intrinsic-exploration method must not be graded on (or helped by) the room bonus")
    g.add_argument("--reward_scale", type = float, default = 1.0, help = "Multiplier on the extrinsic reward")
    # budget
    g = p.add_argument_group("budget")
    g.add_argument("--total_steps", type = int, default = 1_000_000, help = "Env steps to run")
    g.add_argument("--seconds", type = float, default = None, help = "Wall-clock budget of the training loop (stop at whichever comes first)")
    # options / exploration
    g = p.add_argument_group("options and exploration (DCEO)")
    g.add_argument("--d", type = int, default = 8, help = "Number of Laplacian eigenfunctions")
    g.add_argument("--option_directions", choices = ("both", "positive"), default = "both",
                   help = "both: K = 2(d-1) options (+/- each non-constant eigenfunction, Wayfarer); "
                          "positive: K = d-1 (one direction, DCEO-like)")
    g.add_argument("--p_term", type = float, default = 0.1, help = "Per-step option termination probability (DCEO D = 10)")
    g.add_argument("--mu", type = float, default = 0.8, help = "Probability an exploratory step launches an option")
    g.add_argument("--epsilon_final", type = float, default = 0.05, help = "Epsilon floor kept for the whole lifetime")
    g.add_argument("--epsilon_decay_steps", type = int, default = 250_000, help = "Linear 1.0 -> epsilon_final after warm-up")
    g.add_argument("--option_epsilon", type = float, default = 0.0, help = "Random-action prob. inside a running option (official code uses epsilon)")
    # task (main) head
    g = p.add_argument_group("main and skill Q-learning")
    g.add_argument("--gamma", type = float, default = 0.99)
    g.add_argument("--n_step", type = int, default = 3)
    g.add_argument("--lr", type = float, default = 1e-4, help = "Adam step size of the main and skill networks")
    g.add_argument("--adam_eps", type = float, default = 1.5e-4)
    g.add_argument("--batch_size", type = int, default = 32, help = "Main and skill minibatch")
    add_bool(g, "double_dqn", True)
    add_bool(g, "dueling", True)
    g.add_argument("--huber_delta", type = float, default = 1.0)
    g.add_argument("--death_terminal", action = "store_true", help = "Main head: no bootstrap past a death step")
    # representation
    g = p.add_argument_group("representation (ALLO + inverse dynamics, Wayfarer)")
    g.add_argument("--rep_lr", type = float, default = 3e-4, help = "Adam step size of the representation network")
    g.add_argument("--rep_adam_eps", type = float, default = 1e-8)
    g.add_argument("--dual_lr", type = float, default = 3e-5, help = "Gradient-ascent step size of the ALLO duals")
    g.add_argument("--dual_init", type = float, default = -2.0, help = "Initial value of the diagonal duals")
    g.add_argument("--dual_clip", type = float, default = 100.0)
    g.add_argument("--barrier", type = float, default = 0.5, help = "Fixed ALLO barrier coefficient b")
    g.add_argument("--rep_batch", type = int, default = 64, help = "Laplacian / inverse-dynamics pair minibatch (Wayfarer 512)")
    g.add_argument("--separate_constraint_batches", action = "store_true",
                   help = "Sample two extra state batches for the orthonormality constraints (reference ALLO); default "
                          "reuses the two halves of the pair starts (independent uniform samples, half the encoder cost)")
    add_bool(g, "symlog", True, help = "Symmetric-log transform on the Laplacian outputs")
    g.add_argument("--inv_weight", type = float, default = 1.0, help = "Inverse-dynamics loss weight (0 disables)")
    g.add_argument("--gamma_rep", type = float, default = 0.9, help = "P(Delta) prop. to gamma_rep^(Delta-1); 0 = consecutive pairs (DCEO)")
    g.add_argument("--max_delta", type = int, default = 100, help = "Cap on Delta")
    # continuing-env masking
    g = p.add_argument_group("continuing env")
    add_bool(g, "mask_death", True,
                   help = "Cut trajectories after a death frame (pairs, stacks, option bootstrap)")
    add_bool(g, "mask_not_ready", True,
                   help = "Cut around steps with info['ready'] False (death screen / reload / menus: no control)")
    # networks / optimisation
    g = p.add_argument_group("networks")
    g.add_argument("--hidden", type = int, default = 512)
    add_bool(g, "layer_norm", True)
    g.add_argument("--nap", action = "store_true", help = "Normalize-and-Project: keep LayerNorm-ed weights at their initial norm")
    g.add_argument("--max_grad_norm", type = float, default = 30.0, help = "Global-norm clip (all optimisers, Wayfarer)")
    # replay / schedule
    g = p.add_argument_group("replay and schedule")
    g.add_argument("--replay_capacity", type = int, default = 1_000_000, help = "Frames (96x54 uint8: ~5.2 GB at 1M)")
    g.add_argument("--min_replay", type = int, default = 20_000, help = "Frames before learning starts (epsilon = 1 until then)")
    g.add_argument("--updates_per_step", type = float, default = 0.25,
                   help = "Replay ratio: gradient updates (one step of each network) per env step; 0.25 = every 4 env "
                          "steps (DCEO / Wayfarer). Values > 1 run several updates per env step (GPU)")
    g.add_argument("--target_update_period", type = int, default = 10_000, help = "Env steps between target syncs")
    add_bool(g, "parallel_updates", True,
                   help = "Run the three networks' update steps in concurrent threads (same maths, faster on CPU)")
    add_bool(g, "async_updates", True,
                   help = "Run each update in the background while the next env steps are collected "
                          "(acting params at most one update stale)")
    # bookkeeping
    g = p.add_argument_group("bookkeeping")
    g.add_argument("--seed", type = int, default = None, help = "Random if omitted")
    g.add_argument("--log_interval", type = int, default = 1000, help = "Env steps between MLflow logs")
    g.add_argument("--ckpt_interval", type = int, default = 50_000, help = "Env steps between checkpoints")
    g.add_argument("--ckpt_dir", type = str, default = None, help = "Default baselines/dceo/checkpoints/<run_id>")
    g.add_argument("--mlflow_uri", type = str, default = DEFAULT_MLFLOW_URI)
    g.add_argument("--experiment", type = str, default = "rainworld-dceo")
    g.add_argument("--run_name", type = str, default = None)
    args = p.parse_args(argv)
    if args.seed is None:
        args.seed = random.randrange(2**31)
    if args.d < 2:
        p.error("--d must be >= 2 (the first eigenfunction is constant and gets no option)")
    if args.nap and not args.layer_norm:
        p.error("--nap needs --layer_norm")
    if args.updates_per_step <= 0:
        p.error("--updates_per_step must be > 0")
    if args.rep_batch < 4 or args.rep_batch % 2:
        p.error("--rep_batch must be even and >= 4")
    args.min_replay = max(args.min_replay, args.n_step + 2)
    if args.replay_capacity < args.min_replay:
        p.error("--replay_capacity must be >= --min_replay")
    return args


def option_layout(d: int, directions: str) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """(eigenfunction index, sign) of every option. ``both``: option 2(i-1) = +u_i, 2(i-1)+1 = -u_i, i = 1..d-1."""
    dims, signs = [], []
    for i in range(1, d):
        dims.append(i); signs.append(1.0)
        if directions == "both":
            dims.append(i); signs.append(-1.0)
    return tuple(dims), tuple(signs)


# ---------------------------------------------------------------------------
# State / metrics
# ---------------------------------------------------------------------------

class DCEOModels(eqx.Module):
    """What a checkpoint holds."""
    rep: RepresentationNet
    skill: QNetwork
    main: QNetwork
    duals: jax.Array


class TrainState(eqx.Module):
    rep: RepresentationNet
    rep_opt_state: optax.OptState
    duals: jax.Array
    skill: QNetwork
    skill_target: QNetwork
    skill_opt_state: optax.OptState
    main: QNetwork
    main_target: QNetwork
    main_opt_state: optax.OptState
    nap_norms: Optional[Tuple[Tuple[jax.Array, ...], ...]]
    rng: jax.Array
    step: jax.Array  # gradient updates so far (int32)

    rep_optimizer: optax.GradientTransformation = eqx.field(static = True)
    q_optimizer: optax.GradientTransformation = eqx.field(static = True)
    option_dims: Tuple[int, ...] = eqx.field(static = True)
    option_signs: Tuple[float, ...] = eqx.field(static = True)
    barrier: float = eqx.field(static = True)
    dual_lr: float = eqx.field(static = True)
    dual_clip: float = eqx.field(static = True)
    inv_weight: float = eqx.field(static = True)
    double_dqn: bool = eqx.field(static = True)
    huber_delta: float = eqx.field(static = True)
    separate_constraints: bool = eqx.field(static = True)
    nap: bool = eqx.field(static = True)


class StepMetrics(eqx.Module):
    main_loss: jax.Array
    main_q: jax.Array
    main_target: jax.Array
    main_grad_norm: jax.Array
    skill_loss: jax.Array
    skill_q: jax.Array
    skill_grad_norm: jax.Array
    skill_bootstrap_frac: jax.Array    # fraction of option windows that bootstrap (not cut)
    intr_mean: jax.Array               # (K,)
    intr_std: jax.Array                # (K,)
    intr_abs_mean: jax.Array           # scalar
    rep_loss: jax.Array
    graph_loss: jax.Array
    dual_loss: jax.Array
    barrier_loss: jax.Array
    orth_error: jax.Array              # Frobenius norm of the constraint-error matrix
    graph_norms: jax.Array             # (d,)
    inner_diag: jax.Array              # (d,)
    u_std: jax.Array                   # (d,)
    duals_diag: jax.Array              # (d,) after the ascent step
    inv_loss: jax.Array
    inv_acc_key: jax.Array
    inv_acc_exact: jax.Array
    rep_grad_norm: jax.Array
    delta_mean: jax.Array


def build_networks(args: argparse.Namespace, key: jax.Array) -> Tuple[RepresentationNet, QNetwork, QNetwork]:
    obs_shape = (args.frame_stack, args.frame_height, args.frame_width)
    k_rep, k_skill, k_main = jr.split(key, 3)
    num_options = len(option_layout(args.d, args.option_directions)[0])
    rep = RepresentationNet(obs_shape, args.d, args.hidden, args.layer_norm, args.symlog, key = k_rep)
    skill = QNetwork(obs_shape, num_options, NUM_JOINT_ACTIONS, args.hidden, args.layer_norm, args.dueling, key = k_skill)
    main = QNetwork(obs_shape, 1, NUM_JOINT_ACTIONS, args.hidden, args.layer_norm, args.dueling, key = k_main)
    return rep, skill, main


def init_experiment(args: argparse.Namespace) -> TrainState:
    key = jr.PRNGKey(args.seed)
    net_key, rng = jr.split(key)
    rep, skill, main = build_networks(args, net_key)
    dims, signs = option_layout(args.d, args.option_directions)
    rep_optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.rep_lr, eps = args.rep_adam_eps))
    q_optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps = args.adam_eps))
    nap_norms = (nap_reference_norms(rep), nap_reference_norms(skill), nap_reference_norms(main)) if args.nap else None
    return TrainState(
        rep = rep,
        rep_opt_state = rep_optimizer.init(eqx.filter(rep, eqx.is_array)),
        duals = init_duals(args.d, args.dual_init),
        skill = skill,
        skill_target = skill,
        skill_opt_state = q_optimizer.init(eqx.filter(skill, eqx.is_array)),
        main = main,
        main_target = main,
        main_opt_state = q_optimizer.init(eqx.filter(main, eqx.is_array)),
        nap_norms = nap_norms,
        rng = rng,
        step = jnp.asarray(0, dtype = jnp.int32),
        rep_optimizer = rep_optimizer,
        q_optimizer = q_optimizer,
        option_dims = dims,
        option_signs = signs,
        barrier = args.barrier,
        dual_lr = args.dual_lr,
        dual_clip = args.dual_clip,
        inv_weight = args.inv_weight,
        double_dqn = args.double_dqn,
        huber_delta = args.huber_delta,
        separate_constraints = args.separate_constraint_batches,
        nap = args.nap,
    )


# ---------------------------------------------------------------------------
# Losses and the jitted update (all three networks, one gradient step each)
# ---------------------------------------------------------------------------

def intrinsic_rewards(u_start: jax.Array, u_end: jax.Array, dims: Tuple[int, ...], signs: Tuple[float, ...]) -> jax.Array:
    """``(B, K)`` option rewards ``sign_k * (u_{dim_k}(x_end) - u_{dim_k}(x_start))`` (telescoped n-step change)."""
    du = u_end - u_start
    return du[:, jnp.asarray(dims)] * jnp.asarray(signs, dtype = du.dtype)[None, :]


def _take_actions(q: jax.Array, actions: jax.Array) -> jax.Array:
    """``q`` ``(B, H, A)``, ``actions`` ``(B,)`` or ``(B, H)`` -> ``(B, H)``."""
    if actions.ndim == 1:
        actions = jnp.broadcast_to(actions[:, None], q.shape[:2])
    return jnp.take_along_axis(q, actions[..., None], axis = -1)[..., 0]


def q_targets(online: QNetwork, target: QNetwork, next_obs: jax.Array, rewards: jax.Array, discount: jax.Array,
              double_dqn: bool) -> jax.Array:
    """n-step (double) DQN targets ``(B, H)``; ``rewards`` is ``(B, H)``, ``discount`` ``(B,)`` (0 = no bootstrap)."""
    q_next_target = jax.vmap(target)(next_obs)
    q_select = jax.vmap(online)(next_obs) if double_dqn else q_next_target
    next_v = _take_actions(q_next_target, jnp.argmax(q_select, axis = -1))
    return jax.lax.stop_gradient(rewards + discount[:, None] * next_v)


def q_loss(model: QNetwork, obs: jax.Array, actions: jax.Array, targets: jax.Array, huber_delta: float):
    q_sa = _take_actions(jax.vmap(model)(obs), actions)
    loss = optax.huber_loss(q_sa, targets, delta = huber_delta).mean()
    return loss, q_sa.mean()


def rep_loss(rep: RepresentationNet, duals: jax.Array, batch: Dict[str, jax.Array], barrier: float,
             inv_weight: float, separate_constraints: bool):
    z_start = jax.vmap(rep.features)(batch["r_start"])
    z_end = jax.vmap(rep.features)(batch["r_end"])
    u_start = jax.vmap(rep.laplacian)(z_start)
    u_end = jax.vmap(rep.laplacian)(z_end)
    if separate_constraints:
        u_c1 = jax.vmap(rep)(batch["r_c1"])
        u_c2 = jax.vmap(rep)(batch["r_c2"])
    else:
        half = u_start.shape[0] // 2
        u_c1, u_c2 = u_start[:half], u_start[half:2 * half]
    lap_loss, aux = allo_loss(u_start, u_end, u_c1, u_c2, duals, barrier)

    logits = jax.vmap(rep.inverse_logits)(z_start, z_end)
    bits = batch["r_bits"]
    inv_loss = optax.sigmoid_binary_cross_entropy(logits, bits).sum(axis = -1).mean()
    correct = (logits > 0) == (bits > 0.5)
    total = lap_loss + inv_weight * inv_loss
    return total, (aux, inv_loss, correct.mean(), correct.all(axis = -1).mean(), u_start)


def _subset(batch: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {k: v for k, v in batch.items() if k.startswith(prefix)}


@eqx.filter_jit
def main_step(ts: TrainState, batch: Dict[str, jax.Array]):
    """Main (task) head: one n-step double-DQN step on the extrinsic reward."""
    targets = q_targets(ts.main, ts.main_target, batch["m_next_obs"], batch["m_ret"][:, None],
                        batch["m_discount"], ts.double_dqn)
    (loss, q_mean), grads = eqx.filter_value_and_grad(q_loss, has_aux = True)(
        ts.main, batch["m_obs"], batch["m_action"], targets, ts.huber_delta)
    updates, opt_state = ts.q_optimizer.update(grads, ts.main_opt_state, eqx.filter(ts.main, eqx.is_array))
    main = eqx.apply_updates(ts.main, updates)
    if ts.nap:
        main = nap_project(main, ts.nap_norms[2])
    metrics = {"main_loss": loss, "main_q": q_mean, "main_target": targets.mean(), "main_grad_norm": optax.tree.norm(grads)}
    return main, opt_state, metrics


@eqx.filter_jit
def skill_step(ts: TrainState, batch: Dict[str, jax.Array]):
    """Skill heads: intrinsic rewards from the current representation, one n-step double-DQN step for all K heads."""
    u_s = jax.lax.stop_gradient(jax.vmap(ts.rep)(batch["s_obs"]))
    u_e = jax.lax.stop_gradient(jax.vmap(ts.rep)(batch["s_next_obs"]))
    r_int = intrinsic_rewards(u_s, u_e, ts.option_dims, ts.option_signs)
    targets = q_targets(ts.skill, ts.skill_target, batch["s_next_obs"], r_int, batch["s_discount"], ts.double_dqn)
    (loss, q_mean), grads = eqx.filter_value_and_grad(q_loss, has_aux = True)(
        ts.skill, batch["s_obs"], batch["s_action"], targets, ts.huber_delta)
    updates, opt_state = ts.q_optimizer.update(grads, ts.skill_opt_state, eqx.filter(ts.skill, eqx.is_array))
    skill = eqx.apply_updates(ts.skill, updates)
    if ts.nap:
        skill = nap_project(skill, ts.nap_norms[1])
    metrics = {
        "skill_loss": loss, "skill_q": q_mean, "skill_grad_norm": optax.tree.norm(grads),
        "skill_bootstrap_frac": jnp.mean(batch["s_discount"] > 0),
        "intr_mean": r_int.mean(axis = 0), "intr_std": r_int.std(axis = 0), "intr_abs_mean": jnp.abs(r_int).mean(),
    }
    return skill, opt_state, metrics


@eqx.filter_jit
def rep_step(ts: TrainState, batch: Dict[str, jax.Array]):
    """Representation: one step on ALLO + inverse dynamics, then one dual-ascent step."""
    (total, (aux, inv_loss, acc_key, acc_exact, u_start)), grads = eqx.filter_value_and_grad(rep_loss, has_aux = True)(
        ts.rep, ts.duals, batch, ts.barrier, ts.inv_weight, ts.separate_constraints)
    updates, opt_state = ts.rep_optimizer.update(grads, ts.rep_opt_state, eqx.filter(ts.rep, eqx.is_array))
    rep = eqx.apply_updates(ts.rep, updates)
    if ts.nap:
        rep = nap_project(rep, ts.nap_norms[0])
    duals = update_duals(ts.duals, aux.errors, ts.dual_lr, ts.dual_clip)
    metrics = {
        "rep_loss": total, "graph_loss": aux.graph_loss, "dual_loss": aux.dual_loss, "barrier_loss": aux.barrier_loss,
        "orth_error": jnp.linalg.norm(aux.errors), "graph_norms": aux.graph_norms, "inner_diag": jnp.diagonal(aux.inner),
        "u_std": jnp.std(u_start, axis = 0), "duals_diag": jnp.diagonal(duals),
        "inv_loss": inv_loss, "inv_acc_key": acc_key, "inv_acc_exact": acc_exact,
        "rep_grad_norm": optax.tree.norm(grads), "delta_mean": jnp.mean(batch["r_delta"].astype(jnp.float32)),
    }
    return rep, opt_state, duals, metrics


def train_step(train_state: TrainState, batch: Dict[str, Any],
               pool: Optional[ThreadPoolExecutor] = None) -> Tuple[TrainState, StepMetrics]:
    """
    One update of the main, skill and representation networks (+ dual ascent), each
    from its own replay batch and each reading only the pre-update state (the skill
    rewards use the representation before this step's update, as DCEO's
    sequential code effectively does too). The three jitted sub-steps are
    independent, so with ``pool`` they run concurrently (XLA:CPU parallelises
    small convolutions poorly; three threads cut the update time by ~40%).
    """
    ts = train_state
    parts = ((main_step, _subset(batch, "m_")), (skill_step, _subset(batch, "s_")), (rep_step, _subset(batch, "r_")))
    if pool is None:
        results = [fn(ts, b) for fn, b in parts]
    else:
        # block inside each worker so the three computations really overlap
        results = [f.result() for f in [pool.submit(lambda fn, b: jax.block_until_ready(fn(ts, b)), fn, b) for fn, b in parts]]
    (main, main_opt_state, m_main), (skill, skill_opt_state, m_skill), (rep, rep_opt_state, duals, m_rep) = results
    metrics = StepMetrics(**m_main, **m_skill, **m_rep)
    new_state = dataclasses.replace(
        ts, rep = rep, rep_opt_state = rep_opt_state, duals = duals,
        skill = skill, skill_opt_state = skill_opt_state, main = main, main_opt_state = main_opt_state,
        rng = jr.split(ts.rng)[0], step = ts.step + 1,
    )
    return new_state, metrics


def sync_targets(train_state: TrainState) -> TrainState:
    return dataclasses.replace(train_state, main_target = train_state.main, skill_target = train_state.skill)


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------

@eqx.filter_jit
def greedy_main_action(main: QNetwork, obs: jax.Array) -> jax.Array:
    return jnp.argmax(main(obs)[0])


@eqx.filter_jit
def greedy_option_action(skill: QNetwork, obs: jax.Array, option: jax.Array) -> jax.Array:
    return jnp.argmax(skill(obs)[option])


def is_ready(info: Dict[str, Any]) -> bool:
    return bool(info.get("ready", True)) and bool(info.get("in_game", True))


def compute_cut(info: Dict[str, Any], next_info: Dict[str, Any], mask_death: bool, mask_not_ready: bool) -> bool:
    """Is the frame of ``next_info`` NOT a continuation of the frame of ``info`` (for representation purposes)?"""
    if mask_death and bool(info.get("player_dead", False)):
        return True
    if mask_not_ready and not (is_ready(info) and is_ready(next_info)):
        return True
    return False


def sample_batch(replay: FrameReplay, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    batch: Dict[str, np.ndarray] = {}
    for prefix, part in (
        ("m_", replay.sample_task(args.batch_size, args.n_step, args.gamma, args.death_terminal)),
        ("s_", replay.sample_option(args.batch_size, args.n_step, args.gamma)),
        ("r_", replay.sample_pairs(args.rep_batch, args.gamma_rep, args.max_delta, args.separate_constraint_batches)),
    ):
        batch.update({prefix + k: v for k, v in part.items()})
    batch.pop("s_steps")
    return batch


# ---------------------------------------------------------------------------
# Logging / checkpoints
# ---------------------------------------------------------------------------

def summarize_updates(step_metrics: List[StepMetrics]) -> Dict[str, float]:
    if not step_metrics:
        return {}
    m = jax.tree.map(lambda *xs: np.mean(np.stack([np.asarray(x) for x in xs]), axis = 0), *step_metrics)
    eig = np.asarray(eigenvalue_estimates(jnp.diag(jnp.asarray(m.duals_diag))))
    out = {
        "dceo/main_loss": m.main_loss, "dceo/main_q": m.main_q, "dceo/main_target": m.main_target,
        "dceo/main_grad_norm": m.main_grad_norm,
        "dceo/skill_loss": m.skill_loss, "dceo/skill_q": m.skill_q, "dceo/skill_grad_norm": m.skill_grad_norm,
        "dceo/skill_bootstrap_frac": m.skill_bootstrap_frac,
        "intrinsic/abs_mean": m.intr_abs_mean, "intrinsic/mean": np.mean(m.intr_mean), "intrinsic/std": np.mean(m.intr_std),
        "rep/total_loss": m.rep_loss, "rep/graph_loss": m.graph_loss, "rep/dual_loss": m.dual_loss,
        "rep/barrier_loss": m.barrier_loss, "rep/orth_error": m.orth_error,
        "rep/inv_loss": m.inv_loss, "rep/inv_acc_key": m.inv_acc_key, "rep/inv_acc_exact": m.inv_acc_exact,
        "rep/grad_norm": m.rep_grad_norm, "rep/delta_mean": m.delta_mean,
    }
    for k, (mu, sd) in enumerate(zip(np.atleast_1d(m.intr_mean), np.atleast_1d(m.intr_std))):
        out[f"intrinsic/option_{k}_mean"] = mu
        out[f"intrinsic/option_{k}_std"] = sd
    for i in range(len(m.graph_norms)):
        out[f"rep/graph_norm_{i}"] = m.graph_norms[i]
        out[f"rep/eig_est_{i}"] = eig[i]
        out[f"rep/dual_{i}"] = m.duals_diag[i]
        out[f"rep/sq_norm_{i}"] = m.inner_diag[i]
        out[f"rep/u_std_{i}"] = m.u_std[i]
    return {k: float(v) for k, v in out.items()}


def summarize_exploration(explorer: DCEOExplorer, epsilon: float) -> Dict[str, float]:
    stats = explorer.pop_stats()
    total = max(1, int(stats.steps.sum()))
    durations = stats.durations
    return {
        "explore/epsilon": float(epsilon),
        "explore/frac_option": stats.steps[OPTION] / total,
        "explore/frac_random": stats.steps[RANDOM] / total,
        "explore/frac_greedy": stats.steps[GREEDY] / total,
        "explore/options_launched": float(sum(stats.launches.values())),
        "explore/options_finished": float(len(durations)),
        "explore/option_mean_duration": float(np.mean(durations)) if durations else 0.0,
        "explore/distinct_options": float(len(stats.launches)),
    }


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "frame_width": args.frame_width, "frame_height": args.frame_height, "frame_stack": args.frame_stack,
        "grayscale": True, "ticks_per_step": args.ticks_per_step,
        "d": args.d, "option_directions": args.option_directions, "hidden": args.hidden,
        "layer_norm": args.layer_norm, "symlog": args.symlog, "dueling": args.dueling,
        "num_joint_actions": NUM_JOINT_ACTIONS, "key_names": list(KEY_NAMES),
        "mask_death": args.mask_death, "mask_not_ready": args.mask_not_ready,
    }


def save_checkpoint(train_state: TrainState, ckpt_dir: Path, step: int, config: Dict[str, Any]) -> Path:
    ckpt_dir.mkdir(parents = True, exist_ok = True)
    config_path = ckpt_dir / MODEL_CONFIG_FILENAME
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent = 2), encoding = "utf-8")
    models = DCEOModels(rep = train_state.rep, skill = train_state.skill, main = train_state.main, duals = train_state.duals)
    path = ckpt_dir / f"model_{step:09d}.eqx"
    eqx.tree_serialise_leaves(path, models)
    shutil.copyfile(path, ckpt_dir / "latest.eqx")
    return path


def load_checkpoint(path: Path, config: Optional[Dict[str, Any]] = None) -> Tuple[DCEOModels, Dict[str, Any]]:
    path = Path(path)
    if config is None:
        config = json.loads((path.parent / MODEL_CONFIG_FILENAME).read_text(encoding = "utf-8"))
    ns = argparse.Namespace(
        frame_stack = config["frame_stack"], frame_height = config["frame_height"], frame_width = config["frame_width"],
        d = config["d"], option_directions = config["option_directions"], hidden = config["hidden"],
        layer_norm = config["layer_norm"], symlog = config["symlog"], dueling = config["dueling"],
    )
    rep, skill, main = build_networks(ns, jr.PRNGKey(0))
    skeleton = DCEOModels(rep = rep, skill = skill, main = main, duals = jnp.zeros((config["d"], config["d"]), jnp.float32))
    return eqx.tree_deserialise_leaves(path, skeleton), config


def warm_up(train_state: TrainState, args: argparse.Namespace) -> None:
    """Compile the update and the acting functions on dummy data (before the game is launched)."""
    frame_shape = (args.frame_height, args.frame_width)
    n = max(args.n_step + args.frame_stack + 8, 64)
    replay = FrameReplay(n + 1, frame_shape, args.frame_stack, seed = 0)
    rng = np.random.default_rng(0)
    for i in range(n):
        replay.add(rng.integers(0, 256, frame_shape, dtype = np.uint8), int(rng.integers(NUM_JOINT_ACTIONS)), 0.0,
                   False, i % 17 == 16)
    with ThreadPoolExecutor(3) as pool:
        _, m = train_step(train_state, sample_batch(replay, args), pool if args.parallel_updates else None)
    jax.block_until_ready(m.rep_loss)
    obs = jnp.asarray(replay.stack(np.array([n - 1]))[0])
    jax.block_until_ready(greedy_main_action(train_state.main, obs))
    jax.block_until_ready(greedy_option_action(train_state.skill, obs, jnp.asarray(0, dtype = jnp.int32)))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class Learner:
    """
    Owns the ``TrainState`` and runs ``train_step``: synchronously, or (``asynchronous``)
    one update ahead in a background thread so the update overlaps the next
    env steps. In async mode the agent acts with parameters that are at most one
    update stale (``1 / updates_per_step`` env steps); everything else (batches,
    replay ratio, target syncs) is unchanged.
    """

    def __init__(self, train_state: TrainState, parallel: bool, asynchronous: bool):
        self.state = train_state
        self.pool = ThreadPoolExecutor(3, thread_name_prefix = "dceo-update") if parallel else None
        self.background = ThreadPoolExecutor(1, thread_name_prefix = "dceo-learner") if asynchronous else None
        self.inflight = None
        self.started = 0
        self.finished: List[StepMetrics] = []
        self.compute_seconds = 0.0
        self.wait_seconds = 0.0

    def _run(self, state: TrainState, batch: Dict[str, np.ndarray]):
        t = time.perf_counter()
        batch = jax.device_put(batch)   # one host -> device transfer per update (replay stays in host memory)
        new_state, metrics = train_step(state, batch, self.pool)
        jax.block_until_ready((eqx.filter(new_state, eqx.is_array), metrics))
        return new_state, metrics, time.perf_counter() - t

    def _finish(self, result) -> None:
        self.state, metrics, seconds = result
        self.finished.append(metrics)
        self.compute_seconds += seconds

    def collect(self) -> TrainState:
        """Wait for the in-flight update (if any) and adopt its result."""
        if self.inflight is not None:
            t = time.perf_counter()
            result = self.inflight.result()
            self.wait_seconds += time.perf_counter() - t
            self.inflight = None
            self._finish(result)
        return self.state

    def submit(self, batch: Dict[str, np.ndarray]) -> None:
        self.collect()
        if self.started == 0:
            self.state = sync_targets(self.state)
        self.started += 1
        if self.background is None:
            t = time.perf_counter()
            result = self._run(self.state, batch)
            self.wait_seconds += time.perf_counter() - t
            self._finish(result)
        else:
            self.inflight = self.background.submit(self._run, self.state, batch)

    def sync_targets(self) -> None:
        self.collect()
        self.state = sync_targets(self.state)

    def pop_window(self) -> Tuple[List[StepMetrics], float, float]:
        out = (self.finished, self.compute_seconds, self.wait_seconds)
        self.finished, self.compute_seconds, self.wait_seconds = [], 0.0, 0.0
        return out

    def close(self) -> TrainState:
        try:
            self.collect()
        finally:
            for executor in (self.background, self.pool):
                if executor is not None:
                    executor.shutdown(wait = True)
        return self.state


def train(train_state: TrainState, env, args: argparse.Namespace, ckpt_dir: Path) -> TrainState:
    metrics = RolloutMetrics()
    obs, info = env.reset(options = {"wipe": not args.no_wipe})
    metrics.start(info)
    frame = np.asarray(obs)[-1]
    replay = FrameReplay(args.replay_capacity, frame.shape, args.frame_stack, seed = args.seed + 1)
    stacker = StackBuilder(args.frame_stack)
    stacker.reset(frame)
    explorer = DCEOExplorer(len(train_state.option_dims), args.mu, args.p_term, seed = args.seed + 2)
    act_rng = np.random.default_rng(args.seed + 3)
    model_config = model_config_from_args(args)
    learner = Learner(train_state, args.parallel_updates, args.async_updates)

    env_steps = 0
    next_log = args.log_interval
    next_ckpt = args.ckpt_interval
    last_logged: Dict[str, float] = {}
    env_time = 0.0
    update_credit = 0.0
    epsilon = 1.0
    t_start = time.perf_counter()
    deadline = t_start + args.seconds if args.seconds else None
    pbar = tqdm(total = args.total_steps, unit = "step", dynamic_ncols = True, mininterval = 1.0)

    def log_window() -> Dict[str, float]:
        nonlocal last_logged, env_time
        step_metrics, compute_s, wait_s = learner.pop_window()
        out = dict(metrics.summary())
        out.update(summarize_updates(step_metrics))
        out.update(summarize_exploration(explorer, epsilon))
        out.update({
            "time/env_seconds": env_time,
            "time/update_wait_seconds": wait_s,
            "time/update_compute_seconds": compute_s,
            "time/updates": float(len(step_metrics)),
            "time/update_seconds_mean": compute_s / max(1, len(step_metrics)),
            "time/env_steps_per_second": metrics.window_steps / max(1e-9, env_time),
            "time/elapsed_seconds": time.perf_counter() - t_start,
            "replay/size": float(replay.size), "dceo/updates_total": float(learner.started),
        })
        mlflow.log_metrics(out, step = env_steps)
        last_logged = out
        metrics.end_window()
        env_time = 0.0
        return out

    try:
        while env_steps < args.total_steps and (deadline is None or time.perf_counter() < deadline):
            t0 = time.perf_counter()
            ts = learner.state
            epsilon = linearly_decaying_epsilon(args.epsilon_decay_steps, env_steps, args.min_replay, args.epsilon_final)
            decision = explorer.decide(epsilon)
            state = stacker.get()
            if decision.kind == OPTION:
                if args.option_epsilon > 0 and act_rng.random() < args.option_epsilon:
                    action = int(act_rng.integers(NUM_JOINT_ACTIONS))
                else:
                    action = int(greedy_option_action(ts.skill, jnp.asarray(state), jnp.asarray(decision.option, dtype = jnp.int32)))
            elif decision.kind == RANDOM:
                action = int(act_rng.integers(NUM_JOINT_ACTIONS))
            else:
                action = int(greedy_main_action(ts.main, jnp.asarray(state)))

            next_obs, reward, _terminated, _truncated, next_info = env.step(ACTION_BITS[action])
            metrics.update(next_info, float(reward))
            cut = compute_cut(info, next_info, args.mask_death, args.mask_not_ready)
            died = bool(next_info.get("player_dead", False))
            replay.add(frame, action, float(reward) * args.reward_scale, died, cut)
            frame = np.asarray(next_obs)[-1]
            stacker.push(frame, cut)
            if cut or (args.mask_death and died):
                explorer.terminate()   # an option does not survive a trajectory boundary
            info = next_info
            env_steps += 1
            env_time += time.perf_counter() - t0

            learning = replay.size >= args.min_replay
            if learning:
                update_credit += args.updates_per_step
                while update_credit >= 1.0 - 1e-9:
                    learner.submit(sample_batch(replay, args))
                    update_credit -= 1.0
            if learning and env_steps % args.target_update_period == 0:
                learner.sync_targets()

            if env_steps % 100 == 0:
                pbar.update(100)
            if env_steps >= next_log:
                out = log_window()
                while next_log <= env_steps:
                    next_log += args.log_interval
                pbar.set_postfix(
                    rooms = int(out["env/rooms_discovered"]), deaths = int(out["env/deaths"]),
                    eps = f"{epsilon:.2f}", opt = f"{out['explore/frac_option']:.2f}",
                    sps = f"{out['env/steps_per_second']:.0f}",
                    graph = f"{out.get('rep/graph_loss', float('nan')):.3f}",
                )
            if env_steps >= next_ckpt:
                save_checkpoint(learner.collect(), ckpt_dir, env_steps, model_config)
                while next_ckpt <= env_steps:
                    next_ckpt += args.ckpt_interval
    except KeyboardInterrupt:
        logger.warning("interrupted at %d env steps; saving checkpoint", env_steps)
    finally:
        pbar.close()
        train_state = learner.close()
        if metrics.window_steps > 0:
            log_window()   # the last, partial window (e.g. when --seconds runs out)
        path = save_checkpoint(train_state, ckpt_dir, env_steps, model_config)
        logger.info("final checkpoint: %s (%d env steps, %d updates, %.0fs)", path, env_steps, int(train_state.step),
                    time.perf_counter() - t_start)
        if last_logged:
            print("\nlast logged metrics:\n" + format_summary(last_logged))
    return train_state


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("rainworld_rl imported from %s", rainworld_rl.__file__)
    logger.info("jax backend: %s, devices: %s", jax.default_backend(), jax.devices())
    setup_mlflow(args.mlflow_uri, args.experiment)

    train_state = init_experiment(args)
    counts = {name: sum(int(np.prod(a.shape)) for a in jax.tree.leaves(eqx.filter(getattr(train_state, name), eqx.is_array)))
              for name in ("rep", "skill", "main")}
    logger.info("parameters: %s; options K = %d", counts, len(train_state.option_dims))
    logger.info("warming up JIT (before taking the game lock)...")
    tw = time.perf_counter()
    warm_up(train_state, args)
    logger.info("warm-up took %.1fs", time.perf_counter() - tw)

    with game_lock(args.game_lock):
        env = make_env(
            frame_width = args.frame_width, frame_height = args.frame_height, ticks_per_step = args.ticks_per_step,
            frame_stack = 1, grayscale = True, fake = args.fake, seed = args.seed, novelty = args.novelty,
        )
        mlflow.start_run(run_name = args.run_name)
        run_id = mlflow.active_run().info.run_id
        ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else DEFAULT_CKPT_ROOT / run_id
        mlflow.log_params({**vars(args), "run_id": run_id, "ckpt_dir": str(ckpt_dir),
                           "rainworld_rl_path": rainworld_rl.__file__, "num_options": len(train_state.option_dims),
                           **{f"param_count_{k}": v for k, v in counts.items()}, "jax_backend": jax.default_backend(), "device": str(jax.devices()[0])})
        logger.info("run %s, checkpoints -> %s", run_id, ckpt_dir)
        try:
            attach(env, launch = args.launch)
            train(train_state, env, args, ckpt_dir)
        finally:
            mlflow.end_run()
            try:
                env.close()
            finally:
                if args.kill_game and not args.fake:
                    from rainworld_rl.launcher import kill_game
                    kill_game()
    return 0


if __name__ == "__main__":
    sys.exit(main())
