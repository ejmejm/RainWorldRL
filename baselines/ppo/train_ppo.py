"""
PPO baseline for the Rain World RL environment.

Continuing environment: there are no episode boundaries. Rollouts of
``--rollout_steps`` steps are collected in a Python loop (the env is a Python
object) with a jitted policy; GAE is bootstrapped from the value of the final
observation (no dones, ``player_dead`` is *not* terminal). The PPO update over
``--epochs`` x ``--minibatches`` is a jitted ``jax.lax.scan`` whose ``xs`` are
the minibatch indices.

Policy: Nature-CNN-ish conv stack on the stacked uint8 frames (``/255`` inside
the model) -> 256 hidden -> independent Bernoulli logits for the 9 keys
(``MultiBinary(9)``) + a value head. Log-prob and entropy are sums over keys.

Run from the repository root (so the worktree's ``rainworld_rl`` is imported)::

    python -m baselines.ppo.train_ppo --fake --total_steps 2048          # no game
    python -m baselines.ppo.train_ppo --launch --kill_game --total_steps 200000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mlflow
import numpy as np
import optax
from tqdm import tqdm

import rainworld_rl
from rainworld_rl.shared_memory import KEY_NAMES, NUM_KEYS

from baselines.common.env import attach, make_env, obs_shape_for
from baselines.common.game_lock import game_lock
from baselines.common.metrics import RolloutMetrics, format_summary

logger = logging.getLogger("baselines.ppo")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_ROOT = Path(__file__).resolve().parent / "checkpoints"
MLRUNS_DIR = REPO_ROOT / "mlruns"
# MLflow >= 3.16 rejects the plain ./mlruns file store, so the default is a SQLite store kept in the same folder.
DEFAULT_MLFLOW_URI = "sqlite:///" + (MLRUNS_DIR / "mlflow.db").as_posix()
MODEL_CONFIG_FILENAME = "model_config.json"


def setup_mlflow(uri: str, experiment: str) -> None:
    """Point MLflow at ``uri`` (creating the folder of a local SQLite store) and select ``experiment``."""
    if uri.startswith("sqlite:///"):
        Path(uri[len("sqlite:///"):]).parent.mkdir(parents = True, exist_ok = True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog = "python -m baselines.ppo.train_ppo",
        description = "PPO baseline for Rain World RL (continuing env, MultiBinary keys).",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    # env
    p.add_argument("--fake", action = "store_true", help = "Use FakeRainWorldEnv (no game)")
    p.add_argument("--launch", action = "store_true", help = "(Re)start Rain World with the deployed DLL before training")
    p.add_argument("--kill_game", action = "store_true", help = "Close the game when training ends")
    p.add_argument("--no_wipe", action = "store_true", help = "Attach to the game in progress instead of wiping the RL save")
    p.add_argument("--game_lock", type = str, default = None, help = "mkdir-lock directory to hold while the game is in use")
    p.add_argument("--frame_width", type = int, default = 96)
    p.add_argument("--frame_height", type = int, default = 54)
    p.add_argument("--ticks_per_step", type = int, default = 4)
    p.add_argument("--frame_stack", type = int, default = 4)
    p.add_argument("--rgb", action = "store_true", help = "Keep RGB frames instead of grayscale")
    p.add_argument("--reward_scale", type = float, default = 1.0)
    # ppo
    p.add_argument("--total_steps", type = int, default = 200_000)
    p.add_argument("--rollout_steps", type = int, default = 1024)
    p.add_argument("--epochs", type = int, default = 4)
    p.add_argument("--minibatches", type = int, default = 4)
    p.add_argument("--gamma", type = float, default = 0.99)
    p.add_argument("--gae_lambda", type = float, default = 0.95)
    p.add_argument("--clip", type = float, default = 0.2)
    p.add_argument("--clip_vf", type = float, default = 0.0, help = "Value clipping range; 0 disables")
    p.add_argument("--lr", type = float, default = 2.5e-4)
    p.add_argument("--ent_coef", type = float, default = 0.01)
    p.add_argument("--vf_coef", type = float, default = 0.5)
    p.add_argument("--max_grad_norm", type = float, default = 0.5)
    p.add_argument("--hidden", type = int, default = 256)
    # bookkeeping
    p.add_argument("--seed", type = int, default = None, help = "Random if omitted")
    p.add_argument("--log_interval", type = int, default = 1024, help = "Env steps between MLflow logs")
    p.add_argument("--ckpt_interval", type = int, default = 20_000, help = "Env steps between checkpoints")
    p.add_argument("--ckpt_dir", type = str, default = None, help = "Checkpoint directory (default baselines/ppo/checkpoints/<run_id>)")
    p.add_argument("--mlflow_uri", type = str, default = DEFAULT_MLFLOW_URI)
    p.add_argument("--experiment", type = str, default = "rainworld-ppo")
    p.add_argument("--run_name", type = str, default = None)
    args = p.parse_args(argv)
    if args.seed is None:
        args.seed = random.randrange(2**31)
    if args.rollout_steps % args.minibatches != 0:
        p.error("--rollout_steps must be divisible by --minibatches")
    if args.total_steps < args.rollout_steps:
        p.error("--total_steps must be >= --rollout_steps")
    return args


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _orthogonal(layer, gain: float, key: jax.Array):
    """Orthogonal weight init + zero bias for a Conv2d/Linear layer."""
    w = jax.nn.initializers.orthogonal(gain)(key, layer.weight.shape, jnp.float32)
    layer = eqx.tree_at(lambda l: l.weight, layer, w)
    if layer.bias is not None:
        layer = eqx.tree_at(lambda l: l.bias, layer, jnp.zeros_like(layer.bias))
    return layer


class ActorCritic(eqx.Module):
    """Nature-CNN-ish torso -> hidden -> (Bernoulli logits per key, value)."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    fc: eqx.nn.Linear
    policy: eqx.nn.Linear
    value: eqx.nn.Linear

    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int = NUM_KEYS, hidden: int = 256, *, key: jax.Array):
        c, h, w = obs_shape
        k = jr.split(key, 12)
        conv1 = eqx.nn.Conv2d(c, 32, 8, 4, key = k[0])
        conv2 = eqx.nn.Conv2d(32, 64, 4, 2, key = k[1])
        conv3 = eqx.nn.Conv2d(64, 64, 3, 1, key = k[2])
        self.conv1 = _orthogonal(conv1, jnp.sqrt(2.0), k[6])
        self.conv2 = _orthogonal(conv2, jnp.sqrt(2.0), k[7])
        self.conv3 = _orthogonal(conv3, jnp.sqrt(2.0), k[8])

        def torso(x):
            x = jax.nn.relu(conv1(x))
            x = jax.nn.relu(conv2(x))
            x = jax.nn.relu(conv3(x))
            return x.reshape(-1)

        flat = jax.eval_shape(torso, jax.ShapeDtypeStruct((c, h, w), jnp.float32)).shape[0]
        self.fc = _orthogonal(eqx.nn.Linear(flat, hidden, key = k[3]), jnp.sqrt(2.0), k[9])
        self.policy = _orthogonal(eqx.nn.Linear(hidden, num_actions, key = k[4]), 0.01, k[10])
        self.value = _orthogonal(eqx.nn.Linear(hidden, 1, key = k[5]), 1.0, k[11])

    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """``obs``: uint8 ``(C, H, W)`` -> (logits ``(A,)``, value scalar)."""
        x = obs.astype(jnp.float32) / 255.0
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = jax.nn.relu(self.fc(x.reshape(-1)))
        return self.policy(x), self.value(x)[0]


def bernoulli_log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    """Sum over keys of log Bernoulli(sigmoid(logit)) at ``actions`` in {0, 1}."""
    return jnp.sum(actions * jax.nn.log_sigmoid(logits) + (1.0 - actions) * jax.nn.log_sigmoid(-logits), axis = -1)


def bernoulli_entropy(logits: jax.Array) -> jax.Array:
    """Sum over keys of the Bernoulli entropy."""
    p = jax.nn.sigmoid(logits)
    return jnp.sum(p * jax.nn.softplus(-logits) + (1.0 - p) * jax.nn.softplus(logits), axis = -1)


@eqx.filter_jit
def sample_action(model: ActorCritic, obs: jax.Array, key: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Sample keys from the Bernoulli policy for one observation -> (action, log_prob, value)."""
    logits, value = model(obs)
    action = jr.bernoulli(key, jax.nn.sigmoid(logits)).astype(jnp.float32)
    return action, bernoulli_log_prob(logits, action), value


@eqx.filter_jit
def greedy_action(model: ActorCritic, obs: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Press every key whose probability exceeds 0.5 -> (action, log_prob, value)."""
    logits, value = model(obs)
    action = (logits > 0).astype(jnp.float32)
    return action, bernoulli_log_prob(logits, action), value


@eqx.filter_jit
def value_of(model: ActorCritic, obs: jax.Array) -> jax.Array:
    return model(obs)[1]


# ---------------------------------------------------------------------------
# State / metrics
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    model: ActorCritic
    optimizer_state: optax.OptState
    rng: jax.Array
    step: jax.Array  # env steps consumed by updates so far (int32)

    optimizer: optax.GradientTransformation = eqx.field(static = True)
    epochs: int = eqx.field(static = True)
    minibatches: int = eqx.field(static = True)
    gamma: float = eqx.field(static = True)
    gae_lambda: float = eqx.field(static = True)
    clip: float = eqx.field(static = True)
    clip_vf: float = eqx.field(static = True)
    vf_coef: float = eqx.field(static = True)
    ent_coef: float = eqx.field(static = True)
    log_interval: int = eqx.field(static = True)
    ckpt_interval: int = eqx.field(static = True)


class StepMetrics(eqx.Module):
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clipfrac: jax.Array
    grad_norm: jax.Array
    explained_variance: jax.Array
    total_loss: jax.Array


def init_experiment(args: argparse.Namespace) -> TrainState:
    key = jr.PRNGKey(args.seed)
    model_key, rng = jr.split(key)
    obs_shape = obs_shape_for(args.frame_width, args.frame_height, args.frame_stack, not args.rgb)
    model = ActorCritic(obs_shape, NUM_KEYS, args.hidden, key = model_key)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.lr, eps = 1e-5),
    )
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return TrainState(
        model = model,
        optimizer_state = optimizer_state,
        rng = rng,
        step = jnp.asarray(0, dtype = jnp.int32),
        optimizer = optimizer,
        epochs = args.epochs,
        minibatches = args.minibatches,
        gamma = args.gamma,
        gae_lambda = args.gae_lambda,
        clip = args.clip,
        clip_vf = args.clip_vf,
        vf_coef = args.vf_coef,
        ent_coef = args.ent_coef,
        log_interval = args.log_interval,
        ckpt_interval = args.ckpt_interval,
    )


# ---------------------------------------------------------------------------
# PPO maths
# ---------------------------------------------------------------------------

def compute_gae(rewards: jax.Array, values: jax.Array, last_value: jax.Array, gamma: float, lam: float) -> Tuple[jax.Array, jax.Array]:
    """
    GAE for a continuing environment: no dones, bootstrap from ``last_value``
    (value of the observation after the final step). Returns (advantages, returns).
    """
    next_values = jnp.concatenate([values[1:], last_value[None]])
    deltas = rewards + gamma * next_values - values

    def scan_fn(next_adv, delta):
        adv = delta + gamma * lam * next_adv
        return adv, adv

    _, advantages = jax.lax.scan(scan_fn, jnp.zeros_like(last_value), deltas, reverse = True)
    return advantages, advantages + values


def ppo_loss(model: ActorCritic, mb: Dict[str, jax.Array], clip: float, clip_vf: float, vf_coef: float, ent_coef: float):
    logits, values = jax.vmap(model)(mb["obs"])
    log_probs = bernoulli_log_prob(logits, mb["actions"])
    entropy = bernoulli_entropy(logits).mean()

    log_ratio = log_probs - mb["log_probs"]
    ratio = jnp.exp(log_ratio)
    adv = mb["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    policy_loss = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * adv).mean()

    returns = mb["returns"]
    if clip_vf > 0:
        v_clipped = mb["values"] + jnp.clip(values - mb["values"], -clip_vf, clip_vf)
        value_loss = 0.5 * jnp.maximum((values - returns) ** 2, (v_clipped - returns) ** 2).mean()
    else:
        value_loss = 0.5 * ((values - returns) ** 2).mean()

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > clip).astype(jnp.float32))
    return loss, (policy_loss, value_loss, entropy, approx_kl, clipfrac)


@eqx.filter_jit
def train_step(train_state: TrainState, batch: Dict[str, jax.Array]) -> Tuple[TrainState, StepMetrics]:
    """One PPO update: GAE, then ``epochs`` x ``minibatches`` gradient steps via ``lax.scan``."""
    rng, perm_key = jr.split(train_state.rng)
    num_steps = batch["rewards"].shape[0]
    mb_size = num_steps // train_state.minibatches

    advantages, returns = compute_gae(
        batch["rewards"], batch["values"], batch["last_value"], train_state.gamma, train_state.gae_lambda
    )
    explained_variance = 1.0 - jnp.var(returns - batch["values"]) / (jnp.var(returns) + 1e-8)

    perms = jax.vmap(lambda k: jr.permutation(k, num_steps))(jr.split(perm_key, train_state.epochs))
    minibatch_indices = perms.reshape(train_state.epochs * train_state.minibatches, mb_size)

    params, static = eqx.partition(train_state.model, eqx.is_array)
    optimizer = train_state.optimizer
    loss_fn = eqx.filter_value_and_grad(ppo_loss, has_aux = True)

    def minibatch_step(carry, idx):
        params, optimizer_state = carry
        model = eqx.combine(params, static)
        mb = {
            "obs": batch["obs"][idx],
            "actions": batch["actions"][idx],
            "log_probs": batch["log_probs"][idx],
            "values": batch["values"][idx],
            "advantages": advantages[idx],
            "returns": returns[idx],
        }
        (loss, aux), grads = loss_fn(
            model, mb, train_state.clip, train_state.clip_vf, train_state.vf_coef, train_state.ent_coef
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = eqx.apply_updates(params, updates)
        policy_loss, value_loss, entropy, approx_kl, clipfrac = aux
        metrics = StepMetrics(
            policy_loss = policy_loss, value_loss = value_loss, entropy = entropy, approx_kl = approx_kl,
            clipfrac = clipfrac, grad_norm = optax.global_norm(grads), explained_variance = explained_variance,
            total_loss = loss,
        )
        return (params, optimizer_state), metrics

    (params, optimizer_state), metrics = jax.lax.scan(
        minibatch_step, (params, train_state.optimizer_state), minibatch_indices, length = minibatch_indices.shape[0]
    )
    metrics = jax.tree.map(lambda x: x.mean(0), metrics)

    new_state = TrainState(
        model = eqx.combine(params, static),
        optimizer_state = optimizer_state,
        rng = rng,
        step = train_state.step + num_steps,
        optimizer = optimizer,
        epochs = train_state.epochs,
        minibatches = train_state.minibatches,
        gamma = train_state.gamma,
        gae_lambda = train_state.gae_lambda,
        clip = train_state.clip,
        clip_vf = train_state.clip_vf,
        vf_coef = train_state.vf_coef,
        ent_coef = train_state.ent_coef,
        log_interval = train_state.log_interval,
        ckpt_interval = train_state.ckpt_interval,
    )
    return new_state, metrics


# ---------------------------------------------------------------------------
# Rollout collection (Python loop over the env, jitted policy)
# ---------------------------------------------------------------------------

def collect_rollout(
    env, model: ActorCritic, obs: np.ndarray, rng: jax.Array, num_steps: int,
    metrics: RolloutMetrics, reward_scale: float,
) -> Tuple[Dict[str, jax.Array], np.ndarray, jax.Array, Dict[str, Any]]:
    """Step the env ``num_steps`` times; returns (batch, next_obs, rng, last_info)."""
    obs_buf = np.empty((num_steps,) + obs.shape, dtype = np.uint8)
    act_buf = np.empty((num_steps, NUM_KEYS), dtype = np.float32)
    logp_buf = np.empty(num_steps, dtype = np.float32)
    val_buf = np.empty(num_steps, dtype = np.float32)
    rew_buf = np.empty(num_steps, dtype = np.float32)
    info: Dict[str, Any] = {}

    for t in range(num_steps):
        rng, key = jr.split(rng)
        action, log_prob, value = sample_action(model, jnp.asarray(obs), key)
        action_np = np.asarray(action)
        next_obs, reward, _terminated, _truncated, info = env.step(action_np.astype(np.int8))
        metrics.update(info, float(reward))

        obs_buf[t] = obs
        act_buf[t] = action_np
        logp_buf[t] = float(log_prob)
        val_buf[t] = float(value)
        rew_buf[t] = float(reward) * reward_scale
        obs = next_obs

    last_value = value_of(model, jnp.asarray(obs))
    batch = {
        "obs": jnp.asarray(obs_buf),
        "actions": jnp.asarray(act_buf),
        "log_probs": jnp.asarray(logp_buf),
        "values": jnp.asarray(val_buf),
        "rewards": jnp.asarray(rew_buf),
        "last_value": last_value.astype(jnp.float32),
    }
    return batch, obs, rng, info


# ---------------------------------------------------------------------------
# Logging / checkpoints
# ---------------------------------------------------------------------------

def log_metrics(metrics: StepMetrics, env_metrics: Dict[str, float], timing: Dict[str, float], step: int) -> Dict[str, float]:
    out = {
        "ppo/policy_loss": float(metrics.policy_loss),
        "ppo/value_loss": float(metrics.value_loss),
        "ppo/entropy": float(metrics.entropy),
        "ppo/approx_kl": float(metrics.approx_kl),
        "ppo/clipfrac": float(metrics.clipfrac),
        "ppo/grad_norm": float(metrics.grad_norm),
        "ppo/explained_variance": float(metrics.explained_variance),
        "ppo/total_loss": float(metrics.total_loss),
    }
    out.update(env_metrics)
    out.update({f"time/{k}": v for k, v in timing.items()})
    mlflow.log_metrics(out, step = step)
    return out


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "frame_width": args.frame_width,
        "frame_height": args.frame_height,
        "frame_stack": args.frame_stack,
        "grayscale": not args.rgb,
        "ticks_per_step": args.ticks_per_step,
        "hidden": args.hidden,
        "num_actions": NUM_KEYS,
        "key_names": list(KEY_NAMES),
    }


def build_model_from_config(config: Dict[str, Any]) -> ActorCritic:
    obs_shape = obs_shape_for(config["frame_width"], config["frame_height"], config["frame_stack"], config["grayscale"])
    return ActorCritic(obs_shape, config.get("num_actions", NUM_KEYS), config.get("hidden", 256), key = jr.PRNGKey(0))


def save_checkpoint(model: ActorCritic, ckpt_dir: Path, step: int, config: Dict[str, Any]) -> Path:
    ckpt_dir.mkdir(parents = True, exist_ok = True)
    config_path = ckpt_dir / MODEL_CONFIG_FILENAME
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent = 2), encoding = "utf-8")
    path = ckpt_dir / f"model_{step:09d}.eqx"
    eqx.tree_serialise_leaves(path, model)
    shutil.copyfile(path, ckpt_dir / "latest.eqx")
    return path


def load_checkpoint(path: Path, config: Optional[Dict[str, Any]] = None) -> Tuple[ActorCritic, Dict[str, Any]]:
    """Load a serialised model; ``config`` defaults to ``model_config.json`` next to ``path``."""
    path = Path(path)
    if config is None:
        config = json.loads((path.parent / MODEL_CONFIG_FILENAME).read_text(encoding = "utf-8"))
    skeleton = build_model_from_config(config)
    return eqx.tree_deserialise_leaves(path, skeleton), config


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(train_state: TrainState, env, args: argparse.Namespace, ckpt_dir: Path) -> TrainState:
    metrics = RolloutMetrics()
    obs, info = env.reset(options = {"wipe": not args.no_wipe})
    metrics.start(info)
    rollout_rng = jr.PRNGKey(args.seed + 1)
    model_config = model_config_from_args(args)

    env_steps = 0
    next_log = args.log_interval
    next_ckpt = args.ckpt_interval
    last_logged: Dict[str, float] = {}
    pbar = tqdm(total = args.total_steps, unit = "step", dynamic_ncols = True)
    try:
        while env_steps < args.total_steps:
            t0 = time.perf_counter()
            batch, obs, rollout_rng, info = collect_rollout(
                env, train_state.model, obs, rollout_rng, args.rollout_steps, metrics, args.reward_scale
            )
            t1 = time.perf_counter()
            train_state, step_metrics = train_step(train_state, batch)
            jax.block_until_ready(step_metrics.total_loss)
            t2 = time.perf_counter()
            env_steps += args.rollout_steps
            pbar.update(args.rollout_steps)

            timing = {
                "rollout_seconds": t1 - t0,
                "update_seconds": t2 - t1,
                "env_steps_per_second": args.rollout_steps / max(1e-9, t1 - t0),
            }
            if env_steps >= next_log or env_steps >= args.total_steps:
                env_summary = metrics.summary()
                last_logged = log_metrics(step_metrics, env_summary, timing, env_steps)
                metrics.end_window()
                while next_log <= env_steps:
                    next_log += args.log_interval
                pbar.set_postfix(
                    r = f"{env_summary['env/reward_mean']:.4f}",
                    rooms = int(env_summary["env/rooms_discovered"]),
                    deaths = int(env_summary["env/deaths"]),
                    sps = f"{timing['env_steps_per_second']:.0f}",
                    upd = f"{timing['update_seconds']:.1f}s",
                )
            if env_steps >= next_ckpt:
                save_checkpoint(train_state.model, ckpt_dir, env_steps, model_config)
                while next_ckpt <= env_steps:
                    next_ckpt += args.ckpt_interval
    except KeyboardInterrupt:
        logger.warning("interrupted at %d env steps; saving checkpoint", env_steps)
    finally:
        pbar.close()
        path = save_checkpoint(train_state.model, ckpt_dir, env_steps, model_config)
        logger.info("final checkpoint: %s", path)
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
    param_count = sum(int(np.prod(a.shape)) for a in jax.tree.leaves(eqx.filter(train_state.model, eqx.is_array)))
    logger.info("model parameters: %d", param_count)

    with game_lock(args.game_lock):
        env = make_env(
            frame_width = args.frame_width, frame_height = args.frame_height, ticks_per_step = args.ticks_per_step,
            frame_stack = args.frame_stack, grayscale = not args.rgb, fake = args.fake, seed = args.seed,
        )
        mlflow.start_run(run_name = args.run_name)
        run_id = mlflow.active_run().info.run_id
        ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else DEFAULT_CKPT_ROOT / run_id
        mlflow.log_params({**vars(args), "run_id": run_id, "ckpt_dir": str(ckpt_dir),
                           "rainworld_rl_path": rainworld_rl.__file__, "param_count": param_count,
                           "jax_backend": jax.default_backend()})
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
