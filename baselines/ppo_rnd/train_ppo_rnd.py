"""
PPO + RND (Random Network Distillation; Burda, Edwards, Storkey, Klimov,
ICLR 2019, arXiv:1810.12894) for the Rain World RL environment.

A faithful single-env port of CleanRL ``cleanrl/ppo_rnd_envpool.py``
(vwxyzjn/cleanrl master @ fe8d8a03; file last changed in 35896b1) to the
house JAX / Equinox / Optax / MLflow stack. See ``baselines/ppo_rnd/README.md``
for the recipe and every deviation from the reference.

Per rollout of ``--rollout_steps`` env steps (Python loop, jitted policy):

1. RND bonus for every transition, on the single latest frame of s_{t+1},
   whitened with the per-pixel running stats from *before* this rollout:
   ``sum((target - predictor)^2) / 2``.
2. Run the ``RewardForwardFilter`` (gamma_I) along time, fold the filtered
   returns into a running variance and divide the raw bonus by its std
   (no mean subtraction, no clipping, nothing is ever reset).
3. Two-head GAE: intrinsic is non-episodic (never cut, not even at death);
   extrinsic is continuing like the PPO baseline (``--ext_death_terminal``
   makes ``info["player_dead"]`` a terminal for the extrinsic head only).
   Advantages ``ext_coef * A_E + int_coef * A_I``.
4. Update the per-pixel obs stats with the rollout's latest frames, then one
   jitted update: ``epochs x minibatches`` Adam steps on
   ``pg - ent * H + vf * (L_VE + L_VI) + L_fwd``, the predictor's forward loss
   on a random ``--update_proportion`` of each minibatch.

The per-pixel obs stats are warm-started with ``--obs_norm_init_steps``
random-policy steps before training (CleanRL: 50 x 128 steps per env).

Run from the repository root (so the worktree's ``rainworld_rl`` is imported)::

    python -m baselines.ppo_rnd.train_ppo_rnd --fake --total_steps 4096
    python -m baselines.ppo_rnd.train_ppo_rnd --launch --kill_game --seconds 180 \\
        --game_lock E:/projects/rainworld_rl-wt/.game-lock
"""

from __future__ import annotations

import argparse
import json
import logging
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
from baselines.ppo.train_ppo import DEFAULT_MLFLOW_URI, bernoulli_entropy, bernoulli_log_prob, setup_mlflow
from baselines.ppo_rnd.rnd import (
    RNDPredictor,
    RNDStats,
    RNDTarget,
    conv_flat_dim,
    intrinsic_reward,
    latest_frame,
    nature_convs,
    orthogonal_init,
    whiten,
)

logger = logging.getLogger("baselines.ppo_rnd")

DEFAULT_CKPT_ROOT = Path(__file__).resolve().parent / "checkpoints"
MODEL_CONFIG_FILENAME = "model_config.json"
LATEST_STATS_FILENAME = "latest_rnd_stats.npz"
CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
# ``unroll`` of the epochs x minibatches ``lax.scan`` in ``train_step``. On XLA:CPU an op
# inside a while-loop body runs ~3-4x slower than the same op unrolled, so fully unrolling
# the (16-step) update trades a longer one-off compile for a much faster update.
MINIBATCH_UNROLL = True


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog = "python -m baselines.ppo_rnd.train_ppo_rnd",
        description = "PPO + RND (Burda et al. 2019) for Rain World RL: continuing env, MultiBinary keys, "
                      "two value heads, non-episodic intrinsic return.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    # env
    p.add_argument("--fake", action = "store_true", help = "Use FakeRainWorldEnv (no game)")
    p.add_argument("--launch", action = "store_true", help = "(Re)start Rain World with the deployed DLL before training")
    p.add_argument("--kill_game", action = "store_true", help = "Close the game when training ends")
    p.add_argument("--no_wipe", action = "store_true", help = "Attach to the game in progress instead of wiping the RL save")
    p.add_argument("--game_lock", type = str, default = None, help = "mkdir-lock directory to hold while the game is in use")
    p.add_argument("--novelty", dest = "novelty", action = "store_true",
                   help = "Keep the hand-written NewRoom term in the extrinsic reward")
    p.add_argument("--no_novelty", dest = "novelty", action = "store_false",
                   help = "Drop the NewRoom term (DEFAULT: an intrinsic-motivation method should not get the "
                          "hand-written room bonus)")
    p.set_defaults(novelty = False)
    p.add_argument("--frame_width", type = int, default = 96)
    p.add_argument("--frame_height", type = int, default = 54)
    p.add_argument("--ticks_per_step", type = int, default = 4)
    p.add_argument("--frame_stack", type = int, default = 4, help = "Policy frame stack (RND uses the latest frame only)")
    p.add_argument("--rgb", action = "store_true", help = "Keep RGB frames instead of grayscale")
    p.add_argument("--reward_scale", type = float, default = 1.0, help = "Extrinsic reward multiplier before GAE")
    # budget
    p.add_argument("--total_steps", type = int, default = 200_000,
                   help = "Training env steps (excluding the obs-norm warm-up)")
    p.add_argument("--seconds", type = float, default = 0.0,
                   help = "Wall-clock budget for all env interaction incl. the warm-up (0 = unlimited); "
                          "stops at whichever of --total_steps / --seconds comes first")
    # ppo
    p.add_argument("--rollout_steps", type = int, default = 1024)
    p.add_argument("--epochs", type = int, default = 4)
    p.add_argument("--minibatches", type = int, default = 4)
    p.add_argument("--gamma", type = float, default = 0.99,
                   help = "Extrinsic discount gamma_E (paper/CleanRL 0.999; 0.99 as in the PPO baseline)")
    p.add_argument("--int_gamma", type = float, default = 0.99, help = "Intrinsic discount gamma_I")
    p.add_argument("--gae_lambda", type = float, default = 0.95)
    p.add_argument("--clip", type = float, default = 0.1, help = "PPO ratio clip (also the extrinsic value clip)")
    p.add_argument("--clip_vloss", dest = "clip_vloss", action = "store_true", help = "Clip the extrinsic value loss")
    p.add_argument("--no_clip_vloss", dest = "clip_vloss", action = "store_false")
    p.add_argument("--norm_adv", dest = "norm_adv", action = "store_true", help = "Per-minibatch advantage normalisation")
    p.add_argument("--no_norm_adv", dest = "norm_adv", action = "store_false")
    p.add_argument("--lr", type = float, default = 1e-4, help = "Adam lr for the actor-critic")
    p.add_argument("--anneal_lr", dest = "anneal_lr", action = "store_true",
                   help = "Linearly anneal both lrs to 0 over --total_steps (CleanRL default; off for a lifetime run)")
    p.add_argument("--no_anneal_lr", dest = "anneal_lr", action = "store_false")
    p.set_defaults(clip_vloss = True, norm_adv = True, anneal_lr = False)
    p.add_argument("--ent_coef", type = float, default = 0.001)
    p.add_argument("--vf_coef", type = float, default = 0.5)
    p.add_argument("--max_grad_norm", type = float, default = 0.5, help = "Global grad-norm clip over policy + predictor")
    p.add_argument("--hidden", type = int, default = 256, help = "Actor-critic: first FC layer after the convs")
    p.add_argument("--feature_dim", type = int, default = 448, help = "Actor-critic: shared feature width")
    p.add_argument("--ext_death_terminal", action = "store_true",
                   help = "Treat info['player_dead'] as terminal for the EXTRINSIC head only "
                          "(the intrinsic head is always non-episodic)")
    # rnd
    p.add_argument("--int_coef", type = float, default = 1.0, help = "Weight of the intrinsic advantage")
    p.add_argument("--ext_coef", type = float, default = 2.0, help = "Weight of the extrinsic advantage")
    p.add_argument("--update_proportion", type = float, default = 0.25,
                   help = "Keep-probability of each minibatch sample in the predictor loss; effective predictor "
                          "batch = update_proportion * rollout_steps / minibatches")
    p.add_argument("--rnd_lr", type = float, default = None,
                   help = "Adam lr for the RND predictor (default: --lr, i.e. one Adam as in CleanRL); "
                          "the main knob for how fast the bonus decays")
    p.add_argument("--rnd_rep_size", type = int, default = 512, help = "RND target/predictor output size")
    p.add_argument("--rnd_hidden", type = int, default = 512, help = "RND predictor hidden width")
    p.add_argument("--obs_norm_init_steps", type = int, default = 6400,
                   help = "Random-policy steps that warm-start the per-pixel obs stats (CleanRL 50 x 128 per env)")
    # bookkeeping
    p.add_argument("--seed", type = int, default = None, help = "Random if omitted")
    p.add_argument("--log_interval", type = int, default = None, help = "Env steps between MLflow logs (default: one rollout)")
    p.add_argument("--ckpt_interval", type = int, default = 20_000, help = "Training env steps between checkpoints")
    p.add_argument("--ckpt_dir", type = str, default = None,
                   help = "Checkpoint directory (default baselines/ppo_rnd/checkpoints/<run_id>)")
    p.add_argument("--mlflow_uri", type = str, default = DEFAULT_MLFLOW_URI)
    p.add_argument("--experiment", type = str, default = "rainworld-ppo-rnd")
    p.add_argument("--run_name", type = str, default = None)
    args = p.parse_args(argv)
    if args.seed is None:
        args.seed = random.randrange(2**31)
    if args.rnd_lr is None:
        args.rnd_lr = args.lr
    if args.log_interval is None:
        args.log_interval = args.rollout_steps
    if args.rollout_steps % args.minibatches != 0:
        p.error("--rollout_steps must be divisible by --minibatches")
    if args.total_steps < args.rollout_steps:
        p.error("--total_steps must be >= --rollout_steps")
    if not 0.0 < args.update_proportion <= 1.0:
        p.error("--update_proportion must be in (0, 1]")
    return args


def frame_channels_of(args: argparse.Namespace) -> int:
    return 3 if args.rgb else 1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RNDActorCritic(eqx.Module):
    """
    CleanRL's RND ``Agent`` with a factorised Bernoulli head: convs -> ReLU(256)
    -> ReLU(448) = h; policy ReLU(448) -> 9 logits; f = ReLU(extra(h));
    V_E = critic_ext(f + h), V_I = critic_int(f + h).
    """

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    extra: eqx.nn.Linear
    actor_hidden: eqx.nn.Linear
    actor_out: eqx.nn.Linear
    critic_ext: eqx.nn.Linear
    critic_int: eqx.nn.Linear

    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int = NUM_KEYS, hidden: int = 256,
                 feature_dim: int = 448, *, key: jax.Array):
        k = jr.split(key, 20)
        g = float(np.sqrt(2.0))
        self.conv1, self.conv2, self.conv3 = nature_convs(obs_shape[0], k[:6])
        flat = conv_flat_dim(obs_shape)
        self.fc1 = orthogonal_init(eqx.nn.Linear(flat, hidden, key = k[6]), g, k[7])
        self.fc2 = orthogonal_init(eqx.nn.Linear(hidden, feature_dim, key = k[8]), g, k[9])
        self.extra = orthogonal_init(eqx.nn.Linear(feature_dim, feature_dim, key = k[10]), 0.1, k[11])
        self.actor_hidden = orthogonal_init(eqx.nn.Linear(feature_dim, feature_dim, key = k[12]), 0.01, k[13])
        self.actor_out = orthogonal_init(eqx.nn.Linear(feature_dim, num_actions, key = k[14]), 0.01, k[15])
        self.critic_ext = orthogonal_init(eqx.nn.Linear(feature_dim, 1, key = k[16]), 0.01, k[17])
        self.critic_int = orthogonal_init(eqx.nn.Linear(feature_dim, 1, key = k[18]), 0.01, k[19])

    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """``obs``: uint8 ``(C, H, W)`` frame stack -> (logits ``(A,)``, V_E, V_I)."""
        x = obs.astype(jnp.float32) / 255.0
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = jax.nn.relu(self.fc1(x.reshape(-1)))
        hidden = jax.nn.relu(self.fc2(x))
        logits = self.actor_out(jax.nn.relu(self.actor_hidden(hidden)))
        features = jax.nn.relu(self.extra(hidden)) + hidden
        return logits, self.critic_ext(features)[0], self.critic_int(features)[0]


@eqx.filter_jit
def act(agent: RNDActorCritic, obs: jax.Array, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    One policy step as a single jitted call: split ``rng``, sample the keys from
    the Bernoulli policy. Returns (packed ``(A + 3,)`` float32 =
    ``[action..., log_prob, V_E, V_I]``, next rng). The packed vector is the only
    device -> host transfer per env step; the rng stays on the device.
    """
    rng, key = jr.split(rng)
    logits, v_ext, v_int = agent(obs)
    action = jr.bernoulli(key, jax.nn.sigmoid(logits)).astype(jnp.float32)
    packed = jnp.concatenate([action, jnp.stack([bernoulli_log_prob(logits, action), v_ext, v_int])])
    return packed, rng


def unpack_step(packed: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Host-side split of ``act``'s packed output -> (action ``(A,)``, log_prob, V_E, V_I)."""
    return packed[:-3], float(packed[-3]), float(packed[-2]), float(packed[-1])


@eqx.filter_jit
def values_of(agent: RNDActorCritic, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
    _, v_ext, v_int = agent(obs)
    return v_ext, v_int


# ---------------------------------------------------------------------------
# State / metrics
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    agent: RNDActorCritic
    predictor: RNDPredictor
    target: RNDTarget            # fixed random network; never receives gradients
    optimizer_state: optax.OptState
    rng: jax.Array
    step: jax.Array              # env steps consumed by updates so far (int32)

    optimizer: optax.GradientTransformation = eqx.field(static = True)
    epochs: int = eqx.field(static = True)
    minibatches: int = eqx.field(static = True)
    gamma: float = eqx.field(static = True)
    int_gamma: float = eqx.field(static = True)
    gae_lambda: float = eqx.field(static = True)
    clip: float = eqx.field(static = True)
    clip_vloss: bool = eqx.field(static = True)
    norm_adv: bool = eqx.field(static = True)
    vf_coef: float = eqx.field(static = True)
    ent_coef: float = eqx.field(static = True)
    ext_coef: float = eqx.field(static = True)
    int_coef: float = eqx.field(static = True)
    update_proportion: float = eqx.field(static = True)
    ext_death_terminal: bool = eqx.field(static = True)
    frame_channels: int = eqx.field(static = True)
    log_interval: int = eqx.field(static = True)
    ckpt_interval: int = eqx.field(static = True)


class StepMetrics(eqx.Module):
    policy_loss: jax.Array
    ext_value_loss: jax.Array
    int_value_loss: jax.Array
    entropy: jax.Array
    predictor_loss: jax.Array
    predictor_keep_frac: jax.Array
    approx_kl: jax.Array
    clipfrac: jax.Array
    grad_norm: jax.Array
    total_loss: jax.Array
    ext_explained_variance: jax.Array
    int_explained_variance: jax.Array
    ext_adv_std: jax.Array
    int_adv_std: jax.Array
    ext_return_mean: jax.Array
    int_return_mean: jax.Array


def make_optimizer(args: argparse.Namespace) -> optax.GradientTransformation:
    """
    One global-norm clip over (actor-critic, predictor) like CleanRL's single
    optimiser, then Adam(eps 1e-5) per group so the predictor lr can be tuned
    on its own (with ``rnd_lr == lr`` this is exactly one Adam over both).
    """
    steps_per_update = args.epochs * args.minibatches
    num_updates = max(1, args.total_steps // args.rollout_steps)

    def schedule(base_lr: float):
        if not args.anneal_lr:
            return base_lr
        # CleanRL: lr * (1 - (update - 1) / num_updates), constant within an update.
        return lambda count: base_lr * jnp.maximum(0.0, 1.0 - (count // steps_per_update) / num_updates)

    def labels(params):
        agent, predictor = params
        return (jax.tree.map(lambda _: "agent", agent), jax.tree.map(lambda _: "predictor", predictor))

    return optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.multi_transform(
            {"agent": optax.adam(schedule(args.lr), eps = 1e-5),
             "predictor": optax.adam(schedule(args.rnd_lr), eps = 1e-5)},
            labels,
        ),
    )


def build_networks(config: Dict[str, Any], key: jax.Array) -> Tuple[RNDActorCritic, RNDPredictor, RNDTarget]:
    obs_shape = obs_shape_for(config["frame_width"], config["frame_height"], config["frame_stack"], config["grayscale"])
    frame_shape = (config["frame_channels"], config["frame_height"], config["frame_width"])
    k_agent, k_pred, k_tgt = jr.split(key, 3)
    agent = RNDActorCritic(obs_shape, config.get("num_actions", NUM_KEYS), config["hidden"], config["feature_dim"],
                           key = k_agent)
    predictor = RNDPredictor(frame_shape, config["rnd_rep_size"], config["rnd_hidden"], key = k_pred)
    target = RNDTarget(frame_shape, config["rnd_rep_size"], key = k_tgt)
    return agent, predictor, target


def init_experiment(args: argparse.Namespace) -> TrainState:
    key = jr.PRNGKey(args.seed)
    model_key, rng = jr.split(key)
    agent, predictor, target = build_networks(model_config_from_args(args), model_key)
    optimizer = make_optimizer(args)
    optimizer_state = optimizer.init(eqx.filter((agent, predictor), eqx.is_array))
    return TrainState(
        agent = agent,
        predictor = predictor,
        target = target,
        optimizer_state = optimizer_state,
        rng = rng,
        step = jnp.asarray(0, dtype = jnp.int32),
        optimizer = optimizer,
        epochs = args.epochs,
        minibatches = args.minibatches,
        gamma = args.gamma,
        int_gamma = args.int_gamma,
        gae_lambda = args.gae_lambda,
        clip = args.clip,
        clip_vloss = args.clip_vloss,
        norm_adv = args.norm_adv,
        vf_coef = args.vf_coef,
        ent_coef = args.ent_coef,
        ext_coef = args.ext_coef,
        int_coef = args.int_coef,
        update_proportion = args.update_proportion,
        ext_death_terminal = args.ext_death_terminal,
        frame_channels = frame_channels_of(args),
        log_interval = args.log_interval,
        ckpt_interval = args.ckpt_interval,
    )


# ---------------------------------------------------------------------------
# PPO + RND maths
# ---------------------------------------------------------------------------

def compute_gae(rewards: jax.Array, values: jax.Array, last_value: jax.Array, nonterminal: jax.Array,
                gamma: float, lam: float) -> Tuple[jax.Array, jax.Array]:
    """
    GAE over one stream. ``nonterminal[t] = 0`` means the transition t -> t+1
    ended the trajectory: V(s_{t+1}) is not bootstrapped and the trace is cut
    (CleanRL's ``1 - dones[t + 1]``). All ones = continuing. Returns (adv, returns).
    """
    next_values = jnp.concatenate([values[1:], last_value[None]])
    deltas = rewards + gamma * next_values * nonterminal - values

    def scan_fn(next_adv, x):
        delta, nt = x
        adv = delta + gamma * lam * nt * next_adv
        return adv, adv

    _, advantages = jax.lax.scan(scan_fn, jnp.zeros_like(last_value), (deltas, nonterminal), reverse = True)
    return advantages, advantages + values


def compute_dual_gae(batch: Dict[str, jax.Array], gamma: float, int_gamma: float, lam: float,
                     ext_death_terminal: bool) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Extrinsic GAE (continuing, or death-terminal with ``ext_death_terminal``)
    and intrinsic GAE (always non-episodic: death never cuts it).
    Returns (ext_adv, ext_returns, int_adv, int_returns).
    """
    deaths = batch["deaths"]
    ext_nonterminal = (1.0 - deaths) if ext_death_terminal else jnp.ones_like(deaths)
    int_nonterminal = jnp.ones_like(deaths)
    ext_adv, ext_ret = compute_gae(batch["ext_rewards"], batch["ext_values"], batch["last_ext_value"],
                                   ext_nonterminal, gamma, lam)
    int_adv, int_ret = compute_gae(batch["int_rewards"], batch["int_values"], batch["last_int_value"],
                                   int_nonterminal, int_gamma, lam)
    return ext_adv, ext_ret, int_adv, int_ret


def ppo_rnd_loss(learner, target_features: jax.Array, mb: Dict[str, jax.Array], mask_key: jax.Array,
                 clip: float, clip_vloss: bool, norm_adv: bool, vf_coef: float, ent_coef: float,
                 update_proportion: float):
    agent, predictor = learner
    logits, ext_values, int_values = jax.vmap(agent)(mb["obs"])
    log_probs = bernoulli_log_prob(logits, mb["actions"])
    entropy = bernoulli_entropy(logits).mean()

    log_ratio = log_probs - mb["log_probs"]
    ratio = jnp.exp(log_ratio)
    adv = mb["advantages"]
    if norm_adv:
        adv = (adv - adv.mean()) / (jnp.std(adv, ddof = 1) + 1e-8)
    policy_loss = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1.0 - clip, 1.0 + clip)).mean()

    ext_returns = mb["ext_returns"]
    if clip_vloss:
        v_clipped = mb["ext_values"] + jnp.clip(ext_values - mb["ext_values"], -clip, clip)
        ext_value_loss = 0.5 * jnp.maximum((ext_values - ext_returns) ** 2, (v_clipped - ext_returns) ** 2).mean()
    else:
        ext_value_loss = 0.5 * ((ext_values - ext_returns) ** 2).mean()
    int_value_loss = 0.5 * ((int_values - mb["int_returns"]) ** 2).mean()

    # RND predictor: MSE to the fixed target on a random update_proportion of the minibatch.
    pred = jax.vmap(predictor)(mb["rnd_obs"])
    forward = jnp.mean((pred - jax.lax.stop_gradient(target_features)) ** 2, axis = -1)
    mask = (jr.uniform(mask_key, forward.shape) < update_proportion).astype(jnp.float32)
    predictor_loss = jnp.sum(forward * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    loss = policy_loss - ent_coef * entropy + vf_coef * (ext_value_loss + int_value_loss) + predictor_loss
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > clip).astype(jnp.float32))
    return loss, (policy_loss, ext_value_loss, int_value_loss, entropy, predictor_loss, mask.mean(), approx_kl, clipfrac)


def _explained_variance(returns: jax.Array, values: jax.Array) -> jax.Array:
    return 1.0 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)


@eqx.filter_jit
def train_step(train_state: TrainState, batch: Dict[str, jax.Array]) -> Tuple[TrainState, StepMetrics]:
    """One PPO + RND update: dual GAE, then ``epochs x minibatches`` Adam steps via ``lax.scan``."""
    rng, perm_key, mask_key = jr.split(train_state.rng, 3)
    num_steps = batch["ext_rewards"].shape[0]
    mb_size = num_steps // train_state.minibatches
    n_grad = train_state.epochs * train_state.minibatches

    ext_adv, ext_ret, int_adv, int_ret = compute_dual_gae(
        batch, train_state.gamma, train_state.int_gamma, train_state.gae_lambda, train_state.ext_death_terminal
    )
    advantages = train_state.ext_coef * ext_adv + train_state.int_coef * int_adv

    # Predictor inputs: latest frame of each policy observation, whitened with the
    # stats already updated by this rollout (CleanRL); target features once (fixed net).
    rnd_obs = whiten(latest_frame(batch["obs"], train_state.frame_channels), batch["obs_mean"], batch["obs_std"])
    target_features = jax.vmap(train_state.target)(rnd_obs)

    perms = jax.vmap(lambda k: jr.permutation(k, num_steps))(jr.split(perm_key, train_state.epochs))
    minibatch_indices = perms.reshape(n_grad, mb_size)
    mask_keys = jr.split(mask_key, n_grad)

    learner = (train_state.agent, train_state.predictor)
    params, static = eqx.partition(learner, eqx.is_array)
    optimizer = train_state.optimizer
    loss_fn = eqx.filter_value_and_grad(ppo_rnd_loss, has_aux = True)

    def minibatch_step(carry, xs):
        params, optimizer_state = carry
        idx, key = xs
        mb = {
            "obs": batch["obs"][idx],
            "actions": batch["actions"][idx],
            "log_probs": batch["log_probs"][idx],
            "ext_values": batch["ext_values"][idx],
            "advantages": advantages[idx],
            "ext_returns": ext_ret[idx],
            "int_returns": int_ret[idx],
            "rnd_obs": rnd_obs[idx],
        }
        (loss, aux), grads = loss_fn(
            eqx.combine(params, static), target_features[idx], mb, key, train_state.clip, train_state.clip_vloss,
            train_state.norm_adv, train_state.vf_coef, train_state.ent_coef, train_state.update_proportion,
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = eqx.apply_updates(params, updates)
        policy_loss, ext_value_loss, int_value_loss, entropy, predictor_loss, keep, approx_kl, clipfrac = aux
        metrics = StepMetrics(
            policy_loss = policy_loss, ext_value_loss = ext_value_loss, int_value_loss = int_value_loss,
            entropy = entropy, predictor_loss = predictor_loss, predictor_keep_frac = keep, approx_kl = approx_kl,
            clipfrac = clipfrac, grad_norm = optax.tree.norm(grads), total_loss = loss,
            ext_explained_variance = jnp.zeros(()), int_explained_variance = jnp.zeros(()),
            ext_adv_std = jnp.zeros(()), int_adv_std = jnp.zeros(()),
            ext_return_mean = jnp.zeros(()), int_return_mean = jnp.zeros(()),
        )
        return (params, optimizer_state), metrics

    (params, optimizer_state), metrics = jax.lax.scan(
        minibatch_step, (params, train_state.optimizer_state), (minibatch_indices, mask_keys), length = n_grad,
        unroll = MINIBATCH_UNROLL,
    )
    metrics = jax.tree.map(lambda x: x.mean(0), metrics)
    metrics = eqx.tree_at(
        lambda m: (m.ext_explained_variance, m.int_explained_variance, m.ext_adv_std, m.int_adv_std,
                   m.ext_return_mean, m.int_return_mean),
        metrics,
        (_explained_variance(ext_ret, batch["ext_values"]), _explained_variance(int_ret, batch["int_values"]),
         jnp.std(ext_adv), jnp.std(int_adv), ext_ret.mean(), int_ret.mean()),
    )

    agent, predictor = eqx.combine(params, static)
    new_state = TrainState(
        agent = agent,
        predictor = predictor,
        target = train_state.target,
        optimizer_state = optimizer_state,
        rng = rng,
        step = train_state.step + num_steps,
        optimizer = optimizer,
        epochs = train_state.epochs,
        minibatches = train_state.minibatches,
        gamma = train_state.gamma,
        int_gamma = train_state.int_gamma,
        gae_lambda = train_state.gae_lambda,
        clip = train_state.clip,
        clip_vloss = train_state.clip_vloss,
        norm_adv = train_state.norm_adv,
        vf_coef = train_state.vf_coef,
        ent_coef = train_state.ent_coef,
        ext_coef = train_state.ext_coef,
        int_coef = train_state.int_coef,
        update_proportion = train_state.update_proportion,
        ext_death_terminal = train_state.ext_death_terminal,
        frame_channels = train_state.frame_channels,
        log_interval = train_state.log_interval,
        ckpt_interval = train_state.ckpt_interval,
    )
    return new_state, metrics


# ---------------------------------------------------------------------------
# Rollout collection (Python loop over the env, jitted policy) and RND bookkeeping
# ---------------------------------------------------------------------------

def collect_rollout(
    env, agent: RNDActorCritic, obs: np.ndarray, rng: jax.Array, num_steps: int,
    metrics: RolloutMetrics, reward_scale: float, deadline: Optional[float] = None,
) -> Tuple[Optional[Dict[str, np.ndarray]], np.ndarray, jax.Array, Dict[str, Any]]:
    """
    Step the env ``num_steps`` times -> (rollout, next_obs, rng, last_info).
    ``rollout`` is None if ``deadline`` (``time.perf_counter()``) passed first;
    the partial rollout is then discarded (its env steps still went to ``metrics``).
    """
    obs_buf = np.empty((num_steps,) + obs.shape, dtype = np.uint8)
    act_buf = np.empty((num_steps, NUM_KEYS), dtype = np.float32)
    logp_buf = np.empty(num_steps, dtype = np.float32)
    vext_buf = np.empty(num_steps, dtype = np.float32)
    vint_buf = np.empty(num_steps, dtype = np.float32)
    rew_buf = np.empty(num_steps, dtype = np.float32)
    death_buf = np.zeros(num_steps, dtype = np.float32)
    info: Dict[str, Any] = {}

    for t in range(num_steps):
        if deadline is not None and time.perf_counter() >= deadline:
            return None, obs, rng, info
        packed, rng = act(agent, jnp.asarray(obs), rng)
        action, log_prob, v_ext, v_int = unpack_step(np.asarray(packed))
        next_obs, reward, _terminated, _truncated, info = env.step(action.astype(np.int8))
        metrics.update(info, float(reward))

        obs_buf[t] = obs
        act_buf[t] = action
        logp_buf[t] = log_prob
        vext_buf[t] = v_ext
        vint_buf[t] = v_int
        rew_buf[t] = float(reward) * reward_scale
        death_buf[t] = float(bool(info.get("player_dead", False)))
        obs = next_obs

    last_ext, last_int = jax.device_get(values_of(agent, jnp.asarray(obs)))
    rollout = {
        "obs": obs_buf, "actions": act_buf, "log_probs": logp_buf,
        "ext_values": vext_buf, "int_values": vint_buf, "ext_rewards": rew_buf, "deaths": death_buf,
        "last_ext_value": np.float32(last_ext), "last_int_value": np.float32(last_int),
    }
    return rollout, obs, rng, info


def prepare_batch(train_state: TrainState, stats: RNDStats, rollout: Dict[str, np.ndarray],
                  next_obs: np.ndarray) -> Tuple[Dict[str, jax.Array], Dict[str, float]]:
    """
    RND bookkeeping between rollout and update, in CleanRL's order: bonus on
    s_{t+1} with the pre-rollout obs stats -> forward-filter normalisation ->
    obs-stat update with the rollout's latest frames -> device batch for
    ``train_step``. Returns (batch, intrinsic-reward stats for logging).
    """
    c = train_state.frame_channels
    obs = rollout["obs"]
    next_frames = np.concatenate([latest_frame(obs[1:], c), latest_frame(next_obs[None], c)], axis = 0)
    mean, std = stats.obs_mean_std()
    raw = np.asarray(intrinsic_reward(train_state.predictor, train_state.target, jnp.asarray(next_frames),
                                      jnp.asarray(mean), jnp.asarray(std)), dtype = np.float64)
    normalised, rets = stats.normalise_intrinsic(raw)
    stats.update_obs(latest_frame(obs, c))
    mean, std = stats.obs_mean_std()

    batch = {k: jnp.asarray(v) for k, v in rollout.items()}
    batch["int_rewards"] = jnp.asarray(normalised.astype(np.float32))
    batch["obs_mean"] = jnp.asarray(mean)
    batch["obs_std"] = jnp.asarray(std)
    info = {
        "rnd/int_reward_raw_mean": float(raw.mean()),
        "rnd/int_reward_raw_std": float(raw.std()),
        "rnd/int_reward_raw_max": float(raw.max()),
        "rnd/int_reward_norm_mean": float(normalised.mean()),
        "rnd/int_reward_norm_std": float(normalised.std()),
        "rnd/int_reward_norm_max": float(normalised.max()),
        "rnd/int_return_std": stats.int_return_std,
        "rnd/int_return_filtered_mean": float(rets.mean()),
        "rnd/obs_rms_count": float(stats.obs_rms.count),
        "rnd/obs_std_mean": float(std.mean()),
        "rollout/ext_reward_mean": float(rollout["ext_rewards"].mean()),
        "rollout/ext_reward_sum": float(rollout["ext_rewards"].sum()),
        "rollout/deaths": float(rollout["deaths"].sum()),
    }
    return batch, info


def init_obs_norm(env, obs: np.ndarray, stats: RNDStats, steps: int, chunk: int, frame_channels: int,
                  metrics: RolloutMetrics, np_rng: np.random.Generator,
                  deadline: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Warm-start the per-pixel obs stats with ``steps`` uniform-random-key steps
    (CleanRL: ``num_steps * num_iterations_obs_norm_init`` random steps,
    folded in per rollout-sized chunk). Returns (obs, steps taken).
    """
    frames = []
    taken = 0
    for _ in range(steps):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        action = np_rng.integers(0, 2, size = NUM_KEYS, dtype = np.int8)
        obs, reward, _terminated, _truncated, info = env.step(action)
        metrics.update(info, float(reward))
        frames.append(latest_frame(obs, frame_channels))
        taken += 1
        if len(frames) == chunk:
            stats.update_obs(np.stack(frames))
            frames = []
    if frames:
        stats.update_obs(np.stack(frames))
    return obs, taken


def dummy_rollout(num_steps: int, obs_shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    """Zero rollout with the exact dtypes/shapes of ``collect_rollout`` (JIT warm-up, tests)."""
    z = np.zeros(num_steps, dtype = np.float32)
    return {
        "obs": np.zeros((num_steps,) + tuple(obs_shape), dtype = np.uint8),
        "actions": np.zeros((num_steps, NUM_KEYS), dtype = np.float32),
        "log_probs": z.copy(), "ext_values": z.copy(), "int_values": z.copy(),
        "ext_rewards": z.copy(), "deaths": z.copy(),
        "last_ext_value": np.float32(0.0), "last_int_value": np.float32(0.0),
    }


def warm_up_jit(train_state: TrainState, args: argparse.Namespace) -> float:
    """Compile ``act``, ``values_of``, ``intrinsic_reward`` and ``train_step`` (results discarded). Returns seconds."""
    t0 = time.perf_counter()
    obs_shape = obs_shape_for(args.frame_width, args.frame_height, args.frame_stack, not args.rgb)
    obs = jnp.zeros(obs_shape, dtype = jnp.uint8)
    jax.block_until_ready(act(train_state.agent, obs, jr.PRNGKey(0)))
    jax.block_until_ready(values_of(train_state.agent, obs))
    scratch = RNDStats((train_state.frame_channels, args.frame_height, args.frame_width), args.int_gamma)
    batch, _ = prepare_batch(train_state, scratch, dummy_rollout(args.rollout_steps, obs_shape), np.asarray(obs))
    _, m = train_step(train_state, batch)
    jax.block_until_ready(m.total_loss)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Logging / checkpoints
# ---------------------------------------------------------------------------

def log_metrics(metrics: StepMetrics, rnd_info: Dict[str, float], env_metrics: Dict[str, float],
                timing: Dict[str, float], step: int) -> Dict[str, float]:
    out = {
        "ppo/policy_loss": float(metrics.policy_loss),
        "ppo/ext_value_loss": float(metrics.ext_value_loss),
        "ppo/int_value_loss": float(metrics.int_value_loss),
        "ppo/entropy": float(metrics.entropy),
        "ppo/approx_kl": float(metrics.approx_kl),
        "ppo/clipfrac": float(metrics.clipfrac),
        "ppo/grad_norm": float(metrics.grad_norm),
        "ppo/total_loss": float(metrics.total_loss),
        "ppo/ext_explained_variance": float(metrics.ext_explained_variance),
        "ppo/int_explained_variance": float(metrics.int_explained_variance),
        "rnd/predictor_loss": float(metrics.predictor_loss),
        "rnd/predictor_keep_frac": float(metrics.predictor_keep_frac),
        "rnd/ext_adv_std": float(metrics.ext_adv_std),
        "rnd/int_adv_std": float(metrics.int_adv_std),
        "rnd/ext_return_mean": float(metrics.ext_return_mean),
        "rnd/int_return_mean": float(metrics.int_return_mean),
    }
    out.update(rnd_info)
    out.update(env_metrics)
    out.update({f"time/{k}": v for k, v in timing.items()})
    mlflow.log_metrics(out, step = step)
    return out


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "algo": "ppo_rnd",
        "frame_width": args.frame_width,
        "frame_height": args.frame_height,
        "frame_stack": args.frame_stack,
        "grayscale": not args.rgb,
        "frame_channels": frame_channels_of(args),
        "ticks_per_step": args.ticks_per_step,
        "hidden": args.hidden,
        "feature_dim": args.feature_dim,
        "rnd_rep_size": args.rnd_rep_size,
        "rnd_hidden": args.rnd_hidden,
        "num_actions": NUM_KEYS,
        "key_names": list(KEY_NAMES),
    }


def save_checkpoint(train_state: TrainState, stats: RNDStats, ckpt_dir: Path, step: int,
                    config: Dict[str, Any]) -> Path:
    """``(agent, predictor, target)`` -> ``model_<step>.eqx`` + ``latest.eqx``; RND stats -> ``*.npz``."""
    ckpt_dir.mkdir(parents = True, exist_ok = True)
    config_path = ckpt_dir / MODEL_CONFIG_FILENAME
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent = 2), encoding = "utf-8")
    path = ckpt_dir / f"model_{step:09d}.eqx"
    eqx.tree_serialise_leaves(path, (train_state.agent, train_state.predictor, train_state.target))
    shutil.copyfile(path, ckpt_dir / "latest.eqx")
    stats_path = ckpt_dir / f"rnd_stats_{step:09d}.npz"
    stats.save(stats_path)
    shutil.copyfile(stats_path, ckpt_dir / LATEST_STATS_FILENAME)
    return path


def load_checkpoint(path: Path, config: Optional[Dict[str, Any]] = None):
    """Load ``(agent, predictor, target)``; ``config`` defaults to ``model_config.json`` next to ``path``."""
    path = Path(path)
    if config is None:
        config = json.loads((path.parent / MODEL_CONFIG_FILENAME).read_text(encoding = "utf-8"))
    skeleton = build_networks(config, jr.PRNGKey(0))
    return eqx.tree_deserialise_leaves(path, skeleton), config


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(train_state: TrainState, env, args: argparse.Namespace, ckpt_dir: Path) -> Tuple[TrainState, RNDStats, Dict[str, float]]:
    c = frame_channels_of(args)
    stats = RNDStats((c, args.frame_height, args.frame_width), args.int_gamma)
    metrics = RolloutMetrics()
    obs, info = env.reset(options = {"wipe": not args.no_wipe})
    metrics.start(info)
    t_start = time.perf_counter()
    deadline = t_start + args.seconds if args.seconds and args.seconds > 0 else None
    rollout_rng = jr.PRNGKey(args.seed + 1)
    np_rng = np.random.default_rng(args.seed + 2)
    model_config = model_config_from_args(args)

    logger.info("obs-norm warm-up: %d random-policy steps", args.obs_norm_init_steps)
    obs, warm = init_obs_norm(env, obs, stats, args.obs_norm_init_steps, args.rollout_steps, c, metrics, np_rng, deadline)
    warm_seconds = time.perf_counter() - t_start
    logger.info("obs-norm warm-up done: %d steps in %.1fs (%.0f steps/s), mean pixel std %.2f",
                warm, warm_seconds, warm / max(1e-9, warm_seconds), float(stats.obs_mean_std()[1].mean()))
    mlflow.log_metrics({"rnd/warmup_steps": float(warm), "time/warmup_seconds": warm_seconds},
                       step = metrics.total_steps)

    env_steps = 0                     # training steps (excl. warm-up)
    updates = 0
    next_log = args.log_interval
    next_ckpt = args.ckpt_interval
    last_logged: Dict[str, float] = {}
    stop_reason = "total_steps"
    pbar = tqdm(total = args.total_steps, unit = "step", dynamic_ncols = True)
    try:
        while env_steps < args.total_steps:
            if deadline is not None and time.perf_counter() >= deadline:
                stop_reason = "seconds"
                break
            t0 = time.perf_counter()
            rollout, obs, rollout_rng, info = collect_rollout(
                env, train_state.agent, obs, rollout_rng, args.rollout_steps, metrics, args.reward_scale, deadline
            )
            if rollout is None:
                stop_reason = "seconds"
                break
            t1 = time.perf_counter()
            batch, rnd_info = prepare_batch(train_state, stats, rollout, obs)
            t2 = time.perf_counter()
            train_state, step_metrics = train_step(train_state, batch)
            jax.block_until_ready(step_metrics.total_loss)
            t3 = time.perf_counter()
            env_steps += args.rollout_steps
            updates += 1
            pbar.update(args.rollout_steps)

            timing = {
                "rollout_seconds": t1 - t0,
                "intrinsic_seconds": t2 - t1,
                "update_seconds": t3 - t2,
                "env_steps_per_second": args.rollout_steps / max(1e-9, t1 - t0),
                "elapsed_seconds": t3 - t_start,
            }
            if env_steps >= next_log or env_steps >= args.total_steps:
                env_summary = metrics.summary()
                last_logged = log_metrics(step_metrics, rnd_info, env_summary, timing, metrics.total_steps)
                metrics.end_window()
                while next_log <= env_steps:
                    next_log += args.log_interval
                pbar.set_postfix(
                    r = f"{env_summary['env/reward_mean']:.4f}",
                    ri = f"{rnd_info['rnd/int_reward_raw_mean']:.3g}",
                    rooms = int(env_summary["env/rooms_discovered"]),
                    deaths = int(env_summary["env/deaths"]),
                    sps = f"{timing['env_steps_per_second']:.0f}",
                    upd = f"{timing['update_seconds']:.1f}s",
                )
                tqdm.write(
                    f"[update {updates} | step {metrics.total_steps} | t {timing['elapsed_seconds']:.0f}s] "
                    f"r/step {env_summary['env/reward_mean']:+.4f}  "
                    f"int raw {rnd_info['rnd/int_reward_raw_mean']:.4g}+-{rnd_info['rnd/int_reward_raw_std']:.3g}  "
                    f"int norm {rnd_info['rnd/int_reward_norm_mean']:.3f}+-{rnd_info['rnd/int_reward_norm_std']:.3f}  "
                    f"ret std {rnd_info['rnd/int_return_std']:.3g}  fwd {float(step_metrics.predictor_loss):.4g}  "
                    f"VL e/i {float(step_metrics.ext_value_loss):.3g}/{float(step_metrics.int_value_loss):.3g}  "
                    f"H {float(step_metrics.entropy):.3f}  rooms {int(env_summary['env/rooms_discovered'])}  "
                    f"deaths {int(rnd_info['rollout/deaths'])}/{int(env_summary['env/deaths'])}  "
                    f"sps {timing['env_steps_per_second']:.0f}  upd {timing['update_seconds']:.1f}s"
                )
            if env_steps >= next_ckpt:
                save_checkpoint(train_state, stats, ckpt_dir, env_steps, model_config)
                while next_ckpt <= env_steps:
                    next_ckpt += args.ckpt_interval
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        logger.warning("interrupted at %d training steps; saving checkpoint", env_steps)
    finally:
        pbar.close()
        path = save_checkpoint(train_state, stats, ckpt_dir, env_steps, model_config)
        logger.info("stopped (%s) after %d training + %d warm-up env steps, %d updates, %.0fs; final checkpoint: %s",
                    stop_reason, env_steps, warm, updates, time.perf_counter() - t_start, path)
        cumulative = ("rooms_discovered", "deaths", "cycles_survived", "food_eaten", "karma", "reward_total", "total_steps")
        summary = metrics.summary()
        final = {f"env/{k}": summary[f"env/{k}"] for k in cumulative}
        final.update({"train/updates": float(updates), "train/training_steps": float(env_steps),
                      "train/warmup_steps": float(warm), "train/seconds": time.perf_counter() - t_start})
        mlflow.log_metrics({f"final/{k.replace('/', '_')}": v for k, v in final.items()}, step = metrics.total_steps)
        if last_logged:
            print("\nlast logged metrics:\n" + format_summary(last_logged))
        print("\nfinal (cumulative):\n" + format_summary(final))
    return train_state, stats, last_logged


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("rainworld_rl imported from %s", rainworld_rl.__file__)
    logger.info("jax backend: %s, devices: %s", jax.default_backend(), jax.devices())

    setup_mlflow(args.mlflow_uri, args.experiment)
    train_state = init_experiment(args)

    def n_params(tree) -> int:
        return sum(int(np.prod(a.shape)) for a in jax.tree.leaves(eqx.filter(tree, eqx.is_array)))

    counts = {"param_count_agent": n_params(train_state.agent), "param_count_predictor": n_params(train_state.predictor),
              "param_count_target": n_params(train_state.target)}
    predictor_batch = args.update_proportion * args.rollout_steps / args.minibatches
    logger.info("parameters: %s; effective predictor batch %.0f samples/step, %.2f predictor passes per env sample",
                counts, predictor_batch, args.epochs * args.update_proportion)

    # Compile everything before touching the game so the lock is not held while XLA compiles.
    logger.info("JIT warm-up ...")
    logger.info("JIT warm-up took %.1fs", warm_up_jit(train_state, args))

    mlflow.start_run(run_name = args.run_name)
    run_id = mlflow.active_run().info.run_id
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else DEFAULT_CKPT_ROOT / run_id
    mlflow.log_params({**vars(args), **counts, "run_id": run_id, "ckpt_dir": str(ckpt_dir),
                       "rainworld_rl_path": rainworld_rl.__file__, "jax_backend": jax.default_backend(),
                       "device": str(jax.devices()[0]),
                       "predictor_batch_effective": predictor_batch, "cleanrl_commit": CLEANRL_COMMIT})
    logger.info("run %s, checkpoints -> %s", run_id, ckpt_dir)
    try:
        with game_lock(args.game_lock):
            env = make_env(
                frame_width = args.frame_width, frame_height = args.frame_height, ticks_per_step = args.ticks_per_step,
                frame_stack = args.frame_stack, grayscale = not args.rgb, fake = args.fake, seed = args.seed,
                novelty = args.novelty,
            )
            try:
                attach(env, launch = args.launch)
                train(train_state, env, args, ckpt_dir)
            finally:
                try:
                    env.close()
                finally:
                    if args.kill_game and not args.fake:
                        from rainworld_rl.launcher import kill_game
                        kill_game()
    finally:
        mlflow.end_run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
