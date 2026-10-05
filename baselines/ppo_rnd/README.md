# PPO + RND

Random Network Distillation (Burda, Edwards, Storkey, Klimov, *Exploration by
Random Network Distillation*, ICLR 2019, [arXiv:1810.12894](https://arxiv.org/abs/1810.12894))
on top of PPO, for the continuing Rain World env. It is the Tier-1 lifelong
novelty-bonus control in `docs/EXPLORATION_METHODS.md` (section 3, item 1).

```
baselines/ppo_rnd/
  rnd.py             target / predictor CNNs, torch-style orthogonal init, whitening, RunningMeanStd,
                     RewardForwardFilter, RNDStats (all lifetime statistics, never reset)
  train_ppo_rnd.py   trainer: two-head actor-critic, dual GAE, PPO + predictor update, rollout loop, CLI
  checkpoints/       <run_id>/model_<step>.eqx + latest.eqx (agent, predictor, target),
                     rnd_stats_<step>.npz + latest_rnd_stats.npz, model_config.json   (gitignored)
baselines/tests/test_ppo_rnd_{model,stats,gae,train}.py
```

## Reference code

Ported from CleanRL [`cleanrl/ppo_rnd_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py)
(docs: https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/) at **vwxyzjn/cleanrl
master `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`** (2026-04-20; the file
itself last changed in `35896b1fefa9898b904f7e09bcbe6e168e15d2a9`, 2023-11-27).
Secondary reference / tie-breaker: [openai/random-network-distillation](https://github.com/openai/random-network-distillation)
at `f75c0f1efa473d5109d487062fd8ed49ddce6634` (archived TF1). The
Craftax_Baselines JAX RND was deliberately not used (no normalisation, one gamma).

## Algorithm as implemented

Networks (orthogonal init, gain sqrt(2) and zero bias unless noted, exactly as CleanRL's `layer_init`):

* **Actor-critic** (policy input = the normal uint8 frame stack `/255`): Nature
  convs (32x8s4, 64x4s2, 64x3s1, ReLU) -> ReLU(256) -> ReLU(448) = `h`.
  Policy: ReLU(448, gain 0.01) -> 9 logits (gain 0.01), one independent
  Bernoulli per key (same factorised head as `baselines/ppo`).
  `f = ReLU(extra(h))` (gain 0.1); `V_E = critic_ext(f + h)`, `V_I = critic_int(f + h)` (gain 0.01).
* **RND target** (fixed, never trained): same convs with LeakyReLU -> Linear(512).
* **RND predictor** (trained): same convs with LeakyReLU -> ReLU(512) -> ReLU(512) -> Linear(512).

Per rollout of `--rollout_steps` steps (Python loop, jitted `act`):

1. **Bonus**: for each transition, the single **latest frame** of s_{t+1}
   (raw 0..255 values), whitened per pixel with the running mean/std from
   *before* this rollout and clipped to [-5, 5]:
   `i_t = sum((target(x) - predictor(x))^2) / 2`.
2. **Normalisation**: `RewardForwardFilter(gamma_I)` runs along time
   (`R_t = gamma_I * R_{t-1} + i_t`), the rollout's filtered values update a
   scalar `RunningMeanStd`, and `i_t / sqrt(var)` is the intrinsic reward. No
   mean subtraction, no clipping. The filter state and both running stats live
   for the whole lifetime and are never reset.
3. **Two-head GAE** (lambda 0.95): intrinsic with gamma_I, **non-episodic**
   (never cut, including at death); extrinsic with gamma_E, continuing like the
   PPO baseline (no terminals), or with `--ext_death_terminal` cut at the step
   where `info["player_dead"]` is True (extrinsic head only).
   `A = ext_coef * A_E + int_coef * A_I`.
4. **Obs stats update** with the latest frame of every observation in the rollout.
5. **One jitted update**: `epochs x minibatches` Adam steps (a fully unrolled
   `lax.scan`) on `pg_loss - ent_coef * H + vf_coef * (L_VE + L_VI) + L_fwd`;
   clipped surrogate, per-minibatch advantage normalisation, extrinsic value
   clipping (`--clip`), unclipped intrinsic value loss;
   `L_fwd` = per-sample feature MSE on a Bernoulli(`--update_proportion`) mask
   of the minibatch, averaged over the kept samples; one global grad-norm
   clip (0.5) over policy + predictor parameters.

Before training, the per-pixel obs stats are warm-started with
`--obs_norm_init_steps` uniform-random-key steps (CleanRL: 50 x 128 steps per
env -> 6400 for our single env).

## Single-env scaling (the effective predictor batch)

CleanRL collects 128 envs x 128 steps = 16384 samples per update in 4
minibatches of 4096; with `update_proportion = 0.25` each predictor step sees
~1024 samples (= the 32-env predictor batch of the paper) and every sample is
used `epochs * p = 1` time in expectation. We have one env, so:

* `--rollout_steps 1024`, `--minibatches 4` (256 per minibatch), `--epochs 4`,
  as in the PPO baseline (128-step single-env rollouts would give 32-sample
  minibatches).
* `--update_proportion 0.25` is kept: **effective predictor batch =
  `update_proportion * rollout_steps / minibatches` = 64 samples per Adam
  step**, and each env sample still gets `epochs * p = 1` predictor pass, the
  same ratio as CleanRL; the predictor-to-policy batch ratio (0.25) is also the
  same. The value is logged as the `predictor_batch_effective` param.
* What does *not* carry over: per env step the predictor makes 16 Adam steps
  per 1024 env steps instead of 16 per 16384, so with Adam it moves faster per
  env step than in CleanRL, i.e. the bonus can decay faster. The two decay
  knobs are `--rnd_lr` (predictor Adam lr, default = `--lr`; the cleanest
  knob, since Adam step size barely depends on batch size) and
  `--update_proportion` (how much of each minibatch the predictor fits).
  Lower either if `rnd/int_reward_raw_mean` collapses before the policy can
  use it; raise `--update_proportion` to 1.0 to match the paper's 32-env
  setting (4 predictor passes per sample).

## Deviations from CleanRL, and why

| # | CleanRL | here | why |
|---|---------|------|-----|
| 1 | Categorical over 18 Atari actions | factorised Bernoulli over the 9 keys (`MultiBinary(9)`) | env action space; same head as `baselines/ppo` (docs section 2, item 8) |
| 2 | 128 envs x 128 steps, 4 x 4096 minibatches | 1 env x 1024 steps, 4 x 256 | one game instance; see the scaling section |
| 3 | gamma_E = 0.999 | `--gamma 0.99` | matches the PPO baseline; Wan et al. 2025 show 0.999 is fragile in continuing tasks |
| 4 | extrinsic head episodic (`episodic_life=True`) | continuing by default, `--ext_death_terminal` makes death terminal for the extrinsic head only | house default for a continuing env; the flag recovers the paper's "find reward, die, repeat" protection |
| 5 | extrinsic reward sign-clipped (envpool `reward_clip=True`) | not clipped; `--reward_scale` | the drive reward has designed magnitudes (death -3, food 0.1-0.3, sleep ~1); clipping is Atari-specific |
| 6 | `anneal_lr=True` to 0 over 2e9 steps | off by default (`--anneal_lr` to enable) | a lifetime run has no natural end and `--seconds` may stop it early |
| 7 | `RewardForwardFilter` applied to `curiosity_rewards.T`, i.e. along the env axis | along the time axis | the official code (`buf_rews_int.T`, shape `(nsteps, nenvs)`) filters over time; CleanRL's transpose walks envs, which with one env would filter across rollouts element-wise |
| 8 | one Adam over agent + predictor | one global-norm clip over both, then Adam per group (`--rnd_lr`, default = `--lr`) | identical when `rnd_lr == lr` (Adam is per-parameter); exposes the predictor lr as a decay knob |
| 9 | bonus computed per step inside the rollout | computed batched after the rollout | identical numbers: predictor and obs stats do not change during a rollout |
| 10 | target features recomputed per minibatch | computed once per update | identical (fixed network) |
| 11 | 84x84 frames | 96x54 (env default, 16:9) | env; conv output 64x3x8 = 1536 instead of 3136 |
| 12 | sticky actions p = 0.25 | none | not part of this env; Rain World is stochastic already |
| 13 | 50 x 128 x 128 random warm-up steps | `--obs_norm_init_steps 6400` (= 50 x 128 for one env), uniform random keys | single env. Warm-up steps are real env steps: they feed `RolloutMetrics` and the MLflow step axis, but do not count toward `--total_steps` (CleanRL's `global_step` also excludes them) |
| 14 | `torch.nn.init.orthogonal_` | same semantics (conv kernels viewed as `(out, in*kh*kw)`) | note: `baselines/ppo`'s `_orthogonal` applies jax's initializer to the raw 4-D kernel, which orthogonalises over the kernel-width axis; not used here so the random target's feature scale matches the reference |

Everything else follows CleanRL: lr 1e-4 (Adam eps 1e-5), entropy 0.001,
clip 0.1 (also the extrinsic value clip), vf 0.5, max grad norm 0.5, 4 epochs,
4 minibatches, gamma_I 0.99, lambda 0.95, int/ext coef 1/2, update proportion
0.25, per-minibatch advantage normalisation (unbiased std, like `torch.std`),
whitening on raw pixel values with gym's `RunningMeanStd` (float64, initial
count 1e-4, var 1), predictor trained on the latest frame of the policy
observations whitened with the stats after this rollout's update.

Continuing-env notes: there are no episode boundaries anywhere; the intrinsic
head is non-episodic by design so no death mask is applied to it; the death
count per rollout is logged (`rollout/deaths`). `ready=False` / human-override
steps are ordinary steps, as in `baselines/ppo` (`RolloutMetrics` handles them
for the env metrics). Info fields (room, position, region) are never inputs to
the agent or the bonus; they are only logged.

## Run

```
# no game (pipeline check)
python -m baselines.ppo_rnd.train_ppo_rnd --fake --total_steps 8192

# real game, holding the shared lock for launch + run + kill (JIT compiles before the lock is taken)
python -m baselines.ppo_rnd.train_ppo_rnd --launch --kill_game --seconds 180 \
    --game_lock E:/projects/rainworld_rl-wt/.game-lock

# tests (no game)
python -m pytest baselines/tests -q -k ppo_rnd
```

MLflow experiment `rainworld-ppo-rnd` in the default store (`sqlite:///<repo>/mlruns/mlflow.db`).

## Flags

| flag | default | meaning |
|------|---------|---------|
| `--fake`, `--launch`, `--kill_game`, `--no_wipe`, `--game_lock` | off | env lifecycle, as in `baselines/ppo` |
| `--novelty` / `--no_novelty` | no novelty | include the hand-written NewRoom term in the extrinsic reward (off: intrinsic methods are not given the room bonus) |
| `--total_steps` | 200000 | training env steps (excl. warm-up) |
| `--seconds` | 0 (unlimited) | wall-clock budget for all env interaction incl. warm-up; stop at whichever comes first. A partial rollout at the deadline is discarded |
| `--rollout_steps`, `--epochs`, `--minibatches` | 1024, 4, 4 | PPO batch shape |
| `--gamma`, `--int_gamma`, `--gae_lambda` | 0.99, 0.99, 0.95 | gamma_E, gamma_I, lambda |
| `--clip`, `--clip_vloss`/`--no_clip_vloss` | 0.1, on | ratio clip; extrinsic value clipping with the same range |
| `--norm_adv`/`--no_norm_adv` | on | per-minibatch advantage normalisation |
| `--lr`, `--anneal_lr`/`--no_anneal_lr` | 1e-4, off | actor-critic Adam lr; linear anneal of both lrs over `--total_steps` |
| `--ent_coef`, `--vf_coef`, `--max_grad_norm` | 0.001, 0.5, 0.5 | loss weights, global grad-norm clip (policy + predictor) |
| `--hidden`, `--feature_dim` | 256, 448 | actor-critic widths |
| `--ext_death_terminal` | off | `player_dead` is terminal for the extrinsic head only |
| `--int_coef`, `--ext_coef` | 1.0, 2.0 | advantage weights |
| `--update_proportion` | 0.25 | predictor keep-probability per minibatch sample |
| `--rnd_lr` | = `--lr` | predictor Adam lr |
| `--rnd_rep_size`, `--rnd_hidden` | 512, 512 | RND output / predictor hidden width |
| `--obs_norm_init_steps` | 6400 | random-policy warm-up of the pixel stats |
| `--reward_scale` | 1.0 | extrinsic reward multiplier |
| `--frame_width`, `--frame_height`, `--frame_stack`, `--rgb`, `--ticks_per_step` | 96, 54, 4, off, 4 | observation and step size |
| `--seed` | random | JAX, fake-env and warm-up seed |
| `--log_interval`, `--ckpt_interval` | one rollout, 20000 | env steps between MLflow logs / checkpoints |
| `--mlflow_uri`, `--experiment`, `--run_name`, `--ckpt_dir` | repo store, `rainworld-ppo-rnd`, auto, auto | bookkeeping |

## What is logged

Per rollout (MLflow step = total env steps incl. warm-up):

| group | keys |
|-------|------|
| `ppo/` | `policy_loss`, `ext_value_loss`, `int_value_loss`, `entropy`, `approx_kl`, `clipfrac`, `grad_norm`, `total_loss`, `ext_explained_variance`, `int_explained_variance` |
| `rnd/` | `int_reward_raw_{mean,std,max}` (bonus before normalisation: the decay curve), `int_reward_norm_{mean,std,max}`, `int_return_std` (the divisor), `int_return_filtered_mean`, `predictor_loss`, `predictor_keep_frac`, `ext_adv_std`, `int_adv_std` (balance of the two streams), `ext_return_mean`, `int_return_mean`, `obs_rms_count`, `obs_std_mean` |
| `rollout/` | `deaths` (deaths in this rollout), `ext_reward_mean`, `ext_reward_sum` |
| `env/`, `reward_terms/` | `RolloutMetrics`, identical to the other baselines (`rooms_discovered`, `deaths`, ...) |
| `time/` | `rollout_seconds`, `intrinsic_seconds`, `update_seconds`, `env_steps_per_second`, `elapsed_seconds`, `warmup_seconds` |
| `final/` | cumulative env counters, updates, training/warm-up steps, seconds at the end of the run |

Params: all CLI args, parameter counts, `predictor_batch_effective`, `cleanrl_commit`, `run_id`, `ckpt_dir`, `rainworld_rl_path`.
The console prints one line per log (raw and normalised bonus, return std, predictor loss, both value losses, entropy, rooms, deaths, steps/s, update time).

## Performance and device notes

Parameters: actor-critic ~1.0M, predictor ~1.4M (trained), target ~0.9M
(fixed). On this machine's CPU one update over 1024 steps takes ~2 s. The
minibatch `lax.scan` is fully unrolled (`MINIBATCH_UNROLL`): on XLA:CPU the
same loop as a rolled while-loop took ~11 s (the existing PPO baseline's
update has the same issue: 6.2 s rolled vs ~1.7 s unrolled for its 16 steps).

The code is device-agnostic (nothing pins CPU): per env step there is one
jitted `act` call (rng split + policy forward) whose output is packed into a
single `(9 + 3,)` vector, i.e. one host->device frame upload and one
device->host transfer per step; rollout buffers and the RND running statistics
live on the host, and the whole rollout moves to the device once per update
(`prepare_batch`), followed by one jitted `intrinsic_reward` and one jitted
`train_step`. The device is logged as the `device` param.

## Smoke results (2026-10-04, CPU)

* `--fake --total_steps 8192` (defaults otherwise): 6400 warm-up + 8192
  training steps in 25 s after a 15 s JIT warm-up; ~1100 env steps/s, 1.9 s
  per update, all losses finite. Raw bonus 1273 -> 294 over 8 updates (fake
  frames are i.i.d. noise, so the predictor can only fit their mean).
* Live, one run: `--launch --kill_game --game_lock ... --seconds 180
  --obs_norm_init_steps 1024`. JIT compiled before the lock (15 s); game up
  in ~6 s; warm-up 1024 steps at 125 steps/s (mean pixel std 3.6: the start
  room is nearly static); 13 updates / 13312 training steps at ~102 env
  steps/s during rollouts, 2.2 s per update (~18 % of wall-clock). 2 rooms
  discovered (the second during the final, discarded partial rollout), 0
  deaths, extrinsic reward 0, entropy 6.238 -> 6.21. The raw bonus came in
  bursts (4456 in rollout 2, 900 in rollout 9, 1696 in rollout 13, the last
  just before the agent left the start room), each decaying to ~25-70 within
  3-4 rollouts. The normalised bonus was 1e-4 to 2e-2 per step because the
  return-std divisor (1.6e5-3.1e5) is inflated by those bursts.

## Open issues / tuning advice

* **Bonus scale vs extrinsic events.** The running variance of the filtered
  intrinsic return is a lifetime statistic (faithful to the reference), so
  early bursts keep the divisor large for a long time and the normalised bonus
  small. While the extrinsic reward is zero, per-minibatch advantage
  normalisation hides this; once deaths (-3) or food appear, `ext_coef * A_E`
  can dwarf `int_coef * A_I`. Watch `rnd/int_adv_std` vs `rnd/ext_adv_std`.
* **Decay speed.** A burst is consumed within ~3-4k steps. If the policy
  cannot follow it, lower `--rnd_lr` (first knob) or `--update_proportion`.
* **Warm-up.** Use the default 6400 warm-up steps (~1 min live) for real runs;
  1024 was only to fit the 3-minute smoke budget.
