# DCEO + Wayfarer representation (exploration baseline)

Deep Covering Eigenoptions (DCEO) with the representation components of
Wayfarer, adapted to the continuing Rain World env (pixels, one lifetime,
death = respawn, `MultiBinary(9)` keys).

* M. Klissarov, M. C. Machado. *Deep Laplacian-based Options for Temporally-Extended
  Exploration.* ICML 2023. https://proceedings.mlr.press/v202/klissarov23a.html
* E. M. Lintunen, M. C. Machado. *Mastering Atari 2600 Games with Discovered Options*
  ("Wayfarer"). arXiv:2610.03604v1, Oct 2026 (no code released).
* D. Gomez, M. Bowling, M. C. Machado. *Proper Laplacian Representation Learning*
  (ALLO). ICLR 2024. https://arxiv.org/abs/2310.10833

Reference code ported from:

| what | repository | commit | files |
|---|---|---|---|
| DCEO agent, option execution, option training | https://github.com/mklissa/dceo | `2fb85623898a7b3dc51c5ac9e42c808e3489b166` | `dopamine/jax/agents/full_rainbow/full_rainbow_dceo.py`, `configs/full_rainbow_dceo.gin`, README porting guide |
| ALLO loss, dual ascent, Delta-pair sampling | https://github.com/tarod13/laplacian_dual_dynamics | `b8156180133ab1d77e3ca03d7cefd96a4463ba6e` | `src/trainer/generalized_augmented.py`, `src/trainer/al.py`, `src/agent/episodic_replay_buffer.py` (`discounted_sampling`) |
| Wayfarer upgrades | paper only (Sec. 3.2, 3.3, App. D Table 1, App. I) | arXiv v1 | - |

```
baselines/dceo/
  train_dceo.py   trainer: args, TrainState, jitted update sub-steps, Learner, acting loop, logging, checkpoints
  networks.py     conv encoder (+LayerNorm), RepresentationNet (Laplacian + inverse dynamics), QNetwork (K heads), NaP
  allo.py         ALLO Lagrangian, dual ascent, symlog, eigenvalue estimates
  replay.py       circular single-frame replay, segment-aware stacking, task / option / pair samplers
  options.py      DCEO Algorithm 1 option execution, epsilon schedule
  checkpoints/    <run_id>/model_<step>.eqx + latest.eqx + model_config.json (gitignored)
baselines/tests/test_dceo_*.py
```

## Algorithm as implemented

Three networks, **each with its own encoder**, all trained off-policy from one
replay buffer; by default one gradient step of each every 4 env steps
(`--updates_per_step 0.25`).

1. **Representation network** (Wayfarer Sec. 3.3). Encoder psi -> Laplacian head
   phi (`--d 8` outputs, symmetric-log transformed) and an inverse-dynamics head chi.
   * ALLO on pairs `(x_t, x_{t+Delta})`, `P(Delta) ∝ 0.9^(Delta-1)` (`--gamma_rep`):
     `sum_i E[(u_i(x)-u_i(x'))^2] + sum_{j>=k} beta_jk (<u_j,[[u_k]]> - delta_jk) + b sum_{j>=k} (...)_1 (...)_2`
     with stop-gradients and two independent constraint batches as in the
     reference (by default the two halves of the pair starts, see deviations); barrier `b = 0.5` **fixed** (Wayfarer; the reference grows it),
     duals start at `-2.0` on the diagonal, updated by plain gradient ascent
     (`duals += 3e-5 * tril(errors)`, clipped to +/-100) - 10x slower than the
     encoder's Adam step size (3e-4), the ALLO reference's ratio.
   * Multi-step inverse dynamics on the same pairs: predict `a_t` from
     `psi(x_t) ⊕ psi(x_{t+Delta})`. With `MultiBinary(9)` this is 9 sigmoid
     cross-entropies summed (= `-log q(a_t)` under a factorised model), weight 1.0.
     Only the shared encoder receives both gradients.
   * Pairs follow **Wayfarer / ALLO (geometric Delta-step pairs, mean Delta ~ 10)**, not
     DCEO's: the DCEO paper uses consecutive pairs and its Atari code pairs `x_t` with the
     n-step next state `x_{t+3}` (`--gamma_rep 0` gives Delta = 1).
2. **Skill network** (Wayfarer Sec. 3.2). One encoder, `K = 2(d-1) = 14` dueling
   Q-heads over the 512 joint actions (`--option_directions positive` gives DCEO's
   one-directional `K = d-1`). Option `2(i-1)` / `2(i-1)+1` gets
   `+/-(u_i(x_{t+n}) - u_i(x_t))`, i.e. the telescoped n-step change of
   eigenfunction `i` (`i = 1..d-1`; the constant `u_0` gets no option), computed
   with the current representation inside the update. n-step (3) double-DQN
   targets with gamma 0.99, Huber loss; all heads every update, the encoder
   descends on the head-averaged loss.
3. **Main network**. Double DQN + n-step (3) + dueling over the 512 joint actions
   on the extrinsic drive reward (`DriveReward`, NewRoom term **off** unless
   `--novelty`). Target networks (main and skill) synced every 10,000 env steps.
4. **Exploration** (DCEO Algorithm 1, `options.py`). Each step: a running option
   terminates with prob `P_term = 0.1`; with no option running, with prob epsilon
   explore - launch a uniformly random option (prob `mu = 0.8`) that then acts
   greedily w.r.t. its head until termination, else a uniform random primitive -
   and otherwise act greedily w.r.t. the main network. Epsilon is 1.0 until
   `--min_replay` frames, then decays linearly over 250k steps to **0.05 and
   stays there** (continuing; no decay to 0.01). Options are never valued or
   chosen by the main network.

Action space: `index -> bits` with bit `k` = key `k` of `rainworld_rl.KEY_NAMES`
(`networks.index_to_bits` / `bits_to_index`).

Replay: single 96x54 grayscale uint8 frames in a circular buffer (1M frames ~ 5.2 GB),
4-frame stacks rebuilt at sample time. Per step it stores the frame, joint action,
reward, a `died` flag and a `cut` flag (see below).

## Continuing-env adaptations (all flags default ON)

* **No episodes.** Pairs, inverse-dynamics triples and option windows are only
  truncated at *cuts* (and at the newest frame); Delta is capped at `--max_delta 100`
  (P(Delta > 100) ~ 3e-5).
* **Death / respawn masking** (`--mask_death`, `--mask_not_ready`). A cut is placed
  after a frame with `info["player_dead"]` (the step *into* the death frame is a real
  transition and stays), and around every step with `info["ready"]` False (game-over
  reload, sleep screen, menus: the agent has no control and the respawn teleport
  happens there). Effects: no Laplacian / inverse-dynamics pair spans a cut; frame
  stacks never cross one (acting-side `StackBuilder` restarts the stack the same
  way); option n-step windows stop at the last frame before a cut with **no
  intrinsic bootstrap**; pair starts, option starts and constraint states never sit
  on a frame that ends a segment (isolated loading frames would otherwise form
  disconnected graph components, i.e. spurious zero-eigenvalue eigenfunctions); a
  running option is terminated at a cut. `--no_mask_death --no_mask_not_ready`
  ablates all of it (then nothing is ever cut).
* **Task head**: death is non-terminal by default (n-step windows cross deaths, like
  the PPO baseline); `--death_terminal` truncates the window after a death step.
* **Epsilon floor** 0.05 for the whole lifetime.
* **Plasticity**: LayerNorm before every ReLU in all three networks (`--layer_norm`);
  `--nap` adds Normalize-and-Project's weight projection (every LayerNorm-ed weight
  is rescaled to its initial norm after each step; LN gains/biases untouched).
* Running statistics are never reset (there are none besides Adam's).
* `ready` False / human-override steps are otherwise treated like every other step
  (as in the PPO baseline); `RolloutMetrics` handles them for the env metrics.

## Deviations from the references, and why

| vs | deviation | why |
|---|---|---|
| DCEO code | Main agent is double DQN + n-step + dueling + Huber; no C51 (51 atoms), no prioritized replay, no NoisyNets (DCEO's gin has `noisy = False` anyway) | CPU budget and simplicity; C51 on a 512-action head is 26k outputs, PER needs a sum tree over a 1M single-stream buffer. Extrinsic reward is sparse, the method's contribution is the options |
| DCEO code | Options: one shared skill encoder with K heads, all heads updated every step (DCEO: one full network per option, one random option trained per step) | Wayfarer's "unified skill network"; the brief asks for it |
| DCEO code | `K = 2(d-1)` options over `u_1..u_{d-1}` (DCEO: `num_options` one-directional options over `u_0..`, incl. the constant one) | Wayfarer convention; `--option_directions positive` for the one-directional variant |
| DCEO code | Laplacian learned with ALLO (DCEO: generalized graph drawing with `log1p` norm transform) | Wayfarer upgrade; ALLO removes rotation ambiguity |
| DCEO code | Exploration coins follow the paper's Algorithm 1; options act greedily (`--option_epsilon 0`) | The code uses the main net's own epsilon-greedy for the primitive branch and epsilon-greedy inside options; the paper / brief specify the clean version |
| DCEO code | lr 1e-4 (paper) with Adam eps 1.5e-4 (gin), global-norm clip 30 (Wayfarer); gin uses 6.25e-5 | brief: keep the doc's 1e-4 |
| DCEO code | Target sync 10k env steps (gin: 8k; Wayfarer: 10k); epsilon 1 -> 0.05 over 250k steps (gin: -> 0.01) | Wayfarer value; continuing floor |
| DCEO / Wayfarer | 512-way joint categorical over the 9 keys (Atari: 18 actions); inverse dynamics as 9-way multi-label BCE | MultiBinary(9); closest to the reference heads |
| DCEO / Wayfarer | Conv encoder with VALID padding on 96x54 frames, LayerNorm before every ReLU (Dopamine: SAME padding on 84x84, no LN) | house frame size (as PPO); LN for plasticity |
| ALLO code | Barrier `b` fixed at 0.5 (code grows it from 2.0), symlog outputs, diagonal duals start at -2 (code: 0) | Wayfarer Sec. 3.3 / Table 1. Table 1 says "ALLO dual initialisation -2.0"; we read it as the diagonal (off-diagonal multipliers start at 0) |
| ALLO code | Constraint batches = the two halves of the pair starts (`--separate_constraint_batches` restores two extra uniform batches) | Same distribution, still two independent batches (so the barrier stays unbiased), half the encoder cost on CPU |
| Wayfarer | Representation minibatch 64 pairs (`--rep_batch`; Wayfarer 512) | CPU: the rep step is the bottleneck (64: ~45 ms, 128: ~120 ms per step on this machine) |
| Wayfarer | Step sizes: Q nets 1e-4 (Wayfarer 5e-5, IQN), rep 3e-4 constant, duals 3e-5 constant. Wayfarer Table 1 lists `eta_phi = eta_chi = eta_beta : 3e-4 -> 3e-5`, which reads as one shared, decaying schedule | Brief: lr 1e-4 / 3e-4 / 3e-5, slower duals; a decaying step size is at odds with a continuing lifetime. If the eigenvalue estimates (`rep/eig_est_*`) stall, try `--dual_lr 3e-4` |
| Wayfarer | No IQN, no NoisyNets, no PER, no h-transform; options are **not** in the main network's action set | DCEO-style first run (brief). Learning option values in the main network is Wayfarer's phase-two change (future work) |
| everything | Updates run in 3 concurrent threads and one update ahead of acting (`--parallel_updates`, `--async_updates`) | Same maths per update (tested); XLA:CPU parallelises small convs poorly. Async: the agent acts with parameters at most one update (4 env steps) stale. `--no_async_updates --no_parallel_updates` is the plain synchronous loop |

## Run

From the repository (worktree) root:

```
# no game
python -m baselines.dceo.train_dceo --fake --total_steps 5000 --min_replay 1000

# real game, under the shared lock (launch + run + kill), 3-minute budget
python -m baselines.dceo.train_dceo --launch --kill_game --game_lock E:/projects/rainworld_rl-wt/.game-lock --seconds 180 --min_replay 2000

# tests
python -m pytest baselines/tests -q
```

JIT warm-up happens before the lock is taken. MLflow experiment: `rainworld-dceo`
(default store `sqlite:///<repo>/mlruns/mlflow.db`, as PPO). The device is logged as
the `device` param. Nothing pins the CPU: on a CUDA jaxlib the same code runs the
updates on the GPU (replay stays in host memory; one batch is transferred per
update; action selection is one jitted call per step); raise `--updates_per_step`
there.

## Flags (defaults)

| group | flag | default |
|---|---|---|
| env | `--fake`, `--launch`, `--kill_game`, `--no_wipe`, `--game_lock DIR` | off |
| env | `--frame_width`, `--frame_height`, `--ticks_per_step`, `--frame_stack` | 96, 54, 4, 4 |
| env | `--novelty` / `--no_novelty` (NewRoom term in the extrinsic reward) | **off** |
| env | `--reward_scale` | 1.0 |
| budget | `--total_steps`, `--seconds` | 1,000,000, none |
| options | `--d`, `--option_directions` | 8, both (K = 14) |
| options | `--p_term`, `--mu`, `--option_epsilon` | 0.1, 0.8, 0.0 |
| options | `--epsilon_final`, `--epsilon_decay_steps` | 0.05, 250,000 |
| Q-learning | `--gamma`, `--n_step`, `--lr`, `--adam_eps`, `--batch_size` | 0.99, 3, 1e-4, 1.5e-4, 32 |
| Q-learning | `--double_dqn`, `--dueling`, `--huber_delta`, `--death_terminal` | on, on, 1.0, off |
| representation | `--rep_lr`, `--rep_adam_eps`, `--dual_lr`, `--dual_init`, `--dual_clip`, `--barrier` | 3e-4, 1e-8, 3e-5, -2.0, 100, 0.5 |
| representation | `--rep_batch`, `--separate_constraint_batches` | 64, off |
| representation | `--symlog`, `--inv_weight`, `--gamma_rep`, `--max_delta` | on, 1.0, 0.9, 100 |
| continuing | `--mask_death`, `--mask_not_ready` (`--no_...` to ablate) | on, on |
| networks | `--hidden`, `--layer_norm`, `--nap`, `--max_grad_norm` | 512, on, off, 30 |
| schedule | `--replay_capacity`, `--min_replay`, `--updates_per_step`, `--target_update_period` | 1,000,000, 20,000, 0.25, 10,000 |
| schedule | `--parallel_updates`, `--async_updates` | on, on |
| bookkeeping | `--seed`, `--log_interval`, `--ckpt_interval`, `--ckpt_dir`, `--mlflow_uri`, `--experiment`, `--run_name` | random, 1000, 50,000, auto, repo `mlruns/`, `rainworld-dceo`, auto |

Boolean flags come in `--x` / `--no_x` pairs.

## What is logged (every `--log_interval` env steps)

| group | keys |
|---|---|
| `env/`, `reward_terms/` | `RolloutMetrics` (rooms discovered, deaths, cycles, food, karma, steps/s, ...), identical to the other baselines |
| `dceo/` | `main_loss`, `main_q`, `main_target`, `main_grad_norm`, `skill_loss`, `skill_q`, `skill_grad_norm`, `skill_bootstrap_frac` (option windows not cut), `updates_total` |
| `intrinsic/` | raw per-option reward `option_k_mean` / `option_k_std`, overall `mean`, `std`, `abs_mean` (DCEO does not normalise intrinsic rewards, so there is no normalised variant) |
| `rep/` | `total_loss`, `graph_loss`, `dual_loss`, `barrier_loss`, `orth_error` (norm of the constraint-error matrix), `inv_loss`, `inv_acc_key`, `inv_acc_exact`, `grad_norm`, `delta_mean`; per eigenfunction `graph_norm_i`, `eig_est_i` (= -dual_ii / 2), `dual_i`, `sq_norm_i` (<u_i, u_i>), `u_std_i` |
| `explore/` | `epsilon`, `frac_option`, `frac_random`, `frac_greedy`, `options_launched`, `options_finished`, `option_mean_duration`, `distinct_options` |
| `time/` | `env_seconds`, `update_wait_seconds`, `update_compute_seconds`, `update_seconds_mean`, `updates`, `env_steps_per_second`, `elapsed_seconds`; `replay/size` |

**Laplacian health signals.** At an ALLO equilibrium `eig_est_i` equals
`graph_norm_i` (both are 2x the eigenvalue of the Delta-pair Laplacian), `sq_norm_i`
is 1, `orth_error` is ~0, and the estimates increase with `i` (`eig_est_0` ~ 0 for the
constant eigenfunction). With duals initialised at -2 and fixed `b = 0.5` the
squared norms start inflated (up to 1 - beta/(2b) = 3) and settle as the slow duals
move; eigenfunction *directions* converge long before the scale (see
`test_dceo_allo.py`). Inverse-dynamics accuracy above chance (`inv_acc_key` > the
majority-class rate) means the encoder is picking up agent-controlled features.

## Throughput and smoke-test numbers (this machine: 16-core CPU, no GPU)

Per update (main + skill + representation, default sizes): main ~25 ms, skill
~45 ms, representation ~45 ms (rep_batch 64; ~120 ms at 128); ~95 ms with the three
in parallel threads (~120 ms sequential).

* `--fake`, 5,000 steps (`--min_replay 1000`): 38 env steps/s overall with learning,
  update compute 104 ms (overlapped with acting), 1,000 updates in 104 s. Fake frames
  are i.i.d. noise, so the representation numbers there mean nothing (inverse dynamics
  stays at chance, as it should).
* Live smoke, 180 s of interaction (`--launch --kill_game --game_lock ... --seconds 180
  --min_replay 2000 --epsilon_decay_steps 4000 --target_update_period 2000`): 7,111 env
  steps, 1,278 updates. ~95 env steps/s before learning starts, ~32-33 steps/s once
  updates run (98 ms per update in the background; the env + acting side slows from
  ~10 to ~30 ms/step because the update threads, the acting forward pass and the game
  share the CPU; the learner was never the one waited on). Rooms discovered 1, deaths 0
  (3 minutes, mostly epsilon ~ 1). Option share of steps 0.97 while epsilon = 1, 0.2-0.4
  at the 0.05 floor; mean option duration ~10 steps (P_term 0.1). Representation after
  1,278 updates: inverse-dynamics per-key accuracy 0.62 -> 0.73 (exact 9-key match
  0.19; chance ~0.5 / 0.002), graph norms already ordered (`u_0` 0.002, `u_1` 0.05,
  `u_2` 0.11, `u_3` 0.30, `u_4..7` 0.34-0.51), squared norms inflated to ~2.3-2.9
  and `eig_est_*` still ~0.96-0.99: the slow-dual transient described above.
  Per-option intrinsic reward |r| ~0.17 per 3-step window.
* On a synthetic pixel corridor (random walk of a square on a 20-cell strip with
  pixel noise; scratch check, not a unit test) the learned `u_1..u_3` correlate
  0.97-0.999 with the analytic eigenvectors after 500 updates. After 4,000 updates the
  squared norms are ~1.6-2.8 with `--dual_lr 3e-5` (default) and ~1.2-1.6 with
  `--dual_lr 3e-4` (eigenvector correlations identical), so the faster duals of
  Wayfarer's Table 1 shorten the scale transient without hurting the directions.

On a GPU, raise `--updates_per_step` and `--rep_batch` (Wayfarer: 512).

## Future work

* Wayfarer phase two: options as actions of the main network (option values with the
  fixed n-step target), IQN + NoisyNets + PER, h-transform instead of raw rewards.
* Branching (9 x 2) heads instead of the 512-way joint head if the joint head learns slowly.
* NaP on by default once a long run shows plateaus; reward centering for the task head.
* Larger representation batches (Wayfarer 512) once the learner runs on a GPU.
