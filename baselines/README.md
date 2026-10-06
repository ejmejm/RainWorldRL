# Baselines

Reference agents for the Rain World RL environment: PPO, plus the exploration
methods chosen in `docs/EXPLORATION_METHODS.md` (PPO + RND, DCEO). They
exist to (a) prove the env is trainable end to end and (b) give every later
exploration / intrinsic-motivation method a number to beat on the same
metrics. Stack: JAX (CPU) + Equinox + Optax + MLflow + tqdm.

```
baselines/
  common/env.py         make_env(): RainWorldEnv -> default reward -> grayscale -> frame stack; FakeRainWorldEnv
  common/metrics.py     RolloutMetrics: rooms discovered, deaths, cycles, food, karma, reward terms, steps/s
  common/game_lock.py   mkdir lock so only one process drives the single game instance
  common/runner.py      shared evaluation loop (play / random agent)
  ppo/train_ppo.py      PPO trainer (continuing env, MultiBinary Bernoulli policy)
  ppo/play.py           run a checkpoint, greedy or sampled, print the same metrics
  ppo/checkpoints/      <run_id>/model_<step>.eqx + latest.eqx + model_config.json (gitignored)
  random_agent.py       uniform random keys through the same env + metrics
  ppo_rnd/              PPO + Random Network Distillation (Burda et al. 2019; port of CleanRL ppo_rnd_envpool) - see ppo_rnd/README.md
  dceo/                 Deep Covering Eigenoptions (Klissarov & Machado 2023) + Wayfarer representation (ALLO + inverse dynamics) - see dceo/README.md
  tests/                game-free pytest suite (uses FakeRainWorldEnv)
```

## Install

```
python -m pip install -e .                                   # the env itself (numpy, gymnasium), from the repo root
python -m pip install --user -r baselines/requirements.txt   # jax (CPU), equinox, optax, mlflow, tqdm
```

Always run from the **repository root** with `python -m ...` so the checkout's
own `rainworld_rl` is the one imported (the cwd is first on `sys.path`; the
scripts log `rainworld_rl.__file__` at startup so you can check).

## Run

```
# no game, smoke-test the pipeline
python -m baselines.ppo.train_ppo --fake --total_steps 2048 --rollout_steps 256

# real game: (re)start Rain World with the deployed mod DLL, train, close the game afterwards
python -m baselines.ppo.train_ppo --launch --kill_game --total_steps 200000

# attach to a game that is already running with the mod (no launch)
python -m baselines.ppo.train_ppo --total_steps 50000

# play a checkpoint (greedy = press every key with p > 0.5; --sample to sample)
python -m baselines.ppo.play --ckpt baselines/ppo/checkpoints/<run_id>/latest.eqx --steps 2000 --launch --kill_game

# random-agent comparison
python -m baselines.random_agent --steps 1000 --launch --kill_game

# tests (no game needed, ~1-2 min, mostly JAX compile)
python -m pytest baselines/tests -q
```

`--game_lock DIR` (all three scripts) holds an `mkdir`-based lock around the
launch + run + kill so two worktrees never fight over the one game instance.

MLflow writes to `mlruns/mlflow.db` (SQLite) at the repo root (gitignored;
MLflow >= 3.16 refuses the old plain-folder file store by default, so the
default URI is `sqlite:///<repo>/mlruns/mlflow.db`; override with `--mlflow_uri`).
Browse with

```
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

and open http://127.0.0.1:5000. Experiments: `rainworld-ppo` (training),
`rainworld-ppo-eval` (`play.py --mlflow`), `rainworld-random` (`random_agent --mlflow`).

## What is logged

Per rollout (every `--log_interval` env steps, default one rollout):

| group | keys |
|-------|------|
| `ppo/` | `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clipfrac`, `grad_norm`, `explained_variance`, `total_loss` |
| `env/` | `reward_mean` (per step, this window), `reward_sum`, `reward_total`, `rooms_discovered` (cumulative unique `(region, room_index)`), `new_rooms`, `deaths` (+ `_window`), `cycles_survived` (+ `_window`), `food_eaten` (+ `_window`, sum of positive food deltas within a cycle), `karma`, `steps_per_second`, `total_steps` |
| `reward_terms/` | each component of `info["reward_terms"]` summed over the window; `reward_terms_total/` cumulative |
| `time/` | `rollout_seconds`, `update_seconds`, `env_steps_per_second` (rollout only; `env/steps_per_second` is wall-clock over the whole window, i.e. including the update) |

All CLI args are logged as params, plus `run_id`, `ckpt_dir`, `param_count`,
`rainworld_rl_path`, `jax_backend`. `play.py` and `random_agent.py` print the
`env/` + `reward_terms/` block every `--print_every` steps.

## Design notes

* **Continuing environment.** Rain World never terminates: death respawns the
  slugcat, and `terminated`/`truncated` are always False. The trainer has no
  episode boundaries at all. Each rollout of `--rollout_steps` is one slice of
  one endless trajectory; GAE bootstraps from the value of the observation after
  the last step; `player_dead` is *not* treated as terminal (it is just an info
  edge that the metrics count). Resets happen once at startup (`env.reset()`
  wipes the RL save; `--no_wipe` attaches to the game in progress).
* **Observation.** Frames are 96x54 (16:9, the game's aspect), grayscale by
  default (`--rgb` to keep colour), stacked 4 deep, channel-first, and stay
  `uint8` in the rollout buffers; the `/255` happens inside the model.
* **Policy.** The native action space is `MultiBinary(9)` (raw keys). The
  policy head outputs one logit per key and treats the keys as independent
  Bernoullis: `log_prob` and entropy are sums over keys, sampling draws each
  key separately, greedy presses every key with `p > 0.5`. Any combination
  (including none) is representable, which the game needs (e.g. `down+right`
  to crawl, `jump` to dismiss prompts).
* **PPO.** Clipped surrogate, optional value clipping (`--clip_vf`), entropy
  bonus, per-minibatch advantage normalisation, Adam (eps 1e-5) with global
  grad-norm clipping, orthogonal init. Rollout collection is a Python loop
  calling a jitted `sample_action`; the update is a single jitted
  `train_step` whose `epochs x minibatches` gradient steps run in a
  `jax.lax.scan` over the minibatch index array.
* **Reward.** Whatever `rainworld_rl.rewards.make_default_reward_env` returns;
  the trainer only relies on the scalar reward and the `info["reward_terms"]`
  dict (logged per component). `--reward_scale` multiplies the reward before
  GAE.
* **JAX on CPU or GPU.** There is no CUDA jaxlib for native Windows, so there
  JAX runs on CPU. The env also runs on Linux and WSL2 (see the top-level
  README), where the learner can use a GPU build of JAX.
* **Checkpoints.** `eqx.tree_serialise_leaves` every `--ckpt_interval` env
  steps (and at the end / on Ctrl-C) into `baselines/ppo/checkpoints/<run_id>/`,
  next to a `model_config.json` that `play.py` uses to rebuild the model and
  matching env.

## Knobs

| flag | default | meaning |
|------|---------|---------|
| `--total_steps` | 200000 | env steps to train for |
| `--rollout_steps` | 1024 | steps per rollout / update |
| `--epochs`, `--minibatches` | 4, 4 | PPO passes and minibatches per rollout |
| `--gamma`, `--gae_lambda` | 0.99, 0.95 | discount, GAE lambda |
| `--clip`, `--clip_vf` | 0.2, 0 (off) | policy / value clipping |
| `--lr`, `--ent_coef`, `--vf_coef`, `--max_grad_norm` | 2.5e-4, 0.01, 0.5, 0.5 | optimiser and loss weights |
| `--hidden` | 256 | hidden layer after the conv torso |
| `--reward_scale` | 1.0 | reward multiplier before GAE |
| `--frame_width`, `--frame_height`, `--frame_stack`, `--rgb`, `--ticks_per_step` | 96, 54, 4, off, 4 | observation and step size |
| `--log_interval`, `--ckpt_interval` | 1024, 20000 | env steps between MLflow logs / checkpoints |
| `--seed` | random | JAX + fake-env seed |
| `--fake`, `--launch`, `--kill_game`, `--no_wipe`, `--game_lock` | | env lifecycle (see Run) |
| `--mlflow_uri`, `--experiment`, `--run_name`, `--ckpt_dir` | `mlruns/`, `rainworld-ppo`, auto, auto | bookkeeping |

Known limitations: a single env instance (the mod supports one game); the
default reward is sparse (surviving a cycle), so 200k steps of PPO is a
pipeline check rather than a result - the exploration methods are what this
baseline is meant to be compared against.
