# Exploration methods for Rain World RL: literature review synthesis

Date: 2026-10-04. Synthesised from five targeted reviews (full text, with verified
links, in `docs/litreview/`):

| File | Stream |
|---|---|
| `litreview/modelfree_robustness_continuing.md` | Model-free bonuses (RND, E3B, RLE, ...), robustness studies, continuing / no-reset RL |
| `litreview/world_models.md` | World-model-driven exploration (Plan2Explore, LBS, Director, PEG, Curious Replay, ...) |
| `litreview/autotelic.md` | Self-goal-setting agents with learned goal spaces (Skew-Fit, LEXA, Director, PEG, MoReFree, ...) |
| `litreview/competence.md` | Skill discovery and unsupervised RL (METRA, TLDR, HILP, FB, URLB, ...) |
| `litreview/machado_options.md` | Option discovery for exploration (eigenoptions, DCEO, Wayfarer arXiv:2610.03604) |

## 1. Constraints the shortlist was filtered against

- C1 General purpose: no game-specific knowledge, no hand-tuned action repeat, no
  hand-designed goal spaces or cell abstractions.
- C2 Continuing: one lifetime, death = respawn in place, no resets, no state restore.
- C3 Pixel observations only; large, continuous observation space.
- C4 No external knowledge: no LLMs/VLMs, no pretrained foundation models.
- Preference: published methods with reference code over novel combinations.

## 2. Findings that cut across every stream

1. **Nobody has published this setting.** There is no benchmark for continuing,
   pixel-based, no-reset exploration. Every autotelic method evaluates its goal
   selector at a fixed start state once per episode; every episodic bonus (E3B,
   NGU, NovelD, RIDE) resets memory at episode start; Go-Explore, asymmetric
   self-play and Latent Go-Explore are structurally episodic. Only RND, the
   imagination-trained Dreamer explorers, Director, DCEO and MoReFree are
   reset-free by construction. Any result here is new evidence, not a reproduction.
2. **Implementation details beat bonus choice.** RLeXplore (TMLR 2025): nearly all
   bonuses fail without intrinsic-reward normalisation; separate value heads beat
   summing rewards; no universal update proportion. Taiga et al. (ICLR 2020): with
   fair tuning, RND/ICM/pseudo-counts only beat epsilon-greedy on Montezuma.
   Craftax (2024): RND/ICM below PPO, E3B worse. Treat "bonus X works" claims as
   conditional on normalisation, two heads and adaptive weighting.
3. **Singleton world favours global novelty.** Henaff et al. (ICML 2023): a global
   (lifelong) bonus is near-optimal in fixed "singleton" MDPs and collapses as
   procedural contexts grow; episodic bonuses are the reverse. Rain World is a
   singleton world, so RND-style lifelong novelty is on firm ground as the control.
4. **Raw-pixel prediction error is the wrong currency here.** Rain World has heavy
   exogenous dynamics (rain, creatures, lighting). Disagreement vanishes on noise
   only asymptotically; Plan2Explore's own authors say so. The representations that
   win coverage studies are temporal-distance / temporal-contrastive (METRA, TLDR,
   HILP, C-TeC, ETD) or Laplacian (DCEO, Wayfarer), both invariant to pixel noise.
   Posterior-prior KL (LBS) and learning-progress bonuses (LPM, ICLR 2026) are the
   general-purpose noise-robust bonus variants.
5. **Death is a graph shortcut.** Every method that scores a transition
   (disagreement, Laplacian smoothness, temporal-distance constraint, intrinsic
   bootstrap) will treat death -> respawn as a large, partly stochastic jump. An
   option or skill can learn to die to teleport. Standard fix, generic rather than
   game-specific: use the `death` edge in `info` as a trajectory boundary for
   representation losses and intrinsic bootstrapping, and decide deliberately
   whether the extrinsic head treats death as terminal. Run with/without as an
   ablation and report it.
6. **Continuing-task discounting.** Wan et al. (2025) show gamma = 0.999 is fragile
   in continuing tasks with reward offsets; reward centering fixes it. RND's own
   intrinsic gamma of 0.99 should not be raised.
7. **One-lifetime replay.** Uniform replay over a single long stream is dominated by
   old data. Curious Replay (Kauvar 2023) is the published, hyperparameter-insensitive
   fix for Dreamer-family agents; Continual-Dreamer found reservoir sampling matters.
8. **MultiBinary(9) is unaddressed by every paper.** Options: flatten to a 512-way
   categorical head (closest to Atari reference code) or factorise into 9 Bernoulli
   heads (what the PPO baseline already does). Inverse-dynamics heads become 9-way
   multi-label sigmoid cross-entropy.
9. **Evaluation.** Measure coverage over wall-clock lifetime (distinct rooms,
   position bins, regions) from `info`, with the `NewRoom` novelty term switched off,
   so intrinsic methods are not graded on a hand-written novelty bonus. Keep the
   drive reward as the "extrinsic" signal. Log the intrinsic/extrinsic ratio.

## 3. Shortlist

Ordered by how soon to run them. Each entry: what it is in the taxonomy, algorithm,
continuing adaptations, code, main risk.

### Tier 1: controls that run on the current CPU PPO stack

**1. PPO + RND (Burda et al., ICLR 2019).** Novelty bonus, lifelong.
- Signal: MSE between a fixed random target CNN and a trained predictor on the
  single latest frame; divided by running std of the discounted intrinsic return.
- Two value heads; intrinsic head NON-episodic (never zero its bootstrap at
  death), extrinsic head episodic or continuing by deliberate choice.
  gamma_I = 0.99, gamma_E = 0.999 (consider lower or reward centering),
  int_coef 1 / ext_coef 2 on advantages, predictor/target inputs whitened per
  pixel and clipped to [-5, 5] with random-policy warm-up, policy sees raw frames,
  predictor trained on a fixed effective batch (keep-prob 0.25 at 128 envs),
  lr 1e-4, entropy 0.001, clip 0.1.
- Decay speed of the bonus (update proportion / predictor lr) is the first thing
  to tune in a continuing world.
- Code: CleanRL `ppo_rnd_envpool.py` (faithful; PyTorch),
  https://github.com/openai/random-network-distillation (archived TF). The
  Craftax_Baselines JAX RND is NOT faithful (no normalisation, one gamma).
- Risk: exogenous stochasticity attracting the bonus; fallback predictors are
  Disagreement or LPM.

**2. Random Latent Exploration, RLE (Mahankali et al., ICML 2024).** Bonus-free
structured exploration control.
- Samples a random latent z, adds a random reward z^T phi(s) with a fixed random
  phi, trains a z-conditioned policy; z resampled on a fixed step schedule instead
  of episode boundaries. Only method with a statistically reliable gain over PPO
  across all 57 Atari games (RND and NoisyNet were not).
- Code: https://github.com/Improbable-AI/random-latent-exploration (CleanRL based).
- Answers "does any structured exploration help, or specifically a novelty bonus?"

**Optional 2b. E3B x RND (Henaff et al., ICML 2023).** Episodic elliptical bonus
multiplied into RND; most robust across singleton/procedural regimes. Needs a
pseudo-episode: reset covariance on respawn or on a fixed window (1k-4k steps).
Code: https://github.com/facebookresearch/e3b, RLeXplore. E3B's own lifetime
variant is much worse than episodic E3B, so the window is load-bearing.

### Tier 2: the methods the project exists to test

**3. DCEO, Deep Covering Eigenoptions (Klissarov & Machado, ICML 2023), upgraded
with Wayfarer's representation (Lintunen & Machado, arXiv:2610.03604).**
Temporally extended exploration from a learned Laplacian representation; the
single pick from the option-discovery line.
- Three networks from one replay buffer: (a) representation encoder with a
  Laplacian head (d eigenfunctions; ALLO loss with fixed barrier b, dual init -2,
  symmetric-log outputs) plus a multi-step inverse-dynamics head (Delta ~
  Geom(0.9), weight 1.0) so exogenous motion is ignored; (b) K = 2(d-1) option
  policies, option i rewarded by +/-(f_i(x_{t+1}) - f_i(x_t)), trained off-policy;
  (c) main agent over primitives, which on exploratory steps launches a random
  option with prob mu, terminating each step with prob P_term.
- Continuing adaptations: drop episode truncation for pair sampling; mask
  (death, respawn) pairs from both representation losses and intrinsic bootstrap;
  continuing Q-learning or reward centering for the task head; keep epsilon >= 0.05
  for life; LayerNorm + Normalize-and-Project for plasticity; 512-way joint head
  first, branching head second; Atari/Impala encoder.
- Start: d = 8-10, P_term in {0.05, 0.1, 0.2}, mu = 0.8, lr 1e-4 (3e-4 encoder /
  3e-5 duals), replay 1M, gamma 0.99, n-step 3, b sweep {0.5, 2, 5}.
- Code: https://github.com/mklissa/dceo (official, JAX + Dopamine, porting guide
  in README); ALLO loss from https://github.com/tarod13/laplacian_dual_dynamics.
  Wayfarer code is not released. Third-party DCEO in
  https://github.com/bath-reinforcement-learning-lab/jaxhrl (unverified).
- Risk: Laplacian learning from pixels at open-world scale (evidence is MiniGrid,
  MiniWorld and one Atari game; Wayfarer is 10 Atari games at d = 6);
  eigenfunctions rotate as the frontier moves; memoryless termination is the only
  reset-free option scheme with deep evidence. Fallback: the AAAI 2020
  successor-representation count bonus, one extra head.

**4. METRA, then TLDR (Park, Rybkin, Levine, ICLR 2024; Bae et al., CoRL 2024).**
Competence / temporal-distance representation; skills first, far-goal selection second.
- METRA: reward (phi(s') - phi(s))^T z with ||phi(s) - phi(s')|| <= 1 enforced by a
  Lagrangian on one-step transitions (eps 1e-3, lambda_0 30); z uniform on the
  unit sphere (2-4-D) or discrete (16/24); SAC, UTD 1/16; 64x64 pixels validated
  without proprioception.
- Known limit: covers along a few primary directions, loses in mazes (LEADS:
  54.8% hard-maze coverage; TLDR far ahead on AntMaze-Large/Ultra). Rain World is
  a room graph, so expect this failure.
- TLDR: same phi, adds kNN-entropy selection of far goals in phi-space plus two
  per-transition rewards. Best large-structured-world coverage evidence within
  C1-C4. Adaptation: pick a far goal every K steps from the current state, switch
  to the exploration reward on success/timeout.
- Continuing adaptations: resample z every K steps (K in {64,...,512}); exclude
  death/respawn and skill-switch pairs from the Lipschitz constraint and critic
  bootstrap; continuing discount; factorised-Bernoulli SAC or PPO backbone (D3
  shows the objective works with PPO).
- Code: https://github.com/seohongpark/METRA (MIT, PyTorch + garage, dependency
  rot), https://github.com/heatz123/tldr (same codebase, pixel envs, METRA baseline).
- Risk: latent collapse when lambda saturates; fast-movement bias translating into
  reckless deaths; no game-pixel results anywhere in this line.

**5. Plan2Explore head on DreamerV3, with and without Curious Replay (Sekar 2020;
Kauvar 2023).** Knowledge gain inside a world model. Needs a GPU.
- Ensemble of 8-10 heads predicting the next latent from (deter, stoch, action);
  intrinsic reward = ensemble std (mean over dims), normalised; actor-critic trained
  in 15-step imagination with a learned continuation head, so the intrinsic return
  is non-episodic by construction. Mixed reward: extr 1.0, disag 0.1 (2023 JAX
  defaults). Try `disag_log` and the paper's embedding target vs the codebases'
  `stoch` target. LBS (posterior-prior KL) is the noise-robust swap-in.
- Curious Replay: count-and-loss replay priority; orthogonal, hyperparameter
  insensitive, best general Crafter gain (19.4 vs 14.5). Run as a 2x2 with P2E.
- Documented weaknesses: reward-free P2E 2.1% vs random 1.6% on Crafter; "too
  chaotic actions" (Director); 300k+ steps to touch a novel object (Curious
  Replay). It is the universal baseline, not a winner.
- Code: the current `danijar/dreamerv3` main has NO exploration code (dropped in
  the 2024 rewrite). Use the June 2023 JAX commit (`expl_behavior`,
  `expl_rewards`), https://github.com/NM512/dreamerv3-torch (`plan2explore`),
  or https://github.com/Eclectic-Sheep/sheeprl `p2e_dv3` after PR #361
  (multi-discrete actions; earlier versions had real bugs). Curious Replay:
  https://github.com/AutonomousAgentsLab/cr-dv3. LBS: https://github.com/mazpie/lbs-exploration.

### Tier 3: self-goal-setting agents (largest engineering cost)

**6. Director (Hafner et al., NeurIPS 2022).** The only non-LLM self-goal-setting
agent demonstrated from pixels in open-world games (Crafter, DMLab, Atari,
egocentric Ant Maze) with one hyperparameter set.
- Manager picks a goal in a discrete goal-VAE latent of the world model every K
  steps from the current state; worker rewarded by cosine similarity to the goal;
  manager rewarded by extrinsic return plus goal-VAE reconstruction error
  (novelty). Reset-free by design. Consider swapping the goal-AE bonus for P2E
  disagreement, already present in Dreamer codebases.
- Code: https://github.com/danijar/director (TF2, single-commit snapshot);
  Hieros https://github.com/Snagnar/Hieros is a PyTorch stand-in. Watch the
  Schiewer et al. (Sci. Rep. 2024) model-exploitation failure in hierarchies.

**7. PEG -> MoReFree (Hu et al., ICLR 2023; Yang et al., TMLR 2025).** Planned
exploratory goals on the same world model as 5/6; MoReFree is the only published
reset-free go/explore recipe. Both are state-based in their papers, so this is a
research step. Code: https://github.com/penn-pal-lab/peg,
https://github.com/yangzhao-666/MoReFree. Dr. Strategy
(https://github.com/ahn-ml/drstrategy) is the landmark-based pixel alternative
(67% vs LEXA 9.6% on a 25-room maze).

**Model-free alternative to 6:** learned latent + density-skewed goal sampling +
HER with a contrastive temporal critic (Skew-Fit / MEGA / LGE lineage).
https://github.com/qgallouedec/lge (SB3, image obs, discrete actions) for the
sampler and HER plumbing. Adaptations: goal switch on success/timeout, density
over a sliding window, reachability filter V(s_t, g).

## 4. Excluded, with reasons

- Go-Explore, Latent Go-Explore, asymmetric self-play: structurally episodic (C2).
- ICM / raw prediction error, BYOL-Explore: noisy-TV exposure; BYOL-Explore has no
  official code and needs a V-MPO-scale stack.
- DIAYN, DADS, VIC, APS, CIC and other mutual-information skill methods: do not
  produce coverage (URLB, LSD, CIC, METRA Fig. 3).
- FB / HILP / zero-shot RL: offline, do not solve exploration; degrade from pixels.
  HILP is a possible later offline layer over the accumulated replay buffer.
- Empowerment: no scaled pixel results, no maintained code.
- SENSEI, LS-Imagine, DLLM, ELLM, OMNI, MAGELLAN, Voyager, Motif, L-AMIGo: LLM/VLM
  priors (C4).
- Goal GAN, Setter-Solver, AMIGo, SVGG: adversarial / intermediate-difficulty
  generators are the most brittle family; AMIGo failed on hard MiniHack without
  language.
- Action repeat / epsilon-z-greedy: hand-chosen temporal prior (C1); note DCEO
  uses it only as a baseline.

## 5. Goal-selection criteria, for later ablations

No pixel-scale head-to-head exists. Density / entropy of achieved goals (Skew-Fit,
MEGA, LGE, TLDR) is the most repeatedly validated simple criterion; exploration
value / value uncertainty (PEG, DISCOVER) wins the controlled state-based
comparisons; learning progress (IMGEP, CURIOUS, LPM) is the only criterion with a
noise-robustness argument but has no deep-GCRL-from-pixels win. Rain World can
actually inform this question.
