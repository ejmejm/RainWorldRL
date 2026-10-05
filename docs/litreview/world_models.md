# Literature review: exploration driven by learned world models (model-based intrinsic motivation)

Prepared 2026-10-04 for a pixel-observation, continuing (no resets; death = respawn in place), open-world environment (Rain World) with a 9-key MultiBinary action space.

Hard constraints applied throughout:
- no game-specific tricks (no hand-tuned action repeat, hand-designed goal spaces, or cell representations);
- must run in a continuing, non-episodic setting (no Go-Explore state restore; methods that assume resets are flagged with the adaptation needed);
- must work from large, continuous pixel observations;
- no external knowledge (no LLM/VLM, language priors, or pretrained foundation models).

Verification policy. Every arXiv ID, venue and repository URL below was fetched during this review (arXiv abstract/HTML or ar5iv render, GitHub page, or the paper PDF). Where a detail could not be verified from a primary source it is marked "(not verified)". Numbers are quoted from the papers themselves. I did not find any paper literally titled "Dreaming to Explore"; that phrase is treated as shorthand for the Dreamer/Plan2Explore lineage, and the closest named relatives (LWM, GoBI, LEXA) are covered.

---

## 0. Quick reference: how the Dreamer codebases expose exploration

This matters because most "world-model exploration" in practice means "Plan2Explore head inside a Dreamer codebase", and the defaults differ from the paper.

| Codebase | Exploration switch | Default intrinsic target | Reward mixing | Notes |
|---|---|---|---|---|
| `danijar/dreamerv2` (TF2) — `dreamerv2/expl.py`, `configs.yaml` | `expl_behavior: greedy` (options: `greedy`, `Plan2Explore`, `ModelLoss`, `Random`) | `disag_target: stoch`, `disag_models: 10`, `disag_action_cond: True`, `disag_log: False` | `expl_intr_scale: 1.0`, `expl_extr_scale: 0.0` (pure intrinsic by default) | Intrinsic reward = `tf.tensor(preds).std(0).mean(-1)` (std over ensemble, mean over dims), then streaming normalisation. Separate actor-critic trained in imagination on the intrinsic reward. `ModelLoss` = single-head prediction-error curiosity. |
| `danijar/dreamerv3` (JAX, 2023 release; verified at commit `8fa35f83`, June 2023) | `expl_behavior: None` | `disag_target: [stoch]`, `disag_models: 8`, `disag_head` 5x1024 SiLU MLP with inputs `[deter, stoch, action]` | `expl_rewards: {extr: 1.0, disag: 0.1}` | Mixed extrinsic + 0.1 x disagreement by default when enabled. `horizon: 333` (gamma ~ 0.997), `imag_horizon: 15`. |
| `danijar/dreamerv3` (JAX, current `main`, 2024+ rewrite) | none | — | — | Fetched `configs.yaml` on `main`: no `expl*`/`disag*` keys remain. Plan2Explore was dropped in the rewrite. Use the 2023 commit, `cr-dv3`, or a third-party port if you want it. |
| `NM512/dreamerv3-torch` (PyTorch) — `exploration.py`, `configs.yaml` | `expl_behavior: 'greedy'` (option `plan2explore`), `expl_until: 0` | `disag_target: 'stoch'`, `disag_models: 10`, `disag_layers: 4`, `disag_units: 400`, `disag_action_cond: False`, `disag_log: True` | `expl_intr_scale: 1.0`, `expl_extr_scale: 0.0` | Intrinsic = `torch.mean(torch.std(preds, 0), -1)`, optionally `log`. `discount: 0.997`. |
| `Eclectic-Sheep/sheeprl` (PyTorch/Lightning Fabric) — `p2e_dv1`, `p2e_dv2`, `p2e_dv3` | separate algorithms | `p2e_dv2/dv3` predict the posterior `stoch` state; `p2e_dv1` predicts the image embedding (issue #322 flags this inconsistency with the paper) | separate exploration actor and task actor | Supports continuous, discrete and multi-discrete actions and pixel obs. Several correctness bugs were fixed in 2024-25 (PR #346, #358, #361: evaluation ran the untrained exploration actor; intrinsic reward computed on detached trajectories so its gradient never reached a continuous-action exploration actor; ensemble backward crashed under mixed precision). Treat pre-fix results with suspicion. |

Important discrepancy: the Plan2Explore paper trains the ensemble to predict the next image embedding h_{t+1} and uses the variance of ensemble means as reward; all Dreamer codebases default to predicting the stochastic latent (`stoch`) and use the std. Both work in practice, but they are different signals (predicting `stoch` is cheaper and lower-dimensional; the embedding target is closer to "information about the observation").

Continuing-environment note for all Dreamer-family explorers: the explorer's return is computed over 15-step imagined rollouts bootstrapped with a learned critic and discounted via the learned `cont` predictor. In an environment that never terminates, `cont` -> 1 and the intrinsic value is non-episodic by construction (gamma = 0.997, effective horizon ~333). Nothing in the imagination-training loop needs an episode boundary. The practical hazards are instead: (i) the respawn transition after death is a large, partly stochastic discontinuity that an ensemble will flag as "novel" for a long time (see Section 3 on disagreement vs stochasticity); (ii) replay sequences cross the death/respawn boundary, which the RSSM must learn to model. Both are environment properties you will have to characterise empirically, not method defects per se.

---

## 1. Method reports

Template: (1) identity; (2) algorithm; (3) episodic assumptions / continuing-env behaviour; (4) observations and benchmarks; (5) robustness / critiques; (6) code; (7) fit.

### 1.1 VIME — Variational Information Maximizing Exploration (lineage root)

1. Houthooft, Chen, Duan, Schulman, De Turck, Abbeel. NeurIPS 2016. arXiv:1605.09674.
2. Learns a Bayesian neural network dynamics model p(s'|s,a; theta) via variational inference. Intrinsic reward = information gain about dynamics parameters = KL(posterior after seeing (s,a,s') || posterior before), approximated locally. Added to the extrinsic reward with a scaling eta and median normalisation; policy is model-free (TRPO) on real data. The model is used only to score transitions.
3. Episodic in the original experiments but the signal itself is per-transition and non-episodic; nothing requires resets.
4. Low-dimensional state only (MuJoCo locomotion, continuous control). Never validated from pixels.
5. Direct successor comparisons: LBS (Mazzaglia 2022) reports that Bayesian surprise in a latent space outperforms VIME-style parameter-space information gain; disagreement ensembles (MAX, Plan2Explore, MaxInfoRL) are the practical modern approximation of this quantity.
6. `https://github.com/openai/vime` (verified; Theano/rllab-era code, effectively unmaintained).
7. Excluded as-is (states only, obsolete code). Relevant as the conceptual ancestor of every disagreement/information-gain method below.

### 1.2 MAX — Model-Based Active Exploration

1. Shyam, Jaskowski, Gomez. ICML 2019 (PMLR v97). arXiv:1810.12162.
2. Ensemble of probabilistic forward models trained on real transitions. Utility of (s,a) = disagreement among the members' next-state distributions, measured as Jensen-Shannon divergence (discrete) or Jensen-Renyi divergence (continuous Gaussians). An "exploration MDP" is built from the ensemble and an exploration policy (SAC) is trained inside it, then rolled out in the real environment to gather data. The design distinguishes learnable uncertainty from noise: "noise manifests as confusion among all models and not as a conflict".
3. Episodic experiments, but the objective is per-transition and the exploration policy is re-planned repeatedly; no reset dependence in the algorithm. Authors note second-order effects (utilities change as the model updates) are not modelled.
4. State observations only: Chain (discrete), Half Cheetah, Ant Maze. "MAX explores 100% of the transitions in around 15 episodes while baseline methods reach 40% in 60 episodes."
5. Trades compute for data efficiency; never scaled to pixels. Plan2Explore (2020) is essentially MAX re-done in the latent space of a learned pixel world model and is the reference point that superseded it.
6. `https://github.com/nnaisense/max` (verified; PyTorch 1.0, MuJoCo 1.50, Sacred; 81 stars; unmaintained).
7. Excluded as-is (no pixel path, obsolete stack). Use Plan2Explore instead; MAX is the citation for the JSD/JRD disagreement formulation.

### 1.3 Plan2Explore (P2E)

1. Sekar, Rybkin, Daniilidis, Abbeel, Hafner, Pathak. ICML 2020 (PMLR v119). arXiv:2005.05960.
2. World model = Dreamer RSSM (encoder, posterior/prior dynamics, decoder). A bootstrap ensemble of K one-step MLPs each "takes a model state s_t and action a_t as input and predicts the next image embedding h_{t+1}". Intrinsic reward D(s_t,a_t) = Var({mu(w_k, s_t, a_t)}_k), i.e. variance across ensemble means (averaged over dims), interpreted as expected information gain. "The exploration policy is optimized purely from trajectories imagined under the model to maximize the intrinsic rewards computed by the model itself." Key idea: planning for expected future novelty (foresight) rather than rewarding novelty after the fact (retrospective). A separate task policy can be trained zero-shot in imagination from the same model once a reward is available.
3. Experiments are episodic (DMC, 1000-step episodes), but the method is reset-agnostic: imagination-trained explorer, per-step intrinsic reward, bootstrapped critic. Runs unchanged in a continuing env (see Section 0 note on `cont`). The paper itself states the known limitation: "The disagreement is positive for novel states, but given enough samples, it eventually reduces to zero even for stochastic environments because all one-step predictions converge to the mean" - which is the intended behaviour (not a noisy-TV attractor) but only in the limit; in finite data, stochastic transitions (death/respawn) will be rewarded for a while.
4. Pixels. 20 DeepMind Control tasks. Zero-shot (3.5M steps): P2E 563.6 vs Curiosity/ICM 489.3 vs Dreamer oracle 694.8 (averaged); few-shot (1M + 150K): 643.7 vs 538.0 vs 700.3. Ensemble of 5 members, 2 hidden layers, predicting 1024-d encoder features. Worlds are small, fixed-scene control tasks, not open worlds.
5. Evidence of brittleness on open-world/object tasks, all from primary sources:
   - Crafter benchmark paper (Hafner 2021, arXiv:2109.06780): reward-free P2E score 2.1 +- 0.1 % vs Random 1.6 +- 0.0 % at 1M steps (RND 2.0 %). Barely above random in an open world (older DreamerV1-era implementation).
   - Director (Hafner 2022): "Plan2Explore fails in the small maze because the robot flips over too much, a common limitation of low-level exploration methods"; "Plan2Explore chooses too chaotic actions"; struggles at 5-6 pads in Visual Pin Pad.
   - Curious Replay (Kauvar 2023): in a novel-object assay "Plan2Explore agents do not quickly interact with the object" (~300K+ steps vs ~50K with CR).
   - SENSEI (Sancaktar 2025): on Robodesk "Plan2Explore mostly performs arm stretches"; "Plan2Explore does not outperform learning a task policy from scratch with DreamerV3 consistently across environments".
   - Continual-Dreamer (Kessler et al., CoLLAs 2023, arXiv:2211.15944): DreamerV2+P2E on pixel MiniGrid/MiniHack - helped MiniGrid, mixed in MiniHack, sometimes increased forgetting depending on replay strategy. Code: `https://github.com/skezle/continual-dreamer`.
   - Positive: "Mastering URLB from pixels" (Rajeswar, Mazzaglia et al., ICML 2023) finds P2E is a solid pre-training strategy inside DreamerV2 on URLB (93.6 % overall vs the best model-free URLB baseline Disagreement at 39.0 %), and that with a world model "the overall performance improves over time for all categories" of intrinsic objectives.
   - Implementation fragility: sheeprl bugs above; target mismatch (embedding vs stoch) across codebases.
6. Official: `https://github.com/ramanans1/plan2explore` (verified; TF1.14 / CUDA 9 / Python 3.6, 244 stars, last substantive update Jan 2022; README points to a TF2 DreamerV2-based version). Maintained ports: `danijar/dreamerv2` (TF2) `--expl_behavior Plan2Explore`; `danijar/dreamerv3` 2023 JAX release (`expl_behavior`, `expl_rewards: {extr: 1.0, disag: 0.1}`); `NM512/dreamerv3-torch` (`expl_behavior: plan2explore`); `Eclectic-Sheep/sheeprl` `p2e_dv3` (multi-discrete actions supported); `mazpie/mastering-urlb` (PyTorch DreamerV2 with `plan2explore` and six other strategies).
7. Good fit. The baseline everyone compares against; reset-agnostic; pixel-native; several maintained implementations. Expect it to under-explore "interaction" novelty in an open world and to be noisy around stochastic respawns; mitigate via mixed reward (`disag: 0.1`), the `log` transform, and/or an LBS-style signal.

### 1.4 LBS — Latent Bayesian Surprise

1. Mazzaglia, Catal, Verbelen, Dhoedt. AAAI 2022. arXiv:2104.07495.
2. Learns a latent dynamics model with a prior p(z_{t+1}|z_t,a_t) and a posterior q(z_{t+1}|z_t,a_t,o_{t+1}). Intrinsic reward = KL(posterior || prior) of the latent state - information gained about the latent by observing o_{t+1} - as a proxy for Bayesian surprise. No ensemble needed. Motivated as the latent-space analogue of VIME's parameter-space information gain; the paper reports it outperforms VIME and Disagreement and is "robust to stochasticity in the dynamics".
3. Per-transition, non-episodic signal; no reset dependence. Can be used retrospectively (on real data) or, in a Dreamer, as a reward computed in imagination from the prior/posterior pair.
4. Validated in "continuous-control and discrete-action settings", state- and vision-based (per paper and official repo description). Used as one of the pre-training strategies in "Mastering URLB from pixels" (DreamerV2, pixels), where it was among the stronger knowledge-based objectives (exact per-strategy numbers are in that paper's Figure 1b/appendix; I could not extract them from the HTML render - not verified).
5. Fewer independent reproductions than P2E. Its robustness-to-noise claim is the main reason to consider it: the KL(post||prior) collapses on truly random transitions once the prior has matched the noise distribution, whereas finite-ensemble disagreement decays more slowly.
6. `https://github.com/mazpie/lbs-exploration` (verified; PyTorch). Also implemented as the `lbs` strategy in `https://github.com/mazpie/mastering-urlb` (verified; PyTorch DreamerV2).
7. Good fit / drop-in alternative to the disagreement head. Cheap (no ensemble), pixel-validated, reset-agnostic. Less battle-tested than P2E.

### 1.5 LEXA — Latent Explorer Achiever

1. Mendonca, Rybkin, Daniilidis, Hafner, Pathak. NeurIPS 2021. arXiv:2110.09514.
2. One DreamerV2 world model, two imagination-trained policies. Explorer: Plan2Explore ensemble-disagreement intrinsic reward (seeks "unseen surprising states through foresight"). Achiever: goal-conditioned policy trained in imagination to reach goals sampled from replay; reward is either a learned temporal distance ("lexa_temporal") or cosine similarity in latent space ("lexa_cosine"). Real data is collected by alternating explorer and achiever episodes. After unsupervised training the achiever solves goal-image tasks zero-shot.
3. Episodic in design: explorer and achiever alternate per episode, goals are sampled from replay, evaluation is goal-image reaching. In a continuing env you would switch policies on a timer rather than at episode boundaries (MoReFree, 1.8, does exactly this for PEG). The goal space is the learned latent, not hand-designed, so it satisfies the "no hand-designed goal space" constraint.
4. Pixels. RoboYoga, RoboBins, RoboKitchen (+ Joint), 40 goal-image tasks; a single agent trained across four environments. Fixed-scene robotics, not open worlds. Baselines: DDL, DIAYN, GCSL, SkewFit.
5. PEG (2023) shows LEXA's goal selection (sampling from replay) is a weak explorer in long-corridor/stacking tasks (LEXA ~0 % on 3-Block Stack vs PEG ~30 %); Dr. Strategy (2024) reports LEXA at 9.6 % vs 67 % on a 25-room pixel maze and ~20 % vs 94 % on 9-room. Known env-version fragility in the repo ("robobin environment code is known to break with other versions").
6. `https://github.com/orybkin/lexa` (verified; TF2, built on DreamerV2 + Plan2Explore; 90 stars; methods `lexa_temporal`, `lexa_cosine`, `ddl`, `diayn`, `gcsl`).
7. Needs adaptation (timer-based policy switching). Its explorer is just P2E; the achiever adds a goal-reaching capability you may not need for pure exploration. Prefer PEG/Dr. Strategy if you want goal-directed exploration, Director if you want hierarchy.

### 1.6 Director — Deep Hierarchical Planning from Pixels

1. Hafner, Lee, Fischer, Abbeel. NeurIPS 2022. arXiv:2206.04114.
2. DreamerV2-style world model. A goal autoencoder (categorical latent codes; loss = ||dec(z) - s_t||^2 + beta KL) compresses replay-buffer latent states into a discrete goal space. A manager selects a code z every K = 8 steps (decoded into a 1024-d feature-space goal); a worker reaches it with primitive actions using a cosine-similarity reward. "The manager maximizes the task reward and an exploration bonus based on the autoencoder reconstruction error, implementing temporally-extended exploration." Because the autoencoder tracks the replay distribution, states with high reconstruction error are novel. The manager has two critics (extrinsic and exploration). Everything is trained in imagination.
3. No episodic assumptions in the mechanism (manager re-plans every K steps; goals are latent). Runs in a continuing env as-is. Evaluated episodically (e.g., Ant Maze episodes end at 3000 steps).
4. Pixels (egocentric). Egocentric Ant Maze S/M/L/XL, Visual Pin Pad 3-6, DMC, Atari, Crafter, DMLab. Director beats flat Plan2Explore and flat Dreamer on Ant Maze and Pin Pad 5/6; "The exploration bonus of Plan2Explore offers a significant improvement over Dreamer" on Pin Pad but P2E "completely fails with six pads". Ablations: goal autoencoder is essential ("goals are completely uninterpretable and cause the agent to fail in many, but not all, of the tested environments" without it); the exploration bonus is described as crucial in complex environments.
5. Independent evidence on hierarchical world models is sobering: Schiewer, Subramoney, Wiskott, "Exploring the limits of hierarchical world models in RL" (Scientific Reports 2024; arXiv:2406.00483) build a Director-like stacked-RSSM hierarchy and find the higher level "seeks and exploits inaccuracies of the level 1 model", so the hierarchy "matched or underperformed the baseline across all environments" (state-based Nav2d/PointMaze/Reacher/HalfCheetah). Director's official repo has a single commit and is TF2; no widely used reimplementation found. Hyperparameters were kept at Dreamer defaults across domains (paper), which is a point in its favour.
6. `https://github.com/danijar/director` (verified; TF2 on the `embodied` framework; 1 commit; custom envs `pinpad.py`, `loconav.py`). Hieros (1.13) is a PyTorch re-think of the same idea.
7. Good fit (needs engineering). The only officially released hierarchical pixel explorer validated on Crafter/DMLab/Atari; reset-agnostic; novelty signal is a learned autoencoder, not a hand-designed cell. Risk: TF2 code age and the model-exploitation failure mode documented by Schiewer et al.

### 1.7 PEG — Planning Goals for Exploration

1. Hu, Chang, Rybkin, Jayaraman. ICLR 2023 (Spotlight). arXiv:2303.13002.
2. Built on LEXA (DreamerV2 world model + goal-conditioned achiever + P2E explorer). Instead of sampling goals from replay, PEG plans the goal command: a sampling-based optimiser (MPPI-style) searches goal space to maximise the expected exploration value of the trajectory the goal-conditioned policy would produce under the world model, where exploration value is the Plan2Explore disagreement objective ("incentivizes reaching states that cause an ensemble of world models to disagree amongst themselves"). Each episode is Go-Explore-shaped: a "go" phase (goal-conditioned policy reaches the planned goal) then an "explore" phase (P2E policy explores from there). Return is by policy, never by state restore.
3. Episodic by construction: goal planning happens at episode start from the reset state; the go/explore split is per episode. Adaptation: run the go/explore cycle on a fixed timer from the current state - this is precisely MoReFree's "chunking" (1.8). The goal space is the learned latent (no hand design).
4. State observations in the paper's four tasks (Point Maze, Walker, Ant Maze, 3-Block Stack; 2-D to 29-D). Not validated from pixels, although the underlying LEXA/DreamerV2 stack is pixel-capable. Results: beats LEXA, MEGA, SkewFit and P2E; "PEG achieves about 30% success rate [on 3-Block Stack]... all other baselines are close to 0%".
5. No independent pixel reproduction found. Compute: a goal-planning optimisation per episode with imagined rollouts (cost grows with the number of candidate goals).
6. `https://github.com/penn-pal-lab/peg` (verified; TF 2.4, Python 3.6, built on DreamerV2/LEXA; includes model-based SkewFit, MEGA, LEXA baselines; MIT).
7. Needs adaptation (episodic go/explore -> timer-based chunks, and pixel validation). Conceptually the best "directed exploration without state restore" in this family.

### 1.8 MoReFree — Reset-free RL with World Models (PEG adapted to no resets)

1. Yang, Moerland, Preuss, Plaat, Hu. TMLR 2025. arXiv:2408.09807.
2. "PEG is a model-based Go-Explore framework that extends LEXA"; MoReFree adapts it to reset-free training. Back-and-forth Go-Explore: the long reset-free horizon is split into chunks, and the agent "alternates between temporally extended phases of task solving, resetting, and exploration", periodically directing itself "to return to the states relevant to the task (i.e. initial and evaluation goals)". Exploration reward is Plan2Explore's: "the variance of the ensemble predictions averaged across dimension". Task-relevant imagination biases imagined goal sampling toward initial/evaluation-goal states (alpha = 0.2). No environment reward needed (learned dynamical distance), but the evaluation goal distribution must be given.
3. Explicitly designed for a continuing (reset-free) process. The direct answer to "how would PEG/LEXA run without resets".
4. State observations only (7-D to 29-D): PointUMaze, Tabletop, Sawyer Door, Fetch Push/Pick&Place (+Hard), Ant. "Both model-based approaches outperform prior state-of-the-art baselines in 7/8 tasks".
5. Small community footprint (4 stars); not pixel-validated; requires specifying goal states, which in your setting would be a problem-knowledge input unless replaced by purely intrinsic chunk targets.
6. `https://github.com/yangzhao-666/MoReFree` (verified; builds on PEG, TF2).
7. Needs adaptation (pixels; remove the task-goal prior). Mainly valuable as the published recipe for running Go-Explore-style world-model exploration without resets.

### 1.9 Curious Replay (CR)

1. Kauvar, Doyle, Zhou, Haber. ICML 2023. arXiv:2306.15934.
2. Not a reward: a prioritised replay rule for model-based agents. Priority p_i = c * beta^{v_i} + (|L_i| + eps)^alpha, where v_i is the number of times experience i has been trained on (count-based, decays) and L_i is the world-model loss on it (adversarial / model-loss term). Sampling is from p_i; the policy's reward is unchanged. Rationale: focus model training on under-trained or poorly modelled experiences so the world model adapts quickly to novelty, which the imagination-trained policy then exploits.
3. Completely reset-agnostic (operates on the replay buffer). Note the implicit assumption that the buffer contains the novel experience at all; CR speeds adaptation after an encounter, it does not by itself cause the encounter.
4. Pixels. Crafter: DreamerV3 + CR 19.4 vs DreamerV3 14.5 (then SOTA). DMC: "similar mean and median score across all 20 tasks" with some task-level regressions (e.g., Cartpole Swingup Sparse). Novel-object assay: baseline P2E takes 300K+ steps to interact with a newly appeared object, CR ~50K.
5. Authors report "Curious Replay is relatively insensitive to its hyperparameters". Independent follow-ups exist on primacy bias in MBRL (arXiv:2310.15017, which cites CR) but I did not verify an independent Crafter reproduction. Orthogonal to and combinable with any intrinsic reward.
6. `https://github.com/AutonomousAgentsLab/curiousreplay` (hub), `https://github.com/AutonomousAgentsLab/cr-dv3` (verified; JAX, fork of `danijar/dreamerv3` at commit `84ecf19`; `--replay curious-replay`; 40 stars), `https://github.com/AutonomousAgentsLab/cr-dv2` (TF2).
7. Good fit. Cheapest, most robust add-on in this review; best Crafter evidence of anything here. Combine with a P2E head.

### 1.10 BYOL-Explore (and BYOL-Hindsight)

1. Guo, Thakoor, Pislar, Avila Pires, Altche, Tallec, Saade, Calandriello, Grill, Tang, Valko, Munos, Azar, Piot. NeurIPS 2022. arXiv:2206.08332. Follow-up: Jarrett, Tallec, Altche, Mesnard, Munos, Valko, "Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments" (BYOL-Hindsight), ICML 2023 (PMLR v202).
2. A single self-supervised latent prediction loss does everything: an online encoder + RNN "world model" predicts, open-loop over K future steps, the representations produced by an EMA target encoder (BYOL-style bootstrapping; no reconstruction). The per-transition intrinsic reward is the sum of those multi-step prediction losses ("The uncertainty associated to the transition is the sum of the corresponding prediction losses"), normalised by an EMA of its std and clipped so that only above-mean uncertainties are rewarded. The policy is model-free (VMPO) on real data; the world model only supplies rewards and (optionally) shared representations. BYOL-Hindsight adds a learned representation of the unpredictable part of each outcome as extra prediction input so rewards reflect only predictable dynamics (noise robustness).
3. Intrinsic reward is per-transition; reported Montezuma comparisons used the "episodic setting for intrinsic rewards" (RND-style episodic intrinsic return), but nothing prevents a non-episodic intrinsic return. No reset dependence in the model.
4. Pixels, large 3-D and Atari worlds: DM-HARD-8 (partially observable, continuous-action, visually rich 3-D hard-exploration suite) - "human-level performance in the majority of the tasks using only the extrinsic reward augmented with BYOL-Explore's intrinsic reward"; "superhuman performance on the ten hardest exploration games" in Atari. BYOL-Hindsight: SOTA Montezuma with sticky actions.
5. Ablations: horizon K (1-16), EMA rate; fixed (untrained) targets perform poorly. The main brittleness is practical: DeepMind-scale distributed training (VMPO) and no official code; I found no credible open reimplementation (searches for "BYOL-Explore" code return only plain BYOL image-SSL repos). The original is vulnerable to stochasticity (motivation of BYOL-Hindsight).
6. No official repository found (DeepMind publication page lists none; not verified beyond that). Reimplementation required.
7. Needs adaptation (reimplementation; choose non-episodic intrinsic return). Strong evidence on the most Rain-World-like benchmarks (large 3-D, partially observed, hard exploration), but it is model-free in policy learning and expensive to reproduce. Prefer BYOL-Hindsight's noise-robust variant if you build it.

### 1.11 MaxInfoRL

1. Sukhija, Coros, Krause, Abbeel, Sferrazza. ICLR 2025. arXiv:2412.12098.
2. Information-gain bonus from an ensemble of probabilistic forward models over (s', r): r_int = sum_j log(1 + sigma^2_{epistemic,j}(s,a) / sigma^2_{aleatoric}), an upper bound on I(s'; f* | s,a). The policy is model-free (SAC/REDQ/DrQ/DrQv2) on real data: "The learned model is only used to determine the intrinsic reward". Two temperatures alpha_1 (policy entropy) and alpha_2 (information gain) are auto-tuned by constrained optimisation, so the intrinsic/extrinsic trade-off needs no manual schedule.
3. Per-transition, non-episodic signal; off-policy; no reset dependence.
4. State and pixel DMC (DrQ variants), including hard visual humanoid stand/walk/run where MaxInfoDrQv2 reports the highest model-free returns in the literature; HumanoidBench. Continuous actions only in the released code.
5. Authors acknowledge "MaxInfoRL requires training an ensemble of forward dynamics models... which increases computation overhead". No open-world or discrete-action validation; no independent reproduction found yet (2025 paper).
6. `https://github.com/sukhijab/maxinforl_jax` (verified; JAX; MaxInfoSAC/REDQ/DrQ/DrQv2; 29 stars) and `https://github.com/sukhijab/maxinforl_torch` (linked from project page; not individually fetched).
7. Needs adaptation (discrete MultiBinary actor; the policy is not world-model-driven). Worth noting for its principled auto-tuning of the intrinsic weight, which is a recurring pain point for P2E.

### 1.12 Dr. Strategy — Model-Based Generalist Agents with Strategic Dreaming

1. Hamed, Kim, Kim, Ahn (ahn-ml). ICML 2024 (PMLR v235). arXiv:2402.18866.
2. DreamerV2 world model + LEXA-style explorer/achiever, codebase based on Choreographer. A VQ-VAE over model states yields ~64-128 discrete "landmarks". Three imagination-trained policies: a landmark-conditioned highway policy (go to landmark), an explorer ("exploration reward that encourages maximizing disagreement among an ensemble of 1-step dynamics models", i.e. P2E) launched from "curious" landmarks, and an achiever trained with focused (near-state) goal sampling. Addresses "naive dreaming": standard agents start imagination from random replay states, which is undirected in large partially observed worlds.
3. The paper describes exploration as non-episodic in spirit (reach landmark, then explore) within episodes of 500-1000 steps; evaluation is goal-reaching. Like PEG, the go-to-landmark/explore cycle can be driven by a timer in a continuing env. Landmarks are learned, not hand-designed.
4. Pixels, partially observable: 9-room, 25-room, spiral 9-room 2-D navigation (64x64 egocentric), 3-D mazes 7x7 and 15x15 (first person), RoboKitchen. ~2M steps. 25-room: 67.1 % vs LEXA 9.6 %; 9-room ~94 % vs ~20 %.
5. Low adoption so far (official repo 8 stars, 3 commits, Docker + W&B required). No independent reproduction found. Inherits P2E's disagreement signal and its weaknesses.
6. `https://github.com/ahn-ml/drstrategy` (verified; based on Choreographer, PyTorch; envs `9rooms`, `25rooms`, `spiral9`, `mz7x7`, `mz15x15`, `robokitchen`, `dmc_walker`, `dmc_quad`).
7. Needs adaptation (timer-based cycling; engineering on a thin codebase). The most relevant recent pixel result for "big partially observed map" exploration driven by a world model.

### 1.13 Hieros — Hierarchical Imagination on S5 World Models

1. Mattes, Schlosser, Herbrich. ICML 2024. arXiv:2310.05167.
2. Director-style hierarchy on an S5 (structured state-space) world model that imagines at multiple time scales. Two intrinsic terms: a novelty reward r_nov = ||h_t - g_psi(h_t)||_2 (reconstruction error of the subgoal autoencoder, as in Director) and a subgoal-achievement reward (cosine); total r = r_extr + w_g r_g + w_nov r_nov. Adds Efficient Time-Balanced Sampling (O(1) replay sampling favouring slightly older experience, tau = 0.3).
3. Reset-agnostic mechanism (same argument as Director).
4. Pixels, Atari100k: mean HNS 120, median 56 (vs DreamerV3 112 mean); exploration-heavy games improve (Frostbite 2901 vs 909, James Bond 939 vs 445, Private Eye 1507 vs 882). 37.1M params; ~14 h per game on an A100 for 100k steps.
5. 100k-step Atari only; no open-world or long-horizon validation; small codebase (23 stars). No independent reproduction found.
6. `https://github.com/Snagnar/Hieros` (verified; PyTorch, built on dreamerv3-torch and an S5 port).
7. Needs adaptation (scale, long horizons). A PyTorch starting point if you want Director-style hierarchy without TF2.

### 1.14 GoBI — Go Beyond Imagination

1. Fu, Peng, Lee. ICML 2023. arXiv:2308.13661.
2. Learned dynamics model used to imagine k-step random-action rollouts from the current state; an episodic buffer stores visited states and imagined-reachable states; intrinsic reward r_int = (m_{t+1} - m_t) x r_lifelong, i.e. growth of the episodic reachable set times a lifelong novelty term. Base RL: IMPALA (MiniGrid), RAD (DMC).
3. Episodic by construction: "critically depends on maintaining per-episode episodic buffers that reset each episode". Adaptation would require a sliding-window or decaying memory and a non-hand-designed state key for the buffer (MiniGrid used exact states), which is awkward from pixels.
4. MiniGrid (12 tasks, strong gains, e.g. 3x more sample-efficient than NovelD on ObstructedMaze-2Dlhb) and a few sparse pixel DMC tasks.
5. No code URL in the paper (not verified further).
6. None found.
7. Excluded (episodic memory + state keys; no code).

### 1.15 DreamerV3-XP

1. Bierling, Pasero, Bertrand, Van Gerwen. arXiv:2510.21418 (Oct 2025; preprint, no venue stated).
2. DreamerV3 plus (a) prioritised trajectory replay scored by s_i = (lambda_r + lambda_delta delta_i) R_i + lambda_eps eps_i (return, critic error, reconstruction error) and (b) an intrinsic reward from disagreement over an ensemble of reward predictors that roll out deterministic latents: mean predicted reward + variance. Note: reward-disagreement, not dynamics-disagreement, so it is an optimism bonus rather than reward-free novelty.
3. Reset-agnostic mechanisms.
4. Pixels: 3 Atari100k games (Battle Zone, Boxing, Krull) and 2 DMC vision tasks; 2 seeds. Replay prioritisation "consistently reduces the dynamics loss"; intrinsic bonus gave "modest" gains on Krull and Cup Catch; no degradation reported.
5. Very thin evidence (5 tasks, 2 seeds, authors cite compute constraints). Useful mainly for its explicit statement of the problem: in DreamerV3 "exploration is purely guided by environment rewards... Hence, this can hinder broad and thorough exploration, especially in sparse reward settings."
6. `https://github.com/Coluding/dreamerv3` (verified; JAX fork of dreamerv3; 2 stars).
7. Excluded as a method to adopt (reward-dependent bonus, weak evidence); cited as documentation for question (b).

### 1.16 Change-Based Intrinsic Motivation for World Model Agents (CBET on DreamerV3)

1. Ferrao, Cunha. Northern Lights Deep Learning Conference 2025. arXiv:2503.21047.
2. Adapts CBET (change-based exploration transfer; rewards observation changes caused by the agent) to DreamerV3 and compares with IMPALA+CBET.
3. Reset-agnostic reward.
4. Pixels: Crafter and MiniGrid. Helped DreamerV3 returns in Crafter, "produced suboptimal policies in Minigrid"; intrinsic pre-training did not transfer to extrinsic performance in MiniGrid.
5. Mixed/negative result, small venue; no code mentioned (not verified).
6. None found.
7. Excluded (mixed evidence, no code); relevant as another data point that bolting a model-free bonus onto DreamerV3 is environment-dependent.

### 1.17 Optimistic World Models (O-DreamerV3 / O-STORM)

1. Mete, Sheikh, Lin, Kalathil, Kumar. arXiv:2602.10044 (Feb 2026; preprint).
2. Reward-biased maximum-likelihood idea: an "optimistic dynamics loss" biases the world model's imagined transitions toward higher-reward outcomes so the actor is optimistic in imagination. Not an intrinsic reward and not reward-free.
3. Reset-agnostic.
4. Pixels: Atari100k (26 games): O-DreamerV3 152.7 % mean HNS vs 97.5 % for their DreamerV3 run; large gains on sparse Private Eye, Freeway, Cartpole Swingup Sparse; O-STORM 80.7 % vs 75.9 %. DMC proprio/vision also reported.
5. Preprint; no code found (not verified); requires task reward.
6. None found.
7. Excluded for reward-free exploration (needs extrinsic reward); included because it is the most recent published "fix" to DreamerV3's exploration on sparse-reward pixels.

### 1.18 CIG — Exploration via Conditional Information Gain

1. Joseph, Fechner, Stegmaier, Daaboul, Zollner. arXiv:2605.20878 (May 2026; preprint).
2. Model-based exploration where the intrinsic reward is a tractable surrogate for information gain conditioned on both the replay history and the current rollout prefix: a log-determinant over an ensemble-disagreement kernel, Cholesky-factorised to give causal per-step rewards. Explicitly departs from Plan2Explore by damping small disagreements toward a floor so credit concentrates on transitions whose disagreement "clearly exceeds the predictor's own noise scale".
3. Reset-agnostic per-step reward (per abstract).
4. 12 tasks: MiniGrid (discrete) and OGBench (continuous), clean and with stochastic distractors. Pixel observations not confirmed (not verified).
5. Too new for independent evaluation; no code found (not verified).
6. None found.
7. Watch-list. Directly targets the stochastic-transition weakness of disagreement rewards, which is the main hazard in a death-respawn continuing world.

### 1.19 SeeX — Separated World Model for Exploration under Visual Distraction

1. Huang, Wan, Shao, Sun, Gan, Feng, Zhan. NeurIPS 2024.
2. Bi-level: inner loop trains a separated world model splitting endogenous (controllable, task-relevant) from exogenous (distractor) latent factors; outer loop trains an exploration policy on imagined trajectories in the endogenous latent space to maximise task-relevant uncertainty. Addresses intrinsic rewards being hijacked by background distractors in pixel inputs.
3. Reset-agnostic in principle (I could only access the abstract/poster page; details not verified).
4. Locomotion and manipulation from pixels with visual distractors (not verified in detail).
5. Single paper; code status not verified.
6. Not verified.
7. Needs verification. Conceptually relevant: Rain World has heavy non-agent dynamics (rain, creatures), which is exactly the exogenous-noise problem SeeX attacks.

### 1.20 Older/adjacent items (brief)

- LWM, "Latent World Models for Intrinsically Motivated Exploration" (Ermolov, Sebe, NeurIPS 2020, arXiv:2010.02302). Temporal-distance-structured latent representation + forward-model error as novelty, with both episodic and lifelong uncertainty terms; Atari hard-exploration from pixels. Code `https://github.com/htdt/lwm`. Model-free policy; reset-agnostic if the episodic term is dropped.
- Ready Policy One (Ball et al., ICML 2020, arXiv:2002.02693). MBRL as active learning; hybrid reward/exploration objective, state-based. Lineage citation only.
- Choreographer (Mazzaglia et al., ICLR 2023, arXiv:2211.13350; code `https://github.com/mazpie/choreographer`, PyTorch). Skill discovery in imagination on top of a Dreamer world model; the codebase that Dr. Strategy builds on. Skills, not exploration per se.
- THICK (Gumbsch et al., ICLR 2024 Spotlight; code `https://github.com/CognitiveModeling/THICK`, TF2). Hierarchical world model with sparse context changes; evaluated on MiniHack among others; not an exploration method but a candidate backbone for hierarchical exploration.
- "Investigating the role of model-based learning in exploration and transfer" (Walker et al., DeepMind, arXiv:2302.04009). Crafter, RoboDesk, Meta-World; finds model-based agents transfer better and that "intrinsic exploration combined with environment models present a viable direction". Algorithms not identified from the abstract (not verified).
- InDRiVE (Khanzada, Kwon, arXiv:2512.18850, Dec 2025). DreamerV3 + latent ensemble disagreement for reward-free pre-training in CARLA; robust zero/few-shot transfer. Application paper; confirms the P2E recipe still being used as the default in 2025-26.
- RLeXplore (Yuan et al., TMLR 2025; `https://github.com/RLE-Foundation/RLeXplore`, PyTorch). Plug-and-play model-free intrinsic rewards (ICM, RND, RIDE, E3B, Disagreement, ...). Useful as reference implementations of the reward computations, not world-model-driven.
- Craftax (Matthews et al., ICML 2024, arXiv:2402.16801): model-free intrinsic rewards (ICM, E3B, RND) "failed to improve overall performance" over PPO/PPO-RNN and E3B hurt. Not a world-model result, but the strongest recent negative result on intrinsic rewards in an open-world benchmark.

### 1.21 Excluded by the no-foundation-model constraint (listed only to flag)

- SENSEI (Sancaktar, Gumbsch, Zadaianchuk, Kolev, Martius, ICML 2025, arXiv:2503.01584): DreamerV3 + a GPT-4-distilled "interestingness" reward combined with P2E disagreement, adaptive switching between the two; MiniHack and Robodesk. Excluded (VLM prior). Its baselines are nevertheless useful evidence on P2E's weaknesses (quoted in 1.3).
- LS-Imagine (Li, Wang et al., ICLR 2025 Oral, arXiv:2410.03618; code `https://github.com/qiwang067/LS-Imagine`): long-short-term imagination in MineDojo with affordance maps scored against textual goals (MineCLIP-style). Excluded (language/pretrained reward).
- DLLM, "World Models with Hints of Large Language Models for Goal Achieving" (arXiv:2406.07381): LLM subgoals injected into Dreamer rollouts. Excluded.

---

## 2. Summary table

| Method | Year | Intrinsic signal | Policy trained | Pixels validated | Open-world-ish eval | Resets needed? | Official code (framework) | Fit |
|---|---|---|---|---|---|---|---|---|
| VIME | 2016 | BNN param info gain | model-free, real | no | no | no | openai/vime (Theano) | excluded |
| MAX | 2019 | JSD/JRD ensemble disagreement | SAC in model | no | no | no | nnaisense/max (PyTorch) | excluded |
| Plan2Explore | 2020 | latent ensemble disagreement | imagination | yes (DMC) | Crafter (weak), MiniHack via others | no | ramanans1/plan2explore (TF1); dreamerv2/v3 (2023), dreamerv3-torch, sheeprl | good |
| LBS | 2022 | KL(posterior||prior) latent surprise | real or imagination | yes | no | no | mazpie/lbs-exploration (PyTorch) | good |
| LEXA | 2021 | P2E explorer + goal achiever | imagination | yes (robotics) | no | episodic alternation | orybkin/lexa (TF2) | adapt |
| Director | 2022 | goal-AE reconstruction error (manager) | imagination | yes | Crafter, DMLab, Atari, Ant Maze | no | danijar/director (TF2) | good (eng.) |
| PEG | 2023 | planned goals maximising P2E value | imagination + goal planning | no (states) | no | episodic go/explore | penn-pal-lab/peg (TF2) | adapt |
| MoReFree | 2025 | PEG, reset-free chunks | imagination | no (states) | no | no (designed reset-free) | yangzhao-666/MoReFree (TF2) | adapt |
| Curious Replay | 2023 | none (replay priority) | DreamerV3 | yes | Crafter 19.4 | no | cr-dv3 (JAX), cr-dv2 (TF2) | good |
| BYOL-Explore / Hindsight | 2022/23 | multi-step latent prediction loss vs EMA target | VMPO, real | yes | DM-HARD-8, Atari | no | none official | adapt (reimpl.) |
| MaxInfoRL | 2025 | ensemble info-gain bound, auto-tuned | SAC/DrQ, real | yes (DMC) | no | no | maxinforl_jax / _torch | adapt |
| Dr. Strategy | 2024 | P2E from learned landmarks | imagination | yes (mazes, kitchen) | 25-room / 3-D mazes | timer-adaptable | ahn-ml/drstrategy (PyTorch) | adapt |
| Hieros | 2024 | subgoal-AE novelty + subgoal reward | imagination | yes (Atari100k) | no | no | Snagnar/Hieros (PyTorch) | adapt |
| GoBI | 2023 | episodic reachability via WM rollouts | IMPALA/RAD | partly | no | yes (episodic buffer) | none | excluded |
| DreamerV3-XP | 2025 | reward-ensemble disagreement + PER | DreamerV3 | yes (5 tasks) | no | no | Coluding/dreamerv3 (JAX) | excluded |
| O-DreamerV3 | 2026 | optimistic dynamics loss | DreamerV3/STORM | yes (Atari100k) | no | no | none | excluded (needs reward) |
| CIG | 2026 | conditional info gain (log-det kernel) | MBRL | not confirmed | no | no | none | watch |

---

## 3. Cross-cutting issues for a continuing, death-respawn, pixel open world

1. Non-episodic intrinsic return. All imagination-trained explorers (P2E, LEXA, Director, PEG, Dr. Strategy, Hieros) compute value over imagined horizons with a learned continuation predictor; with no terminals this is already a non-episodic return. Model-free-policy methods (BYOL-Explore, MaxInfoRL, LWM) should use a non-episodic intrinsic return (the RND/NGU literature finds non-episodic intrinsic returns explore more when there is no extrinsic reward, because the return is not truncated at "game over").

2. Death as a stochastic, high-surprise transition. Disagreement-based rewards theoretically vanish on irreducibly stochastic transitions (Plan2Explore's own statement) but only asymptotically; in finite data a respawn can be rewarded for a long time, pulling the explorer toward dying. Mitigations that remain "general-purpose": LBS (posterior-prior KL, converges faster on noise), BYOL-Hindsight (explicitly models the unpredictable part), CIG (noise-floor damping), or the `disag_log` transform plus reward normalisation. Treating the death flag from `info` as an episode boundary for the intrinsic critic is a defensible generic choice (every Dreamer codebase already has `is_terminal`/`cont` plumbing), but it is a design decision you should report.

3. Go-Explore without restore. PEG / Dr. Strategy / LEXA achieve "return to a frontier" with a goal-conditioned policy rather than state restore, so they are compatible with a continuing env once the go/explore cycle is driven by a timer; MoReFree is the published template for that change.

4. Replay in a one-lifetime stream. Curious Replay's count-and-loss priority and Hieros's time-balanced sampling both target exactly the "old data dominates uniform replay" issue of a long single lifetime. Continual-Dreamer (Kessler 2023) found reservoir-style uniform coverage mattered more than clever sampling for forgetting in MiniHack; worth an ablation.

5. Action space. DreamerV3 codebases natively handle one-hot discrete actions; sheeprl's P2E supports multi-discrete. MultiBinary(9) must be either flattened (512-way categorical) or factorised into 9 Bernoulli heads; none of the papers above evaluate factorised binary actions (not verified for `danijar/dreamerv3` main).

6. Exogenous dynamics. Rain World has substantial agent-independent dynamics (creatures, rain cycle). Reconstruction-based novelty (Director/Hieros) and prediction-error novelty will partly chase these; SeeX (1.19) and the controllability idea behind ICM/BYOL-Hindsight are the general remedies.

---

## 4. Explicit answers

### (a) State of the art for world-model-driven exploration from pixels in open-ended worlds, 2025-2026

There is no clean winner, and the honest reading of the record is:

- The reproducible, widely used baseline is still Plan2Explore-style latent ensemble disagreement inside a Dreamer (DreamerV2/V3, 2020-2026: LEXA, PEG, Dr. Strategy, MoReFree, SENSEI, InDRiVE all reuse it). Nothing published has displaced it as the default reward-free signal.
- The best reward-free pixel results in large, partially observed worlds come from goal- or landmark-directed exploration layered on that signal (Dr. Strategy 2024 on 25-room and 3-D mazes; Director 2022 on egocentric Ant Maze and Pin Pad), i.e. fixing where imagination starts and how far it reaches rather than changing the novelty formula.
- In Crafter/Craftax/Minecraft specifically, the published reward-free world-model exploration evidence is thin and old: Hafner's 2021 Crafter numbers put P2E at 2.1 % vs random 1.6 %. The strongest Crafter gain from a general-purpose change is Curious Replay (19.4 vs 14.5), but that is with task reward. Craftax (2024) shows model-free intrinsic rewards not helping PPO at all. DreamerV3's Minecraft diamonds came from the entropy regulariser alone, not intrinsic rewards.
- The methods that claim semantically meaningful open-world exploration in 2025 (SENSEI, LS-Imagine, DLLM) do so by importing a VLM/LLM prior, which is exactly what your constraints exclude. As of this review I found no 2025-26 paper demonstrating strong reward-free world-model exploration on Craftax/Minecraft-scale pixel worlds without foundation-model priors.
- 2026 preprints (CIG, Optimistic World Models) target the two known failure modes - stochasticity and sparse-reward exploitation - but have no code and no independent reproduction yet.

### (b) Is "DreamerV3 is weak at exploration" documented, and what fixes are published?

Documented, in several independent places:
- DreamerV3 itself (Hafner et al., arXiv:2301.04104; Nature 2025) explores only "through an entropy regularizer" with a fixed scale eta = 3e-4; the paper does not include an intrinsic reward and does not frame exploration as a limitation. The 2023 JAX release shipped an optional P2E head (`expl_behavior`) that the 2024 rewrite removed.
- DreamerV2 (arXiv:2010.02193) reports that on Montezuma's Revenge, without explicit exploration, it reaches roughly ICM-level performance.
- DreamerV3-XP (2025): "exploration is purely guided by environment rewards... can hinder broad and thorough exploration, especially in sparse reward settings."
- Curious Replay (2023): P2E/Dreamer agents take 300K+ steps to touch a newly appeared object; uniform replay is identified as part of the cause.
- SENSEI (2025): P2E "mostly performs arm stretches" and does not consistently beat DreamerV3-from-scratch.
- Director (2022): flat P2E flips the ant and fails at 6-pad Pin Pad; flat Dreamer fails beyond the smallest maze.
- Hieros (2024) and O-DreamerV3 (2026): the largest Atari100k gains over DreamerV3 are on exploration games (Frostbite, Private Eye, Freeway).

Published fixes, from least to most invasive: Curious Replay (replay priority; Crafter 19.4), Plan2Explore head (built into the 2023 DreamerV3 code, dreamerv3-torch, sheeprl), LBS head (cheaper, noise-robust), Director / Hieros (hierarchical goal novelty), PEG / Dr. Strategy (planned goals or landmarks), DreamerV3-XP (reward-ensemble optimism + PER; weak evidence), O-DreamerV3 (optimistic dynamics loss; reward-dependent), CBET-on-DreamerV3 (mixed).

### (c) Which 2-3 methods to implement first, and why

1. Plan2Explore head on DreamerV3, mixed reward. Use a codebase where it already exists and has been debugged: the 2023 `danijar/dreamerv3` JAX commit (`expl_behavior`, `expl_rewards: {extr: 1.0, disag: 0.1}`), `NM512/dreamerv3-torch` (`expl_behavior: plan2explore`), or `sheeprl p2e_dv3` after PR #361 (supports multi-discrete actions). It is reset-agnostic, pixel-native, the universal baseline, and the thing every later method is measured against. Log the ensemble target you use (embedding vs `stoch`) and try `disag_log: True`; expect to need the LBS or log/normalised variant if respawn stochasticity dominates.

2. Curious Replay on the same agent (`cr-dv3`, flag `--replay curious-replay`). Orthogonal to the reward, reported hyperparameter-insensitive, the best general-purpose Crafter improvement on record, and directly aimed at the one-long-lifetime replay problem. Running P2E with and without CR is a cheap, informative 2x2.

3. Director (official TF2 `danijar/director`), or Hieros as a PyTorch stand-in, if flat P2E dithers. It adds temporally extended exploration from a learned discrete goal space with no reset assumption and was validated on Crafter/DMLab/Atari/egocentric mazes - the closest published setting to a large open pixel world. Budget engineering time for the TF2/`embodied` stack and keep the Schiewer et al. model-exploitation failure mode in mind as a diagnostic.

If you later want directed frontier-seeking without state restore, PEG -> MoReFree (timer-chunked go/explore) is the published path, but both are state-based in their papers, so count it as a research step rather than a reproduction. BYOL-Explore has the most Rain-World-like benchmark results (DM-HARD-8) but no official code and a model-free policy; it is a reimplementation project, not a first experiment.

---

## 5. Source list (fetched during this review)

Papers: arXiv 1605.09674 (VIME), 1810.12162 (MAX), 2005.05960 (Plan2Explore), 2104.07495 (LBS), 2010.02302 (LWM), 2002.02693 (RP1), 2110.09514 (LEXA), 2206.04114 (Director), 2206.08332 (BYOL-Explore), Jarrett et al. PMLR v202 (BYOL-Hindsight), 2303.13002 (PEG), 2306.15934 (Curious Replay), 2308.13661 (GoBI), 2209.12016 (Mastering URLB from pixels), 2110.15191 (URLB), 2211.15944 (Continual-Dreamer), 2211.13350 (Choreographer), 2310.05167 (Hieros), 2402.18866 (Dr. Strategy), 2406.00483 / Sci. Rep. 2024 (limits of hierarchical WMs), 2408.09807 (MoReFree), 2412.12098 (MaxInfoRL), 2503.01584 (SENSEI), 2410.03618 (LS-Imagine), 2503.21047 (CBET-DreamerV3), 2510.21418 (DreamerV3-XP), 2512.18850 (InDRiVE), 2602.10044 (Optimistic WMs), 2605.20878 (CIG), 2301.04104 (DreamerV3), 2109.06780 (Crafter), 2402.16801 (Craftax), 2405.19548 (RLeXplore), 2302.04009 (Walker et al.), NeurIPS 2024 SeeX, 2503.23631 (Lidayan et al., humans vs agents in Crafter).

Code: github.com/danijar/{dreamerv2,dreamerv3,director,crafter}, NM512/dreamerv3-torch, Eclectic-Sheep/sheeprl (issue #322, PRs #346/#358/#361), ramanans1/plan2explore, nnaisense/max, orybkin/lexa, penn-pal-lab/peg, yangzhao-666/MoReFree, AutonomousAgentsLab/{curiousreplay,cr-dv3}, mazpie/{lbs-exploration,mastering-urlb,choreographer}, ahn-ml/drstrategy, Snagnar/Hieros, sukhijab/maxinforl_jax, Coluding/dreamerv3, skezle/continual-dreamer, CognitiveModeling/THICK, openai/vime, htdt/lwm, RLE-Foundation/RLeXplore, qiwang067/LS-Imagine.
