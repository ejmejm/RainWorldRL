# Literature review: autotelic / self-goal-setting agents and goal-conditioned exploration with learned goal spaces (no language)

Prepared 2026-10-04 for a pixel-observation, continuing (no resets; death = respawn in place), open-world Rain World environment with a 9-key MultiBinary action space.

Constraints applied: no game-specific tricks (no hand-designed goal/cell spaces), must work or be adaptable to a continuing setting, must work from pixels, no LLM/VLM/pretrained foundation models. Preference for published, reproducible methods with reference code.

Verification policy: every arXiv/venue/repo claim below was checked against a live web search or fetch during this review unless explicitly marked "(not re-verified)" or "(could not verify)". Several PDFs could not be text-extracted (locked downloads); where I relied on an ar5iv rendering or project page instead I say so. Quantitative claims are the papers' own unless otherwise noted.

---

## 0. Executive summary

- The autotelic literature splits into (i) population-based IMGEP work (Oudeyer lab) where goal spaces are learned by unsupervised representation learning and goals are selected by learning progress (LP); (ii) deep goal-conditioned RL (GCRL) where goals are sampled from the agent's own replay/latent distribution with a density/entropy skew (RIG -> Skew-Fit -> MEGA -> Latent Go-Explore -> TLDR); (iii) adversarial / intermediate-difficulty goal generators (Goal GAN, Setter-Solver, AMIGo, asymmetric self-play, SVGG, ULEE); (iv) world-model agents that imagine or plan goals (LEXA, PEG/MoReFree, Director); (v) learned temporal-distance / quasimetric machinery for the goal-reaching policy itself (DDL, Contrastive RL, QRL, HIQL, METRA/TLDR).
- Almost everything assumes episodic resets to a fixed start distribution for *goal selection* ("which goal do I command at the start of this episode?") even when the *policy learning* (HER, contrastive, imagination) is reset-agnostic. The exceptions that explicitly handle no-reset operation are MoReFree (2024, built on PEG) and, implicitly, Director (manager re-picks a goal every K steps from wherever the agent is) and LEO's "pseudo-termination" goal switching.
- Only one non-LLM autotelic method has been demonstrated from pixels in an open-world game at scale with a single hyperparameter set: **Director** (Crafter, DMLab, Atari, egocentric Ant Maze). DISCERN (Atari, DMLab) is the other pixel/game result but has no code and modest results. Everything else is robot arms, mazes, locomotion, or symbolic Craftax (LEO).
- Best-supported goal-selection criterion: no controlled head-to-head exists at pixel/open-world scale. In deep GCRL the most repeatedly winning *simple* criterion is "low-density / frontier of achieved goals + a reachability filter" (Skew-Fit, MEGA, LGE, TLDR), beaten in turn by criteria that estimate *exploration value or value uncertainty* of a goal (PEG, DISCOVER) on state-based tasks. Learning progress is the best-supported criterion in population-based IMGEP, in CURIOUS, and in human studies, and is the only criterion with a principled noise-robustness argument, but there is no deep-GCRL-from-pixels result showing LP beating density-based selection. Adversarial / intermediate-difficulty generators are the most brittle family.
- Recommended first implementations (Section 5): (1) Director-style manager/worker on a DreamerV3 world model; (2) a model-free "learned-latent + density-skewed goal sampling + HER" agent (Skew-Fit/MEGA/LGE lineage) using a contrastive/temporal-distance critic as the goal-reaching reward; (3) PEG with MoReFree-style reset-free goal planning if (1)'s world model is already in place. Add a CURIOUS-style LP layer over goal clusters as a later ablation, not a first step.

---

## 1. Method-by-method review

Template: (1) identity, (2) core algorithm, (3) episodic assumptions and continuing-env adaptation, (4) observations/benchmarks validated on, (5) robustness evidence, (6) code, (7) fit.

### 1.1 IMGEP — Intrinsically Motivated Goal Exploration Processes (population-based)

1. Forestier, Portelas, Mollard, Oudeyer. "Intrinsically Motivated Goal Exploration Processes with Automatic Curriculum Learning." JMLR 23(152), 2022 (arXiv 2017). https://arxiv.org/abs/1708.02190 ; https://www.jmlr.org/papers/v23/21-0808.html
2. Core: a population-based (non-RL) architecture. The agent samples a goal in one of several *modular* goal spaces (in the original work these are hand-engineered outcome spaces: object positions etc.), picks the policy parameters whose past outcome is nearest to the goal (nearest-neighbour retrieval + perturbation), executes, stores the (params, outcome) pair, and all outcomes are reused as hindsight data for every goal space. Goal-space (module) selection is by absolute learning progress (change in competence on that module, estimated from goal-reaching error over time). Within-module goal sampling is uniform. No intrinsic reward in the RL sense; LP is used only for module selection.
3. Episodic: strongly. Each "experiment" is an episode from the same start; the memory stores whole parameterised policies tried from that start. In a continuing env you cannot "retry parameters from the same start"; the policy-retrieval step is meaningless without resets. Only the *LP-based module selection* transfers (see CURIOUS for the RL version).
4. Observations: low-dimensional engineered outcome spaces; real Poppy/Torso robot and simulated tool-use tasks. Not pixels.
5. Robustness: the LP module selection is the component repeatedly re-used (CURIOUS, ACL survey). The population-based part does not scale to high-dimensional policies/observations; the authors position it as a developmental model rather than a deep-RL baseline.
6. Code: Oudeyer-lab explauto / IMGEP repos exist; I did not verify the exact JMLR-companion repo URL in this review. Python/numpy.
7. Fit: **excluded as-is** (episodic, population-based, engineered goal spaces); the LP-over-goal-modules idea is reusable.

### 1.2 IMGEP-UGL — Unsupervised Goal space Learning (and the disentangled / modular follow-up)

1. Péré, Forestier, Sigaud, Oudeyer. "Unsupervised Learning of Goal Spaces for Intrinsically Motivated Goal Exploration." ICLR 2018. https://arxiv.org/abs/1803.00781 ; https://openreview.net/forum?id=S1DWPP1A- . Follow-up: Laversanne-Finot, Péré, Oudeyer, "Curiosity Driven Exploration of Learned Disentangled Goal Spaces," CoRL 2018 (HAL hal-01891598); journal extension "Intrinsically Motivated Exploration of Learned Goal Spaces," Frontiers in Neurorobotics 2021 (PMC7835425).
2. Core: two-stage. Stage 1: passively observe raw images of outcomes and fit an unsupervised representation (VAE, beta-VAE, AE, Isomap, PCA...); the latent space becomes the goal space. Stage 2: run IMGEP (as 1.1) with goals sampled uniformly in the learned latent box, nearest-neighbour policy retrieval, and (in the CoRL follow-up) LP-based selection among latent *dimensions* treated as modules when the representation is disentangled (beta-VAE). Key finding: exploration in learned latent spaces matched or exceeded engineered feature spaces; disentangled latents explored better than entangled ones, and modular LP over latent dimensions helped. No RL policy and no intrinsic reward.
3. Episodic: yes (as 1.1). The representation is learned *before* exploration from passive observations in the original paper (the Frontiers extension studies online representation learning). In a continuing env the "learn a VAE goal space from the agent's own stream" part transfers directly; the population-based goal-reaching does not.
4. Observations: 70x70 pixel renderings of a 2D ArmBall / ArmArrow toy environment. Tiny.
5. Robustness: results depend on representation quality (VAE vs. Isomap differed); goal-space quality mattered mostly through coverage of valid outcomes. No independent reproduction found.
6. Code: https://github.com/flowersteam/Unsupervised_Goal_Space_Learning (verified; 21 stars, AGPL-3.0, scripts rpe.py / rge_efr.py / rge_rep.py, ArmBall-type env, minimal maintenance).
7. Fit: **excluded as a method, important as a design pattern** — the canonical argument that a VAE latent is an acceptable learned goal space, and that LP over latent *dimensions* can be a goal-space-selection signal.

### 1.3 CURIOUS — modular multi-goal RL with absolute learning progress

1. Colas, Fournier, Sigaud, Chetouani, Oudeyer. "CURIOUS: Intrinsically Motivated Modular Multi-Goal Reinforcement Learning." ICML 2019 (oral). https://arxiv.org/abs/1810.06284 ; http://proceedings.mlr.press/v97/colas19a.html
2. Core: a single UVFA-style DDPG+HER policy conditioned on (module one-hot, goal). Goal space = a set of hand-defined modules (Reach, Push, Pick&Place, Stack, plus distractor modules), each a low-dim target. Module selection is proportional to absolute LP (|competence_now - competence_before| per module, estimated from the agent's own sparse success on attempted goals); goals within a module are uniform. Replay sampling is also biased toward high-LP modules. Reward is the sparse goal-success signal (no extra intrinsic reward). Hindsight relabelling across modules.
3. Episodic: yes (Fetch episodes from a fixed start, 50 steps). The LP estimator needs success/failure outcomes on attempted goals; in a continuing env this can be computed per goal attempt bounded by a timeout, so LP module selection is adaptable. The goal *modules* themselves are hand-designed — that violates the "no hand-designed goal spaces" constraint unless modules are replaced by learned latent clusters (as IMGEP-UGL did with beta-VAE dimensions).
4. Observations: low-dim proprioceptive state in a custom "Modular Goal Fetch Arm" env. Not pixels.
5. Robustness: the paper's selling points are robustness to distracting (impossible) modules, to forgetting, and to body changes (ablations in the paper). The follow-ups by the same group moved to social/language signals ("Help Me Explore", "Language as a Cognitive Tool"), which are excluded here.
6. Code: https://github.com/flowersteam/curious (verified; OpenAI Baselines + custom gym_flowers; 27 stars, 9 commits).
7. Fit: **needs adaptation** — LP-based goal-module selection is directly reusable, but the goal space must be learned (latent clusters/dimensions) and the Baselines/TF1 code is legacy.

### 1.4 Autotelic agents survey (Colas, Karch, Sigaud, Oudeyer, JAIR 2022)

1. "Autotelic Agents with Intrinsically Motivated Goal-Conditioned Reinforcement Learning: a Short Survey." JAIR 74, 2022. https://arxiv.org/abs/2012.09830 ; https://jair.org/index.php/jair/article/view/13554
2. Content relevant here: a typology of IMGEP-RL agents along (a) goal representation (target state / image, latent, language...), (b) goal generation / selection (uniform, achieved-goal distribution skewing, LP, intermediate difficulty, adversarial), (c) the goal-conditioned reward (hand-defined, learned reward model, learned distance), (d) exploration bonuses. The survey lists LP-based, density/entropy-based (Skew-Fit, MEGA, DISCERN), intermediate-difficulty (Goal GAN, Setter-Solver) and adversarial (asymmetric self-play, AMIGo) selection and notes the lack of standardised comparisons across them.
3-7. Not a method; the taxonomy used in this review. Companion: Portelas, Colas, Weng, Hofmann, Oudeyer, "Automatic Curriculum Learning for Deep RL: A Short Survey," IJCAI 2020 (https://www.ijcai.org/proceedings/2020/0671.pdf), covering LP vs. intermediate-difficulty vs. adversarial ACL mechanisms.

### 1.5 RIG — Reinforcement learning with Imagined Goals

1. Nair, Pong, Dalal, Bahl, Lin, Levine. "Visual Reinforcement Learning with Imagined Goals." NeurIPS 2018. https://arxiv.org/abs/1807.04742
2. Core: train a beta-VAE on observed images; latent z is both state representation and goal space. Training goals are *sampled from the VAE prior* ("imagined goals"); reward is negative latent-space distance to the goal latent, with HER relabelling in latent space; off-policy TD3/SAC. Test goals are given as images.
3. Episodic: Fetch-style episodes from a fixed start. Goal sampling from the prior is reset-agnostic, so in a continuing env the only adaptation is to switch goals on a timer or on success. The weakness (goal prior collapses to the data mean) is what Skew-Fit fixes.
4. Observations: 84x84 (later 48x48) pixels; Sawyer reach/push/pickup sims and a real Sawyer. Small, closed worlds.
5. Robustness: known failure — sampling from the VAE prior biases goals to already-common states; Skew-Fit and MEGA both show RIG's goal distribution does not expand. Latent L2 distance is a poor reward when the VAE encodes task-irrelevant pixels (motivates DISCERN, CRL, temporal distances).
6. Code: https://github.com/vitchyr/rlkit (verified; PyTorch; 2.9k stars; RIG preserved at tag v0.1.2, Skew-Fit in master; requires MuJoCo + multiworld).
7. Fit: **superseded** by Skew-Fit; historical baseline only.

### 1.6 Skew-Fit — state-covering goal distribution

1. Pong, Dalal, Lin, Nair, Bahl, Levine. "Skew-Fit: State-Covering Self-Supervised Reinforcement Learning." ICML 2020 (arXiv v1 2019). https://arxiv.org/abs/1903.03698
2. Core: same RIG machinery (VAE latent goal space, latent-distance reward, HER, SAC), but the goal-proposal distribution is the VAE refit on replay states *re-weighted by p(s)^alpha with alpha < 0* (importance resampling toward rare states, rank-based in practice). The objective is maximum entropy of the achieved-goal distribution; the paper proves convergence to uniform over valid states under regularity assumptions. Goals are sampled from the skewed VAE; no separate intrinsic reward.
3. Episodic: the theory is stated for finite-horizon, communicating MDPs; goals are set per episode from a fixed start. The skewing step only needs the replay buffer, so it is reset-free compatible; the missing piece in a continuing env is *reachability from the current state* (a rare state may be unreachable from where the agent is now). MEGA's Q-filter / DISCOVER's value term address this.
4. Observations: 48x48 pixels: Visual Door, Visual Pusher, Visual Pickup; plus 2D Four Rooms and Ant Maze (state); real-robot door opening from pixels. Closed, small.
5. Robustness: the paper sweeps alpha in {-1,-0.75,-0.5,-0.25,0} and reports it "works across a large range of alpha" with alpha=-1 consistently better than 0 (Figure 10, per the ar5iv rendering). Negative evidence: in PEG (ICLR 2023) Skew-Fit and MEGA reach ~0% on 3-Block Stack and are sometimes beaten by Plan2Explore alone, the authors noting that bad Go-phase goals can *deteriorate* exploration. In MEGA (ICML 2020) Skew-Fit's distribution expands slower than MEGA's low-density sampling on Ant Maze / Fetch stack. VAE density estimates are unreliable early (MLE warm-up needed).
6. Code: rlkit (above); re-implementations as baselines in https://github.com/spitis/mrl and https://github.com/penn-pal-lab/peg (model-based Skew-Fit).
7. Fit: **needs adaptation** (continuing goal switching + reachability filter; replace VAE-pixel distance reward with a learned temporal distance). The simplest reproducible entry point in this family.

### 1.7 MEGA — Maximum Entropy Gain goal selection (added; central baseline in later work)

1. Pitis, Chan, Zhao, Stadie, Ba. "Maximum Entropy Gain Exploration for Long Horizon Multi-goal Reinforcement Learning." ICML 2020. https://arxiv.org/abs/2007.02832
2. Core: goals are *past achieved goals* from the buffer, chosen to maximise entropy gain of the historical achieved-goal distribution, implemented as sampling the lowest-density achieved goals under a KDE, filtered by a learned goal-conditioned Q/value so that only goals the current policy can plausibly reach are kept ("frontier"). Goal-reaching policy: DDPG/TD3 + HER with sparse success reward. OMEGA interpolates toward the desired test-goal distribution. No separate intrinsic reward.
3. Episodic: per-episode goal from a fixed start; KDE and Q-filter are evaluated at s0. Continuing adaptation is natural: density over the recent buffer and the Q-filter at the *current* state s_t, goal switch on success/timeout. (MEGA's goal space is an engineered achieved-goal projection — in pixels substitute a learned latent, as LGE does.)
4. Observations: state-based (Point/Ant Maze, Fetch Push/Stack). Not pixels.
5. Robustness: order-of-magnitude sample-efficiency gains over Skew-Fit/HER in the paper; near 0% on 3-Block Stack in PEG's harder setup. Widely reused as the density-based reference baseline (PEG, DISCOVER, SVGG, GEASD).
6. Code: https://github.com/spitis/mrl (verified; PyTorch; 118 stars; sparse commits).
7. Fit: **needs adaptation** (latent goal space from pixels; continuing goal switching). With Skew-Fit it defines the "density + reachability" criterion.

### 1.8 DISCERN — discriminative embedding reward networks

1. Warde-Farley, Van de Wiele, Kulkarni, Ionescu, Hansen, Mnih. "Unsupervised Control Through Non-Parametric Discriminative Rewards." ICLR 2019. https://arxiv.org/abs/1811.11359 ; https://sites.google.com/view/discern-paper
2. Core: model-free IMPALA-style agent. Goals are *past observations* kept in a fixed-size buffer G with a diversity-preserving replacement rule (non-parametric goal proposal, roughly uniform over the buffer). Reward for goal g is the learned cosine-similarity discriminator score of the final state vs. g against negatives from G (a contrastive, learned similarity in controllable features). Policy and reward network are co-trained; HER-style relabelling with the achieved final state.
3. Episodic: goals set per episode; reward computed at episode end. Continuing: switch goal on a timer, compute the discriminative reward at goal switch, keep G updated from the stream. Nothing fundamental requires resets except the implementation.
4. Observations: pixels — Atari (several games), DeepMind Control Suite, DeepMind Lab. One of only two pixel/game autotelic results.
5. Robustness: results are qualitative/modest; LEXA (2021) reports DISCERN as a weak baseline (single seed due to cost). Skew-Fit compares against DISCERN-g and DISCERN and reports better coverage. No independent reproduction found.
6. Code: **no official release found** (GitHub and project page searched). Reimplementation only as a baseline inside LEXA's codebase (model-based variant).
7. Fit: **excluded for lack of code**, but its two ideas (goals = replayed observations; contrastive learned goal-similarity reward) are now standard components (see Contrastive RL).

### 1.9 Goal GAN — automatic goal generation at intermediate difficulty

1. Florensa, Held, Geng, Abbeel. "Automatic Goal Generation for Reinforcement Learning Agents." ICML 2018. https://arxiv.org/abs/1705.06366 ; http://proceedings.mlr.press/v80/florensa18a/florensa18a.pdf
2. Core: goal space is an *engineered* low-dim space (x,y of the ant / end-effector). A GAN generates "GOID" goals — Goals Of Intermediate Difficulty with success rate in [R_min, R_max] — by labelling sampled goals with the current policy's empirical success and training the generator on positive labels. Policy: TRPO on sparse indicator reward. No intrinsic reward.
3. Episodic: fundamentally — difficulty labels require repeated trials from the same start; the goal space is a fixed engineered box.
4. Observations: state-based Ant Maze / point mass / arm. Not pixels.
5. Robustness: GAN instability and label noise are well known; Setter-Solver (2020) and SVGG (2023) exist largely to fix Goal GAN's coverage/validity problems; Skew-Fit's "AutoGoal GAN" baseline collapses on the visual tasks.
6. Code: https://github.com/florensacc/rllab-curriculum (verified; rllab/Theano, 145 stars, inactive).
7. Fit: **excluded** (engineered goal space, episodic, dead framework). The GOID criterion survives in later methods.

### 1.10 Setter-Solver (added; DeepMind's learned-goal-space intermediate-difficulty method)

1. Racanière, Lampinen, Santoro, Reichert, Firoiu, Lillicrap. "Automated curriculum generation through setter-solver interactions." ICLR 2020. https://arxiv.org/abs/1909.12892 ; https://openreview.net/forum?id=H1e0Wp4KvH
2. Core: a Setter generates goals (in a learned goal embedding) conditioned on a desired feasibility level; a Judge predicts the Solver's success probability for a goal; the Setter is trained with three losses — goal *validity* (reconstruct achieved goals), *feasibility* (match a sampled target difficulty), *coverage* (entropy). Solver is a goal-conditioned IMPALA agent on sparse success reward. First automated goal curricula where the goal set varies per episode.
3. Episodic: yes — Judge/feasibility are defined per episode from the episode start.
4. Observations: pixel 2D and 3D (Unity) environments with colour/shape goals; moderately rich but DeepMind-internal.
5. Robustness: ablations show removing validity or coverage losses breaks learning; no independent reproduction found.
6. Code: **none found** (could not verify any release).
7. Fit: **excluded** (no code, episodic). Its validity/feasibility/coverage decomposition is the clearest statement of what an intermediate-difficulty generator must satisfy.

### 1.11 AMIGo — adversarially motivated intrinsic goals

1. Campero, Raileanu, Küttler, Tenenbaum, Rocktäschel, Grefenstette. "Learning with AMIGo: Adversarially Motivated Intrinsic Goals." ICLR 2021. https://arxiv.org/abs/2006.12122
2. Core: a teacher outputs a goal as an (x,y) cell on the (fully observed) grid; the student is a goal-conditioned IMPALA policy rewarded +1 for reaching it (plus extrinsic). The teacher gets +alpha if the student takes more than t* steps (hard enough) and -beta if fewer; t* grows as the student succeeds. Goal space is the observation grid itself, i.e. hand-designed positional goals.
3. Episodic: per-episode goal; teacher reward uses time-to-reach within an episode. Could be made continuing by resetting the per-goal timer at goal switch, but the positional goal space is the real blocker.
4. Observations: MiniGrid (symbolic grids), KeyCorridor and ObstructedMaze. Not pixels, not large.
5. Robustness: Mu et al. 2022 ("Improving Intrinsic Exploration with Language Abstractions," NeurIPS 2022) ran AMIGo on MiniHack and report that non-linguistic AMIGo often failed to learn anything on hard tasks (their L-AMIGo fix is language-based, excluded). E3B (Henaff et al., NeurIPS 2022, https://arxiv.org/abs/2210.05805) set MiniHack SOTA with a non-goal-conditioned episodic bonus; its main comparisons are RND/ICM/RIDE/NovelD (I could not verify an AMIGo row in E3B). Adversarial dynamics are sensitive to alpha/beta and the t* schedule.
6. Code: https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals (verified; TorchBeast/monobeast; archived read-only July 2025; CC-BY-NC).
7. Fit: **excluded** (hand-designed grid goal space; weak transfer evidence to pixel/open worlds).

### 1.12 Asymmetric self-play (OpenAI Alice/Bob)

1. OpenAI (Plappert, Sampedro, Xu, Akkaya, Kosaraju, Welinder, D'Sa, Petron, et al.). "Asymmetric self-play for automatic goal discovery in robotic manipulation." arXiv 2021 (NeurIPS 2020 Deep RL workshop). https://arxiv.org/abs/2101.04882 ; https://robotics-self-play.github.io/
2. Core: Alice acts from the shared initial state and her final state becomes Bob's goal. Bob is goal-conditioned PPO with sparse success reward; Alice is rewarded when Bob fails (and penalised for invalid goals). Alice's trajectory is used as a demonstration for Bob (ABC loss). Goal space = whatever states Alice reaches — learned implicitly, which is the attraction.
3. Episodic: hard requirement — Alice and Bob must start from the *same* initial state so Bob's goal is reachable; with no resets Bob would have to "return" first, which is policy-based Go-Explore.
4. Observations: state-based (object poses), large distributed setup; no pixel version.
5. Robustness: OpenAI-scale compute; Alice collapse/invalid goals needed explicit penalties; follow-ups (e.g. "It Takes Four to Tango," 2022) report instability of two-player curricula.
6. Code: **no official training code**; only environments (https://github.com/openai/robogym). The earlier Sukhbaatar et al. 2018 asymmetric self-play has code at https://github.com/tesatory/selfplay (Torch/Lua, old).
7. Fit: **excluded** (needs shared start state; no code).

### 1.13 SVGG — Stein Variational Goal Generation (brief)

Castanet, Sigaud, Lamprier. ICML 2023. https://proceedings.mlr.press/v202/castanet23a.html . Particle-based goal distribution pushed by SVGD toward regions of intermediate predicted success (learned success model) while keeping diversity; HER policy. State-based; code not verified. Shows intermediate difficulty without GANs, but episodic and state-based. **Excluded / reference only.**

### 1.14 LEXA — Latent Explorer Achiever

1. Mendonca, Rybkin, Daniilidis, Hafner, Pathak. "Discovering and Achieving Goals via World Models." NeurIPS 2021. https://arxiv.org/abs/2110.09514 ; https://orybkin.github.io/lexa/
2. Core: DreamerV2-style RSSM world model from pixels. Two policies trained in imagination: an *explorer* maximising ensemble-disagreement (Plan2Explore) reward, and an *achiever* conditioned on a goal image sampled uniformly from replay, rewarded by cosine similarity in latent space or by a learned *temporal distance* (a network trained to predict the number of steps between two states in imagined rollouts). Data collection alternates explorer and achiever episodes. Zero-shot goal-image evaluation.
3. Episodic: collection is episodic but goal selection is uniform-from-replay, so nothing requires resets; world model + imagination are continuing-friendly (DreamerV3 supports continuing tasks). Adaptation: alternate explorer/achiever on a timer, sample goals from the recent buffer.
4. Observations: 64x64 pixels; RoboBin, RoboKitchen, RoboYoga (Walker/Quadruped) — 40 goal-image tasks across 4 robotics domains. Not open world.
5. Robustness: ~2-5 GPU-days per run, ~200 GPU-days total; DISCERN and Plan2Explore baselines run with a single seed "due to large required compute." In PEG (2023), LEXA's uniform-replay goal selection is a weak goal setter (worse than Plan2Explore alone on hard tasks). Repo warns robobin breaks with other MuJoCo/Python versions.
6. Code: https://github.com/orybkin/lexa (verified; TF2; 90 stars; 18 commits; version-fragile).
7. Fit: **good in principle (pixels, world model, no reset dependence), needs a better goal selector** — exactly what PEG/MoReFree/Director do.

### 1.15 PEG — Planning Exploratory Goals (and MoReFree, its reset-free extension)

1. Hu, Chang, Rybkin, Jayaraman. "Planning Goals for Exploration." ICLR 2023 (spotlight). https://arxiv.org/abs/2303.13002 ; https://github.com/penn-pal-lab/peg . Reset-free extension: Yang, Moerland, Preuss, Plaat, Hu, "Reset-free Reinforcement Learning with World Models" (MoReFree), arXiv 2024, https://arxiv.org/abs/2408.09807 .
2. Core (PEG): LEXA-style world model with an explorer (Plan2Explore reward) and a goal-conditioned achiever. At the start of each episode PEG *plans the goal*: it optimises (MPPI/CEM in the world model) over goal commands g to maximise E[V^E(s_T)], the exploration value of the state the goal-conditioned policy will end up in after the Go phase, so the subsequent Explore phase starts from a maximally informative state. Go-Explore structure without state restore: Go phase (achiever to g), then Explore phase (explorer). MoReFree adopts PEG and adds "Back-and-Forth Go-Explore": with probability alpha (0.2) command initial/eval states as goals (hallucinated resets via the goal-conditioned policy), otherwise PEG exploratory goals; exploration and policy learning are made reset-free.
3. Episodic: PEG plans from the fixed s0 each episode. MoReFree is explicitly reset-free and is the closest published recipe for the Rain World setting: plan goals from the *current* model state and interleave "return-to-useful-state" goals. Both are reported in state space, but the architecture is pixel-native (LEXA/DreamerV2).
4. Observations: PEG — state-based Point Maze (2D), Walker (9D), Ant Maze (29D), 3-Block Stack (14D); authors call image observations a "straightforward extension" but did not evaluate it. MoReFree — state-based (6D-29D) EARL-style reset-free tasks.
5. Robustness: PEG ablations (Ant Maze) — removing the Explore phase gives 0%, random goals and replay ("seen") goals are much worse, MPPI vs CEM is irrelevant; PEG beats MEGA and Skew-Fit on all four tasks (baselines ~0% on 3-Block Stack). MoReFree/reset-free PEG beat MEDAL, IBC, R3L, DreamerV2 on 7/8 reset-free tasks without rewards or demos. Authors note dependence on world-model rollouts for long horizons and higher compute than model-free.
6. Code: https://github.com/penn-pal-lab/peg (verified; TF 2.4; built on DreamerV2 + LEXA; 83 stars; includes Skew-Fit, MEGA, LEXA, P2E baselines). MoReFree code: "in supplemental" per paper; builds on PEG (not independently located on GitHub).
7. Fit: **needs adaptation (pixels + continuing), but the adaptation is published (MoReFree)**. Strong candidate if a world model is part of the plan.

### 1.16 Director — deep hierarchical planning from pixels

1. Hafner, Lee, Fischer, Abbeel. "Deep Hierarchical Planning from Pixels." NeurIPS 2022. https://arxiv.org/abs/2206.04114 ; https://danijar.com/project/director/ ; https://research.google/blog/deep-hierarchical-planning-from-pixels/
2. Core: DreamerV2 world model. A *goal autoencoder* compresses model states into discrete codes. Every K steps the *manager* picks a code that is decoded into a latent goal; the *worker* is rewarded by max-cosine similarity between current model state and the goal in latent space. The manager maximises extrinsic reward plus an *exploration bonus = goal-autoencoder prediction error* of visited model states (novel states are poorly reconstructed), so it tends to request goals that are both decodable (achievable) and novel. Everything is trained in imagination. This is autotelic goal selection by novelty in a learned discrete goal space.
3. Episodic: **no reset dependence** — the manager re-selects a goal every K steps from the current state; DreamerV2/V3 handle continuing tasks. Rain World's respawn-in-place is just a world-model transition.
4. Observations: pixels; Egocentric Ant Maze (first-person camera + proprio, large mazes), Visual Pin Pad, DMLab goal mazes, Crafter, Atari, DMC — the broadest open-world pixel validation in this literature. Reported to match or exceed DreamerV2 across 12 standard tasks including Crafter, and to be the only method to reliably solve the large egocentric ant maze (Plan2Explore destabilised, Dreamer failed). (Details from the project page and Google blog; the NeurIPS PDF could not be text-extracted.)
5. Robustness: the *same hyperparameters* were used across all domains (blog + paper). Follow-ups: THICK (Gumbsch et al., ICLR 2024, https://github.com/CognitiveModeling/THICK) and "Exploring the limits of hierarchical world models in RL" (Sci. Reports 2024) build on / analyse Director-style hierarchies. Negatives: official repo is a single-commit TF2 snapshot; the exploration bonus is tied to the goal-AE and is not an ablated standalone intrinsic reward; hierarchical credit assignment with K=8 can be slow.
6. Code: https://github.com/danijar/director (verified; TF2; 123 stars; 1 commit; DMC + pinpad + loconav configs; no Crafter/DMLab config in the public repo).
7. Fit: **good** — the only non-LLM autotelic method demonstrated from pixels in an open-world game with no reset dependence and discrete-action support (Atari). Cost: a reimplementation on DreamerV3 (JAX) is realistic but non-trivial.

### 1.17 Choreographer (brief, borderline: skills rather than goals)

Mazzaglia, Verbelen, Dhoedt, Lacoste, Rajeswar. "Choreographer: Learning and Adapting Skills in Imagination." ICLR 2023 (notable top-25%). https://openreview.net/forum?id=PhkWyijGi5b ; https://skillchoreographer.github.io/ . World-model agent that discovers skills (code-based latent goals via a skill autoencoder over model states) in imagination from pixels; reports skills "explore the environment thoroughly." Pixel DMC/URLB; not open world. **Borderline; reference only** (skill-conditioned, no explicit goal-selection curriculum).

### 1.18 EDL — Explore, Discover and Learn (and PiCoEDL in Minecraft)

1. Campos, Trott, Xiong, Socher, Giró-i-Nieto, Torres. "Explore, Discover and Learn: Unsupervised Discovery of State-Covering Skills." ICML 2020. https://arxiv.org/abs/2002.03647
2. Core: three decoupled stages: Explore (collect a state-covering dataset, e.g. with state-marginal matching), Discover (fit a VQ-VAE over states so each code is a skill/goal region), Learn (skill-conditioned policy with reward = log p(s | z) from the VQ-VAE decoder). Goal space is learned (VQ codes); goal selection uniform over codes. PiCoEDL (Embodied AI workshop) applied EDL in MineRL with pixel + coordinate observations.
3. Episodic: stages are offline/batch; the Learn stage is standard RL and can be continuing. Decoupling means exploration does not benefit from skills online (no autocurriculum).
4. Observations: 2D mazes (state); PiCoEDL used Minecraft egocentric pixels + coordinates (coordinates = partial privileged state).
5. Robustness: overcomes the coverage problem of DIAYN-type methods, but the Explore stage inherits whatever explorer you use.
6. Code: https://github.com/victorcampos7/edl (verified via search; mirrored at imatge-upc/edl; Python 3.5-era).
7. Fit: **needs adaptation / mostly superseded** by METRA/TLDR; the VQ-code goal space idea appears in Director and LGE.

### 1.19 METRA and TLDR — temporal-distance-aware latent spaces for skills/goals from pixels

1. Park, Rybkin, Levine. "METRA: Scalable Unsupervised RL with Metric-Aware Abstraction." ICLR 2024 (oral). https://arxiv.org/abs/2310.08887 ; https://github.com/seohongpark/METRA . Bae, Park, Lee. "TLDR: Unsupervised Goal-Conditioned RL via Temporal Distance-Aware Representations." CoRL 2024. https://arxiv.org/abs/2407.08464 ; https://github.com/heatz123/tldr
2. Core (METRA): learn phi(s) with ||phi(s)-phi(s')|| <= 1 per step (temporal-distance Lipschitz constraint, dual gradient descent) and train skills z to maximise (phi(s')-phi(s))·z — covering a compact latent metrically connected to the state space by temporal distances. TLDR: goal-conditioned version — learn a temporal-distance representation, pick *far-away goals* (large temporal distance from the current state, i.e. frontier by temporal distance rather than density) to initiate exploration; the exploration policy is rewarded for increasing temporal distance, the goal-conditioned policy for decreasing it.
3. Episodic: trained on episodic locomotion; nothing in the objectives needs resets. TLDR's "far from current state" criterion is *more* natural in a continuing env than density-at-s0 criteria.
4. Observations: METRA is the first unsupervised RL method to learn diverse locomotion from pixels (DMC Quadruped/Humanoid, Kitchen pixels). TLDR: six locomotion/navigation envs incl. pixel Quadruped and Kitchen. Not open-world games.
5. Robustness: METRA has independent reproduction attempts (e.g. https://github.com/pjhae/metra_reproduce) and many follow-ups; TLDR reports large coverage gains over METRA/other unsupervised GCRL. Both are SAC-based continuous-control codebases.
6. Code: verified (METRA 96 stars; TLDR 36 stars, PyTorch, 5 commits).
7. Fit: **needs adaptation** (discrete/MultiBinary heads; continuing goal switching) — TLDR is the most recent reproducible pixel-validated goal-conditioned exploration method.

### 1.20 DDL — Dynamical Distance Learning

1. Hartikainen, Geng, Haarnoja, Levine. "Dynamical Distance Learning for Semi-Supervised and Unsupervised Skill Discovery." ICLR 2020. https://arxiv.org/abs/1907.08225 ; https://sites.google.com/view/dynamical-distance-learning
2. Core: regress d(s,g) onto the empirical number of steps between states on the current policy's trajectories; use -d as a shaped goal-reaching reward (SAC). Unsupervised variant (DDLuS): pick as goal the buffer state *farthest* under the learned distance from the start, reach it, repeat — a frontier criterion by learned temporal distance. Semi-supervised variant uses ~10 human preference labels (excluded as external supervision).
3. Episodic: on-policy distances estimated from episode segments; goal = farthest from start. Continuing adaptation: farthest from current state, distances on sliding windows.
4. Observations: pixel real-robot valve turning (semi-supervised); unsupervised results on state-based locomotion.
5. Robustness: on-policy distances go stale; "Automatic Goal Generation using DDL" (arXiv 2111.04120) and TLDR/QRL address this.
6. Code: in https://github.com/rail-berkeley/softlearning (TF-era SAC framework; DDL branch not individually verified).
7. Fit: **superseded** by TLDR/QRL/CRL; origin of the temporal-distance frontier criterion.

### 1.21 Contrastive RL (CRL), Single-Goal CRL, JaxGCRL

1. Eysenbach, Zhang, Salakhutdinov, Levine. "Contrastive Learning as Goal-Conditioned Reinforcement Learning." NeurIPS 2022. https://arxiv.org/abs/2206.07568 ; https://github.com/google-research/google-research/tree/master/contrastive_rl . Liu, Tang, Eysenbach, "A Single Goal is All You Need: Skills and Exploration Emerge from Contrastive RL without Rewards, Demonstrations, or Subgoals," ICLR 2025, https://arxiv.org/abs/2408.05804 , https://github.com/graliuce/sgcrl . Bortkiewicz et al., "Accelerating Goal-Conditioned RL Algorithms and Research" (JaxGCRL), ICLR 2025 spotlight, https://arxiv.org/abs/2408.11052 , https://github.com/MichalBortkiewicz/JaxGCRL .
2. Core: train encoders phi(s,a), psi(g) with an InfoNCE loss on (state-action, future-state) pairs from the same trajectory vs. random negatives; the inner product is a goal-conditioned Q-function (discounted future-state density ratio). Actor maximises it. No hand-designed distance, no explicit HER (relabelling is implicit). Goal *selection* in CRL is uniform from the commanded/test goal distribution; SGCRL shows that with a *single* commanded goal image the critic's errors induce directed exploration and skills before any success. JaxGCRL is a GPU-vectorised benchmark/codebase (CRL, SAC/TD3 + HER, PPO) that DISCOVER builds on.
3. Episodic: the contrastive objective samples future states within a trajectory window — defined on any stream, so continuing is fine (geometric horizon over the stream; segmenting at goal switches optional). Goal selection is the missing piece (combine with 1.6/1.7/1.19 criteria).
4. Observations: CRL includes image-based Fetch/Sawyer tasks (64x64) in the original code; SGCRL is state-based manipulation; JaxGCRL is state-only (no pixel support).
5. Robustness: widely reimplemented (JaxGCRL reports stability improvements; an "Analyzing and Simplifying Contrastive RL" repo exists); the temporal-contrastive critic is now standard (also in Temporal-Representations-for-Exploration and Episodic-Novelty-via-Temporal-Distance work). Weaknesses: horizon generalisation, large negative-batch requirements; no open-world pixel results.
6. Code: verified for all three; CRL is JAX/Acme (heavy), JaxGCRL is JAX/Brax (287 stars, Apache-2.0, active), SGCRL on GitHub.
7. Fit: **good as the goal-reaching engine** (replaces VAE/pixel distances); needs own pixel encoder and discrete-action actor; needs a goal-selection criterion on top.

### 1.22 QRL — Quasimetric RL

1. Wang, Torralba, Isola, Zhang. "Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning." ICML 2023. https://arxiv.org/abs/2304.01203 ; https://github.com/quasimetric-learning/quasimetric-rl
2. Core: parametrise the goal-conditioned value as a learned quasimetric d(s,g) (IQE/MRN) trained with a constrained objective (maximise distances subject to local one-step consistency), which provably recovers the optimal goal-reaching cost in deterministic MDPs; actor minimises d. Online and offline variants; no goal-selection mechanism.
3. Episodic: none inherent (local transition constraints on any stream).
4. Observations: state-based GCRL benchmarks (maze, Fetch) and offline D4RL; image variant in the repo not verified in detail.
5. Robustness: OGBench (2024) and the 2025 "Offline GCRL with quasimetric representations" follow-up report mixed results across datasets; quasimetric architectures assume near-deterministic dynamics (Rain World has stochastic creatures).
6. Code: verified; PyTorch.
7. Fit: **reference / component only** — a value architecture, not an autotelic method; stochasticity is a concern.

### 1.23 HIQL — hierarchical implicit Q-learning (offline)

1. Park, Ghosh, Eysenbach, Levine. "HIQL: Offline Goal-Conditioned RL with Latent States as Actions." NeurIPS 2023 (spotlight). https://arxiv.org/abs/2307.11949 ; https://github.com/seohongpark/HIQL
2. Core: one action-free goal-conditioned value V(s,g) trained with IQL-style expectile regression on offline data; a high-level policy outputs a latent subgoal representation phi(s_{t+k}) and a low-level policy reaches it; the subgoal representation is learned from the value function. Offline only; no exploration.
3. Episodic: offline — not applicable; could consume a continuing stream as a dataset.
4. Observations: state (AntMaze, Kitchen) and pixels (Procgen, CALVIN).
5. Robustness: OGBench (Park et al. 2024) standardises it and reports HIQL as a strong offline GCRL baseline; the repo points to OGBench's cleaner implementation (Dec 2024 note).
6. Code: verified; JAX; 97 stars.
7. Fit: **excluded as an exploration method**; relevant if you later train a goal-reacher offline from the continuing stream.

### 1.24 Policy-based Go-Explore (borderline) and 1.25 Latent Go-Explore

1. Ecoffet, Huizinga, Lehman, Stanley, Clune. "First return, then explore." Nature 590, 2021. https://arxiv.org/abs/2004.12919 . Gallouédec, Dellandréa. "Cell-Free Latent Go-Explore." ICML 2023. https://arxiv.org/abs/2208.14928 ; https://github.com/qgallouedec/lge
2. Core (Go-Explore): archive of cells (downscaled-pixel or domain-knowledge cells), select a cell with a count-based novelty weight, *return* to it — by simulator restore (excluded) or by a goal-conditioned policy trained from archive trajectories (policy-based variant) — then explore; later "robustify" by backward imitation. Core (LGE): removes cells; a latent representation (inverse dynamics, forward dynamics, or VQ-VAE for pixels) is learned online; goals are sampled from the buffer favouring low latent density via a geometric law over the density rank; a goal-conditioned SAC/DDPG/QR-DQN + HER policy returns by following the chain of subgoals from the trajectory that originally reached the goal, then explores.
3. Episodic: Go-Explore's "return" presupposes starting from the fixed start each episode; LGE's authors state it "assumes the agent is always initialized in the same state" and is not suited to procedurally generated envs. In a continuing env "return" degenerates into "navigate from current state to goal," i.e. plain goal-conditioned exploration with density-based goal choice (1.6/1.7 with a latent). The subgoal-chain trick remains usable if goals are chosen among states reachable from the current region.
4. Observations: Go-Explore — Atari pixels (Montezuma, Pitfall) with downscaled-pixel or hand-coded cells; LGE — Maze and Panda (vector), Montezuma's Revenge and Pitfall (pixels, VQ-VAE latent).
5. Robustness: Go-Explore's dependence on cell design is the headline critique (LGE: if the cell partitioning is not informative enough, Go-Explore can completely fail). LGE reports near-full maze coverage with low variance and beats ICM/Surprise/DIAYN/Skew-Fit baselines, with only a modest gain over Go-Explore on Atari. Policy-based Go-Explore adds training complexity and is slower than restore (Nature paper). "Post-exploration" studies (Yang et al. 2022-23) confirm the Go-then-explore structure helps but were episodic.
6. Code: LGE verified (Stable-Baselines3; MIT; Box/Image/Discrete observations, Box/Discrete actions; 438 commits, 34 stars). Go-Explore official code (uber-research/go-explore) not re-verified here.
7. Fit: Go-Explore **excluded** (state restore / hand cells / episodic); LGE **borderline, needs adaptation** — its learned-latent + density-rank goal sampling + HER goal-reacher is a clean, maintained, discrete-action-capable implementation of the Skew-Fit/MEGA criterion; drop the same-start assumption and the subgoal-chain return.

### 1.26 DISCOVER — value-uncertainty-directed goal curricula (NeurIPS 2025)

1. Diaz-Bone et al. "DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning." NeurIPS 2025. https://arxiv.org/abs/2505.19850 ; https://github.com/LeanderDiazBone/discover
2. Core: an ensemble of goal-conditioned critics (on JaxGCRL's CRL/TD3+HER); the training goal each episode maximises V(s0,g) (achievable) + sigma(s0,g) (uncertain = informative) + a relevance term toward a target goal g* (directed). The undirected special case (drop relevance) is a pure "achievable and uncertain" autotelic criterion.
3. Episodic: goals chosen at s0 per episode; continuing adaptation = evaluate V and sigma at the current state.
4. Observations: state-based JaxGCRL mazes/manipulation.
5. Robustness: reports solving long-horizon mazes beyond MEGA/PEG-style baselines; 11-star repo; no independent reproduction yet.
6. Code: verified; JAX/Brax; CUDA >= 12.3.
7. Fit: **needs adaptation** (pixels, continuing); the "value + value-uncertainty" criterion is the modern successor of MEGA's reachability filter.

### 1.27 LEO — Learning Everything all at Once (ICML 2026; goal-conditioned Craftax)

1. Matthews, Jackson, Beukman, Foster, Letcher, Fujimoto, Colas, Foerster. "Goal-Conditioned Agents that Learn Everything All at Once." ICML 2026. https://arxiv.org/abs/2605.23551 ; https://github.com/MichaelTMatthews/purejaxgcrl
2. Core: a network outputs values/actions for *every* goal at once, enabling all-goals updates without relabelling (>250x faster than all-goals HER). CraftaxGC: 136-512 abstract goals (inventory counts, adjacency, tools, dungeon progression) over *symbolic* Craftax observations; training goals sampled uniformly from goals observed at least once (implicit autocurriculum), with "first return then explore"-style pseudo-termination and goal resampling in place when a goal is achieved. Baselines: PQN(+HER), PPO, CRL, UVFA variants.
3. Episodic: Craftax episodes, but goal switching is *in place* on achievement — exactly the mechanic a continuing env needs.
4. Observations: symbolic Craftax; goals are hand-specified semantic conditions (violates the learned-goal-space constraint).
5. Robustness: new (May 2026); 33-star repo; single-file JAX.
6. Code: verified; JAX; MIT.
7. Fit: **excluded as a goal space (hand-specified semantic goals, symbolic obs)**; useful engineering pattern (all-goals heads, in-place goal switching) and the only recent open-world GCRL result with an autotelic-agents author (Colas).

### 1.28 ULEE — self-imposed goals for meta-exploration (ICLR 2026)

Pappalardo. "Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals." ICLR 2026. https://arxiv.org/abs/2601.19810 ; https://github.com/Octavio-Pappalardo/ulee-jax . Adversarial goal generator keeps goals at the frontier of an in-context learner's competence; XLand-MiniGrid (grid, not pixels); episodic meta-RL. **Excluded** (gridworld, meta-episodic); noted as 2026 evidence that intermediate-difficulty generation is still used without LLMs.

### 1.29 Learning-progress intrinsic rewards (not goal-conditioned, but bears on criterion choice)

Hou, An, Du. "Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring" (LPM). ICLR 2026. https://arxiv.org/abs/2509.25438 ; https://github.com/Akuna23Matata/LPM_exploration . Rewards improvement in dynamics-model error (an error model predicts the previous iteration's error); shown to be a monotone indicator of information gain and robust to noisy-TV; 160x120 RGB 3D maze and Atari. Earlier: Becker-Ehmck et al., "Exploration via Empowerment Gain: Combining Novelty, Surprise and Learning Progress" (OpenReview 2021). Relevant because Rain World has heavy uncontrollable stochasticity (creatures, rain cycle), where novelty/density criteria are fooled and LP-type criteria are the principled fix.

### 1.30 Other 2024-2026 items checked (brief)

- GEASD (Wu, Chen, arXiv 2404.12999): adaptive skill distribution maximising *local* entropy of achieved goals; state-based; no code found.
- Probabilistic Curriculum Learning for goal-based RL (Salt, Gallagher, arXiv 2504.01459): continuous control/navigation; no code found.
- MUN (Duan, Mao, Zhu, NeurIPS 2024, arXiv 2411.02446): world model that models transitions between arbitrary replay "key" states to improve goal-navigation generalisation; PEG-adjacent; observation type not verified.
- Diversity Progress for goal selection (Lintunen, Ady, Guckelsberger, IMOL@NeurIPS 2024, arXiv 2411.01521): LP-style selection over skill discriminability; small-scale.
- SEA (Zhou, Garg, ICLR 2023, arXiv 2305.00508): learns Crafter's achievement structure from *offline* data then explores structurally from pixels; no language, but needs offline achievement labels — **excluded** (external supervision).
- Temporal Representations for Exploration (Mohamed, Ji, Eysenbach, Berseth, arXiv 2603.02008, 2026): temporal-contrastive representations prioritising states with unpredictable futures; locomotion/manipulation/embodied-AI; not goal-conditioned; no code URL on the abstract page.
- Episodic Novelty Through Temporal Distance (Jiang et al., ICLR 2025, arXiv 2501.15418): contrastive temporal distance as the metric for episodic novelty bonuses in contextual MDPs; not goal-conditioned.
- RAMP, "Exploration by Running Away from the Past" (Le Tolguenec et al., arXiv 2411.14085): state-entropy via divergence from past occupancy; not goal-conditioned.
- Decoupling Exploration and Policy Optimization (Mhammedi, Cohan, arXiv 2603.22273, 2026): Go-With-The-Winner tree search + uncertainty, then distillation; Montezuma/Pitfall/Venture — relies on branching/restoring trajectories, so **excluded** for the continuing setting.
- Lidayan et al., "Intrinsically-Motivated Humans and Agents in Open-World Exploration" (arXiv 2503.23631, 2025): in Crafter, entropy and empowerment correlate with human exploration progress; entropy plateaus early while empowerment keeps rising — suggests state-diversity criteria early and control/competence criteria later.
- Craftax paper (Matthews et al., ICML 2024, arXiv 2402.16801): RND, ICM and E3B did not improve (sometimes hurt) PPO on Craftax — a caution that generic novelty bonuses can be noise in open worlds with rich internal structure.
- EARL benchmark (Sharma et al., ICLR 2022, arXiv 2112.09605): the standard formalism/benchmark for reset-free ("autonomous") RL; MoReFree is evaluated in this style.

### 1.31 Excluded (external knowledge / LLM-based) — flagged only

ELLM (Du et al., ICML 2023, arXiv 2302.06692; LLM-suggested goals in Crafter/Housekeep); OMNI (Zhang, Lehman, Clune, arXiv 2306.01711; LLM model of interestingness, reports LP-alone insufficient in large goal spaces) and OMNI-EPIC (arXiv 2405.15568); MAGELLAN (arXiv 2502.07709; metacognitive LP predictions for LLM autotelic agents); Voyager (arXiv 2305.16291; GPT-4 Minecraft); Motif (arXiv 2310.00166; LLM-preference intrinsic reward in NetHack); L-AMIGo / L-NovelD (Mu et al., NeurIPS 2022). All excluded by the no-language-prior constraint; several report that *LP-only* or *novelty-only* goal selection underperforms in very large goal spaces, which is informative about criterion limits even though their fix is excluded.

---

## 2. Cross-cutting: running these methods in a continuing, respawn-in-place environment

What breaks: (a) per-episode goal selection evaluated at a fixed s0 (Skew-Fit, MEGA, Goal GAN, Setter-Solver, AMIGo, PEG, DISCOVER, LGE); (b) "return" by restore or by shared start (Go-Explore, asymmetric self-play, LGE's subgoal chain); (c) success-rate/LP estimators that assume repeated trials of the same goal from the same start (Goal GAN, CURIOUS, SVGG, ULEE).

What does not break: HER / contrastive goal-reaching losses (segment the stream at goal switches or use geometric future sampling); VAE/VQ/temporal-distance goal-space learning from the stream; density/entropy skewing over a sliding replay window; world-model training and imagination (Dreamer family is continuing-native); manager-style re-selection every K steps (Director).

Minimal adaptation recipe supported by published pieces: (1) command a goal, run until success or timeout, then re-select *from the current state* (LEO's pseudo-termination; MoReFree's back-and-forth); (2) choose goals from the recent buffer by low density or large temporal distance (Skew-Fit/MEGA/TLDR), *filtered by reachability from the current state* via V(s_t,g) or an ensemble (MEGA's Q filter; DISCOVER's V+sigma); (3) optionally plan the goal in a world model to maximise exploration value from the current latent (PEG objective at s_t instead of s0, as MoReFree does); (4) for LP, track per-goal-cluster success rates with a timeout and use absolute LP over clusters of the learned latent (CURIOUS module selection with clusters instead of hand modules). Death/respawn is just a transition; the goal-reacher's value function learns that death is a costly detour only if the respawn location is far (in temporal distance) from most goals — worth monitoring.

Action space: most GCRL code is continuous-control (SAC/TD3/Brax). Implementations that already support discrete action heads: LGE (SB3, Discrete), Director/Dreamer (Atari), AMIGo (IMPALA), LEO (Craftax). A 9-key MultiBinary head is a factorised-Bernoulli actor in any of these, but expect to write it yourself.

---

## 3. (a) Which non-LLM autotelic methods have been demonstrated from pixels in large open worlds?

Honest answer: very few.
- **Director** (NeurIPS 2022): pixels, Crafter + DMLab + Atari + egocentric Ant Maze, one hyperparameter set — the only clear case of a self-goal-setting agent in open-world-style pixel games.
- **DISCERN** (ICLR 2019): pixels, Atari + DMLab; modest, no code.
- **Latent Go-Explore** (ICML 2023): pixels (VQ-VAE) on Montezuma/Pitfall — hard-exploration but not open world, and explicitly episodic/same-start.
- **LEXA, Skew-Fit/RIG, METRA, TLDR, Choreographer**: pixels, but closed robotics/locomotion scenes.
- **Open-world but not pixels / not learned goals**: LEO (symbolic Craftax, hand-specified semantic goals); SEA (Crafter pixels, offline achievement labels); PEG/MoReFree/MEGA/DISCOVER (state-based).
- Everything in Minecraft/MineRL that sets its own goals in 2023-2026 is LLM-driven (Voyager, ODYSSEY, MineExplorer) except PiCoEDL (workshop; EDL on pixels+coordinates).
Gap: no published non-LLM method demonstrates learned-goal-space autotelic exploration from pixels in a large open world under a continuing regime. Whatever you run will be partly new; the closest fully-published scaffold is Director (pixels, open world, no resets) and the closest reset-free recipe is MoReFree (state-based).

## 4. (b) Best-supported goal-selection criterion

- **Density / entropy of achieved goals ("frontier")**: the most repeatedly validated in deep GCRL (Skew-Fit ICML'20 with an alpha-robustness sweep; MEGA ICML'20; LGE ICML'23; TLDR CoRL'24 with temporal distance instead of density). Theory: converges to uniform coverage (Skew-Fit). Weakness: ignores reachability unless filtered (MEGA Q-filter), and can be fooled by uncontrollable stochasticity (Rain World creatures/rain) — Craftax results show generic novelty signals can hurt.
- **Exploration value / value uncertainty of the goal**: PEG (ICLR'23) beats MEGA and Skew-Fit on all four of its tasks; DISCOVER (NeurIPS'25) beats density-style baselines on long-horizon mazes. These are the best *controlled* comparisons available, but state-based and episodic.
- **Learning progress**: strongest in population-based IMGEP (JMLR'22), CURIOUS (robust to distractors/forgetting), human studies (Poli et al. 2024; Molinaro et al. 2024), and noise robustness (LPM, ICLR'26). No deep-GCRL-from-pixels result shows LP goal selection beating density; LP needs a goal partition (modules/clusters) and enough repeated attempts per partition to estimate competence, which is expensive in a continuing open world. OMNI (LLM, excluded) reports LP-alone is insufficient in very large goal spaces.
- **Intermediate difficulty / adversarial** (Goal GAN, Setter-Solver, AMIGo, asymmetric self-play, SVGG, ULEE): works in its papers, but is the family with the most reported brittleness (GAN instability, teacher/student collapse, episodic success-rate estimation, AMIGo failing on hard MiniHack without language). Not recommended as a first criterion.
Verdict: the best-supported *and* simplest is density/temporal-distance frontier with a reachability (value) filter; the best-performing in controlled studies is exploration-value / value-uncertainty (PEG, DISCOVER); LP is the principled choice for noise robustness and should be layered over clusters later. There is no pixel-scale head-to-head; treat this as an open question your environment can actually inform.

## 5. (c) What to implement first

1. **Director-style manager/worker on a DreamerV3 world model** (Hafner et al. 2022; danijar/director TF2 as reference; reimplement on DreamerV3 JAX). Why: the only method demonstrated from pixels in open-world games with one hyperparameter set, discrete-action capable, no reset dependence (goal re-selected every K steps from the current latent), interpretable (decode goals to images). Risk: compute; single-commit reference code; the goal-AE error bonus is the only exploration signal (consider swapping in Plan2Explore disagreement, already present in Dreamer codebases).
2. **Model-free "learned latent + density-skewed goal sampling + goal-conditioned policy" (Skew-Fit/MEGA/LGE lineage), with a contrastive temporal critic (CRL) as the goal-reaching reward instead of VAE-pixel distance.** Reference code: qgallouedec/lge (SB3, image obs, discrete actions, MIT) for the goal sampler and HER plumbing; JaxGCRL / google-research contrastive_rl for the critic. Continuing adaptation: goal switch on success/timeout, density over a sliding window, reachability filter V(s_t,g). Why: cheapest to get running, every component is published and reproduced, and it gives the density-vs-reachability baseline against which the others should be judged.
3. **PEG with MoReFree's reset-free back-and-forth, on the same world model as (1)** (penn-pal-lab/peg TF2; MoReFree arXiv 2408.09807). Why: the best controlled evidence for a goal-selection criterion (exploration value) and the only published reset-free goal-exploration recipe; shares most code with (1). Risk: validated only in state space; MPPI over goals in a large latent may need many imagined rollouts.
Then, as ablations rather than first steps: a CURIOUS-style absolute-LP selector over k-means clusters of the learned latent (replacing hand modules), and TLDR's temporal-distance frontier as an alternative to density.

---

## 6. Source list (verified during this review)

arXiv: 1708.02190, 1803.00781, 1810.06284, 2012.09830, 1807.04742, 1903.03698, 2007.02832, 1811.11359, 1705.06366, 1909.12892, 2006.12122, 2101.04882, 2110.09514, 2303.13002, 2408.09807, 2206.04114, 2002.03647, 2310.08887, 2407.08464, 1907.08225, 2206.07568, 2408.05804, 2408.11052, 2304.01203, 2307.11949, 2004.12919, 2208.14928, 2505.19850, 2605.23551, 2601.19810, 2509.25438, 2404.12999, 2504.01459, 2411.02446, 2411.01521, 2305.00508, 2603.02008, 2501.15418, 2411.14085, 2603.22273, 2503.23631, 2402.16801, 2112.09605, 2210.05805, 2003.04664, 2302.06692, 2306.01711, 2502.07709, 2305.16291, 2310.00166.

Repos fetched or confirmed via search: flowersteam/Unsupervised_Goal_Space_Learning, flowersteam/curious, vitchyr/rlkit, spitis/mrl, florensacc/rllab-curriculum, facebookresearch/adversarially-motivated-intrinsic-goals, orybkin/lexa, penn-pal-lab/peg, danijar/director, victorcampos7/edl, seohongpark/METRA, heatz123/tldr, google-research/google-research/contrastive_rl, graliuce/sgcrl, MichalBortkiewicz/JaxGCRL, quasimetric-learning/quasimetric-rl, seohongpark/HIQL, qgallouedec/lge, LeanderDiazBone/discover, MichaelTMatthews/purejaxgcrl, CognitiveModeling/THICK, Octavio-Pappalardo/ulee-jax, Akuna23Matata/LPM_exploration.

Not found / could not verify: official code for DISCERN, Setter-Solver, asymmetric self-play (training), SVGG, GEASD, PCL; the IMGEP JMLR companion repo URL; PEG/Skew-Fit/Director/LGE full PDFs were locked downloads — PEG, Skew-Fit and LGE details come from ar5iv renderings, Director details from the project page and Google Research blog.
