# Competence-based intrinsic motivation, unsupervised skill discovery and zero-shot RL: a literature review for a pixel-based, continuing, open-world environment (Rain World)

Date: 2026-10-04. Stream: competence-based methods (skill discovery, empowerment, zero-shot / unsupervised RL).

## 0. Scope, constraints and how to read this document

Target setting: pixel observations, continuing (no resets; death = respawn in place; one long lifetime), large open world, 9-key MultiBinary action space. Hard constraints on acceptable methods:

- C1: no game-specific tricks (no hand-tuned action repeat, hand-designed goal spaces or cell representations);
- C2: must work (or be adaptable) in continuing, non-episodic settings;
- C3: must handle large, continuous pixel observation spaces;
- C4: no external knowledge (no LLMs/VLMs, language priors, pretrained foundation models, demonstration or reference data).

Verification policy. Every arXiv abstract, project page and GitHub repository marked "verified" below was fetched on 2026-10-04. Where I could not fetch something (e.g., anonymous code links, a PDF that failed to download) I say so explicitly. Numbers are quoted from the papers' text; where a figure (not a table) is the only source, I say "figure" and describe qualitatively. Statements marked "[analysis]" are my own inference, not a citation.

Each method entry follows the requested template: (1) identity, (2) core algorithm, (3) episodic assumptions / continuing-env adaptation, (4) observation type validated, (5) robustness / brittleness evidence, (6) code, (7) fit assessment.

---

## 1. Executive summary

1. The URLB taxonomy (knowledge- / data- / competence-based) remains the lens of the field. On the original URLB (Laskin et al., 2021), "there is no competence-based approach that achieves state-of-the-art mean performance on any of the URLB tasks", and "most algorithms lose 20-50% when learning from pixels compared to state". The authors attribute the competence-based failure to tiny skill spaces (DIAYN 16-d, SMM 4-d, APS 10-d) and to discriminators that latch onto "different configurations of the agent lying on the ground".
2. The pre-2022 competence family (VIC, DIAYN, DADS, APS, SMM) is now well understood to maximise mutual information in a way that is invariant to how far the agent travels; it yields locally distinguishable, often static behaviours rather than coverage (LSD's "lower-hanging fruit" argument; EDL's "poor coverage of the state space" critique; CIC's "max H(z) does not imply max H(tau)").
3. The distance-maximising family (LSD 2022, CSD 2023, METRA 2024, HILP 2024) replaced the MI objective by "move as far as possible in a metric latent space". METRA is the only one of these validated from pixels online (64x64x3 Quadruped, Humanoid, Kitchen; no proprioception) and is "the first unsupervised RL method that discovers diverse locomotion behaviors in pixel-based Quadruped and Humanoid". But the pixel locomotion results used gradient-coloured floors "to allow the agent to infer its location from pixels" and fixed-length episodes (200/400/50 steps) with one skill per episode.
4. Evidence that METRA covers a large world uniformly is weak. METRA's own coverage metric is x-y bin occupancy (a 2-D projection). Follow-ups show it covers along a few "primary directions" (RSD, ICML 2025), prefers fast-moving skills (PSD, NeurIPS 2025; D3, CoRL 2025), and is clearly beaten on AntMaze-Large / AntMaze-Ultra / Quadruped-Escape by TLDR (CoRL 2024) and on mazes by LEADS (NeurIPS 2024; METRA 54.8% hard-maze coverage vs LEADS 90.8%, and below LSD 70.0% and CSD 64.0%).
5. Against knowledge-based bonuses, the picture is mixed and environment dependent: METRA beats RND/APT/ICM/P2E/LBS/APS on total coverage in state-Ant and pixel-Humanoid (METRA Fig. 8) while "pure exploration methods also work decently in the pixel-based Kitchen"; with a world model, knowledge-based LBS/P2E are best on URLB from pixels and "skill-driven methods ... tend to have more limited exploration capabilities" (Rajeswar et al., ICML 2023). In mazes RND is the worst method in LEADS' table (30.8% hard maze). The strongest open-world coverage results in 2025-2026 come from temporal-distance representations used as an exploration reward (TLDR; C-TeC in Craftax-Classic, non-episodic, but with symbolic not pixel observations), not from skill discovery per se.
6. Zero-shot RL (forward-backward representations, HILP, FRE, TD-JEPA) is primarily an offline paradigm; it does not solve exploration. Touati et al. (2023) state that "for large problems, some form of prior seems unavoidable"; all their environments were deterministic and state-based. Online FB variants exist (DVFB ICLR 2025; FB-EE RLJ 2025; FB-MEBE 2026) but are validated only on state-based DMC/URLB or quadruped simulation. TD-JEPA (2025) reports FB and HILP underperform from pixels. FB-CPR / Meta Motivo and DoDont rely on external data (motion capture / instruction videos) and are excluded by C4.
7. Recommendation (details in Section 4c): implement METRA first (official PyTorch/garage code; simplest reward; the only pixel-validated online skill method), with explicit adaptations for the continuing setting (skill resampling every K steps, masking death/respawn transitions from the adjacency constraint, discrete-action backbone). Second, TLDR (same codebase; adds kNN-entropy goal selection on the same temporal-distance representation; strongest maze/large-world coverage evidence). Run a knowledge/data-based control (RND or APT from the URLB codebase) alongside, since every paper that reports coverage shows the answer is environment dependent. Treat HILP/FB as an offline add-on on the replay buffer, not as an exploration driver.

---

## 2. Benchmarks and what they say about knowledge- vs data- vs competence-based methods

### 2.1 URLB (Laskin, Yarats, Liu, Lee, Zhan, Lu, Cang, Pinto, Abbeel; NeurIPS 2021 Datasets & Benchmarks). arXiv:2110.15191 (verified). Code: https://github.com/rll-research/url_benchmark (verified; MIT; 368 stars; PyTorch, DrQ-v2/DDPG; states and pixels; agents ICM, RND, Disagreement, ProtoRL, APT, APS, DIAYN, SMM).

- Protocol: reward-free pre-training on Walker / Quadruped / Jaco (DMC), snapshots at 100k, 500k, 1M, 2M steps, then 100k steps of fine-tuning on 12 downstream tasks. DDPG for states, DrQ-v2 for pixels.
- Categories: knowledge-based (ICM, RND, Disagreement), data-based (APT, ProtoRL), competence-based (DIAYN, APS, SMM).
- Findings (quoted from the paper): "there is no competence-based approach that achieves state-of-the-art mean performance on any of the URLB tasks"; "most algorithms lose 20-50% when learning from pixels compared to state"; from pixels, ICM was the leading method; "for 9 out of 18 experiments ... performance either does not improve or even degrades as a function of pre-training steps", which the authors call potentially "the biggest drawback of current unsupervised RL approaches". The explanation offered for competence-based failure: small skill dimensions (DIAYN 16, SMM 4, APS 10) and that in fixed-length episodes "the most likely skills to be classified are different configurations of the agent lying on the ground", requiring "more powerful discriminators".

### 2.2 ExORL (Yarats, Brandfonbrener, Liu, Laskin, Abbeel, Lazaric, Pinto; 2022). arXiv:2201.13425 (verified). Code: https://github.com/denisyarats/exorl.

- Nine data-collection policies (Random, ICM, Disagreement, RND, Proto, APT, APS, DIAYN, SMM) on Walker, Cheetah, Jaco, Cartpole, PointMass Maze; state observations only.
- Findings: ProtoRL and ICM datasets gave the strongest offline TD3 results; DIAYN data was best "in low-data settings" while "in the high-data regime ICM begins to dominate"; on Jaco Reach "random data collection surprisingly outperforms all of the unsupervised exploration algorithms".
- Relevance: ExORL's RND buffers became the de-facto pre-training data for the zero-shot RL literature (Section 3.12), which means FB/HILP results are conditional on a knowledge-based explorer having collected the data.

### 2.3 Mastering URLB from pixels (Rajeswar, Mazzaglia, Verbelen, Piche, Dhoedt, Courville, Lacoste; ICML 2023). arXiv:2209.12016 (verified). Project: https://masteringurlb.github.io/

- A DreamerV2-style world model is pre-trained with each of eight intrinsic rewards (ICM, LBS, Plan2Explore, RND, APT, DIAYN, APS, random), then fine-tuned with a hybrid planner (Dyna-MPC). Reaches "93.59% overall normalized performance".
- Findings: LBS best overall; "data-based and knowledge-based methods are more effective in the Walker and Quadruped domains, and random actions and competence-based are more effective in the Jaco domain"; "skill-driven methods ... tend to have more limited exploration capabilities falling behind other approaches in some domains (i.e. Walker and Quadruped) of URLB".
- Takeaway: from pixels, the world model is doing most of the work; the choice of intrinsic reward matters less than the representation, and competence-based rewards are at best domain dependent.

### 2.4 CIC's diagnosis of competence-based methods (Laskin et al., 2022; see 3.7)
"most competence-based approaches optimize H(z) - H(z|tau)" and "while this ensures diversity of skill vectors it does not ensure diverse behavior from the policy, meaning max H(z) does not imply max H(tau)"; and "to learn an accurate discriminator q(z|tau), the above methods assume skill spaces that are much smaller than the state space".

---

## 3. Method entries

### 3.1 METRA (deep dive)

1. Identity. METRA: Scalable Unsupervised RL with Metric-Aware Abstraction. Seohong Park, Oleh Rybkin, Sergey Levine. ICLR 2024. arXiv:2310.08887 (v1 Oct 2023, v2 Mar 2024; verified). Project: https://seohong.me/projects/metra/

2. Core algorithm.
- Objective: a Wasserstein dependency measure I_W(S;Z) = W(p(s,z), p(s)p(z)) in place of the Shannon MI used by DIAYN/DADS. Via Kantorovich-Rubinstein duality with score f(s,z) = phi(s)^T psi(z) and psi(z) = z, the paper arrives at the simplified objective
  sup_{pi,phi} E_{p(tau,z)} [ sum_t (phi(s_{t+1}) - phi(s_t))^T z ]  s.t.  ||phi(s) - phi(s')||_2 <= 1 for all adjacent (s,s').
- The constraint is the key idea: "adjacent" means one environment step, so the Lipschitz constraint is with respect to temporal distance ("the minimum number of environment steps to reach s2 from s1"), not Euclidean state distance (LSD) or controllability distance (CSD). This makes the constraint invariant to the observation parameterisation and therefore applicable to pixels. phi: S -> Z (Z = R^D, D in {2,4} for continuous skills) is a "metric-aware abstraction": a compact latent space whose Euclidean geometry lower-bounds temporal distance.
- Constraint enforcement: dual gradient descent with a Lagrange multiplier lambda and relaxation constant eps: phi maximises E[(phi(s')-phi(s))^T z + lambda * min(eps, 1 - ||phi(s)-phi(s')||^2)], lambda minimises E[lambda * min(eps, 1 - ||phi(s)-phi(s')||^2)]. Hyperparameters: eps = 1e-3, initial lambda = 30 (Table 1 of the paper).
- Intrinsic reward: r(s, z, s') = (phi(s') - phi(s))^T z. Nothing else (no entropy bonus, no discriminator).
- Skill sampling (Appendix F.2, quoted): "continuous skills are sampled from the standard Gaussian distribution, and discrete skills are uniformly sampled from the set of zero-centered one-hot vectors"; "METRA and LSD use normalized vectors (i.e., z/||z||_2) for continuous skills, as their objectives are invariant to the magnitude of z". So continuous z is effectively uniform on the unit sphere. Dimensions used: "2-D continuous skills for Ant and Humanoid, 4-D continuous skills for Quadruped, 16 discrete skills for HalfCheetah, and 24 discrete skills for Kitchen".
- RL backbone: vanilla SAC, trained jointly with phi from a replay buffer. Table 1: lr 1e-4 (Adam); 8 episodes per epoch; gradient steps per epoch 200 (Quadruped, Humanoid), 100 (Kitchen), 50 (Ant, HalfCheetah); minibatch 256; gamma 0.99; replay buffer 1e6 (Ant, HalfCheetah), 1e5 (Kitchen), 3e5 (Quadruped, Humanoid); CNN encoder; 2 hidden layers x 1024; target smoothing 0.995; entropy coefficient 0.01 (Kitchen) or auto-tuned. Update-to-data ratio is deliberately small: "1/4 for Kitchen and 1/16 for Quadruped and Humanoid".
- Zero-shot goal reaching: set z = (phi(g) - phi(s)) / ||phi(g) - phi(s)|| (continuous) or z = argmax_dim(phi(g) - phi(s)) (discrete); "We re-compute z every step for locomotion environments, but in Kitchen, we use the same z selected at the first step throughout the episode".
- Hierarchical downstream use: a high-level policy pi^h(z | s, s_task) "selects a skill every K = 25 (Ant and HalfCheetah) or K = 50 (Quadruped and Humanoid) environment steps"; PPO for discrete skills, SAC for continuous.

3. Episodic assumptions. Strongly episodic in the paper: "When sampling a trajectory, we first sample a skill from the prior distribution, z ~ p(z), and then roll out a trajectory with pi(a|s,z), where z is fixed for the entire episode." Episode lengths: "200 for Ant and HalfCheetah, 400 for Quadruped and Humanoid, and 50 for Kitchen" (Appendix F.1); the agent is reset to a start state each episode, and the x-y coverage metric is measured relative to that start. The authors explicitly state they "do not use ... early termination". The paper also states METRA "assume[s] a fixed MDP (i.e., stationary, fully observable dynamics)".
   Continuing-environment adaptation [analysis, with supporting citations]:
   - Resample z every K steps from the prior (K on the order of the training episode length used in the paper, 200-400, or the hierarchical K = 25-50). Cathomen et al. (D3, CoRL 2025), who train a METRA-style objective in 2048 parallel Isaac Lab environments, report "we resample the skill multiple times within an episode. Without this resampling, we observe that agents tend to 'lock in' to the initial skill". This is direct evidence that intra-episode resampling is workable.
   - Rewards and the phi constraint only use (s, s', z) transitions, so no reset is needed for learning; the replay buffer just needs transition tuples. The only objects that are episode-dependent are the SAC bootstrapping (use no terminal, continuing discounting) and the skill-switch boundaries (treat a skill switch as a soft boundary: either discard the transition straddling the switch or store z per transition, which the reward formula already supports).
   - Death = respawn is a hazard specific to METRA's constraint: a (s_death, s_respawn) pair is one "environment step" but may be temporally far apart in the forward dynamics. If such pairs are included in S_adj, the constraint forces ||phi(s_death) - phi(s_respawn)|| <= 1 and collapses the latent geometry between the death location and the respawn location. These transitions should be excluded from the constraint set (they are still fine to store for the critic with appropriate handling). This is my inference; no paper studies METRA with teleport transitions.
   - Asymmetric dynamics (falls, one-way drops, which are abundant in platformers) are explicitly listed by the authors as a limitation: the symmetric Euclidean embedding "considers the minimum of both temporal distances" and "might be overly restrictive in highly asymmetric environments"; they suggest replacing the Euclidean norm with an asymmetric quasimetric (Wang et al., 2023, QRL).
   - The action space: the official code uses continuous SAC. A 9-key MultiBinary space needs a discrete/multi-binary SAC variant or an on-policy backbone; GitHub issue #8 ("Adapting METRA to On-Policy Algorithms for Massively Parallel Simulation") shows others have tried this, and D3 trains a METRA-style objective with on-policy PPO in Isaac Lab, so the reward is backbone-agnostic.

4. Observation type validated. State-based Ant (29-D) and HalfCheetah (18-D); pixel-based (64x64x3, "we do not use any proprioceptive state information") Quadruped and Humanoid from DMC and a pixel Kitchen (LEXA camera). Action repeat 2 for pixel DMC. Important caveat for C1: "In DMC locomotion environments, we use gradient-colored floors to allow the agent to infer its location from pixels". Environments are single-arena locomotion or a single kitchen scene, i.e., not open worlds; total training was on the order of 1e7 steps for state Ant/HalfCheetah, ~6e6 for pixel Quadruped, ~1e7 for pixel Humanoid and ~1e6 for Kitchen (Figure 5 x-axes). Not evaluated on Atari or any game: "we have not evaluated METRA on other different types of environments, such as Atari games".

5. Robustness / brittleness evidence.
   - Within-paper: 8 seeds for coverage, 4 for downstream; coverage measured as 1x1 x-y bins occupied by 48 deterministic rollouts of 48 sampled skills ("policy state coverage"); "total state coverage" over all training trajectories used for the exploration-method comparison. METRA "achieves the best coverage in most of the environments"; "pure exploration methods also work decently in the pixel-based Kitchen, they fail to fully explore the state spaces of state-based Ant and pixel-based Humanoid".
   - Independent reproductions: a Princeton COS435 course project reports re-implementing METRA and reproducing the main claims (https://github.com/tanushreebanerjee/re_METRA_cos435, found via search; the PDF itself returned 404 when fetched, so I could not inspect which environments). TLDR (Bae et al., CoRL 2024), RSD (Zhang et al., ICML 2025), PSD (Park et al., NeurIPS 2025), DoDont (Kim et al., NeurIPS 2024), LEADS (Le Tolguenec et al., NeurIPS 2024) and D3 (Cathomen et al., CoRL 2025) all re-ran METRA as a baseline, several directly on the official codebase; none reports failing to reproduce the locomotion results.
   - Documented failure modes and critiques: (i) RSD: METRA "tends to explore primary directions, consistent with the PCA-like theoretical explanations ... this feature leads to homogeneous skills, resulting in a less diverse skill set", and "as the state dimension increases ... the efficiency of the LSD methods decreases"; RSD beats METRA on Antmaze-large zero-shot navigation by 15% success and on pixel Kitchen (5.08 vs 3.94 tasks at 400k steps). (ii) PSD: "METRA employs a temporal distance as its metric and thus strongly prefers fast-moving skills to maximize temporal state deviations", so multi-timescale / periodic behaviours are missed. (iii) D3 (robotics): the alignment objective "leads to skills that move maximally fast through the state space, which might not be desirable"; their norm-matching replacement "has a considerably weaker alignment component, making it difficult to train from scratch". (iv) TLDR: "METRA learns low-dimensional skills and extends the temporal distance along a few directions specified by the skills, providing a strong inductive bias for simple locomotion tasks like HalfCheetah. On the other hand, TLDR achieves much larger state coverage in complex environments than METRA, including AntMaze-Large, AntMaze-Ultra, and Quadruped-Escape." (v) LEADS Table 1 (state-based, coverage %): METRA 92.8 (easy maze), 78.0 (U-maze), 54.8 (hard maze), 68.7 (Fetch-Reach), 80.0 (Finger), 50.7 (Fetch-Slide), versus LEADS 100 / 96.0 / 90.8 / 88.6 / 81.6 / 91.3, LSD 99.8 / 79.8 / 70.0 / 54.5 / 72.7 / 63.5 and CSD 97.4 / 79.8 / 64.0 / 70.8 / 93.0 / 65.1. METRA is not the best distance-based method in mazes in that study.
   - Hyperparameter sensitivity: not systematically studied in the paper. The two METRA-specific knobs (eps, initial lambda) are reported as single values; the skill dimension is set per environment by hand (2 / 4 / 16 / 24), which is a form of problem knowledge [analysis].
   - Sample efficiency: the authors concede UTD "1/4 for Kitchen and 1/16 for Quadruped and Humanoid" and that "there is room for improvement in terms of sample efficiency".
   - GitHub issues (16 total, fetched): #2 "Reproducing downstream performance of METRA" (user could not find the downstream-learning code; closed), #13 hyperparameters for SAC on HalfCheetahGoal, #16 "Problems with training parameterization" (Jan 2026), #15 / #5 dependency conflicts (the garage stack is dated), #8 on-policy adaptation.

6. Code. Official: https://github.com/seohongpark/METRA (verified; MIT; 96 stars, 17 forks; Python 3.8; PyTorch on a vendored garage fork; "extends LSD"; runs via tests/main.py with flags for METRA / LSD / DADS / DIAYN on Ant, HalfCheetah, pixel Quadruped/Humanoid, Kitchen). Quality notes: research code, dependency rot reported in issues; pixel encoder, Lagrangian and skill sampling are all in one agent class, easy to port. Reimplementations: TLDR (https://github.com/heatz123/tldr, verified, MIT, built on the METRA codebase and includes METRA as a baseline), RSD (https://github.com/ZhHe11/RSD, verified, PyTorch, "builds upon the METRA codebase", 5 stars, no license), DoDont (code link not found on the project page; see 3.14), D3 (https://github.com/leggedrobotics/d3-skill-discovery, found via search, Isaac Lab, not inspected). No JAX reimplementation found.

7. Fit. Needs adaptation. Passes C3 (pixels, with the colored-floor caveat), C4 (no external knowledge), C1 mostly (skill dimension is hand-set; action repeat 2 is inherited from LEXA for DMC). Fails C2 as published (one skill per fixed-length episode with resets) but the adaptation is mechanical (resample z every K steps; exclude death-respawn pairs from the adjacency constraint; continuing discount). Main scientific risk for a large world: the "few primary directions" problem; coverage will be along a 2-4-D latent manifold rather than uniform.

### 3.2 LSD - Lipschitz-constrained Unsupervised Skill Discovery

1. Seohong Park, Jongwook Choi, Jaekyeom Kim, Honglak Lee, Gunhee Kim. ICLR 2022. arXiv:2202.00914 (verified). Project https://shpark.me/projects/lsd/
2. Reward r = (phi(s') - phi(s))^T z with phi 1-Lipschitz in the Euclidean state metric (||phi(x) - phi(y)|| <= ||x - y||, enforced with spectral normalisation). Argument against MI methods: MI "can be fully maximized even with small differences in states" because MI is invariant to scaling, so "discovering skills with such slight or less dynamic state variations is usually a 'lower-hanging fruit'". Skills: continuous unit vectors or zero-centred one-hot. Zero-shot goal following via z = phi(g) - phi(s).
3. Episodic (skill fixed per episode; uses early termination in Humanoid). Same resampling adaptation as METRA.
4. State-based only: Ant, Humanoid, HalfCheetah, FetchPush/Slide/PickAndPlace. The Euclidean Lipschitz constraint is meaningless on pixels (METRA's motivation).
5. Reproduced as a baseline in CSD, METRA, LEADS (where it beats METRA on hard maze: 70.0% vs 54.8%), TLDR. Brittleness: depends on the Euclidean metric of the state space, so sensitive to state scaling / which dimensions dominate.
6. https://github.com/seohongpark/LSD (verified; MIT; 39 stars; garage/PyTorch; Python 3.7.8).
7. Excluded for pixels (constraint is Euclidean in observation space); historically important as METRA's parent.

### 3.3 CSD - Controllability-aware Skill Discovery

1. Seohong Park, Kimin Lee, Youngwoon Lee, Pieter Abbeel. ICML 2023. arXiv:2302.05103 (verified). Project https://seohong.me/projects/csd/
2. Same reward form as LSD but the Lipschitz constraint uses a learned "controllability-aware distance function, which assigns larger values to state transitions that are harder to achieve with the current skills" (a learned dynamics-model density), so the agent "actively seeks complex, hard-to-control skills" and the curriculum shifts to harder behaviours over training.
3. Episodic, as LSD.
4. State-based: six locomotion and manipulation environments (Fetch, Kitchen state, Ant, HalfCheetah, Humanoid).
5. LEADS Table 1: CSD 64.0% hard maze, 93.0% Finger (best of all methods on Finger). PSD compares against CSD and reports it below PSD on hurdle tasks. The learned distance adds a density model that must be fit well; no pixel version.
6. https://github.com/seohongpark/CSD-locomotion (verified; MIT; 30 stars; builds on LSD) and CSD-manipulation (referenced in README, not fetched).
7. Excluded as published for pixels (needs a dynamics density model over states); conceptually interesting for object-manipulation worlds.

### 3.4 DIAYN - Diversity is All You Need

1. Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, Sergey Levine. ICLR 2019 (arXiv Feb 2018). arXiv:1802.06070 (verified). Site https://sites.google.com/view/diayn
2. Maximise I(S;Z) + H(A|S) - I(A;Z|S) with a variational discriminator q(z|s); intrinsic reward r = log q(z|s) - log p(z); skills discrete one-hot from a uniform prior, fixed per episode; SAC backbone.
3. Episodic: one skill per episode; discriminator trained on all states in the episode. Can be run continuing by resampling z every K steps (as later surveys and EDL do), but the discriminator then sees state distributions that depend on the previous skill's final state, which weakens the learning signal [analysis; EDL documents the related initial-state dependence].
4. State-based MuJoCo; URLB (states and pixels via DrQ-v2) and the world-model variant of Rajeswar et al. (pixels).
5. Weak coverage: URLB finds no competence-based method is SOTA on any task; EDL: existing information-theoretic methods "discover options that provide a poor coverage of the state space" and depend on the initial state; LSD: static skills; LEADS Table 1: DIAYN 43.8% hard maze, 68.2% Fetch-Reach (interestingly comparable to METRA's 68.7%). ExORL: DIAYN data is useful "in low-data settings". Sensitive to the number of skills and the discriminator capacity (URLB discussion).
6. Official: https://github.com/ben-eysenbach/sac (verified; TensorFlow/rllab; 164 stars; DIAYN.md documents the DIAYN runs). Maintained reimplementation: URLB's agents/diayn.py (PyTorch).
7. Excluded as a primary candidate (does not produce coverage; needs small skill spaces); keep as a sanity baseline only.

### 3.5 VIC - Variational Intrinsic Control

1. Karol Gregor, Danilo Jimenez Rezende, Daan Wierstra. arXiv Nov 2016, arXiv:1611.07507 (verified; no peer-reviewed venue).
2. Maximise MI between an option Omega and the terminal state s_f given s_0 (empowerment over options): variational lower bound with q(Omega | s_0, s_f); explicit (embedding) and implicit option variants; policy gradient.
3. Episodic by construction (needs a terminal state per option).
4. Grid worlds and simple tasks; not validated from pixels at scale.
5. EDL's coverage critique applies; option collapse is a known failure (EDL, UPSIDE). No official code found.
6. No official code (not located); reimplementations exist in EDL's repo as a baseline.
7. Excluded (episodic terminal-state objective; no pixel validation; superseded by DIAYN/DADS).

### 3.6 DADS - Dynamics-Aware Discovery of Skills

1. Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, Karol Hausman. ICLR 2020. arXiv:1907.01657 (verified).
2. Maximise I(s'; z | s) via a skill-conditioned dynamics model q(s' | s, z); reward log q(s'|s,z) - log sum_i q(s'|s,z_i) over sampled skills; continuous skills; the learned skill dynamics enables model-predictive planning in skill space (zero-shot planning on downstream tasks). Off-DADS is an off-policy variant.
3. Episodic (skill fixed per episode), but the per-transition reward is naturally defined for any (s,z,s'), so continuing use with skill resampling is straightforward [analysis].
4. State-based MuJoCo (Ant, HalfCheetah, Humanoid; often with an "x-y prior" restricting the dynamics model to coordinates - a problem-knowledge trick). Not validated on pixels; METRA's pixel comparison (Figure 3) shows DADS failing on pixel locomotion.
5. METRA Fig. 5: DADS fails to explore pixel locomotion; PSD uses DADS as a baseline and reports it below PSD. Known to prefer predictable (often slow) behaviours (LSD's critique).
6. https://github.com/google-research/dads (verified; Apache-2.0; 203 stars; TensorFlow; archived read-only April 2026).
7. Excluded (needs a learned dynamics model over observations; pixel version not demonstrated; x-y prior violates C1).

### 3.7 CIC - Contrastive Intrinsic Control

1. Michael Laskin, Hao Liu, Xue Bin Peng, Denis Yarats, Aravind Rajeswaran, Pieter Abbeel. 2022 (arXiv:2202.00161, verified; appears in NeurIPS 2022). Site https://sites.google.com/view/cicrl/
2. Decomposes I(tau; z) = H(tau) - H(tau | z): H(tau) is maximised with an APT-style kNN particle entropy over transition embeddings (the intrinsic reward), and H(tau|z) is minimised with a noise-contrastive (CPC/SimCLR-style) discriminator between transitions (s,s') and skills; skills are 64-D continuous (uniform [0,1]), fixed per episode.
3. Episodic in URLB protocol; the reward is per-transition, so resampling z within a lifetime is possible.
4. State-based URLB only: "we only consider MDPs where the full state is observable"; pixel extension listed as future work. On URLB states: 79% over APS (IQM) and 18% over ProtoRL.
5. METRA found CIC needs 64-D skills for good coverage ("using 64-D skills for CIC leads to better state coverage than using 2-D or 4-D skills") and that CIC fails on pixel locomotion (Fig. 5). CeSD (ICML 2024) and BeCL (ICML 2023) report beating CIC on URLB.
6. https://github.com/rll-research/cic (verified; built on URLB; 88 stars; license not shown).
7. Needs adaptation and is unproven from pixels; its exploration component is essentially APT, so prefer APT directly if the skill structure is not needed.

### 3.8 APS - Active Pretraining with Successor Features

1. Hao Liu, Pieter Abbeel. ICML 2021. arXiv:2108.13956 (verified via search; ICML proceedings v139).
2. Decomposes I(s;z) = H(s) - H(s|z): state entropy via APT's particle estimator in a learned feature space, and -H(s|z) via variational successor features (reward phi(s)^T z with z on the unit sphere). After pre-training, a task is solved by inferring w from reward regression (zero-shot in the successor-features sense).
3. Episodic in practice; per-transition reward.
4. Atari 100k (pixels) and URLB (states and pixels). URLB: competence-based, never SOTA; worst group from pixels.
5. "APS does not work well as a zero-shot RL method" (Touati et al., 2023, when APS features are used for SF). Rajeswar et al. (2023): skill-driven methods (APS, DIAYN) fall behind on Walker and Quadruped from pixels.
6. URLB agents/aps.py (verified repo); no separate official repo located.
7. Excluded as a primary candidate; its entropy half is APT.

### 3.9 APT - Behavior From the Void: Unsupervised Active Pre-Training (data-based, included for comparison)

1. Hao Liu, Pieter Abbeel. NeurIPS 2021 (spotlight). arXiv:2103.04551 (verified).
2. Non-parametric (k-NN particle) entropy of a contrastively learned representation as intrinsic reward; no skills.
3. Per-transition reward computed against a batch of buffer samples; no reset dependence. Continuing-friendly.
4. Atari (pixels; "human-level performance on 12 games") and DMC pixels. URLB: top method on states (with ProtoRL).
5. In METRA Fig. 8 APT is one of the exploration methods that "fail to fully explore" state Ant and pixel Humanoid; TLDR Fig. 4 shows APT below METRA on HalfCheetah/Ant. In C-TeC (2026) APT is a baseline beaten by the temporal-contrastive reward.
6. URLB agents/apt (ICM- and inverse-dynamics-feature variants; verified).
7. Good as a data-based control baseline (not competence-based, but the researcher should run it alongside METRA).

### 3.10 ProtoRL (data-based, included for comparison)

1. Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto. ICML 2021. arXiv:2102.11271 (verified).
2. SwAV-style prototypes learned on pixel observations; k-NN entropy over prototype-projected embeddings as the intrinsic reward; representation and exploration are tied.
3. Per-transition reward; continuing-friendly.
4. Pixel DMC. URLB: strong on states; ExORL: ProtoRL datasets among the best.
5. Touati et al. (2023): Proto buffers "show particularly poor offline TD3 performance on Quadruped".
6. https://github.com/denisyarats/proto (verified; MIT; PyTorch; 88 stars).
7. Good data-based control baseline for pixels.

### 3.11 RE3 - Random Encoders for Efficient Exploration (data-based)

1. Younggyo Seo, Lili Chen, Jinwoo Shin, Honglak Lee, Pieter Abbeel, Kimin Lee. ICML 2021. arXiv:2102.09430 (verified).
2. k-NN state entropy in the feature space of a randomly initialised, frozen CNN; added to the task reward or used alone.
3. Per-transition; continuing-friendly.
4. Pixel DMC and MiniGrid.
5. Random features are cheap and stable but not metric-aware; coverage in large worlds untested.
6. https://github.com/younggyoseo/RE3 (verified).
7. Trivial to add as a control; not a coverage driver on its own in a big world [analysis].

### 3.12 Forward-backward (FB) representations and zero-shot RL

#### 3.12.1 Learning One Representation to Optimize All Rewards

1. Ahmed Touati, Yann Ollivier. NeurIPS 2021. arXiv:2103.07945 (verified).
2. Learn F(s,a,z) and B(s') such that the successor measure M^{pi_z}(s,a,ds') ~ F(s,a,z)^T B(s') rho(ds'); pi_z is greedy w.r.t. F(s,a,z)^T z; at test time a reward r gives z_r = E_rho[r(s) B(s)] and pi_{z_r} is near-optimal for r. Trained by TD on the measure; z sampled from a prior (Gaussian, normalised) during training.
3. Off-policy TD on transitions; no reset dependence in the loss. Exploration is not addressed (random or epsilon-greedy data in the paper).
4. Discrete and continuous mazes, FetchReach, pixel MsPacman (F takes pixels but, as Touati et al. 2023 note, "only the agent's (x,y) position for B's input", which is a problem-knowledge trick violating C1).
5. See 3.12.2.
6. https://github.com/facebookresearch/controllable_agent (verified; MIT; 80 stars; archived read-only Jan 2025; built on url_benchmark; PyTorch).
7. Offline/zero-shot representation; not an exploration method. Needs adaptation and pixel validation.

#### 3.12.2 Does Zero-Shot Reinforcement Learning Exist?

1. Ahmed Touati, Jeremy Rapin, Yann Ollivier. ICLR 2023 (top-25% / spotlight). arXiv:2209.14935 (verified).
2. Compares FB to successor features with 10 basic-feature learners (random, autoencoder, ICM, transition model, Laplacian eigenfunctions, low-rank, contrastive, APS/diversity, ...). All trained offline on ExORL/URLB buffers.
3. Offline; "13 tasks in 4 environments" on RND, APS and Proto buffers.
4. State-based only: "all the environments tested here were deterministic".
5. "FB reaches 81% of supervised offline TD3 performance, and 85% on the RND buffer"; Laplacian SF second (74%). "APS does not work well as a zero-shot RL method." Data quality dominates: Proto buffers fail on Quadruped. Stated limitation: "for large problems, some form of prior seems unavoidable".
6. Same repo as above.
7. Excluded as an exploration method; relevant as the theoretical anchor for HILP/FB-style zero-shot adaptation once data exists.

#### 3.12.3 Follow-ups 2023-2026 (verified abstracts)

- Zero-Shot RL from Low Quality Data (Scott Jeen, Tom Bewley, Jonathan Cullen; NeurIPS 2024; arXiv:2309.15178). Value- and measure-conservative FB (VC-FB, MC-FB) for small/homogeneous datasets. Code https://github.com/enjeeneer/zero-shot-rl (verified; MIT; ExORL Walker/Quadruped/PointMass/Jaco with RND/DIAYN/Random datasets; D4RL). State-based.
- Unsupervised Zero-Shot RL via Functional Reward Encodings (Kevin Frans, Seohong Park, Pieter Abbeel, Sergey Levine; ICML 2024; arXiv:2402.17135). Transformer VAE encodes (state, reward) samples of random unsupervised reward functions; offline. Code https://github.com/kvfrans/fre (verified; JAX; AntMaze, ExORL, Kitchen; 58 stars). State-based.
- Unsupervised Zero-Shot RL via Dual-Value Forward-Backward Representation (DVFB; Jingbo Sun, Songjun Tu, Qichao Zhang, Xin Liu, Haoran Li, Yaran Chen, Ke Chen, Dongbin Zhao; ICLR 2025; PDF fetched and parsed). The first online URL use of FB: "poor exploration in forward-backward representations can lead to limited data diversity in online URL, impairing successor measures"; adds an exploration value function with a contrastive-entropy intrinsic reward. 2M pre-training steps on 12 URLB/DMC tasks (Walker, Quadruped, Hopper; state-based); zero-shot beats FB, SF-Lap, CIC, BeCL, ComSD, CeSD on all 12 (e.g., Walker average 686 vs FB 114, CIC 207). Code https://github.com/bofusun/DVFB (verified; 7 stars; no license shown). Not validated from pixels.
- Epistemically-guided forward-backward exploration (Nuria Armengol Urpi, Marin Vlastelica, Georg Martius, Stelian Coros; RLJ 2025; arXiv:2507.05477). Exploration policy minimises posterior variance of the FB representation; "improve sample complexity of the FB algorithm considerably". Site https://sites.google.com/view/fbee-url (not fetched). State-based DMC as far as the abstract indicates.
- Towards Robust Zero-Shot RL (BREEZE; Kexin Zheng, Lauriane Teyssier, Yinan Zheng, Yu Luo, Xianyuan Zhan; NeurIPS 2025; arXiv:2510.15382). FB "lack[s] expressivity" and suffers "extrapolation errors caused by out-of-distribution actions"; adds behaviour regularisation, diffusion policies, attention. ExORL and D4RL Kitchen, states. Code https://github.com/Whiterrrrr/BREEZE (listed, not fetched).
- TD-JEPA (Marco Bagatella, Matteo Pirotta, Ahmed Touati, Alessandro Lazaric, Andrea Tirinzoni; arXiv:2510.00739, Oct 2025). Latent-predictive TD representations with policy-conditioned predictors; the one zero-shot paper with systematic pixel results: DMC-RGB (walker, cheetah, quadruped, pointmass) and OGBench-RGB; "latent-predictive methods tend to be generally preferable in pixel-based domains"; FB and HILP underperform from pixels. Offline. No code URL in the text.
- A Unified Framework for Zero-Shot RL (Jacopo Di Ventura, Jan Felix Kleuker, Aske Plaat, Thomas Moerland; arXiv:2510.20542, v2 Mar 2026). Theory: error decomposition (inference, reward, approximation). No pixel experiments.
- Zero-Shot RL Under Partial Observability (Jeen, Bewley, Cullen; RLC 2025; arXiv:2506.15446): memory-based FB for POMDPs. Relevant because a pixel frame of Rain World is partially observable.
- Soft FB representations for general utilities (arXiv:2602.06769, Feb 2026): extends FB to objectives that are arbitrary functions of the occupancy (e.g., pure exploration). Abstract-level only.
- FB-MEBE (Jiajun Hu, Nuria Armengol Urpi, Jin Cheng, Stelian Coros; arXiv:2603.25464, 2026): online FB with maximum-entropy behaviour exploration on simulated and real quadrupeds; states.
- FB-CPR / Meta Motivo (Zero-shot whole-body humanoid control via behavioral foundation models; NeurIPS 2024 / arXiv:2504.11054) and Fast Adaptation with Behavioral Foundation Models (arXiv:2504.07896) regularise FB with unlabeled motion-capture data: excluded by C4.
- Scott Jeen, On Zero-Shot RL (PhD thesis, arXiv:2508.16496): consolidates the data-quality, observability and data-availability constraints.

Assessment of the FB family for this project [analysis]: it is the most principled way to turn a replay buffer into "any-reward" policies, but every online variant is state-based, the pixel evidence (TD-JEPA) says FB/HILP degrade from pixels, and none has been run in a continuing open world. Use it, if at all, as an offline consumer of data that METRA/RND/APT collect.

### 3.13 HILP - Foundation Policies with Hilbert Representations

1. Seohong Park, Tobias Kreiman, Sergey Levine. ICML 2024. arXiv:2402.15567 (verified; v2 May 2024). Project https://seohong.me/projects/hilp/
2. Two stages, both offline. (i) Hilbert representation: learn phi so that ||phi(s) - phi(g)|| equals the optimal temporal distance d*(s,g) = -V*(s,g), by fitting the goal-conditioned value V(s,g) = -||phi(s) - phi(g)|| with an IQL-style expectile TD loss on hindsight-relabelled goals (isometry rather than METRA's Lipschitz inequality). (ii) Hilbert foundation policy: latent-conditioned policy trained offline (IQL/AWR-style) on reward r(s,z,s') = <phi(s') - phi(s), z> with z "sampled uniformly from the set of unit vectors", so the policy "spans" the latent space. Zero-shot use: goal reaching with z = normalised phi(g) - phi(s); zero-shot RL by treating phi as successor features and regressing a reward onto z; test-time planning in latent space (HILP-Plan). Latent dims D = 32 (goal-conditioned) and D = 50 (zero-shot RL).
3. Offline by design; the authors contrast it with METRA, which requires "on-policy rollouts". It never acts to explore. In a continuing env it could be trained on the agent's replay buffer, but it will only know the parts of the world the data-collecting policy visited [analysis].
4. ExORL (Walker, Cheetah, Quadruped, Jaco) in state and pixel (64x64x3) form; D4RL AntMaze and Kitchen including visual-kitchen-partial/mixed (64x64x3). "HILPs achieve the best IQM in pixel-based environments as well"; visual-kitchen-partial 59.9 +- 4.0, mixed 55.9 +- 9.7 (comparable to GC-IQL).
5. TD-JEPA (2025) reports HILP underperforming its latent-predictive method from pixels; BREEZE and DVFB treat HILP/FB as expressivity-limited. Authors' limitations: Euclidean Hilbert space; "highly asymmetric or disconnected MDPs"; stochasticity. The data dependence (ExORL RND buffers) means HILP's apparent "coverage" is RND's coverage.
6. https://github.com/seohongpark/HILP (verified; MIT; 105 stars): hilp_zsrl (PyTorch; built on facebookresearch/controllable_agent; ExORL state and pixel datasets with a convert.py; agents HILP, HILP-G, FB, FDM) and hilp_gcrl (goal-conditioned, not inspected).
7. Needs adaptation: not an exploration method; good candidate as an offline representation / zero-shot goal-reaching layer over data collected by another explorer; its temporal-distance representation is the same object METRA/TLDR learn online.

### 3.14 DoDont - Do's and Don'ts: Learning Desirable Skills with Instruction Videos

1. Hyunseung Kim, Byungkun Lee, Hojoon Lee, Dongyoon Hwang, Donghu Kim, Jaegul Choo. NeurIPS 2024. arXiv:2406.00324 (verified). Project https://mynsng.github.io/dodont/
2. Trains an instruction network p_psi(s,s') on action-free "do" and "don't" videos (fewer than 8), then "multiplying instruction network p_psi(s,s') to the original learning objective function of METRA", i.e., r = p_psi(s,s') * (phi(s') - phi(s))^T z, "simply adding a single line of code on top of the METRA framework".
3. Inherits METRA's episodic protocol.
4. DMC Cheetah, Quadruped (states for most experiments; pixel DMC variants also use coloured floors), Kitchen.
5. Shows that METRA on its own learns "unsafe or undesirable behaviors" (tripping, undesired locations); the fix needs human-supplied videos.
6. Project page carries no GitHub link (checked); the paper mentions a Google Drive link in the appendix. Code availability unverified.
7. Excluded (C4: requires external instruction videos). Useful only as evidence of METRA's failure modes and of how easily METRA's reward can be reweighted.

### 3.15 Empowerment-based methods

- Variational Information Maximisation for Intrinsically Motivated RL (Shakir Mohamed, Danilo Rezende; NIPS 2015; arXiv:1509.08731, verified). Variational lower bound on empowerment I(a_{t:t+k}; s_{t+k} | s_t) with convolutional networks "from pixels to actions" in grid-world-style tasks. No official code located. Episodic grid tasks; small scale.
- Hierarchical Empowerment (Andrew Levy, Sreehari Rammohan, Alessandro Allievi, Scott Niekum, George Konidaris; arXiv:2307.02728, 2023, under review at ICLR 2024 per the PDF header). Goal-conditioned RL as a variational bound on short-horizon empowerment; a hierarchy extends it to "exponentially longer time scales"; four-level agents "cover a surface area over two orders of magnitude larger than prior work" in ant navigation (state-based). No code URL on arXiv.
- Latent-Predictive Empowerment (Andrew Levy, Alessandro Allievi, George Konidaris; arXiv:2410.11155, 2024). Removes the need for a transition model; "high-dimensional observations and highly stochastic transition dynamics" are claimed; no code URL on arXiv; venue not verified.
- Information-Theoretic Policy Pre-Training with Empowerment (Moritz Schneider et al.; arXiv:2510.05996, Oct 2025): discounted empowerment as a pre-training signal; environments not specified in the abstract.
- Learning to Perceive the World Through Control: Empowerment-Based Representation Learning (Mahsa Bastankhah, Sophie Broderick, Benjamin Eysenbach; arXiv:2605.30656, May 2026): empowerment induces forward and backward representations invariant to uncontrollable features; representation-learning paper, no scale claims verified.
- Experimental Evidence that Empowerment May Drive Exploration in Sparse-Reward Environments (arXiv:2107.07031) is a small-scale study.

Assessment: empowerment is the conceptual root of VIC/DIAYN (skills as a channel from z to future states), but no empowerment method has been shown to produce coverage of a large pixel world, none has public, maintained code at scale, and the natural reading of empowerment (seek states with many reachable futures) is a stationing objective, not a coverage objective. Excluded as a primary candidate; cite as background.

### 3.16 Other skill-discovery methods from 2022-2026 that matter for the questions

- EDL - Explore, Discover and Learn (Victor Campos, Alexander Trott, Caiming Xiong, Richard Socher, Xavier Giro-i-Nieto, Jordi Torres; ICML 2020; arXiv:2002.03647, verified; code https://github.com/victorcampos7/edl, verified, MIT). Shows DIAYN/VIC/VALOR "discover options that provide a poor coverage of the state space" and depend on the initial state; proposes separate exploration (SMM), discovery (VQ-VAE over explored states) and learning phases. Mazes, state-based.
- UPSIDE - Direct then Diffuse (Pierre-Alexandre Kamienny, Jean Tarbouriech, Sylvain Lamprier, Alessandro Lazaric, Ludovic Denoyer; ICLR 2022; arXiv:2110.14457, verified). Skills with a directed part and a diffusing part composed into a tree that incrementally covers the environment; a "coverage-directedness trade-off" framing of MI objectives. Navigation and control, state-based.
- BeCL - Behavior Contrastive Learning (Rushuai Yang et al.; ICML 2023; arXiv:2305.04477, verified) and CeSD - Constrained Ensemble Exploration (Chenjia Bai et al.; ICML 2024; arXiv:2405.16030, verified; code https://github.com/Baichenjia/CeSD, verified, MIT, built on URLB). CeSD states that "empowerment often leads to static skills, and pure exploration only maximizes the state coverage rather than learning useful behaviors"; each skill owns a prototype cluster and maximises particle entropy inside it. URLB states: 91.05% IQM vs BeCL 75.18%. No pixel results.
- LEADS - Exploration by Learning Diverse Skills through Successor State Measures (Paul-Antoine Le Tolguenec, Yann Besse, Florent Teichteil-Konigsbuch, Dennis G. Wilson, Emmanuel Rachelson; NeurIPS 2024; arXiv:2406.10127, verified). Each skill's successor state measure (C-learning) is pushed toward under-visited states while keeping skills' SSMs distinct; "maximizing mutual information might be ambiguous when seeking exploratory behaviors". Coverage table reproduced in 3.1.5; state-based; "relies intrinsically on the quality of the SSM estimator" and "failed on certain MuJoCo environments". Code only as an anonymous review link (not verifiable); no public repo found on the first author's GitHub.
- TLDR - Unsupervised Goal-Conditioned RL via Temporal Distance-Aware Representations (Junik Bae, Kwanyoung Park, Youngwoon Lee; CoRL 2024; arXiv:2407.08464, verified; code https://github.com/heatz123/tldr, verified, MIT, built on METRA). Learns the same temporal-distance phi as METRA (constrained objective with Lagrange multipliers) but uses it for (i) selecting faraway goals with a particle-entropy estimator over phi, (ii) an exploration reward r^E = r_TLDR(s') - r_TLDR(s) (move away from the visited distribution), and (iii) a goal-reaching reward r^G = ||phi(s)-phi(g)|| - ||phi(s')-phi(g)||. Environments: state Ant, HalfCheetah, Humanoid-Run, Quadruped-Escape, AntMaze-Large (300-step episodes), AntMaze-Ultra (600); pixel Quadruped (200 steps) and pixel Kitchen (50). Baselines METRA, PEG, LEXA, RND, APT, Disagreement. Results: TLDR best everywhere except HalfCheetah (METRA slightly better); on AntMaze-Large/Ultra "dramatically exceeds METRA, PEG, RND, APT, Disagreement"; from pixels "mixed": on pixel Quadruped TLDR "learns slower than LEXA and METRA". Episode structure (goal phase then exploration phase) is episodic but both rewards are per-transition and the goal buffer is a replay buffer, so a continuing variant (pick a new far goal every K steps) is natural [analysis].
- RSD - Efficient Skill Discovery via Regret-Aware Optimization (He Zhang, Ming Zhou, Shaopeng Zhai, Ying Sun, Hui Xiong; ICML 2025; arXiv:2506.21044, verified; code https://github.com/ZhHe11/RSD, verified). Min-max game between a skill-generator population and the policy, sampling z from the whole latent space (with magnitude) rather than unit vectors. State Ant, Maze2d-large, Antmaze-medium/large; pixel Kitchen. Beats METRA on Antmaze-large (+15% success) and pixel Kitchen (5.08 vs 3.94 tasks).
- PSD - Periodic Skill Discovery (Jonghae Park, Daesol Cho, Jusuk Lee, Dongseok Shim, Inkyu Jang, H. Jin Kim; NeurIPS 2025; arXiv:2511.03187, verified). Circular latent space for periodic skills; pixel Ant and HalfCheetah (90x90x3); HalfCheetah-hurdle 3.8 +- 2.0 vs METRA 1.9 +- 0.8. Code via project page https://jonghaepark.github.io/psd (GitHub not verified).
- SUSD - Structured Unsupervised Skill Discovery through State Factorization (Hosseini, Soleymani Baghshah; ICLR 2026; arXiv:2602.01619) and DUSDi (Jiaheng Hu, Zizhao Wang, Peter Stone, Roberto Martin-Martin; NeurIPS 2024; arXiv:2410.11251): both exploit a factorised state (objects/entities); whether the factorisation is given is not clear from the abstracts, and neither reports pixel open-world results. Likely violate C1 if factors are hand-specified.
- Unsupervised Skill Discovery through Skill Regions Differentiation (Ting Xiao et al.; arXiv:2506.14420, 2025): maximises deviation of one skill's state density from the others' explored regions with a conditional autoencoder; claims state- and image-based tasks; no code located; venue unverified.
- Robotics deployments of METRA-style objectives: Constrained Skill Discovery for quadruped locomotion (Vassil Atanassov et al.; arXiv:2410.07877; norm-matching replaces latent-transition maximisation; real ANYmal), D3 - Divide, Discover, Deploy (Rafael Cathomen, Mayank Mittal, Marin Vlastelica, Marco Hutter; CoRL 2025; arXiv:2508.19953; factorised METRA-style skills with symmetry and style priors, 2048 Isaac Lab envs, real hardware), SDAX (Seungeun Rho, Kartik Garg, Morgan Byrd, Sehoon Ha; CoRL 2025; arXiv:2508.08982). These show the temporal-distance objective is trainable at scale with on-policy PPO, but all use privileged state and hand-designed factors/regularisers.
- Methods excluded by C4: Reference Grounded Skill Discovery (arXiv:2510.06203; reference motions), Guiding Skill Discovery with Foundation Models (arXiv:2510.23167; VLM/LLM), COLLIE (arXiv:2606.00950; semantic latent space), Open-World Skill Discovery from Unsegmented Demonstrations (ICCV 2025; Minecraft videos), Unsupervised Hierarchical Skill Discovery (Harvey et al., ICML 2026; arXiv:2601.23156; segments unlabeled trajectories in Craftax and Minecraft from pixels - offline, needs trajectories), LOTUS (vision-language segmentation).
- Cross-stream but decisive for the coverage question: Temporal Representations for Exploration / C-TeC (Faisal Mohamed, Catherine Ji, Benjamin Eysenbach, Glen Berseth; ICLR 2026 per the repo; arXiv:2603.02008; code https://github.com/FaisalAhmed0/c-tec, verified, JAX, Brax/MJX and Craftax). Temporal contrastive (InfoNCE over discounted future states) representation; reward = negative similarity to likely futures; explicitly non-episodic ("forward-looking" rewards from a trajectory buffer rather than episodic memory). ant_large_maze: ~2500 unique positions vs RND ~1500, ICM ~1300; Craftax-Classic: more achievements than RND/ICM/APT/E3B, but with symbolic observations, not pixels. Also Episodic Novelty Through Temporal Distance (ETD; Jiang et al., ICLR 2025; arXiv:2501.15418) uses temporal distance for episodic novelty in contextual MDPs.
- Reset-free skill learning: Reset-Free Lifelong Learning with Skill-Space Planning (LiSP; Kevin Lu, Aditya Grover, Pieter Abbeel, Igor Mordatch; ICLR 2021; arXiv:2012.03548; code https://github.com/kzl/lifelong_rl, verified, MIT, PyTorch/rlkit, 109 stars). DADS-style skills learned with intrinsic rewards inside a learned model plus planning in skill space, specifically for non-episodic lifelong RL (state-based MuJoCo). Continual Learning of Control Primitives: Skill Discovery via Reset-Games (Xu et al., 2020; arXiv:2011.05286) learns reset skills in reset-free training. These are the only works I found that study skill discovery in a genuinely continuing setting; both are state-based and small-scale.
- Choreographer (Pietro Mazzaglia, Tim Verbelen, Bart Dhoedt, Alexandre Lacoste, Sai Rajeswar; ICLR 2023 notable top-25%; arXiv:2211.13350, verified; code https://github.com/mazpie/choreographer, verified, MIT, PyTorch, DreamerV2-based). Discovers VQ-coded skills in the imagination of a world model trained on exploration data (data collected by a separate explorer, e.g., LBS/P2E), from pixels on URLB; beats APT, LBS, DIAYN, APS on URLB from pixels. Exploration is delegated to the knowledge-based explorer; the skill discovery itself is "exploration-agnostic" and offline-in-imagination. Fit: needs adaptation (world model from pixels is the heavy part; the exploration driver is not competence-based).

---

## 4. Answers to the three explicit questions

### (a) Do competence-based methods produce state-space coverage in large worlds, or locally distinguishable behaviours?

Mostly the latter, with the distance-maximising subfamily as a partial exception that covers along a low-dimensional latent manifold rather than uniformly.

- MI-based skill discovery (VIC, DIAYN, DADS, SMM, APS, CIC) does not produce coverage. Evidence: URLB ("no competence-based approach that achieves state-of-the-art ... on any of the URLB tasks"; discriminators classify "configurations of the agent lying on the ground"); EDL ("poor coverage of the state space", initial-state dependence); LSD (MI is scale invariant, so "slight or less dynamic state variations" are the "lower-hanging fruit"); CIC ("max H(z) does not imply max H(tau)"); CeSD ("empowerment often leads to static skills"); LEADS ("maximizing mutual information might be ambiguous when seeking exploratory behaviors", with the toy counter-example). METRA's own Figure 3/5 shows DIAYN, DADS, CIC and LSD failing to leave the start region in pixel Quadruped/Humanoid.
- Distance-maximising methods (LSD, CSD, METRA) do travel far, because the reward is proportional to displacement in a metric latent space. But: METRA's coverage metric is 1x1 x-y bins of 48 skill rollouts, a 2-D projection of position; the paper reports approximately 2000 bins for Ant and under 100 bins for pixel Humanoid in 400-step episodes from a fixed start, which is coverage of an open arena, not of a structured world. In structured worlds the limits show: TLDR on AntMaze-Large/Ultra and Quadruped-Escape, LEADS' hard-maze 54.8% for METRA (below LSD and CSD and NGU), RSD's "primary directions ... homogeneous skills", PSD/D3's "maximally fast" bias. METRA's own authors list the "behaviors that move linearly in the latent space" simplification as a limitation. [analysis] In a room-graph world like Rain World, a 2-4-D unit-sphere skill space will produce a handful of "go far in direction z" behaviours; with per-episode skills from a fixed start they fan out radially, which is why mazes defeat them. The remedies in the literature are exactly the hybrids: TLDR (temporal-distance representation + kNN-entropy goal selection), LEADS (successor-state measures + under-visitation targeting), RSD (adversarial skill generator), CeSD (skills partitioned over prototype clusters). None of these hybrids has been run from pixels in an open world; TLDR and RSD have pixel Kitchen / pixel Quadruped results only.

### (b) How do METRA / HILP compare with knowledge-based bonuses (RND, disagreement) when the goal is coverage?

- Direct head-to-heads with METRA (online): METRA Fig. 8 (total state coverage, 8 seeds) shows METRA above ICM, RND, APT, APS, LBS and P2E/Disagreement on state Ant and pixel Humanoid, with exploration methods "decent" in pixel Kitchen; the authors' mechanism: "since it is practically infeasible to completely cover every possible state or transition, pure exploration methods struggle to explore the state space of complex environments". TLDR Fig. 4: RND, APT and Disagreement are below METRA on HalfCheetah and Ant, but on AntMaze-Large/Ultra both METRA and the bonuses are far below TLDR. LEADS Table 1: RND is the worst method on every maze (76.6 / 39.6 / 30.8%) and NGU is mid-pack (86.8 / 73.4 / 57.2%), METRA 92.8 / 78.0 / 54.8%. C-TeC (2026): RND ~1500 vs temporal-contrastive ~2500 positions on ant_large_maze. Conclusion: in locomotion-type spaces where "novelty" is dominated by joint-angle/velocity noise, prediction-error and count-like bonuses dilute, and metric-aware objectives win; in maze-like spaces neither plain METRA nor plain RND is good, and temporal-distance-driven goal selection wins.
- From pixels with a world model, Rajeswar et al. (ICML 2023) rank LBS and Plan2Explore (knowledge-based) above DIAYN/APS; METRA was not in that study and nothing has run METRA against a Dreamer-style knowledge bonus on equal footing.
- HILP cannot be compared on coverage: it is offline and is trained on ExORL buffers collected by RND (and Proto/APS). Its zero-shot results therefore presuppose a knowledge-based explorer. The sensible reading is that HILP/FB and RND are complementary (explorer + representation), not competitors.
- A fair summary for a continuing open world [analysis]: the coverage-relevant ingredient shared by the winners (METRA, TLDR, HILP, C-TeC, ETD) is a temporal-distance / temporal-contrastive representation that is invariant to pixel-level noise; whether it is used to define skills (METRA), to pick far goals (TLDR), or directly as a novelty measure (C-TeC, ETD) matters less than having it. Knowledge-based bonuses on raw pixels lack this invariance.

### (c) Which 2-3 methods to implement first, and why

1. METRA (with the continuing-env adaptations in 3.1.3). Reasons: the only online skill-discovery method with pixel results that did not use proprioception; a single-line reward plus a Lagrangian constraint, so it is easy to port onto any SAC/PPO implementation; official MIT code and several independent reimplementations (TLDR, RSD) to cross-check; yields a reusable temporal-distance phi for zero-shot goal reaching and hierarchical control. Expect: good "go far" behaviours, poor room-to-room coverage; measure coverage with the game's room/position info (eval only) rather than METRA's x-y bins.
2. TLDR (same codebase). Reasons: it is the published fix for exactly the failure mode most likely in a large structured world (METRA on AntMaze), adds only a kNN-entropy goal selector and two per-transition rewards on top of METRA's phi, and has pixel runs (Quadruped, Kitchen). It is the method with the strongest large-state-space coverage evidence that still fits C1-C4. Adaptation: replace "goal phase then explore phase per episode" with "pick a far goal every K steps, switch to exploration reward when reached or timed out".
3. A knowledge/data-based control from the URLB codebase (RND and APT or ProtoRL) under the identical backbone, because every coverage study above shows the ranking flips with environment structure, and because any later FB/HILP stage needs good exploratory data (ExORL's lesson). If zero-shot task adaptation is wanted later, train HILP offline on the accumulated replay buffer (its hilp_zsrl code already handles pixel ExORL-format data) rather than adopting an online FB method, since DVFB/FB-EE/FB-MEBE are unvalidated from pixels and TD-JEPA reports FB/HILP degradation from pixels.

Not recommended as first implementations: DIAYN/DADS/APS/CIC (no coverage, unproven from pixels), CSD/LSD (state-metric constraints), empowerment methods (no scaled code), DoDont / FB-CPR / RGSD (external data), LEADS (SSM estimator fragility; no verifiable code), SUSD/DUSDi (factorised state).

---

## 5. Continuing-environment adaptation checklist for METRA / TLDR [analysis]

- Skill schedule: resample z ~ Uniform(S^{D-1}) every K steps (start with K equal to the paper's episode length for a comparable arena, 200-400 at action repeat 1; try K in {64, 128, 256, 512}). Store z per transition; the reward (phi(s') - phi(s))^T z is already per transition. D3's observation about skill "lock-in" without resampling is the only published data point.
- Constraint set S_adj: include only genuine one-step transitions; exclude (s_death, s_respawn) pairs and any skill-switch boundary from the Lipschitz penalty; also exclude them from the critic's bootstrapping (treat death as a terminal for the critic if the respawn is a teleport).
- Critic: continuing discounting (gamma = 0.99 as in the paper; no time-limit terminals).
- Observation stack / action repeat: the paper uses 64x64x3 single frames (no explicit frame stacking is described) with action repeat 2 inherited from LEXA for DMC; under C1 use the environment's native step and let the encoder see a short frame stack if velocity matters.
- Backbone: discrete/multi-binary SAC (factorised Bernoulli policy) or PPO; D3 shows the objective works with PPO at scale.
- Evaluation: log phi-space spread, room-visit counts and position bins from privileged info (evaluation only).
- Known risks to monitor: latent collapse when lambda saturates (watch ||phi(s) - phi(s')|| histograms vs the eps = 1e-3 slack); asymmetric transitions (falls) compress phi (authors suggest a quasimetric); "fast-moving" bias (PSD/D3) could translate to reckless movement and frequent deaths.

---

## 6. Code verification table

| Method | Repo | Fetched | Framework | License | Notes |
|---|---|---|---|---|---|
| METRA | https://github.com/seohongpark/METRA | yes | PyTorch + garage fork | MIT | 96 stars; dependency issues reported |
| TLDR | https://github.com/heatz123/tldr | yes | PyTorch (METRA codebase) | MIT | 36 stars; includes METRA baseline, pixel envs |
| RSD | https://github.com/ZhHe11/RSD | yes | PyTorch (METRA codebase) | none shown | 5 stars |
| LSD | https://github.com/seohongpark/LSD | yes | PyTorch + garage | MIT | 39 stars |
| CSD | https://github.com/seohongpark/CSD-locomotion | yes | PyTorch (LSD) | MIT | 30 stars; CSD-manipulation referenced |
| HILP | https://github.com/seohongpark/HILP | yes | PyTorch (zsrl, on controllable_agent); gcrl not inspected | MIT | 105 stars; pixel ExORL supported |
| FB / zero-shot RL | https://github.com/facebookresearch/controllable_agent | yes | PyTorch (url_benchmark) | MIT | archived Jan 2025 |
| VC-FB / MC-FB | https://github.com/enjeeneer/zero-shot-rl | yes | PyTorch (per README structure) | MIT | 29 stars |
| FRE | https://github.com/kvfrans/fre | yes | JAX | not shown | 58 stars |
| DVFB | https://github.com/bofusun/DVFB | yes | PyTorch (URLB-style) | not shown | 7 stars |
| URLB (ICM, RND, Disagreement, APT, ProtoRL, DIAYN, APS, SMM) | https://github.com/rll-research/url_benchmark | yes | PyTorch, DrQ-v2/DDPG | MIT | 368 stars |
| CIC | https://github.com/rll-research/cic | yes | PyTorch (URLB) | not shown | 88 stars |
| CeSD | https://github.com/Baichenjia/CeSD | yes | PyTorch (URLB) | MIT | 7 stars |
| DIAYN | https://github.com/ben-eysenbach/sac | yes | TensorFlow / rllab | file present | 164 stars; DIAYN.md |
| DADS | https://github.com/google-research/dads | yes | TensorFlow | Apache-2.0 | archived Apr 2026 |
| ProtoRL | https://github.com/denisyarats/proto | yes | PyTorch | MIT | 88 stars |
| RE3 | https://github.com/younggyoseo/RE3 | yes | PyTorch | - | - |
| SMM | https://github.com/RLAgent/state-marginal-matching | search only | - | - | not fetched |
| EDL | https://github.com/victorcampos7/edl | yes | PyTorch (Sibling Rivalry) | MIT | 34 stars |
| LiSP | https://github.com/kzl/lifelong_rl | yes | PyTorch (rlkit) | MIT | 109 stars |
| Choreographer | https://github.com/mazpie/choreographer | yes | PyTorch (DreamerV2-style) | MIT | 43 stars |
| C-TeC | https://github.com/FaisalAhmed0/c-tec | yes | JAX (Brax/MJX, Craftax) | none shown | 4 stars |
| D3 | https://github.com/leggedrobotics/d3-skill-discovery | search only | Isaac Lab | - | not fetched |
| LEADS | anonymous.4open.science link in arXiv v1 | no | - | - | no public repo located |
| PSD | https://jonghaepark.github.io/psd | project page only | - | - | GitHub not verified |
| DoDont | https://mynsng.github.io/dodont/ | page fetched; no code link | - | - | Google Drive link claimed in appendix |
| BREEZE | https://github.com/Whiterrrrr/BREEZE | listed in abstract | - | - | not fetched |
| SUSD | https://github.com/hadi-hosseini/SUSD | listed in abstract | - | - | not fetched |
| VIC, Mohamed & Rezende 2015, Hierarchical Empowerment, LPE | none located | - | - | - | - |

---

## 7. References (arXiv IDs; all abstracts verified unless noted)

- Park, Rybkin, Levine. METRA. ICLR 2024. arXiv:2310.08887. https://arxiv.org/abs/2310.08887
- Park, Choi, Kim, Lee, Kim. LSD. ICLR 2022. arXiv:2202.00914
- Park, Lee, Lee, Abbeel. CSD. ICML 2023. arXiv:2302.05103
- Park, Kreiman, Levine. HILP. ICML 2024. arXiv:2402.15567
- Eysenbach, Gupta, Ibarz, Levine. DIAYN. ICLR 2019. arXiv:1802.06070
- Gregor, Rezende, Wierstra. VIC. 2016. arXiv:1611.07507
- Sharma, Gu, Levine, Kumar, Hausman. DADS. ICLR 2020. arXiv:1907.01657
- Laskin, Liu, Peng, Yarats, Rajeswaran, Abbeel. CIC. 2022. arXiv:2202.00161
- Liu, Abbeel. APS. ICML 2021. arXiv:2108.13956
- Liu, Abbeel. APT. NeurIPS 2021. arXiv:2103.04551
- Yarats, Fergus, Lazaric, Pinto. ProtoRL. ICML 2021. arXiv:2102.11271
- Seo, Chen, Shin, Lee, Abbeel, Lee. RE3. ICML 2021. arXiv:2102.09430
- Lee, Eysenbach, Parisotto, Xing, Levine, Salakhutdinov. SMM. 2019. arXiv:1906.05274 (search only)
- Laskin et al. URLB. NeurIPS 2021 D&B. arXiv:2110.15191
- Yarats et al. ExORL. 2022. arXiv:2201.13425
- Rajeswar et al. Mastering URLB from Pixels. ICML 2023. arXiv:2209.12016
- Mazzaglia et al. Choreographer. ICLR 2023. arXiv:2211.13350
- Touati, Ollivier. Learning One Representation to Optimize All Rewards. NeurIPS 2021. arXiv:2103.07945
- Touati, Rapin, Ollivier. Does Zero-Shot RL Exist? ICLR 2023. arXiv:2209.14935
- Jeen, Bewley, Cullen. Zero-Shot RL from Low Quality Data. NeurIPS 2024. arXiv:2309.15178
- Frans, Park, Abbeel, Levine. FRE. ICML 2024. arXiv:2402.17135
- Sun et al. DVFB. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/400693b9f90cd47ceae029071950179d-Abstract-Conference.html
- Armengol Urpi, Vlastelica, Martius, Coros. Epistemically-guided FB exploration. RLJ 2025. arXiv:2507.05477
- Zheng et al. BREEZE. NeurIPS 2025. arXiv:2510.15382
- Bagatella, Pirotta, Touati, Lazaric, Tirinzoni. TD-JEPA. 2025. arXiv:2510.00739
- Di Ventura, Kleuker, Plaat, Moerland. A Unified Framework for Zero-Shot RL. 2025/2026. arXiv:2510.20542
- Jeen, Bewley, Cullen. Zero-Shot RL Under Partial Observability. RLC 2025. arXiv:2506.15446
- Soft FB representations for general utilities. 2026. arXiv:2602.06769
- Hu, Armengol Urpi, Cheng, Coros. FB-MEBE. 2026. arXiv:2603.25464
- Jeen. On Zero-Shot RL (PhD thesis). 2025. arXiv:2508.16496
- Mohamed, Rezende. Variational Information Maximisation for Intrinsically Motivated RL. NIPS 2015. arXiv:1509.08731
- Levy, Rammohan, Allievi, Niekum, Konidaris. Hierarchical Empowerment. 2023. arXiv:2307.02728
- Levy, Allievi, Konidaris. Latent-Predictive Empowerment. 2024. arXiv:2410.11155
- Schneider et al. Information-Theoretic Policy Pre-Training with Empowerment. 2025. arXiv:2510.05996
- Bastankhah, Broderick, Eysenbach. Empowerment-Based Representation Learning. 2026. arXiv:2605.30656
- Kim et al. DoDont. NeurIPS 2024. arXiv:2406.00324
- Campos et al. EDL. ICML 2020. arXiv:2002.03647
- Kamienny et al. UPSIDE. ICLR 2022. arXiv:2110.14457
- Yang et al. BeCL. ICML 2023. arXiv:2305.04477
- Bai et al. CeSD. ICML 2024. arXiv:2405.16030
- Le Tolguenec et al. LEADS. NeurIPS 2024. arXiv:2406.10127
- Bae, Park, Lee. TLDR. CoRL 2024. arXiv:2407.08464
- Zhang et al. RSD. ICML 2025. arXiv:2506.21044
- Park et al. PSD. NeurIPS 2025. arXiv:2511.03187
- Hosseini, Soleymani Baghshah. SUSD. ICLR 2026. arXiv:2602.01619
- Hu, Wang, Stone, Martin-Martin. DUSDi. NeurIPS 2024. arXiv:2410.11251
- Xiao et al. Skill Regions Differentiation. 2025. arXiv:2506.14420
- Atanassov et al. Constrained Skill Discovery (quadruped). 2024. arXiv:2410.07877
- Cathomen, Mittal, Vlastelica, Hutter. D3. CoRL 2025. arXiv:2508.19953
- Rho, Garg, Byrd, Ha. SDAX. CoRL 2025. arXiv:2508.08982
- Rho, Trinh, Xu, Ha. Reference Grounded Skill Discovery. 2025. arXiv:2510.06203 (excluded, C4)
- Harvey, Nangue Tasse, Rosman, Ingram, James. Unsupervised Hierarchical Skill Discovery. ICML 2026. arXiv:2601.23156 (offline trajectories)
- Mohamed, Ji, Eysenbach, Berseth. Temporal Representations for Exploration (C-TeC). ICLR 2026. arXiv:2603.02008
- Jiang et al. Episodic Novelty Through Temporal Distance (ETD). ICLR 2025. arXiv:2501.15418
- Lu, Grover, Abbeel, Mordatch. LiSP. ICLR 2021. arXiv:2012.03548
- Xu et al. Continual Learning of Control Primitives: Skill Discovery via Reset-Games. 2020. arXiv:2011.05286 (search only)
