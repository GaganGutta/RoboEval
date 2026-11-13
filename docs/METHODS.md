# Methods

Precise description of every component, written so it can be reimplemented without our
code. Sections marked *(not yet implemented)* describe the plan, not something we ran.

## 1. Policy learning: BC-RNN

Behavior cloning with an LSTM and a Gaussian-mixture action head, as in robomimic.

| Hyperparameter | Value | Source |
|---|---|---|
| Observations | eef pos (3), eef quat (4), gripper qpos (2), object state | robomimic paper low-dim |
| RNN | LSTM, 2 layers, hidden 400 | paper (low-dim) / robomimic default |
| Sequence length (train) | 10 | paper |
| MLP between RNN and head | none | paper |
| Action head | GMM, 5 modes, softplus std, min std 1e-4 | paper / robomimic default |
| Optimizer | Adam, lr 1e-4, no decay, no L2 | paper |
| Batch size | 100 | paper |
| "Epoch" | 100 gradient steps | paper |
| Number of epochs | **600** (paper: 2000) | Exp 1 pilot, select seeds: mean success plateaus from epoch 400 |

Implementation: `train/bc_rnn.py` builds a robomimic `BCConfig` from
`configs/bc_rnn_lowdim.yaml` and calls `robomimic.scripts.train.train`. The training subset
is selected with a robomimic filter key. robomimic's in-training rollouts and validation are
**disabled**; checkpoints are saved at fixed epochs and evaluated separately (Section 2).

Deviations from the paper we must disclose: (1) 600 epochs instead of 2000. (2) The paper
reports the *best* checkpoint over evaluations run every 50 epochs; that number is selected
on the same rollouts it reports and is optimistically biased. We report the mean over a
fixed set of late checkpoints (Section 2).

Learning rate is constant (no schedule), so the training state at epoch k does not depend
on the planned total. The six Exp 1 pilot runs were trained to 1000 epochs; we use their
epoch 400/500/600 checkpoints, which are the same as those of a 600-epoch run (up to GPU
floating-point nondeterminism).

Compatibility shim: robomimic v0.5.0 loads a CLIP text model when
`robomimic.utils.lang_utils` is imported. Our policies are not language-conditioned, so
`train/robomimic_compat.py` replaces that module with a stub before robomimic is imported.
This does not change training or evaluation behavior for non-language policies.

## 2. Evaluation protocol (fixed, held-out)

- A policy is evaluated by running `n` closed-loop episodes in the robosuite environment
  stored in the dataset metadata (robosuite 1.5.1, MuJoCo 3.2.6), horizon 500 steps
  (robomimic registry value for these MH tasks). An episode succeeds if the env's
  `is_success()["task"]` becomes true at any step; the episode stops at first success.
- **Seeds.** Initial object placements in robosuite are drawn from numpy's global RNG.
  Before each episode we seed `random`, `numpy` and `torch` (the GMM head samples) with the
  episode seed. Two disjoint seed sets:
  - *report* set: seeds 100000–100099 (n = 100). Used only for reported numbers.
  - *select* set: seeds 200000+ . Used for any tuning decision (e.g. training budget).
  Every condition is evaluated on the same initial states.
- **Checkpoint rule.** Each run is scored by the mean success of its epoch 400, 500 and 600
  checkpoints, each evaluated on the full report set (3 × 100 episodes). The rule was
  fixed after the pilot and before any report-seed evaluation, from select-seed data only.
  Why: in the pilot, one run's success moved by up to 0.4 between checkpoints 200 epochs
  apart, much more than rollout noise. Averaging three checkpoints reduces that noise
  without choosing a checkpoint by its result. (Changed from the original plan, which was
  "last checkpoint only".)
- **Uncertainty.** Per checkpoint: Wilson 95% interval on k/n. Across training seeds (≥ 3): mean
  and sample standard deviation of per-run success rates.
- Implementation: `eval/rollout.py`. Episodes run in parallel worker processes, each with
  its own env and a CPU copy of the policy. The episode loop only needs
  `reset()`, `step()`, `is_success()`, so a learned simulator can replace the env later.

## 3. Data splits

See `docs/DATA_CARD.md`. Per task: 20 held-out "better" demos (10 per better operator,
sampled with seed 0) form the trusted validation set for Stage 3. All training subsets are
drawn from the remaining 280-demo pool. Random subsets of size k for training seed s are
drawn uniformly from the pool with `numpy.random.default_rng(10000 + s)`.

Note on fairness: the held-out set uses operator labels, which random selection does not.
Methods that use it (Stage 3) get information the random baseline does not.

## 4. Scoring stages

All scores are oriented so that **higher = predicted more useful**. Scores are computed on
the 280-demo pool of a task; the held-out set is never scored or used as reference.

### Stage 1 — Hygiene (`scoring/hygiene.py`; implemented, not yet evaluated)

Inputs per episode: actions a_t ∈ [-1,1]^7 (6 arm + 1 gripper), end-effector position
p_t ∈ R^3, joint velocities q̇_t ∈ R^7. Control period Δt = 0.05 s.

| Feature | Definition |
|---|---|
| length | T |
| idle_ratio | (1/T) Σ_t 1[ max_i |a_t,i| < 0.05 (i ≤ 6)  ∨  ‖q̇_t‖ < 0.05 rad/s ] |
| jerk_mean / jerk_p95 | mean / 95th pct of ‖Δ³p_t‖ / Δt³ |
| action_jerk | mean ‖Δ²a_t,1:6‖ |
| gripper_flips | number of sign changes of a_t,7 |
| dup_dist | distance to nearest other episode after linearly resampling [p_t, object_t] to 50 points, per-dimension standardization, divided by √dim |

hygiene_score = −mean(z̃(idle_ratio), z̃(jerk_mean), z̃(action_jerk), z̃(gripper_flips)),
where z̃ is the robust z-score (x − median) / (1.4826·MAD) over the pool.

Flags: idle_ratio > 0.2; z̃(jerk_mean) > 3 or z̃(action_jerk) > 3; z̃(dup_dist) < −3.

Design choices. (1) Length is **excluded** from hygiene_score and reported as a separate
baseline score (−length), because on MH data length tracks operator tier and would make
the score look informative for a trivial reason. (2) Thresholds are round numbers fixed
before computing any AUROC. We did look at idle-action statistics for 40 better and 40
worse Can demos when choosing what "near zero" means (worse: 7.7% of steps below 0.05;
better: 0.6%), so the idle feature is not fully blind to labels.

### Stage 2 — Consistency (`scoring/consistency.py`; implemented, not yet evaluated)

State s_t = [eef pos, eef quat, gripper qpos, object state], standardized per dimension
over all pool timesteps. For each timestep t of episode i, N_t = the K = 16 nearest
timesteps (Euclidean) belonging to other episodes. With ā = mean of neighbors' arm actions
and σ = mean ‖a_j − ā‖ over neighbors:

  dis_t = ‖a_t,1:6 − ā‖ / (σ + 10⁻³),  consistency_score_i = −mean_t dis_t.

Normalizing by the neighbors' spread avoids penalizing states where demonstrators
legitimately act differently (multimodal regions). Known weakness: it measures agreement
with the majority; if most demos are poor, it rewards poor behavior.

### Stages 3–6 *(not yet implemented)*

- Stage 3 Influence: cosine similarity of per-episode last-layer gradients (random
  projection) with the held-out validation gradient, averaged over checkpoints.
- Stage 4 Closed loop: success/failure classifier on policy rollouts applied to demos.
- Stage 5 Combination: weighted sum of normalized scores; weights tuned on Lift only.
- Stage 6 Greedy selection with diversity term.
