# Experiments

Common protocol for all experiments: `docs/METHODS.md` §1–3 (BC-RNN, fixed-seed
evaluation, last-checkpoint rule, 280-demo pool + 20 held-out "better" demos).

---

## Exp 1 — Baselines on Can MH (answer key)

**Hypothesis.** Training-data quality affects BC-RNN success on Can MH: better-only >
random of the same size > worse-only. If not, the operator labels are not a useful answer
key for the rest of the project.

**Setup.**
- Task: Can (PickPlaceCan), MH low-dim, robosuite 1.5.1.
- Conditions (from the 280-demo pool):

  | Condition | Filter key | # demos |
  |---|---|---|
  | all | `pool` | 280 |
  | better only (oracle) | `better_pool` | 80 |
  | worse only | `worse_pool` | 100 |
  | random 80 | `random80_s<seed>` | 80 |

- Training seeds: 0, 1, 2 (random80 draws a new subset per seed).
- Metric: success of the last checkpoint over 100 report-seed episodes, horizon 500.
  Mean ± std over seeds.
- Script: `experiments/exp1_baselines.py`.

**Pilot (training-budget choice).** 6 runs (all/better/worse/random80 at seed 0; all/worse
at seed 1), 1000 epochs, checkpoints every 100 epochs, learning curves on the *select* seed
set (50 episodes per checkpoint). Purpose: decide the epoch budget without touching
report seeds.

**Pilot result (select seeds only, n = 50 episodes per checkpoint, commit `e2edc4c`).**
Source: `results/exp1/all_evals.csv`; figure `figures/out/exp1_learning_curves.{png,pdf}`.

| Run | ep200 | ep400 | ep600 | ep800 | ep1000 |
|---|---|---|---|---|---|
| all s0 | 0.14 | 0.54 | 0.82 | 0.60 | 0.42 |
| all s1 | 0.68 | 0.96 | 0.58 | 0.72 | 0.80 |
| better s0 | 0.18 | 0.62 | 0.30 | 0.10 | 0.10 |
| worse s0 | 0.24 | 0.60 | 0.50 | 0.38 | 0.64 |
| worse s1 | 0.56 | 0.60 | 0.72 | 0.70 | 0.72 |
| random80 s0 | 0.22 | 0.54 | 0.70 | 0.72 | 0.62 |
| mean of 6 | 0.34 | 0.64 | 0.60 | 0.54 | 0.55 |

Observations (findings, not interpretation):
1. Mean success rises to epoch 400 and is flat or slightly lower afterwards.
2. Large swings between consecutive checkpoints of the same run (all s1: 0.96 → 0.58 →
   0.80; all s0: 0.82 → 0.42). A 50-episode Wilson interval is about ±0.13, so swings of
   0.3–0.4 are larger than rollout noise alone.
3. Large seed-to-seed spread at the same epoch (all at ep1000: 0.42 vs 0.80).
4. better-only (80 demos) peaks at epoch 400 (0.62) and falls to 0.10 by epoch 800.
   Plausible cause (speculation): overfitting, since at a fixed number of gradient steps a
   smaller subset is seen more times. This means the ranking of conditions depends on the
   training budget.
5. Success is well below the near-100% often quoted for BC-RNN on Can MH. The robomimic
   paper reports the best checkpoint over many evaluations and 2000 epochs; we report a
   single checkpoint. We have not verified the paper number ourselves.

Consequence: with the last-checkpoint rule and 3 seeds, differences smaller than roughly
0.3 between conditions will not be distinguishable. The budget and checkpoint rule are
being decided now, on select seeds only, before any report-seed evaluation.

**Results.** *Pending (no report-seed evaluation has been run).*

**Interpretation.** *Pending.*

**Limitations.** See `docs/LIMITATIONS.md` items 1–6.

---

## Exp 2 — Ranking (Stages 1–2 vs operator labels) — *not started*
## Exp 3 — Selection (top 40%) on Lift, Can, Square — *not started*
## Exp 4 — Closed loop vs offline — *not started*
## Exp 5 — Corruption detection — *not started*
## Exp 6 — Human + MimicGen mixture — *not started*
