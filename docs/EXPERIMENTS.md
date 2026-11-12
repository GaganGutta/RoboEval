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

**Results.** *Pending.*

**Interpretation.** *Pending.*

**Limitations.** See `docs/LIMITATIONS.md` items 1–6.

---

## Exp 2 — Ranking (Stages 1–2 vs operator labels) — *not started*
## Exp 3 — Selection (top 40%) on Lift, Can, Square — *not started*
## Exp 4 — Closed loop vs offline — *not started*
## Exp 5 — Corruption detection — *not started*
## Exp 6 — Human + MimicGen mixture — *not started*
