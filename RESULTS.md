# Results log

Every number here comes from a logged run (config, seed, and git commit recorded in the
JSON under `results/`). Negative and null results are reported as they are.

## 2026-11-12 — Sanity: open-loop demo replay (MuJoCo 3.2.6, robosuite 1.5.1)

Replay recorded actions from each demo's first state; 30 evenly spaced demos per task.

| Task | Replay success | Env steps/s (1 process) |
|---|---|---|
| Lift | 30/30 | 36 |
| Can | 29/30 | 41 |
| Square | 24/30 | 38 |

Source: `results/sanity/replay_{lift,can,square}.json`. Square's 80% suggests the sim does
not exactly reproduce collection-time dynamics; see `docs/LIMITATIONS.md` #5.

## Exp 1 — Can MH baselines (2026-11-13)

**Setup.** BC-RNN (paper low-dim settings), 600 epochs, Can MH, conditions drawn from the
280-demo pool. Per run: mean success of the epoch 400/500/600 checkpoints, each on the same
100 report-seed episodes (seeds 100000–100099). 3 training seeds per condition. Protocol
fixed before any report-seed evaluation (`docs/METHODS.md` §2). Evaluation code at commit
`e001287`. Sources: `results/exp1/per_run.csv`, `summary.csv`, `tests.csv`,
`all_evals.csv`; figure `figures/out/exp1_success.{png,pdf}`.

| Condition | # demos | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|---|
| all | 280 | 0.733 | 0.777 | 0.457 | **0.656 ± 0.174** |
| better only (oracle) | 80 | 0.477 | 0.657 | 0.323 | **0.486 ± 0.167** |
| random 80 | 80 | 0.660 | 0.827 | 0.700 | **0.729 ± 0.087** |
| worse only | 100 | 0.593 | 0.543 | 0.617 | **0.584 ± 0.037** |

Pairwise Welch t-tests on per-run scores (n = 3 vs 3), Holm-corrected over 5 planned pairs:

| Comparison | Difference | p | Holm p |
|---|---|---|---|
| random80 − worse | +0.144 | 0.086 | 0.43 |
| better − random80 | −0.243 | 0.111 | 0.44 |
| all − better | +0.170 | 0.289 | 0.87 |
| better − worse | −0.099 | 0.414 | 0.87 |
| all − random80 | −0.073 | 0.561 | 0.87 |

No difference is statistically significant. With 3 seeds, the tests have low power.

Descriptive, per checkpoint (mean over 3 seeds; not a separate test):

| Condition | ep400 | ep500 | ep600 |
|---|---|---|---|
| all | 0.600 | 0.677 | 0.690 |
| better | 0.657 | 0.473 | 0.327 |
| random80 | 0.673 | 0.737 | 0.777 |
| worse | 0.590 | 0.600 | 0.563 |

**Findings.**
1. The hypothesis "better-only > random of the same size > worse-only" is **not supported**.
   better-only had the lowest mean (0.49) and random 80 the highest (0.73). The
   better-vs-random gap points the opposite way from the hypothesis but is not significant
   (Holm p = 0.44).
2. better-only success fell from checkpoint 400 to 600 in all 3 seeds (0.64→0.30,
   0.82→0.49, 0.51→0.19). No other condition shows this. It matches the pilot.
3. All 280 demos did not beat a random 80 (0.66 vs 0.73, not significant).
4. Seed-to-seed spread is large (std up to 0.17) even after averaging 3 checkpoints.

**Training-set sizes in timesteps** (`results/exp1/subset_sizes.csv`): all 59,924;
better 11,432; random 80 16,515 / 17,280 / 19,954 (seeds 0/1/2); worse 30,439. So
"same size" in demos is not the same size in training samples: random 80 has 1.4–1.7×
the timesteps of better-only, because better operators' demos are short.

**Speculation (not tested).** better-only's decline looks like overfitting: at a fixed
60,000 gradient steps, its 11,432 timesteps are revisited more often than any other
subset's. If so, Exp 1 compares data *amount* (in timesteps) as much as data *quality*.

**Relation to the core claim.** This does not test our pipeline yet, but it bears on it
directly: an oracle that keeps exactly the demos labeled best did **not** beat a random
subset of the same size under this protocol. So, at this training budget, matching the
operator labels is not a route to beating random, and Exp 2's AUROC against labels cannot
be read as evidence of usefulness. This is in line with CuraEval (arXiv:2610.38221).

**Suggested next step.** Run Exp 2 (no training, minutes) as planned, reading AUROC as
"agreement with labels" only. Before Exp 3, decide how to set the training budget for
subsets of different sizes (e.g. a fixed number of passes over the data, or a budget chosen
per subset size on select seeds), since Exp 1 suggests the budget rule can decide the
outcome.
