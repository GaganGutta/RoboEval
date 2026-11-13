# [Title TODO] Scoring robot demonstrations by their effect on policy success

*Draft skeleton. Sections fill in as results arrive. No claim appears here unless a logged
run supports it; pending items are marked TODO.*

## Abstract
TODO — write last.

## 1. Introduction
- Problem: imitation-learning datasets mix demonstrations of very different quality;
  training on all of them can hurt. Collecting more data is expensive; choosing which data
  to keep is cheaper.
- Question: can per-episode scores computed from the data (and optionally from a trained
  policy) select a subset that trains a better policy than (a) all data and (b) a random
  subset of the same size?
- Testbed with ground truth: robomimic multi-human datasets, where operators are labeled
  by proficiency and policy success can be measured in simulation.
- Contributions: TODO (only after results).

## 2. Related work
TODO — from `docs/RELATED_WORK.md`.

## 3. Method
TODO — condensed from `docs/METHODS.md` once stages are implemented.

## 4. Experimental setup
- Datasets: robomimic v1.5 MH low-dim, Lift / Can / Square (`docs/DATA_CARD.md`).
- Policy: BC-RNN (robomimic paper hyperparameters; 600 epochs of 100 steps).
- Evaluation: mean of the epoch 400/500/600 checkpoints, each on 100 fixed-seed episodes;
  3 training seeds, mean ± std; Welch t-tests with Holm correction.
- Baselines: all data; random subset of equal size; oracle (better-operator demos only).

## 5. Results
### 5.1 Does data quality matter? (Exp 1)
On Can MH with BC-RNN (600 epochs; mean of 3 late checkpoints; 100 fixed evaluation
episodes; 3 training seeds), training only on the better operators' 80 demos gave
0.49 ± 0.17 success, versus 0.73 ± 0.09 for random subsets of 80 demos, 0.66 ± 0.17 for all
280 demos and 0.58 ± 0.04 for the worse operators' 100 demos. No difference was significant
after Holm correction. The better-only policies declined between epoch 400 and 600 in every
seed. Size-matched subsets were not matched in timesteps: better-only had 11.4k vs
16.5–20.0k for random. TODO: whether a timestep-matched or budget-adjusted comparison
changes this.
### 5.2 Do offline scores recover operator quality? (Exp 2)
TODO
### 5.3 Subset selection (Exp 3)
TODO
### 5.4 Closed-loop scoring (Exp 4)
TODO
### 5.5 Corruption detection (Exp 5)
TODO

## 6. Limitations
TODO — from `docs/LIMITATIONS.md`.

## 7. Conclusion
TODO
