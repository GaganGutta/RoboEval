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
- Policy: BC-RNN (robomimic paper hyperparameters; training budget TODO).
- Evaluation: last checkpoint, 100 fixed-seed episodes, ≥ 3 training seeds, mean ± std.
- Baselines: all data; random subset of equal size; oracle (better-operator demos only).

## 5. Results
### 5.1 Does data quality matter? (Exp 1)
TODO
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
