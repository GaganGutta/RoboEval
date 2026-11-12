# data_eval — scoring robot demonstrations by how much they help a policy

Proof of concept. Score each demonstration episode, select a training subset, and test
whether it beats training on all data and on a random subset of the same size. Ground
truth: robomimic multi-human (MH) datasets (operators labeled worse / okay / better) and
policy success measured in simulation.

Status and numbers: [`RESULTS.md`](RESULTS.md). Full documentation: [`docs/`](docs/).

## Layout
```
configs/       training / eval settings (bc_rnn_lowdim.yaml)
data/          download.py, mh.py (loading + filter keys), splits.py (fixed splits)
scoring/       Stage 1–5 scores            (in progress)
selection/     Stage 6–7 subset / mixture  (in progress)
train/         bc_rnn.py (robomimic wrapper), robomimic_compat.py
eval/          rollout.py (fixed-seed parallel rollouts), replay_check.py
experiments/   one script per experiment
figures/       scripts that build every figure from results/ (out/ holds PNG + PDF)
results/       eval JSON/CSV (small, committed)
docs/          lab notebook, methods, experiments, data card, reproducibility, ...
setup/         setup_wsl.sh
scripts/       wsl_run.sh (activate venv, set PYTHONPATH, run)
```
Not committed: `data/raw`, `data/processed`, `runs/` (checkpoints), `logs/`.

## Reproduce
Runs in Linux (we use WSL2 Ubuntu on Windows; robomimic does not support Windows).
```bash
bash setup/setup_wsl.sh                                   # venv + robomimic v0.5.0 + robosuite v1.5.1
bash scripts/wsl_run.sh python data/download.py           # 0.29 GB: Lift/Can/Square MH low-dim
bash scripts/wsl_run.sh python -m data.splits             # fixed held-out / pool splits
bash scripts/wsl_run.sh python -m experiments.exp1_baselines --phase train --runs full
bash scripts/wsl_run.sh python -m experiments.exp1_baselines --phase eval  --runs full --seedset report
bash scripts/wsl_run.sh python -m experiments.exp1_baselines --phase report
bash scripts/wsl_run.sh python figures/exp1.py
```
Per-experiment commands and runtimes: `docs/REPRODUCIBILITY.md`.
