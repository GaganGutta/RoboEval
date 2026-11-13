# Reproducibility

## Hardware
- Laptop: AMD Ryzen 9 8945HS (8 cores / 16 threads), 31 GB RAM (WSL2 VM capped at 15 GB),
  NVIDIA RTX 4070 Laptop GPU 8 GB (75 W cap), Windows 11 + WSL2 Ubuntu 26.04.
- Exact system info: `docs/env/system.txt`.

## Software
- Python 3.10.22 (uv-managed venv at `~/venvs/data_eval` inside WSL).
- robomimic **v0.5.0** (`ae5799f0`), robosuite **v1.5.1** (`51cc0178`), MuJoCo **3.2.6**,
  torch 2.14.1 (CUDA 13.0 runtime), numpy 1.26.4. Full list: `docs/env/pip_freeze.txt`.
- Setup from scratch (inside WSL): `bash setup/setup_wsl.sh`
- Setup problems we hit and how they are handled (all in the script or code):
  - `egl_probe` fails to build with CMake 4 → `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
  - pip picks MuJoCo 3.17, which breaks robosuite 1.5.1 → pinned 3.2.6.
  - robomimic loads a CLIP model at import → stubbed (`train/robomimic_compat.py`).

## Data
`bash scripts/wsl_run.sh python data/download.py --tasks lift can square`, then
`bash scripts/wsl_run.sh python -m data.splits`. Hashes and HF revision: `data/raw/manifest.json`.

## Measured throughput (2026-11-12, this machine)
| What | Measured |
|---|---|
| BC-RNN training, 1 job | 0.031 min/epoch (100 steps) |
| BC-RNN training, 4 concurrent | 0.068 min/epoch each (≈59 epochs/min total) |
| BC-RNN training, 6 concurrent | 0.067 min/epoch each (≈90 epochs/min total) |
| BC-RNN training, CPU only | 0.22 min/epoch |
| Env step, one process (robosuite OSC, 25 substeps) | ≈ 40–50 steps/s |
| Rollout eval, 8 workers | ≈ 270 env steps/s total; ≈ 1.1 GB RAM per worker |

## Results provenance
Every eval JSON under `results/` records the checkpoint path, seed set, training seed and
the git commit of the code that produced it (`git_commit`, suffixed `-dirty` if the
working tree had uncommitted changes). Each training run writes `train_done.json` with the
full config and runtime.

## Regenerating results
| Result | Command (inside WSL, from repo root, prefix `bash scripts/wsl_run.sh`) |
|---|---|
| Replay sanity check | `python -m eval.replay_check --task can --n 30 --out results/sanity/replay_can.json` |
| Exp 1 train | `python -m experiments.exp1_baselines --phase train --runs full` |
| Exp 1 eval | `python -m experiments.exp1_baselines --phase eval --runs full --seedset report` |
| Exp 1 tables (summary, per-run, tests, subset sizes) | `python -m experiments.exp1_baselines --phase report` |
| Exp 1 figures | `python -m figures.exp1` |
