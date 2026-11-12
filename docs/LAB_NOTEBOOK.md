# Lab notebook

Dated entries, newest last. Includes dead ends.

## 2026-11-12 — Environment, data, calibration, Exp 1 pilot

**Goal.** Set up the stack, verify data and labels, measure compute costs, start Exp 1.

**Environment.** Windows 11 laptop with RTX 4070 Laptop (8 GB). robomimic only supports
Linux/macOS; `egl_probe` (a robomimic dependency) is source-only and does not build on
Windows, so everything runs in WSL2 Ubuntu with a uv venv (Python 3.10).

Install problems, in order:
1. `egl_probe` build failed: CMake 4 removed support for `cmake_minimum_required < 3.5`.
   Fixed with `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
2. A dependency pulled torch 2.14.1 instead of the 2.5.1 we asked for. Kept 2.14.1 and
   pinned it so the environment is reproducible.
3. pip resolved MuJoCo 3.17, which fails inside robosuite 1.5.1
   (`binding_utils.get_joint_qpos_addr` assertion on joint type). Pinned MuJoCo 3.2.6.

**Data.** Downloaded Lift/Can/Square MH low-dim (0.29 GB total, HF revision `74fa0184`).
Confirmed operator-tier filter keys `worse`/`okay`/`better` and also per-operator keys,
which the docs did not mention. Observation: episode length tracks tier strongly
(Can mean: worse 304, okay 181, better 143 steps). Any score that correlates with length
will look like a quality detector. Decided to report length alone as a baseline in Exp 2.

**Replay sanity check.** Open-loop replay of recorded actions from each demo's first state:
Lift 30/30, Can 29/30, Square 24/30 succeed. Open-loop replay is not exact in robosuite, so
some failures are expected; recorded as a limitation (Square).

**Compute calibration.**
- Env stepping is slow: ~40–50 control steps/s per process. Profiled: raw `mj_step` runs at
  11k/s; the cost is robosuite's Python OSC controller running on all 25 substeps of each
  20 Hz control step. Not a bug. Workaround: parallel rollout workers.
- First attempt at 6–12 parallel rollout workers ran WSL out of memory (15 GB cap) and
  crashed the WSL service (`Wsl/Service/E_UNEXPECTED`); needed `wsl --shutdown`.
  Cause found: robomimic v0.5.0 loads a CLIP-ViT-L/14 text model on import of
  `lang_utils`. Stubbed it (`train/robomimic_compat.py`). Remaining per-worker memory is
  ~1.1 GB (torch import ~600 MB, env ~380 MB). 8 workers peak at ~8.8 GB → we use 8.
- A bug in our first rollout code: `multiprocessing.Pool` respawns workers forever if the
  initializer raises, so a failing init hung instead of erroring. Fixed: errors are stored
  and re-raised from the task function.
- Training: 0.031 min/epoch alone; 6 concurrent jobs at 0.067 min/epoch each (≈2.8× total
  throughput). GPU memory use is tiny; the bottleneck is per-step overhead.

**Decision: training budget.** At the paper's 2000 epochs, Exp 1 (12 runs) needs ~4.4 h of
training; at 1000 epochs ~2.2 h. Both exceed the 2-hour approval threshold. Running a pilot
first (6 runs × 1000 epochs, ~70 min), evaluated every 100–200 epochs on the *select* seed
set, to decide whether 1000 epochs is enough. Pilot runs: all/better/worse/random80 at seed
0, all/worse at seed 1.

**Status:** pilot training in progress.

**Literature check (background agent; key items re-verified by hand via the arXiv API).**
Close priors exist for almost every stage (details and citations in `docs/RELATED_WORK.md`):
- *CuraEval* (Xu et al., arXiv:2610.38221, CoRL 2026 workshop) tests our core claim on
  the same data (robomimic MH, BC-RNN and Diffusion Policy, size-matched random subsets,
  full-data baseline). Abstract: no curated condition significantly beats full-data
  training after Holm correction; the PSD smoothness score is nearly identical to episode
  length on Square; a length-based gain flipped sign when checkpoint reporting, rollout count
  and eval seeds changed together.
- *RoboDrop* (Xu et al., arXiv:2609.10021) is close to our Stage 3: per-sample gradient
  cosine similarity to a trusted clean validation set, evaluated partly on robomimic MH.
- DemInf ≈ Stage 2, Demo-SCORE = Stage 4, CUPID combines influence with rollouts,
  Re-Mix reweights MH operators.
- Per the agent's reading of the robomimic paper (not yet re-checked by us): BC-RNN
  low-dim on full MH reaches 100% on Lift and Can; on Square "better"-only was below
  full MH.

What this changes: (1) we cannot claim novelty for individual stages; the contribution,
if any, is a controlled head-to-head comparison and the combination. (2) A positive result
for the core claim is not the expected outcome; a careful null result is plausible and still
worth reporting. (3) It validates protocol choices already made: fixed held-out seeds,
last-checkpoint reporting, length as an explicit baseline, size-matched random subsets.
(4) We should add multiple-comparison correction (Holm) to Exp 3.
