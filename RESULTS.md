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

## Exp 1 — Can MH baselines

*Pilot running (training-budget choice). No results yet.*
