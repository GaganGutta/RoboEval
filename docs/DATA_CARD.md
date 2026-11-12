# Data card

## Datasets used

### robomimic v1.5 multi-human (MH), low-dim

| Task | File (HF path) | Size | SHA-256 (first 12) | Demos |
|---|---|---|---|---|
| Lift | `v1.5/lift/mh/low_dim_v15.hdf5` | 50.7 MB | `d8bc0de2ca48` | 300 |
| Can (PickPlaceCan) | `v1.5/can/mh/low_dim_v15.hdf5` | 112.8 MB | `c4a34c837913` | 300 |
| Square (NutAssemblySquare) | `v1.5/square/mh/low_dim_v15.hdf5` | 123.3 MB | `da7dee0b6da4` | 300 |

- Source: Hugging Face dataset repo `amandlek/robomimic`, pinned revision
  `74fa018461f479cd9fd15b924a16103012096203`. Full hashes in `data/raw/manifest.json`
  (written by `data/download.py`).
- License: **MIT** (Hugging Face dataset card `license: mit`). robomimic code: MIT.
  robosuite code: MIT. MIT permits publishing results and derived artifacts with
  attribution; cite robomimic (Mandlekar et al., CoRL 2021) and robosuite.
- Collection (from the robomimic docs, `docs/datasets/robomimic_v0.1.md`): 6 operators on
  RoboTurk, 50 successful demos each; operators labeled 2 "worse", 2 "okay", 2 "better".
  All 300 demos are successful task completions.
- Environment metadata in the files: robosuite env version **1.5.1**, OSC_POSE controller,
  control frequency 20 Hz.

### Labels present in the files (verified 2026-11-12)

`mask/` filter keys include `worse`, `okay`, `better` (100 demos each), per-operator keys
`{worse,okay,better}_operator_{1,2}` (50 each), pairwise unions (`okay_better`, ...),
robomimic's own `train`/`valid` split, and `20_percent`/`50_percent` subsets.

Mean episode length by tier (steps at 20 Hz):

| Task | worse | okay | better |
|---|---|---|---|
| Lift | 145 | 94 | 72 |
| Can | 304 | 181 | 143 |
| Square | 357 | 265 | 185 |

Episode length differs strongly by tier, so any score correlated with length will look
like a "quality" score. We report length alone as a baseline score in Exp 2.

### Observations used
Low-dim only: `robot0_eef_pos` (3), `robot0_eef_quat` (4), `robot0_gripper_qpos` (2),
`object` (task dependent; 10-D for Lift, 14-D for Can). Same keys as the robomimic paper's low-dim runs.
Actions: 7-D (OSC pose delta + gripper).

## Processing
1. `data/download.py`: download, hash, write manifest. Raw files are never modified.
2. `data/mh.py::working_path`: copy each raw file to `data/processed/<task>_mh_low_dim.hdf5`.
3. `data/splits.py` (seed 0) adds filter keys to the working copy:
   - `heldout_val`: 20 "better" demos (10 from each better operator). Trusted validation set
     for Stage 3 only. Never trained on or selected in any condition.
   - `pool`: the other 280 demos. All subsets are drawn from here.
   - `{worse,okay,better}_pool`: tiers within the pool (100 / 100 / 80).
4. Experiment scripts add more filter keys (e.g. `random80_s0`), one per subset.

## Not yet used
- MimicGen datasets (`amandlek/mimicgen_datasets`, CC-BY 4.0; code under NVIDIA Source Code
  License). Planned for Exp 6. Generated with robosuite 1.4.x, which MimicGen states is
  incompatible with robosuite 1.5+. Mixing requires a separate environment and an action
  convention check. Note: robomimic's own `mg` datasets are "machine-generated" policy
  rollouts, not MimicGen.
