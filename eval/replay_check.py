"""Sanity check: replay recorded demo actions open-loop from each demo's initial sim state
and check the task succeeds. If many fail, the installed simulator (e.g. MuJoCo version)
does not match the one the data was collected with, and policy success numbers would not
be comparable to published ones. Also reports raw sim throughput (env steps/sec).

Usage: python -m eval.replay_check --task can --n 20
"""
import argparse
import json
import os
import time

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import train.robomimic_compat  # noqa: F401
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    import mujoco
    from data.mh import raw_path, list_demos

    path = raw_path(args.task)
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": ["robot0_eef_pos"]})
    env_meta = FileUtils.get_env_metadata_from_dataset(path)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)

    demos = list_demos(path)
    # Evenly spaced demos so all operators are covered.
    pick = [demos[i] for i in np.linspace(0, len(demos) - 1, args.n).astype(int)]
    results, steps, t0 = [], 0, time.time()
    with h5py.File(path, "r") as f:
        for d in pick:
            g = f[f"data/{d}"]
            env.reset()
            env.reset_to({"states": g["states"][0], "model": g.attrs["model_file"]})
            success = False
            for a in g["actions"][:]:
                env.step(a)
                steps += 1
                success = success or env.is_success()["task"]
            results.append({"demo": d, "success": bool(success)})
    dt = time.time() - t0
    rate = np.mean([r["success"] for r in results])
    summary = {"task": args.task, "n": len(results), "replay_success_rate": float(rate),
               "env_steps_per_sec": steps / dt, "mujoco": mujoco.__version__, "results": results}
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
