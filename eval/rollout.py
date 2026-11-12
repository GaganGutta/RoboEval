"""Closed-loop evaluation of a trained policy with FIXED environment seeds.

Why fixed seeds: every condition (all data, random subset, selected subset, ...) is scored
on the exact same set of initial object placements, which removes one source of variance
from comparisons. Reported numbers use seeds report_seed_base + [0, n); anything used for
tuning (e.g. picking a training budget) uses a disjoint seed range (select_seed_base).

Seeding: robosuite's placement samplers draw from numpy's GLOBAL RNG
(robosuite/utils/placement_samplers.py), so we seed np.random right before env.reset().
We also seed torch because the GMM policy head samples actions.

The episode loop (run_episode) only needs an env with reset()/step()/is_success(), so a
learned world model can be dropped in later (see scoring/closed_loop.py).

Usage:
    python -m eval.rollout --ckpt runs/.../model_epoch_1000.pth --n 100 --seed_base 100000
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

_POLICY = None
_ENV = None
_HORIZON = None
_SAVE_TRAJ = False


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(policy, env, horizon, seed, save_traj=False):
    """One rollout. Returns dict with success, length, and optionally the trajectory."""
    seed_everything(seed)
    policy.start_episode()
    obs = env.reset()
    traj = {"obs": [], "actions": []} if save_traj else None
    success, t = False, 0
    for t in range(horizon):
        act = policy(ob=obs)
        if save_traj:
            traj["obs"].append({k: np.asarray(v).copy() for k, v in obs.items()})
            traj["actions"].append(np.asarray(act).copy())
        obs, _, _, _ = env.step(act)
        if env.is_success()["task"]:
            success = True
            break
    out = {"seed": int(seed), "success": bool(success), "length": int(t + 1)}
    if save_traj:
        out["traj"] = traj
    return out


_INIT_ERROR = None


def _init_worker(ckpt, horizon, save_traj):
    """Load policy + env once per worker process.

    Errors are stored, not raised: multiprocessing.Pool silently respawns workers whose
    initializer raises, which hangs forever. Raising inside _worker instead makes
    pool.map fail loudly.
    """
    global _POLICY, _ENV, _HORIZON, _SAVE_TRAJ, _INIT_ERROR
    try:
        import torch
        torch.set_num_threads(1)   # many workers in parallel; avoid CPU oversubscription
        import train.robomimic_compat  # noqa: F401  (must precede robomimic imports)
        import robomimic.utils.file_utils as FileUtils
        _POLICY, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=torch.device("cpu"), verbose=False)
        _ENV, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False)
        _HORIZON, _SAVE_TRAJ = horizon, save_traj
    except Exception as e:  # noqa: BLE001
        import traceback
        _INIT_ERROR = traceback.format_exc()


def _worker(seed):
    if _INIT_ERROR is not None:
        raise RuntimeError("worker init failed:\n" + _INIT_ERROR)
    return run_episode(_POLICY, _ENV, _HORIZON, seed, _SAVE_TRAJ)


def evaluate(ckpt, n, seed_base, horizon=500, workers=12, save_traj=False):
    seeds = [seed_base + i for i in range(n)]
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(workers, initializer=_init_worker, initargs=(ckpt, horizon, save_traj)) as pool:
        episodes = pool.map(_worker, seeds, chunksize=1)
    return {
        "ckpt": os.path.abspath(ckpt), "n": n, "seed_base": seed_base, "horizon": horizon,
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "eval_seconds": time.time() - t0,
        "episodes": episodes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed_base", type=int, required=True)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = evaluate(args.ckpt, args.n, args.seed_base, args.horizon, args.workers)
    print(json.dumps({k: v for k, v in res.items() if k != "episodes"}))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
