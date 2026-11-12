"""Fixed base splits shared by every experiment.

heldout_val : 20 "better" demos (10 from each better operator), sampled once with seed 0.
              Used ONLY as the trusted validation set for Stage 3 (influence). Excluded from
              training and selection in every condition, so no condition sees them.
pool        : the remaining 280 demos. All subsets (all / random / selected / oracle) are
              drawn from here.
{worse,okay,better}_pool : quality tiers within the pool (100 / 100 / 80 demos).

Usage: python -m data.splits --tasks lift can square
"""
import argparse
import json
import os

import numpy as np

from data.mh import QUALITY_KEYS, read_filter_key, working_path, write_filter_key, list_demos

SPLIT_SEED = 0
HELDOUT_PER_BETTER_OPERATOR = 10


def make_base_splits(task):
    path = working_path(task)
    rng = np.random.default_rng(SPLIT_SEED)
    heldout = []
    for op in ["better_operator_1", "better_operator_2"]:
        demos = read_filter_key(path, op)
        heldout += list(rng.choice(demos, HELDOUT_PER_BETTER_OPERATOR, replace=False))
    pool = [d for d in list_demos(path) if d not in set(heldout)]
    write_filter_key(path, "heldout_val", heldout)
    write_filter_key(path, "pool", pool)
    sizes = {"heldout_val": len(heldout), "pool": len(pool)}
    for q in QUALITY_KEYS:
        tier = [d for d in read_filter_key(path, q) if d in set(pool)]
        write_filter_key(path, f"{q}_pool", tier)
        sizes[f"{q}_pool"] = len(tier)
    return path, sizes


def random_subset(task, size, seed, key=None):
    """Uniform random subset of the pool. Seeded so it is reproducible and independent
    of the training seed stream."""
    path = working_path(task)
    pool = read_filter_key(path, "pool")
    rng = np.random.default_rng(10_000 + seed)
    demos = list(rng.choice(pool, size, replace=False))
    key = key or f"random{size}_s{seed}"
    write_filter_key(path, key, demos)
    return key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["lift", "can", "square"])
    args = p.parse_args()
    for t in args.tasks:
        path, sizes = make_base_splits(t)
        print(t, json.dumps(sizes))


if __name__ == "__main__":
    main()
