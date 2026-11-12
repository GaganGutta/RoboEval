"""Exp 2: do Stage 1 (hygiene) and Stage 2 (consistency) scores rank "better" operators'
demos above "worse" operators' demos?

For each task, on the 280-demo pool:
  - compute every score (higher = predicted better quality)
  - AUROC(better vs worse), with a 95% bootstrap CI (2000 resamples of demos)
  - also AUROC with an operator-level bootstrap (resample the 4 operators' demo sets),
    because demos from one operator are not independent (docs/LIMITATIONS.md #3)
  - Spearman correlation with tier (worse=0, okay=1, better=2)

Scores compared:
  neg_length        -episode length                (trivial baseline: longer = worse)
  hygiene_score     Stage 1 (excludes length by design)
  -idle_ratio, -jerk_mean, -action_jerk, -gripper_flips   (Stage 1 components)
  consistency_score Stage 2
  random            uniform noise (sanity: should be ~0.5)

Usage: python -m experiments.exp2_ranking --tasks lift can square
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from data.mh import read_filter_key, working_path
from experiments.common import REPO, git_commit
from scoring import consistency, hygiene

RES = os.path.join(REPO, "results", "exp2")
N_BOOT = 2000
TIER = {"worse": 0, "okay": 1, "better": 2}


def task_scores(task):
    path = working_path(task)
    pool = read_filter_key(path, "pool")
    pool_set = set(pool)
    tier = {}
    for q in TIER:
        for d in read_filter_key(path, q):
            if d in pool_set:
                tier[d] = q
    op = {}
    for q in TIER:
        for k in (1, 2):
            for d in read_filter_key(path, f"{q}_operator_{k}"):
                if d in pool_set:
                    op[d] = f"{q}_{k}"
    h = pd.DataFrame(hygiene.score(hygiene.load_for_hygiene(path, pool)))
    c = pd.DataFrame(consistency.score(path, pool))
    df = h.merge(c, on="demo")
    df["task"], df["tier"], df["operator"] = task, df.demo.map(tier), df.demo.map(op)
    df["neg_length"] = -df["length"]
    for col in ["idle_ratio", "jerk_mean", "action_jerk", "gripper_flips"]:
        df[f"neg_{col}"] = -df[col]
    df["random"] = np.random.default_rng(0).random(len(df))
    return df


SCORES = ["neg_length", "hygiene_score", "neg_idle_ratio", "neg_jerk_mean", "neg_action_jerk",
          "neg_gripper_flips", "consistency_score", "random"]


def auroc_table(df):
    rng = np.random.default_rng(0)
    bw = df[df.tier.isin(["better", "worse"])].reset_index(drop=True)
    y = (bw.tier == "better").astype(int).values
    ops = sorted(bw.operator.unique())
    rows = []
    for s in SCORES:
        x = bw[s].values
        auc = roc_auc_score(y, x)
        boots = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(x), len(x))
            if y[i].min() != y[i].max():
                boots.append(roc_auc_score(y[i], x[i]))
        op_boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(ops, len(ops), replace=True)
            i = np.concatenate([np.where(bw.operator.values == o)[0] for o in pick])
            if y[i].min() != y[i].max():
                op_boots.append(roc_auc_score(y[i], x[i]))
        rho = spearmanr(df[s], df.tier.map(TIER)).correlation
        rows.append({"task": df.task.iloc[0], "score": s, "auroc": auc,
                     "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                     "op_ci_lo": np.percentile(op_boots, 2.5), "op_ci_hi": np.percentile(op_boots, 97.5),
                     "spearman_tier": rho, "n_better": int(y.sum()), "n_worse": int((1 - y).sum())})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["lift", "can", "square"])
    args = p.parse_args()
    os.makedirs(RES, exist_ok=True)
    commit = git_commit()
    tables = []
    for t in args.tasks:
        df = task_scores(t)
        df["git_commit"] = commit
        df.to_csv(os.path.join(RES, f"{t}_scores.csv"), index=False)
        tab = auroc_table(df)
        tab["git_commit"] = commit
        tables.append(tab)
        print(tab[["task", "score", "auroc", "ci_lo", "ci_hi", "op_ci_lo", "op_ci_hi", "spearman_tier"]]
              .round(3).to_string(index=False), flush=True)
    pd.concat(tables).to_csv(os.path.join(RES, "auroc.csv"), index=False)


if __name__ == "__main__":
    main()
