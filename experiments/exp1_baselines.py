"""Exp 1: does data quality matter on Can MH? (answer key for everything after)

Conditions (all drawn from the 280-demo pool; the 20 held-out "better" demos are never used):
    all       : pool              (280 demos)
    better    : better_pool       (80)   oracle
    worse     : worse_pool        (100)
    random80  : random80_s<seed>  (80)   same size as better; a fresh random draw per seed

Phases (each one command, resumable; finished work is skipped):
    python -m experiments.exp1_baselines --phase train  --runs all:0 better:0 ...
    python -m experiments.exp1_baselines --phase eval   --runs ... --epochs 200 400 --seedset select --n 50
    python -m experiments.exp1_baselines --phase eval   --runs full          # report seeds, report_epochs
    python -m experiments.exp1_baselines --phase report
Convenience: --runs full  ==  every condition x seeds 0,1,2.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import yaml

from experiments.common import PY, REPO, git_commit, run_jobs, write_json

TASK = "can"
CONDITIONS = {"all": "pool", "better": "better_pool", "worse": "worse_pool", "random80": None}
SEEDS = [0, 1, 2]
RUN_ROOT = os.path.join(REPO, "runs", "exp1")
RES_ROOT = os.path.join(REPO, "results", "exp1")
CFG = yaml.safe_load(open(os.path.join(REPO, "configs", "bc_rnn_lowdim.yaml")))
REPORT_EPOCHS = CFG["eval"]["report_epochs"]


def parse_runs(spec):
    if spec == ["full"]:
        return [(c, s) for c in CONDITIONS for s in SEEDS]
    out = []
    for item in spec:
        c, s = item.split(":")
        assert c in CONDITIONS, c
        out.append((c, int(s)))
    return out


def filter_key(cond, seed):
    from data.splits import random_subset
    if cond == "random80":
        return random_subset(TASK, 80, seed)   # writes mask/random80_s<seed>
    return CONDITIONS[cond]


def run_dir(cond, seed):
    return os.path.join(RUN_ROOT, f"{cond}_s{seed}")


def ckpt_for_epoch(cond, seed, epoch):
    paths = glob.glob(os.path.join(run_dir(cond, seed), "run", "*", "models", f"model_epoch_{epoch}.pth"))
    assert len(paths) == 1, (cond, seed, epoch, paths)
    return paths[0]


def phase_train(runs, args):
    from data.mh import working_path
    dataset = working_path(TASK)
    jobs = []
    for cond, seed in runs:
        fk = filter_key(cond, seed)
        argv = [PY, "-m", "train.bc_rnn", "--dataset", dataset, "--filter_key", fk,
                "--seed", str(seed), "--out", run_dir(cond, seed)]
        if args.num_epochs:
            argv += ["--num_epochs", str(args.num_epochs)]
        if args.save_epochs:
            argv += ["--save_epochs"] + [str(e) for e in args.save_epochs]
        jobs.append((f"train_{cond}_s{seed}", argv))
    run_jobs(jobs, args.parallel, os.path.join(REPO, "logs", "exp1"))


def phase_eval(runs, args):
    from eval.rollout import evaluate
    base = CFG["eval"]["report_seed_base"] if args.seedset == "report" else CFG["eval"]["select_seed_base"]
    for cond, seed in runs:
        for ep in args.epochs:
            out = os.path.join(RES_ROOT, "evals", f"{cond}_s{seed}_ep{ep}_{args.seedset}.json")
            if os.path.exists(out):
                continue
            res = evaluate(ckpt_for_epoch(cond, seed, ep), args.n, base, CFG["eval"]["horizon"], args.workers)
            res.update({"task": TASK, "condition": cond, "train_seed": seed, "epoch": ep,
                        "seedset": args.seedset, "git_commit": git_commit()})
            write_json(out, res)
            print(f"{cond} s{seed} ep{ep} [{args.seedset}] success={res['success_rate']:.2f} "
                  f"({res['eval_seconds']:.0f}s)", flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def subset_sizes(df):
    """Demos and timesteps per training subset. Why: subsets with equal demo counts can
    differ a lot in timesteps (episode length tracks operator tier), which matters at a
    fixed gradient-step budget."""
    import h5py
    import pandas as pd
    from data.mh import read_filter_key, working_path
    path = working_path(TASK)
    rows = []
    with h5py.File(path, "r") as f:
        for (cond, seed, key) in df[["condition", "train_seed", "filter_key"]].drop_duplicates().values:
            demos = read_filter_key(path, key)
            T = [int(f["data"][d].attrs["num_samples"]) for d in demos]
            rows.append({"condition": cond, "train_seed": seed, "filter_key": key,
                         "n_demos": len(demos), "timesteps": sum(T), "mean_length": sum(T) / len(T)})
    return pd.DataFrame(rows).sort_values(["condition", "train_seed"])


PAIRS = [("better", "random80"), ("all", "random80"), ("better", "worse"),
         ("random80", "worse"), ("all", "better")]


def pairwise_tests(per_run):
    """Welch t-test on per-run scores (n = training seeds) for each planned pair, with Holm
    correction over the pairs. With 3 seeds per arm these tests have little power; they are
    reported so that "no significant difference" is stated with a number, not a guess."""
    import pandas as pd
    from scipy.stats import ttest_ind
    g = {c: per_run[per_run.condition == c].score.values for c in per_run.condition.unique()}
    rows = []
    for a, b in PAIRS:
        if a in g and b in g and len(g[a]) > 1 and len(g[b]) > 1:
            t = ttest_ind(g[a], g[b], equal_var=False)
            rows.append({"a": a, "b": b, "mean_a": g[a].mean(), "mean_b": g[b].mean(),
                         "diff": g[a].mean() - g[b].mean(), "t": t.statistic, "p": t.pvalue,
                         "n_a": len(g[a]), "n_b": len(g[b])})
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(df)
    # Holm step-down: adjusted p_(k) = max_{j<=k} min(1, (m - j) * p_(j))
    adj, running = [], 0.0
    for j, p in enumerate(df.p):
        running = max(running, min(1.0, (m - j) * p))
        adj.append(running)
    df["holm_p"] = adj
    return df


def phase_report(args):
    """Collect every eval json into one tidy CSV; summary table uses the REPORT seed set only."""
    import pandas as pd
    rows = []
    for p in sorted(glob.glob(os.path.join(RES_ROOT, "evals", "*.json"))):
        r = json.load(open(p))
        k = sum(e["success"] for e in r["episodes"])
        lo, hi = wilson(k, r["n"])
        train_info = json.load(open(os.path.join(run_dir(r["condition"], r["train_seed"]), "train_done.json")))
        rows.append({"task": r["task"], "condition": r["condition"], "train_seed": r["train_seed"],
                     "epoch": r["epoch"], "seedset": r["seedset"], "n": r["n"], "successes": k,
                     "success_rate": r["success_rate"], "wilson_lo": lo, "wilson_hi": hi,
                     "filter_key": train_info["filter_key"], "train_minutes": train_info["train_seconds"] / 60,
                     "git_commit": r.get("git_commit", "unknown")})
    df = pd.DataFrame(rows)
    os.makedirs(RES_ROOT, exist_ok=True)
    df.to_csv(os.path.join(RES_ROOT, "all_evals.csv"), index=False)
    # Per-run score = mean success over the fixed report_epochs checkpoints (METHODS §2).
    # Only runs with every report checkpoint evaluated are scored; nothing is imputed.
    rep = df[(df.seedset == "report") & (df.epoch.isin(REPORT_EPOCHS))]
    if len(rep):
        per_run = (rep.groupby(["condition", "train_seed"])
                   .agg(n_ckpts=("epoch", "nunique"), score=("success_rate", "mean"),
                        successes=("successes", "sum"), episodes=("n", "sum"),
                        git_commit=("git_commit", lambda c: ",".join(sorted(set(c)))))
                   .reset_index())
        per_run = per_run[per_run.n_ckpts == len(REPORT_EPOCHS)]
        per_run.to_csv(os.path.join(RES_ROOT, "per_run.csv"), index=False)
        summ = per_run.groupby("condition").score.agg(["mean", "std", "count"]).reset_index()
        summ["report_epochs"] = " ".join(map(str, REPORT_EPOCHS))
        summ.to_csv(os.path.join(RES_ROOT, "summary.csv"), index=False)
        print(per_run.to_string(index=False))
        print(summ.to_string(index=False))
        subset_sizes(df).to_csv(os.path.join(RES_ROOT, "subset_sizes.csv"), index=False)
        tests = pairwise_tests(per_run)
        tests.to_csv(os.path.join(RES_ROOT, "tests.csv"), index=False)
        print(tests.round(3).to_string(index=False))
    print(f"wrote {len(df)} rows -> results/exp1/all_evals.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["train", "eval", "report"], required=True)
    p.add_argument("--runs", nargs="+", default=["full"])
    p.add_argument("--parallel", type=int, default=6, help="concurrent training jobs (6 measured best)")
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--save_epochs", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, nargs="+", default=None, help="checkpoints to evaluate")
    p.add_argument("--seedset", choices=["select", "report"], default="report")
    p.add_argument("--n", type=int, default=CFG["eval"]["n_rollouts"])
    p.add_argument("--workers", type=int, default=8, help="rollout processes (8 fits in WSL's 15 GB)")
    args = p.parse_args()

    if args.phase == "train":
        phase_train(parse_runs(args.runs), args)
    elif args.phase == "eval":
        args.epochs = args.epochs or REPORT_EPOCHS
        phase_eval(parse_runs(args.runs), args)
    else:
        phase_report(args)


if __name__ == "__main__":
    main()
