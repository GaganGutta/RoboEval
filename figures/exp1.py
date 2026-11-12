"""Figures for Exp 1, generated only from results/exp1/all_evals.csv (never edited by hand).

Outputs (PNG + PDF) in figures/out/:
  exp1_learning_curves  success vs training epoch per condition (SELECT seed set: these
                        curves are what we used to choose the training budget)
  exp1_success          final-checkpoint success per condition on the REPORT seed set,
                        mean ± std over training seeds, individual seeds as dots

Usage: python figures/exp1.py [--final_epoch 1000]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "figures", "out")
ORDER = ["all", "better", "random80", "worse"]
COLORS = {"all": "#4C72B0", "better": "#55A868", "random80": "#8C8C8C", "worse": "#C44E52"}
LABELS = {"all": "all (280)", "better": "better only (80)", "random80": "random (80)", "worse": "worse only (100)"}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--final_epoch", type=int, default=1000)
    args = p.parse_args()
    df = pd.read_csv(os.path.join(REPO, "results", "exp1", "all_evals.csv"))
    commit = ",".join(sorted(set(df.git_commit.astype(str).str[:8])))

    sel = df[df.seedset == "select"]
    if len(sel):
        fig, ax = plt.subplots(figsize=(6, 4))
        for (cond, seed), g in sel.groupby(["condition", "train_seed"]):
            g = g.sort_values("epoch")
            ax.plot(g.epoch, g.success_rate, marker="o", ms=3, color=COLORS[cond],
                    ls="-" if seed == 0 else "--", label=f"{LABELS[cond]}, seed {seed}")
        ax.set_xlabel("training epoch (×100 gradient steps)")
        ax.set_ylabel(f"success rate (select seeds, n={int(sel.n.iloc[0])})")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Exp 1 pilot: Can MH, BC-RNN learning curves")
        ax.legend(fontsize=7, loc="lower right")
        ax.text(0.01, 0.01, f"commit {commit}", transform=ax.transAxes, fontsize=6, color="gray")
        save(fig, "exp1_learning_curves")

    rep = df[(df.seedset == "report") & (df.epoch == args.final_epoch)]
    if len(rep):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        conds = [c for c in ORDER if c in set(rep.condition)]
        for i, c in enumerate(conds):
            g = rep[rep.condition == c]
            ax.bar(i, g.success_rate.mean(), yerr=g.success_rate.std() if len(g) > 1 else 0,
                   color=COLORS[c], alpha=0.7, capsize=4)
            ax.scatter([i] * len(g), g.success_rate, color="k", s=10, zorder=3)
        ax.set_xticks(range(len(conds)), [LABELS[c] for c in conds], rotation=15, fontsize=8)
        ax.set_ylabel(f"success (report seeds, n={int(rep.n.iloc[0])}/run)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Exp 1: Can MH, epoch {args.final_epoch}, mean ± std over seeds")
        ax.text(0.01, 0.97, f"commit {commit}", transform=ax.transAxes, fontsize=6, color="gray", va="top")
        save(fig, "exp1_success")
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
