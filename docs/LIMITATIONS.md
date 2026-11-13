# Limitations and threats to validity

Running list. Each entry: what, why it matters, what we do about it.

1. **Quality labels are per operator, not per demo.** All 300 MH demos are successful; a
   "worse" operator's individual demo may be fine and vice versa. AUROC against these
   labels measures agreement with operator tier, not demo usefulness. Policy success
   after selection (Exp 3) is the primary metric.

2. **Episode length confound.** Better operators' demos are about half as long as worse
   operators' (Can: 143 vs 304 steps mean). Scores correlated with length will separate
   tiers without measuring anything else. We report length alone as a baseline score.

3. **Operator identity confound.** With only 6 operators, a score can learn an operator's
   style and appear to measure quality. Results are effectively over 6 "clusters", not 300
   independent samples.

4. **Ceiling effects.** If BC-RNN on all data already succeeds near 100% on Lift/Can, no
   subset can be shown to beat "all data" there. Exp 1 measures this.

5. **Simulator version.** Data was collected/converted for robosuite 1.5.1; we use MuJoCo
   3.2.6. Open-loop replay of 30 demos per task succeeds 100% (Lift), 97% (Can), 80%
   (Square) (`results/sanity/replay_*.json`). Open-loop replay is not expected to be exact,
   but the Square figure means the sim may differ slightly from collection. Our numbers
   are not directly comparable to published robomimic numbers (also because of the
   checkpoint rule, see METHODS §1).

6. **Reduced training budget (pending).** The paper trains 2000 epochs; compute limits may
   force fewer. The budget is chosen from learning curves on selection seeds only.

7. **Held-out validation uses labels.** Stage 3 uses 20 "better" demos as a trusted set;
   random selection gets no such information.

8. **Single policy class.** All results are for BC-RNN on low-dim observations. Data
   value can depend on the learner (e.g. diffusion policies, image observations).

10. **Prior negative evidence.** CuraEval (arXiv:2610.38221) found no curated subset that
   significantly beats full data on robomimic MH after Holm correction, and showed results
   can flip with evaluation-protocol changes. Our results must be read against this; we will
   apply Holm correction to the Exp 3 comparisons.

9. **Small number of seeds.** 3 training seeds per condition; with 100 rollouts per run,
   per-run 95% intervals are about ±10 percentage points at 50% success.

11. **Checkpoint-to-checkpoint variance and budget dependence.** In the Exp 1 pilot, success
   of one run changed by up to 0.4 between checkpoints 200 epochs apart, and better-only
   (80 demos) fell from 0.62 to 0.10 between epochs 400 and 800 while larger sets did not.
   A single-checkpoint result is therefore noisy, and the comparison between subsets of
   different sizes depends on the training budget. All subset conditions use the same
   number of gradient steps, which gives smaller subsets more passes over each demo.
