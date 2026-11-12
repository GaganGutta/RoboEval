# Related Work

Notes for the paper's related-work section. BibTeX keys refer to `docs/references.bib`.

**How this was checked (2026-11-12).** Titles, authors, arXiv IDs, dates and arXiv comments were taken from the arXiv API (`export.arxiv.org/api/query`) or arXiv abs pages. Venue details come from the proceedings page named in each entry (PMLR, NeurIPS proceedings, RSS proceedings, or the ICLR/ICML/NeurIPS virtual site). Method and result summaries come from the paper's own HTML or PDF (arXiv HTML or ar5iv). Anything we could not confirm against a primary source is marked `TODO: unverified`. Numbers quoted below are the authors' reported numbers, not reproduced by us.

**Our pipeline, for reference.** (1) hygiene: idle ratio, jerk, length, near-duplicates; (2) consistency: kNN action disagreement; (3) training influence: cosine similarity between per-episode training gradients and the gradient on a trusted validation set (held-out "better"-operator demos), last-layer gradients, random projection, averaged over checkpoints; (4) closed-loop: Demo-SCORE-style success/failure classifier applied to demos; (5) score normalization and combination; (6) greedy subset selection with a diversity term; (7) later, mixture-ratio search over human vs MimicGen data. Testbed: RoboMimic MH low-dim Lift/Can/Square, BC-RNN, simulated rollouts. Claim under test: the selected subset beats both all data and a same-size random subset across tasks and seeds.

---

## 1. Benchmarks and data sources

### RoboMimic — `mandlekar2021robomimic`
Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, Roberto Martín-Martín. "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation." CoRL 2021 (Oral, per arXiv comment); PMLR vol. 164, pp. 1678–1690 (PMLR lists the year as 2022). arXiv:2108.03298.
Verified at: https://proceedings.mlr.press/v164/mandlekar22a.html ; https://arxiv.org/abs/2108.03298 ; dataset details at https://robomimic.github.io/docs/datasets/robomimic_v0.1.html

The paper benchmarks six offline learning methods (BC, BC-RNN, HBC, BCQ, CQL, IRIS) on simulated tasks (Lift, Can, Square, Transport, Tool Hang) and some real-robot versions. It releases Proficient-Human (PH), Multi-Human (MH) and Machine-Generated (MG) datasets. MH has 300 successful demos per task from six operators (50 each), two each labeled "worse", "okay" and "better". Main findings: history-dependent policies (BC-RNN in particular) are the strongest on human data, offline RL methods do poorly on human data, and checkpoint selection matters a lot. Reported success is the maximum over checkpoints evaluated during training (50 rollouts per evaluation, 3 seeds). In the MH quality-subset study, most methods got worse when lower-quality operators were added, but BC-RNN improved as data was added. For low-dim BC-RNN the paper reports: full MH 100% on Lift and Can, 78.0% on Square; Square "better"-only (100 demos) 66.0%, Worse-Better (200) 73.3%.

**Relation to our work.** This is our testbed, data source (MH low-dim Lift/Can/Square with operator labels) and policy class (BC-RNN). Two of its results matter directly for our claim. First, under its protocol BC-RNN on full MH already reaches 100% on Lift and Can, so there is little room to beat "all data" there unless we change the protocol (e.g., final-checkpoint evaluation, fewer epochs, smaller budgets). Second, BC-RNN gained from adding lower-quality operators, and "better"-only training on Square was below full MH. Filtering by operator quality is therefore not known to help BC-RNN on these datasets; our experiments have to show this, not assume it. We should state which evaluation protocol we use (max-over-checkpoints vs final checkpoint), since it changes the comparison.

### robosuite — `zhu2020robosuite`
Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Kevin Lin, Abhiram Maddukuri, Soroush Nasiriany, Yifeng Zhu. "robosuite: A Modular Simulation Framework and Benchmark for Robot Learning." arXiv:2009.12293 (v1 Sep 2020; v3 Jan 2025). No journal reference or venue is listed on arXiv.
Verified at: https://arxiv.org/abs/2009.12293

robosuite is a MuJoCo-based simulation framework with modular components for building manipulation tasks and a set of benchmark environments. The current arXiv version (v3) describes the robosuite v1.5 release. It is a systems paper; it does not study data curation.

**Relation to our work.** The RoboMimic tasks (Lift, Can, Square) and our rollout evaluation run in robosuite. Cite it as the simulator. We should state the robosuite version we use, since the arXiv version now describes v1.5 and the RoboMimic v0.1 datasets were collected with an earlier version.

### MimicGen — `mandlekar2023mimicgen`
Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, Dieter Fox. "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations." CoRL 2023; PMLR vol. 229, pp. 1820–1864. arXiv:2310.17596.
Verified at: https://proceedings.mlr.press/v229/mandlekar23a.html ; https://arxiv.org/abs/2310.17596 ; method details from https://ar5iv.labs.arxiv.org/html/2310.17596

MimicGen generates new demonstrations from a few human demos. It splits each source demo into object-centric subtask segments, re-expresses each segment relative to new object poses, connects segments by interpolation, executes them, and keeps only trajectories that succeed. From about 200 source demos it produces over 50K demos across 18 tasks (typically about 10 source demos per task, with D0/D1/D2 reset-distribution variants). Policies are BC-RNN from RoboMimic. The paper reports that 200 generated demos (from 10 human demos) perform comparably to 200 human demos in most cases, and that scaling generated data from 200 to 1000 demos helps, with diminishing returns beyond. The only filter is task success, and the authors note generated data can carry biases and artifacts. In the parts of the paper we read, we did not find experiments that train on mixtures of human and generated data at varying ratios.

**Relation to our work.** MimicGen is the planned second data source for stage 7 (human vs generated mixture ratio). Its success-only filter means generated data are all "successful" but may differ in style from human data. Our scores (e.g., jerk, kNN disagreement) may score the two sources differently for reasons unrelated to usefulness, so scores should probably be normalized per source before mixing. The MimicGen source demos for Square came from RoboMimic Square PH (single operator), not MH.

---

## 2. Offline (training-free) data-quality scoring

### DemInf — `hejna2025deminf`
Joey Hejna, Suvir Mirchandani, Ashwin Balakrishna, Annie Xie, Ayzaan Wahid, Jonathan Tompson, Pannag R. Sanketi, Dhruv Shah, Coline Manon Devin, Dorsa Sadigh. "Robot Data Curation with Mutual Information Estimators." Robotics: Science and Systems (RSS) 2025, DOI 10.15607/RSS.2025.XXI.023. arXiv:2502.08623.
Verified at: https://roboticsproceedings.org/rss21/p023.html ; https://arxiv.org/abs/2502.08623 ; method details from https://arxiv.org/html/2502.08623v3

DemInf scores each demo by its average contribution to the mutual information between states and actions in the dataset. This rewards demos whose actions are both diverse across the dataset and predictable given the state. It trains two VAEs (states, action chunks) and applies the KSG k-nearest-neighbor MI estimator in the latent space. Per-sample estimates are averaged over each demo's timesteps, and demos that contribute positively are kept. It needs no policy training and no rollouts. It is evaluated on RoboMimic MH Lift, Can and Square, where operator labels (worse/okay/better) serve as ground-truth quality, and on real Franka and RoboCrowd/ALOHA data. Baselines include ensemble uncertainty, demonstration compatibility (Gandhi et al.), VIP, and policy loss. The authors report that DemInf separates high- and low-quality demos according to expert labels, and that filtering improves downstream policies by 5–10% on RoboMimic.

**Relation to our work.** This is the closest prior to our stage 2. Our kNN action disagreement and DemInf's KSG estimator both use nearest neighbours to measure how predictable actions are given states. The differences: we compute disagreement directly in raw low-dim state/action space without VAEs, and we do not include DemInf's action-diversity term. DemInf uses the same MH Lift/Can/Square operator labels we use, so it should be a baseline in our stage-2 comparison. Its downstream RoboMimic policies were BC MLPs, not BC-RNN, so its gains do not directly transfer to our setting.

### Data Quality in Imitation Learning — `belkhale2023dataquality`
Suneel Belkhale, Yuchen Cui, Dorsa Sadigh. "Data Quality in Imitation Learning." NeurIPS 2023 (Advances in Neural Information Processing Systems 36, pp. 80375–80395). arXiv:2306.02437.
Verified at: NeurIPS proceedings BibTeX for paper hash fe692980c5d9732cf153ce27947653a7 ; https://arxiv.org/abs/2306.02437

The paper frames dataset quality for imitation learning in terms of distribution shift at test time. It defines two properties: action divergence (mismatch between expert and learned policy at given states) and transition diversity (system noise for a given state and action). It analyzes their combined effect theoretically and empirically, and finds that more state diversity is not always beneficial. It proposes a framework, not a curation algorithm.

**Relation to our work.** This is the conceptual motivation for our stage-2 consistency score: demos whose actions disagree with similar states' actions add action variance that the policy must average over. We borrow the motivation, not a method.

### Demonstration compatibility — `gandhi2022compatible`
Kanishk Gandhi, Siddharth Karamcheti, Madeline Liao, Dorsa Sadigh. "Eliciting Compatible Demonstrations for Multi-Human Imitation Learning." arXiv:2210.08073; arXiv comment says CoRL 2022. PMLR entry: `TODO: unverified`.
Verified at: arXiv API record for 2210.08073

The paper measures how compatible a new demonstration is with a base policy trained on earlier demos, and uses this measure both to filter incompatible demos after the fact and to guide new users toward compatible demonstrations. It is evaluated on two simulated long-horizon tasks and a real Franka food-plating task. It requires a trained base policy but not rollouts for scoring.

**Relation to our work.** This is an early instance of "conflicting operators hurt multi-human imitation learning", which is the motivation for our stage 2. DemInf uses it as a baseline. It is optional for us.

### Learning to Discern (L2D) — `kuhar2023l2d`
Sachit Kuhar, Shuo Cheng, Shivang Chopra, Matthew Bronars, Danfei Xu. "Learning to Discern: Imitating Heterogeneous Human Demonstrations with Preference and Representation Learning." CoRL 2023; PMLR vol. 229, pp. 1437–1449.
Verified at: https://proceedings.mlr.press/v229/kuhar23a.html

L2D learns a latent representation of short trajectory segments and trains a quality evaluator with preference learning, using a small set of demos with sparse quality labels. The evaluator is meant to generalize to demonstrators with different styles, and it is used to choose which demos to train on. The abstract reports better policies in simulation and on a real robot.

**Relation to our work.** Like our stage 3, L2D assumes a small trusted or labeled subset. It trains a quality classifier on that subset, whereas we use the subset only as a gradient target. This makes it a reasonable baseline for "how much does a small set of better-operator demos buy you".

### SCIZOR — `zhang2025scizor`
Yu Zhang, Yuqi Xie, Huihan Liu, Rutav Shah, Michael Wan, Linxi Fan, Yuke Zhu. "SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning." arXiv:2505.22626 (v2 Sep 2025). The UT Austin RPL lab page lists it as ICRA 2026; proceedings entry `TODO: unverified`.
Verified at: https://arxiv.org/abs/2505.22626

SCIZOR filters at the level of state-action pairs. It does not score whole trajectories. It targets two kinds of low-quality data: suboptimal samples, found with a self-supervised task-progress predictor that flags segments without progress, and redundant samples, found by deduplication on joint state-action representations. It needs no manual labels and no rollouts for scoring. The authors report an average 15.4% improvement across multiple benchmarks, and the paper scales to Open X-Embodiment.

**Relation to our work.** This is the closest prior to our stage 1. Its no-progress filter is related to our idle ratio, and its deduplication is related to our near-duplicate detection. We score whole episodes and do not remove individual samples.

### RINSE (smoothness metrics) — `kulkarni2026rinse`
Soham Kulkarni, Raayan Dhar, Yuchen Cui. "Learning from the Best: Smoothness-Driven Metrics for Data Quality in Imitation Learning." arXiv:2604.23000 (Apr 2026). Also a poster at the ICML 2025 workshop "DataWorld: Unifying data curation frameworks across domains".
Verified at: https://arxiv.org/abs/2604.23000 ; https://icml.cc/virtual/2025/48703

RINSE ranks demos by trajectory smoothness using Spectral Arc Length (SAL) and a contact-aware Trajectory-Envelope Distance (TED). It needs no policy training. The authors argue that smoothness filtering reduces the conditional action variance of the retained data. They report that SAL filtering gives 16% higher success on RoboMimic using one-sixth of the data, and that TED gives a 20% improvement on real-world tasks with half the data. They also use RINSE as a retrieval filter in STRAP and as soft weights in Re-Mix.

**Relation to our work.** RINSE overlaps with our jerk hygiene feature. See Section 6: later audits found that smoothness on RoboMimic MH is largely confounded with episode length.

### PSD metric — `sojib2026psd`
Noushad Sojib, Sajay Arthanat, Momotaz Begum. "An Efficient Metric for Data Quality Measurement in Imitation Learning." arXiv:2605.01544 (v2 Sep 2026).
Verified at: https://arxiv.org/abs/2605.01544

The paper ranks demos by the power spectral density of the trajectory: lower PSD means smoother and higher quality. It needs no policy, rollouts or labels. It is evaluated on two benchmark IL datasets with expert and lay-user demos, and in a user study that fine-tunes π0.5. It reports higher success than uncurated data and two data-ranking baselines.

**Relation to our work.** This is another training-free smoothness signal in the same family as our jerk feature, and it is a cheap baseline.

### Diversity-aware selection: FAKTUAL and Geometric Entropy — `sirigiri2026faktual`, `luo2026geometric`
Sreevardhan Sirigiri, Nathan Samuel de Lara, Christopher Agia, Florian Shkurti, Fabio Ramos. "Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for Robotics Datasets." arXiv:2603.11634 (v2 Oct 2026).
Qian Luo, Ruizhe Liu, Pei Zhou, Xunzhe Zhou, Yanchao Yang. "Geometric Entropy: When Trajectory Diversity Helps and Hurts in Imitation Learning." arXiv:2606.20871; arXiv comment says accepted to IROS 2026 (proceedings entry `TODO: unverified`).
Verified at: https://arxiv.org/abs/2603.11634 ; https://arxiv.org/abs/2606.20871

FAKTUAL defines dataset entropy from a signature-kernel Gram matrix over demos and selects a size-budgeted subset that maximizes this entropy. It is model-free, with no policy and no rollouts. It reports gains over random selection on RoboMimic, MetaWorld and four real tasks. Geometric Entropy measures the diversity of trajectory shapes and reports an inverted-U relationship between diversity and success: diversity helps at first, then hurts once it creates strategy ambiguity.

**Relation to our work.** These are direct priors for our stage 6 diversity term. Geometric Entropy's inverted-U result, and CuraEval's finding that a diversity selector did worse than random on Can at 25% retention (Section 6), mean the weight on the diversity term should be tuned and reported, not assumed helpful.

---

## 3. Influence and gradient-based selection

### Influence functions — `koh2017influence`
Pang Wei Koh, Percy Liang. "Understanding Black-box Predictions via Influence Functions." ICML 2017; PMLR vol. 70, pp. 1885–1894. arXiv:1703.04730.
Verified at: https://proceedings.mlr.press/v70/koh17a.html ; https://arxiv.org/abs/1703.04730

The paper uses influence functions from robust statistics to estimate how upweighting or removing a training point changes the loss on a test point. It uses inverse-Hessian-vector products, computed with only gradient and Hessian-vector-product access. It shows that the approximation remains informative for non-convex models and applies it to model understanding, debugging, detecting dataset errors, and training-set attacks.

**Relation to our work.** This is the theoretical basis for influence-based curation (CUPID builds on it). Our stage 3 drops the inverse Hessian and uses a first-order, TracIn/LESS-style gradient similarity instead. We should say so explicitly and not call our score an "influence function".

### TracIn — `pruthi2020tracin`
Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan. "Estimating Training Data Influence by Tracing Gradient Descent." NeurIPS 2020 (Advances in Neural Information Processing Systems 33, pp. 19920–19930). arXiv:2002.08484. Note: the author order differs between sources. The NeurIPS proceedings list Kale before Sundararajan, and arXiv lists Sundararajan before Kale. The bib file uses the proceedings order.
Verified at: NeurIPS proceedings BibTeX for hash e6385d39ec9394f2f3a354d9d2b88eec ; https://arxiv.org/abs/2002.08484 ; method details from https://ar5iv.labs.arxiv.org/html/2002.08484

TracIn estimates the influence of a training example on a test example by tracing how the test loss changes when that example is used during training. The practical version, TracInCP, sums over saved checkpoints the learning-rate-weighted dot product of the training-example gradient and the test-example gradient. To keep it cheap, the paper restricts gradients to selected layers (e.g., only the last layer) and describes random projection of gradients, whose dot products are unbiased estimates of the originals.

**Relation to our work.** Our stage 3 is essentially TracInCP with a validation set as the "test" set. It borrows the checkpoint averaging, last-layer gradients and random projection. It differs in using cosine similarity rather than the raw dot product (following LESS) and in aggregating per-sample gradients to episodes.

### LESS — `xia2024less`
Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen. "LESS: Selecting Influential Data for Targeted Instruction Tuning." ICML 2024; PMLR vol. 235, pp. 54104–54132. arXiv:2402.04333.
Verified at: https://proceedings.mlr.press/v235/xia24c.html ; https://arxiv.org/abs/2402.04333 ; method details from https://arxiv.org/html/2402.04333v3

LESS selects instruction-tuning data for a target capability defined by a few examples. It warm-up trains with LoRA on a 5% random subset and saves a checkpoint each epoch. It then computes Adam-adjusted LoRA gradients for training examples and plain gradients for target examples, projects both to 8192 dimensions with a random Rademacher matrix, and scores each candidate by the cosine similarity to the averaged target gradient. Scores are summed over checkpoints, weighted by learning rate. The authors use cosine rather than inner product because gradient norms correlate with sequence length, which biases the inner product toward short sequences. Training on the top 5% often beats training on the full dataset, and the selected data transfer across model sizes and families.

**Relation to our work.** Our stage 3 follows the LESS recipe closely: a small target set, random projection, cosine similarity, and multiple checkpoints. The differences: robot BC-RNN rather than LLMs, last-layer gradients rather than LoRA, (as far as we know) no Adam adjustment, and episode-level aggregation. LESS's length-bias argument has an analogue for us. Gradient magnitude may scale with episode length or action magnitude, and RoboMimic MH quality tiers are correlated with episode length (Section 6). We should check whether the stage-3 score is just a proxy for length.

### CUPID — `agia2025cupid`
Christopher Agia, Rohan Sinha, Jingyun Yang, Rika Antonova, Marco Pavone, Haruki Nishimura, Masha Itkina, Jeannette Bohg. "CUPID: Curating Data your Robot Loves with Influence Functions." CoRL 2025; PMLR vol. 305, pp. 2907–2932. arXiv:2506.19121.
Verified at: https://proceedings.mlr.press/v305/agia25a.html ; https://arxiv.org/abs/2506.19121 ; method details from https://arxiv.org/html/2506.19121v2

CUPID estimates each training demo's influence on the policy's expected return. It computes action-level influence (gradient of the action log-likelihood, combined with an inverse Hessian of the BC loss) between training demos and states in evaluation rollouts, and weights each rollout's contribution by its return in a REINFORCE-style estimator. TRAK (random projection, Gauss-Newton Hessian) makes this tractable. It therefore needs a trained policy and rollouts. It supports filtering harmful demos and selecting new ones. Simulation experiments use RoboMimic MH Lift, Square and Transport with the official diffusion policy; hardware experiments use a Franka arm, plus a π0 fine-tuning experiment. Baselines are DemInf, Demo-SCORE, a success-similarity heuristic, random, and an oracle. A variant, CUPID-Quality, averages the performance influence with a reward-agnostic quality score derived from the influence scores. The authors report that it helps on mixed-quality data but can hurt elsewhere. Headline: training on under 33% of curated data gives state-of-the-art diffusion policies on RoboMimic.

**Relation to our work.** CUPID is the closest prior to the combination of our stages 3 and 4. It ties gradient-based attribution to closed-loop return, where we tie it to a trusted validation-set loss in stage 3 and treat rollouts separately in stage 4. It also already combines two scores (CUPID-Quality). Differences: we use first-order gradient similarity without a Hessian, BC-RNN rather than diffusion policies, Lift/Can/Square (CUPID uses Lift/Square/Transport), and multiple signals plus a diversity term. CUPID should be a primary baseline. If our influence stage does not beat it, we should say so.

### DataMIL — `dass2026datamil`
Shivin Dass, Alaa Khaddaj, Logan Engstrom, Aleksander Madry, Andrew Ilyas, Roberto Martín-Martín. "DataMIL: Selecting Data for Robot Imitation Learning with Datamodels." ICLR 2026 (poster). arXiv:2505.09603 (v2 Jun 2026). OpenReview forum: AcTsKglDdh (the page itself could not be loaded; venue confirmed from the ICLR virtual site).
Verified at: https://iclr.cc/virtual/2026/poster/10011019 ; https://arxiv.org/abs/2505.09603 ; method details from https://arxiv.org/html/2505.09603v2

DataMIL applies the datamodels framework to robot data selection. It estimates how including each prior-data sample (grouped into short sub-trajectory clusters) changes a target metric, either by regression over many policies trained on random subsets or with metagradients. To avoid rollouts during selection, it uses the BC loss on held-out target-task demos as a surrogate objective. The setting is selecting from a large multi-task prior dataset and co-training with target data, not filtering a single-task dataset. It is evaluated on MetaWorld, LIBERO (LIBERO-90 → LIBERO-10) and real-world OXE tasks with Octo. The authors report gains over retrieval baselines (Behavior Retrieval, FlowRetrieval, STRAP), and they note that estimation costs several times more than training one policy on all data.

**Relation to our work.** DataMIL also uses held-out target-demo loss as a rollout-free proxy for usefulness, which is what our stage 3 does. Their estimator is datamodels or metagradients over a multi-task prior; ours is a first-order gradient similarity within a single-task dataset. The problem setting differs (co-training selection vs filtering), so it is a conceptual prior more than a direct baseline.

### RoboDrop — `xu2026robodrop`
Runze Xu, Yuanfan Xu, Cuijie Xu, Shuang Dai, Yining Li, Yu Wang, Jincheng Yu. "RoboDrop: Curating VLA Post-Training Data via Local Gradient Compatibility." arXiv:2609.10021 (v1, 9 Sep 2026). No venue is listed on arXiv.
Verified at: https://arxiv.org/abs/2609.10021 ; method details from https://arxiv.org/html/2609.10021v1

The paper exists under this name; it is a September 2026 arXiv preprint. RoboDrop curates post-training data for vision-language-action models. During a one-epoch warm-up run, it scores each training sample by the cosine similarity between its gradient (action-expert parameters only, compressed to 4096 dimensions with CountSketch) and a reference gradient. The reference is built from samples in a small trusted clean validation set that share the task instruction and are visually similar (DINO features), weighted with a softmax. Sample scores are averaged per episode. Episodes are then removed either by a fixed ratio or with a two-component Gaussian-mixture rule. Scoring needs no environment interaction. Evaluation covers LIBERO with injected temporal or action corruptions, RoboMimic-MH (Can, Lift, Square, Transport, with proficient operators treated as reliable), and four real ARX-arm tasks, mainly with π0.5 (also GR00T and X-VLA). Baselines are episode length, Behavior Retrieval, DataMIL, SCIZOR and QoQ. On RoboMimic-MH it reports mean AUROC 88.33% for identifying unreliable demos. On real robots, average success rose from 35.0% unfiltered to 63.75% (20% removal) and 67.5% (automatic rule). We did not confirm whether the paper reports downstream policy success on RoboMimic-MH or only detection metrics: `TODO: unverified`.

**Relation to our work.** RoboDrop is the closest published prior to our stage 3. Both compute cosine similarity between training gradients and gradients on a trusted clean validation set, use projected gradients, aggregate to episodes, and evaluate on RoboMimic MH with operator quality as ground truth. Differences: RoboDrop uses a per-sample, context-matched local reference, where we use one global validation gradient. It scores once along a single warm-up epoch, where we average over checkpoints. It targets large VLA models, where we use a small BC-RNN with last-layer gradients. And our validation set is held-out "better"-operator demos rather than a separately collected clean set. Stage 3 on its own should not be presented as novel. RoboDrop should be cited as the direct precedent, and ideally reproduced as a baseline.

---

## 4. Rollout-based (closed-loop) curation

### Demo-SCORE — `chen2025demoscore`
Annie S. Chen, Alec M. Lessing, Yuejiang Liu, Chelsea Finn. "Curating Demonstrations using Online Experience." Robotics: Science and Systems (RSS) 2025, DOI 10.15607/RSS.2025.XXI.071. arXiv:2503.03707 (v2 Jul 2025).
Verified at: https://roboticsproceedings.org/rss21/p071.html ; https://arxiv.org/abs/2503.03707 ; method details from https://arxiv.org/html/2503.03707v2

Demo-SCORE trains a policy on all demos, collects rollouts at several checkpoints, and trains a classifier on individual states (proprioceptive) to predict whether they come from a successful or failed rollout. Cross-validation across checkpoints picks the classifier: classifiers are trained on rollouts from earlier checkpoints and validated on the last checkpoint's rollouts. Each demo is scored by the mean classifier output over its states and kept if above a threshold set from the classifier's training rollouts. The final policy is retrained on the kept demos, plus successful rollouts. It requires policy training and rollouts, but only binary success labels. Experiments use a simulated square-peg task (diffusion policy; a proficient-human dataset plus scripted strategies), a simulated bimanual peg-insertion task (ACT), and real ALOHA tasks. Baselines include training-loss weighting, Auto-IL and return-conditioned policies. Reported gains: 15–35% absolute over the base policy in simulation, and about 30% on average on real tasks. The paper does not evaluate on RoboMimic MH operator-quality data.

**Relation to our work.** Our stage 4 is a reimplementation of Demo-SCORE. We borrow the method and should credit it as such. The new parts are applying it to RoboMimic MH with BC-RNN and combining its score with the other stages. Demo-SCORE alone should be a baseline. One caveat: with BC-RNN on MH Lift/Can near 100% success, there may be too few failed rollouts to train the classifier.

---

## 5. Data mixtures

### Re-Mix — `hejna2025remix`
Joey Hejna, Chethan Anand Bhateja, Yichen Jiang, Karl Pertsch, Dorsa Sadigh. "ReMix: Optimizing Data Mixtures for Large Scale Imitation Learning." CoRL 2024; PMLR vol. 270, pp. 145–164 (PMLR lists the year as 2025). arXiv:2408.14037 (titled "Re-Mix" on arXiv).
Verified at: https://proceedings.mlr.press/v270/hejna25a.html ; https://arxiv.org/abs/2408.14037 ; details from https://arxiv.org/html/2408.14037v1

Re-Mix learns domain weights for robot pre-training mixtures with group distributionally robust optimization over domains. Each domain's excess BC loss is measured relative to a reference model trained on the uniform mixture and early-stopped, and actions are normalized per domain and discretized. It uses only offline BC losses, with no rollouts. On Open X-Embodiment (RT-X mix) and Bridge V2, the learned weights beat uniform weights by 38% and human-chosen weights by 32% on average on real WidowX and Franka robots. In an appendix experiment on RoboMimic Square MH (images), the six operators are treated as domains. Re-Mix upweights "better" operators and downweights "worse" ones, and at 50% and 25% subsetting it beats uniform weights (reported 77 vs 53 and 59 vs 39 successes out of 100).

**Relation to our work.** Re-Mix is the main prior for stage 7 and also an existing result on our exact data. Its MH appendix shows that operator-level reweighting from BC losses alone can recover the quality ordering on Square MH, so an operator-level (rather than episode-level) mixture baseline belongs in our comparisons. For human vs MimicGen mixing, Re-Mix's DRO weighting is an alternative to grid search over ratios.

### DoReMi (optional) — `xie2023doremi`
Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu. "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining." NeurIPS 2023 (Advances in Neural Information Processing Systems 36, pp. 69798–69818). arXiv:2305.10429.
Verified at: NeurIPS proceedings BibTeX for hash dcba6be91359358c2355cd920da3fcbd ; https://arxiv.org/abs/2305.10429

DoReMi trains a small proxy model with Group DRO over pretraining domains to obtain domain weights, then trains a larger model on data resampled with those weights. On The Pile, weights from a 280M proxy improve an 8B model's average few-shot accuracy by 6.5 points and reach baseline accuracy with 2.6x fewer steps.

**Relation to our work.** This is the language-model origin of the DRO mixture method that Re-Mix adapts. Cite it only if we discuss mixture search methodology.

### DataComp (optional) — `gadre2023datacomp`
Samir Yitzhak Gadre et al. (34 authors; full list in bib). "DataComp: In search of the next generation of multimodal datasets." NeurIPS 2023 Datasets and Benchmarks Track (Advances in Neural Information Processing Systems 36, pp. 27092–27112). arXiv:2304.14108.
Verified at: NeurIPS proceedings BibTeX for hash 56332d41d55ad7ad8024aac625881be7 ; https://arxiv.org/abs/2304.14108

DataComp is a benchmark where participants propose filtering or data sources from a fixed 12.8B image-text candidate pool. Training code and compute are fixed (CLIP), and the resulting data are evaluated on 38 downstream tasks. The best baseline subset (DataComp-1B) beats OpenAI's CLIP ViT-L/14 on ImageNet zero-shot by 3.7 points with the same training recipe.

**Relation to our work.** We share its evaluation principle: hold the learner and compute fixed and compare only the data. Our "same training budget" choice (fixed steps vs fixed epochs for different subset sizes) should be stated explicitly, as DataComp and CuraEval do.

---

## 6. Audits and evaluation protocols for curation

These are recent (2026), mostly workshop or preprint papers. They test the same kind of claim we make on the same data, so they matter more than their venue suggests.

### CuraEval — `xu2026curaeval`
Jiuyi Xu, Meida Chen, Yang Ye, Yangming Shi. "CuraEval: Evaluating Demonstration Curation on Mixed-Skill Robot Data." arXiv:2610.38221 (v1, 28 Oct 2026); arXiv comment: accepted by the CoRL 2026 Workshop "Oops, I Erred".
Verified at: https://arxiv.org/abs/2610.38221 ; code at https://github.com/jiuyixu25/CuraEval

CuraEval compares demonstration scores on RoboMimic MH low-dim (Can, Square, Transport; 300 demos, better/okay/worse operators) with Diffusion Policy and BC-RNN. It keeps 25/50/75% of demos and compares against size-matched random subsets and full-data training, at a fixed step budget, with Welch tests and Holm correction over training seeds. The scores tested include PSD, SAL, signature-kernel entropy (FAKTUAL), geometric entropy, length, path inefficiency, jerk, and a CUPID port. Findings: one score was effectively a function of episode length. Keeping the shortest half raised Diffusion Policy success on Square over random selection (76.5% vs 70.7%). A diversity selector was much worse than random on Can at 25%. After changing the evaluation protocol (checkpoint reporting, rollout count, evaluation seed), length-based selection no longer beat full-data training. After Holm correction, no curated condition significantly beat full-data training.

**Relation to our work.** This is the closest prior to our core claim and evaluation protocol: same datasets (MH low-dim), same policy class (BC-RNN), same two baselines (random same-size, full data). Its result is negative for full-data comparisons. Our paper has to (a) include episode length as a baseline, (b) use the same multiple-comparison discipline, (c) report sensitivity to the evaluation protocol, and (d) cite this as the existing benchmark of single-score curation on MH. It does not test gradient-similarity influence against a validation set, Demo-SCORE, kNN consistency, or a combined score, so those remain open.

### Auditing offline demonstration pruning — `yang2026auditing`
Joy Z. Yang, Socrates Osorio. "Auditing Offline Demonstration Pruning for Online Robot Deployment." Poster, ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation: Black-Box Optimization to Reinforcement Learning. No arXiv or OpenReview link found. Only the virtual-site abstract was read; PDF `TODO: unverified`.
Verified at: https://icml.cc/virtual/2026/83657

The paper proposes a four-step audit for pruning claims: a paired comparison against random pruning, a comparison against a cheap baseline, a held-out stress test, and a mechanism check. The case study is TRAK-Traj (trajectory-level TRAK attribution) on RoboMimic Can MH. TRAK-Traj beats matched random pruning by 4.7 points over 10 paired seeds (300 rollouts per condition; one-sided p = 0.019), but a TracIn-style scorer ties it, and trajectory-length pruning does better. On a mixed-quality held-out split, length pruning removed 260 of 270 pruned trajectories from the worse tier.

**Relation to our work.** This directly tests a TracIn-style gradient score against random pruning on RoboMimic Can MH, which overlaps with our stage 3, and finds that episode length explains much of the effect. We should report a length-only baseline and check how much of each of our scores is explained by length.

### Smoothness audit — `bassi2026smoothness`
Jaiveer Bassi. "Curation Metrics Are a Post-Training Decision: Auditing Trajectory Smoothness for Adapting Robot Foundation Models." Poster, NeurIPS 2026 Workshop RoboPAD (Post-Training Adaptation of Robot Foundation Models). OpenReview forum wj9Kj3hsJs (not loaded); only the virtual-site abstract was read, PDF `TODO: unverified`.
Verified at: https://neurips.cc/virtual/2026/162882

On RoboMimic can/mh, can/mg and square/mh, the paper finds that SAL separates better from worse operators (AUROC 0.952 on can/mh), but episode length alone does as well (0.957). Within duration-matched bands, SAL drops toward chance, while raw mean jerk stays informative (e.g., 0.896). Downstream, with GMM BC policies over 10 seeds against random and no curation, TED curation was significantly worse than random, and SAL and jerk were not distinguishable from no curation.

**Relation to our work.** This audits our jerk hygiene feature on our data. Jerk may carry operator-skill signal beyond length, but on its own it did not improve downstream BC on these tasks. It supports reporting duration-matched analyses for our hygiene scores.

### Curation-metric audit with injected defects — `bedi2026auditing`
Aarav Bedi. "Auditing Demonstration Curation Metrics: Action-Only Scorers Fail on the Structural Defects That Degrade Imitation Policies." arXiv:2606.05588 (v1, Jun 2026).
Verified at: https://arxiv.org/abs/2606.05588

The paper injects defects of known type into demonstrations and audits seven curation metrics for detection and for downstream BC success. Subtle action perturbations (correlated noise, tremor, truncation) are caught by multivariate outlier scoring, and removing them recovers the downstream gap. Structural errors (a wrong action at a key moment) are missed by every action-only metric, and two metrics rank them as higher quality. State-trajectory metrics detect them but recover only about a third of the gap. High detection accuracy did not guarantee downstream gains.

**Relation to our work.** Our stage 1 features (idle ratio, jerk) and stage 2 (kNN action disagreement in state-action space) are mostly action-based and may miss structural errors. The stage 4 closed-loop classifier is our only state-based, outcome-aware signal. The paper also cautions against reporting detection AUROC against operator labels as evidence of downstream benefit.

---

## 7. Gaps and open questions (including close priors to our exact pipeline)

**Close priors found.** No paper we found combines all seven of our stages. Several papers cover individual stages, and some cover our evaluation claim on the same data:

1. **Stage 3 has a direct precedent: RoboDrop (arXiv:2609.10021, Sep 2026).** It computes cosine similarity between projected training gradients and gradients on a trusted clean validation set, aggregates to episodes, and evaluates on RoboMimic MH (Can/Lift/Square/Transport) with proficient operators as ground truth. Our stage 3 differs in details (global vs context-matched reference, checkpoint averaging vs one warm-up pass, BC-RNN last layer vs VLA action expert, "better"-operator validation set). We should not claim gradient-to-validation-set similarity for demo curation as new. The method also follows from TracIn plus LESS.
2. **Stages 3+4 combined (attribution tied to rollouts) and score combination: CUPID (CoRL 2025)**, evaluated on RoboMimic MH Lift/Square/Transport. CUPID-Quality already averages two scores.
3. **Stage 4 is Demo-SCORE (RSS 2025)**, used as-is.
4. **Stage 2 is close to DemInf (RSS 2025)**: kNN-based state-action predictability on RoboMimic MH Lift/Can/Square with the same operator labels.
5. **Stage 1 overlaps SCIZOR (deduplication, no-progress samples), RINSE/PSD (smoothness)**, and the smoothness audits.
6. **Stage 6 overlaps FAKTUAL (greedy entropy-maximizing selection)**, which CuraEval reports can hurt on Can at 25% retention.
7. **Stage 7 overlaps Re-Mix**, which already reweights the six MH operators on Square and beats uniform weighting under subsetting.
8. **Our core claim (subset beats full data and same-size random on MH low-dim with BC-RNN) has already been tested for single scores by CuraEval (Oct 2026)**, with a negative result against full data after multiple-comparison correction. A related ICML 2026 workshop audit on Can MH found that a TracIn/TRAK-style score's advantage over random is matched or exceeded by episode length.

**What appears not to have been done (based on our search; not exhaustive).** (a) Combining hygiene, kNN consistency, validation-gradient similarity and a rollout classifier into one normalized score with diversity-aware greedy selection, evaluated on MH with BC-RNN. (b) A head-to-head comparison of Demo-SCORE, CUPID, DemInf, RoboDrop-style gradient similarity and length under one protocol on MH Lift/Can/Square with BC-RNN. (c) Human-vs-MimicGen mixture-ratio search combined with episode-level curation. We searched for (c) and found no paper that sweeps human/MimicGen ratios; MimicGen itself compares pure datasets.

**Risks to our core claim from prior results.**
- **Ceiling effects.** RoboMimic reports 100% success for BC-RNN (low-dim) on full MH Lift and Can, using max-over-checkpoints evaluation. Beating "all data" there may be impossible under that protocol. Square is the only task among ours with headroom in their tables.
- **BC-RNN uses lower-quality data well.** RoboMimic found that BC-RNN improved when worse/okay operators were added, and "better"-only Square (66.0%) was below full MH (78.0%). Removing data may hurt BC-RNN more than the diffusion policies or MLPs used in DemInf/CUPID.
- **Length confound.** Three independent audits (CuraEval, Yang & Osorio, Bassi) find that episode length explains much of the operator-quality signal on MH, and that length is a strong baseline. Our hygiene features include length directly, and gradient-based scores may correlate with it. We need a length-only baseline and should report score–length correlations.
- **Evaluation-protocol sensitivity.** CuraEval found that changing checkpoint reporting, rollout count and evaluation seed changed its conclusion. We should fix and report the protocol in advance (final vs best checkpoint, number of rollouts, seeds, step budget across subset sizes), and correct for multiple comparisons.
- **Validation-set leakage.** Our trusted validation set is held-out "better"-operator demos, and the evaluation ground truth is also operator quality. Any score that pulls the subset toward "better" operators is partly rewarded by construction. We should report results against rollout success, not only agreement with operator labels, and consider a validation set that is not defined by the same label.
- **Rollout classifier data.** With high base success on Lift/Can, Demo-SCORE-style classifiers may have very few failures to learn from.

**Unverified items to resolve before submission.**
- Gandhi et al. 2022: PMLR (CoRL 2022) volume and pages not checked.
- SCIZOR: ICRA 2026 venue seen only on a lab page.
- Geometric Entropy: IROS 2026 venue taken from the arXiv comment only.
- DataMIL: the OpenReview page could not be loaded; venue confirmed from the ICLR 2026 virtual site.
- Yang & Osorio (ICML 2026 workshop) and Bassi (NeurIPS 2026 workshop): only virtual-site abstracts were read, no PDFs.
- RoboDrop: whether downstream RoboMimic-MH policy success (not only detection AUROC) is reported.
- MimicGen: we read roughly the first half of the paper's HTML. A human/generated mixture experiment in the remaining appendices cannot be ruled out.
