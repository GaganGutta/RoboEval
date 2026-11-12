"""Stage 2 — consistency (v0: kNN action disagreement). No training.

Idea: if similar states map to very different actions across demos, at least one of those
demos is inconsistent with the rest. An episode whose actions consistently disagree with
what other episodes do in similar states gets a low score.

For every timestep t of episode i:
    s_t   = standardized low-dim state (eef pos, eef quat, gripper qpos, object state)
    N_t   = the K nearest timesteps (Euclidean in s) from OTHER episodes
    dis_t = || a_t[:6] - mean_{j in N_t} a_j[:6] || / (mean_{j in N_t} || a_j[:6] - mean a_N ||  + eps)
            i.e. distance from the neighbors' mean action, in units of the neighbors' own
            spread, so naturally multimodal regions (high spread) are not penalized.
episode disagreement = mean_t dis_t ; consistency_score = -disagreement (higher = better).

Arm action dims only (gripper is binary and handled by hygiene). Reference set = the same
pool being scored (self-referential: it measures agreement with the majority behavior).
Caveat: if most demos come from a bad mode, consistency rewards the bad mode.

Hyperparameters: K = 16, eps = 1e-3. Fixed a priori.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

K = 16
EPS = 1e-3
STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


def load_for_consistency(path, demos):
    import h5py
    S, A, ep = [], [], []
    with h5py.File(path, "r") as f:
        for i, d in enumerate(demos):
            g = f[f"data/{d}"]
            s = np.concatenate([g[f"obs/{k}"][:] for k in STATE_KEYS], axis=1)
            S.append(s)
            A.append(g["actions"][:, :6])
            ep.append(np.full(len(s), i))
    return np.concatenate(S).astype(np.float32), np.concatenate(A).astype(np.float32), np.concatenate(ep)


def score(path, demos, k=K):
    S, A, ep = load_for_consistency(path, demos)
    S = (S - S.mean(0)) / (S.std(0) + 1e-8)
    # Query extra neighbors so that after dropping same-episode ones we still have k.
    # Within an episode consecutive steps are near-identical, so many neighbors are
    # same-episode (e.g. while the robot pauses). First pass 8k; timesteps still short
    # get a second, wider query (64k). Any remaining shortfall is reported.
    nn = NearestNeighbors(n_neighbors=8 * k).fit(S)
    _, idx = nn.kneighbors(S)
    dis = np.empty(len(S))
    short = 0
    wide = {}
    need = [t for t in range(len(S)) if (ep[idx[t]] != ep[t]).sum() < k]
    if need:
        _, idx_w = nn.kneighbors(S[need], n_neighbors=min(64 * k, len(S)))
        wide = dict(zip(need, idx_w))
    for t in range(len(S)):
        cand = wide.get(t, idx[t])
        nb = cand[ep[cand] != ep[t]][:k]
        if len(nb) < k:
            short += 1
        an = A[nb]
        mu = an.mean(0)
        spread = np.linalg.norm(an - mu, axis=1).mean()
        dis[t] = np.linalg.norm(A[t] - mu) / (spread + EPS)
    rows = []
    for i, d in enumerate(demos):
        m = dis[ep == i]
        rows.append({"demo": d, "disagreement": float(m.mean()),
                     "consistency_score": float(-m.mean())})
    if short:
        print(f"consistency: {short}/{len(S)} timesteps had < {k} other-episode neighbors")
    return rows
