"""Stage 1 — hygiene. Per-episode statistics that need no training.

Features (per episode):
    length          number of control steps (20 Hz)
    idle_ratio      fraction of steps where the robot is essentially not moving:
                    max |arm action| < IDLE_ACT  OR  ||joint velocity|| < IDLE_QVEL
    jerk_mean       mean ||d^3 eef_pos / dt^3|| (m/s^3), finite differences
    jerk_p95        95th percentile of the same
    action_jerk     mean ||second difference of arm actions|| (smoothness of commands)
    gripper_flips   number of gripper open/close switches (chatter)
    dup_dist        distance to the nearest OTHER episode in a resampled trajectory space

Score: hygiene_score = -mean(robust z of idle_ratio, jerk_mean, action_jerk, gripper_flips).
Higher = cleaner. LENGTH IS DELIBERATELY NOT IN THE SCORE: on MH data length alone tracks
operator tier, so it is reported as its own baseline score instead of being hidden inside
this one (see docs/LIMITATIONS.md #2).

Flags (boolean): idle > FLAG_IDLE, jerk_mean or action_jerk beyond the robust-z cutoff,
near-duplicate (dup_dist below DUP_Z robust-z under the median).

Thresholds were fixed a priori as round numbers and are not tuned on labels.
"""
import numpy as np

DT = 1.0 / 20.0          # robosuite control frequency in these datasets
IDLE_ACT = 0.05          # actions are in [-1, 1]; 5% of full scale = "near zero"
IDLE_QVEL = 0.05         # rad/s, joint-velocity norm
FLAG_IDLE = 0.2          # flag episodes idle >20% of the time
FLAG_Z = 3.0             # robust-z cutoff for outlier flags
DUP_Z = -3.0
RESAMPLE = 50            # points per trajectory for duplicate detection


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-12
    return (x - med) / mad


def episode_features(actions, eef_pos, joint_vel):
    T = len(actions)
    arm = actions[:, :6]
    idle = (np.abs(arm).max(1) < IDLE_ACT) | (np.linalg.norm(joint_vel, axis=1) < IDLE_QVEL)
    jerk = np.linalg.norm(np.diff(eef_pos, n=3, axis=0), axis=1) / DT**3 if T > 3 else np.zeros(1)
    a_jerk = np.linalg.norm(np.diff(arm, n=2, axis=0), axis=1) if T > 2 else np.zeros(1)
    grip = np.sign(actions[:, 6])
    return {
        "length": T,
        "idle_ratio": float(idle.mean()),
        "jerk_mean": float(jerk.mean()),
        "jerk_p95": float(np.percentile(jerk, 95)),
        "action_jerk": float(a_jerk.mean()),
        "gripper_flips": int((np.diff(grip) != 0).sum()),
    }


def _resample(x, n=RESAMPLE):
    """Linear resample a (T, d) trajectory to (n, d) so episodes of any length compare."""
    t_old = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n)
    return np.stack([np.interp(t_new, t_old, x[:, j]) for j in range(x.shape[1])], axis=1)


def duplicate_distance(trajs):
    """For each episode, distance to its nearest other episode after resampling
    (eef position + object state, standardized). Small = possible near-duplicate."""
    X = np.stack([_resample(t).ravel() for t in trajs])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min(1)) / np.sqrt(X.shape[1])


def score(episodes):
    """episodes: {name: {'actions', 'eef_pos', 'joint_vel', 'object'}} -> list of row dicts."""
    names = list(episodes)
    rows = [dict(demo=n, **episode_features(episodes[n]["actions"], episodes[n]["eef_pos"],
                                            episodes[n]["joint_vel"])) for n in names]
    dup = duplicate_distance([np.concatenate([episodes[n]["eef_pos"], episodes[n]["object"]], 1) for n in names])
    comps = ["idle_ratio", "jerk_mean", "action_jerk", "gripper_flips"]
    Z = {c: robust_z([r[c] for r in rows]) for c in comps}
    dz = robust_z(dup)
    for i, r in enumerate(rows):
        r["dup_dist"] = float(dup[i])
        r["hygiene_score"] = float(-np.mean([Z[c][i] for c in comps]))
        r["flag_idle"] = r["idle_ratio"] > FLAG_IDLE
        r["flag_jerk"] = bool(Z["jerk_mean"][i] > FLAG_Z or Z["action_jerk"][i] > FLAG_Z)
        r["flag_duplicate"] = bool(dz[i] < DUP_Z)
    return rows


def load_for_hygiene(path, demos):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for d in demos:
            g = f[f"data/{d}"]
            out[d] = {"actions": g["actions"][:], "eef_pos": g["obs/robot0_eef_pos"][:],
                      "joint_vel": g["obs/robot0_joint_vel"][:], "object": g["obs/object"][:]}
    return out
