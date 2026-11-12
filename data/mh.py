"""Loading robomimic MH (multi-human) low-dim datasets and managing filter keys.

robomimic hdf5 layout (docs: robomimic.github.io/docs/datasets/overview.html):
    data/demo_<i>/obs/<obs_key>   (T, d)
    data/demo_<i>/actions         (T, 7)
    data/demo_<i>/states          (T, n)   full MuJoCo state, used to reset the sim exactly
    mask/<filter_key>             array of demo names (bytes) belonging to a subset

Operator-quality labels live in mask/{worse, okay, better} (verified on download, see
docs/DATA_CARD.md). We never modify the raw download: subsets we create are written as
extra filter keys into a *working copy* under data/processed/.
"""
import os
import shutil

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
PROC_DIR = os.path.join(HERE, "processed")

TASKS = ["lift", "can", "square"]
QUALITY_KEYS = ["worse", "okay", "better"]
# Default low-dim observation keys used by the robomimic paper for these tasks.
LOW_DIM_OBS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


def raw_path(task):
    return os.path.join(RAW_DIR, "v1.5", task, "mh", "low_dim_v15.hdf5")


def working_path(task):
    """Working copy that holds our extra filter keys. Created on first use."""
    p = os.path.join(PROC_DIR, f"{task}_mh_low_dim.hdf5")
    if not os.path.exists(p):
        os.makedirs(PROC_DIR, exist_ok=True)
        shutil.copyfile(raw_path(task), p)
    return p


def _decode(names):
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def demo_sort_key(name):
    return int(name.split("_")[-1])


def list_demos(path):
    with h5py.File(path, "r") as f:
        return sorted(f["data"].keys(), key=demo_sort_key)


def read_filter_key(path, key):
    with h5py.File(path, "r") as f:
        return sorted(_decode(f[f"mask/{key}"][:]), key=demo_sort_key)


def list_filter_keys(path):
    with h5py.File(path, "r") as f:
        return list(f["mask"].keys()) if "mask" in f else []


def write_filter_key(path, key, demos, overwrite=True):
    """Write (or replace) mask/<key>. Keys must not contain '/' (h5py would nest them)."""
    assert "/" not in key, key
    demos = sorted(demos, key=demo_sort_key)
    with h5py.File(path, "a") as f:
        all_demos = set(f["data"].keys())
        missing = set(demos) - all_demos
        assert not missing, f"unknown demos: {sorted(missing)[:5]}"
        if f"mask/{key}" in f:
            if not overwrite:
                raise KeyError(key)
            del f[f"mask/{key}"]
        f.create_dataset(f"mask/{key}", data=np.array(demos, dtype="S"))


def quality_labels(path):
    """Return {demo_name: 'worse'|'okay'|'better'}."""
    labels = {}
    for q in QUALITY_KEYS:
        for d in read_filter_key(path, q):
            assert d not in labels, f"{d} in two quality groups"
            labels[d] = q
    return labels


def load_episodes(path, demos=None, obs_keys=LOW_DIM_OBS):
    """Load episodes into memory: {demo: {'obs': (T, d) concat of obs_keys, 'actions': (T, 7)}}.

    Low-dim datasets are small (~100 MB), so loading everything is fine.
    """
    out = {}
    with h5py.File(path, "r") as f:
        names = demos if demos is not None else sorted(f["data"].keys(), key=demo_sort_key)
        for d in names:
            g = f[f"data/{d}"]
            obs = np.concatenate([g[f"obs/{k}"][:].reshape(len(g["actions"]), -1) for k in obs_keys], axis=1)
            out[d] = {"obs": obs.astype(np.float32), "actions": g["actions"][:].astype(np.float32)}
    return out
