"""Download robomimic v1.5 multi-human (MH) low-dim datasets from Hugging Face.

Source: https://huggingface.co/datasets/amandlek/robomimic  (path v1.5/{task}/mh/low_dim_v15.hdf5)
We pin the Hugging Face commit so every result traces back to the exact bytes used.

Usage:
    python data/download.py --tasks lift can square
"""
import argparse
import hashlib
import json
import os

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "amandlek/robomimic"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["lift", "can", "square"])
    parser.add_argument("--revision", default=None,
                        help="HF commit to pin; default = current main (recorded in manifest)")
    args = parser.parse_args()

    revision = args.revision or HfApi().dataset_info(REPO_ID).sha
    manifest_path = os.path.join(RAW_DIR, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    for task in args.tasks:
        rel = f"v1.5/{task}/mh/low_dim_v15.hdf5"
        path = hf_hub_download(REPO_ID, rel, repo_type="dataset", revision=revision,
                               local_dir=RAW_DIR)
        digest = sha256(path)
        manifest[task] = {"repo": REPO_ID, "file": rel, "revision": revision,
                          "sha256": digest, "bytes": os.path.getsize(path)}
        print(f"{task}: {path}  {os.path.getsize(path)/1e6:.1f} MB  sha256={digest[:12]}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
