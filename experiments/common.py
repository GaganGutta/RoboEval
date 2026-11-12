"""Shared helpers for experiment scripts: job pool, run bookkeeping, provenance."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_commit():
    try:
        out = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        dirty = subprocess.call(["git", "-C", REPO, "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL) != 0
        return out.decode().strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def run_jobs(jobs, max_parallel, log_dir):
    """jobs: list of (name, argv). Runs up to max_parallel at once, each logging to
    <log_dir>/<name>.log. Returns {name: returncode}. Failures are reported, not hidden."""
    os.makedirs(log_dir, exist_ok=True)

    def _one(job):
        name, argv = job
        t0 = time.time()
        with open(os.path.join(log_dir, f"{name}.log"), "w") as log:
            rc = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: rc={rc} ({(time.time()-t0)/60:.1f} min)", flush=True)
        return name, rc

    with ThreadPoolExecutor(max_parallel) as ex:
        results = dict(ex.map(_one, jobs))
    failed = [n for n, rc in results.items() if rc != 0]
    if failed:
        print("FAILED:", failed, flush=True)
    return results


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


PY = sys.executable
