"""Thin wrapper around robomimic's BC-RNN training.

One call = one training run on one subset (a robomimic filter key) with one seed.
We turn off robomimic's own rollouts/validation: evaluation is done separately by
eval/rollout.py with fixed environment seeds, so every condition is scored on the
exact same initial states.

Usage:
    python -m train.bc_rnn --dataset data/processed/can_mh_low_dim.hdf5 \
        --filter_key better --seed 0 --out runs/exp1/can_better_s0
"""
import argparse
import glob
import json
import os
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CFG = os.path.join(REPO, "configs", "bc_rnn_lowdim.yaml")


def build_config(dataset, filter_key, seed, out_dir, cfg):
    import train.robomimic_compat  # noqa: F401
    from robomimic.config import config_factory

    config = config_factory("bc")
    a, t = cfg["algo"], cfg["train"]
    with config.values_unlocked():
        config.experiment.name = "run"
        config.experiment.validate = False
        config.experiment.rollout.enabled = False   # we evaluate separately (fixed seeds)
        config.experiment.render_video = False
        config.experiment.logging.log_tb = False
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = None
        config.experiment.save.epochs = list(t["save_epochs"])
        config.experiment.save.on_best_rollout_success_rate = False
        config.experiment.epoch_every_n_steps = t["steps_per_epoch"]

        config.train.data = [{"path": os.path.abspath(dataset), "filter_key": filter_key}]
        config.train.hdf5_filter_key = filter_key
        config.train.output_dir = os.path.abspath(out_dir)
        config.train.num_data_workers = 0
        config.train.hdf5_cache_mode = "all"
        config.train.batch_size = t["batch_size"]
        config.train.num_epochs = t["num_epochs"]
        config.train.seed = seed
        config.train.seq_length = a["rnn_horizon"]

        config.algo.optim_params.policy.learning_rate.initial = a["lr"]
        config.algo.actor_layer_dims = ()
        config.algo.gmm.enabled = a["gmm"]
        config.algo.gmm.num_modes = a["gmm_num_modes"]
        config.algo.rnn.enabled = True
        config.algo.rnn.horizon = a["rnn_horizon"]
        config.algo.rnn.hidden_dim = a["rnn_hidden_dim"]
        config.algo.rnn.rnn_type = a["rnn_type"]
        config.algo.rnn.num_layers = a["rnn_num_layers"]

        config.observation.modalities.obs.low_dim = [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    return config


def find_checkpoints(out_dir):
    """robomimic writes <out>/run/<timestamp>/models/model_epoch_<k>.pth."""
    paths = glob.glob(os.path.join(out_dir, "run", "*", "models", "model_epoch_*.pth"))
    return sorted(paths, key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--filter_key", default=None)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=DEFAULT_CFG)
    p.add_argument("--num_epochs", type=int, default=None, help="override config num_epochs")
    p.add_argument("--save_epochs", type=int, nargs="+", default=None, help="override config save_epochs")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.num_epochs is not None:
        cfg["train"]["num_epochs"] = args.num_epochs
        cfg["train"]["save_epochs"] = [e for e in cfg["train"]["save_epochs"] if e <= args.num_epochs] or [args.num_epochs]
    if args.save_epochs is not None:
        cfg["train"]["save_epochs"] = args.save_epochs

    done_marker = os.path.join(args.out, "train_done.json")
    if os.path.exists(done_marker):
        print("already trained:", args.out)
        return

    import torch
    import train.robomimic_compat  # noqa: F401  (must precede robomimic imports)
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.scripts.train import train

    os.makedirs(args.out, exist_ok=True)
    config = build_config(args.dataset, args.filter_key, args.seed, args.out, cfg)
    config.lock()
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    t0 = time.time()
    train(config, device=device)
    elapsed = time.time() - t0

    info = {
        "dataset": os.path.abspath(args.dataset), "filter_key": args.filter_key,
        "seed": args.seed, "config_file": os.path.abspath(args.config), "config": cfg,
        "train_seconds": elapsed, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoints": find_checkpoints(args.out),
    }
    with open(done_marker, "w") as f:
        json.dump(info, f, indent=2)
    print(f"trained in {elapsed/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
