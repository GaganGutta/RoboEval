#!/usr/bin/env bash
# One-time environment setup inside WSL2 Ubuntu (also works on native Linux).
# Why WSL: robomimic only supports Linux/macOS; its dependency egl_probe ships
# only as a source tarball that does not build on Windows.
#
# Creates:
#   ~/venvs/data_eval        Python 3.10 venv (on the Linux filesystem: much faster than /mnt/c)
#   ~/src/robomimic          robomimic v0.5.0  (pinned tag)
#   ~/src/robosuite          robosuite v1.5.1  (pinned tag; required by robomimic v1.5 datasets)
#
# Usage (from WSL):  bash setup/setup_wsl.sh
set -euo pipefail

VENV="$HOME/venvs/data_eval"
SRC="$HOME/src"
ROBOMIMIC_TAG="v0.5.0"
ROBOSUITE_TAG="v1.5.1"
TORCH_VERSION="2.5.1"   # stable, ships CUDA 12.4 wheels on PyPI for Linux

# uv manages the Python interpreter and venv (no conda on this machine).
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.10
if [ ! -d "$VENV" ]; then
  uv venv --python 3.10 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

mkdir -p "$SRC"
if [ ! -d "$SRC/robosuite" ]; then
  git clone https://github.com/ARISE-Initiative/robosuite.git "$SRC/robosuite"
fi
git -C "$SRC/robosuite" fetch --tags -q
git -C "$SRC/robosuite" checkout -q "$ROBOSUITE_TAG"

if [ ! -d "$SRC/robomimic" ]; then
  git clone https://github.com/ARISE-Initiative/robomimic.git "$SRC/robomimic"
fi
git -C "$SRC/robomimic" fetch --tags -q
git -C "$SRC/robomimic" checkout -q "$ROBOMIMIC_TAG"

# numpy<2: robomimic's pinned transformers/diffusers and numba predate numpy 2.
uv pip install "torch==$TORCH_VERSION" "numpy<2"
uv pip install -e "$SRC/robosuite"
# egl_probe (a robomimic dependency, only used for image rendering) has
# cmake_minimum_required < 3.5, which CMake 4 rejects. This env var tells CMake 4 to
# configure it anyway.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
uv pip install -e "$SRC/robomimic"
# Our own analysis deps.
uv pip install scikit-learn pandas pyyaml matplotlib seaborn scipy

python - <<'EOF'
import torch, robomimic, robosuite, mujoco, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("robomimic", robomimic.__version__, "robosuite", robosuite.__version__, "mujoco", mujoco.__version__, "numpy", numpy.__version__)
EOF
