#!/bin/bash
#SBATCH --job-name=food_shapley_toy
#SBATCH --output=slurm/logs/output_logs/output_food_shapley_toy_%j.log
#SBATCH --error=slurm/logs/error_logs/error_food_shapley_toy_%j.log
#SBATCH -p NvidiaAll
#SBATCH -t 00:15:00

# PLAN.md Milestone 1, step 0: confirm the pipeline runs on a node before any
# real experiment depends on it. It checks that uv, the package, the research
# stack (shapiq, torch, open_clip), the GPU, the dataset and the array
# mechanics all work from inside a batch job.
#
# No model checkpoint is downloaded and no dataset row is read: the research
# libraries are imported and their versions printed, nothing more. A failing
# environment must cost seconds, and a compute node may have no network.
#
# Submit from the repo root:
#     bash slurm/prepare.sh && sbatch slurm/toy_job.slurm
#
# Partition and walltime are copied from a script known to work on this
# cluster. No --gres line: -p NvidiaAll alone was sufficient there.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Batch jobs don't source ~/.bashrc or ~/.zshrc, so uv (standalone installer,
# ~/.local/bin) is not on PATH by default.
export PATH="$HOME/.local/bin:$PATH"

# $HOME is NFS-mounted and quota-limited on this cluster. Keep scratch local.
export TMPDIR="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"

# Where run artifacts and the dataset live. Defaults sit inside the repo; point
# both at persistent scratch once runs produce real volume (one .npz per image)
# -- see slurm/README.md on the NFS quota.
export FOOD_SHAPLEY_RUNS="${FOOD_SHAPLEY_RUNS:-$(pwd)/runs}"
export FOOD_SHAPLEY_DATA="${FOOD_SHAPLEY_DATA:-$(pwd)/data}"

# Keep model/dataset caches off $HOME. Nothing here should download anything,
# but if a library decides to, it must not fill the home quota.
export HF_HOME="${HF_HOME:-$TMPDIR/hf}"

CONFIG="${CONFIG:-configs/smoke.toml}"

echo "=== node environment ==="
echo "host:        $(hostname)"
echo "job id:      ${SLURM_JOB_ID:-<none>}"
echo "partition:   ${SLURM_JOB_PARTITION:-<none>}"
echo "workdir:     $(pwd)"
echo "config:      $CONFIG"
echo "runs root:   $FOOD_SHAPLEY_RUNS"
echo "data root:   $FOOD_SHAPLEY_DATA"
echo "TMPDIR:      $TMPDIR"

echo "=== nvidia-smi ==="
# Driver-level view, separate from torch's. When the two disagree -- a GPU is
# listed here but torch reports none -- the problem is the CUDA/driver pairing,
# not the allocation. Non-fatal; the authoritative check is below.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || echo "WARNING: nvidia-smi failed"
else
    echo "WARNING: nvidia-smi not found on this node"
fi

echo "=== uv ==="
# --no-sync: never touch the network from a compute node. slurm/prepare.sh has
# already synced the environment from the login node.
uv --version

echo "=== research stack, GPU and dataset ==="
# --require-gpu: a GPU-partition job that quietly fell back to CPU would still
# "succeed" and would then take days instead of hours. Fail here instead.
# --require-dataset: catches a data root that is simply not mounted on the node.
uv run --frozen --no-sync python -m food_shapley.envcheck \
    --require-gpu --dataset-split test --require-dataset

echo "=== infrastructure self-test (single task) ==="
uv run --frozen --no-sync python -m food_shapley.selftest --config "$CONFIG" task --task-id 0
uv run --frozen --no-sync python -m food_shapley.selftest --config "$CONFIG" verify --num-tasks 1

echo "=== toy job completed successfully ==="
