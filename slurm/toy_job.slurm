#!/bin/bash
#SBATCH --job-name=food_shapley_toy
#SBATCH --output=slurm/logs/output_logs/output_food_shapley_toy_%j.log
#SBATCH --error=slurm/logs/error_logs/error_food_shapley_toy_%j.log
#SBATCH -p NvidiaAll
#SBATCH -t 00:15:00

# PLAN.md Milestone 1, step 0: confirm the pipeline runs on a node before any
# real experiment depends on it. Touches no dataset and no model -- it checks
# that uv, the package, the run directory and GPU visibility all work from a
# batch job.
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

# Where run artifacts go. Default is the repo's runs/ directory; point this at
# persistent scratch once runs produce real volume (one .npz per image).
export FOOD_SHAPLEY_RUNS="${FOOD_SHAPLEY_RUNS:-$(pwd)/runs}"

CONFIG="${CONFIG:-configs/smoke.toml}"

echo "=== node environment ==="
echo "host:        $(hostname)"
echo "job id:      ${SLURM_JOB_ID:-<none>}"
echo "partition:   ${SLURM_JOB_PARTITION:-<none>}"
echo "workdir:     $(pwd)"
echo "config:      $CONFIG"
echo "runs root:   $FOOD_SHAPLEY_RUNS"
echo "TMPDIR:      $TMPDIR"

echo "=== GPU visibility ==="
# Informational only: this toy job has no GPU workload yet, because torch is
# not a dependency until Week 1. Non-fatal so the job still validates the rest.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || echo "WARNING: nvidia-smi failed"
else
    echo "WARNING: nvidia-smi not found on this node"
fi

echo "=== uv and package ==="
# --no-sync: never touch the network from a compute node. slurm/prepare.sh has
# already synced the environment from the login node.
uv --version
uv run --frozen --no-sync python -c "import food_shapley; print('food_shapley', food_shapley.__version__)"

echo "=== infrastructure self-test (single task) ==="
uv run --frozen --no-sync python -m food_shapley.selftest --config "$CONFIG" task --task-id 0
uv run --frozen --no-sync python -m food_shapley.selftest --config "$CONFIG" verify --num-tasks 1

echo "=== toy job completed successfully ==="
