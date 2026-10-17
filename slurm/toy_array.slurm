#!/bin/bash
#SBATCH --job-name=food_shapley_toy_array
#SBATCH --output=slurm/logs/output_logs/output_food_shapley_toy_array_%A_%a.log
#SBATCH --error=slurm/logs/error_logs/error_food_shapley_toy_array_%A_%a.log
#SBATCH -p NvidiaAll
#SBATCH -t 00:15:00
#SBATCH --array=0-9

# PLAN.md Milestone 1, step 0: "get the SLURM submit/array mechanics working on
# a trivial job now". Steps 5 and 12 scale this same pattern to thousands of
# images across several models, so the mechanics validated here are:
#
#   1. one output file per task, so tasks never collide and a partial failure
#      is re-runnable per task rather than all-or-nothing;
#   2. per-task seeds derived from (config seed, task index), so a task's seed
#      does not depend on execution order;
#   3. the run directory is created once by prepare.sh, not raced for by N
#      simultaneously starting tasks.
#
# Submit from the repo root:
#     bash slurm/prepare.sh && sbatch slurm/toy_array.slurm
# Then check completeness:
#     uv run --frozen python -m food_shapley.selftest --config configs/smoke.toml \
#         verify --num-tasks 10
#
# %A is the array job id, %a the task index -- unlike %j in a non-array job,
# this keeps each task's log separate.
#
# Note: --array=0-9 must match --num-tasks in the verify step above. Raise
# both together, and add e.g. %20 (0-9%20) to throttle concurrency if the
# partition requires it.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export PATH="$HOME/.local/bin:$PATH"

export TMPDIR="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"

export FOOD_SHAPLEY_RUNS="${FOOD_SHAPLEY_RUNS:-$(pwd)/runs}"

CONFIG="${CONFIG:-configs/smoke.toml}"

echo "array job ${SLURM_ARRAY_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?} on $(hostname)"

# --task-id is omitted on purpose: selftest reads $SLURM_ARRAY_TASK_ID.
# --no-sync keeps compute nodes off the network.
uv run --frozen --no-sync python -m food_shapley.selftest --config "$CONFIG" task

echo "task ${SLURM_ARRAY_TASK_ID:-?} completed successfully"
