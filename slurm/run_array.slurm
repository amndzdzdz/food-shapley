#!/bin/bash
#SBATCH --job-name=food_shapley_run
#SBATCH --output=slurm/logs/output_logs/output_food_shapley_run_%A_%a.log
#SBATCH --error=slurm/logs/error_logs/error_food_shapley_run_%A_%a.log
#SBATCH -p NvidiaAll
#SBATCH -t 04:00:00
#SBATCH --array=0-27

# PLAN.md Milestone 1, step 5: the full run. One exact decomposition per image
# of the configured split, one .npz per image, as a job array.
#
# Submit from the repo root:
#     bash slurm/prepare.sh configs/week01-clip-blur.toml
#     CONFIG=configs/week01-clip-blur.toml sbatch slurm/run_array.slurm
# Then -- always, not optionally -- check completeness:
#     uv run --frozen python -m food_shapley.run \
#         --config configs/week01-clip-blur.toml verify
#
# toy_array.slurm rehearsed these mechanics without a dataset or a model. The
# differences here are that the work is real and that it is divided by image
# rather than by task index: `run.py` derives each image's output name and seed
# from the image itself, so the array size below is a throughput choice and
# does not change a single stored number. Raising or lowering --array and
# resubmitting resumes the same run.
#
# --array and the task count must agree, and they do so without being written
# twice: `run task` reads $SLURM_ARRAY_TASK_COUNT. Edit --array alone, or
# override it per submission with `sbatch --array=0-N`. An array indexed from 1
# rather than 0 is rejected loudly rather than silently skipping the last
# image's worth of work.
#
# Why 28 and not more. Measured on this cluster, 2026-10-23, for account
# stud_ifi / QOS normal: MaxSubmit = 30 and MaxJobs = 15. Array tasks count
# individually against MaxSubmit, so --array=0-31 is rejected outright with
# `AssocMaxSubmitJobLimit` -- the submission never reaches a node. 28 leaves
# two slots for an interactive or smoke job. Only 15 run at once whatever is
# requested; the surplus pends and starts as tasks finish, which is worth
# having because per-image cost spans three orders of magnitude and a shorter
# per-task runtime is a smaller walltime bet. A `%` throttle does NOT help
# here: it limits concurrency, not the submitted count.
#
# Walltime is an estimate and has never been measured on this cluster. 2,135
# images and 91,828 coalitions (W01-11) over 32 tasks is ~2,870 coalitions per
# task, which is minutes of GPU time -- but the checkpoint load, the dataset
# read and the per-image blur are not in that figure. Four hours is deliberate
# headroom for the first submission; correct it from the measured runtime and
# record the number in insights/week-01.md. A task that runs out of walltime
# loses nothing permanently: finished images stay on disk and a resubmission
# picks up the rest.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export PATH="$HOME/.local/bin:$PATH"

export TMPDIR="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"

export FOOD_SHAPLEY_RUNS="${FOOD_SHAPLEY_RUNS:-$(pwd)/runs}"
export FOOD_SHAPLEY_DATA="${FOOD_SHAPLEY_DATA:-$(pwd)/data}"

# The checkpoint must be readable from the node. prepare.sh cached it under
# $FOOD_SHAPLEY_DATA/hf-cache on shared scratch precisely because compute nodes
# may be offline; pointing HF_HOME at node-local TMPDIR here would make every
# task try to download it.
export HF_HOME="${HF_HOME:-$FOOD_SHAPLEY_DATA/hf-cache}"

# Measured 2026-10-23 on adakit: the node *does* have network access, and the
# model load contacted the Hub anyway ("unauthenticated requests to the HF
# Hub" in the task log) even though prepare.sh had already cached the weights.
# That is worse than it looks with 28 tasks starting at once: the run becomes
# dependent on Hub availability and rate limits at exactly the moment it is
# most parallel, and a silent re-download would make two tasks' weights
# un-attributable. Offline mode turns that into a loud, immediate failure on
# a cache miss instead -- which is the right trade, because the cache miss is
# fixable on the login node and the rate limit is not.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CONFIG="${CONFIG:-configs/smoke.toml}"

echo "array ${SLURM_ARRAY_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?}/${SLURM_ARRAY_TASK_COUNT:-?} on $(hostname)"
echo "config:    $CONFIG"
echo "runs root: $FOOD_SHAPLEY_RUNS"

# --task-id and --num-tasks are omitted on purpose: run.py reads
# $SLURM_ARRAY_TASK_ID and $SLURM_ARRAY_TASK_COUNT, so the array size lives in
# exactly one place, the #SBATCH line above.
#
# --no-sync keeps compute nodes off the network. A non-zero exit here means at
# least one image failed; the task report in <run>/logs/task-*.json names them.
uv run --frozen --no-sync python -m food_shapley.run --config "$CONFIG" task

echo "task ${SLURM_ARRAY_TASK_ID:-?} completed"
