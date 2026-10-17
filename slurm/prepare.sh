#!/bin/bash
# Login-node preparation. Idempotent: safe to re-run after a failed job.
#
#     bash slurm/prepare.sh [config]
#
# Does everything that needs the network or must happen exactly once, so the
# compute-node scripts can run with `uv run --no-sync` and never reach out:
#
#   1. creates the SLURM log directories (sbatch fails at submit time if the
#      --output/--error directories do not exist);
#   2. syncs the environment from uv.lock;
#   3. downloads and verifies the FoodSeg103 split;
#   4. creates the run directory and writes its provenance record, once,
#      instead of letting N array tasks race to create it.
#
# Steps 3 and 4 are both idempotent. Step 3 re-hashes an existing copy instead
# of re-fetching it, so a second run costs seconds rather than minutes. Step 4
# passes --allow-existing, so resubmitting a partially failed array reuses the
# run directory rather than failing. It never rewrites the existing provenance
# record, and individual tasks still refuse to overwrite their own finished
# output -- so resuming is safe, but silently replacing results is still
# impossible. To start a genuinely fresh run, change the config (the run
# directory name includes the config fingerprint) or remove the old directory
# by hand, consciously.
#
# Disk: the synced environment is ~6 GB (torch and its CUDA wheels) and uv's
# download cache grows to tens of GB. On a quota-limited NFS home, export
# UV_CACHE_DIR -- and, if the repository itself lives on the home filesystem,
# UV_PROJECT_ENVIRONMENT -- to scratch before running this. That scratch must
# be *shared*, not node-local: the compute nodes read the same .venv.
#
#     export UV_CACHE_DIR=/scratch/$USER/uv-cache
#     export UV_PROJECT_ENVIRONMENT=/scratch/$USER/food-shapley-venv

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export PATH="$HOME/.local/bin:$PATH"

CONFIG="${1:-${CONFIG:-configs/smoke.toml}}"
SPLIT="${SPLIT:-test}"
export FOOD_SHAPLEY_RUNS="${FOOD_SHAPLEY_RUNS:-$(pwd)/runs}"
export FOOD_SHAPLEY_DATA="${FOOD_SHAPLEY_DATA:-$(pwd)/data}"

# `datasets` downloads the mirror's parquet shards before converting them, and
# caches them under ~/.cache/huggingface by default -- roughly 1.25 GB for the
# test split, on top of the 110 MB that is kept. Keep that off a quota-limited
# home; it is a cache and can be deleted once the split verifies.
export HF_HOME="${HF_HOME:-$FOOD_SHAPLEY_DATA/hf-cache}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

echo "[1/4] creating log directories"
mkdir -p slurm/logs/output_logs slurm/logs/error_logs

echo "[2/4] syncing environment from uv.lock"
uv sync --frozen --all-groups

echo "[3/4] downloading FoodSeg103 '$SPLIT' into $FOOD_SHAPLEY_DATA"
uv run --frozen python -m food_shapley.download --split "$SPLIT"

echo "[4/4] preparing run directory for $CONFIG"
uv run --frozen python -m food_shapley.selftest --config "$CONFIG" prepare --allow-existing

echo "prepare complete -- now submit:"
echo "  sbatch slurm/toy_job.slurm"
echo "  sbatch slurm/toy_array.slurm"
