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
#   4. downloads and verifies the Food-101 dish prompt set (names only, ~12 kB);
#   5. loads the configured CLIP checkpoint once, which populates the weight
#      cache under HF_HOME and fails loudly if the architecture and the
#      pretrained tag disagree (see W01-10);
#   6. creates the run directory and writes its provenance record, once,
#      instead of letting N array tasks race to create it. Both the self-test
#      and the real run share that directory, so both are prepared here.
#
# Steps 3-6 are all idempotent. Steps 3 and 4 re-hash an existing copy instead
# of re-fetching it, so a second run costs seconds rather than minutes; step 5
# reads the cache. Step 6 passes --allow-existing, so resubmitting a partially
# failed array reuses the run directory rather than failing. It never rewrites
# the existing provenance record, and individual tasks still refuse to
# overwrite their own finished output -- so resuming is safe, but silently
# replacing results is still impossible. To start a genuinely fresh run, change
# the config (the run directory name includes the config fingerprint) or remove
# the old directory by hand, consciously.
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
#
# The CLIP weights land here too (~600 MB for ViT-B-32), and unlike the parquet
# shards they are *not* disposable: compute nodes may be offline, so this is
# the only copy they can load from. It must therefore be on shared scratch, not
# node-local TMPDIR, and must survive between prepare.sh and the job.
export HF_HOME="${HF_HOME:-$FOOD_SHAPLEY_DATA/hf-cache}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

echo "[1/6] creating log directories"
mkdir -p slurm/logs/output_logs slurm/logs/error_logs

echo "[2/6] syncing environment from uv.lock"
uv sync --frozen --all-groups

echo "[3/6] downloading FoodSeg103 '$SPLIT' into $FOOD_SHAPLEY_DATA"
uv run --frozen python -m food_shapley.download --split "$SPLIT"

echo "[4/6] downloading the Food-101 dish prompt set"
uv run --frozen python -m food_shapley.prompts

echo "[5/6] caching the CLIP checkpoint named in $CONFIG"
uv run --frozen python -m food_shapley.scoring --config "$CONFIG"

echo "[6/6] preparing run directory for $CONFIG"
# Both entry points resolve the same directory -- RunLayout.for_run(name,
# fingerprint) -- and both pass --allow-existing, so this is one idempotent
# directory, written once, whichever job runs next.
uv run --frozen python -m food_shapley.selftest --config "$CONFIG" prepare --allow-existing
uv run --frozen python -m food_shapley.run --config "$CONFIG" prepare --allow-existing

echo "prepare complete -- now submit:"
echo "  sbatch slurm/toy_job.slurm                      # infrastructure only"
echo "  sbatch slurm/toy_array.slurm                    # array mechanics only"
echo "  CONFIG=$CONFIG sbatch slurm/run_array.slurm     # the real run"
echo
echo "then always check completeness:"
echo "  uv run python -m food_shapley.run --config $CONFIG verify"
