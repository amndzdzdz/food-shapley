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
#   3. creates the run directory and writes its provenance record, once,
#      instead of letting N array tasks race to create it.
#
# Step 3 passes --allow-existing, so resubmitting a partially failed array
# reuses the run directory rather than failing. It never rewrites the existing
# provenance record, and individual tasks still refuse to overwrite their own
# finished output -- so resuming is safe, but silently replacing results is
# still impossible. To start a genuinely fresh run, change the config (the run
# directory name includes the config fingerprint) or remove the old directory
# by hand, consciously.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export PATH="$HOME/.local/bin:$PATH"

CONFIG="${1:-${CONFIG:-configs/smoke.toml}}"
export FOOD_SHAPLEY_RUNS="${FOOD_SHAPLEY_RUNS:-$(pwd)/runs}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

echo "[1/3] creating log directories"
mkdir -p slurm/logs/output_logs slurm/logs/error_logs

echo "[2/3] syncing environment from uv.lock"
uv sync --frozen --all-groups

echo "[3/3] preparing run directory for $CONFIG"
uv run --frozen python -m food_shapley.selftest --config "$CONFIG" prepare --allow-existing

echo "prepare complete -- now submit:"
echo "  sbatch slurm/toy_job.slurm"
echo "  sbatch slurm/toy_array.slurm"
