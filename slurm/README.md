# SLURM scripts

Covers [`PLAN.md`](../PLAN.md) Milestone 1, step 0 — get the submit and array
mechanics working on a trivial job before any real experiment depends on them.
Steps 5 and 12 scale this same pattern up.

**Status: written but never submitted.** These scripts have not been run on a
cluster. The subcommands they call are unit-tested locally, and the cluster
settings are copied from a script known to work on this cluster, but the
scripts themselves are unverified until someone submits them. Record the
outcome of the first submission in
[`insights/week-01.md`](../insights/week-01.md).

## Usage

From the repository root:

```bash
bash slurm/prepare.sh                 # login node: logs, env sync, run dir
sbatch slurm/toy_job.slurm            # single job: node sanity check
sbatch slurm/toy_array.slurm          # 10-task array: array mechanics

uv run --frozen python -m food_shapley.selftest \
    --config configs/smoke.toml verify --num-tasks 10
```

`verify` exits non-zero and lists what is wrong if any task is missing or any
seed is wrong or duplicated.

## Division of labour

| Where | What | Why |
|---|---|---|
| `prepare.sh` (login node) | log dirs, `uv sync`, create run directory | Needs the network; must happen exactly once |
| `toy_job.slurm` (compute) | environment + GPU + import checks | Fails fast on node misconfiguration |
| `toy_array.slurm` (compute) | one output file per task | Tasks never collide; failures are re-runnable per task |
| `selftest verify` (login node) | completeness and seed checks | Catches silent partial failure of an array |

The run directory is created once by `prepare.sh` rather than by the tasks, so
N simultaneously starting array tasks cannot race to create it.

## Cluster conventions assumed

Taken from a script that runs without problems on this cluster. Revisit each
if the cluster changes.

- **Partition `-p NvidiaAll`**, no `--gres` line — the working reference script
  requests the partition alone.
- **Batch jobs do not source `~/.bashrc` or `~/.zshrc`.** `uv` lives in
  `~/.local/bin` (standalone installer) and must be put on `PATH` explicitly.
  Every script does `export PATH="$HOME/.local/bin:$PATH"`.
- **`$HOME` is NFS-mounted and quota-limited.** Keep caches and bulk artifacts
  off it: `TMPDIR` is set to node-local scratch, and `FOOD_SHAPLEY_RUNS` should
  point at persistent scratch once runs produce real volume.
- **Compute nodes may be offline.** Anything needing the network happens in
  `prepare.sh` on a login node; compute-node steps use `uv run --no-sync` so
  they never try to resolve or download.
- **Scripts are submitted from the repository root**, and use
  `cd "${SLURM_SUBMIT_DIR:-$(pwd)}"`.
- **Preparation is idempotent**, so a resubmission after a partial failure
  skips finished work instead of redoing or clobbering it.
- `set -euo pipefail` everywhere.

### Directory naming

The reference script keeps scripts and logs under `run/`. This repository uses
`slurm/` instead, because `runs/` is already the artifact root
(`runs/<name>-<fingerprint>/`) and `run/` next to `runs/` is a footgun. Logs
therefore live in `slurm/logs/{output_logs,error_logs}/`, mirroring the
reference layout otherwise. Rename if cross-project consistency matters more.

## Before scaling up (steps 5 and 12)

- Point `FOOD_SHAPLEY_RUNS` at persistent scratch, not the repo.
- Set a realistic `-t` walltime; the toy jobs use 15 minutes.
- Consider throttling concurrency with `--array=0-7117%N` if the partition
  requires it.
- Keep one output file per image, so a partial failure is re-runnable per task.
- Re-run `verify` (or its successor) after every array; an array that silently
  drops 3 % of its tasks looks exactly like one that succeeded.
