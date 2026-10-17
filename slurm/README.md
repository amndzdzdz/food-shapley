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
bash slurm/prepare.sh                 # login node: logs, env sync, dataset, run dir
sbatch slurm/toy_job.slurm            # single job: node sanity check
sbatch slurm/toy_array.slurm          # 10-task array: array mechanics

uv run --frozen python -m food_shapley.selftest \
    --config configs/smoke.toml verify --num-tasks 10
```

`verify` exits non-zero and lists what is wrong if any task is missing or any
seed is wrong or duplicated.

## Disk, and where to put it

Measured locally, not on the cluster:

| Thing | Size | Lives for |
|---|---|---|
| `.venv` (torch + CUDA wheels dominate) | ~6 GB | The project |
| `~/.cache/uv` | tens of GB | Can be deleted after `uv sync` |
| Hugging Face parquet cache (`HF_HOME`) | ~1.25 GB per split | Can be deleted after `download --verify-only` passes |
| `data/raw/foodseg103/test` | ~110 MB | The project |

Against a quota-limited NFS `$HOME` that matters. Before the first
`prepare.sh`:

```bash
export UV_CACHE_DIR=/scratch/$USER/uv-cache
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/food-shapley-venv   # if the repo is on $HOME
export FOOD_SHAPLEY_DATA=/scratch/$USER/food-shapley-data
export FOOD_SHAPLEY_RUNS=/scratch/$USER/food-shapley-runs
```

`UV_PROJECT_ENVIRONMENT` must point at **shared** scratch, not node-local
`/tmp`: the compute nodes read the same `.venv`. `TMPDIR` and `HF_HOME` inside
the job scripts are node-local on purpose, because nothing in them needs to
outlive the job.

## Division of labour

| Where | What | Why |
|---|---|---|
| `prepare.sh` (login node) | log dirs, `uv sync`, dataset download, create run directory | Needs the network; must happen exactly once |
| `toy_job.slurm` (compute) | `envcheck`: imports, GPU, dataset integrity | Fails fast on node misconfiguration |
| `toy_array.slurm` (compute) | one output file per task | Tasks never collide; failures are re-runnable per task |
| `selftest verify` (login node) | completeness and seed checks | Catches silent partial failure of an array |

The run directory is created once by `prepare.sh` rather than by the tasks, so
N simultaneously starting array tasks cannot race to create it.

### The environment check

`toy_job.slurm` runs `python -m food_shapley.envcheck --require-gpu
--dataset-split test --require-dataset`, which can also be run by hand on any
machine:

```bash
uv run python -m food_shapley.envcheck --dataset-split test
```

It imports `shapiq`, `torch`, `open_clip`, `numpy`, `PIL` and `datasets`,
prints their versions, reports CUDA availability and the GPUs it can see, and
re-hashes the dataset split against its manifest. It **downloads nothing** —
no checkpoint, and not a single dataset row — so it costs seconds and is safe
on an offline node.

`--require-gpu` turns a missing CUDA device into a failure. A GPU-partition
job that quietly fell back to CPU would otherwise still exit 0 and then take
days rather than hours. `nvidia-smi` is printed separately because when the
driver lists a GPU that torch cannot see, the fault is the CUDA/driver
pairing, not the allocation — and the two outputs side by side say so.

The array script does **not** run the check: importing torch in each of N
tasks costs seconds per task for information that does not vary.

## Cluster conventions assumed

Taken from a script that runs without problems on this cluster. Revisit each
if the cluster changes.

- **Partition `-p NvidiaAll`**, no `--gres` line — the working reference script
  requests the partition alone.
- **Batch jobs do not source `~/.bashrc` or `~/.zshrc`.** `uv` lives in
  `~/.local/bin` (standalone installer) and must be put on `PATH` explicitly.
  Every script does `export PATH="$HOME/.local/bin:$PATH"`.
- **`$HOME` is NFS-mounted and quota-limited.** Keep caches and bulk artifacts
  off it: `TMPDIR` and `HF_HOME` are set to node-local scratch, and
  `FOOD_SHAPLEY_RUNS`, `FOOD_SHAPLEY_DATA`, `UV_CACHE_DIR` and
  `UV_PROJECT_ENVIRONMENT` should point at shared scratch. See *Disk* above
  for the measured sizes.
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
