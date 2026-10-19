# Why Does the Model Say "Burger"?

Exact Shapley values and Shapley interactions over **semantic food-ingredient
segments**, used to ask whether a vision-language model recognises a dish from
a single telltale ingredient or from a genuine combination of ingredients.

## Motivation and objectives

Compositionality in vision-language models is usually measured through
*accuracy* on benchmarks such as SugarCrepe and ARO — they tell us *whether* a
model composes, not *how*. This project measures the *how* directly, by
decomposing a model's dish score into per-ingredient contributions and
higher-order interactions.

Research questions, as derived from [`PLAN.md`](PLAN.md):

1. When CLIP predicts a dish, how much of that prediction comes from a single
   ingredient's main effect versus from interactions between ingredients?
2. Are those conclusions stable across removal operators (blur-based masking
   versus token dropping inside the ViT)?
3. *(Milestone 2)* Does an interaction-based "compositionality index" vary
   across model families, and how does it relate to accuracy-based
   compositionality benchmarks?
4. *(Milestone 2, secondary)* Does reliance on single parts rather than
   interactions predict misclassification under distribution shift, and does
   it beat a plain-saliency baseline?

## Intended contribution

Per `PLAN.md`, the combination of FoodSeg103-style ingredient segmentation with
exact Shapley-interaction analysis is not covered by prior work. The intended
contribution is distinguished from the closest related work as:
segment-level (vs. pixel- or patch-level), food domain, **exact** computation
via `shapiq` rather than approximation, and the explicit
*telltale-ingredient vs. combination* framing.

Target: **at least a workshop paper** (Milestone 1), with a conference-length
extension as Milestone 2.

## Project status

**Infrastructure, data acquisition and the segment→player mapping. No model
has been loaded, no experiment run, no result exists.** Week 1 task 1.2 is
done; task 1.1 is still open on its cluster-submission step. What exists
today:

| Area | Status |
|---|---|
| Repository skeleton and tooling | Implemented |
| Reproducibility scaffolding (config, seeding, run provenance, artifact layout) | Implemented |
| Research dependencies (`shapiq`, `torch`, `open_clip`, `datasets`) | Installed and locked |
| FoodSeg103 download + integrity verification (`food_shapley.download`) | Implemented; test split downloaded and verified |
| Environment / GPU / dataset check (`food_shapley.envcheck`) | Implemented |
| SLURM toy job + array scaffolding (`slurm/`) | Written and locally simulated — **never submitted to a cluster** |
| Weekly implementation plans (`docs/`) | Drafted from `PLAN.md` |
| Insight log (`insights/`) | Dataset, player-construction and masking observations — **no experimental findings** |
| FoodSeg103 loading (`food_shapley.dataset`) | Implemented |
| Segment→player mapping, CLIP geometry (`food_shapley.segments`) | Implemented; alignment verified against torchvision **and** by eye. The scene is a player, never an ingredient ([W01-09](insights/week-01.md)) |
| Blur removal operator + coalition renderer (`masking.py`, `preview.py`) | Implemented; visual check passed on 19 coalitions / 5 images |
| CLIP value function, `shapiq.Game`, exact SV/k-SII/Möbius | Planned (Week 1, tasks 1.3–1.5) |
| Full SLURM run, second removal operator, analysis | Planned (Weeks 1–2) |
| Cross-model scaling, compositionality index, benchmarks | Planned (Weeks 4–6) |

A complete repository skeleton is *not* evidence of publication-quality
results.

## Repository structure

```
PLAN.md                 Original project plan (source of truth for scope)
CLAUDE.md               Instructions for Claude Code
configs/                Experiment configurations (TOML), one per run
docs/                   Weekly implementation plans + dataset documentation
insights/               Dated research findings, one file per week
slurm/                  Cluster job scripts + the infrastructure self-test
src/food_shapley/       Installable Python package
tests/unit/             Fast, isolated unit tests
tests/integration/      Multi-component tests (marked `integration`)
data/                   Inputs — git-ignored, never written by the code
runs/                   Generated artifacts — git-ignored, one dir per run
```

## Prerequisites

- Python >= 3.12 — the exact version is pinned in `.python-version` (3.12) and
  `uv` installs it for you, so checkouts cannot drift. The floor is 3.12
  because `shapiq` >= 1.5 requires it.
- [`uv`](https://docs.astral.sh/uv/) for environments, dependencies and locking
- ~6 GB of disk for `.venv` (torch and its CUDA wheels), plus tens of GB of
  transient `uv` cache. On a quota-limited home directory, see the *Disk*
  section of [`slurm/README.md`](slurm/README.md).
- For the actual experiments (not yet implemented): access to a SLURM cluster
  with GPUs.

## Setup

```bash
uv sync --all-groups          # create .venv from uv.lock, exactly
uv run pre-commit install     # enable the pre-commit hooks
```

## Data

FoodSeg103 is fetched from a revision-pinned mirror into `data/raw/`, which is
git-ignored:

```bash
uv run python -m food_shapley.download --split test   # ~3 min, idempotent
uv run python -m food_shapley.download --split test --verify-only
```

Each split gets a `manifest.json` recording the mirror, the pinned revision,
the example count, the license and a content hash; `--verify-only` re-hashes
the files against it, so a truncated download fails here rather than as a
confusing model error later. Provenance and licensing:
[`docs/datasets.md`](docs/datasets.md).

To check that a machine can actually run the pipeline — imports, GPU, dataset
integrity, downloading nothing:

```bash
uv run python -m food_shapley.envcheck --dataset-split test
```

## Looking at what the model will see

`PLAN.md` requires a human to confirm that "only bun visible" really shows only
the bun before any values are computed, because the efficiency check cannot
detect mis-registered masks — a game built on them stays perfectly
self-consistent. This renders one labelled sheet per image, with the original,
the empty coalition, every singleton, the ingredients without the scene, and
the grand coalition:

```bash
uv run python -m food_shapley.preview --config configs/week01-clip-blur.toml --num-images 4
uv run python -m food_shapley.preview --config configs/week01-clip-blur.toml \
    --index 1569 --scale 3 --open
```

Sheets go to `runs/visual-check/<fingerprint>/`. Panels are magnified with
nearest-neighbour (`--scale`, default 2) because 224 px is too small to judge a
segment boundary on screen, and because a smoothing resize would hide exactly
the one-pixel misalignment the check is looking for. `--open` hands each sheet
to the system image viewer.

It loads no model and computes no value. Sampled indices are drawn from the
config seed, so repeating a command gives the same images. Re-run it whenever
the geometry, the player construction or the removal operator changes; the
outcome of the Week-1 inspection is recorded as
[W01-07](insights/week-01.md).

Under `background_mode = "player"` the first and last panels must be
pixel-identical: the grand coalition *is* the photograph. If they differ, some
pixel belongs to no player and the whole decomposition is explaining an image
the model never saw.

## Checks

```bash
uv run ruff check .           # lint
uv run ruff format --check .  # formatting
uv run mypy                   # strict type checking (paths come from pyproject.toml)
uv run pytest                 # tests
```

Useful variants:

```bash
uv run pytest -m "not integration"   # unit tests only
uv run pytest -m integration         # integration tests only
uv run ruff format .                 # apply formatting
uv run pre-commit run --all-files    # everything at once
```

**There is no CI.** Pre-commit is the only automated gate, and it is filtered
by file type: a commit touching only Markdown runs neither mypy nor pytest,
and `slow` tests are excluded from the hook entirely. Run
`uv run pre-commit run --all-files` and `uv run pytest -m "not gpu"` by hand
before anything that matters. A clean commit is not evidence that the suite
is green.

## Running on the cluster

`PLAN.md` step 0 calls for the SLURM submit and array mechanics to work on a
trivial job before any real experiment depends on them. That toy job exists:

```bash
bash slurm/prepare.sh          # login node: log dirs, env sync, dataset, run dir
sbatch slurm/toy_job.slurm     # single job: uv, research stack, GPU, dataset
sbatch slurm/toy_array.slurm   # 10-task array: per-task outputs and seeds

uv run --frozen python -m food_shapley.selftest \
    --config configs/smoke.toml verify --num-tasks 10
```

The toy job loads no model and reads no dataset row — it imports the research
libraries, checks the GPU, and re-hashes the dataset. See
[`slurm/README.md`](slurm/README.md) for the cluster conventions these scripts
assume, all of which are still unverified: **nothing here has been submitted
to a cluster yet.**

## Reproducibility conventions

- Every run is described by a TOML file in `configs/`; no hidden defaults.
- Seeds are explicit. `seed_everything` pins process-wide generators and
  `derive_seed(base, label)` gives each image / array task a stable sub-seed.
- Each run writes `runs/<name>-<fingerprint>/` containing `config.json` and
  `run_metadata.json` (git commit, dirty flag, Python and package versions,
  timestamp), so a number can be traced back to its config, code and inputs.
- Existing run directories are never silently overwritten.
- `data/` and `runs/` are git-ignored; pre-commit blocks files over 512 kB.

## Documentation

- Weekly implementation plans: [`docs/`](docs/) —
  [Week 1](docs/week-01.md) ·
  [Week 2](docs/week-02.md) ·
  [Week 3](docs/week-03.md) ·
  [Week 4](docs/week-04.md) ·
  [Week 5](docs/week-05.md) ·
  [Week 6](docs/week-06.md)
- Dataset provenance and licensing: [`docs/datasets.md`](docs/datasets.md)
- Research insights: [`insights/README.md`](insights/README.md)

## License

MIT (see `pyproject.toml`). Dataset licenses are separate and are documented in
[`docs/datasets.md`](docs/datasets.md).
