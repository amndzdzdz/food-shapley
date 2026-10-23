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

**Infrastructure, data acquisition, the segment→player mapping, the CLIP value
function, the coalition game, the exact indices and the full-split job array.
No experiment has been run and no result exists — exact Shapley values, k-SII
and Möbius coefficients have been computed on three images, as a correctness
check.** Week 1 tasks 1.2, 1.3, 1.4 and 1.5 are done; task 1.6's machinery is
written and rehearsed locally but has never been submitted, because task 1.1
is still open on its cluster-submission step. What exists today:

| Area | Status |
|---|---|
| Repository skeleton and tooling | Implemented |
| Reproducibility scaffolding (config, seeding, run provenance, artifact layout) | Implemented |
| Research dependencies (`shapiq`, `torch`, `open_clip`, `datasets`) | Installed and locked |
| FoodSeg103 download + integrity verification (`food_shapley.download`) | Implemented; test split downloaded and verified |
| Environment / GPU / dataset / prompt-set check (`food_shapley.envcheck`) | Implemented |
| SLURM toy job + array scaffolding (`slurm/`) | Written and locally simulated — **never submitted to a cluster** |
| Weekly implementation plans (`docs/`) | Drafted from `PLAN.md` |
| Insight log (`insights/`) | Data, players, masking, the value function on 200 unmasked images, and the game's and indices' numerics on 3 — **no experimental result yet** |
| FoodSeg103 loading (`food_shapley.dataset`) | Implemented |
| Segment→player mapping, CLIP geometry (`food_shapley.segments`) | Implemented; alignment verified against torchvision **and** by eye. The scene is a player, never an ingredient ([W01-09](insights/week-01.md)) |
| Blur removal operator + coalition renderer (`masking.py`, `preview.py`) | Implemented; visual check passed on 19 coalitions / 5 images |
| Dish prompt set (`food_shapley.prompts`) | Implemented; 101 Food-101 names at a pinned revision, hashed. **Names only, no images** |
| CLIP loading + zero-shot scoring (`food_shapley.scoring`) | Implemented; checkpoint fully pinned and weight-hashed, architecture mismatches are a hard error ([W01-10](insights/week-01.md)) |
| Value function (`food_shapley.value`) | Implemented; `v(S) = log p(target dish)`, sanity-checked on 200 real unmasked images |
| Coalition game (`food_shapley.game`) | Implemented; efficiency holds on 3 real images ([W01-11](insights/week-01.md)) |
| Exact SV / k-SII / Möbius indices (`food_shapley.indices`) | Implemented; one `ExactComputer` evaluation per image, stored in a shape fixed by the config because `shapiq` drops zero-valued k-SII entries ([W01-12](insights/week-01.md)) |
| Full-split job array (`food_shapley.run`, `slurm/run_array.slurm`) | Written and rehearsed on 3 real images — **never submitted; no run exists** |
| Second removal operator, analysis | Planned (Weeks 1–2) |
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

The dish prompt set — the 101 Food-101 class names, which *are* the softmax
denominator of the value function — is fetched the same way, from dataset
**metadata only**; none of Food-101's images are downloaded:

```bash
uv run python -m food_shapley.prompts            # ~12 kB, idempotent
uv run python -m food_shapley.prompts --verify-only
```

To check that a machine can actually run the pipeline — imports, GPU, dataset
and prompt-set integrity, downloading nothing:

```bash
uv run python -m food_shapley.envcheck --dataset-split test --require-prompts
```

## The value function

`v(S) = log p(target dish | image with only the segments in S visible)`, with
`p` a softmax over all 101 dish prompts. FoodSeg103 has **no dish annotation**,
so the target is the model's own top-1 on the *unmasked* photograph: the
decomposition explains the model's prediction, which is not always the dish.
That limitation is measured rather than hidden — see
[W01-10](insights/week-01.md).

The checkpoint is named in full in the config, as
`<backend>/<architecture>/<pretrained>`, and loading it prints a provenance
record including a sha256 over the weights:

```bash
uv run python -m food_shapley.scoring --config configs/week01-clip-blur.toml
```

Run this once on a networked machine before submitting anything: it warms the
weight cache under `HF_HOME` (`slurm/prepare.sh` step 5) and it fails loudly if
the architecture and the pretrained tag disagree. That guard is not theoretical
— `ViT-B-32/openai`, which open_clip accepts with only a warning, runs OpenAI's
QuickGELU weights through standard GELU and changed the predicted dish on 15 of
40 test images.

## The game

`food_shapley.game.IngredientGame` is the `shapiq.Game` that joins the three
pieces: one image's players, the blur removal operator, and the value
function. One instance is one image and one target dish, and the model is
loaded once per process rather than once per game.

Under `background_mode = "player"` the grand coalition's rendering *is* the
photograph, so `v(N)` reproduces the target's own log-probability bit for bit,
and `v(∅)` is a fully blurred frame. The game is normalised, so `shapiq`
reports `v(S) − v(∅)` and the empty coalition is worth exactly 0.

Exact Shapley values over the full powerset of three real images satisfy
`sum(SV) == v(N) − v(∅)` to within `4.5e-16`, against a `1e-9` gate. Two
things are worth knowing before reading or extending the code:

- The game has no opinion about coalition ordering. `shapiq` passes a boolean
  `(n_coalitions, n_players)` matrix and gets one value per row, in row order.
  A game that *consistently* mislabels its coalitions passes the efficiency
  check perfectly, so that property is tested by decoding the coalition back
  out of the rendering, not by checking numbers.
- CLIP's float32 forward pass is not batch-invariant. Rebatching moves a
  coalition value by up to ~8e-06 nats and a Shapley value by up to ~3e-06,
  which is why `chunk_size` is a runtime argument rather than a config field,
  and why an efficiency check must read `v(N)` and `v(∅)` from the same
  evaluation. Measured in [W01-11](insights/week-01.md).

## Exact indices

`food_shapley.indices.compute_indices` turns one game into one decomposition:
the full coalition vector, exact Shapley values, k-SII at
`interaction_order`, the Möbius transform at full order, and the efficiency
residual. All three index families come from a single `ExactComputer`
evaluation, so the Möbius transform is free once the coalitions are scored.

```bash
uv run python -m food_shapley.indices --config configs/week01-clip-blur.toml \
    --image 32 --save runs/one-image.npz
```

`save_indices` / `load_indices` round-trip it as one `.npz` per image — the
artifact the full run will write, one per array task.

**A stored block's shape depends on the configuration, never on the
numbers.** `shapiq` omits k-SII entries whose value is exactly zero: on a
4-player game with an inert player it returns 6 of the 11 subsets of size
≤ 2, missing that player's own singleton, and its `interaction_lookup` is not
in powerset order. Reading those rows directly would take a value-dependent
sample of the interactions with no symptom, so every block is rebuilt against
`powerset(range(n), max_size=order)` with omissions restored as `0.0`. An
analysis may therefore index a stored file positionally. Measured in
[W01-12](insights/week-01.md).

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

### The full run

Same three subcommands, over the real split:

```bash
bash slurm/prepare.sh configs/week01-clip-blur.toml
CONFIG=configs/week01-clip-blur.toml sbatch slurm/run_array.slurm

uv run --frozen python -m food_shapley.run \
    --config configs/week01-clip-blur.toml verify
```

The `verify` step is not optional. An array that silently drops 3 % of its
tasks looks exactly like one that succeeded, so `verify` re-opens every
expected file, checks it holds the image its name claims, re-checks the stored
efficiency residual, writes `summary.json`, and exits non-zero on any problem.

One `.npz` is written per **image**, not per task, so a partial failure is
re-runnable at image granularity: resubmitting skips what is already on disk
and recomputes only the rest. The array size in `#SBATCH --array` is therefore
a throughput choice and changes no stored number — each image's seed and
filename are derived from the image itself, never from the task. Writes are
atomic, so a task killed by walltime leaves no truncated file for a resumed
run to mistake for finished work.

`max_images` in the config truncates the run to a prefix of the split for a
rehearsal. It is a config field rather than a flag so that the rehearsal gets
its own run directory; a prefix is not a sample, and is never a basis for a
reported statistic.

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
