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

**Repository initialization only.** No research code, no experiments, no
results. What exists today:

| Area | Status |
|---|---|
| Repository skeleton, tooling, CI | Implemented |
| Reproducibility scaffolding (config, seeding, run provenance, artifact layout) | Implemented |
| Weekly implementation plans (`docs/`) | Drafted from `PLAN.md` |
| Insight log (`insights/`) | Templates only — **no findings recorded yet** |
| FoodSeg103 loading, segment→player mapping | Planned (Week 1) |
| CLIP value function, `shapiq.Game`, exact SV/k-SII/Möbius | Planned (Week 1) |
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
src/food_shapley/       Installable Python package
tests/unit/             Fast, isolated unit tests
tests/integration/      Multi-component tests (marked `integration`)
.github/workflows/      CI
data/                   Inputs — git-ignored, never written by the code
runs/                   Generated artifacts — git-ignored, one dir per run
```

## Prerequisites

- Python >= 3.11 — the exact version is pinned in `.python-version` (3.11) and
  `uv` installs it for you, so local and CI environments cannot drift
- [`uv`](https://docs.astral.sh/uv/) for environments, dependencies and locking
- For the actual experiments (not yet implemented): access to a SLURM cluster
  with GPUs, plus the FoodSeg103 dataset — see [`docs/datasets.md`](docs/datasets.md)

## Setup

```bash
uv sync --all-groups          # create .venv from uv.lock, exactly
uv run pre-commit install     # enable the pre-commit hooks
```

## Checks

The same four checks run locally, in pre-commit, and in CI:

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
