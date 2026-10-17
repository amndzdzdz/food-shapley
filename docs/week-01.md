# Week 1 — From scaffolding to a full CLIP Shapley run

**`PLAN.md` steps:** 0–5 · **Budget:** 5.0 days · **Milestone:** 1 (workshop)
**Depends on:** nothing (first implementation week; repository skeleton exists)

## Objectives

Get from an empty pipeline to exact Shapley values and interactions for every
test image, computed by CLIP over FoodSeg103 ingredient segments.

### Research questions addressed

- RQ1 (setup): Can a dish prediction be decomposed into per-ingredient main
  effects and interactions at all, at the scale of the FoodSeg103 test split?

This week does not answer RQ1 substantively — it produces the values that
Week 2's analysis interprets.

## Implementation tasks

| # | `PLAN.md` | Task | Days |
|---|---|---|---|
| 1.1 | 0 | Cluster scaffolding: environment on the SLURM nodes, `shapiq` + CLIP installed from the lockfile, FoodSeg103 downloaded, one toy array job submitted end-to-end | 0.5 |
| 1.2 | 1 | FoodSeg103 loading; masks → coalition players (merge small segments, add a "rest" player, align masks with the resized image) | 1.0 |
| 1.3 | 2 | Dish prompt set (e.g. the 101 Food-101 class names), CLIP setup, value function definition: log-probability of the target dish against all dish prompts | 0.5 |
| 1.4 | 3 | The game: hide segments by blurring, batch the ~128 coalitions per image, return the dish score; implemented as a `shapiq.Game` subclass | 1.0 |
| 1.5 | 4 | Exact SV, k-SII and Möbius via `shapiq.ExactComputer`, plus the two validity checks below | 1.0 |
| 1.6 | 5 | Scale out: SLURM job array over the ~2,000–7,118 test images, one `.npz` of game values per image | 1.0 |

### Status of task 1.1 (updated 2026-10-17)

Done and verified locally:

- Research dependencies added and locked: `shapiq`, `torch`, `open-clip-torch`,
  `numpy`, `datasets`, `pillow`. Python floor raised to 3.12 (shapiq ≥ 1.5
  requires it).
- `food_shapley.download` fetches and verifies FoodSeg103 from a
  revision-pinned mirror, writing a manifest with a content hash. The real
  test split (2,135 images) has been downloaded and verified. See
  [`datasets.md`](datasets.md).
- `food_shapley.envcheck` reports the research stack, GPU and dataset
  integrity from inside a job, downloading nothing.
- `slurm/prepare.sh` now also downloads the dataset; `slurm/toy_job.slurm` runs
  `envcheck --require-gpu --require-dataset`.

**Not done: nothing has been submitted to a cluster.** The scripts pass
shellcheck and have been run locally as far as a GPU-less machine allows —
`toy_job.slurm` reaches the `--require-gpu` gate and stops there, as designed.
Partition, walltime, shared-scratch `.venv` and `uv run --no-sync` on an
offline node all remain assumptions. Submitting them is the remaining part of
task 1.1; record the outcome in `insights/week-01.md`.

Dependencies to add to `pyproject.toml` this week (with the lockfile updated and
committed): `shapiq`, a CLIP implementation (`open_clip_torch` or
`transformers`), `torch`, `numpy`, and whatever is needed for FoodSeg103 mask
loading. Add them when the first code imports them, not before. If a package
ships no type information, add a narrow `[[tool.mypy.overrides]]` entry naming
that module only.

*Done in task 1.1.* `open_clip_torch` was chosen over `transformers`: the
configuration already names open_clip checkpoints, and Week 2's token-dropping
removal operator (`PLAN.md` step 6) needs access to ViT internals. `datasets`
is the only one without a `py.typed` marker and has a narrow override.

## Experimental design and hypotheses

The value function is `v(S) = log p(target dish | image with only segments S
visible)` under the blur removal operator, with `p` taken over the full dish
prompt set. `v(∅)` is the all-blurred image.

No scientific hypothesis is tested this week. The two checks below are
*validity* checks on the implementation, not findings:

1. **Efficiency.** For every image, `sum(SV) == v(N) - v(∅)` within numerical
   tolerance. A violation means the game or the computation is wrong.
2. **Visual.** Render a handful of masked coalitions and confirm that "only bun
   visible" really shows only the bun. This catches mask/image misalignment,
   which the efficiency check cannot.

Per `PLAN.md`: catching a masking bug here is far cheaper than after the full
run. Do not start task 1.6 until both checks pass.

## Data, resources, dependencies

- **FoodSeg103** test split (~2,000–7,118 images, depending on the subset used).
  See [`datasets.md`](datasets.md) for provenance and licensing.
- A CLIP checkpoint, pinned by name in the run configuration.
- SLURM cluster with GPUs. Per `PLAN.md`, compute is not the bottleneck; task
  1.6 is an orchestration day, not a GPU-wait day.

## Expected outputs

- `src/food_shapley/` modules for dataset loading, player construction, the
  value function, and the `shapiq.Game` subclass.
- `configs/` entry for the full Week-1 run (in addition to `smoke.toml`).
- One `.npz` per image under `runs/<name>-<fingerprint>/values/`, containing the
  coalition values and the exact SV / k-SII / Möbius indices.
- A record in [`insights/week-01.md`](../insights/week-01.md) of anything
  surprising — especially value-function sanity checks that fail.

## Evaluation metrics and success criteria

| Criterion | Target |
|---|---|
| Efficiency check | `\|sum(SV) − (v(N) − v(∅))\|` below float tolerance on 100 % of sampled images |
| Visual check | Manual inspection of ≥ 10 coalitions across ≥ 3 images, all correct |
| Prompt sanity check | CLIP ranks the ground-truth dish plausibly on a few obvious unmasked images |
| Coverage of the full run | ≥ 99 % of targeted test images produce a readable `.npz`; failures enumerated, not ignored |

The prompt sanity check is qualitative on purpose — `PLAN.md` says to sanity
check the value function "on a few obvious images first". Record what was
checked, not just that it was.

## Testing and validation

- Unit: segment merging, the "rest" player, mask/image alignment after resize,
  player-count bounds, coalition encoding/decoding.
- Unit: the coalition ordering the game produces matches what `shapiq` expects
  (`PLAN.md` flags this explicitly as an easy thing to get wrong).
- Unit: efficiency on a tiny synthetic game with a known closed-form answer —
  no CLIP, no GPU, fully deterministic.
- Integration (marked `integration`): the whole pipeline on 1–2 small fixture
  images with a stubbed or tiny model, asserting shapes, finiteness and
  efficiency.
- The real model must never be downloaded inside the test suite. Mark anything
  needing a GPU with `@pytest.mark.gpu`; CI excludes it.

## Risks and alternatives

| Risk | Mitigation / alternative |
|---|---|
| Label 0 is background, not a player (`PLAN.md`) | Explicit test asserting label 0 is excluded from the player set |
| Mask and image resized with different interpolation | Nearest-neighbour for masks, test alignment on a synthetic checkerboard |
| Coalition ordering mismatch with `shapiq` | Test against `shapiq`'s own indexing; fail loudly on mismatch |
| Images with too many segments make exact computation expensive | Merge small segments into "rest" to cap player count (`max_players` in the config); record how many images were capped |
| Blur does not actually remove information (CLIP still "sees" the shape) | Known limitation; Week 2's token-dropping operator is the designed response |
| Job array partially fails | Per-image output files + an explicit re-run list; never silently overwrite completed outputs |

## Definition of done

- [ ] A toy SLURM array job has run end-to-end on a node, and
      `selftest verify` reported every task complete.
- [ ] FoodSeg103 loads and converts to players, with tests.
- [ ] The value function is defined, documented, and sanity-checked on real images.
- [ ] The `shapiq.Game` subclass produces exact SV, k-SII and Möbius values.
- [ ] Efficiency and visual checks both pass and are documented.
- [ ] The full test-split run has completed, with failures enumerated.
- [ ] `ruff check`, `ruff format --check`, `mypy`, `pytest` all pass.
- [ ] New dependencies are justified in the commit message and locked in `uv.lock`.
- [ ] `insights/week-01.md` records what was actually observed, including
      negative or inconclusive outcomes.
