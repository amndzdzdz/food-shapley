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

*Added in task 1.2.* `huggingface-hub`, imported directly to fetch the
mirror's `id2label.json` at the pinned revision; and `torchvision` in the dev
group, because the geometry test asserts equality against its transforms.
Both ship no `py.typed` for the parts used, and `torchvision` has its own
narrow override.

### Status of task 1.2 (updated 2026-10-18)

Done and verified locally:

- `food_shapley.dataset` loads the pinned split into typed records (image plus
  a `uint8` label array) and carries the ingredient names. `id2label.json` is
  now downloaded with the split and hashed into the manifest, so the names are
  pinned to the same revision as the pixels.
- `food_shapley.segments` reproduces open_clip's geometry — shortest-side
  bicubic resize to 224 then centre crop — for image and mask together, with
  nearest-neighbour resampling for the mask. Equality with
  `torchvision.transforms.Resize` + `CenterCrop` is asserted byte-for-byte in
  the tests, so this is equality with CLIP's own preprocessing.
- Players are ingredient **classes**; small segments and any overflow past
  `max_players` merge into a `rest` player that cannot collide with
  FoodSeg103 class 103, "other ingredients".
- The non-ingredient region (plate, table, cutlery, a mug) is a **`scene`
  player**, not a blurred void and not a free rider. It is 39 % of the model's
  input on average, so neither alternative was neutral; as a player, `v(N)` is
  the unmodified photograph and `v(∅)` is a fully blurred frame at the same
  time. It is still **not an ingredient**: `PlayerSet.ingredient_indices`
  excludes it and every ingredient-level statistic must use those indices.
  This deviates from `PLAN.md`'s wording; see
  [W01-09](../insights/week-01.md) for the measurements and the trade-off.
- `configs/week01-clip-blur.toml` holds the Week-1 settings:
  `image_size = 224`, `max_players = 11`, `min_area_fraction = 0.005`,
  `blur_radius = 12.0`, `background_mode = "player"`.
- Measured over all 2,135 test images: 2,135 valid player sets, 0 failures,
  91,828 coalitions in total, 0 images capped, and 44 losing an ingredient to
  the centre crop. See [W01-05](../insights/week-01.md), W01-06 and W01-09.

### Status of the visual check (updated 2026-10-18)

**Done.** The blur removal operator (`food_shapley.masking`, the first half of
task 1.4) and a rendering entry point (`food_shapley.preview`) were written so
that the check could be performed, and 19 coalitions across 5 images were
rendered and inspected. All showed the region they claimed to; see
[W01-07](../insights/week-01.md). The images were chosen to cover the cases a
random sample would miss: no crop, heavy crop, a dropped ingredient, a `rest`
player, and aspect ratios from 0.56 to 1.78.

Two things this did **not** establish, both recorded in the insight:

- It is a 5-image spot check, not a sweep, and "correct" is a human judgement
  from a 224 px panel.
- It confirmed rather than refuted the silhouette confound: hidden ingredients
  stay legible as shapes, and at `blur_radius = 12` the empty coalition still
  reads as a recognisable kind of dish. Week 2's token-dropping operator is the
  designed response.

The efficiency check is still outstanding — it needs the value function
(task 1.3) — so task 1.6 remains blocked.

`blur_radius` and `background_mode` became required configuration fields in the
process, so the removal operator enters the run fingerprint.

### Status of task 1.3 (updated 2026-10-19)

Done and verified locally, including against the real checkpoint on CPU.

**The prompt set.** `food_shapley.prompts` fetches the 101 Food-101 class names
from `ethz/food101`'s *builder metadata* at a pinned revision — no images — and
writes them with a manifest and a content hash under `data/raw/food101/`. The
list is validated, not trusted: exactly 101 unique non-empty names, in order,
because the prompt set *is* the softmax denominator and `DishTarget.index`
indexes into it. Provenance and the license position are in
[`datasets.md`](datasets.md).

**Which dish is the target.** FoodSeg103 annotates ingredients and carries **no
dish label at all**, so there is no ground-truth dish to score. The target is
the model's own top-1 over the prompt set on the *unmasked* photograph, which
makes `v(N) − v(∅)` the amount of the model's own decision that the removal
operator can account for. Two consequences, both load-bearing:

- The target is **model-dependent**, so Milestone 2 must compare interaction
  *structure* across model families, not per-class values.
- Food-101's vocabulary is **borrowed** and does not cover what FoodSeg103
  photographs. Measured, not assumed: over 200 test images the model used 66 of
  101 classes and `pork_chop` took 13 % of images, including plates of chicken
  and rice.

**The value function.** `food_shapley.value.DishValueFunction` computes
`v(S) = log softmax(CLIP logits over the 101 prompts)[ŷ]`. Log-probability
rather than probability or raw logit because it is normalised (so "blurrier ⇒
less like every caption" divides out), unbounded below (so it keeps resolving
small coalitions instead of saturating at 0), and its differences are log-odds,
i.e. nats of evidence. The log-softmax is computed row-wise in **float64**,
since efficiency sums up to 2,048 terms per image.

**A silent misconfiguration was found and fixed.** `ViT-B-32/openai` — the
checkpoint name the configs previously carried — runs OpenAI's weights through
the wrong non-linearity: those weights were trained with QuickGELU, and since
open_clip 2.24 the plain `ViT-B-32` config uses standard GELU. open_clip emits
only a `UserWarning`. On 40 real test images the two architectures **disagree
on the top-1 dish for 15 of 40 (37.5 %)**, and the top-1 *is* the target, so
37 % of images would have decomposed the wrong dish. Three fixes: the configs
now say `open_clip/ViT-B-32-quickgelu/openai`; `ClipScorer.load` turns any
open_clip mismatch warning into a `RuntimeError`, so no future checkpoint can
repeat it silently; and an integration test asserts the bad name is refused.
This matters beyond Week 1 — Milestone 2 loads many checkpoints.

**New configuration fields.** `prompt_template` is required (and validated to
contain `{}`), and `model` must now be a fully explicit
`"<backend>/<architecture>/<pretrained>"` triple rather than an underspecified
name. The Week-1 fingerprint moved from `f6f33ec14b1f` to **`0c2f69fe4d3a`**.

**The sanity check was run on 200 real unmasked images** (CPU, real
checkpoint). Headlines: top-1 confidence median 0.580, min 0.167; the game is
non-degenerate, with `v(N) − v(∅)` positive on **all 200** (median +4.13 nats);
blur leakage is usually fine but has a tail (median `v(∅)` probability 0.0071,
*below* the 1/101 = 0.0099 uniform baseline, but above 0.10 on 5 of 200 and up
to 0.416); re-scoring is bit-identical. A **negative result**: top-1 confidence
does *not* separate apt targets from inapt ones, so the borrowed-vocabulary
problem cannot be filtered away and no threshold is applied anywhere. Full
numbers, including a first preliminary single-ingredient recovery signal and
its confounds, in [W01-10](../insights/week-01.md).

Not done, deliberately: the efficiency check still needs the `shapiq.Game`
(task 1.4), so task 1.6 remains blocked.

### Status of task 1.4 (updated 2026-10-20)

Done and verified locally, including against the real checkpoint on CPU.
`food_shapley.game.IngredientGame` is a `shapiq.Game` binding one image's
players to the blur operator and the value function: `v(S) = log p(target |
only S visible)`, target fixed once on the unmasked photograph so every
coalition of an image is scored against the same dish.

**The efficiency check now runs, and passes at zero tolerance.** Exact
`shapiq` Shapley values over the full powerset of three real images satisfy
`sum(SV) == v(N) − v(∅)` *bit for bit* — see
[W01-11](../insights/week-01.md). `v(N)` also reproduces the target's own
log-probability exactly, which is only true because the scene is a player
(W01-09) and is the sharpest available check that masking, geometry and
scoring agree. Task 1.6 is no longer blocked on the game; it is still blocked
on 1.1's cluster submission.

**Coalition ordering — the risk `PLAN.md` names for this step.** The defence
is to have no opinion: `shapiq` passes a boolean `(n_coalitions, n_players)`
matrix and gets one value per row, in row order, with no sorting, grouping,
deduplication or caching anywhere. This cannot be validated numerically, since
a game that consistently mislabels its coalitions satisfies efficiency
perfectly. It is validated instead by a value function that *recovers the
coalition from the rendering* and is checked against a closed form, pushed
through the real `ExactComputer`. The suite was mutation-tested with four
deliberate breakages; one of them exposed a weak fixture, which was fixed
rather than the assertion.

**One thing is not exact, and it is measured rather than tolerated.** CLIP's
float32 forward pass is not batch-invariant: the same coalition scored at
`chunk_size=1` and `chunk_size=32` differs by up to 8.1e-06 nats, which moves
an exact Shapley value by up to 2.9e-06 — about 1e-06 of the span being
decomposed. An integration test asserted exact agreement first and failed; the
tolerance was not widened until it passed, the drift was quantified and the
claim in the code was corrected. This is the justification for keeping
`chunk_size` a runtime argument rather than an `ExperimentConfig` field: a
throughput knob in the fingerprint would make two runs of the same experiment
look like different experiments.

Tests: 26 unit (no model) and 22 integration against the real checkpoint
(`-m gpu`), all passing; suite now 463 tests, 420 without `gpu`.

### Carried into tasks 1.5 and 1.6 from reading `shapiq`'s source

Four properties of `shapiq.Game` and `ExactComputer` that are not visible from
their documented interface, found while implementing 1.4 and recorded here
because each one fails *silently* in the task that will next touch it. None is
a problem today; `game.py` touches none of them.

- **`Game.save_values()` writes `.npz` rows in dict insertion order, not
  powerset order**, and round-trips coalitions through `tuple(sorted(...))`.
  Task 1.6 stores one `.npz` per image, so a reader must recover each
  coalition from the stored `coalitions` array and never assume row `i` is the
  `i`-th member of any enumeration. The same applies to comparing two runs'
  stored vectors element-wise.
- **`_precompute_flag` can be `True` while only a subset of coalitions is
  stored.** The flag makes `__call__` switch to dict lookup, so a partially
  written array task fails with a `KeyError` at the first missing coalition
  rather than at load time. This is the shape the "always check array
  completeness" rule exists for; completeness must be checked against
  `2^n_players`, not against the flag.
- **`ExactComputer` reads its baseline as `game_values[0]`**, i.e. it assumes
  the empty coalition is the first row of the powerset. True of
  `shapiq.utils.powerset`, and relied on by task 1.5; worth knowing if a
  future index is ever computed from a hand-built coalition matrix.
- **`ExactComputer('k-SII', order=2)` raises `IndexError` on a one-player
  game.** No image in this split has one player (2–11, W01-11), but task 1.5
  should clamp `order` to `min(order, n_players)` rather than depend on a
  dataset property.

### Status of task 1.5 (updated 2026-10-23)

Done and verified locally, including against the real checkpoint on CPU.
`food_shapley.indices.compute_indices` turns one `IngredientGame` into an
`ImageIndices`: the full coalition vector, exact SV, k-SII at
`config.interaction_order`, the Möbius transform at full order, the
efficiency residual, and the target. `save_indices` / `load_indices`
round-trip it as one `.npz` per image — the artifact task 1.6 will write.
`python -m food_shapley.indices --config <cfg> --image <n> --save <path>`
decomposes a single image end to end, which is what `PLAN.md` step 4 asks for
before the full run.

One `ExactComputer` serves all three index families; the game is evaluated
`2**n` times in total, not `3 × 2**n`, which a unit test asserts by counting
forward passes. Both items carried over from 1.4 were acted on: `order` is
clamped to `min(order, n_players)`, and completeness is checked against
`2**n_players` rather than against `_precompute_flag`.

**`interaction_order` is now an `ExperimentConfig` field**, so it enters the
run fingerprint. Both configs moved: `week01-clip-blur` `0c2f69fe4d3a` →
**`9171f04cd6a0`**, `smoke` `8ff0d4d5cbbc` → **`eb657bff090b`**. Fingerprints
recorded in earlier entries are historical and still correct for them.

**Stored blocks have a shape fixed by the configuration, never by the
numbers.** `shapiq` omits k-SII entries whose value is exactly zero — on a
4-player game with an inert player it returned 6 of the 11 subsets of size
≤ 2, missing that player's own singleton — and its `interaction_lookup` is
not in powerset order. Reading those rows directly would take a
value-dependent sample of the interactions with no symptom, so every block is
rebuilt against `powerset(range(n), max_size=order)`, omissions restored as
`0.0`, unexpected keys raising. See
[W01-12](../insights/week-01.md).

**The efficiency check now runs through the function the full run will
call**, with `DEFAULT_EFFICIENCY_TOLERANCE = 1e-9`. Residuals on the three
sample images: `0.0`, `−4.441e-16`, `0.0`. This corrects W01-11's "exactly
zero" generalisation — the summation order differs — and it is six orders of
magnitude inside the gate.

Tests: 41 unit (no model) and 24 integration against the real checkpoint
(`-m gpu`), all passing; suite now 534 tests. Mutation-tested in three
rounds; three survivors were found and killed, one of which is only
detectable on real images because a synthetic fixture normalises `v(∅)` to
exactly `0.0` while CLIP does not.

### Status of task 1.6 (updated 2026-10-23)

**The machinery is written and rehearsed locally. The run itself has not
happened, and cannot until 1.1's cluster submission closes.** Nothing below is
a result.

`food_shapley.run` mirrors `selftest.py`'s three subcommands — `prepare`,
`task`, `verify` — over the real split, and `slurm/run_array.slurm` submits it.
`slurm/prepare.sh` now also prepares the run directory for `run.py`, so a
login-node `prepare` covers both the self-test and the real run.

The research computation is one line per image (`compute_indices`); everything
else in the module is the bookkeeping `PLAN.md` says to budget the day for.
Five properties, each chosen against a specific way an array job fails
silently:

- **One output file per image, never one per task.** `values/image-00042.npz`
  is written by whichever task owns image 42. A partial failure is re-runnable
  per image rather than all-or-nothing.
- **A stored file does not depend on how the work was divided.** The image set
  is a function of the config alone, and each image's seed is derived from its
  own label, not from its task. Re-running at `--array=0-7` and at
  `--array=0-399` produces the same files — asserted in both the unit suite
  and, against the real checkpoint, bit for bit in
  `tests/integration/test_run_on_real_images.py`. That is what makes a
  resubmission a resumption rather than a second, subtly different experiment.
- **Writes are atomic.** A task killed by walltime mid-write would otherwise
  leave a truncated `.npz` that a resumed run skips as "already done". Each
  file is written under a temporary name in the same directory and renamed.
  The integration test truncates a real file and confirms `verify` reports it,
  so "unreadable" is a detectable state and not a hope.
- **A failing image fails loudly but alone.** It is recorded in the task's
  `logs/task-<id>.json`, the other images still run, and the task exits
  non-zero so SLURM marks it failed.
- **`verify` is not optional.** It re-opens every expected file, checks it
  holds the image its name claims, re-checks the stored efficiency residual
  against `1e-9`, writes `summary.json`, and exits non-zero on *any* problem.
  The "≥ 99 % coverage" criterion below is the bar for reporting a run; it is
  not a licence for this command to pass with images missing.

Shards are strided (`images[task_id::num_tasks]`), not contiguous: per-image
cost is `2**n_players` forward passes over a 2–11 player range (W01-11), so
contiguous blocks would hand one task a run of expensive images while the rest
of the array idled.

**`max_images` is now an `ExperimentConfig` field**, so a truncated rehearsal
gets its own run directory and cannot be mistaken for the full run. It takes a
*prefix*, not a sample, and is therefore never a basis for a reported
statistic. Both configs moved again: `week01-clip-blur` `9171f04cd6a0` →
**`deb3a76af736`**, `smoke` `eb657bff090b` → **`4f6b612d9de8`**
(`smoke.toml` sets `max_images = 20`). Fingerprints in earlier entries are
historical and still correct for them.

One real defect was found by the integration test and fixed: `np.load` leaks
the file descriptor it opened when the archive turns out to be corrupt, so
`load_indices` now opens the stream itself. `verify` is expected to survive
corrupt files and enumerate them, and would otherwise have leaked one
descriptor per corrupt file.

Tests: 60 unit (no model) and 7 integration against the real checkpoint
(`-m gpu`), all passing; suite now 602 tests. The unit suite was
mutation-tested with 15 targeted mutations of `run.py` — contiguous sharding,
an off-by-one shard, the task-id range check removed, `max_images` ignored,
finished images recomputed, the seed taken from the task instead of the image,
the wrong-image guard removed, a non-atomic write, the partial file left
behind, a failed task reporting success, the prepare check removed, and four
ways for `verify` to pass a broken run — and **caught all 15**.

Not done, and not claimable: the run. No `.npz` exists outside a temporary
test directory, and the four-hour walltime is still a guess.

**First submission, 2026-10-23 (W01-13).** Array 225119 was submitted with
`--array=0-1` and both tasks failed — the login-node `prepare` had been run
for `week01-clip-blur` and the array submitted with `CONFIG=smoke.toml`, so
`run_task`'s "prepare must have run" guard fired exactly as designed. No
image was decomposed. The failure is nonetheless informative, because it
happens *after* `ImageDecomposer` is constructed: the partition, `PATH`, `uv
run --frozen --no-sync`, the imports, the FoodSeg103 split and the CLIP
checkpoint all worked on the node. Three script changes came out of it —
`--array=0-27` (`MaxSubmit = 30` rejects 32 at submit time),
`HF_HUB_OFFLINE=1` (the node has network access and the load contacted the
Hub despite a warm cache), and a `device:` line printed at task start so that
an array silently running on CPU is visible in the first second rather than
at the walltime. Still unverified: that a GPU is allocated and used, and the
per-image runtime `-t` must be set from.

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

**Prompt sanity check: met, with a caveat the criterion as written could not
express (2026-10-19).** The criterion assumes a ground-truth dish; FoodSeg103
has none, so what was actually checked was whether the model's own top-1 is a
*plausible* reading of the photograph, over 200 unmasked test images, plus
three properties the value function needs in order to be usable at all:

- **Plausibility, by eye on a sample:** image 32 → `grilled_salmon`, 158 →
  `pulled_pork_sandwich`, 242 → `risotto`, all correct. These three are
  asserted as an integration test so a regression is caught, not re-eyeballed.
- **Confidence:** top-1 probability median 0.580 (mean 0.581, min 0.167);
  nothing below 0.10, 19 of 200 below 0.25. The model is not guessing.
- **Dynamic range:** `v(N) − v(∅)` is positive on 200 of 200 images, median
  +4.13 nats. The blur operator genuinely destroys the evidence, so the game is
  non-degenerate on every image checked.
- **Not met, and reported as such:** 66 of 101 classes are used and `pork_chop`
  fires on 13 % of images including non-pork dishes, and confidence does not
  flag those cases. The prompt set is the weakest link in the Week-1 design.

See [W01-10](../insights/week-01.md) for the measurements and their limits.

**Efficiency check: met on three images, at a tolerance of zero
(2026-10-20).** The criterion asks for the residual to be "below float
tolerance"; measured, `sum(SV) − (v(N) − v(∅))` is exactly `0.0` on images 32,
158 and 242, so the integration test asserts equality rather than closeness.
One qualification the criterion as written does not cover: this only holds if
both ends are read from the same powerset evaluation. Re-scoring the grand
coalition in a batch of its own shifts it by ~1e-06, because CLIP's float32
forward pass is not batch-invariant (W01-11). Still outstanding is the "100 %
of sampled images" part, which needs task 1.6's full run.

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
  needing a GPU with `@pytest.mark.gpu`; the pre-commit hook excludes it.

## Risks and alternatives

| Risk | Mitigation / alternative |
|---|---|
| Label 0 is background, not an ingredient (`PLAN.md`) | It *is* a player (`background_mode = "player"`, W01-09), and it is still never an ingredient: `Player.is_ingredient` and `PlayerSet.ingredient_indices` exclude it, asserted per image in the integration tests |
| Mask and image resized with different interpolation | Nearest-neighbour for masks, test alignment on a synthetic checkerboard |
| Coalition ordering mismatch with `shapiq` | Test against `shapiq`'s own indexing; fail loudly on mismatch |
| Images with too many segments make exact computation expensive | Merge small segments into "rest" to cap player count (`max_players` in the config); record how many images were capped |
| Blur does not actually remove information (CLIP still "sees" the shape) | Known limitation; Week 2's token-dropping operator is the designed response |
| Job array partially fails | Per-image output files + an explicit re-run list; never silently overwrite completed outputs. Check completeness against `2^n_players`, not against `shapiq`'s `_precompute_flag`, which can be `True` with coalitions missing |
| A stored `.npz` is read back with the wrong coalition→row convention | `shapiq` writes rows in dict insertion order, not powerset order; always recover the coalition from the stored `coalitions` array rather than from an index |

## Definition of done

- [ ] A toy SLURM array job has run end-to-end on a node, and
      `selftest verify` reported every task complete.
- [x] FoodSeg103 loads and converts to players, with tests. *(2026-10-18;
      `dataset.py`, `segments.py`, 69 unit + 41 integration tests. The scene
      is a player — W01-09.)*
- [x] The value function is defined, documented, and sanity-checked on real
      images. *(2026-10-19; `prompts.py`, `scoring.py`, `value.py`, 98 unit +
      21 real-checkpoint integration tests. Sanity-checked on 200 unmasked
      images — W01-10. A QuickGELU checkpoint mismatch affecting the target on
      37 % of images was found and guarded against in the process.)*
- [x] The `shapiq.Game` subclass produces exact SV, k-SII and Möbius values.
      *(2026-10-23; `game.py` + `indices.py`, 67 unit + 46 real-checkpoint
      integration tests. All three index families come from one
      `ExactComputer` evaluation and are stored in a shape fixed by the
      configuration, because `shapiq` drops zero-valued k-SII entries —
      W01-12.)*
- [ ] Efficiency and visual checks both pass and are documented. *(Visual check
      done 2026-10-18, W01-07. Efficiency passes on three real images — W01-11,
      residuals corrected in W01-12 to `0.0`, `−4.441e-16`, `0.0` against a
      `1e-9` gate — and now runs inside `compute_indices`. The criterion's
      "100 % of sampled images" needs task 1.6's full run, so this stays
      unticked.)*
- [ ] The full test-split run has completed, with failures enumerated.
- [ ] `ruff check`, `ruff format --check`, `mypy`, `pytest` all pass.
- [ ] New dependencies are justified in the commit message and locked in `uv.lock`.
- [ ] `insights/week-01.md` records what was actually observed, including
      negative or inconclusive outcomes.
