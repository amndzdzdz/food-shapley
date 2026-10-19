# Week 1 insights — Pipeline, value function, first full run

Plan: [docs/week-01.md](../docs/week-01.md) · Entry format:
[insights/README.md](README.md)

**Nine entries, all about the input data, how it becomes coalition players,
and how a coalition is rendered. No experiment has been run and no model has
been evaluated.** W01-03 partially contradicts W01-02 and W01-09 partially
supersedes W01-06; read both pairs.

## Questions this week's entries should answer

Derived from the Week 1 plan; each needs a dated entry once evidence exists.

- Did the toy SLURM job and array run on the cluster as written? Which
  assumptions in `slurm/README.md` turned out to be wrong — partition,
  walltime, `uv run --no-sync` on an offline node, scratch paths? This is the
  first thing to record, because every later run inherits these settings.
  **Still open: nothing has been submitted.**
- Does the efficiency check (`sum(SV) == v(N) − v(∅)`) hold, and at what
  numerical tolerance?
- Does the visual check confirm that masked coalitions show what they claim to?
  Which coalitions were inspected? **Answered by W01-07: 19 coalitions over 5
  images, all correct, listed there by name.**
- Is the log-probability value function sensible on obvious unmasked images?
  What exactly was checked, and what was the outcome?
- How many images exceeded `max_players` and had segments merged into "rest"?
  **Answered by W01-04 and confirmed through the real pipeline by W01-06:
  zero capped at `max_players = 10`; 3 images get a "rest" player from the
  area threshold.** The run must still report the figure it actually observed.
- What fraction of the test split produced usable output, and why did the rest
  fail?

Record failures and surprises here as readily as successes — a failed masking
check is one of the most valuable entries this week can contain.

---

## W01-09 — The non-ingredient region is 39 % of CLIP's input, so it is made a player rather than blurred away or left free

- **Date:** 2026-10-18
- **Status:** validated (the measurements are exhaustive over the test split;
  the design choice they justify is a decision, not a result)
- **Plan reference:** docs/week-01.md, task 1.2. **Deviates from `PLAN.md`**,
  which says only "label 0 is background, not a player" — see *Relation to the
  plan* below.
- **Run / artifact:** `configs/week01-clip-blur.toml`, fingerprint
  `f6f33ec14b1f` (was `2ad46618605e`) · commit `a2eb35c` plus the uncommitted
  working tree · sheets rendered to `/tmp/scene-check/` with
  `python -m food_shapley.preview --config configs/week01-clip-blur.toml
  --index 656 --index 728 --index 1510 --scale 2 --columns 4`

### Observation

Measured over all 2,135 test images, after CLIP's resize and centre crop, for
the pixels that no ingredient mask claims:

- The non-ingredient region is **39.5 % of the model's input on average**
  (median 38.0 %, min 0.06 %, max 94.6 %).
- It is the **majority of the frame in 631 images (29.6 %)**.
- It is **not empty in 2,134 of 2,135 images**; exactly one image (index 2005)
  is covered by food edge to edge.
- It is *not* a neutral void. On image 1510 the region contains the coffee
  mug, its saucer, the granite counter and the window; the only ingredient
  pixels are the toast and the coffee *inside* the mug.

With the region promoted to a `scene` player under `max_players = 11`:

| players (incl. `scene`) | images |
|---|---|
| 2 | 7 |
| 3 | 503 |
| 4 | 662 |
| 5 | 460 |
| 6 | 314 |
| 7 | 111 |
| 8 | 56 |
| 9 | 19 |
| 10 | 2 |
| 11 | 1 |

- 2,135 / 2,135 images built, **zero failures, zero images capped**.
- **Σ 2^players = 91,828 coalitions** for the split, against 45,916 before
  (W01-06). Exactly 2× minus the one scene-less image.
- Ingredient players alone still peak at 10, so `max_players` had to go from
  10 to 11; at 10 the cap would have started folding real ingredients into
  `rest` on one image.

### Context and motivation

Noticed by eye while reviewing the W01-07 sheets: every rendered coalition
blurred the plate, the table and a visible mug, in every coalition including
the grand one. That makes `v(N)` an image CLIP never sees, so the
decomposition would explain a prediction the model never made. The obvious
repair — keep the region sharp everywhere — is no better: the region would be
present in every coalition and attributed in none, a free rider carrying 39 %
of the pixels.

### Setup

`food_shapley.segments.build_players(..., background_player=True)`, driven by
the new `background_mode` config field. Deterministic, no model, no GPU.
`ExperimentConfig` now carries one three-valued field rather than a boolean, so
the two broken options and the chosen one are mutually exclusive by
construction:

| `background_mode` | `v(∅)` | `v(N)` | scene attributed? |
|---|---|---|---|
| `hidden` | fully blurred | **not the photograph** | no |
| `visible` | **scene sharp** | the photograph | **no — free rider** |
| `player` | fully blurred | the photograph | **yes** |

### Evidence

- The distribution table above, from a full-split sweep.
- `tests/integration/test_players_on_real_images.py::test_the_grand_coalition_is_the_photograph`
  asserts `render(players, N) == players.image` byte-for-byte and
  `render(players, ∅) == blurred(image)` on 8 real images under the shipped
  config. This is the property the mode exists for, so it is checked against
  real masks, not only synthetic ones.
- `tests/unit/test_segments.py` adds 16 scene-player cases (ordering,
  disjointness, cap accounting, `ingredient_indices`, the empty-leftover case).
- Rendered sheets for images 656, 728 and 1510. In each, panel 1 ("original
  input") and the last panel ("all (v of N)") are pixel-identical, the "none"
  panel is uniformly blurred, and the new "only scene" panel shows the
  surroundings sharp with a blurred food-shaped hole.
- `uv run ruff check . && uv run ruff format --check . && uv run mypy &&
  uv run pytest -m "not gpu"` — all pass, 289 tests.

### Interpretation

*Interpretation, not measurement:* a 39 % share means the choice is not a
detail. Blurring the scene in `v(N)` would bias every Shapley value by an
unknown amount in an unknown direction, because the reference point of the
whole decomposition would be an out-of-distribution image. Keeping it sharp
would bias them in a knowable direction — downward, since the scene's
contribution gets silently folded into the baseline — but would also make the
mug on image 1510 unattributable, and a mug is exactly the kind of contextual
cue the compositionality question is about. Making it a player is the only
option of the three under which the question "how much of the prediction comes
from context rather than from food?" is answerable at all.

### Limitations and alternative explanations

- **This deviates from `PLAN.md`'s "label 0 is background, not a player".**
  The plan's intent — the scene must never be counted as an ingredient — is
  preserved and now enforced by `Player.is_ingredient` and
  `PlayerSet.ingredient_indices`, which the integration tests check per image.
  What changed is that it is a player *and* not an ingredient. Every
  ingredient-level headline number must be computed over `ingredient_indices`;
  computing it over `players` would be the exact mistake the plan warns about,
  now one attribute away instead of impossible.
- **Two images get a near-degenerate scene player.** Index 310's scene is 28
  pixels (0.06 % of the frame) and index 1884's is 158 (0.32 %) — both below
  `min_area_fraction = 0.005`, which the scene bypasses because it is not
  subject to area-based merging. Their Shapley values will be near zero and
  will slightly dilute any "mean scene contribution" statistic. 2 of 2,134 is
  not worth a special case, but the number should be quoted if that statistic
  is reported.
- **The cost doubled** to 91,828 coalitions. Still small, but the headroom
  argument in W01-06 is now half as large.
- **The silhouette confound is unchanged and arguably more visible.** The
  "only scene" panels show a sharp plate with a blurred food-shaped hole: the
  shape of every ingredient survives in every coalition. That is a property of
  blur as a removal operator, not of this decision, and it is why Week 2's
  token-drop operator exists.
- No model has been run. Nothing here says the scene *will* carry a large
  Shapley value — only that it now *can*, and that the question is measurable.

### Relation to earlier entries

- **W01-06 is partially superseded.** Its coalition total (45,916) and its
  `max_players = 10` setting were measured without a scene player; the current
  configuration gives 91,828 and `max_players = 11`. Its per-image failure
  rate (0 %) and its "player = ingredient class, not connected component"
  decision still stand.
- **W01-04**'s "no test image has more than 10 ingredients" still holds — it
  is a statement about ingredients, and it is why 11 is the right cap.

### Follow-up questions and next steps

1. Once CLIP is wired in (task 1.4), report the scene's mean Shapley value and
   its mean |interaction| with ingredients as a first-class result. If it is
   large, context is doing the work and that is a finding; if it is near zero,
   the earlier `hidden`/`visible` debate was moot and that is worth saying too.
2. Check whether the scene's value correlates with its area fraction. A near
   perfect correlation would suggest the model is responding to how much of
   the frame is blurred rather than to what is in it — a confound the
   token-drop operator would have to settle.
3. Re-measure under `background_mode = "hidden"` on a subsample once a model
   exists, as a sensitivity check on this decision. The config field makes
   that a one-line change and a new fingerprint.

---

## W01-08 — FoodSeg103 ingredient *names* are sometimes wrong even where the *mask* is right

- **Date:** 2026-10-18
- **Status:** preliminary (two images, found incidentally during W01-07)
- **Plan reference:** docs/week-01.md, task 1.2 / the visual check in task 1.5
- **Run / artifact:** `runs/visual-check/2ad46618605e/image-01569.png` and
  `image-00386.png`; commit `a2eb35c` plus the uncommitted working tree;
  dataset manifest `4d2b3127f607…`

### Observation

While inspecting rendered coalitions (W01-07), two segments were correctly
delineated but carry an ingredient name that does not describe the object:

| Image | Class id / name | What the pixels actually show |
|---|---|---|
| 1569 | `ice cream` | a scoop of macaroni or potato salad |
| 386 | `pie` | a pancake |

In both cases the mask outline follows the object precisely; only the class
label is wrong. Two further names were *correct but surprising* and should not
be confused with these: image 1569's `pork` is the bacon wrapped around the
chicken, and image 656's `chocolate` is a second dessert at the edge of frame,
not the chocolate glaze on the main cake (the glaze is annotated `cake`).

### Context and motivation

Not sought. The visual check looks at *where* a mask is; these were noticed
because a correctly placed region had an implausible caption.

### Setup

As W01-07: `configs/week01-clip-blur.toml`, FoodSeg103 test split at manifest
`4d2b3127f607…`, names from the `id2label.json` pinned at the same revision.

### Evidence

`uv run python -m food_shapley.preview --config configs/week01-clip-blur.toml
--index 1569 --index 386`, then look at the `only ice cream` and `only pie`
panels.

### Interpretation

*Interpretation, not measurement.* This is a property of the dataset, not of
the pipeline. The geometry is unaffected: a Shapley value computed over the
`ice cream` segment of image 1569 is a perfectly valid statement about *that
region of pixels*. What breaks is the **report**: "ice cream contributes 0.4 to
the dish score" would be a false sentence about a potato salad, and it is
exactly the kind of sentence this project intends to write.

### Limitations and alternative explanations

- Two images is anecdote, not a rate. The frequency is unmeasured.
- FoodSeg103 class names are coarse by design (`pork` legitimately covers
  bacon, `pie` may be intended as a broad pastry class), so some of these may
  be vocabulary breadth rather than error. `ice cream` for potato salad is hard
  to read that way; `pie` for a pancake is arguable.
- The judgement is mine, from a 224 px thumbnail, with no ground truth beyond
  the photograph.

### Implications

- Per-ingredient claims in the paper must be attached to a **class id**, and
  any named example shown in a figure must be eyeballed first.
- Aggregate statistics ("how often is a single ingredient sufficient?") are
  robust to this, because they never read the name.
- If Week 2 or 3 wants to report per-ingredient rankings by name, the error
  rate has to be estimated first — otherwise a headline like "cheese is the
  most telltale ingredient" rests on unverified labels.

### Follow-up questions and next steps

- Estimate the rate: sample ~50 segments stratified by class and score
  name-against-pixels. Cheap, and it bounds what the naming can support.
- Check whether the mislabels concentrate in a few classes; if so, an
  exclusion list is simpler than a correction.

---

## W01-07 — Visual check passes: 19 rendered coalitions across 5 images all show the ingredient they name

- **Date:** 2026-10-18
- **Status:** validated (for the geometry; the sample is small and hand-picked)
- **Plan reference:** docs/week-01.md, task 1.5, validity check 2 ("Visual");
  success criterion "≥ 10 coalitions across ≥ 3 images, all correct"
- **Run / artifact:** `runs/visual-check/2ad46618605e/`, config
  `configs/week01-clip-blur.toml` (fingerprint `2ad46618605e`), commit
  `a2eb35c` plus the uncommitted working tree adding
  `src/food_shapley/masking.py` and `src/food_shapley/preview.py`; dataset
  manifest `4d2b3127f607…`

### Observation

19 coalitions were rendered and inspected across 5 images: 16 singletons
("only X visible"), the empty coalition on 2 images, and the grand coalition on
1. **Every one showed the region its caption named, with the sharp/blurred
boundary following the object outline.** No misalignment, no off-by-one shift,
no stale mask.

| Image | Size (w×h) | Case it covers | Coalitions inspected |
|---|---|---|---|
| 656 | 256×256 | square, no crop at all | `chocolate`, `cake`, `strawberry` |
| 1569 | 577×1024 | portrait crop, 6 players | `ice cream`, `pork`, `chicken duck`, `bread`, `carrot`, `broccoli`, empty, grand coalition |
| 377 | 768×1024 | portrait crop; `lemon` dropped by the crop | `steak`, `shrimp`, `asparagus` |
| 386 | 1820×1024 | widest aspect ratio; has a `rest` player | `pie`, `other ingredients`, `rest` |
| 69 | 307×204 | smallest tested; `juice` dropped by the crop | empty, `cheese butter` |

Aspect ratios covered: 0.56, 0.75, 1.00, 1.50, 1.78.

Specific things confirmed rather than assumed:

- **Crop path is correct.** The three images where the centre crop discards
  most of the frame (386 and 1569 lose ≈ 44 % of their long side, 377 ≈ 25 %)
  place every segment correctly in the surviving 224 px window.
- **`rest` is visually a scatter of specks.** On image 386 it is the union of
  sub-threshold `ice cream` and `milk` segments (0.90 % of the frame combined)
  and renders as a handful of disconnected slivers between strawberry slices.
- **A suspicious-looking result was a correct one.** Image 656's `chocolate`
  player is a small sharp patch in the top-left corner, nowhere near the
  chocolate-glazed cake. Checking the raw mask showed the annotation is at
  x∈[0,27], y∈[0,14] of the 256×256 original, and zooming the original shows a
  second dessert at the edge of frame there. The renderer is faithful; the
  surprise was in the data.

A blur-radius sweep (4, 8, 12, 20, 32 px) was rendered on the empty coalition
of images 69 and 1569. At 4–8 px the dish remains plainly identifiable — the
biscuit, carrots and broccoli of image 1569 are individually recognisable. At
20–32 px the frame collapses to a near-uniform colour field. `blur_radius` was
set to **12.0** on this evidence.

### Context and motivation

`PLAN.md` requires both an efficiency check and a visual check before the full
run, and is explicit that the two catch different faults: efficiency is a
statement about the *game*, and a game built on mis-registered masks is
perfectly self-consistent. Shift every mask ten pixels and efficiency still
holds to machine precision while every attribution is wrong. W01-05 established
byte-equality with `torchvision`'s transforms, but that is equality between two
pieces of code; it says nothing about whether label 31 is the bun.

### Setup

- Config `configs/week01-clip-blur.toml`: `image_size = 224`,
  `max_players = 10`, `min_area_fraction = 0.005`, `blur_radius = 12.0`,
  `keep_background = false`.
- Removal operator: `food_shapley.masking.BlurRemoval`, whole-image Gaussian
  blur with the coalition's pixels pasted back sharp.
- FoodSeg103 test split, manifest `4d2b3127f607…`. **No model was loaded and no
  value was computed** — this check is about pixels only.
- Images 656, 728, 1465 and 1569 came from the config-seeded sample; 69, 377
  and 386 were requested by index to cover crop loss, extreme aspect ratio and
  the `rest` player, which a random sample of four would have missed. 728 and
  1465 were rendered but not inspected panel by panel, so they are not counted
  above.
- **Caveat on the sample.** The seeded sampler was switched from
  `random.Random` to `numpy.random.default_rng` during lint cleanup *after*
  these images were inspected, so re-running `--num-images 4` today draws
  694, 998, 1026 and 1510 instead. The seed derivation is unchanged; only the
  generator is. The five inspected images are therefore reproducible **by
  explicit index only**, as in the command below. Said plainly: the sample was
  reproducible when it was drawn and is not reproducible from `--num-images`
  now, which is a flaw in how this entry's evidence was produced, not a
  finding.

### Evidence

Exactly the five sheets this entry is based on:

```bash
uv run python -m food_shapley.preview --config configs/week01-clip-blur.toml \
    --index 656 --index 1569 --index 377 --index 386 --index 69
```

They land in `runs/visual-check/2ad46618605e/`, which is git-ignored, so the
command above is the artifact. Programmatic companions:
`tests/unit/test_masking.py` (18 tests, including "a sharp region keeps its own
stripe's colour") and
`tests/unit/test_segments.py::test_image_resize_matches_torchvision`.

### Interpretation

*Interpretation, not measurement.* The segment→player→render path puts the
right pixels in the right place, including through the centre crop, for images
spanning the aspect ratios in this split. The failure mode `PLAN.md` warns
about — "only bun visible" not showing the bun — does not occur in this sample,
and the one apparent violation traced to an unusual annotation rather than to
the code. Combined with W01-05's byte-equality result, the geometry is now
checked both against torchvision and against the eye.

### Limitations and alternative explanations

- **Five images out of 2,135.** The criterion in `docs/week-01.md` is met, but
  this is a spot check. A defect confined to, say, images with exactly one
  player (6 exist) would survive it.
- **"Correct" is my judgement from a 224 px panel.** Boundary errors of one or
  two pixels are invisible at that scale and would not have been caught.
- **The silhouette confound is confirmed, not refuted.** In every singleton the
  hidden ingredients remain legible as shapes: the strawberry ring of image 656
  is clearly outlined in the `only cake` panel. At radius 12 the empty
  coalition of image 69 still reads as "something red on something pale",
  i.e. an open sandwich. `PLAN.md` lists "blur does not actually remove
  information" as a known risk; this is direct visual evidence for it, not
  against it. A positive Shapley value may therefore reflect *sharpening* an
  already-visible shape rather than *revealing* an ingredient.
- The radius choice was made by eye on two images and was **not** tuned against
  model behaviour, because no model has been loaded.
- This validates the blur operator only. Week 2's token-dropping operator needs
  its own check.

### Implications

- The visual half of `PLAN.md`'s pre-run gate is satisfied; task 1.6 is still
  blocked on the efficiency check, which needs the value function (task 1.3).
- `blur_radius` and `keep_background` are now required `ExperimentConfig`
  fields, so they enter the run fingerprint and cannot drift silently between
  runs. Changing either changes the game, not just the picture.
- `food_shapley.preview` should be re-run whenever the geometry, the player
  construction or the removal operator changes. It is cheap and needs no GPU.

### Follow-up questions and next steps

- Re-run the check on the Week-2 token-dropping operator before trusting any
  cross-operator agreement claim.
- Render the 6 single-player images and the 3 `rest` images in full; they are
  the degenerate cases and there are few enough to inspect exhaustively.
- Quantify the silhouette confound once a model exists: compare `v(∅)` against
  a uniform-grey baseline. If the blurred frame scores much higher, blur is
  leaking scene information into every coalition.

---

## W01-06 — Player construction runs over the whole test split: 2,135/2,135 images, 45,916 coalitions, max 10 players

> **Partially superseded by W01-09 (2026-10-18).** The counts below were
> measured before the scene became a player. Under the current configuration
> (`background_mode = "player"`, `max_players = 11`) the split has 91,828
> coalitions, not 45,916, and up to 11 players. The 0 % failure rate, the 0 %
> capping rate and the "a player is an ingredient class, not a connected
> component" decision are unaffected.

- **Date:** 2026-10-18
- **Status:** superseded in part (see the note above)
- **Plan reference:** docs/week-01.md, task 1.2
- **Run / artifact:** `configs/week01-clip-blur.toml`, commit `a2eb35c` plus
  the uncommitted working tree adding `src/food_shapley/segments.py`; dataset
  manifest `4d2b3127f607…`

### Observation

`build_players` applied to every test image under
`configs/week01-clip-blur.toml` (`image_size = 224`, `max_players = 10`,
`min_area_fraction = 0.005`):

| players | images |
|---|---|
| 1 | 6 |
| 2 | 504 |
| 3 | 662 |
| 4 | 460 |
| 5 | 314 |
| 6 | 111 |
| 7 | 56 |
| 8 | 19 |
| 9 | 2 |
| 10 | 1 |

- **2,135 of 2,135 images produced a valid player set; zero failures.**
- Mean 3.58 players, max 10.
- **`max_players` was never reached**: 0 images capped, as W01-04 predicted.
- A `rest` player appears in **3 images (0.14 %)**, absorbing 6 segments in
  total.
- **Σ 2^players = 45,916 coalitions** for the entire split.

### Context and motivation

W01-04 predicted these numbers from `classes_on_image` alone, before any
resizing. This is the measurement after the real pipeline — centre crop,
nearest-neighbour mask resampling, area thresholding and merging — so the two
can be compared.

### Setup

Deterministic: no model, no randomness, no GPU. Reads the pinned split and
calls `food_shapley.segments.build_players` once per image.

**Decision recorded here because it determines the numbers above:** a player
is an **ingredient class**, not a connected component. One `carrot` id covering
three disjoint blobs is one player. W01-04 listed this as unresolved; it is
resolved this way because the research question is whether a dish is
recognised from an ingredient or from a combination of ingredients, so "the
carrot in this image" is the unit the question is about. Per-component players
would answer a different question — which *region* mattered — and would raise
the player count by an unmeasured amount. The counts in W01-04 and above are
therefore exact for the chosen definition, and a lower bound for the
alternative one.

### Evidence

Reproduce with the sweep script recorded in this entry's commit message; the
per-image assertions it checks are also enforced as tests in
`tests/integration/test_players_on_real_images.py` (8 sampled images) and
`tests/unit/test_segments.py` (50 synthetic cases).

### Interpretation

*Interpretation, not measurement:*

- W01-04's prediction held: the pipeline does not change the ingredient-count
  distribution materially (the counts shift by at most a few images, from
  segments lost to the crop — see W01-05).
- 45,916 coalitions is small. At even 100 forward passes per second this is
  under ten minutes of GPU time for the whole split, which supports `PLAN.md`'s
  claim that task 1.6 is an orchestration exercise rather than a compute one.
- The merging machinery is effectively inert on this split: 3 images, 6
  segments. That is the desired outcome — it means almost no result will carry
  a "rest" caveat — but it also means **the merging code is barely exercised by
  the real data**, which is why it is tested synthetically instead.

### Limitations and alternative explanations

- **6 images have exactly one player.** Their game is degenerate: a single
  player has Shapley value `v({1}) − v(∅)` by definition and no interactions
  at all. They cannot contribute to the main-effect-vs-interaction question
  and must be excluded from, or reported separately in, any Week-2 analysis.
  They are *not* an error, so the pipeline keeps them.
- Counting coalitions is not counting cost: preprocessing and blurring per
  coalition are ignored, and they may dominate the forward pass.
- Test split only; the train split has not been measured.
- `min_area_fraction = 0.005` is a judgement call, not a measured optimum —
  see W01-05.

### Implications

- `max_players = 10` and `min_area_fraction = 0.005` are the Week-1 settings,
  with the evidence that neither distorts the split.
- The full-run coverage criterion in docs/week-01.md ("≥ 99 % of images
  produce a readable `.npz`") has no *player-construction* obstacle: the
  failure rate at this stage is 0 %. Any shortfall in the real run will come
  from the model or the job array, not from the masks.
- Week 2 must handle the 6 single-player images explicitly.

### Follow-up questions and next steps

1. Record the per-image wall-clock cost once CLIP is wired in (task 1.4).
2. Decide in Week 2 whether single-player images are excluded or reported
   separately.

---

## W01-05 — CLIP's centre crop destroys 46 of 7,697 ingredient instances; they are dropped from the player set rather than kept empty

- **Date:** 2026-10-18
- **Status:** validated (exhaustive over the test split)
- **Plan reference:** docs/week-01.md, task 1.2; risk "Mask and image resized
  with different interpolation"
- **Run / artifact:** `configs/week01-clip-blur.toml`; geometry verified
  against `open_clip.create_model_and_transforms("ViT-B-32")`

### Observation

open_clip's ViT-B-32 preprocessing is
`Resize(224, bicubic, antialias) → CenterCrop(224)` — a **shortest-side**
resize followed by a square crop, which discards image content. Applying that
geometry to the masks over all 2,135 test images:

- **46 of 7,697 ingredient instances (0.60 %) vanish entirely**, affecting
  **44 images (2.06 %)**. 42 images lose one ingredient, 2 lose two.
- The food-pixel fraction *rises* from 0.529 to 0.606: the crop preferentially
  removes background at the edges.
- Aspect ratios are mild — median 1.33, p95 1.61, max 1.92.
- Post-crop segment sizes are large: median 12.9 % of the frame, p10 3.1 %.
  Only 0.94 % of segments fall below 0.5 % of the frame (≈ a quarter of one
  32-pixel ViT patch).

### Context and motivation

Before writing `segments.py` I checked what geometry CLIP actually uses rather
than assuming "resize to 224×224". The answer changes the player set, because
an ingredient outside the crop is not in the model's input at all.

### Setup

Deterministic, no model weights loaded: the transform was read off
`create_model_and_transforms`, and the geometry reimplemented in
`food_shapley.segments.clip_geometry`.

### Evidence

`tests/unit/test_segments.py::test_image_resize_matches_torchvision` and
`::test_mask_resize_matches_torchvision_nearest` assert **byte-equality**
against `torchvision.transforms.Resize` + `CenterCrop` over seven image
shapes. torchvision's transforms are what open_clip composes, so this is
equality with CLIP's own preprocessing, not with a reimplementation of it.

### Interpretation

*Interpretation, not measurement:*

- An ingredient that is cropped away but kept as a player would have an
  **empty mask, hence Shapley value exactly 0 by construction** — not because
  the model ignored it, but because nothing was removed when it was removed.
  Every "how often is a single ingredient enough?" statistic would be diluted
  by these. They are therefore dropped from the player set and recorded in
  `PlayerSet.dropped_labels`.
- Two rounding details matter and are easy to get wrong: torchvision
  **truncates** the resized long side (`int`) but **rounds** the crop offset.
  A one-pixel error here misaligns every mask against every image, and the
  efficiency check cannot detect it — the game stays perfectly self-consistent
  while measuring the wrong pixels. This is the `PLAN.md` masking bug in its
  most silent form.

### Limitations and alternative explanations

- **The alternative was not tested empirically.** Squashing to 224×224 drops
  no ingredient, at the cost of feeding CLIP a distorted aspect ratio it was
  not trained on. I chose crop-matching on the reasoning that a confound
  affecting 100 % of images is worse than losing 2 % of them — that is an
  argument, not a measurement. If a reviewer presses on it, the honest
  response is to re-run a subset under both geometries and compare.
- The 0.60 % figure counts ingredients that vanish *completely*. Ingredients
  that are heavily but not totally cropped are not counted and are arguably a
  bigger distortion, since they stay players while losing most of their
  evidence.
- `min_area_fraction = 0.005` is justified by "below a quarter of a ViT patch
  a segment cannot plausibly carry its own evidence". That is a plausibility
  argument about patch granularity, not a measured threshold.

### Implications

- The 44 affected images must be reported, not silently handled: whatever the
  Week-2 analysis says about ingredient coverage, 2.06 % of images are
  answering it about a subset of their annotated ingredients.
- The same geometry must be reused verbatim by Week 2's token-dropping
  operator, which maps mask pixels to 7×7 ViT patches. Any divergence there
  reintroduces the misalignment this entry rules out.

### Follow-up questions and next steps

1. Render masked coalitions for several of the 44 affected images as part of
   the task-1.5 visual check; they are the worst case for alignment.
2. Consider a squash-geometry robustness run if the crop becomes a reviewer
   objection.

---

## W01-04 — No test image has more than 10 ingredients, so exact Shapley needs no segment merging

- **Date:** 2026-10-17
- **Status:** validated (exhaustive over the test split)
- **Plan reference:** docs/week-01.md, task 1.2; risk "Images with too many
  segments make exact computation expensive"; `PLAN.md` steps 3 and 5
- **Run / artifact:** `data/raw/foodseg103/test/`, manifest
  `3fc784f31a1c…`
- **Note added 2026-10-18:** that content hash no longer matches the directory.
  `id2label.json` is now downloaded into the split and is covered by the hash,
  so the same data hashes to `4d2b3127f607…`. The pinned mirror revision is
  unchanged and no pixel or label changed; the finding below stands, and
  [W01-06](#w01-06--player-construction-runs-over-the-whole-test-split-21352135-images-45916-coalitions-max-10-players)
  re-measures it through the real pipeline.

### Observation

Ingredient counts over **all 2,135 test images**, taking
`set(classes_on_image) - {0}` (background excluded per W01-03):

| ingredients | images | cumulative |
|---|---|---|
| 1 | 5 | 0.2 % |
| 2 | 500 | 23.7 % |
| 3 | 658 | 54.5 % |
| 4 | 452 | 75.6 % |
| 5 | 323 | 90.8 % |
| 6 | 116 | 96.2 % |
| 7 | 55 | 98.8 % |
| 8 | 21 | 99.8 % |
| 9 | 4 | 100.0 % |
| 10 | 1 | 100.0 % |

**The maximum is 10. The median is 3.** Images capped at a given
`max_players`: 7 → 26 images (1.2 %), 8 → 5 (0.2 %), 9 → 1, **10 → 0**.

Summing 2^k over the split gives **≈ 47,800 coalitions for the entire test
split** at `max_players = 10`, i.e. with no merging at all.

### Context and motivation

`configs/smoke.toml` carries `max_players = 7`, and `PLAN.md` estimates "~128
coalitions per image", which is 2^7. Before implementing the merge-into-rest
logic I checked what the data actually requires.

### Setup

Deterministic read of the `classes_on_image` column for all 2,135 rows. No
mask decoding, no model, no randomness.

### Interpretation

*Interpretation, not measurement:*

- The "~128 coalitions per image" figure in `PLAN.md` looks like a worst-case
  estimate being treated as typical. The mean is ~22 coalitions per image;
  128 is the 99th percentile, not the middle.
- **Segment merging is not needed to make exact computation affordable.** At
  `max_players = 10` nothing is capped and the whole split is ~47,800 CLIP
  forward passes — minutes of GPU time, not a cluster-scale job. The risk row
  "images with too many segments make exact computation expensive" does not
  materialise on this split.
- That removes a confound rather than just saving time. Every merged image
  would have carried an artificial "rest" player whose Shapley value means
  something different from an ingredient's, and any claim about interaction
  structure would have needed a caveat for 1.2 % of the data.
- There may still be a *non*-compute reason to merge: a segment of 50 pixels
  is arguably not a visible ingredient, and blurring it may be a no-op. That
  is an area threshold, a different parameter from `max_players`, and it
  should be justified on visibility grounds rather than cost.

### Limitations and alternative explanations

- Test split only. The train split (4,983 images) has not been measured and
  may have a longer tail; it is not used for the Milestone 1 run.
- This counts **distinct label ids**, not connected components. One `carrot`
  id may cover several disjoint blobs. If players are later defined per
  connected component rather than per ingredient class, these counts are a
  lower bound and the conclusion must be rechecked.
- The forward-pass estimate ignores preprocessing and blurring cost, which
  may well dominate the model itself.

### Implications

- Set `max_players = 10` for the real Week-1 config, not 7, and record in the
  run that zero images were capped — a stronger statement than "few were".
- Still implement the cap and the "rest" player: it is a guard for other
  splits and other datasets, and `PLAN.md` asks for the count of capped
  images to be recorded. It should simply never fire on this split.
- Task 1.6's "scale out" is an orchestration exercise, as `PLAN.md` says, not
  a compute-bound one. Worth confirming once the real per-image cost is known.

### Follow-up questions and next steps

1. Add a non-smoke `configs/` entry with `max_players = 10`.
2. Decide players-per-class vs players-per-connected-component before 1.2;
   it changes these numbers.
3. Measure the train split if Milestone 2 needs it.

*Resolved 2026-10-18:* (1) `configs/week01-clip-blur.toml`. (2) Per ingredient
class — see W01-06, "Setup". (3) Still open.

---

## W01-03 — The mirror does ship an ingredient-name list after all; label 0 is background

- **Date:** 2026-10-17
- **Status:** validated
- **Plan reference:** docs/week-01.md, task 1.2 and 1.3
- **Run / artifact:** `id2label.json` at
  `EduardoPacheco/FoodSeg103@34e1208e14bc…`, sha256
  `b6b6458cd2b259a47374f8ad5cf331528caf8764eb50397b2c1e2bb2fa7a0a50`
- **Corrects:** [W01-02](#w01-02--classes_on_image-includes-background-and-no-ingredient-names-ship-with-the-mirror)

### Observation

The mirror repository contains a top-level `id2label.json` (1,823 bytes, blob
`82cb3539`). It holds **104 entries, ids 0–103, contiguous, no gaps**. Spot
checks:

| id | name |
|---|---|
| 0 | `background` |
| 20 | `dried cranberries` |
| 56 | `shrimp` |
| 66 | `rice` |
| 84 | `carrot` |
| 96 | `French beans` |
| 103 | `other ingredients` |

`datasets.load_dataset` does not fetch this file — it reads only the parquet
shards listed in the card's `configs` block — which is why inspecting the
loaded `Dataset.features` (as W01-02 did) showed no mapping.

### Context and motivation

W01-02 concluded that the ingredient names were unavailable and that this
blocked tasks 1.2 and 1.3. Before acting on that, I listed the repository's
files through the Hub API rather than only the loaded dataset object. The
conclusion was wrong: I had inferred "absent from the repository" from "absent
from `Dataset.features`".

### Setup

```bash
curl -sSL "https://huggingface.co/datasets/EduardoPacheco/FoodSeg103/resolve/\
34e1208e14bc3595d544fc8c3f3c6673253fd9ef/id2label.json"
```

The `-L` matters: without it the Hub returns a 307 whose body is the redirect
notice, which parses as "not JSON" rather than as a failure.

### Evidence

- Hub tree listing at the pinned revision shows `id2label.json`, blob
  `82cb3539e971ed1fbde2d9f3464271bd3a92d497`.
- 104 entries, keys exactly `range(104)`, verified programmatically.
- Example 0's mask ids `[0, 20, 56, 66, 84, 96]` resolve to background, dried
  cranberries, shrimp, rice, carrot, French beans — a coherent dish, which is
  independent corroboration that the mapping aligns with the mask ids rather
  than being some other ordering.

### Interpretation

*Interpretation, not measurement:*

- **Label 0 is literally named `background`.** The `PLAN.md` warning is now
  confirmed by the data, not just asserted; the planned exclusion test is
  grounded.
- **Id 103 is `other ingredients`** — a catch-all that already exists in the
  annotations. This collides conceptually with the "rest" player task 1.2 is
  meant to create by merging small segments. Two different notions of "other"
  in one player set would be confusing at best and double-counted at worst.
- The names are human-readable and mostly prompt-ready (`French beans`,
  `dried cranberries`), so task 1.3's prompt construction is unblocked.

### Limitations and alternative explanations

- This is the **mirror's** `id2label.json`, not the official `category_id.txt`,
  which is still unreachable (W01-01). The ordering has not been checked
  against the FoodSeg103 paper. The coherent dish in example 0 is evidence
  that it is right, not proof.
- 104 names for "103 food categories" is consistent with 103 ingredients plus
  background, but I have not confirmed that reading against the paper.
- The file is not currently downloaded or hashed by `food_shapley.download`,
  so it is not yet covered by the manifest.

### Implications

- Tasks 1.2 and 1.3 are **not** blocked. W01-02's "blocker" framing was wrong.
- `download.py` should fetch `id2label.json` alongside the split and include
  it in the manifest hash, so the names are pinned to the same revision as the
  pixels.
- Task 1.2 must decide explicitly what to do about id 103 `other ingredients`
  versus the merged "rest" player, and record the decision.

### Follow-up questions and next steps

1. Add `id2label.json` to the download and the manifest.
2. Decide and document the id-103 / "rest" question before building players.
3. Still worth comparing against the official `category_id.txt` if the zip
   ever returns.

---

## W01-02 — `classes_on_image` includes background, and no ingredient names ship with the mirror

- **Date:** 2026-10-17
- **Status:** **partially contradicted** by [W01-03](#w01-03--the-mirror-does-ship-an-ingredient-name-list-after-all-label-0-is-background).
  The `classes_on_image` finding stands. The claim that no name mapping ships
  with the mirror is **wrong**: `id2label.json` exists in the repository, it is
  simply not exposed through `load_dataset`. Original text left intact below.
- **Plan reference:** docs/week-01.md, task 1.2; risk "Label 0 is background,
  not a player"
- **Run / artifact:** `data/raw/foodseg103/test/` ·
  manifest `content_sha256: 3fc784f31a1cbad4e93455d155f0018739a53be89626614d3d589bbc9361a1e8`

### Observation

On the FoodSeg103 test split (2,135 images):

- For example 0, the unique mask label ids are `[0, 20, 56, 66, 84, 96]` and
  `classes_on_image` is **the same list, including the 0**.
- Over the first 200 rows: 78 distinct label ids, min 0, max 103. Image and
  mask sizes agree in 200/200 rows. Image mode `RGB`, mask mode `L` (uint8).
- The dataset features are
  `{image: Image, label: Image, classes_on_image: List(int64), id: int64}`.
  The mask is stored as an image, not as a `ClassLabel`, so **no label-id →
  ingredient-name mapping is present at all.**

### Context and motivation

`PLAN.md` warns that label 0 is background and not a player. Before writing
the segment→player conversion (task 1.2), I checked what the data actually
contains rather than assuming the warning's shape.

### Setup

`datasets.load_from_disk("data/raw/foodseg103/test")`, mirror
`EduardoPacheco/FoodSeg103` at revision `34e1208e14bc…`. No model, no seed,
no randomness: a deterministic read of the first 200 rows plus the full
feature schema.

### Evidence

Reproducible with:

```python
from datasets import load_from_disk
from food_shapley.download import split_dir

ds = load_from_disk(str(split_dir("test")))
print(ds.features)
print(sorted(ds[0]["classes_on_image"]))
```

### Interpretation

*Interpretation, not measurement:* `classes_on_image` is the convenient field
to reach for when building the player set, and it is booby-trapped — using it
directly yields a "background" player that would absorb Shapley mass and
silently corrupt every per-ingredient number downstream. The efficiency check
would **not** catch this: a game with background as a legitimate player is
internally consistent. Only an explicit exclusion test catches it.

The missing name mapping is the more awkward finding. An interaction reported
between "class 56" and "class 84" is not a result anyone can read.

### Limitations and alternative explanations

- The 200-row scan is a sample, not the full split; the size-agreement claim
  holds for those 200 rows only. It is plausible, but unverified, that some
  later row disagrees. Full-split verification belongs in task 1.2.
- Only the test split was inspected. The train split is downloaded by the same
  code path but has not been read.
- "Label 0 is background" is inferred from `PLAN.md` plus the fact that 0
  appears in essentially every image; it has not been confirmed against the
  official `category_id.txt`, which is inside the zip that is currently
  unreachable (see W01-01).

### Implications

- Task 1.2 must derive players from the unique ids in the mask **minus 0**,
  never from `classes_on_image` as given. The planned unit test asserting that
  label 0 is excluded is now known to be load-bearing, not ceremonial.
- Obtaining the ingredient-name list is a blocker for task 1.3's prompts and
  for every figure in the paper. It is not a Week-2 nicety.

### Follow-up questions and next steps

1. Source the 104-entry id→name list and record its provenance in
   `docs/datasets.md`. Preferred: the official `category_id.txt`; fallback:
   the mirror's dataset card, clearly labelled as such.
2. Confirm label 0 really is background by rendering a few masks with 0
   highlighted — cheap, and it settles the question by inspection.
3. Check image/mask size agreement over all 2,135 rows during task 1.2.

---

## W01-01 — Official FoodSeg103 distribution is unreachable; a pinned mirror was used instead

- **Date:** 2026-10-17
- **Status:** validated (as a statement about availability on this date)
- **Plan reference:** docs/week-01.md, task 1.1; docs/datasets.md
- **Run / artifact:** `data/raw/foodseg103/test/manifest.json` ·
  `src/food_shapley/download.py`

### Observation

- `https://research.larc.smu.edu.sg/downloads/FoodSeg103/FoodSeg103.zip`
  returned **HTTP 502** on 2026-10-17. Confirmed with a HEAD request, a ranged
  GET, and a request to the host root — which returned 200, so the host was up
  and the file specifically was not served.
- The Hugging Face mirror `EduardoPacheco/FoodSeg103` served the data. Its
  split sizes are **train 4,983 / validation 2,135 = 7,118**, matching the
  FoodSeg103 paper exactly. It is labelled Apache-2.0, as is the original.
- The mirror names the 2,135-image split `validation`; the paper and the
  official release call it `test`.
- Downloading the test split took 3 min 15 s and left 110 MB on disk (one
  Arrow shard), after ~1.25 GB of parquet passed through the `HF_HOME` cache.

### Context and motivation

`PLAN.md` step 0 requires FoodSeg103 on the cluster. "We used FoodSeg103" has
to mean a specific, checkable copy of the data, so the source had to be
established before anything downstream was built on it.

### Setup

`python -m food_shapley.download --split test`, mirror pinned to revision
`34e1208e14bc3595d544fc8c3f3c6673253fd9ef` (lastModified 2024-04-29),
`datasets` 5.1.0.

### Evidence

- `data/raw/foodseg103/test/manifest.json` records the source, repo id,
  revision, both split counts, the license, the columns, the `datasets`
  version, and `content_sha256 = 3fc784f31a1cbad4e93455d155f0018739a53be89626614d3d589bbc9361a1e8`.
- `python -m food_shapley.download --split test --verify-only` exits 0.
- Corruption is detected. Copying the split to a scratch data root, flipping
  one byte of `data-00000-of-00001.arrow`, and re-running `--verify-only`
  gives exit code 1 and
  `content hash <h> does not match manifest 3fc784f31a1c; the copy is corrupt
  or was modified`. (`<h>` depends on which byte was flipped; two separate
  trials produced `b27cf1d55fd8` and `eca8c6b2691f`.)
- `tests/unit/test_download.py` covers the same failure modes offline against
  a fake `datasets` module: 31 tests, all passing.

### Interpretation

*Interpretation, not measurement:* the matching split sizes and license are
good evidence that the mirror is a faithful copy, but they are not proof. A
re-upload could differ in image encoding, compression, or EXIF handling in
ways that leave the row counts untouched and still perturb CLIP's input. I
consider that unlikely but unexcluded.

### Limitations and alternative explanations

- **The mirror has not been byte-compared against the original**, because the
  original could not be fetched. This is the main caveat and it has to appear
  in the paper's data section, not just here.
- The 502 is a single-day observation. The URL may well come back.
- Only the test split has been downloaded; the train-split count comes from
  the mirror's metadata, not from a download.

### Implications

- The paper must say "FoodSeg103 via the `EduardoPacheco/FoodSeg103` mirror at
  revision `34e1208e`", not "FoodSeg103". The manifest makes that statement
  checkable by a reader.
- Bumping `HF_REVISION` is a change of input data: new fingerprint, new run.
  `verify_split` enforces this by failing when an on-disk copy's recorded
  revision no longer matches the pinned one.

### Follow-up questions and next steps

1. Retry the official URL periodically. If it returns, download it and compare
   image bytes against the mirror on a sample; record the outcome here either
   way.
2. Download the train split before any analysis that needs it, so its count is
   measured rather than taken from metadata.
