# Datasets: provenance, licensing, preprocessing

FoodSeg103 has been downloaded and verified (2026-10-17) and the Food-101 class
names have been downloaded and verified (2026-10-19); the remaining sections
are still unverified and are marked as such. Fill in each unverified row before
that dataset is first used, and record the verification date. Dataset files
live under `data/` and are never committed.

## FoodSeg103

**Verified 2026-10-17** by `python -m food_shapley.download --split test`.

| Field | Value |
|---|---|
| Role | Primary dataset: images + ingredient segmentation masks (all weeks) |
| Introduced in `PLAN.md` | Milestone 1, steps 0–1 |
| Source used | Hugging Face mirror [`EduardoPacheco/FoodSeg103`](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103) |
| Revision pinned | `34e1208e14bc3595d544fc8c3f3c6673253fd9ef` (lastModified 2024-04-29) |
| Authoritative source | `https://research.larc.smu.edu.sg/downloads/FoodSeg103/FoodSeg103.zip` (password-protected). **Returned HTTP 502 on 2026-10-17** — see below |
| License | Apache-2.0 (both the original release and the mirror) |
| Split used | `test` in this project's vocabulary = `validation` on the mirror; 2,135 images |
| Split sizes observed | train 4,983 / test 2,135 = 7,118 — matches the FoodSeg103 paper exactly |
| Columns | `image`, `label`, `classes_on_image`, `id` |
| Label id range observed | 0–103; 78 distinct ids present in the 200 test rows sampled |
| On-disk size | ~110 MB (single Arrow shard) after conversion; ~1.25 GB of parquet passes through the `HF_HOME` cache first |
| Local path | `data/raw/foodseg103/<split>/` (override via `FOOD_SHAPLEY_DATA`) |
| Integrity record | `data/raw/foodseg103/<split>/manifest.json`; `--verify-only` re-hashes the payload against it |

### Why the mirror, and what that costs

The authoritative distribution is the authors' password-protected zip. On
2026-10-17 that URL returned HTTP 502 (confirmed with a HEAD, a ranged GET,
and a 200 from the host root, so the host was up and the file was not). The
mirror was used instead.

What this does and does not establish:

- The mirror's split sizes match the paper exactly, and it carries the same
  Apache-2.0 license.
- It is a third-party re-upload and has **not** been byte-compared against the
  original. If the official URL comes back, compare the two and record the
  result here.
- The mirror is pinned to a commit. "We used FoodSeg103" is not a reproducible
  statement without a revision; a mirror can be updated or deleted.

### Preprocessing facts and assumptions

Verified on the downloaded test split:

- **`classes_on_image` includes background class 0.** On example 0 it is
  identical to the set of unique mask ids, `[0, 20, 56, 66, 84, 96]`. This is
  exactly the trap `PLAN.md` flags — taking `classes_on_image` as the player
  set would make "background" a player. Label 0 must be dropped explicitly,
  and there is a test for it.
- **Images and masks agree in size in the raw data** (checked on 200 rows).
  Misalignment, if it appears, will be introduced by our own resizing, not
  inherited.
- Image mode `RGB`, mask mode `L` (uint8).

Still to confirm in Week 1:

- Images and masks must be resized with consistent alignment; masks require
  nearest-neighbour interpolation, images do not.
- Small segments are merged and a "rest" player is added, capping the player
  count at `max_players` (see `configs/`). The merge threshold is a
  configuration value, not a hidden constant, and the number of images affected
  must be recorded.
- **The loaded dataset carries no label names, but the repository does.** The
  features are `image: Image`, `label: Image`, `classes_on_image: List(int64)`,
  `id: int64` — the mask is an image, not a `ClassLabel`. The names live in a
  separate top-level `id2label.json` that `load_dataset` does not fetch:
  **104 entries, ids 0–103, contiguous**, sha256
  `b6b6458cd2b259a47374f8ad5cf331528caf8764eb50397b2c1e2bb2fa7a0a50` at the
  pinned revision. Id 0 is literally `background`; id 103 is
  `other ingredients`. See insight W01-03.
- **Id 103 `other ingredients` already exists in the annotations.** Task 1.2's
  plan to merge small segments into a "rest" player creates a second,
  different notion of "other". Decide and document how the two relate before
  building the player set.
- `id2label.json` is **not yet downloaded or covered by the manifest hash**.
  It should be, so the names are pinned to the same revision as the pixels.
- **Ingredients per image (test split, background excluded): median 3,
  maximum 10.** Measured over all 2,135 images; see insight W01-04.
- The non-ingredient region is a **`scene` player** under the Week-1 config
  (W01-09), so the player count is one higher than the ingredient count on
  2,134 of 2,135 images. With `max_players = 11` the cap fires on none of
  them, and exact Shapley over the whole split needs **91,828 coalitions**.

## Food-101 class names

**Verified 2026-10-19** by `python -m food_shapley.prompts`.

| Field | Value |
|---|---|
| Role | Dish prompt set for the CLIP value function: it *is* the softmax denominator |
| Introduced in `PLAN.md` | Milestone 1, step 2 ("e.g. Food-101's 101 class names") |
| Source used | Hugging Face [`ethz/food101`](https://huggingface.co/datasets/ethz/food101) **builder metadata only** |
| Revision pinned | `83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9` (lastModified 2026-03-09) |
| What is fetched | `load_dataset_builder(...).info.features["label"].names` — the `ClassLabel` name list. **No images**: builder metadata only, none of Food-101's 101,000 images are downloaded |
| Authoritative source | ETHZ, [Food-101 project page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (Bossard, Guillaumin & Van Gool, ECCV 2014) |
| License | **No license is declared on the Hub repository.** The original release states the images were scraped from Foodspotting and are provided for *non-commercial research use only*; the authors do not own them. Recorded in the manifest as `"unknown (images: ETHZ/Foodspotting, scientific fair use; names only are used)"` |
| License risk for this project | Low but non-zero: only the 101 **strings** are used, never an image. A list of English dish names is not plausibly the protected element of that dataset, but the paper should cite Food-101 as the source of the label vocabulary rather than imply the dataset was used |
| Classes observed | exactly **101**, all unique, all non-empty |
| Name format | lowercase with underscores (`apple_pie`, `beef_carpaccio`, `huevos_rancheros`). `prompts.dish_display_name` converts underscores to spaces before templating |
| On-disk size | ~12 kB (`dish_names.json` 1,692 B + `manifest.json` 426 B) |
| Local path | `data/raw/food101/` (override via `FOOD_SHAPLEY_DATA`) |
| Integrity record | `data/raw/food101/manifest.json`; `names_sha256 = 92d5cd07a7856279655bc77e852337ad501f98c8a0e8cee7c969c304a6b19255`, `datasets` 5.1.0 |

### What the hash does and does not pin

`names_sha256` is computed over **this project's own serialisation** of the
name list (`json.dumps(list(names), indent=2)` plus a trailing newline), not
over an upstream file — the names come out of builder metadata, not out of a
downloadable artifact with a stable byte layout. So the hash detects a local
copy that was edited or truncated, and it
detects an upstream reordering or renaming if the pinned revision is ever
moved. It is **not** a hash anyone else can independently reproduce from the
Hub without running this code. The revision pin is what makes the source
reproducible; the hash is what makes the local copy checkable.

The order matters and is preserved: the prompt set is the softmax denominator
and `DishTarget.index` is an index into it, so a reordered list would silently
relabel every stored target. `verify_dish_names` therefore hashes the ordered
list, not a set.

### Prompt template

The template is a required `ExperimentConfig` field (`prompt_template`), so it
enters the run fingerprint and cannot drift between runs. Week 1 uses:

```toml
prompt_template = "a photo of {}, a type of food."
```

This is the Food-101 template from the CLIP paper's prompt-engineering
appendix, used rather than invented because a hand-tuned template would be an
unreported degree of freedom in every number the project reports. It was not
tuned here and no alternative template has been measured — changing it changes
every value in the game and produces a new fingerprint, by design.

### Why a borrowed vocabulary, and what it costs

FoodSeg103 annotates **ingredients, not dishes**: there is no dish label in the
data at all (see `docs/week-01.md`, task 1.3). The target dish is therefore the
model's own top-1 over this prompt set on the unmasked photograph. Two
consequences, both measured in insight W01-10:

- Food-101's 101 dishes are **not** a cover of what FoodSeg103 photographs.
  Over 200 test images the model used 66 of the 101 classes, and `pork_chop`
  alone absorbed 13 % of images, firing on plates of chicken and rice.
- Top-1 confidence does **not** separate the apt targets from the inapt ones
  (minimum 0.167 over 200 images; a frame containing cake was called
  `baby_back_ribs` at 0.79), so no confidence threshold is applied anywhere.

A dish-labelled food dataset would remove this caveat and is the obvious
Milestone-2 improvement; `PLAN.md` step 2 explicitly names Food-101 as the
default, so Week 1 follows it and records the cost.

## Recipe1M *(optional, Week 2 stretch goal)*

| Field | Value |
|---|---|
| Role | Linking FoodSeg103 images to recipes; visible/invisible ingredient split |
| Introduced in `PLAN.md` | Milestone 1, step 8 (optional) |
| Source URL | *unverified* |
| License / access restrictions | *unverified — Recipe1M access is commonly gated; confirm terms before use* |
| **Unresolved** | `PLAN.md` does not specify how FoodSeg103 images are linked to Recipe1M recipes. Resolve before committing to the stretch goal. |

## Shifted / harder test set *(Week 5)*

| Field | Value |
|---|---|
| Role | Distribution-shift evaluation for the secondary generalization analysis |
| Introduced in `PLAN.md` | Milestone 2, step 15 |
| Candidate | A non-Western-dish split, or a FoodSeg154-style extension |
| **Unresolved** | The plan lists candidates but does not choose. Decide before Week 5; document construction, seed, and selection criteria here so the split is reproducible. |

## Compositionality benchmarks: SugarCrepe, ARO *(Week 5)*

| Field | Value |
|---|---|
| Role | Accuracy-based compositionality scores to contrast against the interaction-based index |
| Introduced in `PLAN.md` | Milestone 2, step 14 |
| Source URL | *unverified* |
| License | *unverified* |
| Notes | `PLAN.md` permits either running them or sourcing reported numbers. Record which was used **per model**; a table mixing the two needs an explicit caveat. |

## Handling rules

- Datasets are read-only inputs. Code never writes into `data/raw/`.
- Derived artifacts go to `data/interim/`; results go to `runs/`.
- Record the dataset version/checksum used by a run in its `run_metadata.json`
  once dataset loading exists.
- No dataset files, crops, or checkpoints are committed to git. Pre-commit
  blocks files over 512 kB as a backstop.
- Before publishing example image crops, confirm the license permits it.
