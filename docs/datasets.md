# Datasets: provenance, licensing, preprocessing

FoodSeg103 has been downloaded and verified (2026-10-17); the remaining
sections are still unverified and are marked as such. Fill in each unverified
row before that dataset is first used, and record the verification date.
Dataset files live under `data/` and are never committed.

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
  maximum 10.** Measured over all 2,135 images; see insight W01-04. Exact
  Shapley over the whole split therefore needs ~47,800 coalitions in total,
  and `max_players = 10` caps nothing.

## Food-101 class names

| Field | Value |
|---|---|
| Role | Candidate dish prompt set for the CLIP value function |
| Introduced in `PLAN.md` | Milestone 1, step 2 ("e.g. Food-101's 101 class names") |
| Source URL | *unverified* |
| License | *unverified* |
| Notes | Only the class-name list is needed, not the images. The exact prompt template is a configuration value and must be recorded with every run. |

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
