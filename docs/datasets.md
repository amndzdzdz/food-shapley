# Datasets: provenance, licensing, preprocessing

Nothing in this file has been verified yet — no dataset has been downloaded.
Fill in the unverified rows before the dataset is first used, and record the
verification date. Dataset files live under `data/` and are never committed.

## FoodSeg103

| Field | Value |
|---|---|
| Role | Primary dataset: images + ingredient segmentation masks (all weeks) |
| Introduced in `PLAN.md` | Milestone 1, steps 0–1 |
| Source URL | *unverified — fill in before first use* |
| Version / release | *unverified* |
| License and redistribution terms | *unverified — check before publishing any image crops in the paper* |
| Split used | Test split; `PLAN.md` gives ~2,000–7,118 images |
| Local path | `data/raw/foodseg103/` (override via `FOOD_SHAPLEY_DATA`) |
| Verified on | — |

### Preprocessing assumptions (to be confirmed in Week 1)

- **Label 0 is background, not a player** (`PLAN.md`). It must be excluded from
  the player set.
- Images and masks must be resized with consistent alignment; masks require
  nearest-neighbour interpolation, images do not.
- Small segments are merged and a "rest" player is added, capping the player
  count at `max_players` (see `configs/`). The merge threshold is a
  configuration value, not a hidden constant, and the number of images affected
  must be recorded.

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
