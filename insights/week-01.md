# Week 1 insights — Pipeline, value function, first full run

Plan: [docs/week-01.md](../docs/week-01.md) · Entry format:
[insights/README.md](README.md)

**Four entries, all about the input data. No experiment has been run and no
model has been evaluated.** W01-03 partially contradicts W01-02; read both.

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
  Which coalitions were inspected?
- Is the log-probability value function sensible on obvious unmasked images?
  What exactly was checked, and what was the outcome?
- How many images exceeded `max_players` and had segments merged into "rest"?
  **Answered in advance by W01-04: zero, at `max_players = 10`.** The run must
  still report the figure it actually observed.
- What fraction of the test split produced usable output, and why did the rest
  fail?

Record failures and surprises here as readily as successes — a failed masking
check is one of the most valuable entries this week can contain.

---

## W01-04 — No test image has more than 10 ingredients, so exact Shapley needs no segment merging

- **Date:** 2026-10-17
- **Status:** validated (exhaustive over the test split)
- **Plan reference:** docs/week-01.md, task 1.2; risk "Images with too many
  segments make exact computation expensive"; `PLAN.md` steps 3 and 5
- **Run / artifact:** `data/raw/foodseg103/test/`, manifest
  `3fc784f31a1c…`

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
