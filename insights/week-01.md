# Week 1 insights — Pipeline, value function, first full run

Plan: [docs/week-01.md](../docs/week-01.md) · Entry format:
[insights/README.md](README.md)

**Two entries, both about the input data. No experiment has been run and no
model has been evaluated.**

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
- What fraction of the test split produced usable output, and why did the rest
  fail?

Record failures and surprises here as readily as successes — a failed masking
check is one of the most valuable entries this week can contain.

---

## W01-02 — `classes_on_image` includes background, and no ingredient names ship with the mirror

- **Date:** 2026-10-17
- **Status:** validated (direct inspection of the downloaded split)
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
