# Week 1 insights — Pipeline, value function, first full run

Plan: [docs/week-01.md](../docs/week-01.md) · Entry format:
[insights/README.md](README.md)

**Thirteen entries. W01-01 to W01-09 are about the input data, how it becomes
coalition players, and how a coalition is rendered. W01-10 is the first entry
in which a model was actually loaded and run: it records the value function's
behaviour on 200 unmasked images, a checkpoint misconfiguration that would
have corrupted 37 % of targets, and the limits of a borrowed dish
vocabulary. W01-11 is the first entry containing an actual Shapley value: the
coalition game exists, exact values over three real images satisfy efficiency
to the last bit, and the one thing that is *not* exact — CLIP's float32
forward pass under rebatching — is bounded there. W01-12 is about the first
numbers the project intends to *store*: `shapiq` returns k-SII in a shape
that depends on the values rather than on the game, so every index block is
rebuilt to a fixed shape before it is written. W01-13 is the only entry about
the cluster: the first submission failed, but it failed late enough to prove
most of the environment. No experiment in the `PLAN.md` sense has been run:
three images are a correctness check, not a result, and the cluster has not
yet decomposed one.** W01-03 partially contradicts W01-02, W01-09 partially
supersedes W01-06, and W01-12 corrects a generalisation in W01-11; read all
three pairs.

## Questions this week's entries should answer

Derived from the Week 1 plan; each needs a dated entry once evidence exists.

- Did the toy SLURM job and array run on the cluster as written? Which
  assumptions in `slurm/README.md` turned out to be wrong — partition,
  walltime, `uv run --no-sync` on an offline node, scratch paths? This is the
  first thing to record, because every later run inherits these settings.
  **Partly answered by W01-13.** The toy jobs were never submitted; the *real*
  array was, on 2026-10-23, and both its tasks failed on a config mismatch
  before decomposing anything. Three documented assumptions were wrong —
  a 32-task array is unsubmittable (`MaxSubmit = 30`), the node is *not*
  offline and the model load contacted the Hub, and `TMPDIR` is forced by
  `slurmstepd`. Partition, `uv run --no-sync`, the imports, the dataset and
  the checkpoint all worked. **Still open: whether a GPU was used, and the
  per-image runtime that `-t` should be set from.**
- Does the efficiency check (`sum(SV) == v(N) − v(∅)`) hold, and at what
  numerical tolerance? **Answered by W01-11 and corrected by W01-12 for three
  real images: the residual is `0.0`, `−4.441e-16`, `0.0` as task 1.6 will
  compute it — six orders of magnitude inside the `1e-9` gate — provided both
  ends are read from the same powerset evaluation. Not exactly zero in
  general; W01-11's `0.0` came from a different summation order. Still open
  over the whole split, which needs task 1.6.**
- Does the visual check confirm that masked coalitions show what they claim to?
  Which coalitions were inspected? **Answered by W01-07: 19 coalitions over 5
  images, all correct, listed there by name.**
- Is the log-probability value function sensible on obvious unmasked images?
  What exactly was checked, and what was the outcome? **Answered by W01-10:
  checked on 200 unmasked images — top-1 confidence median 0.580 (min 0.167),
  `v(N) − v(∅)` positive on 200/200 with median +4.13 nats, re-scoring
  bit-identical. The value function is sound; the *prompt set* is not a good
  fit for these photographs and confidence does not reveal when it misses.**
- How many images exceeded `max_players` and had segments merged into "rest"?
  **Answered by W01-04 and confirmed through the real pipeline by W01-06:
  zero capped at `max_players = 10`; 3 images get a "rest" player from the
  area threshold.** The run must still report the figure it actually observed.
- What fraction of the test split produced usable output, and why did the rest
  fail?

Record failures and surprises here as readily as successes — a failed masking
check is one of the most valuable entries this week can contain.

---

## W01-13 — The first cluster submission: three of `slurm/README.md`'s assumptions were wrong, and the one that cost a job was mine

- **Date:** 2026-10-23
- **Status:** validated for the four observations below, each read from a
  SLURM log or an `sacctmgr` table; **inconclusive** on everything the run was
  supposed to measure — no image was decomposed, so there is still no
  per-image runtime, no GPU confirmation and no walltime estimate
- **Plan reference:** docs/week-01.md, task 1.1 (`PLAN.md` step 0, "verify
  SLURM array submission with a trivial job"); the first open question at the
  top of this file, which this entry partially answers
- **Run / artifact:** cluster login node `datolith`, compute node
  `adakit.cip.ifi.lmu.de` · account `stud_ifi`, user `dziri`, QOS `normal` ·
  SLURM array job **225119**, `--array=0-1`, partition `NvidiaAll` ·
  `CONFIG=configs/smoke.toml`, fingerprint `4f6b612d9de8` ·
  `slurm/logs/output_logs/output_food_shapley_run_225119_{0,1}.log` and the
  matching `error_logs` · commit `b0945bc` plus the uncommitted task-1.3–1.6
  working tree, copied to the cluster by hand
- **Not a research result.** Nothing here is about food, CLIP or Shapley
  values. It is recorded because the first open question above asks for it
  explicitly — "every later run inherits these settings" — and because two of
  the four findings changed a committed file.

### Observation

Both array tasks started, ran, and exited non-zero in under a minute.

#### 1. Array tasks count individually against `MaxSubmit`, so `--array=0-31` never reaches a node

The first submission attempt was rejected outright:

```
sbatch: error: AssocMaxSubmitJobLimit
```

Measured afterwards with
`sacctmgr show assoc user=$USER format=maxjobs,maxsubmit,grpjobs,grpsubmit -p`:

| Limit | Value |
|---|---|
| `MaxJobs` (running at once) | **15** |
| `MaxSubmit` (submitted at all) | **30** |
| `MaxWall` | none set |

The QOS table carries no per-user limits, so the association limits are the
binding ones. A 32-task array is 32 submitted jobs, not one. A `%` throttle
does **not** help: `--array=0-31%8` caps concurrency, not the submitted count,
and is rejected identically.

`slurm/run_array.slurm` now ships `--array=0-27`, leaving two slots free for
an interactive or smoke job.

#### 2. The compute node has network access, and the model load used it

`stderr` carried the Hub's unauthenticated-requests warning during
`ImageDecomposer` construction — *after* `prepare.sh` had already cached the
same weights under `$FOOD_SHAPLEY_DATA/hf-cache`. So:

- `slurm/README.md`'s "compute nodes may be offline" is **false on adakit**;
  the design was conservative in the right direction but for the wrong reason.
- A cached checkpoint did not prevent a Hub round-trip. At 28 tasks starting
  together this makes the run depend on Hub availability and rate limits at
  exactly its most parallel moment, and a silent re-download would make two
  tasks' weights un-attributable to one hash.

`slurm/run_array.slurm` now sets `HF_HUB_OFFLINE=1`, converting a cache miss
into an immediate loud failure. That is the better trade in one direction
only: a cache miss is fixable on the login node, a rate limit is not.

#### 3. `slurmstepd` cannot create `/tmp/user/<uid>` and falls back to `TMPDIR=/tmp`

An error line to that effect appears before any of our output, so our
`export TMPDIR="${TMPDIR:-...}"` never fires — the variable is already set.
Harmless here, and worth writing down only to stop a future reader treating
it as the cause of something else: `_atomic_save` writes its temporary file
into the *output* directory, not `TMPDIR`, precisely so that the rename is on
one filesystem.

#### 4. The failure that actually killed both tasks was a config mismatch, and the guard caught it correctly

```
FileNotFoundError: no run metadata at
  .../runs/smoke-4f6b612d9de8/run_metadata.json; run the 'prepare' subcommand
  on the login node before submitting the array
```

`prepare.sh` had been run with `configs/week01-clip-blur.toml` and the array
submitted with `CONFIG=configs/smoke.toml`. `prepare.sh` prepares the run
directory for *the config it is given*, so `runs/` contained only
`week01-clip-blur-deb3a76af736`. The two commands named different experiments.

This is the "tasks must not create the run directory" guard (`run.py:456`)
doing its job: two tasks starting together would otherwise have raced to
write two provenance records. The mistake was in the instructions, not in the
code.

### Context and motivation

Task 1.1's cluster step had been deferred since the start of the week, so
`slurm/README.md` described a cluster nobody had submitted to. Its settings
were copied from a script known to work for a different project, which is
evidence but not verification. The run array for task 1.6 inherits every one
of those settings.

### Setup

`bash slurm/prepare.sh configs/week01-clip-blur.toml` on the login node, then
`CONFIG=configs/smoke.toml sbatch --array=0-1 slurm/run_array.slurm`.
Diagnosis from the two stdout logs, the two stderr logs, and `ls runs/`.
`sacct -X --starttime today` returned an empty table and was briefly
misleading: the check was run at 00:04, so the jobs had finished "yesterday".

### Evidence

The strongest line in the whole run is the least dramatic one:

```
array 225119 task 0/2 on adakit.cip.ifi.lmu.de
```

and then a traceback whose innermost frame is `run_task`, reached from
`_command_task` at `run.py:641` — **after** the `ImageDecomposer(...)`
construction on the line above it. Construction loads the FoodSeg103 split
and the CLIP checkpoint. So the traceback is positive evidence for everything
that had to happen first:

| Assumption | Status after 225119 |
|---|---|
| `uv` reachable without `~/.bashrc` (`PATH` export) | **confirmed** |
| `uv run --frozen --no-sync` resolves on a node | **confirmed** |
| `shapiq`, `torch`, `open_clip`, `datasets` import there | **confirmed** |
| FoodSeg103 split loads from `$FOOD_SHAPLEY_DATA` | **confirmed** |
| CLIP checkpoint loads on the node | **confirmed** |
| `-p NvidiaAll` with no `--gres` is accepted | **confirmed** |
| `$SLURM_ARRAY_TASK_ID` / `_COUNT` reach `run.py` | **confirmed** (`0/2`) |
| Log directories exist at submit time | **confirmed** |
| Compute nodes may be offline | **false** (§2) |
| Node-local `TMPDIR` is ours to set | **false** (§3) |
| 32-task array is submittable | **false** (§1) |
| A GPU was allocated and used | **unknown** — nothing printed it |
| Per-image runtime; `-t 04:00:00` is right | **unknown** |

### Interpretation

The submission failed and the day's unknowns mostly closed anyway. A
`FileNotFoundError` thrown by our own guard, at a known line, after the
expensive imports, is close to the most informative failure available: it
proves the whole environment stack without spending GPU time.

The one finding with teeth is §2. Everything else is a number to edit. A
28-way parallel Hub dependency is a reproducibility hazard rather than a
throughput one — the run would still finish, but the provenance record would
claim a hash for weights that were fetched, not read, and nothing downstream
could tell.

§4 is a procedural finding about me, not about the cluster: the two-command
recipe in `slurm/run_array.slurm`'s header names one config twice for exactly
this reason, and the instruction I gave did not.

### Limitations and alternative explanations

- **One node, one submission.** `adakit` is one machine in `NvidiaAll`.
  Another node in the same partition may well have no network access, in which
  case `HF_HUB_OFFLINE=1` is load-bearing rather than merely prudent — and
  would have failed there without §2 ever being observed.
- **The limits in §1 are an association snapshot.** They are not a cluster
  policy document and can be changed by an administrator without notice.
- **§2 identifies that the Hub was contacted, not what for.** The warning is
  consistent with a metadata revision check that would have succeeded offline
  from cache. I did not establish whether any weight bytes crossed the wire;
  `HF_HUB_OFFLINE=1` makes the question moot rather than answering it.
- **Nothing was computed.** No `.npz`, no timing, no GPU confirmation. Every
  "unknown" row above stays unknown.

### Implications

- `slurm/run_array.slurm`: `--array=0-27`, `HF_HUB_OFFLINE=1`, and a
  `device:` line printed at task start — a whole array silently on CPU looks
  exactly like one on GPU until the walltime expires with the split undone.
- `slurm/README.md` records `MaxSubmit`/`MaxJobs` as *measured*, and no longer
  suggests sizing an array to the split.
- `-t 04:00:00` remains an unmeasured guess. It must be corrected from a run
  that actually decomposes images, and the script's "never been measured on
  this cluster" sentence deleted only when that happens.
- `FOOD_SHAPLEY_RUNS` was `/home/d/dziri/food-shapley/runs` — quota-limited
  NFS. It must point at scratch before the real run, which is what
  `slurm/README.md`'s *Disk* table already says.

### Follow-up questions and next steps

1. Re-run the rehearsal with **the same config on both commands**, confirm
   `device: cuda`, and read the real per-task walltime from
   `seff <jobid>` / `sacct --starttime yesterday`.
2. Does another `NvidiaAll` node have network access, or only `adakit`? If
   not, §2's fix was necessary rather than precautionary.
3. Does the cached checkpoint actually satisfy `HF_HUB_OFFLINE=1` on a node?
   Untested — the next submission tests it, and will fail loudly if not.
4. Check `squeue -u $USER` before every submission: the 28-task array leaves
   two slots, and anything already queued eats into them.

### Relation to earlier entries

First entry in this file about the cluster rather than about data or models.
It converts the first open question at the top from "nothing has been
submitted" to a partial answer; the rows still marked **unknown** in §Evidence
are what remains of it.

---

## W01-12 — `shapiq` returns k-SII in a value-dependent shape: exactly-zero entries are dropped, including whole singletons

- **Date:** 2026-10-23
- **Status:** mixed, by part:
  - *validated:* the four `shapiq` behaviours in §1–§3, each pinned by a test
  - *validated (on three images, on CPU):* the two residual-scale measurements
    in §4
  - *preliminary:* that the densified storage format is the right one for
    Milestone 2's cross-model comparison, which has not been attempted
- **Plan reference:** docs/week-01.md, task 1.5 (`PLAN.md` step 4, "compute
  exact SV, k-SII and Möbius via `shapiq.ExactComputer`; build in the two
  checks now"); `PLAN.md`'s risk note, "catching a masking bug here is much
  cheaper than after the full run"
- **Run / artifact:** `configs/week01-clip-blur.toml`, fingerprint
  **`9171f04cd6a0`** (was `0c2f69fe4d3a` before `interaction_order` was added
  to `ExperimentConfig`; `configs/smoke.toml` moved `8ff0d4d5cbbc` →
  `eb657bff090b` for the same reason) · commit `b0945bc` plus the uncommitted
  working tree adding `indices.py`, `tests/unit/test_indices.py`,
  `tests/unit/synthetic_game.py` and
  `tests/integration/test_indices_on_real_images.py` · checkpoint
  `open_clip/ViT-B-32-quickgelu/openai` · FoodSeg103 test split images
  **32, 158, 242** · CPU, float32 · `shapiq` version as locked in `uv.lock`
- **Fingerprint note, 2026-10-23 (same day, later).** Task 1.6 added
  `max_images` to `ExperimentConfig`, which moved the fingerprints again:
  `week01-clip-blur` `9171f04cd6a0` → **`deb3a76af736`** and `smoke`
  `eb657bff090b` → **`4f6b612d9de8`**. The values above are left as written
  because they identify the configuration this entry's numbers were measured
  under. Nothing measured here changes — `max_images` selects *which* images a
  run covers and does not touch any stored number — but the run directory a
  future reader finds on disk will carry the newer name.

### Observation

#### 1. `shapiq` omits k-SII entries whose value is exactly zero

The number of rows `ExactComputer(game)("k-SII", order=k)` returns is a
function of **the values**, not of the game's size. On a 4-player synthetic
game with weights `[1.0, 0.0, 300.0, 4000.0]` and a single interacting pair
`(0, 2)`:

| | Count |
|---|---|
| Subsets of size ≤ 2 of a 4-player game | 11 |
| Rows actually returned | **6** |

Returned: `(), (0,), (0,2), (0,3), (2,), (3,)`. **Missing: `(1,)`** — a
player's own main effect — along with `(0,1)`, `(1,2)`, `(1,3)` and `(2,3)`.

`SV` and `Moebius` came back dense on every game tried, but nothing in the
API promises that, and the k-SII case proves the promise is not made.

#### 2. The k-SII lookup is not in powerset order either

For a 3-player game at order 2:

```
{(): 0, (0,2): 1, (0,): 2, (1,): 3, (2,): 4}
```

and for 4 players the key order is
`[(), (0,2), (0,3), (0,), (1,2), (1,3), (1,), (2,3), (2,), (3,)]`. Neither is
`shapiq.utils.powerset`'s ascending-size-then-lexicographic order, which
W01-11 §4 recorded for coalitions. The `interaction_lookup` dict is the only
authority on which row is which.

#### 3. `InteractionValues.values` returns a fresh array on every access

`result.values[i] += x` is a silent no-op. Found by writing a test fixture
that perturbed a returned value to prove the efficiency check could fail, and
watching it not fail. The fixture now substitutes its own object.

#### 4. Two small real-data residuals that a synthetic game cannot show

Both are the rebatching effect of W01-11 §3, seen from new angles:

| Quantity | Image 32 | Image 158 | Image 242 |
|---|---|---|---|
| Normalised `v(∅)` inside the powerset | `1.56e-06` | `−2.40e-06` | `6.19e-06` |

and `v(N)` inside the powerset differs from `target.log_prob` by between
`2.1e-08` and `5.5e-07` nats across the same three images.

The first means `span = v(N) − v(∅)` is **not** interchangeable with
`v(N)` on real data, though it is on any synthetic fixture. The second means
`baseline + span == target.log_prob` holds to ~1e-07, not bit for bit —
even though scoring the grand coalition *alone* reproduces it exactly
(W01-11, "Relation to earlier entries").

#### 5. Mutation testing: three rounds, four survivors, all now killed

The two new suites were checked by deliberately breaking the production code:

| Mutation | Initially caught? |
|---|---|
| Efficiency summed over *all* Shapley rows, including the empty set | **no** → killed by a fixture that shifts only the empty-set entry |
| Möbius block stored truncated to `interaction_order` | yes |
| `span` read as `v(N)` instead of `v(N) − v(∅)` | **no** on synthetic, yes on real → killed by a new integration test |
| A partial (non-powerset) coalition evaluation accepted | **no** → killed by a computer stub that hides half the lookup |
| k-SII rows read in returned order rather than densified | yes (§1, §2) |

The third is the interesting one: the mutant is genuinely undetectable on a
fixture where `v(∅)` normalises to exactly 0, and detectable on every real
image, for the reason in §4.

### Context

Task 1.5 is the first place the project stores a number it intends to report.
Everything before it computed values and threw them away. The question this
entry answers is not "are the indices correct" — `tests/unit/test_indices.py`
settles that against a closed-form game — but "does the *stored shape* mean
the same thing from one image to the next".

### Setup

`compute_indices(game, order=config.interaction_order)` over the full
powerset of each image, via one `shapiq.ExactComputer` shared by all three
index families (§1 of the evidence below confirms the game is evaluated
`2**n` times in total, not `3 × 2**n`). `interaction_order` is clamped to
`min(order, n_players)`, which W01-11's follow-up flagged as necessary.
Synthetic measurements use `tests/unit/synthetic_game.py`, a game whose value
is a known function of the coalition it renders.

### Evidence

- `tests/unit/test_indices.py` — 41 tests, no model involved.
  `test_shapiq_drops_exactly_zero_k_sii_entries`,
  `test_the_k_sii_lookup_is_not_in_powerset_order` and
  `test_a_dropped_entry_is_restored_as_zero` pin §1 and §2; the `Perturbed`
  stand-in in that file documents §3.
- `tests/integration/test_indices_on_real_images.py` — 24 tests against the
  real checkpoint (`-m gpu`). §4's two measurements are assertions there:
  `test_the_normalised_empty_coalition_is_near_zero_but_not_zero` and
  `test_the_baseline_recovers_the_target_s_own_log_probability`.
- Full suite **534 passed**; `ruff check`, `ruff format --check` and `mypy`
  clean.

### Interpretation

§1 is the finding that changed the design. Read naively, a downstream
statistic like "how many ingredient pairs interact?" would be computed over a
row set that *shrinks when the answer is zero* — the exact direction that
inflates an interaction rate, and with no symptom: every value present would
be correct. `indices.py` therefore rebuilds every block to a fixed shape
determined by `n_players` and `order`, with omissions restored as `0.0` and
any unexpected key raising. The shape of a stored file is now a function of
the configuration alone.

§4 is interpretation-light: the numbers are ~1e-06 against a ~4-nat span, and
nothing this project would report is sensitive to them. What they buy is the
knowledge that two tolerances in the suite are load-bearing rather than
defensive, which is the difference between a check that can fail and a check
that has been pre-widened into decoration.

### Limitations and alternative explanations

- §1 was found on a *synthetic* game constructed to have an inert player.
  Real images are unlikely to produce an exactly-zero interaction, so the
  practical blast radius may be small — but "unlikely" is not "never", and
  the densification costs nothing.
- The 6-of-11 figure is one game at one order. Whether the omission rule is
  "exactly zero" or some internal threshold was not determined; the fix does
  not depend on which.
- §4's residuals are three images on CPU, inheriting every limitation
  W01-11 lists. The GPU question is still unmeasured.
- §5 is mutation testing by hand, not a tool. It establishes that five
  specific bugs are caught, not that the suite is complete.

### Implications

- Stored `.npz` files have a shape fixed by `(n_players, interaction_order)`.
  Any analysis may index them positionally against
  `powerset(range(n), max_size=order)` without consulting `shapiq`.
- `interaction_order` is now an `ExperimentConfig` field and therefore in the
  run fingerprint, which is why both configs' fingerprints moved. The
  pre-1.5 values `0c2f69fe4d3a` and `8ff0d4d5cbbc` remain correct for every
  entry written before this one.
- W01-11's prediction that the task-1.6 residual "should be exactly `0.0`
  every time" is **too strong**; see the correction at the head of that
  entry. `DEFAULT_EFFICIENCY_TOLERANCE = 1e-9` is the gate, with six orders
  of magnitude of measured headroom.
- Any future check of this kind should be tried on real values before it is
  trusted, because §4 §5 together show a synthetic fixture can make a real
  bug invisible.

### Relation to earlier entries

- **Corrects W01-11 §1** on the exactness of the residual — not the
  measurement, which stands, but the generalisation drawn from it. See the
  dated correction there.
- Uses **W01-11 §3**'s rebatching bound to explain both residuals in §4.
- Depends on **W01-09**: `ingredient_indices` is stored alongside every
  decomposition so that an ingredient-level statistic cannot silently count
  the scene.

### Follow-up questions

- Does any real image produce an exactly-zero k-SII entry, and therefore a
  densified row that is a genuine zero rather than a present one? Worth a
  count during task 1.6; it is one line and it bounds §1's practical reach.
- Is `shapiq`'s omission rule a hard zero or a threshold? Only matters if a
  near-zero value is being dropped, which would be a silent loss rather than
  a shape change.
- Does the residual stay inside `1e-9` at 11 players? Six is the largest
  measured, and the error accumulates over `2**n` terms.

---

## W01-11 — Efficiency holds exactly on real games; the only inexact thing is CLIP's response to rebatching, at ~8e-06 nats

> **Correction, 2026-10-23 (W01-12).** §1's measurement stands, but the
> generalisation does not: the residual is exactly `0.0` only for the
> particular summation W01-11 used — reading `shapiq`'s own Shapley dict.
> Computing it as task 1.6 will, by summing the *densified* main-effect
> vector, gives `0.0`, `−4.441e-16`, `0.0` on the same three images. The
> difference is float accumulation order, six orders of magnitude inside the
> `1e-9` tolerance. The follow-up below that predicts "exactly `0.0` every
> time" in task 1.6 should be read as "inside `DEFAULT_EFFICIENCY_TOLERANCE`".

- **Date:** 2026-10-20
- **Status:** mixed, by part:
  - *validated:* exact efficiency on three real games, at zero tolerance
  - *validated (as a bound on this sample, on CPU):* the size of the
    rebatching drift and its effect on Shapley values
  - *preliminary:* that the bound transfers to GPU inference, which has not
    been measured because nothing has run on the cluster
- **Plan reference:** docs/week-01.md, task 1.4 (`PLAN.md` step 3); `PLAN.md`'s
  risk note for that step, "coalition ordering must match what `shapiq`
  expects internally"; the efficiency check that gates task 1.6
- **Run / artifact:** `configs/week01-clip-blur.toml`, fingerprint
  **`0c2f69fe4d3a`** · commit `b0945bc` plus the uncommitted working tree
  adding `game.py`, `tests/unit/test_game.py` and
  `tests/integration/test_game_on_real_images.py` · checkpoint
  `open_clip/ViT-B-32-quickgelu/openai`, `parameter_sha256 a3ce3c4a2245…` ·
  FoodSeg103 test split images **32, 158, 242** (chosen for small player
  counts, so the whole powerset is affordable on CPU) · CPU, float32 ·
  diagnostics `/tmp/diag_eff2.py`, `/tmp/diag_eff3.py` (not committed; the
  numbers below are re-asserted by the integration tests, which are)

### Observation

#### 1. Exact Shapley values satisfy efficiency with no tolerance at all

Full powerset, real CLIP, exact `shapiq.ExactComputer`:

| Image | Players | `v(N) − v(∅)` | `sum(SV) − (v(N) − v(∅))` |
|---|---|---|---|
| 32 | 6 | 7.337981364127453 | **`+0.000e+00`** |
| 158 | 3 | 3.1933557565675375 | **`+0.000e+00`** |
| 242 | 3 | 4.017425444613295 | **`+0.000e+00`** |

Not "within tolerance" — bit-identical. This is `PLAN.md`'s first validity
check, and the strongest form of it available.

#### 2. It only holds exactly if both ends come from the same evaluation

The first version of the test read `v(N)` from a fresh one-row call instead,
and **failed**. Same game, same coalition, different batch:

| Image | `sum(SV) − (v(N) − v(∅))`, `v(N)` re-scored alone |
|---|---|
| 32 | `−1.014e-06` |
| 158 | `+2.345e-06` |
| 242 | `+6.170e-06` |

The same effect shows up in `shapiq`'s own bookkeeping: after an exact
computation `ExactComputer.baseline_value` is `1.56e-06`, `−2.40e-06` and
`6.19e-06` rather than the exactly `0.0` that normalisation should produce.
It is the `v(∅)` computed at construction time, in a batch of one, compared
against the `v(∅)` sitting in row 0 of the powerset batch.

#### 3. CLIP's float32 forward pass is not batch-invariant

Scoring every coalition of each image at `chunk_size=1` and at
`chunk_size=32`:

| Image | Max abs difference, coalition values | Max abs difference, exact Shapley values | Relative to `v(N) − v(∅)` |
|---|---|---|---|
| 32 | 8.124e-06 | 1.839e-06 | 2.5e-07 |
| 158 | 7.444e-06 | 2.845e-06 | 8.9e-07 |
| 242 | 7.316e-06 | 2.441e-06 | 6.1e-07 |

Within a *fixed* batching the game is bit-deterministic: re-scoring the same
matrix twice gives `0.00e+00` difference, as W01-10 found for the value
function. The drift is entirely a function of what a coalition is batched
with.

#### 4. The coalition-ordering risk `PLAN.md` flags is real and is not detectable numerically

A game that scores row `i` as some *other* subset, consistently, still
satisfies efficiency exactly — §1 would pass unchanged. The unit suite
therefore does not check numbers against numbers: it runs a value function
that recovers the coalition from the rendering by byte-matching against every
coalition rendered independently in the test, with deliberately asymmetric
per-player weights, and compares `shapiq`'s output to the closed form. Four
deliberate mutations were introduced to confirm the suite can fail: reversing
the column convention, sorting rows by coalition size, subtracting the
baseline twice, and replacing the baseline with `0.0`. The first three were
caught immediately. **The fourth was caught by only one test, and that test
was about call counts, not about values** — because the fixture had
`v(∅) == 0`, which makes "subtract the baseline" and "subtract nothing"
indistinguishable. Fixing the *fixture* (`base = −5.25`, in the range W01-10
measured for real images) brought it to five failing tests.

For the record, `shapiq.utils.powerset` enumerates **ascending by coalition
size, then lexicographically** — `(), (0,), (1,), (2,), (0,1), (0,2), (1,2),
(0,1,2)` — not by binary counting. Nothing in `game.py` depends on that, which
is the point, but anyone matching a stored value vector against an index will
need it.

### Context

Task 1.4 joins the three halves built in 1.2 and 1.3 — players, the blur
operator, the value function — into a `shapiq.Game`. Until this entry no
Shapley value had ever been computed in this project, so `PLAN.md`'s
efficiency check had never actually been run against anything.

### Setup

`IngredientGame.for_image` builds one game per image: target fixed by the
model's top-1 on the unmasked photograph, `v(S) = log p(target | only S
visible)`, blur-removal operator, `normalize=True` with an explicit
`normalization_value = v(∅)`. Coalitions are rendered and scored in chunks of
32. Measurements on CPU, float32, three images, full powerset (64 + 8 + 8 =
80 coalitions).

### Evidence

- `tests/unit/test_game.py` — 26 tests, all passing, no model involved
- `tests/integration/test_game_on_real_images.py` — 22 tests, all passing
  against the real checkpoint (`-m gpu`); §1 and §3 are assertions in there,
  so they are re-checked rather than merely recorded
- Full suite **463 passed** (420 without the `gpu` marker); `ruff check`,
  `ruff format --check` and `mypy` clean

### Interpretation

Efficiency passing at zero tolerance says the game's arithmetic and `shapiq`'s
agree completely. It says *nothing* about whether the coalitions are labelled
correctly, which is why §4 exists and why the unit suite is built the way it
is; a permuted game passes §1 perfectly.

§3 is a limitation of the measurement apparatus, not of the method. 8e-06 nats
against a ~4-nat span is ~2e-06 relative, four orders of magnitude below any
difference between two ingredients that this project would report. It is,
however, not zero, and the correct response was to measure it rather than to
widen a tolerance until the test went green — the claim "`chunk_size` is a
throughput knob" is now backed by a number instead of by an assumption.

### Limitations

- **Three images.** Chosen for small player counts, not sampled. §1 and §3 are
  correctness properties that should hold for every image, but "should" is not
  "was measured".
- **CPU only.** GPU kernels reduce in different orders and often use TF32;
  the drift in §3 could plausibly be larger there. Nothing has run on the
  cluster, so this is unmeasured.
- §3 compares two chunk sizes, not two machines. Cross-machine reproducibility
  of a stored value vector is a stronger property and has not been tested.
- The entry says nothing about whether the Shapley values are *meaningful* —
  only that they are correctly computed. The target is still the borrowed
  top-1 of W01-10, and every number here explains the model's prediction
  rather than the dish.

### Implications

- `chunk_size` stays out of `ExperimentConfig`, and therefore out of the run
  fingerprint, with §3 as the justification. Two runs differing only in
  batching are the same experiment.
- Any future efficiency check — in task 1.6, and in the Milestone 2
  comparisons — must read `v(N)` and `v(∅)` out of the same evaluation, or it
  is measuring §3 instead.
- Stored results should record the chunk size used anyway, so that a future
  bit-level comparison that fails can be attributed rather than investigated.
- When task 1.6 runs the whole split, the per-image efficiency residual is
  worth storing: it should be exactly `0.0` every time, and anything else is a
  bug rather than noise.

### Relation to earlier entries

- Extends **W01-10**: determinism under re-scoring, measured there for the
  value function, survives into the game — but only at fixed batching, which
  W01-10 did not distinguish.
- Depends on **W01-09**: `v(N)` equals the target's own log-probability *bit
  for bit* only because the scene is a player, so the grand coalition's
  rendering is the photograph. That identity is asserted in the integration
  suite and passes on all three images.
- Does not contradict anything.

### Follow-up questions

- Does the drift in §3 hold at the same order on GPU, and does TF32 need to be
  disabled for stored results to be comparable across machines?
- Is the efficiency residual exactly `0.0` across all 2,135 images, or does it
  degrade with player count? Six players is the largest game measured here;
  the split reaches 11 (`max_players = 10` ingredients plus the scene).
- `ExactComputer('k-SII', order=2)` raises `IndexError` on a one-player game.
  No image in this split has one: re-running player construction over all
  2,135 images under the current config gives 2–11 players (2 on 7 images, 11
  on 1, 91,828 coalitions in total — confirming the figure in W01-06's
  superseding note). Task 1.5 must still clamp `order` to
  `min(order, n_players)` rather than rely on a dataset property.

---

## W01-10 — The value function works and the game is non-degenerate, but the dish target itself is borrowed and cannot be filtered

- **Date:** 2026-10-19
- **Status:** mixed, by part — stated separately because they do not stand or
  fall together:
  - *validated:* the QuickGELU mismatch and its effect on the target; the
    dynamic range of the game; determinism of re-scoring
  - *validated (as a statement about this sample):* the top-1 confidence
    distribution; the vocabulary-coverage figures
  - *inconclusive, leaning negative:* whether confidence can be used to filter
    inapt targets — it cannot, and no replacement filter is known
  - *preliminary:* the single-ingredient recovery signal
- **Plan reference:** docs/week-01.md, task 1.3 (`PLAN.md` step 2); success
  criterion "Prompt sanity check"
- **Run / artifact:** `configs/week01-clip-blur.toml`, fingerprint
  **`0c2f69fe4d3a`** (was `f6f33ec14b1f` before `prompt_template` and the
  checkpoint tag became fields) · commit `b0945bc` plus the uncommitted working
  tree adding `prompts.py`, `scoring.py`, `value.py` · checkpoint
  `open_clip/ViT-B-32-quickgelu/openai`, `parameter_sha256
  a3ce3c4a2245ed2a572d2eb864a62f0a4ca62b3123b83f8a52a523c3b1dd32a4`, open_clip
  3.3.0, torch 2.14.1+cu130 · prompt set `ethz/food101@83488de741c1`,
  `names_sha256 92d5cd07a785…` · FoodSeg103 test split, manifest
  `4d2b3127f607…` · 200 images, seeded sample, CPU

### Observation

#### 1. The checkpoint name in the config was running the weights through the wrong non-linearity

`ViT-B-32/openai` — the name the configs carried from the start — emits only
this, and then proceeds:

```
UserWarning: QuickGELU mismatch between final model config (quick_gelu=False)
and pretrained tag 'openai' (quick_gelu=True)
```

OpenAI's CLIP weights were trained with QuickGELU; since open_clip 2.24 the
plain `ViT-B-32` config uses standard GELU. Scoring 40 real test images under
both architectures:

- **The two disagree on the top-1 dish for 15 of 40 images (37.5 %).**
- The top-1 *is* the target of the decomposition (see §2), so 37 % of images
  would have had their Shapley values computed for a dish the correctly
  configured model does not predict.

#### 2. There is no ground-truth dish, so the target is the model's own top-1

FoodSeg103 annotates ingredients. It has **no dish label in any column** —
features are `image`, `label` (a mask image), `classes_on_image`, `id`. The
value function therefore scores

> `v(S) = log softmax(CLIP logits over the 101 Food-101 prompts)[ŷ]`,
> `ŷ = argmax` on the **unmasked** photograph.

#### 3. On 200 unmasked images the model is confident and the game is non-degenerate

| Quantity | Result |
|---|---|
| top-1 probability | median **0.580**, mean 0.581, min **0.167** |
| images with top-1 < 0.10 | **0 / 200** |
| images with top-1 < 0.25 | 19 / 200 (9.5 %) |
| `v(N) − v(∅)` | **positive on 200 / 200**, median **+4.13**, mean +4.18 nats |
| re-score of the same images | **bit-identical**, max abs difference `0.00e+00`; target stable |

#### 4. Blur usually does remove the evidence — but there is a tail

Probability assigned to the target on the **fully blurred** frame, i.e. `v(∅)`
exponentiated, against the 1/101 = 0.0099 chance level:

| Quantity | Result |
|---|---|
| median `p(target \| blurred)` | **0.0071** — *below* chance |
| images above 0.10 | **5 / 200 (2.5 %)** |
| maximum | **0.416** |

#### 5. The prompt set does not fit the photographs, and confidence does not reveal when

| Quantity | Result |
|---|---|
| distinct Food-101 classes used across 200 images | **66 / 101** |
| most frequent target | **`pork_chop`, 26 / 200 = 13 %** |
| worst case inspected | a frame containing cake → `baby_back_ribs` at **p = 0.79** |

`pork_chop` fires on plates of chicken and rice. The confidence distribution in
§3 and the misfires here are **not separable**: the minimum confidence over 200
images is 0.167, so no threshold rejects the bad cases without rejecting almost
everything, and the worst case observed is also one of the most confident.

#### 6. Preliminary: one ingredient alone typically recovers most of the span

Fraction of `v(N) − v(∅)` recovered by unblurring a single player alone,
i.e. `(v({i}) − v(∅)) / (v(N) − v(∅))`:

| Quantity | Result |
|---|---|
| best single **ingredient** (scene excluded) | median **0.627** |
| images where the best ingredient recovers ≥ 0.5 | **121 / 200 (60.5 %)** |
| the **scene** player alone | median **−0.080** |
| images where the scene beats every ingredient | 23 / 200 (11.5 %) |
| corr(player area fraction, fraction recovered) | **+0.259** |

The scene figure is negative: unblurring only the plate, table and surroundings
makes the target *less* likely than a fully blurred frame.

### Context and motivation

Task 1.3's acceptance criterion is a sanity check: "CLIP ranks the
ground-truth dish plausibly on a few obvious unmasked images". Writing the
value function forced the prior question — *which* dish — and the criterion
turned out to presuppose an annotation this dataset does not have. The checks
above are what was run instead. §1 was not sought: it came from reading a
warning instead of letting `filterwarnings = ["error"]` hide it in tests while
production code ignored it.

### Setup

- `food_shapley.value.DishValueFunction.from_config`, real checkpoint, CPU,
  no GPU required. `log_softmax` row-wise in **float64**, because efficiency
  will sum up to 2^11 = 2,048 terms per image.
- 200 images drawn by the config seed; the three named integration images (32,
  158, 242) are fixed by index.
- §1 scored the same 40 images twice, once per architecture, everything else
  held equal.
- §6 required 1 + P forward passes per image (empty coalition, grand
  coalition, each singleton) — **not** a Shapley computation; `shapiq` has not
  been wired in yet.

### Evidence

- `tests/integration/test_value_on_real_images.py` — **21 tests, all passing**
  against the real checkpoint (`-m gpu`, 21.23 s on CPU). Includes
  `test_the_underspecified_openai_name_is_refused`, which asserts
  `ViT-B-32/openai` raises rather than warns, and the three named top-1
  assertions.
- `tests/unit/test_scoring.py` (36 tests) checks the logit formula in closed
  form against a stub model with `logit_scale = log(100)`, batch invariance,
  and the geometry contract. `tests/unit/test_value.py` (30) checks
  `log_softmax` against `torch.log_softmax` and the target/tie-breaking rules.
  `tests/unit/test_prompts.py` (32) covers the manifest and hash paths offline
  against a fake `datasets` module.
- Full gate run: `uv run ruff check . && uv run ruff format --check . &&
  uv run mypy && uv run pytest` — **415 tests pass** (394 of them without the
  `gpu` marker); `pre-commit run --all-files` — all 12 hooks pass.
- `uv run python -m food_shapley.scoring --config configs/week01-clip-blur.toml`
  prints the checkpoint provenance quoted in the header, and is the step
  `slurm/prepare.sh` now uses to warm the weight cache on a login node.

### Interpretation

*Interpretation, not measurement.*

- **The value function is fit for purpose and the game is worth computing.**
  §3's "positive on 200/200, median +4.13 nats" is the property the whole
  decomposition rests on: if blur had failed to move the score, every Shapley
  value would have been noise around zero. It did not.
- **§1 is the most important result in this entry**, and it is an engineering
  result rather than a scientific one. A 37 % disagreement rate on the target
  would have invalidated roughly a third of every per-ingredient number,
  undetectably — the efficiency check would have passed perfectly, because a
  game built on the wrong target is still internally consistent. This is the
  same class of silent failure as the mask-misalignment risk in W01-05, and it
  arrived as a warning rather than an error. Milestone 2 loads several
  checkpoints, so this is now a hard failure in `ClipScorer.load`.
- **§5 is the real limitation of the Week-1 design.** The pipeline answers
  "which ingredients explain the model's prediction", which is a well-posed
  question. It does *not* answer "which ingredients explain the dish", because
  for an unknown fraction of images the predicted dish is not the dish. I
  expected confidence to bound that fraction; it does not, which means the
  fraction is currently **unmeasured** rather than small.
- **§4 refines W01-07's silhouette confound rather than confirming or refuting
  it.** The median blurred frame scores *below* chance, so blur is doing its
  job on most images — which is better than W01-07's visual impression
  suggested. But 2.5 % of images retain > 0.10 and one retains 0.416, so the
  confound is real and confined to a tail, not pervasive and not absent.
- **§6 is a genuine first signal for RQ1** and it points toward "one telltale
  ingredient" over "a genuine combination": a median of 0.627 of the span from
  a single ingredient is a lot. It is also exactly the kind of number that a
  confound could manufacture, so it is marked preliminary and the two obvious
  confounds are measured rather than waved at. The scene result is a useful
  negative control: if the model were merely responding to *how much of the
  frame is sharp*, the scene — 39 % of the pixels (W01-09) — would score high.
  It scores slightly negative.

### Limitations and alternative explanations

- **§6 is not a Shapley value.** `v({i}) − v(∅)` is a singleton marginal, which
  equals a Shapley value only in a game with no interactions — precisely the
  thing this project exists to measure. Taking it as evidence about main
  effects versus interactions is circular; it is reported as "a single
  ingredient suffices to recover the score", which is a statement about the
  value function, not a decomposition. The real answer waits for task 1.5.
- **The area correlation of +0.259 is weak but not nothing.** It leaves open
  that part of §6 is "bigger regions carry more pixels", and `r` over 200
  images with a bounded response is a crude instrument. A size-matched
  comparison would be better.
- **200 images of 2,135, one checkpoint, one prompt template, one removal
  operator.** Every number here is conditional on all four. The template was
  taken from CLIP's own Food-101 prompt (not tuned), so template sensitivity
  is entirely unmeasured.
- **§4 does not test what W01-07 asked for.** W01-07's follow-up was to compare
  `v(∅)` against a **uniform-grey image**; this compares it against the
  **1/101 uniform-probability chance level**, which is a different and weaker
  control. A grey-image baseline would isolate "blur leaks the silhouette" from
  "this dish is a priori likely"; it has not been run.
- **"Plausible" in §5 is my judgement from 224 px panels**, on a handful of
  images, with no dish ground truth — the same weakness as W01-08. The 13 %
  `pork_chop` rate is a measurement; the claim that it is *wrong* on chicken
  and rice is an inspection of a few of those 26 images.
- **§1's 37.5 % is 15 of 40 images**, so its confidence interval is wide
  (roughly 23–54 %). The qualitative conclusion — "this changes the target on a
  large fraction of images" — is robust to that; the number should not be
  quoted as precise.
- No Shapley value, interaction index or efficiency check exists yet. Nothing
  here says the decomposition is correct, only that there is a non-degenerate
  game to decompose.

### Implications

- **Every result must be framed as explaining the model's prediction**, not the
  dish. The paper needs this sentence explicitly; it is a scope statement, not
  a caveat to bury.
- **The target is model-dependent**, so Milestone 2 cannot compare per-class
  values across model families — two models may be decomposing different
  dishes on the same photograph. It must compare interaction *structure*
  (main-effect share, k-SII distribution), which is well defined regardless of
  which dish each model chose. This constrains the compositionality index
  before it is designed.
- **Store `ŷ`, its probability, and the checkpoint's `parameter_sha256` in
  every per-image `.npz`.** Without the target, a stored game cannot be
  interpreted at all; without the weight hash, §1 could recur unnoticed across
  a resubmission.
- **A dish-labelled dataset is the highest-value fix available** and it is
  cheap relative to its benefit: it would convert §5 from an unmeasured
  limitation into a measured accuracy figure, and allow the "model is right"
  and "model is wrong" populations to be analysed separately. Worth raising
  before Milestone 2 commits to Food-101.
- `slurm/prepare.sh` must cache the checkpoint on the login node (step 5, now
  implemented): §1's guard raises at load time, and a compute node with no
  network cannot recover from either a missing weight file or a bad name.

### Relation to earlier entries

- **W01-07's follow-up 3** ("quantify the silhouette confound once a model
  exists") is **partially** addressed by §4 — see the limitation above; the
  grey-image baseline it actually asked for is still open.
- **W01-09's follow-up 2** ("check whether the scene's value correlates with
  its area") is **pre-answered in the negative** by §6: the scene's singleton
  marginal is slightly *negative* at a median of −0.080 despite being 39 % of
  the frame, which is evidence against "the model responds to how much of the
  frame is blurred". W01-09's follow-up 1 — the scene's actual Shapley value
  and its interactions — still needs task 1.5.
- **W01-08** (ingredient names are sometimes wrong) and §5 here are the same
  kind of problem at two levels: FoodSeg103's ingredient names are unreliable
  on an unmeasured minority, and Food-101's dish names are unreliable on an
  unmeasured minority. Aggregate statistics that never read a name are immune
  to both; every named claim inherits both.

### Follow-up questions and next steps

1. **Run the grey-image baseline** W01-07 asked for, on the same 200 images:
   `p(target | uniform grey)` versus `p(target | blurred)`. Cheap, and it
   separates silhouette leakage from dish priors. Do it before Week 2 reads
   anything into small coalition values.
2. **Bound §5.** Hand-label the dish on ~50 images, stratified by predicted
   class, and report top-1 accuracy. That converts the headline limitation from
   "unmeasured" to a number with an interval, which is what a reviewer will
   ask for.
3. **Re-measure §6 as a real Shapley decomposition** in task 1.5, and keep both
   numbers: a large gap between "best singleton marginal" and "largest Shapley
   value" is itself a measure of interaction.
4. **Size-match the §6 comparison** — within-image, compare ingredients of
   similar area fraction — to settle the +0.259 correlation.
5. Record the per-image wall-clock cost of the real forward passes, which
   W01-06's follow-up 1 asked for and this entry still does not supply (§6
   ran 1 + P passes per image on CPU, which is not a useful estimate for the
   GPU array job).

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
