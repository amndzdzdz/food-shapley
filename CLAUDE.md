# CLAUDE.md

Instructions for Claude Code working in this repository.

## Project purpose

This project computes **exact Shapley values and Shapley interactions over
semantic food-ingredient segments** (FoodSeg103 masks → CLIP → `shapiq`) to
determine whether a vision-language model recognises a dish from one telltale
ingredient or from a genuine combination of ingredients. Milestone 2 extends
this across model families and contrasts an interaction-based
"compositionality index" against accuracy-based benchmarks (SugarCrepe, ARO).

The scope is defined by [`PLAN.md`](PLAN.md). Treat it as the source of truth.
Do not invent research requirements it does not contain.

**The goal is to produce at least a workshop paper.** Development therefore
prioritises:

- **Scientific rigor** — claims follow evidence, not the other way round.
- **Reproducibility** — any number must be reconstructible from a config, a
  commit, and an input.
- **Reliable, well-tested implementations** — a masking bug invalidates every
  downstream result.
- **Transparent reporting** — negative and inconclusive results are reported.
- **Traceability** — research question → implementation → experiment →
  observation → conclusion, each link visible in the repository.

A complete repository skeleton and passing tests say nothing about the quality
of the research. Never describe the project as having publication-quality
results on the basis of its infrastructure.

## Current state

Infrastructure, data acquisition, the segment→player mapping, the CLIP value
function, the coalition game and the exact indices. `src/food_shapley/`
contains configuration,
seeding, artifact layout, run provenance, a SLURM self-test (`selftest.py`,
array mechanics without a dataset or a model), the FoodSeg103 downloader
(`download.py`), the dish prompt set (`prompts.py`), the node environment
check (`envcheck.py`), split loading (`dataset.py`), player construction under
CLIP's geometry (`segments.py`), the blur removal operator (`masking.py`), the
coalition contact-sheet renderer (`preview.py`), CLIP loading and zero-shot
scoring (`scoring.py`), the value function (`value.py`), the `shapiq.Game`
subclass (`game.py`), the exact SV / k-SII / Möbius decomposition
(`indices.py`) and the full-split job array (`run.py`). The FoodSeg103 test
split and the Food-101 prompt set have both been downloaded and verified.

**No experiment has been run and no result exists.** Week 1 tasks 1.2, 1.3,
1.4 and 1.5 are done; 1.1 is still open on its cluster-submission step, and
nothing has ever been submitted to a SLURM cluster. Exact Shapley values,
k-SII and Möbius coefficients have now been computed — on **three** images, as
a correctness check, not as a result (W01-11, W01-12). Task 1.6's machinery
(`run.py`, `slurm/run_array.slurm`) is written and rehearsed locally on three
real images, but **the full run has not been executed** and is blocked behind
1.1. No `.npz` exists outside a temporary test directory.

`game.py` deliberately has no opinion about coalition ordering: `shapiq` passes
a boolean `(n_coalitions, n_players)` matrix and gets one value per row, in row
order. Do not add sorting, grouping, deduplication or caching by coalition — a
game that consistently mislabels its coalitions satisfies the efficiency check
perfectly, so no downstream number can catch it. `chunk_size` is a runtime
argument and **not** an `ExperimentConfig` field: it is a throughput knob, and
putting it in the run fingerprint would make two runs of the same experiment
look different. It is not quite inert — CLIP's float32 forward pass shifts a
coalition value by up to ~8e-06 nats under rebatching (W01-11) — so any
efficiency check must read `v(N)` and `v(∅)` from the same evaluation.

`indices.py` rebuilds every index block to a shape fixed by `n_players` and
`interaction_order`, with a boolean membership matrix saying which players
each row belongs to. Do not store what `shapiq` returned: its
`interaction_lookup` is not in powerset order, and it **omits k-SII entries
whose value is exactly zero** — 6 of 11 rows on a 4-player test game,
including a player's own singleton (W01-12). A downstream statistic reading
those rows directly would take a value-dependent sample with no symptom.
`interaction_order` *is* an `ExperimentConfig` field, unlike `chunk_size`,
because it changes which numbers exist.

`run.py` is the full-split array and is almost entirely bookkeeping; the
research computation is one `compute_indices` call per image. Four invariants
hold it together, and each exists because its absence fails silently. **One
output file per image, never one per task** — so a partial failure is
re-runnable per image. **A stored file does not depend on how the work was
divided** — the image set comes from the config alone and each image's seed is
derived from its own label, which is what makes a resubmission at a different
`--array` a resumption and not a second experiment; do not derive anything
from the task id. **Writes go through `_atomic_save`** — a truncated `.npz`
left by a killed task would be skipped by the resumed run as "already done".
**`verify` exits non-zero on any problem at all**; the "≥ 99 % coverage"
criterion in `docs/week-01.md` is the bar for *reporting* a run, not for this
command passing. Shards are strided, not contiguous, because per-image cost
spans three orders of magnitude.

`max_images` and `interaction_order` are both `ExperimentConfig` fields; both
changed the fingerprints, which are now `deb3a76af736` for
`configs/week01-clip-blur.toml` and `4f6b612d9de8` for `configs/smoke.toml`.
`max_images` takes a **prefix, not a sample**, so a truncated run is a
rehearsal device and never a basis for a reported statistic — it is a config
field rather than a flag precisely so that a 20-image rehearsal cannot share a
directory with the real run.

The value function is `v(S) = log p(target dish | coalition S visible)` over
all 101 dish prompts. **FoodSeg103 has no dish annotation**, so the target is
the model's own top-1 on the unmasked photograph. Every result therefore
explains *the model's prediction*, not the dish — say so rather than eliding
it, and never write a per-ingredient claim as though it explained a dish. The
costs of the borrowed vocabulary (66 of 101 classes used, `pork_chop` on 13 %
of images including non-pork dishes, and confidence that does *not* flag the
misses) are measured in `insights/week-01.md` W01-10.

Checkpoints are named in full — `<backend>/<architecture>/<pretrained>` — and
`ClipScorer.load` turns any open_clip architecture/weight mismatch warning into
a `RuntimeError`. Do not weaken that guard: `ViT-B-32/openai` loads with only a
warning and changes the predicted dish on 37 % of images (W01-10), which the
efficiency check cannot detect. When adding a checkpoint for Milestone 2, load
it once via `python -m food_shapley.scoring --config <cfg>` and record the
printed `parameter_sha256`.

The non-ingredient region is a **player** (`background_mode = "player"`), a
deliberate, recorded deviation from `PLAN.md`'s "label 0 is background, not a
player". It is still never an *ingredient*: filter on
`PlayerSet.ingredient_indices` for every ingredient-level statistic, or the
scene silently counts as food. See `insights/week-01.md` W01-09.

The scripts in `slurm/` have been simulated locally and pass shellcheck, but
**have never been submitted to a cluster**. Do not describe them as verified
until someone has, and records the outcome in `insights/week-01.md`.

## Cluster conventions

Derived from a script known to work on this SLURM cluster. Apply these to
every job script; see [`slurm/README.md`](slurm/README.md) for detail.

- Partition `-p NvidiaAll`, no `--gres` line.
- Batch jobs do **not** source `~/.bashrc` / `~/.zshrc`. Always
  `export PATH="$HOME/.local/bin:$PATH"` so `uv` is reachable.
- `$HOME` is NFS-mounted and quota-limited. Keep caches and bulk artifacts off
  it: set `TMPDIR` to node-local scratch, and point `FOOD_SHAPLEY_RUNS` at
  persistent scratch for real runs.
- Assume compute nodes may be offline. Anything needing the network goes in
  `slurm/prepare.sh` on a login node; compute-node steps use
  `uv run --frozen --no-sync`.
- Submit from the repository root; scripts `cd "${SLURM_SUBMIT_DIR:-$(pwd)}"`.
- `set -euo pipefail` in every script, and keep preparation idempotent so a
  resubmission after a partial failure resumes instead of redoing or clobbering.
- Array jobs write **one output file per task** and derive their seed from
  `derive_seed(config.seed, task_label(task_id))`, so tasks never collide and a
  partial failure is re-runnable per task. Always check completeness afterwards
  — an array that silently drops 3 % of its tasks looks like one that
  succeeded.
- Log directories must exist before `sbatch`, or submission fails;
  `slurm/logs/{output_logs,error_logs}/` are tracked via `.gitkeep` and their
  `*.log` contents are git-ignored.

## Development workflow

Before implementing anything substantial:

1. Read the relevant weekly plan in [`docs/`](docs/) (see
   [`docs/README.md`](docs/README.md) for the week ↔ plan-step mapping).
2. Understand the research objective and the acceptance criteria it serves —
   the "Evaluation metrics and success criteria" and "Definition of done"
   sections, not just the task list.
3. Inspect the existing implementation and tests before proposing changes.
4. Identify what must change and how it will be validated.

After implementing:

1. Write or update tests covering the new behaviour.
2. Run the checks: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`.
3. Update documentation when behaviour, interfaces, or assumptions change.
4. Record any meaningful new experimental observation in the corresponding
   weekly file under [`insights/`](insights/).
5. Update the weekly plan if the actual work materially deviates from it, and
   say what changed and why.

## Research insight maintenance

When an implementation, experiment, or analysis produces a meaningful new
observation, add a dated entry to `insights/week-NN.md` using the template in
[`insights/README.md`](insights/README.md). Record evidence, context,
interpretation, limitations, and implications.

- Separate **measured results** from **interpretation** and **hypotheses**.
- Every entry carries a status: `preliminary`, `validated`, `inconclusive`,
  `contradicted`, or `superseded`.
- Every entry links to its evidence: run directory, config fingerprint, commit,
  logs, or figures.
- Do not create insights for routine code changes or infrastructure work.
- Inconclusive, negative, and contradictory results must be documented too.
- When a new result overturns an old entry, mark the old one and link forward.
  Do not silently rewrite history.
- **Never mark an experiment successful without evidence, and never invent
  numbers, results, or citations.**

## Testing requirements

- Write tests for new implementations; update existing tests when behaviour
  changes.
- Prefer focused unit tests in `tests/unit/`; add integration tests in
  `tests/integration/` (marked `@pytest.mark.integration`) where components
  must be exercised together.
- Tests must be deterministic. Seed anything stochastic via
  `food_shapley.seeding`.
- Isolate tests from external services: no network calls, no checkpoint
  downloads, no dataset reads outside fixtures. Use `tmp_path` for filesystem
  work.
- For stochastic or expensive research code, test correctness properties
  (efficiency of Shapley values, coalition encoding, mask alignment, index
  behaviour on synthetic games with known structure) instead of rerunning full
  experiments. A tiny synthetic game with a closed-form answer is worth more
  than a slow end-to-end smoke test.
- Mark tests needing a GPU or cluster with `@pytest.mark.gpu` and anything slow
  with `@pytest.mark.slow`. The pre-commit hook excludes both.
- **There is no CI.** Pre-commit is the only automated gate, and it is
  filtered by file type — a Markdown-only commit runs neither mypy nor
  pytest, and `slow` tests never run automatically. Run
  `uv run pytest -m "not gpu"` by hand before anything that matters, and do
  not treat a clean commit as evidence the suite is green.
- **Never skip a failing test, loosen an assertion, or delete a test to make
  the suite pass.** If a test fails, either the code or the expectation is
  wrong — find out which and say so.

## Engineering standards

- Follow the configured Ruff rules and strict mypy. Both are configured in
  `pyproject.toml`; do not relax them to get code through.
- Annotate public interfaces and meaningful internal abstractions.
- Avoid `Any`, blanket `# type: ignore`, and `# noqa` without a reason. If a
  suppression is genuinely needed, make it narrow and comment why.
- If a third-party library lacks type information, add a targeted
  `[[tool.mypy.overrides]]` entry naming that module only — never a global
  `ignore_missing_imports`.
- Keep implementations modular and readable, but do not add abstraction layers
  for hypothetical future needs.
- Prefer explicit, reproducible configuration over hidden defaults. Experiment
  settings belong in `configs/*.toml` and must flow through
  `ExperimentConfig`, so that they are captured in the run fingerprint.
- Seeds are explicit: `seed_everything` for process-wide state,
  `derive_seed(base, label)` for per-image / per-array-task determinism.
- Never commit secrets, credentials, private datasets, model checkpoints, or
  large generated artifacts. `data/` and `runs/` are git-ignored and
  pre-commit blocks files over 512 kB — do not work around either.
- Do not add a dependency without a clear justification. State the reason in
  the commit message, add it to `pyproject.toml`, and commit the updated
  `uv.lock`.
- Never overwrite an existing run directory. `write_run_metadata` refuses to;
  keep that property as the pipeline grows.

## Reporting

When a task cannot be completed, or a check fails, **say so explicitly**.
Report what was run, what passed, and what did not. Do not describe work as
done when a check failed, was skipped, or was never executed. If something
could not be verified (no GPU, no dataset, no network), state that limitation
rather than implying it was verified.

## Commands

```bash
uv sync --all-groups          # environment from the lockfile
uv run pytest                 # all tests
uv run pytest -m "not integration"   # unit tests only
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy                   # strict type check
uv run pre-commit run --all-files
```

`uv` manages the environment, dependencies, locking, and tool execution. Do not
call `pip` directly or create virtualenvs by hand.
