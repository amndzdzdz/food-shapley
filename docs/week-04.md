# Week 4 — Scaling across model families and defining the compositionality index

**`PLAN.md` steps:** 11–13 · **Budget:** 4.5 days · **Milestone:** 2 (conference)
**Depends on:** Weeks 1–3 — a validated game, both removal operators, the
workshop result as the single-model baseline

## Objectives

Generalize the pipeline from CLIP to several model families, and define the
quantity that Milestone 2's main claim rests on.

### Research questions addressed

- **RQ3 (part 1).** Can the interaction-based decomposition be computed
  comparably across model families?
- **RQ3 (part 2).** Does a single scalar "compositionality index" capture the
  main-effect-vs-interaction balance in a way that behaves sensibly on
  hand-picked cases?

## Implementation tasks

| # | `PLAN.md` | Task | Days |
|---|---|---|---|
| 4.1 | 11 | Generalize the value function to more model families: a CNN (e.g. ConvNeXt/ResNet fine-tuned on the same dish labels), a ViT classifier, and an open-weight VLM such as LLaVA scored via log-prob of the dish token/phrase | 2.5 |
| 4.2 | 12 | SLURM job arrays computing exact coalition values for all models over the full image set; keep storage/paths organised per model | 1.0 |
| 4.3 | 13 | Define a per-image/per-dish compositionality index from the interaction values (e.g. share of total value from the single largest main effect vs. from interaction terms) and validate it against the Week-1 checks | 1.0 |

`PLAN.md` flags LLaVA as the odd one out: plan for it needing a different
wrapper than CLIP/CNN/ViT. The CNN requires fine-tuning on the same dish labels
— **unresolved:** the plan does not specify the fine-tuning data or protocol.
Decide and document it before task 4.2, since every cross-model comparison
inherits that choice.

## Experimental design and hypotheses

All models share the same images, the same segment-derived players, and the
same removal operator, so that differences are attributable to the model and
not to the setup. Each model needs its own documented score→value mapping; the
mappings must be comparable in kind (log-probability of the dish against a
fixed alternative set) even though the architectures differ.

- **H4.1** The compositionality index is well defined and finite for
  essentially all images across all models.
- **H4.2** The index separates hand-picked "obvious single-cue" dishes from
  hand-picked "obvious combination" dishes.

`PLAN.md`: "This index is the number your whole scaling claim rests on —
pressure-test it on hand-picked obvious cases before trusting it at scale." If
H4.2 fails, the index definition is wrong, not the data. Revise the definition
and record both attempts.

Exact computation is the default. Per `PLAN.md`, fall back to `shapiq`'s
SVARMIQ / KernelSHAPIQ approximators **only** if a model's per-coalition cost
makes exact computation impractical even with the available GPU budget. If a
fallback is used, record which models used it and report approximation error.

## Data, resources, dependencies

- The same FoodSeg103 split as Milestone 1, so results are directly comparable.
- Checkpoints for the CNN, ViT classifier, and the VLM — each pinned by name
  and revision in the run configuration.
- Substantially more storage: one value set per (model × removal operator ×
  image). Plan the layout before launching, not after.

## Expected outputs

- A model-agnostic scoring interface with one implementation per model family.
- Per-model run directories with exact coalition values.
- A documented compositionality index with a reference implementation.
- `insights/week-04.md` entries for each model's setup quirks and for the
  index's behaviour on the hand-picked cases.

## Evaluation metrics and success criteria

| Criterion | Target |
|---|---|
| Model coverage | CLIP + CNN + ViT classifier + VLM all produce valid games |
| Efficiency check | Holds per model; any model failing it is not reported until fixed |
| Index validity | Correct ordering on the hand-picked obvious cases, with the cases listed explicitly |
| Index coverage | Defined and finite for ≥ 99 % of images per model; exceptions characterised |
| Run organisation | Every value file is traceable to model, config fingerprint and code commit |
| Approximation | Exact everywhere, or the fallback documented per model with its error |

## Testing and validation

- Unit: each model wrapper satisfies a shared scoring-interface contract
  (parametrised over wrappers, using stubs — no checkpoint downloads in tests).
- Unit: the compositionality index on synthetic interaction values with known
  structure — a purely additive game must score at one extreme, a purely
  interactive game at the other.
- Unit: the index is invariant to irrelevant rescaling if that is intended;
  if it is not invariant, test and document that too.
- Integration: a miniature multi-model run with stub models, asserting the
  per-model artifact layout.
- Mark anything requiring real checkpoints or GPUs with `gpu`; the pre-commit
  hook excludes it.

## Risks and alternatives

| Risk | Mitigation / alternative |
|---|---|
| LLaVA scoring needs a different interface (`PLAN.md`) | Separate wrapper budgeted in task 4.1; validate its log-prob scoring independently |
| Is the VLM's log-prob score even valid as a dish score? | `PLAN.md` lists this as a likely reviewer objection — design a sanity check now and write it up in Week 6 |
| CNN fine-tuning protocol unspecified | Decide and document before task 4.2; it is a confound if left implicit |
| Exact computation infeasible for a model | Documented approximator fallback with reported error |
| Storage blow-up across models | Per-model layout planned up front; nothing committed to git |
| The index is a definitional artifact | Pressure-test on obvious cases; consider reporting a second index definition as a robustness check |

## Definition of done

- [ ] All four model families produce valid games under a shared interface.
- [ ] Efficiency checks pass per model.
- [ ] Full cross-model runs complete, with per-model storage layout documented.
- [ ] The compositionality index is defined in writing, implemented, and tested
      against synthetic ground truth.
- [ ] The index's behaviour on hand-picked obvious cases is recorded in
      `insights/week-04.md`, including any failed first definition.
- [ ] Any use of approximators is documented with its error.
- [ ] All quality checks pass.
