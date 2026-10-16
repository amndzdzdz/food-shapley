# Week 2 — Removal-operator robustness, analysis, and the recipe stretch goal

**`PLAN.md` steps:** 6–8 · **Budget:** 5.0 days (3.5 core + 1.5 optional)
**Milestone:** 1 (workshop)
**Depends on:** Week 1 — a validated game, exact values for the full test split

## Objectives

Turn Week 1's values into the paper's actual findings, and show that those
findings do not depend on one arbitrary choice of removal operator.

### Research questions addressed

- **RQ1.** How much of a dish prediction comes from a single ingredient's main
  effect versus from interactions between ingredients?
- **RQ2.** Are the RQ1 conclusions stable across removal operators?
- **RQ-opt** *(optional, step 8)*: Is a visible ingredient still predicted
  strongly when its own region is masked — i.e. is the model reading it off
  co-occurrence rather than pixels?

## Implementation tasks

| # | `PLAN.md` | Task | Days |
|---|---|---|---|
| 2.1 | 6 | Second removal operator: token dropping inside CLIP's ViT instead of blur/inpainting; rerun on a subset | 1.0 |
| 2.2 | 7 | Analysis and figures: per-ingredient SV distributions, interaction heatmaps / `si_graph_plot` with ingredient crops, the "bun-detector vs. combination" histogram across the dataset, example `upset_plot` / `waterfall_plot` cases | 2.5 |
| 2.3 | 8 *(optional)* | Recipe stretch goal: link FoodSeg103 images to Recipe1M recipes, split ingredients into visible/invisible, test whether a visible ingredient is still predicted strongly when its region is masked, compare against ingredient co-occurrence statistics | 1.5 |

Task 2.3 must stay cleanly separable (`PLAN.md`): a separate module, separate
config, separate figures. It is a bonus section, never a dependency of the main
result. Drop it first if the week runs long.

## Experimental design and hypotheses

**Removal-operator comparison (2.1).** The same images, the same players, the
same value function — only the removal operator changes (blur → ViT token
dropping). Run on a subset; `PLAN.md` does not fix the subset size, so choose
one, justify it, and record it. Ideally the subset is sampled with a seed
derived via `derive_seed`.

Hypotheses, to be confirmed or refuted by the data rather than assumed:

- **H2.1** Per-image rankings of ingredient main effects correlate strongly
  across the two operators.
- **H2.2** The qualitative split between "telltale-ingredient" and
  "combination" dishes is preserved across operators, even if absolute values
  shift.

A *refutation* of H2.1 or H2.2 is a real result and must be reported, not
buried — it would mean the conclusions are operator-dependent, which is exactly
the critique `PLAN.md` wants to pre-empt.

**Main analysis (2.2).** Descriptive and comparative; the "bun-detector vs.
combination" histogram across the dataset is the headline figure. Do not invent
a significance test the plan does not call for, but do report dispersion, not
just means.

**Stretch goal (2.3).** Hypothesis: for ingredients that are *visible* in the
image, masking their region should sharply reduce their predicted presence;
if it does not, and the prediction tracks co-occurrence statistics instead,
that is evidence of recipe-prior reliance rather than visual evidence.

## Data, resources, dependencies

- Week 1's `.npz` outputs for the blur operator.
- CLIP ViT internals access for token dropping (may require a different wrapper
  than the Week-1 code path).
- *(Optional)* Recipe1M, and a usable image↔recipe linkage for FoodSeg103.
  **Unresolved:** `PLAN.md` does not specify how that linkage is established.
  Resolve this before committing to task 2.3; see [`datasets.md`](datasets.md).

## Expected outputs

- A second removal-operator implementation behind the same interface as blur,
  selected via the `removal` config field.
- A comparison of SVs/interactions across operators on the subset.
- The figure set listed in task 2.2, written to
  `runs/<name>-<fingerprint>/figures/`.
- *(Optional)* The recipe-based analysis and its figures.
- Findings recorded in [`insights/week-02.md`](../insights/week-02.md).

## Evaluation metrics and success criteria

| Criterion | Target |
|---|---|
| Cross-operator agreement | Rank correlation of per-image ingredient main effects, reported with its distribution across images — not a single aggregate number |
| Efficiency check (token dropping) | Holds for the new operator too |
| Figure set | Each figure in task 2.2 exists, is regenerable from a config + run directory, and is readable at print size |
| Headline claim | The telltale-vs-combination split is quantified across the dataset, with the measure's definition stated precisely |

No threshold is set for "strong" correlation in advance, because `PLAN.md`
offers none. State the observed value and interpret it honestly.

## Testing and validation

- Unit: the token-dropping operator removes exactly the intended tokens
  (test against a synthetic patch grid, no real model).
- Unit: both removal operators satisfy the same interface contract; a shared
  parametrised test suite over operators.
- Unit: figure-generating helpers are tested for data correctness (what goes
  into the plot), not pixel output.
- Integration: a small run reproduces identical numbers from the same config
  and seed — the reproducibility contract.
- Regression: pin a few per-image SV vectors from Week 1 as fixtures so later
  refactors cannot silently change results.

## Risks and alternatives

| Risk | Mitigation / alternative |
|---|---|
| Token dropping needs invasive access to CLIP internals | Budget time for a separate wrapper; fall back to a documented alternative (e.g. attention masking) and say so |
| Operators disagree | Report it as the finding it is; do not pick the operator that gives the nicer story |
| "Hiding segments creates unnatural images" critique | This week's whole point; address it in text with the cross-operator evidence |
| Analysis compressed to recover lost time | `PLAN.md` explicitly warns against this — cut task 2.3 instead |
| Recipe1M linkage turns out to be unavailable | Drop task 2.3; it is optional by design |

## Definition of done

- [ ] Token-dropping operator implemented, tested, and validated by the
      efficiency check.
- [ ] Cross-operator comparison run on a stated, justified subset.
- [ ] All figures from task 2.2 generated and regenerable from a config.
- [ ] The telltale-vs-combination measure is precisely defined in writing.
- [ ] *(Optional)* Recipe analysis done, or explicitly dropped with a reason
      recorded.
- [ ] `insights/week-02.md` separates measured results from interpretation, and
      records any cross-operator disagreement.
- [ ] All quality checks pass.
