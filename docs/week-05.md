# Week 5 — Benchmark comparison and the generalization analysis

**`PLAN.md` steps:** 14–15 · **Budget:** 4.0 days · **Milestone:** 2 (conference)
**Depends on:** Week 4 — cross-model values and a validated compositionality index

## Objectives

Deliver Milestone 2's main contribution — contrasting interaction-based
compositionality with accuracy-based benchmarks — and run the secondary,
deliberately cautious generalization analysis.

### Research questions addressed

- **RQ3 (part 3).** How does the interaction-based index relate to
  accuracy-based compositionality scores (SugarCrepe, ARO) on the same models?
- **RQ4 (secondary).** Does low-interaction / single-part reliance predict
  misclassification under distribution shift — and does it beat a plain
  saliency baseline?

## Implementation tasks

| # | `PLAN.md` | Task | Days |
|---|---|---|---|
| 5.1 | 14 | Run (or source reported numbers for) SugarCrepe and ARO on the same models; correlate and contrast their accuracy-based scores against the interaction-based index | 1.5 |
| 5.2 | 15 | Secondary analysis: build or source a harder/shifted test set (e.g. a non-Western-dish split or a FoodSeg154-style extension); test whether low-interaction/single-part reliance predicts misclassification there; replicate a plain-saliency baseline and compare | 2.5 |

## Experimental design and hypotheses

**5.1 — the headline contribution.** `PLAN.md`: this is the explicit
"we measure *how*, not just *whether*" contribution; make it a real analysis,
not a one-line mention. With only a handful of models, the correlation has very
few points — report it as such and avoid over-claiming from a tiny *n*.

- **H5.1** The interaction-based index is *not* a simple restatement of
  accuracy-based compositionality scores; it carries information those scores
  do not.

A high correlation would weaken the contribution ("is this just accuracy in
disguise?" is listed in `PLAN.md` as an anticipated reviewer objection). Report
it either way.

**5.2 — the cautious secondary analysis.** `PLAN.md` explicitly frames this
against Viviano et al. (ICLR 2021), who found that saliency-based diagnostics
often do not predict generalization.

- **H5.2** Images/dishes whose predictions rely on a single part generalize
  worse to the shifted set than those relying on interactions.
- **H5.3** The interaction-based diagnostic predicts misclassification better
  than a plain-saliency baseline.

`PLAN.md`: do not claim this link holds without showing the saliency baseline
comparison. The baseline is a required deliverable, not an optional extra. A
null result here is publishable as a careful negative finding and should be
written up as one.

## Data, resources, dependencies

- SugarCrepe and ARO — either run on the same models or sourced as reported
  numbers. **Unresolved:** `PLAN.md` allows either. Record which, per model;
  mixing run and reported numbers in one comparison needs an explicit caveat.
- A harder or shifted test set. **Unresolved:** `PLAN.md` gives examples
  (non-Western-dish split, FoodSeg154-style extension) but does not choose.
  Decide early — task 5.2 cannot start without it. Document its provenance in
  [`datasets.md`](datasets.md).
- A plain-saliency baseline implementation for the same models.

## Expected outputs

- A per-model table: interaction-based index alongside SugarCrepe and ARO
  scores, with the source (run vs. reported) marked per cell.
- Correlation/contrast analysis with the sample size stated prominently.
- Shifted-set results: misclassification rates grouped by reliance type, for
  both the interaction diagnostic and the saliency baseline.
- `insights/week-05.md` entries, including negative results.

## Evaluation metrics and success criteria

| Criterion | Target |
|---|---|
| Benchmark coverage | SugarCrepe and ARO available for every model in the comparison, with provenance marked |
| Comparison quality | A real analysis with stated *n*, uncertainty, and the limits of a small model sample |
| Shifted-set validity | The shift is characterised, not just asserted; its construction is documented and reproducible |
| Baseline | A plain-saliency baseline is implemented and compared — non-negotiable |
| Claim discipline | No generalization claim stated without the baseline comparison alongside it |

"Success" here is a well-supported answer, not a positive answer. Do not mark
H5.2 or H5.3 as confirmed without evidence.

## Testing and validation

- Unit: benchmark score parsing/ingestion, including the reported-numbers path
  (a transcription error here silently corrupts the headline table).
- Unit: the correlation/statistics helpers on synthetic data with known answers.
- Unit: the saliency baseline on a synthetic input with a known salient region.
- Integration: the full comparison pipeline on stub data, asserting the table
  structure and the provenance annotations.
- Determinism: the shifted-set construction must be seeded and reproducible.

## Risks and alternatives

| Risk | Mitigation / alternative |
|---|---|
| "Is this just accuracy in disguise?" (`PLAN.md`) | The contrast analysis is designed to answer it; report high correlation honestly if found |
| Tiny *n* for the cross-model correlation | State *n* everywhere; prefer descriptive contrast over a significance claim |
| Viviano et al.'s counterpoint applies to our diagnostic too | Engage directly in text; the saliency baseline is the empirical response |
| Shifted set not comparable to the original split | Document construction and confounds; report per-dish breakdowns |
| Benchmark numbers mix sources | Mark provenance per cell; caveat in the text |
| Secondary analysis crowds out the main one | Task 5.1 is the contribution; protect its time |

## Definition of done

- [ ] Benchmark scores obtained for every compared model, with provenance marked.
- [ ] The index-vs-benchmark contrast is a full analysis with stated *n* and
      uncertainty.
- [ ] The shifted/harder test set is built or sourced, documented, and
      reproducible.
- [ ] The plain-saliency baseline is implemented and compared.
- [ ] No generalization claim appears without its baseline comparison.
- [ ] `insights/week-05.md` records results including null/negative outcomes
      and their alternative explanations.
- [ ] All quality checks pass.
