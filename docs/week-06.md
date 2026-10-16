# Week 6 — Cross-model statistics, conference paper, review

**`PLAN.md` steps:** 16–18 · **Budget:** 6.5 days · **Milestone:** 2 (conference)
**Depends on:** Weeks 4–5 — cross-model values, the index, benchmark comparison,
shifted-set results

Per [`docs/README.md`](README.md), this week is over a nominal 5-day budget by
design: it absorbs the slack left by Week 3. Rebalance if the schedule allows.

## Objectives

Consolidate all Milestone 2 evidence into a statistically defensible analysis
and an 8–10 page conference submission.

### Research questions addressed

All of RQ1–RQ4, now across model families and both datasets. No new questions.

## Tasks

| # | `PLAN.md` | Task | Days |
|---|---|---|---|
| 6.1 | 16 | Full statistical analysis and figures across model families and both datasets: `aggregate_interaction_values`, cross-model comparison plots, correlation plots against benchmark accuracy | 2.0 |
| 6.2 | 17 | Extend the writing to conference length (8–10 pages): broaden related work to PixelSHAP, Sumiyasu et al., SugarCrepe/ARO, Faith-Shap and the Viviano et al. counterpoint; add methods and limitations sections for the cross-model and generalization claims | 3.0 |
| 6.3 | 18 | Internal/advisor review round; pre-empt likely reviewer objections in the text itself; polish and format | 1.5 |

## Required objections to pre-empt (from `PLAN.md`)

The text itself must address, not merely acknowledge:

1. **Removal artifacts** — masking creates unnatural images. Evidence: the
   Week-2 cross-operator comparison.
2. **"Is this just accuracy in disguise?"** — Evidence: the Week-5 contrast
   between the index and SugarCrepe/ARO.
3. **Validity of the VLM's log-prob scoring** — Evidence: the Week-4 wrapper
   validation.
4. **Viviano et al.'s saliency counterpoint** — Evidence: the Week-5 saliency
   baseline comparison.

If any of these lacks supporting evidence by the time of writing, state the
limitation explicitly rather than arguing around it.

## Expected outputs

- Cross-model aggregate analysis and the full figure set, all regenerable from
  configs and run directories.
- An 8–10 page manuscript with expanded related work, methods, and limitations.
- A review round completed and its comments addressed.
- `insights/week-06.md` capturing cross-model findings and review feedback.

## Evaluation metrics and success criteria

| Criterion | Target |
|---|---|
| Statistical rigour | Every reported comparison states its *n*, its uncertainty, and whether it is descriptive or inferential |
| Traceability | Each figure and number maps to a run directory + config fingerprint |
| Related work | PixelSHAP, Sumiyasu et al., SugarCrepe/ARO, Faith-Shap, Viviano et al. all covered and differentiated |
| Objection coverage | All four objections above answered in the text with evidence, or declared as limitations |
| Format | Complies with the target venue's template and page limit |

## Validation

- Regenerate the entire figure set from configs before submission.
- Re-run the full check suite; analysis code written this week is code too.
- Cross-check the paper's numbers against `insights/` entries and the run
  artifacts they cite.
- Verify that no claim in the manuscript has been strengthened beyond what the
  corresponding insight entry records.

## Risks and alternatives

| Risk | Mitigation |
|---|---|
| Aggregate results contradict the workshop paper's conclusions | Report the contradiction and explain it; prior publication is not a reason to suppress it |
| Writing overruns the 6.5-day budget | Protect task 6.1; a weaker analysis cannot be fixed by better prose |
| Reviewer objections not answerable with available evidence | Downgrade the corresponding claim and move it to limitations |
| Venue/deadline unknown | **Unresolved in `PLAN.md`** — fix the target venue at the start of Week 6 |

## Definition of done

- [ ] Cross-model aggregate analysis complete, with *n* and uncertainty stated
      throughout.
- [ ] Full figure set regenerated from configs.
- [ ] Manuscript extended to conference length with the required related work.
- [ ] All four anticipated objections addressed with evidence or declared as
      limitations.
- [ ] Review round completed and comments addressed.
- [ ] Every number traced to a run directory and config fingerprint.
- [ ] `insights/week-06.md` updated, including any claim weakened during review.
- [ ] All quality checks pass.
