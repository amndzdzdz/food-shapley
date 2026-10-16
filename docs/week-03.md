# Week 3 — Workshop paper: writing, review, submission

**`PLAN.md` steps:** 9–10 · **Budget:** 3.5 days · **Milestone:** 1 (workshop)
**Depends on:** Weeks 1–2 — final figures, numbers, and recorded insights

This is the milestone-completing week. It is mostly writing, not implementation.

## Objectives

Produce a submittable 4–6 page workshop paper whose every claim is traceable to
a run directory and an entry in `insights/`.

### Research questions addressed

No new ones. The paper answers RQ1 and RQ2 using Week 2's results, and states
clearly what it does *not* answer (that is Milestone 2's job).

## Tasks

| # | `PLAN.md` | Task | Days |
|---|---|---|---|
| 3.1 | 9 | Write the paper: intro, method, results, limitations, and explicit related-work positioning against PixelSHAP, Sumiyasu et al. (CVPR 2024), and the `shapiq` ViT tutorial | 2.5 |
| 3.2 | 10 | Internal review pass: fix numbers and figures, format for the target venue | 1.0 |

`PLAN.md` names the candidate venues as MADiMa/CEA or a CVPR/NeurIPS XAI
workshop, and says to check this year's call. **Unresolved:** the venue and its
deadline are not fixed in the plan. Resolve this at the start of the week — the
page limit and format follow from it.

`PLAN.md` also says step 10 is "real slack, not a rounding error". Treat the
1.0 day as buffer for the submission itself, not as extra writing time.

## Required positioning (from `PLAN.md`)

The paper must state plainly how the work differs from **PixelSHAP**:

- segment-level rather than pixel-level,
- food domain,
- **exact** computation rather than approximation,
- the telltale-vs-combination framing.

`PLAN.md` warns that reviewers who know PixelSHAP will ask. Also cite and
differentiate: Sumiyasu et al. (pixel-level Shapley interactions), the `shapiq`
NeurIPS 2024 paper and its ViT tutorial (patches, not semantic segments), and
Faith-Shap (the FSII definition in use).

## Expected outputs

- A paper draft in the venue's format. **Unresolved:** the plan does not say
  whether the manuscript lives in this repository. Decide in week 3; if it
  does, add a `paper/` directory and extend `.gitignore` for LaTeX build
  artifacts.
- A results table / figure set whose source run directory and config
  fingerprint are recorded for every reported number.
- A limitations section covering, at minimum: removal-operator artifacts, the
  value-function choice, dataset scope, and the absence of cross-model evidence.

## Success criteria

| Criterion | Target |
|---|---|
| Traceability | Every number and figure in the paper maps to a run directory + config fingerprint recorded in `insights/` |
| Related work | PixelSHAP, Sumiyasu et al., `shapiq`/ViT tutorial, Faith-Shap all cited and explicitly differentiated |
| Limitations | Removal artifacts and value-function sensitivity discussed with the Week-2 evidence |
| Format | Complies with the chosen venue's template and page limit |
| Honesty | No claim in the text exceeds what the recorded evidence supports |

## Validation

- Re-generate every figure from its config before submission; a figure that
  cannot be regenerated cannot be trusted.
- Cross-check each reported number against the run artifact it came from.
- Keep the code quality checks green even though this is a writing week —
  analysis scripts written for the paper are code too.

## Risks and alternatives

| Risk | Mitigation |
|---|---|
| Venue deadline does not match the schedule | Pick the venue in week 3 day 1; have a fallback venue |
| A number in the paper no longer matches the artifacts | Regenerate from configs before submission; treat a mismatch as a bug |
| Reviewer objection: "this is just PixelSHAP for food" | Explicit differentiation section, as required by `PLAN.md` |
| Writing overruns into Milestone 2 | Accept the overrun and update `docs/README.md`; do not cut the limitations section |

## Definition of done

- [ ] Target venue and deadline confirmed against this year's call.
- [ ] Full draft written, including limitations and related-work positioning.
- [ ] Every reported number traced to a run directory and config fingerprint.
- [ ] All figures regenerated from configs immediately before submission.
- [ ] Internal review pass completed and its comments addressed.
- [ ] Manuscript formatted for the venue.
- [ ] `insights/week-03.md` records review feedback and any claim that had to be
      weakened or withdrawn.
