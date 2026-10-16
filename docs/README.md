# Implementation plans

These documents break [`PLAN.md`](../PLAN.md) into weekly units of work. They
describe *intended* work. Nothing here is a result — actual observations belong
in [`insights/`](../insights/).

## Week ↔ plan-step mapping

`PLAN.md` is organised by numbered steps with day estimates, not by weeks. The
mapping below assumes a 5-day focused work week and preserves the milestone
boundary (Milestone 1 finishes before Milestone 2 begins).

| Week | `PLAN.md` steps | Days | Milestone | Theme |
|---|---|---|---|---|
| [1](week-01.md) | 0–5 | 5.0 | M1 | Scaffolding → working pipeline → full CLIP run |
| [2](week-02.md) | 6–8 | 5.0 | M1 | Second removal operator, analysis/figures, recipe stretch goal |
| [3](week-03.md) | 9–10 | 3.5 | M1 | Write and polish the workshop paper |
| [4](week-04.md) | 11–13 | 4.5 | M2 | Additional model families, cross-model runs, compositionality index |
| [5](week-05.md) | 14–15 | 4.0 | M2 | SugarCrepe/ARO comparison, generalization analysis + saliency baseline |
| [6](week-06.md) | 16–18 | 6.5 | M2 | Cross-model statistics, conference-length writing, review |

**Assumption, not from the plan.** The week boundaries are a scheduling choice.
`PLAN.md` gives 12 core + 1.5 optional days for M1 and 15 days for M2
(~27–28.5 days, ~5.5 weeks) but does not enumerate weeks. Week 3 is
deliberately light (3.5 days) to keep the milestone boundary clean; Week 6 is
correspondingly heavy (6.5 days) and absorbs the writing crunch. Rebalance
freely — but update this table and the affected weekly plans when you do.

## Conventions

- Each weekly plan has a **Definition of done** section. Do not consider a week
  closed until every item is checked or explicitly renegotiated in the document.
- When actual work deviates materially from the plan, edit the plan and note
  what changed and why. Do not let the plan silently rot.
- Deliverables that are "optional" in `PLAN.md` stay optional here and are
  marked as such. They must never become dependencies of the main result.

## Other documents

- [`datasets.md`](datasets.md) — dataset provenance, licensing, preprocessing
  assumptions.
