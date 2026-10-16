# Research insights

This directory is the project's lab notebook. It holds what was *actually
observed*. Planned work lives in [`docs/`](../docs/); nothing here should be
written before the evidence for it exists.

**No experiments have been run yet. All weekly files are empty templates.**

## Index

| File | Corresponding plan | Entries |
|---|---|---|
| [week-01.md](week-01.md) | [docs/week-01.md](../docs/week-01.md) | none yet |
| [week-02.md](week-02.md) | [docs/week-02.md](../docs/week-02.md) | none yet |
| [week-03.md](week-03.md) | [docs/week-03.md](../docs/week-03.md) | none yet |
| [week-04.md](week-04.md) | [docs/week-04.md](../docs/week-04.md) | none yet |
| [week-05.md](week-05.md) | [docs/week-05.md](../docs/week-05.md) | none yet |
| [week-06.md](week-06.md) | [docs/week-06.md](../docs/week-06.md) | none yet |

Keep the "Entries" column current — it is the fastest way to find a result when
writing the paper.

## When to write an entry

Write one when an implementation, experiment, or analysis produces a
**meaningful new observation**. That includes:

- a result that supports or undermines a hypothesis from the weekly plan,
- a validity check that fails,
- a surprising property of the data, the model, or the method,
- a methodological decision whose outcome changes what can be claimed,
- a negative, inconclusive, or contradictory result.

Do **not** write one for routine code changes, refactors, or infrastructure
work with no bearing on the research claims.

## Rules

1. **Never fabricate.** No invented numbers, no placeholder results presented as
   real, no "expected" values filled in ahead of the measurement.
2. **Separate measurement from interpretation.** The template has distinct
   sections. Keep them distinct.
3. **Always state a status.** One of: `preliminary`, `validated`,
   `inconclusive`, `contradicted`, `superseded`.
4. **Always link the evidence.** A run directory, config fingerprint, commit
   hash, log, figure, or artifact path. An entry with no evidence pointer
   cannot be used in the paper.
5. **Never silently revise.** When a later result overturns an earlier one, set
   the old entry's status to `contradicted` or `superseded`, link forward to
   the new entry, and leave the original text intact.
6. **Never mark an experiment successful without evidence.**

## Entry template

Copy this block to the top of the relevant week's file. Entry ids are
`W<week>-<nn>`, e.g. `W01-03`.

```markdown
## W01-01 — <short, factual title>

- **Date:** YYYY-MM-DD
- **Status:** preliminary | validated | inconclusive | contradicted | superseded
- **Plan reference:** docs/week-01.md, task 1.x / hypothesis Hx.x
- **Run / artifact:** runs/<name>-<fingerprint>/ · commit `<sha>` · config `<fingerprint>`

### Observation

What was measured. Numbers, not adjectives.

### Context and motivation

Why this was run and what question it was meant to answer.

### Setup

Model, dataset split, removal operator, seed, number of images/coalitions, and
anything else needed to repeat it. Prefer pointing at the config over
retyping it.

### Evidence

Paths to logs, `.npz` files, figures, notebooks, or test output. Enough that
someone else could check the claim.

### Interpretation

What this plausibly means — explicitly marked as interpretation, not
measurement.

### Limitations and alternative explanations

Confounds, sample-size limits, and at least one competing explanation that the
evidence does not rule out.

### Implications

What this changes for later experiments or implementation decisions.

### Follow-up questions and next steps

Open questions and the concrete next action.
```
