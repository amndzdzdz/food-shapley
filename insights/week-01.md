# Week 1 insights — Pipeline, value function, first full run

Plan: [docs/week-01.md](../docs/week-01.md) · Entry format:
[insights/README.md](README.md)

**No entries yet. No experiments have been run.**

## Questions this week's entries should answer

Derived from the Week 1 plan; each needs a dated entry once evidence exists.

- Did the toy SLURM job and array run on the cluster as written? Which
  assumptions in `slurm/README.md` turned out to be wrong — partition,
  walltime, `uv run --no-sync` on an offline node, scratch paths? This is the
  first thing to record, because every later run inherits these settings.
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

<!-- Add entries below, newest first, using the template in README.md. -->
