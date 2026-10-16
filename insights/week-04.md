# Week 4 insights — Model families and the compositionality index

Plan: [docs/week-04.md](../docs/week-04.md) · Entry format:
[insights/README.md](README.md)

**No entries yet. No experiments have been run.**

## Questions this week's entries should answer

- What did each model family's scoring wrapper require, and is its score
  comparable in kind to CLIP's? The VLM wrapper in particular — record the
  evidence that its log-prob scoring is a valid dish score.
- Does the efficiency check hold for every model? Any model where it does not
  must be recorded and excluded from reported results until fixed.
- **H4.1** Is the compositionality index finite and well defined across images
  and models? What fraction of images are exceptions, and why?
- **H4.2** Does the index order the hand-picked obvious cases correctly? List
  the cases. If a first definition failed, record it — the discarded
  definition is itself informative.
- What CNN fine-tuning protocol was used, and how might it confound cross-model
  comparison?
- Was exact computation feasible for all models? For any approximator fallback,
  record the model, the approximator, and the measured error.

---

<!-- Add entries below, newest first, using the template in README.md. -->
