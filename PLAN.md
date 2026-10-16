# Project Plan — Why Does the Model Say Burger?

Two milestones, each building directly on the previous one:

- **Milestone 1 — Workshop-ready.** The core pipeline from the original sketch (FoodSeg103 masks → CLIP → exact Shapley values/interactions via `shapiq`), plus the recipe-based stretch goal as an optional add-on, plus actually writing the paper.
- **Milestone 2 — Conference-ready.** Adds Upgrade Idea 2 (scaling across model families, measured against accuracy-based compositionality benchmarks) as the main new contribution, with Upgrade Idea 1 (linking interactions to misclassification/generalization) folded in as a secondary, more cautious analysis.

Day estimates assume focused full-time work. Since you're on a SLURM cluster with ample GPUs, raw compute (running CLIP/ViT/CNN/LLaVA forward passes over thousands of coalitions) is not the bottleneck anywhere in this plan — the time sinks are job orchestration/engineering, analysis, and writing, not waiting for GPUs. That shifts a few line items (noted below) relative to a laptop-bound estimate.

| Milestone | Core days | Optional add-on | Total |
|---|---|---|---|
| 1 — Workshop | 12 | +1.5 (recipe stretch goal) | ~12–13.5 days (~2.5 weeks) |
| 2 — Conference (on top of M1) | 15 | — | +15 days |
| **Combined, M1 → M2** | | | **~27–28.5 days (~5.5 weeks)** |

---

## Milestone 1: Workshop-level

| # | Step | Days | Watch out for |
|---|---|---|---|
| 0 | Repo + cluster scaffolding: conda/venv env, install `shapiq` + CLIP + FoodSeg103 download, submit one toy SLURM job end-to-end to confirm the pipeline runs on a node | 0.5 | Get the SLURM submit/array mechanics working on a trivial job now, before any real experiments depend on it |
| 1 | Load FoodSeg103, turn masks into players (merge small segments, add "rest", align masks with the resized image) | 1 | Label 0 is background, not a player; resizing images and masks must use the same interpolation/alignment |
| 2 | Pick dish prompts (e.g. Food-101's 101 class names), set up CLIP, decide the value function (log-probability of "burger" against all dish prompts) | 0.5 | This choice of value function is what everything downstream inherits — sanity check it on a few obvious images first |
| 3 | Write the game: hide segments (blur), batch the ~128 coalitions per image, return the dish score, as a `shapiq.Game` subclass | 1 | Coalition ordering must match what `shapiq` expects internally |
| 4 | Compute exact SV, k-SII, and Möbius via `shapiq.ExactComputer`; build in the two checks now (efficiency: SV sums to v(all) − v(∅); visual: render a few masked coalitions and confirm "only bun visible" really shows only the bun) | 1 | Catching a masking bug here is much cheaper than after the full run |
| 5 | Scale to the full run as a SLURM job array across the ~2,000–7,118 test images; save each image's 128 game values as `.npz` | 1 | With your cluster this is an orchestration day, not a compute-bound day — the actual CLIP calls are fast and embarrassingly parallel across array tasks; budget the day for job-array bookkeeping, retries, and storage layout, not for GPU wait time |
| 6 | Second removal strategy: token dropping inside CLIP's ViT instead of blur/inpainting; rerun on a subset and compare conclusions across removal operators | 1 | This directly pre-empts the "hiding segments creates unnatural images" critique — worth doing even for a workshop paper |
| 7 | Analysis and figures: per-ingredient SV distributions, interaction heatmaps / `si_graph_plot` with ingredient crops, a "bun-detector vs. combination" histogram across the dataset, example `upset_plot`/`waterfall_plot` cases | 2.5 | This is where the actual findings come from — don't compress it to make up time elsewhere |
| 8 | *(Optional)* Stretch goal: link FoodSeg103 images to their Recipe1M recipes, split ingredients into visible/invisible, test whether a visible ingredient is still predicted strongly when its own region is masked, compare against ingredient co-occurrence statistics | 1.5 | Keep this cleanly separable — it's a nice extra section, not a dependency for the main result |
| 9 | Write the workshop paper (4–6 pages): intro, method, results, limitations, explicit related-work positioning against PixelSHAP, Sumiyasu et al.'s pixel-interaction paper, and the `shapiq` ViT tutorial | 2.5 | State plainly how this differs from PixelSHAP (segment-level + food domain + exact computation + the telltale-vs-combination framing) — reviewers who know that paper will ask |
| 10 | Internal review pass, fix numbers/figures, format for the target venue (MADiMa/CEA or a CVPR/NeurIPS XAI workshop — check this year's call) | 1 | Leave this as real slack, not a rounding error |

**Milestone 1 total: 12 days core (+1.5 optional) ≈ 2.5 weeks full-time, or 4–5 weeks part-time.**

---

## Milestone 2: Conference-level

Builds on everything in Milestone 1. Leads with Upgrade Idea 2 (the cleaner, less contested gap — measuring compositionality via interactions rather than accuracy); keeps Upgrade Idea 1 (interactions → generalization/misclassification) as a secondary analysis, explicitly engaging with the published skepticism about attribution-based generalization diagnostics (Viviano et al., ICLR 2021).

| # | Step | Days | Watch out for |
|---|---|---|---|
| 11 | Generalize the game's value function to additional model families: a CNN (e.g. ConvNeXt/ResNet fine-tuned on the same dish labels), a ViT classifier, and an open-weight VLM such as LLaVA scored via log-prob of the dish token/phrase | 2.5 | LLaVA's scoring interface is the odd one out — plan for it needing a different wrapper than CLIP/CNN/ViT |
| 12 | SLURM job arrays to compute exact coalition values for all models across the full image set (fall back to `shapiq`'s SVARMIQ/KernelSHAPIQ approximators only if any model's per-coalition cost makes exact computation impractical even with your GPU budget) | 1 | Again an orchestration day given your cluster — multiply the array size by number-of-models and keep storage/paths organized per model |
| 13 | Define a per-image/per-dish "compositionality index" from the interaction values (e.g. share of total value from the single largest main effect vs. from interaction terms); validate it against the two checks from Step 4 | 1 | This index is the number your whole scaling claim rests on — pressure-test it on hand-picked obvious cases before trusting it at scale |
| 14 | Run (or source reported numbers for) SugarCrepe and ARO on the same models; correlate/contrast their accuracy-based compositionality scores against your interaction-based index | 1.5 | This is the explicit "we measure *how*, not just *whether*" contribution — make the comparison a real analysis, not a one-line mention |
| 15 | Upgrade Idea 1 as secondary analysis: build/source a harder or shifted test set (e.g. a non-Western-dish split or FoodSeg154-style extension), test whether low-interaction/single-part reliance predicts misclassification there; replicate a plain-saliency baseline and show whether the interaction-based diagnostic does better, directly addressing Viviano et al.'s finding that saliency-based diagnostics often don't predict generalization | 2.5 | Don't claim this link holds without showing the saliency baseline comparison — that's exactly the objection a reviewer familiar with that paper will raise |
| 16 | Full statistical analysis and figures across model families and both datasets: `aggregate_interaction_values`, cross-model comparison plots, correlation plots against benchmark accuracy | 2 | — |
| 17 | Extend writing to conference length (8–10 pages): broaden related work to cover PixelSHAP, Sumiyasu et al., SugarCrepe/ARO, Faith-Shap, and the Viviano et al. counterpoint; add methods/limitations sections for the cross-model and generalization claims | 3 | — |
| 18 | Internal/advisor review round; pre-empt likely reviewer objections (removal artifacts, "is this just accuracy in disguise", validity of LLaVA's log-prob scoring) in the text itself; polish and format | 1.5 | — |

**Milestone 2 total: 15 days ≈ 3 weeks full-time, on top of Milestone 1.**

**Combined total, Milestone 1 → Milestone 2: ~27–28.5 days (~5.5 weeks) full-time, or roughly double that part-time.**

---

## Notes carried over from the novelty check

- Closest prior work to cite and differentiate from throughout: **PixelSHAP** (segment-level Shapley interactions for VLMs, not food, not exact computation), **Sumiyasu et al. CVPR 2024** (pixel-level Shapley interactions), the **`shapiq`** NeurIPS 2024 paper and its ViT tutorial (patches, not semantic segments), **SugarCrepe**/**ARO** (CLIP compositionality via accuracy, not interactions), **Faith-Shap** (the FSII definition you're using), and **Viviano et al., ICLR 2021** (the "saliency is a possible red herring" counterpoint relevant to Upgrade Idea 1.
- No prior work combines FoodSeg103 (or any food-ingredient segmentation) with Shapley/Shapley-interaction analysis — this holds for both milestones.
