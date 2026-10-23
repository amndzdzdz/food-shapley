"""The value function a coalition is scored by.

`PLAN.md` Milestone 1, step 2: "value function = log-probability of the target
dish". Concretely, for a coalition ``S`` of players,

    v(S) = log softmax(CLIP logits over the dish prompt set)[y-hat]

where the rendered image for ``S`` comes from
:class:`~food_shapley.masking.BlurRemoval` and the logits from
:class:`~food_shapley.scoring.ClipScorer`. This module owns exactly one
decision -- *which* logit -- and the two consequences of it.

Which dish is the target
------------------------
FoodSeg103 annotates **ingredients, not dishes**: its records carry an image, a
per-pixel ingredient label map and the set of ingredient classes present, and
nothing that names the dish. There is therefore no ground-truth label to score,
and no way to derive one without inventing an ingredient-set to dish mapping
that the dataset does not license.

The target is instead **the model's own top-1 on the unmasked photograph**: run
the full image once, take ``argmax`` over the prompt set, and keep that class
fixed for every coalition of that image. This makes the decomposition answer
"*why did this model predict this dish on this photograph*", which is the
question `PLAN.md` actually asks -- whether the prediction rests on one
telltale ingredient or on a combination. It deliberately does **not** answer
"did the model get it right"; that would need dish labels.

Two consequences, both load-bearing:

* The target is a function of the model. The same photograph scored by a
  different checkpoint can get a different target, so Shapley values are not
  directly comparable across models at the level of "the same dish". Milestone
  2's cross-model comparison has to compare *structure* (how concentrated the
  attribution is) rather than per-class values, or re-target explicitly.
* Food-101's 101 classes are a borrowed vocabulary, not FoodSeg103's. Plenty of
  FoodSeg103 photographs depict no Food-101 dish at all, in which case the
  top-1 is the least-bad of 101 wrong answers and the decomposition explains an
  arbitrary choice. Measured on 200 test images, ``pork_chop`` alone takes 13 %
  of the predictions and fires on chicken-and-rice; and **confidence does not
  identify these cases** -- the minimum top-1 probability over the sample is
  0.167, and one frame containing cake is called ``baby_back_ribs`` at 0.79.
  :attr:`DishTarget.probability` is therefore recorded per image as evidence,
  but it is *not* a usable filter, and no threshold on it is applied anywhere.
  Separating "explained the dish" from "explained an arbitrary choice" needs a
  dish-level annotation this dataset does not have; see W01-10 for the full
  measurement and the open question.

Why log-probability
-------------------
Three properties, none of them cosmetic:

* **Normalised.** A raw cosine similarity drifts with overall image quality: a
  blurred frame is less like *every* caption, so every coalition's score sinks
  together and the Shapley values measure blurriness. The softmax over the
  prompt set divides that out -- the value is how much the target dish wins by,
  relative to the other 100.
* **Unbounded below.** A probability is squeezed into ``[0, 1]``, so once the
  evidence is gone all small coalitions look equally close to zero and their
  differences vanish into float noise. ``log p`` keeps resolving them.
* **Differences are log-odds-like.** ``v(N) - v(empty)``, which efficiency says
  the Shapley values sum to, is then a log-likelihood ratio: how many nats of
  evidence for this dish the visible ingredients carry over a blurred frame.

The softmax temperature is *not* a free parameter here: CLIP's ``logit_scale``
is a trained part of the checkpoint, applied by
:meth:`~food_shapley.scoring.ClipScorer.logits`. It is nonlinear in its effect
-- a sharper scale pushes ``v(N)`` toward 0 and sinks the uninformative
coalitions -- so it is one more reason the checkpoint has to be pinned down to
its pretrained tag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from food_shapley.prompts import build_prompts, read_dish_names
from food_shapley.scoring import DEFAULT_BATCH_SIZE, CheckpointSpec, ClipScorer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from food_shapley.config import ExperimentConfig
    from food_shapley.scoring import PromptEmbedding

__all__ = ["DishTarget", "DishValueFunction", "log_softmax"]


def log_softmax(logits: NDArray[np.float32]) -> NDArray[np.float64]:
    """Row-wise log-softmax, computed in float64 without overflowing.

    Done here rather than in torch so that the arithmetic that produces the
    numbers Shapley values are computed from is visible, and in float64 because
    ``shapiq``'s efficiency check compares sums of up to 2,048 terms: float32
    accumulates enough error over that to make a correct implementation look
    broken.

    Args:
        logits: Array of shape ``(n, num_prompts)``.

    Returns:
        Array of the same shape whose rows exponentiate to 1.

    Raises:
        ValueError: If ``logits`` is not two-dimensional, has no prompts, or
            contains a non-finite value. A ``nan`` logit would otherwise
            propagate into every Shapley value of the image and still satisfy
            efficiency.
    """
    if logits.ndim != 2:
        msg = f"logits must be 2-D (images, prompts), got shape {logits.shape}"
        raise ValueError(msg)
    if logits.shape[1] == 0:
        msg = "logits must score at least one prompt"
        raise ValueError(msg)
    if not np.isfinite(logits).all():
        msg = "logits contain non-finite values; refusing to build a value function from them"
        raise ValueError(msg)

    wide = logits.astype(np.float64)
    shifted = wide - wide.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


@dataclass(frozen=True, slots=True)
class DishTarget:
    """The dish whose log-probability one image's coalitions are scored by.

    Attributes:
        index: Position in the prompt set, i.e. the Food-101 class index.
        name: Food-101 class name, e.g. ``"baby_back_ribs"``.
        prompt: The exact rendered prompt that was scored, e.g.
            ``"a photo of baby back ribs, a type of food."``. Kept so that a
            value can be traced to its text without re-deriving it from a
            template.
        log_prob: ``v(N)`` -- the target's log-probability on the **unmasked**
            photograph. Not recomputed later; it is what the target was chosen
            by.
    """

    index: int
    name: str
    prompt: str
    log_prob: float

    @property
    def probability(self) -> float:
        """Top-1 confidence on the unmasked photograph.

        Recorded per image as evidence about how firm the prediction was. It is
        **not** a filter for the borrowed-vocabulary problem, which is what it
        was originally added for: measured over 200 test images the minimum is
        0.167 and a confidently wrong prediction reaches 0.79, so a low-
        confidence cut removes nothing and a high-confidence cut keeps the
        wrong ones. See :mod:`food_shapley.value`'s module docstring and
        W01-10.
        """
        return float(np.exp(self.log_prob))


class DishValueFunction:
    """Scores coalition renderings by the log-probability of one dish.

    Stateless with respect to images: the target is chosen per image by
    :meth:`choose_target` and passed back in, so one loaded model serves the
    whole split and no image's target can leak into another's.
    """

    def __init__(
        self,
        scorer: ClipScorer,
        embedding: PromptEmbedding,
        names: Sequence[str],
    ) -> None:
        """Bind a scorer to one prompt set.

        Args:
            scorer: The loaded checkpoint.
            embedding: Embedded prompts, in class order.
            names: Dish names in the same order as ``embedding.prompts``.

        Raises:
            ValueError: If the names and prompts disagree in length, which
                would silently mislabel every target.
        """
        if len(names) != len(embedding):
            msg = (
                f"got {len(names)} dish names for {len(embedding)} prompts; "
                "names and prompts must be the same prompt set in the same order"
            )
            raise ValueError(msg)
        self._scorer = scorer
        self._embedding = embedding
        self._names = tuple(names)

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        *,
        device: str | None = None,
        base: Path | None = None,
    ) -> DishValueFunction:
        """Build the value function the configuration describes.

        Every ingredient of it -- checkpoint, input size, prompt template -- is
        read from ``config``, so the object cannot differ from what the run
        fingerprint claims.

        Args:
            config: The run configuration.
            device: Torch device; defaults to CUDA when available, else CPU.
            base: Override for the data root holding the prompt set.

        Returns:
            A ready value function, with the prompts already embedded.
        """
        names = read_dish_names(base=base)
        prompts = build_prompts(names, config.prompt_template)
        scorer = ClipScorer.load(
            CheckpointSpec.parse(config.model),
            image_size=config.image_size,
            device=device,
        )
        return cls(scorer, scorer.embed_prompts(prompts), names)

    @property
    def scorer(self) -> ClipScorer:
        """The underlying checkpoint wrapper."""
        return self._scorer

    @property
    def prompts(self) -> tuple[str, ...]:
        """The prompt set, which *is* this value function's output space."""
        return self._embedding.prompts

    def choose_target(self, image: NDArray[np.uint8]) -> DishTarget:
        """Pick the dish to decompose: the model's top-1 on the unmasked image.

        Args:
            image: The photograph in the checkpoint's geometry, i.e.
                ``PlayerSet.image`` -- *not* a rendered coalition. Passing a
                masked image here would target whatever the blur happens to
                look like.

        Returns:
            The chosen target, carrying ``v(N)``.

        Note:
            Ties go to the lowest class index, inherited from ``argmax``. Exact
            ties between two of 101 float logits do not occur in practice; the
            behaviour is fixed rather than left to chance so that a re-run
            cannot pick a different dish.
        """
        scores = log_softmax(self._scorer.logits([image], self._embedding))[0]
        index = int(np.argmax(scores))
        return DishTarget(
            index=index,
            name=self._names[index],
            prompt=self._embedding.prompts[index],
            log_prob=float(scores[index]),
        )

    def values(
        self,
        images: Sequence[NDArray[np.uint8]],
        target: DishTarget,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> NDArray[np.float64]:
        """Score rendered coalitions: ``v(S)`` for each image, in order.

        Args:
            images: Coalition renderings from
                :meth:`~food_shapley.masking.BlurRemoval.render`, in the
                caller's coalition order. The order is preserved, because task
                1.4 relies on position to map a value back to its coalition.
            target: The dish from :meth:`choose_target`.
            batch_size: Images per forward pass.

        Returns:
            Array of shape ``(len(images),)`` of log-probabilities, all
            ``<= 0``.

        Raises:
            IndexError: If ``target.index`` is outside this prompt set, which
                means the target was chosen against different prompts.
        """
        if not 0 <= target.index < len(self._embedding):
            msg = (
                f"target index {target.index} is outside this prompt set of "
                f"{len(self._embedding)}; the target was chosen against other prompts"
            )
            raise IndexError(msg)
        if not images:
            return np.zeros((0,), dtype=np.float64)
        logits = self._scorer.logits(images, self._embedding, batch_size=batch_size)
        return log_softmax(logits)[:, target.index]
