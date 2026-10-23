from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from food_shapley.scoring import CheckpointSpec, ClipScorer, PromptEmbedding
from food_shapley.value import DishTarget, DishValueFunction, log_softmax

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

# --------------------------------------------------------------------------- #
# A scorer whose logits are chosen by the test.
#
# The value function's job is "take the right logit and log-normalise it", so
# what it needs to be tested against is a known logit matrix, not a model. This
# stub returns rows the test writes down, which makes the expected values
# closed-form -- and keeps the real checkpoint out of the suite, as
# `CLAUDE.md` requires.
# --------------------------------------------------------------------------- #

IMAGE_SIZE = 4
NUM_PROMPTS = 3
PROMPTS = ("a photo of pizza, a type of food.", "a photo of ramen, a type of food.")
NAMES = ("pizza", "ramen")


class _ScriptedScorer(ClipScorer):
    """A `ClipScorer` that returns logits keyed by the image's first pixel.

    Subclassed rather than duck-typed so that the contract the value function
    relies on -- the signature and the geometry check -- is the real one.
    """

    def __init__(self, table: Mapping[int, Sequence[float]], *, num_prompts: int) -> None:
        super().__init__(
            model=None,
            preprocess=None,
            tokenizer=None,
            spec=CheckpointSpec.parse("open_clip/Stub-B-32/stub"),
            device="cpu",
            image_size=IMAGE_SIZE,
        )
        self._table = table
        self._num_prompts = num_prompts
        self.batch_sizes: list[int] = []

    def logits(
        self,
        images: Sequence[NDArray[np.uint8]],
        embedding: PromptEmbedding,  # noqa: ARG002 - fixed by the base signature.
        *,
        batch_size: int = 64,
    ) -> NDArray[np.float32]:
        self.batch_sizes.append(batch_size)
        self._check_images(images)
        if not images:
            return np.zeros((0, self._num_prompts), dtype=np.float32)
        return np.array([self._table[int(image[0, 0, 0])] for image in images], dtype=np.float32)

    def embed_prompts(self, prompts: Sequence[str]) -> PromptEmbedding:
        return PromptEmbedding(prompts=tuple(prompts), matrix=torch.zeros((len(prompts), 2)))


def keyed(key: int) -> NDArray[np.uint8]:
    """An image the scripted scorer recognises by its first pixel."""
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[0, 0, 0] = key
    return image


def value_function(
    table: Mapping[int, Sequence[float]],
    *,
    prompts: Sequence[str] = PROMPTS,
    names: Sequence[str] = NAMES,
) -> DishValueFunction:
    scorer = _ScriptedScorer(table, num_prompts=len(prompts))
    return DishValueFunction(scorer, scorer.embed_prompts(prompts), names)


# --------------------------------------------------------------------------- #
# log_softmax
# --------------------------------------------------------------------------- #


def test_log_softmax_rows_exponentiate_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(np.exp(log_softmax(logits)).sum(axis=1), [1.0, 1.0], atol=1e-12)


def test_log_softmax_is_never_positive() -> None:
    """A log-probability above 0 would mean a probability above 1."""
    logits = np.array([[100.0, -100.0, 0.0]], dtype=np.float32)
    assert (log_softmax(logits) <= 0.0).all()


def test_log_softmax_matches_the_closed_form_on_a_uniform_row() -> None:
    uniform = np.zeros((1, NUM_PROMPTS), dtype=np.float32)
    np.testing.assert_allclose(log_softmax(uniform), np.full((1, NUM_PROMPTS), -np.log(3.0)))


def test_log_softmax_matches_torch() -> None:
    """Checked against an independent implementation, not against itself."""
    rng = np.random.default_rng(0)
    logits = (rng.normal(size=(7, 101)) * 20.0).astype(np.float32)
    expected = torch.log_softmax(torch.from_numpy(logits).double(), dim=1).numpy()
    np.testing.assert_allclose(log_softmax(logits), expected, atol=1e-10)


def test_log_softmax_survives_large_logits() -> None:
    """CLIP's temperature of 100 puts logits where a naive exp() overflows."""
    logits = np.array([[800.0, 0.0, -800.0]], dtype=np.float32)
    result = log_softmax(logits)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-12)


def test_log_softmax_is_shift_invariant() -> None:
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(log_softmax(logits), log_softmax(logits + 50.0), atol=1e-10)


def test_log_softmax_widens_to_float64() -> None:
    """Efficiency sums up to 2,048 terms; float32 error alone can break it."""
    assert log_softmax(np.zeros((1, 2), dtype=np.float32)).dtype == np.float64


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_log_softmax_rejects_non_finite_logits(bad: float) -> None:
    """A nan here would reach every Shapley value and still pass efficiency."""
    logits = np.array([[1.0, bad, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        log_softmax(logits)


def test_log_softmax_rejects_a_one_dimensional_array() -> None:
    with pytest.raises(ValueError, match="2-D"):
        log_softmax(np.zeros(3, dtype=np.float32))


def test_log_softmax_rejects_an_empty_prompt_axis() -> None:
    with pytest.raises(ValueError, match="at least one prompt"):
        log_softmax(np.zeros((2, 0), dtype=np.float32))


# --------------------------------------------------------------------------- #
# Target selection
# --------------------------------------------------------------------------- #


def test_target_is_the_models_top_one() -> None:
    """FoodSeg103 has no dish label, so the target is the model's own choice."""
    function = value_function({1: (0.0, 5.0)})
    target = function.choose_target(keyed(1))
    assert (target.index, target.name) == (1, "ramen")
    assert target.prompt == PROMPTS[1]


def test_target_carries_v_of_n() -> None:
    """`log_prob` is v(N): what the efficiency check is anchored at."""
    function = value_function({1: (0.0, np.log(4.0))})
    target = function.choose_target(keyed(1))
    np.testing.assert_allclose(target.probability, 4.0 / 5.0, atol=1e-6)


def test_target_probability_exposes_a_hopeless_prompt_set() -> None:
    """A top-1 at chance is how an off-vocabulary photograph has to look."""
    function = value_function({1: (0.0, 0.0, 0.0)}, prompts=("a", "b", "c"), names=("a", "b", "c"))
    target = function.choose_target(keyed(1))
    np.testing.assert_allclose(target.probability, 1.0 / 3.0, atol=1e-6)


def test_target_breaks_ties_at_the_lowest_index() -> None:
    """Fixed rather than left to chance, so a re-run cannot pick another dish."""
    function = value_function({1: (3.0, 3.0)})
    assert function.choose_target(keyed(1)).index == 0


def test_target_rejects_an_image_outside_the_geometry() -> None:
    function = value_function({1: (0.0, 1.0)})
    with pytest.raises(ValueError, match="in the checkpoint's geometry"):
        function.choose_target(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Coalition values
# --------------------------------------------------------------------------- #


def test_values_are_the_targets_log_probability() -> None:
    table = {1: (0.0, np.log(9.0)), 2: (0.0, np.log(1.0))}
    function = value_function(table)
    target = function.choose_target(keyed(1))
    values = function.values([keyed(1), keyed(2)], target)
    np.testing.assert_allclose(values, [np.log(0.9), np.log(0.5)], atol=1e-6)


def test_values_preserve_coalition_order() -> None:
    """Task 1.4 maps a value back to its coalition by position alone."""
    table = {key: (0.0, float(key)) for key in (1, 2, 3, 4)}
    function = value_function(table)
    target = DishTarget(index=1, name="ramen", prompt=PROMPTS[1], log_prob=-0.1)
    order = [3, 1, 4, 2]
    together = function.values([keyed(key) for key in order], target)
    alone = [function.values([keyed(key)], target)[0] for key in order]
    np.testing.assert_allclose(together, alone, atol=1e-12)


def test_values_are_monotone_in_the_targets_logit() -> None:
    """More evidence for the dish must not lower its value."""
    table = {key: (0.0, float(key) / 10.0) for key in range(1, 20)}
    function = value_function(table)
    target = DishTarget(index=1, name="ramen", prompt=PROMPTS[1], log_prob=-0.1)
    values = function.values([keyed(key) for key in range(1, 20)], target)
    assert (np.diff(values) > 0).all()


def test_values_are_never_positive() -> None:
    table = {key: (float(key), -float(key)) for key in (1, 2, 3)}
    function = value_function(table)
    target = DishTarget(index=0, name="pizza", prompt=PROMPTS[0], log_prob=-0.1)
    assert (function.values([keyed(key) for key in (1, 2, 3)], target) <= 0.0).all()


def test_values_are_float64() -> None:
    function = value_function({1: (0.0, 1.0)})
    target = function.choose_target(keyed(1))
    assert function.values([keyed(1)], target).dtype == np.float64


def test_values_of_no_coalitions_is_empty() -> None:
    function = value_function({1: (0.0, 1.0)})
    target = function.choose_target(keyed(1))
    assert function.values([], target).shape == (0,)


def test_values_pass_the_batch_size_through() -> None:
    function = value_function({1: (0.0, 1.0)})
    target = function.choose_target(keyed(1))
    function.values([keyed(1)], target, batch_size=8)
    assert function.scorer.batch_sizes[-1] == 8  # type: ignore[attr-defined]


@pytest.mark.parametrize("index", [-1, 2, 99])
def test_values_reject_a_target_from_another_prompt_set(index: int) -> None:
    """A target chosen against other prompts must not index into this set."""
    function = value_function({1: (0.0, 1.0)})
    target = DishTarget(index=index, name="x", prompt="x", log_prob=-1.0)
    with pytest.raises(IndexError, match="outside this prompt set"):
        function.values([keyed(1)], target)


def test_grand_coalition_value_equals_the_targets_log_prob() -> None:
    """Read off the target or recomputed, v(N) must be the same number.

    Otherwise the efficiency check compares two different games.
    """
    function = value_function({1: (0.3, 2.7)})
    target = function.choose_target(keyed(1))
    np.testing.assert_allclose(function.values([keyed(1)], target)[0], target.log_prob, atol=1e-12)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_names_and_prompts_must_agree_in_length() -> None:
    """Otherwise every target is labelled with the wrong dish."""
    scorer = _ScriptedScorer({}, num_prompts=3)
    embedding = scorer.embed_prompts(["a", "b", "c"])
    with pytest.raises(ValueError, match="2 dish names for 3 prompts"):
        DishValueFunction(scorer, embedding, ["a", "b"])


def test_prompts_are_exposed_as_the_output_space() -> None:
    function = value_function({1: (0.0, 1.0)})
    assert function.prompts == PROMPTS
