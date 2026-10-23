"""The value function against the real checkpoint, on real FoodSeg103 images.

Marked ``gpu`` *and* ``slow`` so that neither the pre-commit hook nor a default
``pytest`` run touches them: they download a ~600 MB CLIP checkpoint, which
`CLAUDE.md` forbids inside the ordinary suite. The unit tests cover the
arithmetic against a stub; what only these can cover is whether the real
checkpoint, the real prompt set and the real masks fit together -- and whether
the properties the Week-1 efficiency check will rest on actually hold on data.

Run them deliberately:

    uv run pytest tests/integration/test_value_on_real_images.py -m gpu

They do not in fact need a GPU -- everything here runs on CPU in a couple of
minutes -- but they do need the network on first run, and ``gpu`` is this
repository's existing marker for "not in the automated gate".
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from food_shapley.config import ExperimentConfig, load_config
from food_shapley.dataset import load_split
from food_shapley.download import read_id2label, verify_split
from food_shapley.masking import BlurRemoval
from food_shapley.prompts import build_prompts, read_dish_names, verify_dish_names
from food_shapley.scoring import CheckpointSpec, ClipScorer
from food_shapley.segments import PlayerSet, build_players
from food_shapley.value import DishValueFunction, log_softmax

if TYPE_CHECKING:
    from collections.abc import Callable

    from food_shapley.dataset import FoodSegSplit

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]

CONFIG = Path("configs/week01-clip-blur.toml")

#: A small fixed sample. Three images whose top-1 was inspected by hand and
#: found correct (W01-10), so a regression that scrambles the prompt order or
#: the geometry shows up as a changed dish rather than as a changed number.
SAMPLE = (32, 158, 242)
EXPECTED_TOP1 = {32: "grilled_salmon", 158: "pulled_pork_sandwich", 242: "risotto"}


@pytest.fixture(scope="module")
def settings() -> ExperimentConfig:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def split(settings: ExperimentConfig) -> FoodSegSplit:
    if verify_split(settings.split):
        pytest.skip("FoodSeg103 split not downloaded; run food_shapley.download")
    return load_split(settings.split)


@pytest.fixture(scope="module")
def value_function(settings: ExperimentConfig) -> DishValueFunction:
    """The configured value function, loaded once for the whole module."""
    if verify_dish_names():
        pytest.skip("Food-101 prompt set not downloaded; run food_shapley.prompts")
    return DishValueFunction.from_config(settings, device="cpu")


@pytest.fixture(scope="module")
def players(split: FoodSegSplit, settings: ExperimentConfig) -> Callable[[int], PlayerSet]:
    def build(index: int) -> PlayerSet:
        return build_players(
            split[index],
            read_id2label(settings.split),
            image_size=settings.image_size,
            max_players=settings.max_players,
            min_area_fraction=settings.min_area_fraction,
            background_player=settings.background_is_player,
        )

    return build


@pytest.fixture(scope="module")
def operator(settings: ExperimentConfig) -> BlurRemoval:
    return BlurRemoval(radius=settings.blur_radius, keep_background=settings.background_stays_sharp)


# --------------------------------------------------------------------------- #
# The checkpoint is the one the configuration names
# --------------------------------------------------------------------------- #


def test_configured_checkpoint_loads_without_a_mismatch(settings: ExperimentConfig) -> None:
    """The guard that caught `ViT-B-32/openai` must pass on what we ship.

    If this fails, the configured architecture and pretrained tag disagree and
    every number in the run would come from the wrong non-linearity. See
    W01-10.
    """
    scorer = ClipScorer.load(
        CheckpointSpec.parse(settings.model), image_size=settings.image_size, device="cpu"
    )
    assert scorer.image_size == settings.image_size


def test_plain_vit_b_32_with_openai_weights_is_refused(settings: ExperimentConfig) -> None:
    """The mismatch the guard exists for, against the real loader."""
    with pytest.raises(RuntimeError, match="mismatch"):
        ClipScorer.load(
            CheckpointSpec.parse("open_clip/ViT-B-32/openai"),
            image_size=settings.image_size,
            device="cpu",
        )


def test_provenance_records_a_weight_hash(value_function: DishValueFunction) -> None:
    provenance = value_function.scorer.provenance()
    assert provenance["pretrained"] == "openai"
    assert len(provenance["parameter_sha256"]) == 64


def test_prompt_set_is_the_full_food101_vocabulary(
    value_function: DishValueFunction, settings: ExperimentConfig
) -> None:
    """The prompt set *is* the softmax denominator, so its size is a claim."""
    expected = build_prompts(read_dish_names(), settings.prompt_template)
    assert value_function.prompts == expected
    assert len(value_function.prompts) == 101


# --------------------------------------------------------------------------- #
# Properties the Shapley computation will rest on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index", SAMPLE)
def test_target_is_the_inspected_dish(
    value_function: DishValueFunction, players: Callable[[int], PlayerSet], index: int
) -> None:
    """Pin three hand-checked predictions.

    A silent reordering of the prompt set, or a broken resize, then changes a
    dish name rather than just a digit.
    """
    target = value_function.choose_target(players(index).image)
    assert target.name == EXPECTED_TOP1[index]
    assert target.prompt.startswith("a photo of ")


@pytest.mark.parametrize("index", SAMPLE)
def test_grand_coalition_value_is_the_targets_log_prob(
    value_function: DishValueFunction,
    players: Callable[[int], PlayerSet],
    operator: BlurRemoval,
    index: int,
) -> None:
    """Read off the target or rendered, v(N) must be the same number.

    Otherwise efficiency is checked against the wrong anchor.

    This is only true because the scene is a player: in ``visible`` or
    ``hidden`` mode the grand coalition is not the photograph. See W01-09.
    """
    result = players(index)
    target = value_function.choose_target(result.image)
    rendered = operator.render(result, range(len(result)))
    value = value_function.values([rendered], target)[0]
    assert value == pytest.approx(target.log_prob, abs=1e-9)


@pytest.mark.parametrize("index", SAMPLE)
def test_hiding_everything_costs_the_target_evidence(
    value_function: DishValueFunction,
    players: Callable[[int], PlayerSet],
    operator: BlurRemoval,
    index: int,
) -> None:
    """v(N) > v(empty): the game must have a positive span to decompose.

    A non-positive span would mean blurring the dish away did not reduce the
    model's belief in it, which is the degenerate case where every Shapley
    value sums to zero or less and the decomposition says nothing. Measured
    across 200 images the span is positive on all of them, mean +4.2 nats
    (W01-10).
    """
    result = players(index)
    target = value_function.choose_target(result.image)
    v_empty = value_function.values([operator.render(result, [])], target)[0]
    assert v_empty < target.log_prob


@pytest.mark.parametrize("index", SAMPLE)
def test_every_coalition_value_is_a_finite_log_probability(
    value_function: DishValueFunction,
    players: Callable[[int], PlayerSet],
    operator: BlurRemoval,
    index: int,
) -> None:
    """One inf or nan makes every Shapley value of the image meaningless.

    And the efficiency check would still pass.
    """
    result = players(index)
    target = value_function.choose_target(result.image)
    coalitions = [[], [0], list(range(len(result))), result.ingredient_indices]
    values = value_function.values([operator.render(result, c) for c in coalitions], target)
    assert np.isfinite(values).all()
    assert (values <= 0.0).all()


@pytest.mark.parametrize("index", SAMPLE)
def test_scoring_is_deterministic(
    value_function: DishValueFunction,
    players: Callable[[int], PlayerSet],
    operator: BlurRemoval,
    index: int,
) -> None:
    """Bit-identical on a re-score, or no run is reproducible from its config."""
    result = players(index)
    target = value_function.choose_target(result.image)
    images = [result.image, operator.render(result, [0]), operator.render(result, [])]
    first = value_function.values(images, target)
    second = value_function.values(images, target)
    np.testing.assert_array_equal(first, second)
    assert value_function.choose_target(result.image) == target


def test_batching_does_not_change_the_values(
    value_function: DishValueFunction,
    players: Callable[[int], PlayerSet],
    operator: BlurRemoval,
) -> None:
    """Batching must stay a throughput knob, not a parameter of the experiment.

    At ~2,000 coalitions per image it is unavoidable, so it has to be inert.
    """
    result = players(SAMPLE[0])
    target = value_function.choose_target(result.image)
    images = [operator.render(result, [j]) for j in range(len(result))]
    one = value_function.values(images, target, batch_size=1)
    many = value_function.values(images, target, batch_size=len(images))
    np.testing.assert_allclose(one, many, atol=1e-5)


def test_the_probabilities_of_one_image_sum_to_one(
    value_function: DishValueFunction, players: Callable[[int], PlayerSet]
) -> None:
    """The normalisation is what makes the value a dish score.

    Without it the value would track how blurry the frame is.
    """
    logits = value_function.scorer.logits(
        [players(SAMPLE[0]).image], value_function.scorer.embed_prompts(value_function.prompts)
    )
    assert np.exp(log_softmax(logits)).sum() == pytest.approx(1.0, abs=1e-9)
