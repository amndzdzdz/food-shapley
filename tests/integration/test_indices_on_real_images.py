"""Exact indices over real FoodSeg103 games, against the real checkpoint.

Marked ``gpu`` and ``slow`` for the same reason as
``test_game_on_real_images.py``: these load a ~600 MB CLIP checkpoint, which
`CLAUDE.md` keeps out of the ordinary suite.

`tests/unit/test_indices.py` already proves the indices are the right ones, on
a game whose answer is known in closed form. What a synthetic game cannot show
is how the arithmetic behaves on values CLIP actually produced -- log
probabilities a few nats apart, decomposed into Möbius coefficients that
alternate in sign and cancel. So what is checked here is reconstruction and
tolerance on real numbers:

* the efficiency residual `PLAN.md` gates task 1.6 on, through the function
  that will actually compute it;
* the Möbius transform re-sums to every coalition value, which is the
  strongest available statement that each stored value is attached to the
  right set of players;
* the stored blocks have the shape the configuration implies, on games where
  `shapiq` really does drop zero-valued interactions;
* a saved file reloads bit for bit.

Run them deliberately:

    uv run pytest tests/integration/test_indices_on_real_images.py -m gpu
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from shapiq.utils import powerset

from food_shapley.config import ExperimentConfig, load_config
from food_shapley.dataset import load_split
from food_shapley.download import read_id2label, verify_split
from food_shapley.game import IngredientGame
from food_shapley.indices import (
    DEFAULT_EFFICIENCY_TOLERANCE,
    ImageIndices,
    compute_indices,
    load_indices,
    main,
    save_indices,
)
from food_shapley.prompts import verify_dish_names
from food_shapley.segments import build_players
from food_shapley.value import DishValueFunction

if TYPE_CHECKING:
    from collections.abc import Callable

    from food_shapley.dataset import FoodSegSplit
    from food_shapley.segments import PlayerSet

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]

CONFIG = Path("configs/week01-clip-blur.toml")

#: The same three images as `test_game_on_real_images.py`, so that an
#: observation about one file can be read against the other.
SAMPLE = (32, 158, 242)


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
    if verify_dish_names():
        pytest.skip("Food-101 prompt set not downloaded; run food_shapley.prompts")
    return DishValueFunction.from_config(settings, device="cpu")


@pytest.fixture(scope="module")
def decomposed(
    split: FoodSegSplit,
    value_function: DishValueFunction,
    settings: ExperimentConfig,
) -> Callable[[int], ImageIndices]:
    """One decomposition per image, computed once and shared.

    Each costs a full powerset of CLIP forward passes, so recomputing it per
    test would make this file minutes slower for no extra coverage.
    """
    cache: dict[int, ImageIndices] = {}

    def build(index: int) -> ImageIndices:
        if index not in cache:
            players: PlayerSet = build_players(
                split[index],
                read_id2label(settings.split),
                image_size=settings.image_size,
                max_players=settings.max_players,
                min_area_fraction=settings.min_area_fraction,
                background_player=settings.background_is_player,
            )
            game = IngredientGame.for_image(players, value_function, settings)
            cache[index] = compute_indices(game, order=settings.interaction_order)
        return cache[index]

    return build


# --------------------------------------------------------------------------- #
# `PLAN.md`'s efficiency check, through the function the run will use
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index", SAMPLE)
def test_the_efficiency_residual_is_far_inside_the_tolerance(
    decomposed: Callable[[int], ImageIndices], index: int
) -> None:
    """The check task 1.6 is gated on, measured rather than merely passed.

    `test_game_on_real_images.py` asserts the identity holds exactly when both
    ends come from one evaluation. This asserts the same thing through
    `compute_indices`, which is what the full run calls, and additionally
    records how much headroom the default tolerance has: if a future change
    moved the residual to just under it, the check would still pass while
    having stopped meaning anything.
    """
    result = decomposed(index)

    assert abs(result.efficiency_residual) <= DEFAULT_EFFICIENCY_TOLERANCE
    # Six orders of magnitude of headroom. Measured: 0.0, -4.4e-16, 0.0 on
    # the three images -- not identically zero, because summing the densified
    # main effects accumulates in a different order than W01-11's read of
    # `shapiq`'s own dict did. Float noise, and the gap to the tolerance is
    # what says so.
    assert abs(result.efficiency_residual) < DEFAULT_EFFICIENCY_TOLERANCE / 1000


@pytest.mark.parametrize("index", SAMPLE)
def test_the_normalised_empty_coalition_is_near_zero_but_not_zero(
    decomposed: Callable[[int], ImageIndices], index: int
) -> None:
    """Why the span subtracts `v(empty)` instead of just reading `v(N)`.

    On paper the normalised empty coalition is 0 and the subtraction is
    redundant -- which is what a synthetic game shows, and why a mutation
    removing it survives the unit suite. On real data it is not redundant:
    the baseline was scored alone at construction time and `ExactComputer`
    re-scores it inside a batch of 32, so the two differ by CLIP's
    batch non-invariance. Measured: 1.6e-06, -2.4e-06 and 6.2e-06 nats.

    Small enough to be rounding, and large enough that dropping the
    subtraction would quietly bias every span in the run in the same
    direction.
    """
    result = decomposed(index)
    empty = float(result.values[~result.coalitions.any(axis=1)][0])

    assert empty != 0.0
    assert abs(empty) < 1e-4


@pytest.mark.parametrize("index", SAMPLE)
def test_the_span_is_the_evidence_the_blur_removed(
    decomposed: Callable[[int], ImageIndices], index: int
) -> None:
    """A degenerate game would make every Shapley value noise around zero."""
    result = decomposed(index)

    assert result.span > 0.0
    assert np.isfinite(result.values).all()


@pytest.mark.parametrize("index", SAMPLE)
def test_the_baseline_recovers_the_target_s_own_log_probability(
    decomposed: Callable[[int], ImageIndices], index: int
) -> None:
    """`baseline + v(N) - v(empty)` must recover the unmasked score.

    The stored values are normalised, so this is the property that makes them
    convertible back to log probabilities.

    It is deliberately *not* asserted as bit equality, and the reason is worth
    stating because the obvious intuition is wrong. Under
    `background_mode = "player"` the grand coalition's rendering really is the
    photograph, and scoring it on its own does return the target's log
    probability bit for bit -- `test_game_on_real_images.py` asserts exactly
    that. But inside `ExactComputer` the grand coalition is scored in a batch
    of 32, and CLIP's float32 forward pass is not batch-invariant (W01-11).
    Measured here: 2.1e-08, 5.5e-08 and 5.5e-07 nats on the three images.

    The bound is ~200x the largest of those so that it is not flaky, and still
    four orders of magnitude below the ~4-nat span being decomposed. Widening
    it further would stop it from catching a real normalisation error.
    """
    result = decomposed(index)

    assert result.baseline + result.span == pytest.approx(result.target.log_prob, abs=1e-4)


# --------------------------------------------------------------------------- #
# Reconstruction: every value is attached to the right set of players
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index", SAMPLE)
def test_the_moebius_transform_re_sums_to_every_coalition_value(
    decomposed: Callable[[int], ImageIndices], index: int
) -> None:
    """`v(S) - v(empty) == sum of the Möbius terms over subsets of S`.

    The definition of the Möbius transform, so it is a tautology for the right
    pairing of values to player sets and false for any other. Checking it over
    *all* coalitions, rather than just the grand one, is what makes it a
    labelling check: the grand coalition's sum is correct under any
    permutation of the players, while an intermediate coalition's is not.

    On real values the terms alternate in sign and are several nats wide, so
    this also says the arithmetic survives the cancellation.
    """
    result = decomposed(index)
    moebius = result.moebius

    for row in range(len(result.values)):
        coalition = result.coalitions[row]
        inside = ~(moebius.members & ~coalition).any(axis=1)
        assert float(moebius.values[inside].sum()) == pytest.approx(
            float(result.values[row]), abs=1e-9
        )


@pytest.mark.parametrize("index", SAMPLE)
def test_the_shapley_values_are_the_ones_the_moebius_terms_imply(
    decomposed: Callable[[int], ImageIndices], index: int
) -> None:
    """Each Möbius term splits evenly among its members -- the Shapley value.

    An independent derivation of the headline number from a different block of
    the same file. The two are computed by separate `shapiq` code paths, so
    agreement means both were read out with the same player convention.
    """
    result = decomposed(index)
    moebius = result.moebius
    sizes = np.maximum(moebius.sizes, 1)

    shares = moebius.values / sizes
    expected = (moebius.members * shares[:, None]).sum(axis=0)

    np.testing.assert_allclose(result.shapley.main_effects(), expected, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Shape: fixed by the configuration, never by the numbers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index", SAMPLE)
def test_the_stored_blocks_have_the_shape_the_configuration_implies(
    decomposed: Callable[[int], ImageIndices], settings: ExperimentConfig, index: int
) -> None:
    """Real games are where `shapiq`'s zero-dropping would actually bite."""
    result = decomposed(index)
    order = min(settings.interaction_order, result.n_players)

    assert len(result.shapley) == 1 + result.n_players
    assert len(result.interactions) == len(list(powerset(range(result.n_players), max_size=order)))
    assert len(result.moebius) == 2**result.n_players
    assert len(result.values) == 2**result.n_players


@pytest.mark.parametrize("index", SAMPLE)
def test_every_ingredient_has_a_shapley_value_and_the_scene_is_not_one(
    decomposed: Callable[[int], ImageIndices], index: int
) -> None:
    """Ingredient-level statistics must filter on `ingredient_indices` (W01-09)."""
    result = decomposed(index)

    assert len(result.shapley.main_effects()) == result.n_players
    assert len(result.ingredient_indices) == result.n_players - 1
    assert set(result.ingredient_indices) < set(range(result.n_players))


def test_real_images_are_not_perfectly_additive(
    decomposed: Callable[[int], ImageIndices],
) -> None:
    """Guards the whole project's premise against a silently additive game.

    If every pairwise interaction came out zero there would be no research
    question left, and the most likely cause would be a bug -- a value
    function that ignored its coalition, say -- not a discovery. This asserts
    only that the interactions are not all zero; **how** interactive these
    models are is a result, and belongs in `insights/`, not in an assertion.
    """
    pairs = [
        float(np.abs(result.interactions.values[result.interactions.sizes == 2]).max())
        for result in (decomposed(index) for index in SAMPLE)
    ]

    assert min(pairs) > 1e-6


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def test_a_real_decomposition_round_trips_bit_for_bit(
    decomposed: Callable[[int], ImageIndices], tmp_path: Path
) -> None:
    """Float64 values from a real model must not be quantised by storage."""
    result = decomposed(SAMPLE[0])
    path = tmp_path / "image.npz"

    save_indices(path, result)
    reloaded = load_indices(path)

    assert reloaded.target == result.target
    assert reloaded.player_names == result.player_names
    assert reloaded.baseline == result.baseline
    assert reloaded.efficiency_residual == result.efficiency_residual
    np.testing.assert_array_equal(reloaded.values, result.values)
    np.testing.assert_array_equal(reloaded.coalitions, result.coalitions)
    np.testing.assert_array_equal(reloaded.shapley.values, result.shapley.values)
    np.testing.assert_array_equal(reloaded.interactions.values, result.interactions.values)
    np.testing.assert_array_equal(reloaded.moebius.values, result.moebius.values)


def test_the_command_line_entry_point_decomposes_one_image(tmp_path: Path) -> None:
    """`PLAN.md` step 4 asks for a single image end to end before the full run."""
    path = tmp_path / "image.npz"

    code = main(
        [
            "--config",
            str(CONFIG),
            "--image",
            str(SAMPLE[0]),
            "--device",
            "cpu",
            "--save",
            str(path),
        ]
    )

    assert code == 0
    assert load_indices(path).image_index == SAMPLE[0]
