"""The dataset -> players seam, on real FoodSeg103 masks.

The unit tests write their own masks, which is the right way to test the
merging rules but cannot catch a wrong assumption about what the real data
looks like. These read the downloaded split -- they are skipped, not failed, on
a machine that has not downloaded it, and they load no model.

The settings come from ``configs/week01-clip-blur.toml`` rather than being
repeated here, so that changing the run configuration cannot leave this test
quietly validating something else.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from food_shapley.config import ExperimentConfig, load_config
from food_shapley.dataset import load_split
from food_shapley.download import BACKGROUND_LABEL, verify_split
from food_shapley.masking import BlurRemoval
from food_shapley.segments import PlayerSet, build_players

if TYPE_CHECKING:
    from collections.abc import Callable

    from food_shapley.dataset import FoodSegSplit

pytestmark = pytest.mark.integration

CONFIG = Path("configs/week01-clip-blur.toml")

# A fixed, small sample: enough to exercise real masks, fast enough to stay in
# the default test run. Indices are stable for the pinned mirror revision.
SAMPLE = (0, 1, 7, 42, 100, 500, 1000, 2134)


@pytest.fixture(scope="module")
def settings() -> ExperimentConfig:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def split() -> FoodSegSplit:
    if verify_split("test"):
        pytest.skip("FoodSeg103 test split not downloaded; run food_shapley.download")
    return load_split("test")


@pytest.fixture(scope="module")
def players(split: FoodSegSplit, settings: ExperimentConfig) -> Callable[[int], PlayerSet]:
    """Build the player set for one image under the Week-1 settings."""

    def build(index: int) -> PlayerSet:
        return build_players(
            split[index],
            split.names,
            image_size=settings.image_size,
            max_players=settings.max_players,
            min_area_fraction=settings.min_area_fraction,
            background_player=settings.background_is_player,
        )

    return build


@pytest.mark.parametrize("index", SAMPLE)
def test_players_partition_the_frame(players: Callable[[int], PlayerSet], index: int) -> None:
    """Coalitions are subsets of a partition; overlap would double-count pixels."""
    result = players(index)
    stack = np.stack([player.mask for player in result.players])
    assert stack.sum(axis=0).max() == 1
    np.testing.assert_array_equal(stack.any(axis=0), ~result.background)


@pytest.mark.parametrize("index", SAMPLE)
def test_players_are_non_empty_named_and_within_the_cap(
    players: Callable[[int], PlayerSet], settings: ExperimentConfig, index: int
) -> None:
    """An empty player would have Shapley value zero by construction."""
    result = players(index)
    assert 1 <= len(result) <= settings.max_players
    for player in result.players:
        assert player.mask.any()
        assert player.name.strip()


@pytest.mark.parametrize("index", SAMPLE)
def test_the_scene_is_a_player_but_never_an_ingredient(
    players: Callable[[int], PlayerSet], settings: ExperimentConfig, index: int
) -> None:
    """`PLAN.md`'s rule, restated for ``background_mode = "player"``."""
    result = players(index)
    ingredients = [result.players[i] for i in result.ingredient_indices]
    assert BACKGROUND_LABEL not in [player.label for player in ingredients]
    if settings.background_is_player:
        assert result.scene_index is not None
        assert result.players[result.scene_index].name == "scene"
        assert len(ingredients) == len(result) - 1
    else:
        assert result.scene_index is None


@pytest.mark.parametrize("index", SAMPLE)
def test_the_grand_coalition_is_the_photograph(
    players: Callable[[int], PlayerSet], settings: ExperimentConfig, index: int
) -> None:
    """Why the scene is a player: v(N) must be an image the model could have seen.

    Checked on real masks because it depends on every pixel having an owner,
    which is a property of the data as much as of the merging rules.
    """
    if not settings.background_is_player:
        pytest.skip("only meaningful when the scene is a player")
    result = players(index)
    operator = BlurRemoval(
        radius=settings.blur_radius, keep_background=settings.background_stays_sharp
    )
    np.testing.assert_array_equal(operator.render(result, range(len(result))), result.image)
    np.testing.assert_array_equal(operator.render(result, []), operator.blurred(result))


@pytest.mark.parametrize("index", SAMPLE)
def test_image_and_mask_share_the_model_geometry(
    players: Callable[[int], PlayerSet], settings: ExperimentConfig, index: int
) -> None:
    result = players(index)
    size = settings.image_size
    assert result.image.shape == (size, size, 3)
    assert result.background.shape == (size, size)
    assert result.coalition_mask(range(len(result))).shape == (size, size)


def test_row_order_is_pinned_by_the_revision(split: FoodSegSplit) -> None:
    """A pinned revision must give a pinned row order, or no index is reproducible."""
    assert len(split) == 2135
    assert split[0].ingredient_labels == (20, 56, 66, 84, 96)
    assert split.ingredient_name(20) == "dried cranberries"
