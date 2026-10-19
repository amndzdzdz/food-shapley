from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from food_shapley.config import ExperimentConfig
from food_shapley.dataset import FoodSegExample
from food_shapley.masking import BlurRemoval
from food_shapley.preview import contact_sheet, select_indices
from food_shapley.segments import PlayerSet, build_players

if TYPE_CHECKING:
    from pathlib import Path

# The sheets are a diagnostic for a human, so these tests check only the
# properties a human cannot verify by looking: that every coalition is present,
# that the sampled images are reproducible from the config, and that nothing is
# silently dropped. Whether the bun looks like a bun is the point of the manual
# check and cannot be asserted here.

NAMES = {0: "background", **{i: f"ingredient-{i}" for i in range(1, 104)}}


@pytest.fixture
def players() -> PlayerSet:
    mask = np.zeros((448, 448), dtype=np.uint8)
    mask[:, 112:224] = 11
    mask[:, 224:336] = 22
    image = Image.new("RGB", (448, 448), (120, 60, 30))
    return build_players(
        FoodSegExample(index=3, image=image, mask=mask), NAMES, image_size=224, max_players=10
    )


@pytest.fixture
def scene_players() -> PlayerSet:
    """The same layout under ``background_mode = "player"``.

    Textured rather than flat: a flat image is its own blur, which would make
    every "the operator changed something" assertion vacuously false.
    """
    mask = np.zeros((448, 448), dtype=np.uint8)
    mask[:, 112:224] = 11
    mask[:, 224:336] = 22
    rng = np.random.default_rng(3)
    image = Image.fromarray(rng.integers(0, 256, (448, 448, 3), dtype=np.uint8))
    return build_players(
        FoodSegExample(index=3, image=image, mask=mask),
        NAMES,
        image_size=224,
        max_players=10,
        background_player=True,
    )


@pytest.fixture
def operator() -> BlurRemoval:
    return BlurRemoval(radius=12.0, keep_background=False)


def panel_count(sheet: Image.Image) -> int:
    """How many panels a one-column sheet holds, read off its height."""
    cell = 224 + 34
    return (sheet.height - 6) // (cell + 6)


def test_sheet_holds_original_empty_every_singleton_and_full(
    players: PlayerSet, operator: BlurRemoval
) -> None:
    """Panels = original + empty + n singletons + grand coalition."""
    sheet = contact_sheet(players, operator, columns=10)
    assert len(players) == 2
    expected = 1 + 1 + len(players) + 1
    assert sheet.width == 10 * 224 + 11 * 6
    # One row is enough for `expected` panels at 10 columns.
    assert sheet.height == (224 + 34) + 2 * 6
    assert expected <= 10


def test_sheet_gains_a_scene_panel_when_the_scene_is_a_player(
    players: PlayerSet, scene_players: PlayerSet, operator: BlurRemoval
) -> None:
    """One extra player, and one extra panel: "every ingredient, no scene"."""
    assert len(scene_players) == len(players) + 1
    plain = panel_count(contact_sheet(players, operator, columns=1))
    with_scene = panel_count(contact_sheet(scene_players, operator, columns=1))
    # +1 for the scene's own singleton, +1 for the ingredients-only comparison.
    assert with_scene == plain + 2


def test_scene_sheet_shows_the_untouched_photograph_twice(
    scene_players: PlayerSet, operator: BlurRemoval
) -> None:
    """The first and last panels must match, or the sheet's v(N) is not the photograph."""
    empty = operator.render(scene_players, [])
    grand = operator.render(scene_players, range(len(scene_players)))
    np.testing.assert_array_equal(grand, scene_players.image)
    assert not np.array_equal(empty, grand)


def test_sheet_wraps_onto_several_rows(players: PlayerSet, operator: BlurRemoval) -> None:
    narrow = contact_sheet(players, operator, columns=2)
    wide = contact_sheet(players, operator, columns=5)
    assert narrow.height > wide.height


def test_single_player_sheet_omits_the_duplicate_grand_coalition() -> None:
    """With one player the singleton *is* N; two identical panels mislead."""
    mask = np.zeros((448, 448), dtype=np.uint8)
    mask[:, 100:300] = 11
    players = build_players(
        FoodSegExample(index=0, image=Image.new("RGB", (448, 448)), mask=mask),
        NAMES,
        image_size=224,
        max_players=10,
    )
    operator = BlurRemoval(radius=12.0, keep_background=False)
    one = contact_sheet(players, operator, columns=10)
    assert one.width == 10 * 224 + 11 * 6
    # original + empty + the one singleton, and no separate "all" panel.
    assert len(players) == 1


@pytest.mark.parametrize("columns", [0, -1])
def test_sheet_rejects_a_non_positive_column_count(
    players: PlayerSet, operator: BlurRemoval, columns: int
) -> None:
    with pytest.raises(ValueError, match="columns must be at least 1"):
        contact_sheet(players, operator, columns=columns)


@pytest.mark.parametrize("scale", [0, -2])
def test_sheet_rejects_a_non_positive_scale(
    players: PlayerSet, operator: BlurRemoval, scale: int
) -> None:
    with pytest.raises(ValueError, match="scale must be at least 1"):
        contact_sheet(players, operator, scale=scale)


def test_scale_magnifies_the_panels(players: PlayerSet, operator: BlurRemoval) -> None:
    one = contact_sheet(players, operator, columns=10, scale=1)
    three = contact_sheet(players, operator, columns=10, scale=3)
    assert three.width == 10 * 224 * 3 + 11 * 6
    assert three.width > one.width


def test_scale_does_not_interpolate(players: PlayerSet, operator: BlurRemoval) -> None:
    """Smoothing the panels would hide exactly the errors this check looks for.

    Nearest-neighbour magnification can invent no colour that is not already in
    the 224 px panel, so the set of colours is unchanged.
    """
    one = np.asarray(contact_sheet(players, operator, columns=10, scale=1))
    four = np.asarray(contact_sheet(players, operator, columns=10, scale=4))
    assert set(map(tuple, np.unique(four.reshape(-1, 3), axis=0))) <= set(
        map(tuple, np.unique(one.reshape(-1, 3), axis=0))
    )


def test_sheet_is_deterministic(players: PlayerSet, operator: BlurRemoval, tmp_path: Path) -> None:
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    contact_sheet(players, operator).save(first)
    contact_sheet(players, operator).save(second)
    assert first.read_bytes() == second.read_bytes()


# --------------------------------------------------------------------------- #
# Image selection
# --------------------------------------------------------------------------- #


def test_explicit_indices_win_and_are_deduplicated(config: ExperimentConfig) -> None:
    assert select_indices(config, 100, 3, [7, 2, 7]) == [2, 7]


def test_sample_is_reproducible_from_the_config(config: ExperimentConfig) -> None:
    """'The images I looked at' must be recoverable from the config alone."""
    assert select_indices(config, 500, 4, []) == select_indices(config, 500, 4, [])


def test_sample_depends_on_the_seed(config: ExperimentConfig) -> None:
    other = ExperimentConfig(**(config.to_dict() | {"seed": 1}))
    assert select_indices(config, 500, 4, []) != select_indices(other, 500, 4, [])


def test_sample_is_clamped_to_the_split_size(config: ExperimentConfig) -> None:
    assert select_indices(config, 2, 10, []) == [0, 1]


@pytest.mark.parametrize("explicit", [[-1], [100], [0, 100]])
def test_out_of_range_indices_are_rejected(config: ExperimentConfig, explicit: list[int]) -> None:
    with pytest.raises(ValueError, match="out of range"):
        select_indices(config, 100, 1, explicit)


def test_non_positive_sample_size_is_rejected(config: ExperimentConfig) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        select_indices(config, 100, 0, [])
