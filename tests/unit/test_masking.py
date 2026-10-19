from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from food_shapley.dataset import FoodSegExample
from food_shapley.masking import BlurRemoval
from food_shapley.segments import PlayerSet, build_players

# Synthetic throughout. The removal operator is a pure function of a player set
# and a coalition, so a hand-written mask pins it down completely; nothing here
# needs the real split or a model.

NAMES = {0: "background", **{i: f"ingredient-{i}" for i in range(1, 104)}}

#: Saturated, far apart, and distinguishable after a 12 px blur mixes them.
COLOURS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    11: (255, 0, 0),
    22: (0, 255, 0),
    33: (0, 0, 255),
}


def striped_example(labels: tuple[int, ...] = (0, 11, 22, 33)) -> FoodSegExample:
    """An image whose colour and label encode the same vertical stripes.

    Colour-label agreement is what makes the assertions below alignment checks
    rather than mere consistency checks: a sharp pixel of the wrong colour means
    the operator kept the wrong region.
    """
    width, height = 448, 448
    mask = np.zeros((height, width), dtype=np.uint8)
    edges = np.linspace(0, width, len(labels) + 1).astype(int)
    for label, start, stop in zip(labels, edges[:-1], edges[1:], strict=True):
        mask[:, start:stop] = label
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for label, colour in COLOURS.items():
        rgb[mask == label] = colour
    return FoodSegExample(index=0, image=Image.fromarray(rgb), mask=mask)


@pytest.fixture
def players() -> PlayerSet:
    """Three ingredient players plus background, in CLIP's geometry."""
    return build_players(striped_example(), NAMES, image_size=224, max_players=10)


@pytest.fixture
def operator() -> BlurRemoval:
    return BlurRemoval(radius=12.0, keep_background=False)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_non_positive_radius_is_rejected(radius: float) -> None:
    """Radius 0 makes every coalition identical, so every Shapley value is 0.

    Efficiency would still hold, so nothing downstream would notice.
    """
    with pytest.raises(ValueError, match="blur radius must be positive"):
        BlurRemoval(radius=radius, keep_background=False)


# --------------------------------------------------------------------------- #
# What the operator keeps and what it hides
# --------------------------------------------------------------------------- #


def test_visible_players_are_pixel_identical_to_the_input(
    players: PlayerSet, operator: BlurRemoval
) -> None:
    rendered = operator.render(players, [0])
    kept = players.players[0].mask
    np.testing.assert_array_equal(rendered[kept], players.image[kept])


def test_hidden_players_are_changed(players: PlayerSet, operator: BlurRemoval) -> None:
    rendered = operator.render(players, [0])
    hidden = players.players[1].mask
    assert not np.array_equal(rendered[hidden], players.image[hidden])


def test_each_player_region_keeps_its_own_colour(players: PlayerSet, operator: BlurRemoval) -> None:
    """The alignment check, done programmatically.

    If the mask were off by a resize or a crop offset, the sharp region would
    carry a neighbouring stripe's colour. The assertion is on the dominant
    channel rather than the exact triple because bicubic resampling rings at
    the stripe boundaries -- that happens in ``players.image`` itself, before
    any masking, so demanding one exact colour would test the resize, not the
    operator.
    """
    for index, player in enumerate(players.players):
        assert player.label is not None
        dominant = int(np.argmax(np.asarray(COLOURS[player.label])))
        rendered = operator.render(players, [index])
        pixels = rendered[player.mask]
        assert (np.argmax(pixels, axis=1) == dominant).mean() > 0.95, player.name


def test_hidden_region_does_not_keep_its_own_colour(
    players: PlayerSet, operator: BlurRemoval
) -> None:
    """The converse: a blurred stripe must not still read as a pure stripe.

    Without this, an operator that silently did nothing would pass the test
    above.
    """
    rendered = operator.render(players, [0])
    hidden = players.players[1]
    assert hidden.label is not None
    pure = np.all(rendered[hidden.mask] == np.asarray(COLOURS[hidden.label]), axis=1)
    assert pure.mean() < 0.9


# --------------------------------------------------------------------------- #
# The two extremes of the lattice, which are v(N) and v(empty)
# --------------------------------------------------------------------------- #


def test_full_coalition_with_background_kept_is_the_original(players: PlayerSet) -> None:
    """v(N) must be the unmodified image, or every Shapley sum is off by a blur."""
    operator = BlurRemoval(radius=12.0, keep_background=True)
    rendered = operator.render(players, range(len(players)))
    np.testing.assert_array_equal(rendered, players.image)


def test_full_coalition_without_background_still_blurs_the_background(
    players: PlayerSet, operator: BlurRemoval
) -> None:
    background = players.background
    assert background.any()
    assert not np.array_equal(
        operator.render(players, range(len(players)))[background], players.image[background]
    )


def test_empty_coalition_is_the_fully_blurred_image(
    players: PlayerSet, operator: BlurRemoval
) -> None:
    """`docs/week-01.md`: with keep_background False, v(empty) is all blur."""
    np.testing.assert_array_equal(operator.render(players, []), operator.blurred(players))


def test_empty_coalition_keeps_the_scene_when_background_is_kept(players: PlayerSet) -> None:
    operator = BlurRemoval(radius=12.0, keep_background=True)
    rendered = operator.render(players, [])
    background = players.background
    np.testing.assert_array_equal(rendered[background], players.image[background])
    for player in players.players:
        assert not np.array_equal(rendered[player.mask], players.image[player.mask])


# --------------------------------------------------------------------------- #
# Properties the game relies on
# --------------------------------------------------------------------------- #


def test_coalitions_are_unions_of_their_members(players: PlayerSet, operator: BlurRemoval) -> None:
    """Rendering {0, 1} must keep exactly what {0} and {1} keep, and no more."""
    both = operator.render(players, [0, 1])
    for index in (0, 1):
        kept = players.players[index].mask
        np.testing.assert_array_equal(both[kept], players.image[kept])
    other = players.players[2].mask
    assert not np.array_equal(both[other], players.image[other])


def test_render_is_deterministic(players: PlayerSet, operator: BlurRemoval) -> None:
    np.testing.assert_array_equal(
        operator.render(players, [0, 2]), operator.render(players, [0, 2])
    )


def test_coalition_order_does_not_matter(players: PlayerSet, operator: BlurRemoval) -> None:
    np.testing.assert_array_equal(
        operator.render(players, [2, 0]), operator.render(players, [0, 2])
    )


def test_render_preserves_shape_and_dtype(players: PlayerSet, operator: BlurRemoval) -> None:
    rendered = operator.render(players, [1])
    assert rendered.shape == players.image.shape
    assert rendered.dtype == np.uint8


def test_render_does_not_mutate_the_player_set(players: PlayerSet, operator: BlurRemoval) -> None:
    before = players.image.copy()
    operator.render(players, [0, 1])
    np.testing.assert_array_equal(players.image, before)


def test_out_of_range_player_is_rejected(players: PlayerSet, operator: BlurRemoval) -> None:
    """A silently ignored index would produce a plausible but wrong game."""
    with pytest.raises(IndexError, match="out of range"):
        operator.render(players, [len(players)])


def test_a_larger_radius_hides_more(players: PlayerSet) -> None:
    hidden = players.players[1].mask
    original = players.image[hidden].astype(np.int32)
    distances = [
        float(
            np.abs(
                BlurRemoval(radius=radius, keep_background=False)
                .render(players, [0])[hidden]
                .astype(np.int32)
                - original
            ).mean()
        )
        for radius in (2.0, 24.0)
    ]
    assert distances[0] < distances[1]


def test_blurred_is_independent_of_the_coalition(players: PlayerSet, operator: BlurRemoval) -> None:
    """The baseline is cached and reused across coalitions; it must not drift."""
    baseline = operator.blurred(players)
    operator.render(players, [0])
    np.testing.assert_array_equal(operator.blurred(players), baseline)
