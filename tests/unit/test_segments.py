from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from food_shapley.dataset import FoodSegExample
from food_shapley.masking import BlurRemoval
from food_shapley.segments import (
    REST_PLAYER_NAME,
    SCENE_PLAYER_NAME,
    PlayerSet,
    build_players,
    clip_geometry,
    resize_image,
    resize_mask,
)

# Everything here is synthetic. The real split is not read: these tests are
# about the geometry and the merging rules, both of which are fully determined
# by a mask the test wrote itself.

NAMES = {0: "background", **{i: f"ingredient-{i}" for i in range(1, 104)}}


def make_example(
    mask: np.ndarray, *, index: int = 0, image: Image.Image | None = None
) -> FoodSegExample:
    """A FoodSeg103-shaped example from a hand-written mask."""
    height, width = mask.shape
    if image is None:
        image = Image.new("RGB", (width, height), (10, 20, 30))
    return FoodSegExample(index=index, image=image, mask=mask.astype(np.uint8))


def stripes(width: int, height: int, labels: tuple[int, ...]) -> np.ndarray:
    """A mask of equal-width vertical stripes, left to right."""
    mask = np.zeros((height, width), dtype=np.uint8)
    edges = np.linspace(0, width, len(labels) + 1).astype(int)
    for label, start, stop in zip(labels, edges[:-1], edges[1:], strict=True):
        mask[:, start:stop] = label
    return mask


# --------------------------------------------------------------------------- #
# Geometry
#
# The highest-value tests in this file. A one-pixel disagreement with the
# model's own preprocessing misaligns every mask against every image, and no
# efficiency check downstream can detect it -- the game stays perfectly
# self-consistent while measuring the wrong pixels.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "size",
    [(512, 485), (485, 512), (224, 224), (640, 480), (480, 640), (1000, 521), (225, 224)],
)
def test_image_resize_matches_torchvision(size: tuple[int, int]) -> None:
    """open_clip's transform is torchvision's; equality here is equality with CLIP."""
    from torchvision.transforms import CenterCrop, Compose, Resize
    from torchvision.transforms.functional import InterpolationMode

    rng = np.random.default_rng(0)
    width, height = size
    image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))

    reference = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224)])(
        image
    )
    np.testing.assert_array_equal(resize_image(image, 224), np.asarray(reference))


@pytest.mark.parametrize("size", [(512, 485), (485, 512), (640, 480), (1000, 521)])
def test_mask_resize_matches_torchvision_nearest(size: tuple[int, int]) -> None:
    from torchvision.transforms import CenterCrop, Compose, Resize
    from torchvision.transforms.functional import InterpolationMode

    rng = np.random.default_rng(1)
    width, height = size
    mask = rng.integers(0, 104, (height, width), dtype=np.uint8)

    reference = Compose([Resize(224, interpolation=InterpolationMode.NEAREST), CenterCrop(224)])(
        Image.fromarray(mask)
    )
    np.testing.assert_array_equal(resize_mask(mask, 224), np.asarray(reference))


def test_geometry_truncates_the_long_side() -> None:
    """Torchvision truncates here; rounding would shift the crop by a pixel."""
    geometry = clip_geometry((512, 485), 224)
    assert geometry.resized == (236, 224)  # int(224 * 512 / 485) == 236, not 237
    assert (geometry.left, geometry.top) == (6, 0)


def test_geometry_is_a_no_op_on_a_square_at_target_size() -> None:
    geometry = clip_geometry((224, 224), 224)
    assert geometry.resized == (224, 224)
    assert geometry.box == (0, 0, 224, 224)


@pytest.mark.parametrize(("size", "target"), [((0, 10), 224), ((10, 10), 0), ((10, -1), 224)])
def test_geometry_rejects_degenerate_input(size: tuple[int, int], target: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        clip_geometry(size, target)


def test_mask_resize_invents_no_labels() -> None:
    """Interpolating a label map would blend two ingredients into a third."""
    mask = stripes(300, 200, (0, 5, 9))
    resized = resize_mask(mask, 224)
    assert set(np.unique(resized)) <= {0, 5, 9}


def test_image_and_mask_stay_aligned() -> None:
    """The failure `PLAN.md` warns about: 'only bun visible' not showing the bun.

    Colour and label encode the same stripes, so any divergence in resampling
    or crop offset shows up as a pixel whose colour contradicts its label.
    """
    labels = (11, 22, 33, 44)
    colours = {11: (255, 0, 0), 22: (0, 255, 0), 33: (0, 0, 255), 44: (255, 255, 0)}
    mask = stripes(448, 336, labels)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, colour in colours.items():
        rgb[mask == label] = colour

    image_out = resize_image(Image.fromarray(rgb), 224)
    mask_out = resize_mask(mask, 224)

    # Bicubic ringing smears stripe boundaries, so compare away from them: the
    # question is whether the regions line up, not whether bicubic is sharp.
    for label, colour in colours.items():
        region = mask_out == label
        if not region.any():
            continue
        eroded = region.copy()
        eroded[:, :-3] &= region[:, 3:]
        eroded[:, 3:] &= region[:, :-3]
        if not eroded.any():
            continue
        dominant = np.argmax(np.asarray(colour))
        pixels = image_out[eroded]
        assert (np.argmax(pixels, axis=1) == dominant).mean() > 0.95, label


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #


def build(mask: np.ndarray, **kwargs: object) -> PlayerSet:
    """`build_players` with the arguments these tests do not care about fixed."""
    options: dict[str, object] = {
        "image_size": 224,
        "max_players": 10,
        "min_area_fraction": 0.0,
    }
    options.update(kwargs)
    return build_players(make_example(mask), NAMES, **options)  # type: ignore[arg-type]


def test_background_is_not_an_ingredient() -> None:
    """`PLAN.md` names this explicitly as the easy mistake."""
    players = build(stripes(448, 448, (0, 7, 8)))
    assert 0 not in players.labels
    assert players.labels == (7, 8)


def test_background_pixels_belong_to_no_player() -> None:
    players = build(stripes(448, 448, (0, 7, 8)))
    assert players.background.any()
    for player in players.players:
        assert not (player.mask & players.background).any()


def test_players_are_disjoint_and_cover_the_food_region() -> None:
    """Coalitions are subsets of a partition; overlap would double-count pixels."""
    players = build(stripes(448, 448, (0, 7, 8, 9)))
    stack = np.stack([player.mask for player in players.players])
    assert stack.sum(axis=0).max() == 1
    np.testing.assert_array_equal(stack.any(axis=0), ~players.background)


def test_player_order_is_ascending_class_id() -> None:
    """Coalition indices are stored on disk, so the order must not drift."""
    players = build(stripes(448, 448, (9, 3, 7, 0)))
    assert players.labels == (3, 7, 9)


def test_names_come_from_the_pinned_mapping() -> None:
    players = build(stripes(448, 448, (0, 42)))
    assert players.names == ("ingredient-42",)


def test_ingredients_cropped_away_are_dropped_not_kept_empty() -> None:
    """An empty player has Shapley value 0 by construction and would dilute every statistic."""
    # A wide image: the centre crop keeps the middle, so the leftmost stripe goes.
    mask = stripes(1000, 224, (5, 6, 7, 8, 9))
    players = build(mask)
    assert 5 in players.dropped_labels
    assert 5 not in players.labels
    assert all(player.mask.any() for player in players.players)


def test_no_surviving_ingredient_is_an_error_not_an_empty_game() -> None:
    mask = np.zeros((224, 1000), dtype=np.uint8)
    mask[:, :50] = 5
    with pytest.raises(ValueError, match="no ingredient pixels survive"):
        build(mask)


# --------------------------------------------------------------------------- #
# The scene player
#
# `background_mode = "player"` exists so that v(N) is the photograph and
# v(empty set) is a fully blurred frame at the same time. Those two properties
# are the whole point of the mode, so they are tested directly against the
# removal operator rather than inferred from the masks.
# --------------------------------------------------------------------------- #


def textured(width: int, height: int) -> Image.Image:
    """An image blur actually changes, so blur-based assertions are not vacuous."""
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def test_scene_becomes_a_player_when_asked() -> None:
    players = build(stripes(448, 448, (0, 7, 8)), background_player=True)
    assert players.labels == (7, 8, 0)
    assert players.names[-1] == SCENE_PLAYER_NAME
    assert players.players[-1].is_scene


def test_scene_is_last_so_it_cannot_renumber_an_ingredient() -> None:
    """Coalition indices are written to disk; adding a player must not shift them."""
    mask = stripes(448, 448, (0, 3, 7, 9))
    without = build(mask)
    with_scene = build(mask, background_player=True)
    assert with_scene.labels[: len(without)] == without.labels
    assert with_scene.ingredient_indices == tuple(range(len(without)))


def test_scene_is_not_an_ingredient() -> None:
    """`PLAN.md`'s rule survives: it is a player, and it is still not food."""
    players = build(stripes(448, 448, (0, 7, 8)), background_player=True)
    assert players.scene_index == 2
    assert players.ingredient_indices == (0, 1)
    assert not players.players[-1].is_ingredient
    assert all(players.players[i].is_ingredient for i in players.ingredient_indices)


def test_rest_counts_as_an_ingredient() -> None:
    """`rest` is a union of real ingredients, so it belongs in food statistics."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:, 20:100] = 7
    mask[:, 100:220] = 8
    mask[:, 220:222] = 9
    mask[:, 222:] = 10
    players = build(mask, min_area_fraction=0.05, background_player=True)
    rest = players.players[-2]
    assert rest.is_rest
    assert rest.is_ingredient
    assert players.ingredient_indices == (0, 1, 2)


def test_scene_index_is_none_without_the_scene_player() -> None:
    players = build(stripes(448, 448, (0, 7, 8)))
    assert players.scene_index is None
    assert players.ingredient_indices == (0, 1)


def test_every_pixel_has_an_owner_when_the_scene_is_a_player() -> None:
    players = build(stripes(448, 448, (0, 7, 8)), background_player=True)
    assert not players.background.any()
    assert players.coalition_mask(range(len(players))).all()


def test_scene_is_disjoint_from_the_ingredients() -> None:
    players = build(stripes(448, 448, (0, 7, 8, 9)), background_player=True)
    stack = np.stack([player.mask for player in players.players])
    assert stack.sum(axis=0).max() == 1


def test_grand_coalition_reproduces_the_photograph() -> None:
    """The reason for the mode: v(N) must be an image the model could have seen."""
    mask = stripes(448, 448, (0, 7, 8))
    players = build_players(
        make_example(mask, image=textured(448, 448)),
        NAMES,
        image_size=224,
        max_players=10,
        background_player=True,
    )
    operator = BlurRemoval(radius=6.0, keep_background=False)
    np.testing.assert_array_equal(operator.render(players, range(len(players))), players.image)


def test_empty_coalition_is_a_fully_blurred_frame() -> None:
    """The other half: v(empty set) hides the scene as well as the food."""
    mask = stripes(448, 448, (0, 7, 8))
    players = build_players(
        make_example(mask, image=textured(448, 448)),
        NAMES,
        image_size=224,
        max_players=10,
        background_player=True,
    )
    operator = BlurRemoval(radius=6.0, keep_background=False)
    rendered = operator.render(players, [])
    np.testing.assert_array_equal(rendered, operator.blurred(players))
    assert not np.array_equal(rendered, players.image)


def test_keep_background_is_inert_once_the_scene_is_a_player() -> None:
    """No pixel is ownerless, so the operator's background switch has nothing to act on."""
    mask = stripes(448, 448, (0, 7, 8))
    players = build_players(
        make_example(mask, image=textured(448, 448)),
        NAMES,
        image_size=224,
        max_players=10,
        background_player=True,
    )
    sharp = BlurRemoval(radius=6.0, keep_background=True)
    blurred = BlurRemoval(radius=6.0, keep_background=False)
    np.testing.assert_array_equal(
        sharp.render(players, [0]),
        blurred.render(players, [0]),
    )


def test_no_scene_player_when_nothing_is_left_over() -> None:
    """An empty player has Shapley value 0 by construction; the same rule as dropping."""
    players = build(stripes(448, 448, (7, 8)), background_player=True)
    assert players.scene_index is None
    assert players.labels == (7, 8)
    assert not players.background.any()


def test_scene_takes_a_slot_from_the_cap() -> None:
    """A cap is a statement about 2**n, and the scene costs what any player costs."""
    labels = tuple(range(1, 9))
    players = build(stripes(448, 448, (0, *labels)), max_players=4, background_player=True)
    assert len(players) == 4
    assert players.capped
    assert players.players[-1].is_scene
    assert len(players.ingredient_indices) == 3


@pytest.mark.parametrize("max_players", [2, 3, 4, 5, 8])
def test_the_cap_counts_the_scene_too(max_players: int) -> None:
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:, :60] = 1
    mask[:, 60:110] = 2
    mask[:, 110:150] = 3
    mask[:, 150:152] = 4
    mask[:, 152:200] = 5
    players = build(mask, max_players=max_players, min_area_fraction=0.05, background_player=True)
    assert len(players) <= max_players
    assert players.players[-1].is_scene


def test_a_scene_player_needs_at_least_two_slots() -> None:
    """max_players=1 with a scene player is a game with no food in it at all."""
    with pytest.raises(ValueError, match="max_players must be at least 2"):
        build(stripes(448, 448, (0, 7)), max_players=1, background_player=True)


def test_area_fractions_sum_to_one_with_the_scene() -> None:
    players = build(stripes(448, 448, (0, 7, 8, 9)), background_player=True)
    total = sum(player.area_fraction for player in players.players)
    assert total == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Merging and capping
# --------------------------------------------------------------------------- #


def speck_mask(*, speck_columns: int) -> np.ndarray:
    """Two large stripes plus a narrow speck of class 9."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:, :100] = 7
    mask[:, 100:-speck_columns] = 8
    mask[:, -speck_columns:] = 9
    return mask


def test_min_area_fraction_keeps_a_large_segment() -> None:
    players = build(speck_mask(speck_columns=40), min_area_fraction=0.005)
    assert players.labels == (7, 8, 9)


def test_a_lone_small_segment_keeps_its_name() -> None:
    """Renaming one segment to "rest" covers identical pixels under a worse name."""
    players = build(speck_mask(speck_columns=1), min_area_fraction=0.05)
    assert players.labels == (7, 8, 9)
    assert REST_PLAYER_NAME not in players.names


def test_two_small_segments_merge_into_rest() -> None:
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:, :100] = 7
    mask[:, 100:220] = 8
    mask[:, 220:222] = 9
    mask[:, 222:] = 10
    players = build(mask, min_area_fraction=0.05)

    assert players.labels == (7, 8, None)
    rest = players.players[-1]
    assert rest.is_rest
    assert rest.name == REST_PLAYER_NAME
    assert rest.merged_labels == (9, 10)


def test_rest_covers_exactly_the_merged_pixels() -> None:
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:, :100] = 7
    mask[:, 100:220] = 8
    mask[:, 220:222] = 9
    mask[:, 222:] = 10
    players = build(mask, min_area_fraction=0.05)
    expected = (resize_mask(mask, 224) == 9) | (resize_mask(mask, 224) == 10)
    np.testing.assert_array_equal(players.players[-1].mask, expected)


def test_rest_is_not_named_after_class_103() -> None:
    """FoodSeg103 class 103 is called "other ingredients"; the two must stay apart."""
    assert NAMES[103] != REST_PLAYER_NAME
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:, :100] = 7
    mask[:, 100:220] = 8
    mask[:, 220:222] = 9
    mask[:, 222:] = 103
    players = build(mask, min_area_fraction=0.05)
    assert players.names[-1] == REST_PLAYER_NAME
    assert players.players[-1].merged_labels == (9, 103)


def test_everything_tiny_still_leaves_a_named_player() -> None:
    """A game about nothing but "rest" answers no research question."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:, :120] = 7  # largest, but still under the threshold below
    mask[:, 120:180] = 8
    mask[:, 180:] = 9
    players = build(mask, min_area_fraction=0.9)
    assert players.labels == (7, None)
    assert not players.players[0].is_rest


def test_cap_bounds_the_player_count() -> None:
    labels = tuple(range(1, 9))
    players = build(stripes(448, 448, labels), max_players=4)
    assert len(players) == 4
    assert players.capped


def test_cap_puts_the_overflow_into_rest() -> None:
    labels = tuple(range(1, 9))
    players = build(stripes(448, 448, labels), max_players=4)
    rest = players.players[-1]
    assert rest.is_rest
    assert len(rest.merged_labels) == len(labels) - 3


def test_cap_keeps_the_largest_segments_as_themselves() -> None:
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:, :120] = 1  # largest
    mask[:, 120:190] = 2
    mask[:, 190:210] = 3
    mask[:, 210:] = 4
    players = build(mask, max_players=3)
    assert players.labels == (1, 2, None)


def test_capping_is_reported_separately_from_merging() -> None:
    """Which images the cap distorted must be countable after the fact."""
    uncapped = build(stripes(448, 448, (1, 2, 3)), max_players=10)
    assert not uncapped.capped


def test_max_players_of_one_yields_a_single_rest_player() -> None:
    """The cap is explicit, so it overrides keeping a named ingredient."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[:, :150] = 1
    mask[:, 150:] = 2
    players = build(mask, max_players=1)
    assert len(players) == 1
    assert players.players[0].merged_labels == (1, 2)


@pytest.mark.parametrize("max_players", [1, 2, 3, 4, 5, 8])
def test_the_cap_counts_rest_too(max_players: int) -> None:
    """A cap that `rest` can exceed by one is not a bound on 2**n."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    # Five ingredients, two of them specks that min_area_fraction will merge.
    mask[:, :80] = 1
    mask[:, 80:140] = 2
    mask[:, 140:190] = 3
    mask[:, 190:192] = 4
    mask[:, 192:] = 5
    players = build(mask, max_players=max_players, min_area_fraction=0.05)
    assert len(players) <= max_players


@pytest.mark.parametrize(("field", "value"), [("max_players", 0), ("min_area_fraction", 1.5)])
def test_invalid_arguments_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        build(stripes(224, 224, (0, 7)), **{field: value})


# --------------------------------------------------------------------------- #
# Coalitions
# --------------------------------------------------------------------------- #


def test_empty_coalition_shows_no_food() -> None:
    players = build(stripes(448, 448, (0, 7, 8)))
    assert not players.coalition_mask([]).any()


def test_full_coalition_plus_background_is_the_whole_frame() -> None:
    players = build(stripes(448, 448, (0, 7, 8)))
    full = players.coalition_mask(range(len(players)))
    assert (full | players.background).all()


def test_coalition_mask_is_the_union_of_its_players() -> None:
    players = build(stripes(448, 448, (0, 7, 8, 9)))
    expected = players.players[0].mask | players.players[2].mask
    np.testing.assert_array_equal(players.coalition_mask([0, 2]), expected)


def test_coalition_mask_rejects_an_out_of_range_player() -> None:
    """Silently ignoring one would produce a plausible-looking but wrong game."""
    players = build(stripes(448, 448, (0, 7, 8)))
    with pytest.raises(IndexError, match="out of range"):
        players.coalition_mask([0, 5])


def test_coalition_mask_matches_the_image_shape() -> None:
    players = build(stripes(448, 448, (0, 7)))
    assert players.coalition_mask([0]).shape == players.image.shape[:2] == (224, 224)


def test_area_fractions_sum_with_background_to_one() -> None:
    players = build(stripes(448, 448, (0, 7, 8, 9)))
    total = sum(player.area_fraction for player in players.players)
    assert total + float(players.background.mean()) == pytest.approx(1.0)
