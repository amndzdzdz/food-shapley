from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pytest
import shapiq
from shapiq.utils import powerset
from synthetic_game import (
    SIZE,
    TARGET,
    DecodingValueFunction,
    as_value_function,
    brute_force_shapley,
    build_game,
    coalition_rows,
    make_players,
)

from food_shapley.game import DEFAULT_CHUNK_SIZE, IngredientGame
from food_shapley.masking import BlurRemoval
from food_shapley.segments import PlayerSet

if TYPE_CHECKING:
    from food_shapley.config import ExperimentConfig


# --------------------------------------------------------------------------- #
# Coalition ordering: the bug this module is built to prevent.
# --------------------------------------------------------------------------- #


def test_shapiq_enumerates_by_size_then_lexicographically() -> None:
    """Pin `shapiq`'s enumeration order, which the stored results inherit.

    Not a test of our code. It is here so that a `shapiq` upgrade that changed
    the convention -- silently renumbering every stored value vector -- fails
    loudly in this suite instead of in a figure six weeks later.
    """
    assert list(powerset(range(3))) == [
        (),
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]


def test_exact_computer_reproduces_the_closed_form_shapley_values() -> None:
    """The end-to-end ordering check: known game in, known values out.

    `shapiq` sees only the game; the expected numbers come from the definition
    of the Shapley value applied to the closed form. Any disagreement between
    our column convention and `shapiq`'s moves weight onto the wrong player,
    and the weights are orders of magnitude apart so that it cannot cancel.
    """
    game, value_fn, _ = build_game()
    computed = shapiq.ExactComputer(game)("SV")

    expected = brute_force_shapley(value_fn, game.n_players)
    actual = [computed[(index,)] for index in range(game.n_players)]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)


def test_the_interaction_lands_on_the_pair_that_actually_interacts() -> None:
    """A permuted column convention would attribute the bonus to another pair."""
    game, value_fn, _ = build_game(pair=(1, 3), bonus=9.0)
    moebius = shapiq.ExactComputer(game).moebius_transform

    assert moebius[(1, 3)] == pytest.approx(9.0)
    for pair in combinations(range(game.n_players), 2):
        if pair != (1, 3):
            assert moebius[pair] == pytest.approx(0.0, abs=1e-9)
    for index in range(game.n_players):
        assert moebius[(index,)] == pytest.approx(value_fn.weights[index])


def test_efficiency_holds_on_the_synthetic_game() -> None:
    """`sum(SV) == v(N) - v(empty)`, the check `PLAN.md` requires."""
    game, _, _ = build_game()
    computed = shapiq.ExactComputer(game)("SV")

    span = float(game(np.ones((1, game.n_players), dtype=np.bool_))[0])
    total = sum(computed[(index,)] for index in range(game.n_players))
    assert total == pytest.approx(span, abs=1e-9)


def test_values_follow_row_order_not_coalition_size() -> None:
    """Row i out scores row i in, even when the rows are not in powerset order."""
    game, value_fn, _ = build_game()
    order = [[3, 1], [], [0, 1, 2, 3], [2], [0, 2], [1]]

    values = game.value_function(coalition_rows(order, game.n_players))

    assert list(values) == [value_fn.worth(group) for group in order]


def test_repeated_rows_are_scored_repeatedly_and_identically() -> None:
    """No deduplication: a repeated row must come back at every position."""
    game, value_fn, _ = build_game()
    order = [[0, 2], [1], [0, 2], [0, 2]]

    values = game.value_function(coalition_rows(order, game.n_players))

    assert values[0] == values[2] == values[3] == pytest.approx(value_fn.worth([0, 2]))
    assert values[1] == pytest.approx(value_fn.worth([1]))


# --------------------------------------------------------------------------- #
# Chunking must be invisible.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 1000])
def test_chunk_size_changes_nothing_about_the_values(chunk_size: int) -> None:
    """A throughput knob that changed a value would change the science."""
    reference, _, _ = build_game(chunk_size=DEFAULT_CHUNK_SIZE)
    chunked, _, _ = build_game(chunk_size=chunk_size)

    rows = coalition_rows(powerset(range(reference.n_players)), reference.n_players)

    np.testing.assert_array_equal(
        reference.value_function(rows),
        chunked.value_function(rows),
    )


def test_chunking_actually_splits_the_work() -> None:
    """Guard the test above from passing because chunking never happens."""
    game, value_fn, _ = build_game(chunk_size=3)
    value_fn.seen.clear()

    game.value_function(coalition_rows(powerset(range(game.n_players)), game.n_players))

    assert value_fn.seen == [3, 3, 3, 3, 3, 1]


def test_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        build_game(chunk_size=0)


# --------------------------------------------------------------------------- #
# Normalisation: what `v(empty)` and `v(N)` mean to a caller.
# --------------------------------------------------------------------------- #


def test_the_empty_coalition_is_worth_zero_after_normalisation() -> None:
    game, _, _ = build_game()

    assert game(np.zeros((1, game.n_players), dtype=np.bool_))[0] == pytest.approx(0.0)


def test_the_normalisation_value_is_the_raw_empty_coalition_value() -> None:
    """The baseline must be `v(empty)`, not zero and not `v(N)`."""
    game, value_fn, _ = build_game()

    # Guards the assertion below from being satisfied by a zero baseline, which
    # would make "subtract v(empty)" and "subtract nothing" indistinguishable.
    assert value_fn.worth([]) != 0.0
    assert game.normalization_value == pytest.approx(value_fn.worth([]))


def test_value_function_returns_raw_values_so_shapiq_can_subtract_once() -> None:
    """Normalising inside `value_function` too would subtract the baseline twice."""
    game, value_fn, _ = build_game()
    rows = coalition_rows([[0, 1]], game.n_players)

    assert game.value_function(rows)[0] == pytest.approx(value_fn.worth([0, 1]))
    assert game(rows)[0] == pytest.approx(value_fn.worth([0, 1]) - value_fn.worth([]))


def test_constructing_the_game_costs_exactly_one_extra_evaluation() -> None:
    """The baseline is computed eagerly, once, not per call."""
    players = make_players(3)
    removal = BlurRemoval(radius=2.0, keep_background=False)
    value_fn = DecodingValueFunction(players, removal, [1.0, 2.0, 3.0], (0, 1), 1.0)

    IngredientGame(players, as_value_function(value_fn), removal, TARGET)

    assert value_fn.seen == [1]


# --------------------------------------------------------------------------- #
# Input validation. A wrong-shaped matrix must never be silently reinterpreted.
# --------------------------------------------------------------------------- #


def test_a_coalition_matrix_for_another_image_is_rejected() -> None:
    game, _, _ = build_game(num_players=4)

    with pytest.raises(ValueError, match="coalition matrix is for another image"):
        game.value_function(np.ones((2, 3), dtype=np.bool_))


def test_a_one_dimensional_coalition_array_is_rejected() -> None:
    """A single coalition passed unwrapped would score as `n_players` rows."""
    game, _, _ = build_game()

    with pytest.raises(ValueError, match="coalitions must be 2-D"):
        game.value_function(np.ones(game.n_players, dtype=np.bool_))


def test_no_coalitions_yields_no_values() -> None:
    game, _, _ = build_game()

    assert game.value_function(np.zeros((0, game.n_players), dtype=np.bool_)).shape == (0,)


def test_an_image_without_players_has_no_game() -> None:
    players = PlayerSet(
        index=11,
        image=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
        players=(),
        background=np.ones((SIZE, SIZE), dtype=np.bool_),
        dropped_labels=(),
        capped=False,
    )
    removal = BlurRemoval(radius=2.0, keep_background=False)
    value_fn = DecodingValueFunction(players, removal, [], (0, 1), 0.0)

    with pytest.raises(ValueError, match="image 11 has no players"):
        IngredientGame(players, as_value_function(value_fn), removal, TARGET)


# --------------------------------------------------------------------------- #
# The rendering the game scores is the rendering the operator produces.
# --------------------------------------------------------------------------- #


def test_the_game_scores_the_operator_s_own_renderings() -> None:
    """Reusing one blurred frame must not change a single pixel.

    `IngredientGame` blurs once per image instead of once per coalition. This
    asserts the shortcut is exact, because the decoding value function would
    otherwise raise `KeyError` rather than return a wrong number -- and a
    `KeyError` would be easy to misread as a test-harness problem.
    """
    game, _, players = build_game()
    removal = game.removal

    for coalition in powerset(range(len(players))):
        expected = removal.render(players, coalition)
        rendered = removal.render(players, coalition, blurred=removal.blurred(players))
        np.testing.assert_array_equal(rendered, expected)


def test_two_players_minimum_still_gives_a_usable_game() -> None:
    """Seven test images have exactly two players (W01-09); they must work."""
    game, value_fn, _ = build_game(num_players=2, weights=[3.0, 11.0], pair=(0, 1), bonus=5.0)
    computed = shapiq.ExactComputer(game)("SV")

    expected = brute_force_shapley(value_fn, 2)
    actual = [computed[(index,)] for index in range(2)]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------- #
# `for_image`: the configuration decides the operator, not the caller.
# --------------------------------------------------------------------------- #


def test_for_image_builds_the_operator_the_config_describes(config: ExperimentConfig) -> None:
    players = make_players(3)
    removal = BlurRemoval(radius=config.blur_radius, keep_background=False)
    value_fn = DecodingValueFunction(players, removal, [1.0, 2.0, 3.0], (0, 1), 1.0)
    value_fn.choose_target = lambda _image: TARGET  # type: ignore[attr-defined]

    game = IngredientGame.for_image(players, value_fn, config)  # type: ignore[arg-type]

    assert game.removal.radius == config.blur_radius
    assert game.removal.keep_background is False
    assert game.target is TARGET


def test_for_image_keeps_the_background_sharp_only_in_visible_mode(
    config: ExperimentConfig,
) -> None:
    visible = replace_mode(config, "visible")
    players = make_players(3)
    removal = BlurRemoval(radius=visible.blur_radius, keep_background=True)
    value_fn = DecodingValueFunction(players, removal, [1.0, 2.0, 3.0], (0, 1), 1.0)
    value_fn.choose_target = lambda _image: TARGET  # type: ignore[attr-defined]

    game = IngredientGame.for_image(players, value_fn, visible)  # type: ignore[arg-type]

    assert game.removal.keep_background is True


def test_an_unimplemented_removal_operator_is_refused(config: ExperimentConfig) -> None:
    """Week 2's operator must fail loudly here rather than silently blur."""
    from dataclasses import replace

    players = make_players(3)
    removal = BlurRemoval(radius=2.0, keep_background=False)
    value_fn = DecodingValueFunction(players, removal, [1.0, 2.0, 3.0], (0, 1), 1.0)

    with pytest.raises(ValueError, match="'token-drop' is not implemented"):
        IngredientGame.for_image(
            players,
            value_fn,  # type: ignore[arg-type]
            replace(config, removal="token-drop"),
        )


def replace_mode(config: ExperimentConfig, mode: str) -> ExperimentConfig:
    """A copy of `config` with a different background mode."""
    from dataclasses import replace

    return replace(config, background_mode=mode)
