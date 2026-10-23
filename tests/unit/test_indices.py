from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
import pytest
import shapiq
from shapiq.utils import powerset
from synthetic_game import brute_force_shapley, build_game

from food_shapley.indices import (
    INTERACTION,
    MOEBIUS,
    SHAPLEY,
    ImageIndices,
    IndexBlock,
    _blocks_of,
    _dense_block,
    _membership_matrix,
    compute_indices,
    load_indices,
    save_indices,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from food_shapley.game import IngredientGame

# --------------------------------------------------------------------------- #
# The fixture game is `v(S) = base + sum(weights in S) + bonus if pair <= S`.
#
# Its Mobius transform is known in closed form -- `weights[i]` on each
# singleton, `bonus` on `pair`, zero everywhere else -- and the weights are
# orders of magnitude apart so that a value landing on the wrong player cannot
# cancel out. `synthetic_game` explains how a coalition is recovered from its
# rendering, which is what makes a mislabelled coalition visible here at all.
# --------------------------------------------------------------------------- #

PAIR = (0, 2)
BONUS = 6.0
WEIGHTS = (1.0, 20.0, 300.0, 4000.0)


def as_dict(block: IndexBlock) -> dict[tuple[int, ...], float]:
    """The block keyed by the players each value belongs to.

    Every assertion below goes through this rather than through a row index,
    for the reason the module exists: the stored order is not the powerset
    order and must never be assumed.
    """
    return {
        tuple(np.flatnonzero(block.members[row]).tolist()): float(block.values[row])
        for row in range(len(block))
    }


# --------------------------------------------------------------------------- #
# What `shapiq` guarantees. Pinned so an upgrade fails here, loudly, rather
# than silently renumbering every stored result.
# --------------------------------------------------------------------------- #


def test_the_k_sii_lookup_is_not_in_powerset_order() -> None:
    """The fact the whole storage format is built around.

    If this ever starts passing by accident -- that is, if `shapiq` starts
    returning k-SII in powerset order -- the membership matrices become
    redundant rather than wrong, and that is worth knowing deliberately
    instead of discovering it from a figure.
    """
    game, _, _ = build_game(num_players=3)
    lookup = shapiq.ExactComputer(game)("k-SII", order=2).interaction_lookup

    assert list(lookup) != list(powerset(range(3), max_size=2))
    assert lookup == {(): 0, (0, 2): 1, (0,): 2, (1,): 3, (2,): 4}


def test_shapiq_drops_exactly_zero_k_sii_entries() -> None:
    """The other reason the blocks are rebuilt, and the nastier of the two.

    With a player that contributes nothing, `shapiq`'s order-2 k-SII result is
    missing most of its pairs *and* that player's own singleton. Anything
    reading those rows directly would be taking a value-dependent sample of
    the interactions and would not be able to tell.
    """
    game, _, _ = build_game(weights=[1.0, 0.0, 300.0, 4000.0], pair=(0, 2), bonus=9.0)

    lookup = shapiq.ExactComputer(game)("k-SII", order=2).interaction_lookup

    assert len(lookup) < len(list(powerset(range(4), max_size=2)))
    assert (1,) not in lookup


def test_a_dropped_entry_is_restored_as_zero() -> None:
    """The fix for the above: a fixed shape, with the omissions filled in."""
    game, _, _ = build_game(weights=[1.0, 0.0, 300.0, 4000.0], pair=(0, 2), bonus=9.0)

    interactions = compute_indices(game, order=2).interactions

    assert len(interactions) == len(list(powerset(range(4), max_size=2)))
    assert interactions.value_of([1]) == pytest.approx(0.0)
    assert interactions.value_of([0, 1]) == pytest.approx(0.0)
    assert interactions.value_of([0, 2]) == pytest.approx(9.0)


def test_every_block_has_the_shape_its_order_implies() -> None:
    """The stored shape must depend on the game, never on the numbers."""
    game, _, _ = build_game()

    result = compute_indices(game, order=2)

    assert len(result.shapley) == 1 + game.n_players
    assert len(result.interactions) == len(list(powerset(range(game.n_players), max_size=2)))
    assert len(result.moebius) == 2**game.n_players


def test_an_entry_above_the_requested_order_is_refused() -> None:
    """Silently dropping it would discard a real value; keeping it breaks the shape."""
    game, _, _ = build_game(num_players=3)
    result = shapiq.ExactComputer(game)("Moebius")

    with pytest.raises(ValueError, match="outside the expected subsets"):
        _dense_block("Moebius", 2, 3, result)


def test_the_exact_computer_evaluates_the_game_once_for_all_three_indices() -> None:
    """Three index families must not cost three powerset evaluations.

    This is the assumption that makes it reasonable to always compute the
    Mobius transform at full order: it is free once the coalitions are scored.
    If `shapiq` ever stopped caching, the full run's cost would triple without
    any other symptom.
    """
    game, value_fn, _ = build_game()
    value_fn.seen.clear()

    compute_indices(game, order=2)

    # One batched pass over the 16 coalitions, at the default chunk size.
    assert sum(value_fn.seen) == 2**game.n_players


# --------------------------------------------------------------------------- #
# The indices themselves, against the closed form.
# --------------------------------------------------------------------------- #


def test_shapley_values_match_the_definition() -> None:
    """`compute_indices` must reproduce the brute-force Shapley values."""
    game, value_fn, _ = build_game()

    result = compute_indices(game, order=2)

    np.testing.assert_allclose(
        result.shapley.main_effects(),
        brute_force_shapley(value_fn, game.n_players),
        rtol=0,
        atol=1e-9,
    )


def test_the_moebius_transform_is_the_closed_form_one() -> None:
    """Singletons carry the weights; the one interacting pair carries the bonus."""
    game, _, _ = build_game(pair=PAIR, bonus=BONUS)

    moebius = as_dict(compute_indices(game, order=2).moebius)

    for player, weight in enumerate(WEIGHTS):
        assert moebius[(player,)] == pytest.approx(weight)
    assert moebius[PAIR] == pytest.approx(BONUS)
    for pair in combinations(range(game.n_players), 2):
        if pair != PAIR:
            assert moebius[pair] == pytest.approx(0.0, abs=1e-9)


def test_the_interaction_lands_on_the_pair_that_interacts() -> None:
    """The headline claim of the project, on a game whose answer is known.

    A column convention that disagreed with `shapiq`'s would put the bonus on
    some other pair, and no numerical check elsewhere would notice -- the
    efficiency residual stays exactly zero either way.
    """
    game, _, _ = build_game(pair=(1, 3), bonus=9.0)

    interactions = as_dict(compute_indices(game, order=2).interactions)

    assert interactions[(1, 3)] == pytest.approx(9.0)
    for pair in combinations(range(game.n_players), 2):
        if pair != (1, 3):
            assert interactions[pair] == pytest.approx(0.0, abs=1e-9)


def test_k_sii_equals_the_moebius_transform_on_a_two_additive_game() -> None:
    """A property of k-SII, not of this implementation.

    The fixture game has no interaction above order 2, so its 2-SII values
    *are* its Mobius coefficients. Asserting it here means the two blocks are
    being read out of `shapiq` consistently: if the k-SII block were paired
    with the wrong membership matrix, the two would disagree even though each
    looks plausible alone.
    """
    game, _, _ = build_game()

    result = compute_indices(game, order=2)

    moebius = as_dict(result.moebius)
    for members, value in as_dict(result.interactions).items():
        assert value == pytest.approx(moebius[members], abs=1e-9)


def test_k_sii_at_order_one_is_the_shapley_value() -> None:
    """The other end of the same property, and a check that `order` is obeyed."""
    game, _, _ = build_game()

    result = compute_indices(game, order=1)

    assert result.interactions.order == 1
    assert max(result.interactions.sizes) == 1
    np.testing.assert_allclose(
        result.interactions.main_effects(),
        result.shapley.main_effects(),
        rtol=0,
        atol=1e-9,
    )


def test_the_k_sii_values_sum_to_the_span() -> None:
    """k-SII's own efficiency, which the Shapley check does not cover."""
    game, _, _ = build_game()

    result = compute_indices(game, order=2)

    assert float(result.interactions.values.sum()) == pytest.approx(result.span, abs=1e-9)


def test_the_moebius_block_is_always_computed_at_full_order() -> None:
    """Storing it truncated would mean re-running CLIP to raise `interaction_order`."""
    game, _, _ = build_game()

    result = compute_indices(game, order=2)

    assert result.moebius.order == game.n_players
    assert len(result.moebius) == 2**game.n_players


# --------------------------------------------------------------------------- #
# The coalition values the indices were derived from.
# --------------------------------------------------------------------------- #


def test_every_stored_coalition_value_matches_its_membership_row() -> None:
    """The stored values must be readable by *what* they are, not by position.

    This is the check that a permuted coalition matrix fails. It compares each
    row's value against the closed-form worth of the players that row says are
    present, so a consistent relabelling is caught even though it satisfies
    efficiency perfectly.
    """
    game, value_fn, _ = build_game()

    result = compute_indices(game, order=2)

    assert len(result.coalitions) == 2**game.n_players
    for row in range(len(result.values)):
        members = np.flatnonzero(result.coalitions[row]).tolist()
        expected = value_fn.worth(members) - value_fn.worth([])
        assert result.values[row] == pytest.approx(expected, abs=1e-9)


def test_stored_values_are_normalised_and_the_baseline_recovers_the_raw_ones() -> None:
    """`baseline` must be `v(empty)`, so a raw log-probability is recoverable."""
    game, value_fn, _ = build_game()

    result = compute_indices(game, order=2)

    # Guards against a zero baseline, which would make "normalised" and "raw"
    # indistinguishable.
    assert value_fn.worth([]) != 0.0
    assert result.baseline == pytest.approx(value_fn.worth([]))
    assert result.values[~result.coalitions.any(axis=1)][0] == pytest.approx(0.0)
    assert result.span == pytest.approx(value_fn.worth(range(game.n_players)) - value_fn.worth([]))


def test_the_image_metadata_travels_with_the_numbers() -> None:
    """Player names, the scene/ingredient split and the target identify the result."""
    game, _, players = build_game()

    result = compute_indices(game, order=2)

    assert result.image_index == players.index
    assert result.player_names == players.names
    assert result.ingredient_indices == players.ingredient_indices
    # The fixture's player 0 carries label 0, so it is not an ingredient. An
    # empty or complete tuple here would make the assertion vacuous.
    assert 0 < len(result.ingredient_indices) < game.n_players
    assert result.target is game.target
    assert result.chunk_size == game.chunk_size


# --------------------------------------------------------------------------- #
# The efficiency check.
# --------------------------------------------------------------------------- #


def test_the_efficiency_residual_is_recorded_not_just_asserted() -> None:
    """The full run reports a distribution, so the number must be kept."""
    game, _, _ = build_game()

    result = compute_indices(game, order=2)

    assert abs(result.efficiency_residual) < 1e-12
    assert result.shapley.main_effects().sum() - result.span == pytest.approx(
        result.efficiency_residual
    )


class Perturbed(NamedTuple):
    """The two attributes `indices` reads off a `shapiq` index result."""

    values: NDArray[np.float64]
    interaction_lookup: dict[tuple[int, ...], int]


class BrokenComputer:
    """A real `ExactComputer` whose Shapley values are off by a known amount.

    Breaking the check from outside is the only way to see it fire: the real
    computer is efficient by construction, so a correct pipeline can never
    produce a violation on demand. Without this the `raise` is untested and
    could be inverted, mistyped, or comparing against the wrong quantity.

    The perturbed result is a stand-in rather than a mutated real one.
    `InteractionValues.values` hands back a fresh array on every access, so
    writing to it changes nothing -- which would make this fixture silently
    test the unbroken path.
    """

    error = 0.5
    #: Which Shapley entry to nudge. A singleton makes the decomposition
    #: genuinely wrong; the empty set does not (see `ShiftedBaseline`).
    moved: tuple[int, ...] = (0,)
    # Captured before the monkeypatch, or wrapping the real computer would
    # wrap this class instead and recurse.
    real = shapiq.ExactComputer

    def __init__(self, game: IngredientGame) -> None:
        self._inner = BrokenComputer.real(game)

    @property
    def game_values(self) -> NDArray[np.float64]:
        """The real coalition values, untouched."""
        values: NDArray[np.float64] = self._inner.game_values
        return values

    @property
    def coalition_lookup(self) -> dict[tuple[int, ...], int]:
        """The real coalition lookup, untouched."""
        lookup: dict[tuple[int, ...], int] = self._inner.coalition_lookup
        return lookup

    def __call__(self, index: str, order: int | None = None) -> Any:  # noqa: ANN401
        """Forward to the real computer, nudging one Shapley value off."""
        # `shapiq` types this argument as a literal of its index names; the
        # stub takes a plain `str` because it is handed one by production code
        # that has already narrowed it.
        name = cast("Any", index)
        result = self._inner(name) if order is None else self._inner(name, order=order)
        if index != "SV":
            return result
        values = np.array(result.values, dtype=np.float64)
        values[result.interaction_lookup[self.moved]] += self.error
        return Perturbed(values=values, interaction_lookup=result.interaction_lookup)


class ShiftedBaseline(BrokenComputer):
    """The same nudge, applied to the Shapley result's empty-set entry."""

    moved = ()


class PartialComputer(BrokenComputer):
    """A computer that scored only half the powerset.

    `shapiq` has a `_precompute_flag` that can be set with coalitions still
    missing, so "it was precomputed" does not mean "all 2^n were evaluated".
    An exact computation on a partial store would produce indices that look
    entirely normal.
    """

    @property
    def coalition_lookup(self) -> dict[tuple[int, ...], int]:
        """Half the coalitions, as a partially populated store would give."""
        full = super().coalition_lookup
        return {members: row for members, row in full.items() if row < len(full) // 2}


@pytest.fixture
def partial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shapiq, "ExactComputer", PartialComputer)


@pytest.mark.usefixtures("partial")
def test_an_incomplete_powerset_is_refused() -> None:
    """Exactness is the whole claim; a partial evaluation is not an approximation."""
    game, _, _ = build_game()

    with pytest.raises(ValueError, match="evaluated 8 coalitions, not the 16"):
        compute_indices(game, order=2)


@pytest.fixture
def broken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shapiq, "ExactComputer", BrokenComputer)


@pytest.fixture
def shifted_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shapiq, "ExactComputer", ShiftedBaseline)


@pytest.mark.usefixtures("shifted_baseline")
def test_the_efficiency_check_ignores_the_shapley_empty_set_entry() -> None:
    """Efficiency is a statement about the *players*, not about every row.

    `shapiq`'s SV result carries an empty-set entry, which is zero only
    because the game is normalised. Summing the whole array therefore gives
    the same answer today and would silently stop being an efficiency check
    the moment that stopped holding -- a mutation replacing `main_effects()`
    with `values.sum()` passes the rest of this file untouched.

    So the entry is moved off zero here. A correct check does not notice;
    the lazy one reports a violation that is not there.
    """
    game, _, _ = build_game()

    result = compute_indices(game, order=2)

    assert abs(result.efficiency_residual) < 1e-12
    # Guards against the shift not arriving: if it were not applied, the
    # mutation this test exists to kill would survive it too.
    assert result.shapley.value_of([]) == pytest.approx(ShiftedBaseline.error)


@pytest.mark.usefixtures("broken")
def test_a_violated_efficiency_check_stops_the_image() -> None:
    """A wrong decomposition must raise, not be written to disk with a warning."""
    game, _, _ = build_game()

    with pytest.raises(ValueError, match="efficiency check failed for image 0"):
        compute_indices(game, order=2)


@pytest.mark.usefixtures("broken")
def test_the_efficiency_failure_reports_the_residual_and_the_span() -> None:
    """The message has to say how wrong, or it cannot be triaged from a log."""
    game, _, _ = build_game()

    with pytest.raises(ValueError, match=r"\+5\.000e-01 more than v\(N\) - v\(empty\)"):
        compute_indices(game, order=2)


@pytest.mark.usefixtures("broken")
def test_a_tolerance_wide_enough_lets_the_violation_through() -> None:
    """Guards the two tests above from passing for some unrelated reason.

    If the error were not reaching the comparison, widening the tolerance
    past it would change nothing.
    """
    game, _, _ = build_game()

    result = compute_indices(game, order=2, tolerance=1.0)

    assert result.efficiency_residual == pytest.approx(BrokenComputer.error)


# --------------------------------------------------------------------------- #
# Order handling.
# --------------------------------------------------------------------------- #


def test_an_order_above_the_player_count_is_clamped() -> None:
    """`shapiq` raises an opaque `IndexError` instead of clamping itself.

    Two-player images exist in the split (W01-09), so an `interaction_order`
    of 3 must not crash the run on them.
    """
    game, _, _ = build_game(num_players=2, weights=[3.0, 11.0], pair=(0, 1), bonus=5.0)

    result = compute_indices(game, order=3)

    assert result.interactions.order == 2
    assert as_dict(result.interactions)[(0, 1)] == pytest.approx(5.0)


def test_shapiq_really_does_raise_on_an_order_above_the_player_count() -> None:
    """The reason the clamp exists; without it this is what the run would hit."""
    game, _, _ = build_game(num_players=2, weights=[3.0, 11.0], pair=(0, 1), bonus=5.0)

    with pytest.raises(IndexError):
        shapiq.ExactComputer(game)("k-SII", order=3)


@pytest.mark.parametrize("order", [0, -1])
def test_a_non_positive_order_is_refused(order: int) -> None:
    game, _, _ = build_game()

    with pytest.raises(ValueError, match="interaction order must be >= 1"):
        compute_indices(game, order=order)


# --------------------------------------------------------------------------- #
# `IndexBlock` itself.
# --------------------------------------------------------------------------- #


def test_main_effects_are_found_by_membership_not_by_row() -> None:
    """Row `j + 1` is player `j` for SV and is not for k-SII."""
    block = IndexBlock(
        index="k-SII",
        order=2,
        members=np.array([[False, False], [True, True], [False, True], [True, False]]),
        values=np.array([0.0, 7.0, 2.0, 1.0]),
    )

    np.testing.assert_array_equal(block.main_effects(), [1.0, 2.0])


def test_a_block_missing_a_main_effect_is_refused() -> None:
    block = IndexBlock(
        index="k-SII",
        order=2,
        members=np.array([[True, False], [True, True]]),
        values=np.array([1.0, 7.0]),
    )

    with pytest.raises(ValueError, match=r"no main effect for players \[1\]"):
        block.main_effects()


def test_value_of_identifies_an_interaction_by_its_members() -> None:
    game, _, _ = build_game(pair=(1, 3), bonus=9.0)

    interactions = compute_indices(game, order=2).interactions

    assert interactions.value_of([3, 1]) == pytest.approx(9.0)
    assert interactions.value_of([1, 3, 3]) == pytest.approx(9.0)


def test_value_of_refuses_an_interaction_the_block_does_not_carry() -> None:
    """At order 2 a triple has no entry, and returning 0.0 would be a lie."""
    game, _, _ = build_game()

    interactions = compute_indices(game, order=2).interactions

    with pytest.raises(KeyError, match=r"no entry for players \(0, 1, 2\)"):
        interactions.value_of([0, 1, 2])


@pytest.mark.parametrize(
    ("members", "values", "message"),
    [
        (np.zeros((3, 2), dtype=np.bool_), np.zeros(2), "3 membership rows for 2 values"),
        (np.zeros(2, dtype=np.bool_), np.zeros(2), "members must be 2-D"),
        (np.zeros((2, 2), dtype=np.bool_), np.zeros((2, 1)), "values must be 1-D"),
    ],
)
def test_a_block_whose_arrays_disagree_is_refused(
    members: NDArray[np.bool_], values: NDArray[np.float64], message: str
) -> None:
    """Both arrays are plausible alone, so nothing downstream could notice."""
    with pytest.raises(ValueError, match=message):
        IndexBlock(index="SV", order=1, members=members, values=values)


# --------------------------------------------------------------------------- #
# `_membership_matrix`: the translation every stored array depends on.
# --------------------------------------------------------------------------- #


def test_a_lookup_is_turned_into_the_rows_it_names() -> None:
    rows = _membership_matrix({(): 0, (0, 2): 1, (1,): 2}, 3)

    np.testing.assert_array_equal(
        rows,
        [[False, False, False], [True, False, True], [False, True, False]],
    )


def test_a_lookup_pointing_past_its_own_length_is_refused() -> None:
    """An unfilled row is all-`False`, which reads as the empty interaction."""
    with pytest.raises(ValueError, match=r"maps to row 2, outside 0\.\.1"):
        _membership_matrix({(): 0, (0,): 2}, 2)


def test_a_lookup_with_a_duplicated_row_is_refused() -> None:
    with pytest.raises(ValueError, match="claimed by more than one interaction"):
        _membership_matrix({(0,): 0, (1,): 0}, 2)


def test_a_lookup_naming_an_unknown_player_is_refused() -> None:
    """A lookup from another image would otherwise silently drop that player."""
    with pytest.raises(ValueError, match="names player 5 in a 2-player game"):
        _membership_matrix({(): 0, (5,): 1}, 2)


# --------------------------------------------------------------------------- #
# Round-trip to disk.
# --------------------------------------------------------------------------- #


def assert_same(left: ImageIndices, right: ImageIndices) -> None:
    """Every field a later analysis reads must survive the round trip."""
    assert left.image_index == right.image_index
    assert left.player_names == right.player_names
    assert left.ingredient_indices == right.ingredient_indices
    assert left.target == right.target
    assert left.baseline == right.baseline
    assert left.efficiency_residual == right.efficiency_residual
    assert left.chunk_size == right.chunk_size
    np.testing.assert_array_equal(left.coalitions, right.coalitions)
    np.testing.assert_array_equal(left.values, right.values)

    reloaded = _blocks_of(left)
    original = _blocks_of(right)
    assert set(reloaded) == {SHAPLEY, INTERACTION, MOEBIUS}
    for name, block in reloaded.items():
        assert block.index == original[name].index
        assert block.order == original[name].order
        np.testing.assert_array_equal(block.members, original[name].members)
        np.testing.assert_array_equal(block.values, original[name].values)


def test_a_saved_result_reloads_identically(tmp_path: Path) -> None:
    game, _, _ = build_game()
    result = compute_indices(game, order=2)
    path = tmp_path / "image-00000.npz"

    save_indices(path, result)

    assert_same(load_indices(path), result)


def test_the_round_trip_preserves_values_bit_for_bit(tmp_path: Path) -> None:
    """`assert_array_equal`, not `allclose`: storage must not quantise."""
    game, _, _ = build_game()
    result = compute_indices(game, order=2)
    path = tmp_path / "image-00000.npz"
    save_indices(path, result)

    reloaded = load_indices(path)

    assert reloaded.values.dtype == np.float64
    np.testing.assert_array_equal(reloaded.shapley.values, result.shapley.values)


def test_a_file_whose_arrays_disagree_about_the_player_count_is_refused(
    tmp_path: Path,
) -> None:
    """Two images' arrays in one file would load cleanly and be wrong."""
    game, _, _ = build_game()
    path = tmp_path / "image-00000.npz"
    save_indices(path, compute_indices(game, order=2))

    with np.load(path) as handle:
        arrays = dict(handle)
    arrays["coalitions"] = arrays["coalitions"][:, :-1]
    np.savez_compressed(path, allow_pickle=False, **arrays)

    with pytest.raises(ValueError, match="disagree about the player count"):
        load_indices(path)


def test_the_saved_file_needs_no_pickle_to_read(tmp_path: Path) -> None:
    """A pickled artifact is neither safe to load nor stable across versions."""
    game, _, _ = build_game()
    path = tmp_path / "image-00000.npz"
    save_indices(path, compute_indices(game, order=2))

    with np.load(path, allow_pickle=False) as handle:
        assert set(handle) >= {"coalitions", "values", "meta", "SV_values"}
