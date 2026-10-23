"""The coalition game against the real checkpoint, on real FoodSeg103 images.

Marked ``gpu`` and ``slow`` for the same reason as
``test_value_on_real_images.py``: these load a ~600 MB CLIP checkpoint, which
`CLAUDE.md` keeps out of the ordinary suite.

What only these can establish is that the three halves line up on real data:

* the grand coalition's rendering really is the photograph, so ``v(N)``
  reproduces the target's own log-probability **bit for bit** -- the sharpest
  available check that masking, geometry and scoring agree (W01-09);
* exact Shapley values over a real game satisfy efficiency, which is
  `PLAN.md`'s first validity check and the remaining blocker on task 1.6;
* coalition values are finite and the game is not degenerate;
* batching is a throughput knob to within float32 noise, which is a property
  of CLIP's numerics that no stub can establish, and which is *measured* here
  rather than assumed (W01-11).

Run them deliberately:

    uv run pytest tests/integration/test_game_on_real_images.py -m gpu
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import shapiq
from shapiq.utils import powerset

from food_shapley.config import ExperimentConfig, load_config
from food_shapley.dataset import load_split
from food_shapley.download import read_id2label, verify_split
from food_shapley.game import IngredientGame
from food_shapley.prompts import verify_dish_names
from food_shapley.segments import build_players
from food_shapley.value import DishValueFunction

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from food_shapley.dataset import FoodSegSplit
    from food_shapley.segments import PlayerSet

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]

CONFIG = Path("configs/week01-clip-blur.toml")

#: Images with few enough players that the whole powerset is cheap on CPU.
#: Chosen by player count rather than at random: these tests are about
#: correctness properties that hold for every image, so the sample only needs
#: to be real, not representative.
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
def game(
    players: Callable[[int], PlayerSet],
    value_function: DishValueFunction,
    settings: ExperimentConfig,
) -> Callable[[int], IngredientGame]:
    cache: dict[int, IngredientGame] = {}

    def build(index: int) -> IngredientGame:
        if index not in cache:
            cache[index] = IngredientGame.for_image(players(index), value_function, settings)
        return cache[index]

    return build


def _powerset_rows(n_players: int) -> NDArray[np.bool_]:
    """Every coalition, in `shapiq`'s own order: empty first, grand last."""
    rows = np.zeros((2**n_players, n_players), dtype=np.bool_)
    for position, coalition in enumerate(powerset(range(n_players))):
        rows[position, list(coalition)] = True
    return rows


# --------------------------------------------------------------------------- #
# The grand coalition is the photograph
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index", SAMPLE)
def test_the_grand_coalition_reproduces_the_target_s_own_log_probability(
    game: Callable[[int], IngredientGame], index: int
) -> None:
    """`v(N)` must equal the score of the unmasked image, exactly.

    The target is chosen by scoring `players.image`; `v(N)` is produced by
    scoring the grand coalition's *rendering*. Under
    `background_mode = "player"` those are the same array, so the two numbers
    must agree bit for bit -- not approximately. Any difference means some
    pixel belongs to no player, and the whole decomposition would be
    explaining an image the model never saw.
    """
    one = game(index)
    grand = np.ones((1, one.n_players), dtype=np.bool_)

    assert one.value_function(grand)[0] == one.target.log_prob


@pytest.mark.parametrize("index", SAMPLE)
def test_hiding_everything_costs_the_model_evidence(
    game: Callable[[int], IngredientGame], index: int
) -> None:
    """`v(N) > v(empty)`: the operator removes something the model was using.

    A game where it does not is degenerate -- every Shapley value would be
    noise around zero. W01-10 measured this as positive on all 200 sampled
    images; this holds the property in place.
    """
    one = game(index)

    assert one.normalization_value < one.target.log_prob
    assert one(np.ones((1, one.n_players), dtype=np.bool_))[0] > 0.0


# --------------------------------------------------------------------------- #
# `PLAN.md`'s efficiency check, on a real game
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index", SAMPLE)
def test_exact_shapley_values_satisfy_efficiency(
    game: Callable[[int], IngredientGame], index: int
) -> None:
    """`sum(SV) == v(N) - v(empty)` exactly, the check `PLAN.md` gates 1.6 on.

    Two details decide whether this test means anything.

    Summed over the *player* entries, deliberately: `shapiq`'s result also
    carries an empty-set entry, which is zero only because the game is
    normalised. Writing `values.sum()` would pass here and quietly stop being
    an efficiency check if the normalisation ever changed.

    And the span is read off a powerset evaluation identical to the one the
    computer made -- same rows, same order, so bit-identical by the
    determinism asserted below -- not off a fresh one-row call. Scoring the
    grand coalition again on its own returns a number that differs in the last
    bits, because CLIP's forward pass is not batch-invariant (W01-11). A test
    comparing across batch shapes would measure that noise rather than
    efficiency, and would need a tolerance loose enough to stop checking
    anything. Evaluated consistently the identity is exact, so it is asserted
    as equality.
    """
    one = game(index)
    rows = _powerset_rows(one.n_players)
    values = one.value_function(rows)

    computed = shapiq.ExactComputer(one)("SV")
    total = sum(computed[(player,)] for player in range(one.n_players))

    assert total == float(values[-1] - values[0])


@pytest.mark.parametrize("index", SAMPLE)
def test_every_coalition_value_is_finite_and_a_log_probability(
    game: Callable[[int], IngredientGame], index: int
) -> None:
    """Raw values are log-probabilities, so finite and non-positive."""
    one = game(index)
    rows = _powerset_rows(one.n_players)

    values = one.value_function(rows)

    assert np.isfinite(values).all()
    assert (values <= 0.0).all()
    assert values.shape == (2**one.n_players,)


@pytest.mark.parametrize("index", SAMPLE)
def test_the_game_is_deterministic(game: Callable[[int], IngredientGame], index: int) -> None:
    """Re-scoring the same coalitions must be bit-identical, not merely close.

    Exact Shapley values are differences of these numbers; drift here would
    show up as spurious interaction structure.
    """
    one = game(index)
    rows = np.eye(one.n_players, dtype=np.bool_)

    np.testing.assert_array_equal(one.value_function(rows), one.value_function(rows))


@pytest.mark.parametrize("index", SAMPLE)
def test_chunking_moves_a_real_value_only_in_the_last_bits(
    players: Callable[[int], PlayerSet],
    value_function: DishValueFunction,
    settings: ExperimentConfig,
    game: Callable[[int], IngredientGame],
    index: int,
) -> None:
    """Chunk size is a throughput knob, to within float32 noise -- *measured*.

    `tests/unit/test_game.py` asserts exact chunk-invariance against a stub,
    which pins our indexing. It cannot pin CLIP's numerics, and those are not
    invariant: batching changes the tensor a forward pass sees, and the same
    coalition scored in a batch of 1 and a batch of 32 differs in the last few
    bits. This test asserted equality at `atol=1e-6` first, and failed; the
    honest fix was to measure the drift, not to widen the tolerance until it
    passed (W01-11).

    The bound here is ~10x the largest drift observed (8.2e-06 nats over the
    full powerset of all three sample images) so that it is not flaky, and
    still four orders of magnitude below the ~4-nat quantity being decomposed.
    If it ever fires, something changed that is not float32 rounding.
    """
    reference = game(index)
    one_at_a_time = IngredientGame.for_image(players(index), value_function, settings, chunk_size=1)
    rows = _powerset_rows(reference.n_players)

    np.testing.assert_allclose(
        one_at_a_time.value_function(rows),
        reference.value_function(rows),
        rtol=0,
        atol=1e-4,
    )


@pytest.mark.parametrize("index", SAMPLE)
def test_chunking_does_not_move_the_shapley_values_that_matter(
    players: Callable[[int], PlayerSet],
    value_function: DishValueFunction,
    settings: ExperimentConfig,
    game: Callable[[int], IngredientGame],
    index: int,
) -> None:
    """The consequence of the test above, on the numbers the project reports.

    Coalition-level drift is only tolerable if it does not survive into the
    attributions, so this measures it there: the exact Shapley values of the
    same image computed at two chunk sizes must agree far more tightly than
    any difference between ingredients. Observed drift is <= 2.9e-06, about
    1e-06 of the span it is divided among.

    This is the evidence for keeping `chunk_size` out of
    `ExperimentConfig` -- and therefore out of the run fingerprint -- rather
    than an assumption that it is inert.
    """
    reference = game(index)
    one_at_a_time = IngredientGame.for_image(players(index), value_function, settings, chunk_size=1)
    span = float(reference(np.ones((1, reference.n_players), dtype=np.bool_))[0])

    computed = [
        np.array([shapiq.ExactComputer(one)("SV")[(player,)] for player in range(one.n_players)])
        for one in (reference, one_at_a_time)
    ]

    drift = float(np.abs(computed[0] - computed[1]).max())
    assert drift < 1e-4
    assert drift / abs(span) < 1e-4


def test_the_scene_is_a_player_but_never_an_ingredient(
    game: Callable[[int], IngredientGame],
) -> None:
    """Every ingredient-level statistic must filter on `ingredient_indices`.

    Asserted here because the game is where the two sets first meet: columns
    index *players*, and reading a per-ingredient number straight off them
    would count the plate and the table as food (W01-09).
    """
    one = game(SAMPLE[0])
    members = one.players

    assert members.scene_index is not None
    assert members.scene_index not in members.ingredient_indices
    assert len(members.ingredient_indices) == one.n_players - 1
