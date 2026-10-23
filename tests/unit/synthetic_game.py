"""A coalition game with a closed-form Shapley decomposition.

Shared by `test_game.py`, which uses it to check that the game labels its
coalitions correctly, and `test_indices.py`, which uses it to check that the
indices derived from those coalitions are the right ones. Both need the same
property and it is awkward to get right, so it lives in one place: a game whose
value is a *known* function of the coalition, recovered from the rendering
rather than from the row index.

That indirection is the whole point. The failure this guards against is a
coalition the pipeline mislabels -- row `i` scored, stored or attributed as
some other subset. It cannot be caught by looking at numbers, because a
consistently permuted game satisfies efficiency exactly. It can be caught by
making the value depend on a known function of the coalition and comparing
against the closed form.

The link from a rendering back to its coalition is made the only way that
proves the game rendered what it claimed: by rendering every coalition
independently, here in the test, and matching bytes. A game that rendered row
`i` as the wrong subset produces a lookup hit for the wrong subset, and the
closed-form assertions then fail.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import TYPE_CHECKING, cast

import numpy as np
from shapiq.utils import powerset

from food_shapley.game import DEFAULT_CHUNK_SIZE, IngredientGame
from food_shapley.masking import BlurRemoval
from food_shapley.segments import Player, PlayerSet
from food_shapley.value import DishTarget

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

    from food_shapley.value import DishValueFunction

SIZE = 12

TARGET = DishTarget(index=0, name="pizza", prompt="a photo of pizza.", log_prob=-0.5)


def make_players(num: int, *, seed: int = 7) -> PlayerSet:
    """A player set of `num` disjoint stripe players covering the whole frame.

    The image is high-contrast noise so that blurring changes every pixel and
    no two coalitions can render identically.
    """
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    owner = np.arange(SIZE) % num
    players = tuple(
        Player(
            name=f"player-{index}",
            label=index,
            mask=np.repeat((owner == index)[:, None], SIZE, axis=1),
        )
        for index in range(num)
    )
    return PlayerSet(
        index=0,
        image=image,
        players=players,
        background=np.zeros((SIZE, SIZE), dtype=np.bool_),
        dropped_labels=(),
        capped=False,
    )


class DecodingValueFunction:
    """Scores a rendering by a closed-form function of the coalition it shows.

    Attributes:
        weights: Per-player additive weight.
        pair: The one interacting pair.
        bonus: Value added when both members of `pair` are present.
        base: Value of the empty coalition. Non-zero on purpose: the real
            `v(empty)` is the log-probability of the target on a fully blurred
            frame, which is strongly negative (median -4.9 nats, W01-10). A
            fixture with `v(empty) == 0` would make every normalisation bug
            invisible, because subtracting the baseline and not subtracting it
            give the same answer.
    """

    def __init__(
        self,
        players: PlayerSet,
        removal: BlurRemoval,
        weights: Sequence[float],
        pair: tuple[int, int],
        bonus: float,
        base: float = -5.25,
    ) -> None:
        self.weights = tuple(weights)
        self.pair = pair
        self.bonus = bonus
        self.base = base
        self.seen: list[int] = []
        self._lookup = {
            self._key(removal.render(players, coalition)): frozenset(coalition)
            for coalition in powerset(range(len(players)))
        }

    @staticmethod
    def _key(image: NDArray[np.uint8]) -> bytes:
        return image.tobytes()

    def worth(self, coalition: Iterable[int]) -> float:
        """The closed-form value of a coalition, independent of any rendering."""
        members = frozenset(coalition)
        total = self.base + sum(self.weights[index] for index in members)
        if members.issuperset(self.pair):
            total += self.bonus
        return total

    def values(
        self,
        images: Sequence[NDArray[np.uint8]],
        target: DishTarget,  # noqa: ARG002 - fixed by the real signature.
        **_: object,
    ) -> NDArray[np.float64]:
        """Recover each rendering's coalition and return its closed-form worth.

        Recording the batch sizes in `seen` is what lets a test tell real
        chunking apart from a loop that happens to produce the right answers.
        """
        self.seen.append(len(images))
        return np.array(
            [self.worth(self._lookup[self._key(image)]) for image in images],
            dtype=np.float64,
        )


def as_value_function(stub: DecodingValueFunction) -> DishValueFunction:
    """Hand the stub to the game under the game's real parameter type.

    `IngredientGame` calls exactly one method on its value function --
    `values` -- so the stub is a faithful stand-in. Only mypy needs
    convincing. Done with a cast rather than by widening the game's
    annotation to a `Protocol`, because loosening a production signature so
    that a test can pass is how a real type error gets in later.
    """
    return cast("DishValueFunction", stub)


def build_game(
    num_players: int = 4,
    *,
    weights: Sequence[float] | None = None,
    pair: tuple[int, int] = (0, 2),
    bonus: float = 6.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[IngredientGame, DecodingValueFunction, PlayerSet]:
    """A game with a known closed-form Shapley decomposition."""
    players = make_players(num_players)
    removal = BlurRemoval(radius=2.0, keep_background=False)
    # Deliberately asymmetric: equal weights would hide a permuted column
    # convention, because every player would have the same Shapley value.
    chosen = weights if weights is not None else [1.0, 20.0, 300.0, 4000.0][:num_players]
    value_fn = DecodingValueFunction(players, removal, chosen, pair, bonus)
    game = IngredientGame(
        players, as_value_function(value_fn), removal, TARGET, chunk_size=chunk_size
    )
    return game, value_fn, players


def coalition_rows(indices: Iterable[Iterable[int]], num_players: int) -> NDArray[np.bool_]:
    """Build a coalition matrix in exactly the order given."""
    rows = [list(group) for group in indices]
    matrix = np.zeros((len(rows), num_players), dtype=np.bool_)
    for position, group in enumerate(rows):
        matrix[position, group] = True
    return matrix


def brute_force_shapley(value_fn: DecodingValueFunction, num_players: int) -> list[float]:
    """Shapley values from the definition, with no `shapiq` involved."""
    everyone = range(num_players)
    result = []
    for player in everyone:
        others = [index for index in everyone if index != player]
        total = 0.0
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(num_players - size - 1)
            share = weight / math.factorial(num_players)
            for subset in combinations(others, size):
                marginal = value_fn.worth([*subset, player]) - value_fn.worth(subset)
                total += share * marginal
        result.append(total)
    return result
