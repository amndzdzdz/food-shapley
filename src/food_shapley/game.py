"""The cooperative game: one image, one dish, one coalition value per subset.

`PLAN.md` Milestone 1, step 3. This is the join between the three halves built
so far -- players (:mod:`food_shapley.segments`), a removal operator
(:mod:`food_shapley.masking`) and a value function
(:mod:`food_shapley.value`) -- expressed in the interface ``shapiq`` consumes.

What the game is
----------------
For one image with players ``N``:

``v(S) = log p(target dish | only the players in S are visible)``

The target is fixed per image by :meth:`~food_shapley.value.DishValueFunction.
choose_target` on the **unmasked** photograph, so every coalition of one image
is scored against the same dish. Under ``background_mode = "player"`` the grand
coalition *is* the photograph, so ``v(N)`` reproduces the target's own
log-probability exactly, and ``v(empty)`` is a fully blurred frame.

Coalition ordering
------------------
`PLAN.md` flags this as the easy thing to get wrong, and it is the most
dangerous bug available here: a game that mislabels its own coalitions is still
perfectly self-consistent, so the efficiency check passes while every
attribution is wrong -- exactly the silent-failure shape of W01-05's mask
misalignment and W01-10's checkpoint mismatch.

The defence is to *not have an opinion*. ``shapiq`` passes a boolean matrix of
shape ``(n_coalitions, n_players)`` and expects one value per row, in row
order. :meth:`IngredientGame.value_function` therefore never sorts, groups,
deduplicates or caches by coalition: row ``i`` of the output is the score of
row ``i`` of the input, and chunking preserves position by construction.
``tests/unit/test_game.py`` pins this down by running a game whose value
*encodes its own coalition* through the real ``ExactComputer`` and checking
where each value lands, which is a test of ``shapiq``'s convention rather than
of a restatement of our own.

For the record, ``shapiq``'s own enumeration (``shapiq.utils.powerset``) is
ascending by coalition size and then lexicographic -- ``(), (0,), (1,), (2,),
(0,1), (0,2), (1,2), (0,1,2)`` -- **not** binary counting. Nothing in this
module relies on that, which is the point; it is recorded because a reader
comparing a stored value vector against an index will need it.

Normalisation
-------------
The game is constructed with ``normalize=True`` and an explicit
``normalization_value = v(empty)``, so ``shapiq`` reports ``v(S) - v(empty)``
and the empty coalition is worth exactly 0.

This is not cosmetic. Left unnormalised, the empty-set entry of a
``shapiq`` ``SV`` result carries ``v(empty)`` rather than 0, and
``values.sum()`` is then *not* ``v(N) - v(empty)`` -- an efficiency check
written the obvious way would fail for a correct game, or, worse, be "fixed"
by loosening the tolerance. Passing the baseline explicitly also sidesteps a
``RuntimeWarning`` that ``shapiq`` emits for ``normalize=True`` with no value,
which this project's ``filterwarnings = ["error"]`` turns into a failure.

Cost
----
One CLIP forward pass per coalition, plus one Gaussian blur per *image* --
:meth:`~food_shapley.masking.BlurRemoval.render` takes the blurred frame as an
argument so the powerset does not recompute it. Measured over the test split
the whole of Milestone 1 is 91,828 coalitions -- 2 to 11 players per image,
counted under the current configuration in W01-11 -- so this is not a compute
problem; the reason for chunking is memory, since materialising 2,048
renderings of a 224x224x3 image at once is ~300 MB per image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import shapiq

from food_shapley.masking import BlurRemoval
from food_shapley.value import DishValueFunction

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from food_shapley.config import ExperimentConfig
    from food_shapley.segments import PlayerSet
    from food_shapley.value import DishTarget

__all__ = ["DEFAULT_CHUNK_SIZE", "IngredientGame"]

#: Coalitions rendered and scored per batch.
#:
#: A throughput knob, and deliberately **not** an
#: :class:`~food_shapley.config.ExperimentConfig` field: tuning it for a
#: different GPU would change the run fingerprint, which would make two runs of
#: the same experiment look like different experiments.
#:
#: That is justified by a measurement rather than by an exact invariance.
#: *Indexing* is chunk-invariant exactly -- row ``i`` is scored as row ``i``
#: whatever it was batched with, which ``tests/unit/test_game.py`` asserts
#: against a stub -- but CLIP's float32 forward pass is **not** batch-invariant
#: at the last few bits. Measured on the three sample images of
#: ``tests/integration/test_game_on_real_images.py`` (W01-11), ``chunk_size=1``
#: against ``chunk_size=32`` over the whole powerset moves a coalition value by
#: at most 8.2e-06 nats, and moves an exact Shapley value by at most 2.9e-06 --
#: about 1e-06 of the ``v(N) - v(empty)`` span it is divided among. Small enough
#: to stay out of the fingerprint; too large to call it zero.
DEFAULT_CHUNK_SIZE: Final = 32


class IngredientGame(shapiq.Game):
    """The coalition game of one image, ready for ``shapiq``.

    One instance is one image and one target dish. The value function and the
    loaded checkpoint are shared across images by passing the same
    :class:`~food_shapley.value.DishValueFunction` to each game, so the model
    is loaded once per process rather than once per image.
    """

    def __init__(
        self,
        players: PlayerSet,
        value_fn: DishValueFunction,
        removal: BlurRemoval,
        target: DishTarget,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Bind one image's players to the operator and value function.

        Computes ``v(empty)`` eagerly -- one forward pass -- because
        ``shapiq`` needs the baseline at construction time to normalise.

        Args:
            players: The image's players, carrying the aligned image.
            value_fn: The dish value function; supplies the model.
            removal: The removal operator deciding what "hidden" looks like.
            target: The dish to decompose, from
                :meth:`~food_shapley.value.DishValueFunction.choose_target`
                applied to ``players.image``. Passed in rather than chosen here
                so that a caller can reuse one target across operators, which
                is what Week 2's blur-versus-token-drop comparison needs: two
                games of the same image must decompose the *same* dish or the
                comparison is meaningless.
            chunk_size: Coalitions rendered and scored per batch.

        Raises:
            ValueError: If the player set is empty, or ``chunk_size`` is not
                positive.
        """
        if len(players) == 0:
            msg = f"image {players.index} has no players, so there is no game to play"
            raise ValueError(msg)
        if chunk_size < 1:
            msg = f"chunk_size must be >= 1, got {chunk_size}"
            raise ValueError(msg)

        self._players = players
        self._value_fn = value_fn
        self._removal = removal
        self._target = target
        self._chunk_size = chunk_size
        # Blur once per image rather than once per coalition. The result is
        # the empty coalition's rendering whenever the scene is a player.
        self._blurred = removal.blurred(players)

        empty = np.zeros((1, len(players)), dtype=np.bool_)
        super().__init__(
            n_players=len(players),
            normalize=True,
            normalization_value=float(self.value_function(empty)[0]),
            player_names=list(players.names),
        )

    @classmethod
    def for_image(
        cls,
        players: PlayerSet,
        value_fn: DishValueFunction,
        config: ExperimentConfig,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> IngredientGame:
        """Build the game the configuration describes, choosing the target.

        Args:
            players: The image's players.
            value_fn: The dish value function.
            config: The run configuration; supplies the removal operator's
                settings, so they enter the run fingerprint.
            chunk_size: Coalitions rendered and scored per batch.

        Returns:
            A game whose target is the model's top-1 on ``players.image``.

        Raises:
            ValueError: If the configuration names a removal operator this
                module cannot build. Week 2's ``"token-drop"`` lands here until
                it exists, which is better than silently blurring instead.
        """
        if config.removal != "blur":
            msg = (
                f"removal operator {config.removal!r} is not implemented; only 'blur' exists so far"
            )
            raise ValueError(msg)
        removal = BlurRemoval(
            radius=config.blur_radius,
            keep_background=config.background_stays_sharp,
        )
        return cls(
            players,
            value_fn,
            removal,
            value_fn.choose_target(players.image),
            chunk_size=chunk_size,
        )

    @property
    def players(self) -> PlayerSet:
        """The image's players, in the order coalition columns index."""
        return self._players

    @property
    def target(self) -> DishTarget:
        """The dish being decomposed."""
        return self._target

    @property
    def removal(self) -> BlurRemoval:
        """The removal operator every value in this game is relative to."""
        return self._removal

    @property
    def chunk_size(self) -> int:
        """Coalitions scored per forward pass.

        Exposed so that a stored result can record which batching produced it.
        Not an experiment setting -- see :data:`DEFAULT_CHUNK_SIZE` -- but it
        moves a value in the last few bits, so a bit-level difference between
        two runs should be attributable without rerunning either.
        """
        return self._chunk_size

    def value_function(self, coalitions: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Score each coalition: ``v(S)`` for every row, in row order.

        Returns **raw** log-probabilities. ``shapiq`` subtracts the baseline
        itself in ``Game.__call__``; doing it here too would subtract it twice.

        Args:
            coalitions: Boolean matrix of shape ``(n_coalitions, n_players)``.
                Column ``j`` is :attr:`players`'s player ``j``.

        Returns:
            Array of shape ``(n_coalitions,)``; entry ``i`` scores row ``i``.

        Raises:
            ValueError: If the matrix is not 2-D or its width is not the player
                count. Both would otherwise silently mis-assign players to
                columns, which no downstream check can detect.
        """
        if coalitions.ndim != 2:
            msg = f"coalitions must be 2-D (coalitions, players), got shape {coalitions.shape}"
            raise ValueError(msg)
        if coalitions.shape[1] != len(self._players):
            msg = (
                f"coalitions have {coalitions.shape[1]} columns but the image has "
                f"{len(self._players)} players; the coalition matrix is for another image"
            )
            raise ValueError(msg)
        if coalitions.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        values = np.empty(coalitions.shape[0], dtype=np.float64)
        for start in range(0, coalitions.shape[0], self._chunk_size):
            rows = coalitions[start : start + self._chunk_size]
            renders = [
                self._removal.render(
                    self._players,
                    np.flatnonzero(row).tolist(),
                    blurred=self._blurred,
                )
                for row in rows
            ]
            values[start : start + len(renders)] = self._value_fn.values(renders, self._target)
        return values
