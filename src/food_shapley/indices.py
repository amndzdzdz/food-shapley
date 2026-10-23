"""Exact Shapley values and interaction indices for one image.

`PLAN.md` Milestone 1, step 4: "compute exact SV, k-SII and Möbius via
``shapiq.ExactComputer``; build in the two checks now". This module owns the
first half of that sentence and the efficiency half of the second; the visual
check lives in :mod:`food_shapley.preview` and was done in W01-07.

One evaluation, three indices
-----------------------------
``ExactComputer`` calls the game **once** with all ``2^n`` coalitions and
derives every index from those cached values -- verified by instrumenting a
game and counting calls, not assumed. So asking for all three families costs
exactly as many CLIP forward passes as asking for one, which is why the
Möbius transform is always computed at full order even when ``k-SII`` is
reported at order 2: the expensive part is already paid for, and re-deriving
it later would mean re-running the model.

Nothing here is an approximation. ``shapiq`` also ships samplers
(``SVARMIQ``, ``KernelSHAPIQ``); `PLAN.md` reserves those for Milestone 2 if a
model's per-coalition cost makes exact computation impractical. At 2 to 11
players the whole split is 91,828 coalitions, so it is not needed and is not
used -- an approximated value would quietly turn the efficiency check into a
statement about the sampler.

Why interactions are re-densified and stored as a membership matrix
-------------------------------------------------------------------
``shapiq`` returns each index as a flat value array plus an
``interaction_lookup`` dict, and that dict is wrong to trust in two separate
ways. Both were found by probing the library, not by reading its docs.

**It is not in a predictable order.** For a three-player game, ``k-SII`` at
order 2 comes back as ``{(): 0, (0,2): 1, (0,): 2, (1,): 3, (2,): 4}`` -- the
pair before the singletons. Powerset order happens to hold for the coalition
values and for Möbius; relying on it for ``k-SII`` would attribute a pair's
interaction to a single ingredient.

**It is not complete, and what it omits depends on the numbers.** ``k-SII``
silently drops interactions that come out exactly zero. On a four-player
fixture whose player 1 contributes nothing, the order-2 result has 6 entries
instead of 11 and player 1 has no *singleton* row at all. ``SV`` and Möbius
are dense today, but that is the same bet that just lost.

So every block is rebuilt here over the full set of subsets up to its order,
in powerset order, with absent entries filled in as ``0.0`` and any
unrecognised entry rejected. A stored array whose *shape* depends on its
values would make every downstream count value-dependent: "how many ingredient
pairs interact at all" would quietly exclude the pairs that do not, and with
float noise the omission would bite on some images and not others.

Nothing stored is positional even so. Every value array is paired with an
explicit boolean ``(n_interactions, n_players)`` matrix saying which players
each value belongs to, and that matrix is what gets written to disk. Same
defence as :mod:`food_shapley.game` uses for coalitions, for the same reason:
a consistent mislabelling passes every numerical check there is.

The efficiency check
--------------------
``sum of the per-player Shapley values == v(N) - v(empty)``. Two things make
it a real check rather than a tautology:

* Both ends are read out of ``ExactComputer``'s single evaluation. Re-scoring
  the grand coalition separately shifts it by ~1e-06, because CLIP's float32
  forward pass is not batch-invariant (W01-11), and a test written that way
  measures rounding instead of correctness.
* The sum runs over the *player* entries. ``shapiq``'s ``SV`` result also
  carries an empty-set entry, so ``values.sum()`` would silently stop being an
  efficiency check if the game's normalisation ever changed.

It is checked on every image, not sampled: a violation means the game or the
computation is wrong, and finding that out after the full run is exactly what
`PLAN.md` warns against.

What it does *not* catch: a game that consistently mislabels its coalitions
satisfies it perfectly. That failure is covered in ``tests/unit/test_game.py``
and ``tests/unit/test_indices.py`` by checking values against a closed form,
not against each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import shapiq
from shapiq.utils import powerset

from food_shapley.config import load_config
from food_shapley.dataset import load_split
from food_shapley.game import DEFAULT_CHUNK_SIZE, IngredientGame
from food_shapley.segments import build_players
from food_shapley.value import DishTarget, DishValueFunction

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from numpy.typing import NDArray
    from shapiq.interaction_values import InteractionValues

__all__ = [
    "DEFAULT_EFFICIENCY_TOLERANCE",
    "ImageIndices",
    "IndexBlock",
    "compute_indices",
    "load_indices",
    "save_indices",
]

#: ``shapiq`` index names, spelled exactly as it expects them.
SHAPLEY: Final = "SV"
INTERACTION: Final = "k-SII"
MOEBIUS: Final = "Moebius"

#: Largest ``|sum(SV) - (v(N) - v(empty))|`` accepted before a result is
#: rejected as wrong.
#:
#: Chosen from a measurement. On random exact games of 2 to 14 players the
#: residual is float-accumulation noise: at most 2.7e-15 absolute and 3.1e-14
#: relative, and on the three real images of W01-11 it is exactly ``0.0``.
#: This default sits six orders of magnitude above that and six below the
#: ~4-nat quantity being decomposed, so it rejects a genuine bug without ever
#: firing on arithmetic. It is deliberately *not* an
#: :class:`~food_shapley.config.ExperimentConfig` field: a tolerance is a
#: statement about float arithmetic, not about the experiment, and making it
#: configurable would invite loosening it to make a failure go away.
DEFAULT_EFFICIENCY_TOLERANCE: Final = 1e-9


def _membership_matrix(lookup: Mapping[tuple[int, ...], int], n_players: int) -> NDArray[np.bool_]:
    """Turn ``shapiq``'s ``{members: row}`` dict into an explicit bool matrix.

    Args:
        lookup: Mapping from a player-index tuple to its row in the value
            array.
        n_players: Number of players, i.e. the width of the result.

    Returns:
        Boolean array of shape ``(len(lookup), n_players)``; row ``i`` marks
        the players the ``i``-th value belongs to.

    Raises:
        ValueError: If the lookup is not a bijection onto
            ``range(len(lookup))``, or names a player outside the game. Either
            would leave an all-``False`` row, which is indistinguishable from
            the empty interaction and would misattribute a value.
    """
    rows = np.zeros((len(lookup), n_players), dtype=np.bool_)
    filled = np.zeros(len(lookup), dtype=np.bool_)
    for members, position in lookup.items():
        if not 0 <= position < len(lookup):
            msg = f"interaction {members} maps to row {position}, outside 0..{len(lookup) - 1}"
            raise ValueError(msg)
        if filled[position]:
            msg = f"row {position} is claimed by more than one interaction, including {members}"
            raise ValueError(msg)
        for player in members:
            if not 0 <= player < n_players:
                msg = f"interaction {members} names player {player} in a {n_players}-player game"
                raise ValueError(msg)
        rows[position, list(members)] = True
        filled[position] = True
    # No completeness check is needed: `len(lookup)` distinct positions, each
    # inside `range(len(lookup))`, fill it by the pigeonhole principle. The
    # caller is responsible for knowing how long the lookup should have been.
    return rows


@dataclass(frozen=True, slots=True)
class IndexBlock:
    """One index family: values, each tied to the players it belongs to.

    Attributes:
        index: ``shapiq``'s name for the family -- ``"SV"``, ``"k-SII"`` or
            ``"Moebius"``.
        order: Largest interaction size in this block, *as computed*. For
            ``k-SII`` this is the configured ``interaction_order`` clamped to
            the image's player count; see :func:`compute_indices`.
        members: Boolean ``(n_interactions, n_players)`` matrix; row ``i``
            marks which players value ``i`` belongs to. An all-``False`` row is
            the empty interaction, which ``shapiq`` includes and which is 0 for
            a normalised game.
        values: Array of shape ``(n_interactions,)``, aligned to ``members``.
    """

    index: str
    order: int
    members: NDArray[np.bool_]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate that the values and their membership rows line up.

        Raises:
            ValueError: If the shapes disagree. Nothing downstream could detect
                a misalignment, because both arrays are plausible on their own.
        """
        if self.members.ndim != 2:
            msg = f"members must be 2-D (interactions, players), got {self.members.shape}"
            raise ValueError(msg)
        if self.values.ndim != 1:
            msg = f"values must be 1-D, got shape {self.values.shape}"
            raise ValueError(msg)
        if self.members.shape[0] != self.values.shape[0]:
            msg = (
                f"{self.members.shape[0]} membership rows for {self.values.shape[0]} "
                f"values in index {self.index!r}"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        """Number of values in this block."""
        return int(self.values.shape[0])

    @property
    def n_players(self) -> int:
        """Players in the game this block decomposes."""
        return int(self.members.shape[1])

    @property
    def sizes(self) -> NDArray[np.int64]:
        """Interaction size per row: 0 for the empty set, 1 for a main effect."""
        return self.members.sum(axis=1).astype(np.int64)

    def main_effects(self) -> NDArray[np.float64]:
        """Return the single-player values, indexed by player.

        The per-player Shapley value is the project's headline number, and
        recovering it means finding the row whose membership is exactly that
        one player -- not row ``j + 1``, which is true for ``SV`` and false for
        ``k-SII``. Doing that lookup here once stops every caller from
        reinventing it, correctly or otherwise.

        Returns:
            Array of shape ``(n_players,)``; entry ``j`` is player ``j``'s
            main effect.

        Raises:
            ValueError: If some player has no singleton row. Every index family
                computed here reports main effects, so a missing one means the
                block is not what it claims to be.
        """
        found = np.full(self.n_players, np.nan, dtype=np.float64)
        singletons = np.flatnonzero(self.sizes == 1)
        for row in singletons:
            found[np.argmax(self.members[row])] = self.values[row]
        if np.isnan(found).any():
            missing = np.flatnonzero(np.isnan(found)).tolist()
            msg = f"index {self.index!r} has no main effect for players {missing}"
            raise ValueError(msg)
        return found

    def value_of(self, members: Iterable[int]) -> float:
        """Return the value attributed to exactly this set of players.

        Args:
            members: Player indices. Order and repetition are irrelevant; the
                set is what identifies the interaction.

        Returns:
            The stored value for that interaction.

        Raises:
            KeyError: If this block has no row for that set -- which for
                ``k-SII`` simply means the set is larger than :attr:`order`.
        """
        wanted = np.zeros(self.n_players, dtype=np.bool_)
        wanted[list(set(members))] = True
        match = np.flatnonzero((self.members == wanted).all(axis=1))
        if match.size == 0:
            present = tuple(np.flatnonzero(wanted).tolist())
            msg = f"index {self.index!r} has no entry for players {present}"
            raise KeyError(msg)
        return float(self.values[match[0]])


def _dense_block(index: str, order: int, n_players: int, result: InteractionValues) -> IndexBlock:
    """Rebuild one ``shapiq`` result over every subset up to ``order``.

    ``shapiq`` omits exactly-zero ``k-SII`` entries, so the rows it returns
    depend on the numbers rather than on the game. Re-densifying makes the
    stored shape a function of ``n_players`` and ``order`` alone, which is
    what lets a later analysis iterate "all ingredient pairs" and mean it.

    Args:
        index: ``shapiq``'s name for the family, kept for the block.
        order: Largest interaction size this family carries.
        n_players: Players in the game.
        result: What ``shapiq`` returned.

    Returns:
        A block with one row per subset of size ``0..order``, in powerset
        order, with omitted entries set to ``0.0``.

    Raises:
        ValueError: If ``shapiq`` returned an entry outside that set -- a
            subset larger than ``order``, or a player the game does not have.
            Dropping it silently would discard a real value; keeping it would
            break the fixed shape.
    """
    expected = list(powerset(range(n_players), max_size=order))
    position = {members: row for row, members in enumerate(expected)}

    unexpected = sorted(set(result.interaction_lookup) - position.keys())
    if unexpected:
        msg = (
            f"index {index!r} at order {order} returned entries outside the expected "
            f"subsets of a {n_players}-player game: {unexpected}"
        )
        raise ValueError(msg)

    members = np.zeros((len(expected), n_players), dtype=np.bool_)
    for row, group in enumerate(expected):
        members[row, list(group)] = True

    values = np.zeros(len(expected), dtype=np.float64)
    for group, source in result.interaction_lookup.items():
        values[position[group]] = result.values[source]

    return IndexBlock(index=index, order=order, members=members, values=values)


@dataclass(frozen=True, slots=True)
class ImageIndices:
    """Everything one image's exact decomposition produced.

    This is the unit task 1.6 writes one ``.npz`` per image of.

    Attributes:
        image_index: Position of the image in the dataset split.
        player_names: Player names, in coalition-column order.
        ingredient_indices: Positions in ``player_names`` that are real
            ingredients. The scene is a player but never an ingredient, so
            every ingredient-level statistic must filter on this; see W01-09.
        target: The dish that was decomposed, and its log-probability on the
            unmasked photograph.
        coalitions: Boolean ``(2^n, n_players)`` matrix of every coalition, in
            the order ``shapiq`` evaluated them.
        values: ``v(S) - v(empty)`` for each row of ``coalitions``. Stored
            normalised because that is what the indices were computed from;
            add :attr:`baseline` to recover the raw log-probability.
        baseline: ``v(empty)``, the raw log-probability of the target on a
            fully blurred frame.
        shapley: Exact Shapley values.
        interactions: Exact k-SII values at :attr:`IndexBlock.order`.
        moebius: The Möbius transform, always at full order.
        efficiency_residual: ``sum(main effects) - (v(N) - v(empty))``, both
            ends from the same evaluation. Recorded rather than merely
            asserted, so the full run can report its distribution instead of
            just "it passed".
        chunk_size: Coalitions per forward pass. Not part of the experiment --
            see :data:`~food_shapley.game.DEFAULT_CHUNK_SIZE` -- but recorded
            so that a bit-level difference between two runs can be attributed
            rather than investigated.
    """

    image_index: int
    player_names: tuple[str, ...]
    ingredient_indices: tuple[int, ...]
    target: DishTarget
    coalitions: NDArray[np.bool_]
    values: NDArray[np.float64]
    baseline: float
    shapley: IndexBlock
    interactions: IndexBlock
    moebius: IndexBlock
    efficiency_residual: float
    chunk_size: int

    @property
    def n_players(self) -> int:
        """Number of coalition players."""
        return len(self.player_names)

    @property
    def span(self) -> float:
        """``v(N) - v(empty)``: the evidence the Shapley values divide up."""
        return float(self.values[self.coalitions.all(axis=1)][0])


def compute_indices(
    game: IngredientGame,
    *,
    order: int,
    tolerance: float = DEFAULT_EFFICIENCY_TOLERANCE,
) -> ImageIndices:
    """Compute the exact SV, k-SII and Möbius values of one image's game.

    The game is evaluated once, for all ``2^n`` coalitions; every index is
    derived from those values.

    Args:
        game: The image's coalition game.
        order: The ``k`` of k-SII, from
            :attr:`~food_shapley.config.ExperimentConfig.interaction_order`.
            Clamped to the image's player count, because ``shapiq`` raises an
            opaque ``IndexError`` when asked for order 2 on a one-player game.
            The clamped value is what :attr:`IndexBlock.order` reports.
        tolerance: Largest acceptable efficiency residual; see
            :data:`DEFAULT_EFFICIENCY_TOLERANCE`.

    Returns:
        The image's decomposition, with the efficiency residual recorded.

    Raises:
        ValueError: If ``order`` is below 1, or the efficiency check fails.
            A failure means the game or the computation is wrong, so it stops
            the image rather than storing a number nobody can trust.
    """
    if order < 1:
        msg = f"interaction order must be >= 1, got {order}"
        raise ValueError(msg)

    players = game.players
    n_players = game.n_players
    effective_order = min(order, n_players)

    computer = shapiq.ExactComputer(game)
    # Touching `game_values` is what triggers the single powerset evaluation.
    # Reading the coalition values from the computer, rather than evaluating
    # the game again, is what makes the efficiency check below exact.
    values = np.asarray(computer.game_values, dtype=np.float64)
    if len(computer.coalition_lookup) != 2**n_players:
        msg = (
            f"image {players.index}: shapiq evaluated {len(computer.coalition_lookup)} "
            f"coalitions, not the {2**n_players} an exact computation needs"
        )
        raise ValueError(msg)
    coalitions = _membership_matrix(computer.coalition_lookup, n_players)

    built = {
        SHAPLEY: _dense_block(SHAPLEY, 1, n_players, computer(SHAPLEY)),
        INTERACTION: _dense_block(
            INTERACTION,
            effective_order,
            n_players,
            computer(INTERACTION, order=effective_order),
        ),
        MOEBIUS: _dense_block(MOEBIUS, n_players, n_players, computer(MOEBIUS)),
    }

    empty_row = int(np.flatnonzero(~coalitions.any(axis=1))[0])
    grand_row = int(np.flatnonzero(coalitions.all(axis=1))[0])
    span = float(values[grand_row] - values[empty_row])
    residual = float(built[SHAPLEY].main_effects().sum() - span)
    if abs(residual) > tolerance:
        msg = (
            f"efficiency check failed for image {players.index}: the Shapley values sum "
            f"to {residual:+.3e} more than v(N) - v(empty) = {span:.6f}, which exceeds "
            f"the tolerance of {tolerance:.1e}"
        )
        raise ValueError(msg)

    return ImageIndices(
        image_index=players.index,
        player_names=tuple(players.names),
        ingredient_indices=tuple(players.ingredient_indices),
        target=game.target,
        coalitions=coalitions,
        values=values,
        baseline=float(game.normalization_value),
        shapley=built[SHAPLEY],
        interactions=built[INTERACTION],
        moebius=built[MOEBIUS],
        efficiency_residual=residual,
        chunk_size=game.chunk_size,
    )


def save_indices(path: Path, result: ImageIndices) -> None:
    """Write one image's decomposition to a compressed ``.npz``.

    Written by hand rather than with ``shapiq.Game.save_values``, which stores
    rows in dict-insertion order and so cannot be read back positionally.
    Every value array here is accompanied by the membership matrix that says
    what its rows mean.

    Args:
        path: Destination file. Its parent must exist.
        result: The decomposition to store.
    """
    meta = {
        "image_index": result.image_index,
        "player_names": list(result.player_names),
        "target": {
            "index": result.target.index,
            "name": result.target.name,
            "prompt": result.target.prompt,
            "log_prob": result.target.log_prob,
        },
        "baseline": result.baseline,
        "efficiency_residual": result.efficiency_residual,
        "chunk_size": result.chunk_size,
        "orders": {name: block.order for name, block in _blocks_of(result).items()},
    }
    arrays = {
        "coalitions": result.coalitions,
        "values": result.values,
        "ingredient_indices": np.asarray(result.ingredient_indices, dtype=np.int64),
        "meta": np.asarray(json.dumps(meta, sort_keys=True)),
    }
    for name, block in _blocks_of(result).items():
        arrays[f"{name}_members"] = block.members
        arrays[f"{name}_values"] = block.values
    # `allow_pickle=False` is a real guard, not boilerplate: every array here is
    # numeric or a JSON string, so pickling would mean something unintended got
    # in, and a pickled artifact is neither safe to load nor stable to read.
    np.savez_compressed(path, allow_pickle=False, **arrays)


def load_indices(path: Path) -> ImageIndices:
    """Read back a file written by :func:`save_indices`.

    Args:
        path: The ``.npz`` to read.

    Returns:
        The stored decomposition.

    Raises:
        ValueError: If the file's arrays disagree about the player count. A
            file assembled from two different images would otherwise load
            cleanly and attribute one image's values to another's ingredients.
    """
    # The stream is opened here rather than handed to `np.load` as a path
    # because `np.load` leaks the descriptor it opened when the archive turns
    # out to be corrupt -- observed on a half-written `.npz`. `run_verify`
    # expects to survive corrupt files and report them, and it would otherwise
    # leak one descriptor per corrupt file in a 2,135-image run.
    with path.open("rb") as stream, np.load(stream) as handle:
        meta = json.loads(str(handle["meta"]))
        coalitions = handle["coalitions"].astype(np.bool_)
        values = handle["values"].astype(np.float64)
        ingredient_indices = handle["ingredient_indices"].astype(np.int64)
        blocks = {
            name: IndexBlock(
                index=name,
                order=int(meta["orders"][name]),
                members=handle[f"{name}_members"].astype(np.bool_),
                values=handle[f"{name}_values"].astype(np.float64),
            )
            for name in (SHAPLEY, INTERACTION, MOEBIUS)
        }

    names = tuple(meta["player_names"])
    widths = {coalitions.shape[1]} | {block.n_players for block in blocks.values()}
    if widths != {len(names)}:
        msg = (
            f"{path}: arrays disagree about the player count -- {len(names)} names "
            f"but widths {sorted(widths)}"
        )
        raise ValueError(msg)

    return ImageIndices(
        image_index=int(meta["image_index"]),
        player_names=names,
        ingredient_indices=tuple(int(index) for index in ingredient_indices),
        target=DishTarget(**meta["target"]),
        coalitions=coalitions,
        values=values,
        baseline=float(meta["baseline"]),
        shapley=blocks[SHAPLEY],
        interactions=blocks[INTERACTION],
        moebius=blocks[MOEBIUS],
        efficiency_residual=float(meta["efficiency_residual"]),
        chunk_size=int(meta["chunk_size"]),
    )


def _blocks_of(result: ImageIndices) -> dict[str, IndexBlock]:
    """The three index families, keyed by ``shapiq``'s name for each."""
    return {
        SHAPLEY: result.shapley,
        INTERACTION: result.interactions,
        MOEBIUS: result.moebius,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Run configuration (TOML).")
    parser.add_argument("--image", type=int, required=True, help="Index of the image in the split.")
    parser.add_argument("--device", default=None, help="Torch device; default auto.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Coalitions per forward pass; a throughput knob, not an experiment setting.",
    )
    parser.add_argument(
        "--save", type=Path, default=None, help="Optional .npz to write the result to."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Decompose a single image and print its indices.

    The one-image entry point `PLAN.md` step 4 asks for: enough to see a real
    decomposition and its efficiency residual before task 1.6 commits the
    cluster to the whole split.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero if the image cannot be decomposed.
    """
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    split = load_split(config.split)
    players = build_players(
        split[args.image],
        split.names,
        image_size=config.image_size,
        max_players=config.max_players,
        min_area_fraction=config.min_area_fraction,
        background_player=config.background_is_player,
    )
    value_fn = DishValueFunction.from_config(config, device=args.device)
    game = IngredientGame.for_image(players, value_fn, config, chunk_size=args.chunk_size)

    try:
        result = compute_indices(game, order=config.interaction_order)
    except ValueError as error:
        print(f"image {args.image}: {error}", file=sys.stderr)
        return 1

    print(f"image               {result.image_index}")
    print(f"target              {result.target.name} (p = {result.target.probability:.3f})")
    print(f"players             {result.n_players}: {', '.join(result.player_names)}")
    print(f"v(N) - v(empty)     {result.span:+.6f} nats")
    print(f"efficiency residual {result.efficiency_residual:+.3e}")
    print("\nShapley value per player")
    effects = result.shapley.main_effects()
    for position in np.argsort(-effects):
        marker = " " if position in result.ingredient_indices else "*"
        print(f"  {marker} {result.player_names[position]:<24} {effects[position]:+.4f}")
    print(f"\nTop k-SII interactions (order <= {result.interactions.order})")
    pairs = np.flatnonzero(result.interactions.sizes >= 2)
    for row in pairs[np.argsort(-np.abs(result.interactions.values[pairs]))][:5]:
        members = " + ".join(
            result.player_names[position]
            for position in np.flatnonzero(result.interactions.members[row])
        )
        print(f"    {members:<40} {result.interactions.values[row]:+.4f}")
    print("\n* = not an ingredient")

    if args.save is not None:
        save_indices(args.save, result)
        print(f"\nwrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
