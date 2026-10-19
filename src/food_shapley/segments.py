"""Segmentation masks to coalition players.

`PLAN.md` Milestone 1, step 1, second half: "turn masks into players (merge
small segments, add 'rest', align masks with the resized image)". A player is
one ingredient as the model actually sees it; a coalition is a subset of
players; everything downstream -- the removal operator, the value function, the
Shapley computation -- indexes into :attr:`PlayerSet.players`.

Geometry, and why it is copied rather than invented
---------------------------------------------------
CLIP's own preprocessing is a shortest-side resize to 224 followed by a centre
crop (verified from ``open_clip.create_model_and_transforms("ViT-B-32")``).
This module reproduces that geometry exactly, down to torchvision's truncating
resize arithmetic, and applies it to the image and the mask together. Masks are
resampled **nearest-neighbour**: any smoothing invents label ids that name no
ingredient, and bicubic on a label map is meaningless.

A centre crop discards image content, so some ingredients are not in the input
the model receives. Measured on the full 2,135-image test split: 46 of 7,697
ingredient instances (0.60 %), affecting 44 images (2.06 %), disappear
entirely. They are **dropped from the player set**, not kept as empty players:
an empty player has Shapley value exactly zero by construction, which would
silently dilute every "how often is one ingredient enough?" statistic. The
dropped ids are recorded in :attr:`PlayerSet.dropped_labels` so the loss can be
reported rather than assumed away.

Squashing to 224x224 instead would drop nothing, but it feeds CLIP a distorted
aspect ratio it was not trained on, which is a confound on 100 % of images to
avoid one on 2 %. See ``insights/week-01.md``.

Players are ingredient classes, not connected components
--------------------------------------------------------
One ``carrot`` id can cover several disjoint blobs; they form a single player.
The research question is whether a dish is recognised from an ingredient or
from a combination of ingredients, so "the carrot in this image" is the unit
that question is about. Per-component players would answer a different
question (which *region* mattered) and would inflate the player count.

Merging and the "rest" player
-----------------------------
Merging exists to bound the exact-computation cost. Measured on the test split
after cropping, the ingredient count peaks at 10 -- 11 players once ``scene``
takes its slot -- and the median segment covers 12.9 % of the frame, so at
``max_players = 11`` the cap never fires on this data; the mechanism is kept
because `PLAN.md` asks for it and the train split is not the test split.
``min_area_fraction`` is the part that does fire, and only on sub-patch
specks (0.94 % of segments below 0.5 % of the frame, a quarter of one 32-pixel
ViT patch).

The synthetic merged player is called ``rest``. It is deliberately not called
"other ingredients", which is the name of real FoodSeg103 class 103; the two
must stay distinguishable in any figure. It carries ``label=None`` so the
collision cannot happen by accident.

The scene as a player
---------------------
Label 0 is not an ingredient -- `PLAN.md` is explicit, and this module honours
that: the ``scene`` player is never counted as one. But "not an ingredient" and
"not a player" are different claims, and treating them as the same forces a
choice between two broken games:

* blur the scene in every coalition, and ``v(N)`` is no longer the photograph,
  so the decomposition explains a prediction the model never made;
* keep it sharp in every coalition, and it becomes a free rider -- present
  everywhere, attributed nowhere. Measured over the test split the scene is
  39 % of the model's input on average and the majority of pixels in 30 % of
  images, and on image 1510 (88 % scene) the coffee *mug* is scene, so the most
  diagnostic object in the frame would never receive a value.

Making it a player dissolves the choice: ``v(N)`` is the unmodified photograph,
``v(empty)`` is a fully blurred frame, and the scene's contribution and its
interactions with ingredients are measured rather than assumed. The cost is one
extra player on 2,134 of 2,135 images, which doubles the test split from 45,916
to 91,828 coalitions -- affordable, since `PLAN.md` holds that "compute is not
the bottleneck". See ``insights/week-01.md`` W01-09.

It is kept identifiable so that no ingredient-level statistic can silently
absorb it: the player is named :data:`SCENE_PLAYER_NAME`, carries
``label = BACKGROUND_LABEL``, answers :attr:`Player.is_scene`, and is excluded
from :attr:`PlayerSet.ingredient_indices`. Any claim of the form "one
ingredient was enough" must be computed over those indices, never over all
players.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from food_shapley.download import BACKGROUND_LABEL

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import NDArray

    from food_shapley.dataset import FoodSegExample

__all__ = [
    "REST_PLAYER_NAME",
    "SCENE_PLAYER_NAME",
    "Player",
    "PlayerSet",
    "build_players",
    "clip_geometry",
    "resize_image",
    "resize_mask",
]

#: Name of the synthetic player holding merged-away ingredients. Distinct from
#: FoodSeg103 class 103, "other ingredients", on purpose.
REST_PLAYER_NAME = "rest"

#: Name of the player covering everything that is not an ingredient: plate,
#: table, cutlery, a mug. Called "scene" rather than "background" so that a
#: figure legend cannot be confused with FoodSeg103's class 0.
SCENE_PLAYER_NAME = "scene"


@dataclass(frozen=True, slots=True)
class _Geometry:
    """Where the crop window sits in the resized image.

    Attributes:
        resized: ``(width, height)`` after the shortest-side resize.
        left: Left edge of the crop window in the resized image.
        top: Top edge of the crop window in the resized image.
        size: Side length of the square crop.
    """

    resized: tuple[int, int]
    left: int
    top: int
    size: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Crop box in PIL's ``(left, upper, right, lower)`` form."""
        return (self.left, self.top, self.left + self.size, self.top + self.size)


def clip_geometry(size: tuple[int, int], target: int) -> _Geometry:
    """Compute CLIP's resize-and-centre-crop geometry for one image size.

    Reproduces ``torchvision.transforms.Resize(target)`` followed by
    ``CenterCrop(target)``, including their rounding: the long side truncates,
    the crop offset rounds. Getting this wrong by one pixel would misalign
    every mask against every image, which is exactly the failure `PLAN.md`
    warns about and which no efficiency check can detect.

    Args:
        size: ``(width, height)`` of the original image.
        target: Side length of the model's square input, e.g. 224.

    Returns:
        The geometry, shared by the image and its mask.

    Raises:
        ValueError: If ``target`` is not positive or the image is degenerate.
    """
    if target <= 0:
        msg = f"target size must be positive, got {target}"
        raise ValueError(msg)
    width, height = size
    if width <= 0 or height <= 0:
        msg = f"image size must be positive, got {size}"
        raise ValueError(msg)

    short, long = (width, height) if width <= height else (height, width)
    # int(), not round(): torchvision truncates here.
    new_long = int(target * long / short)
    new_width, new_height = (target, new_long) if width <= height else (new_long, target)
    # round(), not int(): torchvision's centre crop rounds the offset while its
    # resize truncates the long side. The two are genuinely different.
    left = round((new_width - target) / 2.0)
    top = round((new_height - target) / 2.0)
    return _Geometry(resized=(new_width, new_height), left=left, top=top, size=target)


def resize_image(image: Image.Image, target: int) -> NDArray[np.uint8]:
    """Apply CLIP's geometry to a photograph.

    Bicubic with antialiasing, matching open_clip's transform.

    Returns:
        RGB array of shape ``(target, target, 3)``.
    """
    geometry = clip_geometry(image.size, target)
    resized = image.convert("RGB").resize(geometry.resized, Image.Resampling.BICUBIC)
    return np.asarray(resized.crop(geometry.box), dtype=np.uint8)


def resize_mask(mask: NDArray[np.uint8], target: int) -> NDArray[np.uint8]:
    """Apply CLIP's geometry to a label mask.

    Nearest-neighbour, so that the output contains only label ids that were in
    the input. Interpolating a label map would produce ids naming no
    ingredient, and would blend two ingredients into a third at every boundary.

    Args:
        mask: Label ids, shape ``(height, width)``.
        target: Side length of the model's square input.

    Returns:
        Label ids of shape ``(target, target)``, aligned with
        :func:`resize_image` applied to the same image.
    """
    height, width = mask.shape
    geometry = clip_geometry((width, height), target)
    resized = Image.fromarray(mask).resize(geometry.resized, Image.Resampling.NEAREST)
    return np.asarray(resized.crop(geometry.box), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class Player:
    """One coalition player: an ingredient as the model sees it.

    Attributes:
        label: FoodSeg103 class id; ``None`` for the synthetic ``rest`` player
            and :data:`~food_shapley.download.BACKGROUND_LABEL` for ``scene``.
        name: Ingredient name, :data:`REST_PLAYER_NAME`, or
            :data:`SCENE_PLAYER_NAME`.
        mask: Boolean mask over the model's square input. Disjoint from every
            other player's mask.
        merged_labels: Class ids folded into this player. Empty for an
            ordinary player; populated only for ``rest``.
    """

    label: int | None
    name: str
    mask: NDArray[np.bool_]
    merged_labels: tuple[int, ...] = ()

    @property
    def is_rest(self) -> bool:
        """Whether this is the synthetic merged player."""
        return self.label is None

    @property
    def is_scene(self) -> bool:
        """Whether this is the non-ingredient player: plate, table, cutlery."""
        return self.label == BACKGROUND_LABEL

    @property
    def is_ingredient(self) -> bool:
        """Whether this player stands for food.

        True for ``rest``, which is a union of real ingredients, and false only
        for ``scene``. Ingredient-level claims must filter on this.
        """
        return not self.is_scene

    @property
    def area_fraction(self) -> float:
        """Share of the model's input covered by this player."""
        return float(self.mask.mean())


@dataclass(frozen=True, slots=True)
class PlayerSet:
    """The players of one image, with the input they are aligned to.

    The image is carried alongside the players because their alignment is the
    property that matters and is easiest to break: both come out of a single
    call to :func:`build_players` under one geometry.

    Attributes:
        index: Row index of the source image within its split.
        image: RGB array of shape ``(size, size, 3)``, in CLIP's geometry.
        players: Players in a fixed order -- ascending class id, then ``rest``,
            then ``scene``. Coalitions index into this tuple, so the order is
            part of the stored results and must not depend on dictionary
            iteration. ``scene`` goes last so that adding it cannot renumber an
            ingredient.
        background: Boolean mask of pixels belonging to no player. **Empty when
            ``scene`` is a player**, because then every pixel has an owner;
            that is what makes ``v(N)`` the unmodified photograph.
        dropped_labels: Class ids present in the original mask but absent from
            the model's input because the centre crop removed them.
        capped: Whether ``max_players`` forced a merge, as opposed to
            ``min_area_fraction`` alone.
    """

    index: int
    image: NDArray[np.uint8]
    players: tuple[Player, ...]
    background: NDArray[np.bool_]
    dropped_labels: tuple[int, ...]
    capped: bool

    def __len__(self) -> int:
        """Number of players, i.e. the size of the ground set ``N``."""
        return len(self.players)

    @property
    def names(self) -> tuple[str, ...]:
        """Player names in player order, for labelling results and figures."""
        return tuple(player.name for player in self.players)

    @property
    def labels(self) -> tuple[int | None, ...]:
        """Player class ids in player order; ``None`` marks ``rest``."""
        return tuple(player.label for player in self.players)

    @property
    def scene_index(self) -> int | None:
        """Index of the ``scene`` player, or ``None`` if the scene is not one."""
        for index, player in enumerate(self.players):
            if player.is_scene:
                return index
        return None

    @property
    def ingredient_indices(self) -> tuple[int, ...]:
        """Indices of the players that stand for food.

        Every "was one ingredient enough?" statistic must be computed over
        these, not over :attr:`players`, or the scene is counted as an
        ingredient and the headline number is wrong.
        """
        return tuple(i for i, player in enumerate(self.players) if player.is_ingredient)

    def coalition_mask(self, coalition: Iterable[int]) -> NDArray[np.bool_]:
        """Return the pixels kept visible by a coalition.

        Only the selected players' pixels are included. What happens to the
        rest -- blurred, token-dropped, left alone -- is the removal operator's
        decision, not this module's.

        Args:
            coalition: Player indices into :attr:`players`.

        Returns:
            Boolean mask of shape ``(size, size)``.

        Raises:
            IndexError: If a player index is out of range. Silently ignoring
                one would produce a plausible-looking but wrong game.
        """
        keep = np.zeros(self.image.shape[:2], dtype=np.bool_)
        for index in coalition:
            if not 0 <= index < len(self.players):
                msg = f"player index {index} out of range for {len(self.players)} players"
                raise IndexError(msg)
            keep |= self.players[index].mask
        return keep


def build_players(
    example: FoodSegExample,
    names: Mapping[int, str],
    *,
    image_size: int,
    max_players: int,
    min_area_fraction: float = 0.0,
    background_player: bool = False,
) -> PlayerSet:
    """Build the coalition players for one image.

    Steps, in order: apply CLIP's geometry to image and mask together; take the
    ingredient classes still present, excluding background; mark the ones below
    ``min_area_fraction``; mark the smallest remaining ones if there are more
    than ``max_players``; fold everything marked into a single ``rest`` player.

    Two rules keep the result interpretable:

    * ``min_area_fraction`` never merges away every ingredient. If all of them
      are below it, the largest is kept named; a game about nothing but
      ``rest`` answers no research question. ``max_players`` is a deliberate
      bound and does override this, so ``max_players=1`` really does give a
      single ``rest`` player.
    * A single marked ingredient is **not** turned into ``rest``. ``rest``
      would then cover exactly the same pixels under a less informative name,
      and the player count would be identical either way.

    The cap applies to the final count *including* ``rest`` and ``scene``, so a
    player set never has more than ``max_players`` members and
    ``2**len(players)`` never exceeds what the cap promised. The scene takes
    its slot first, leaving ``max_players - 1`` for ingredients: a cap is a
    statement about compute, and the scene costs exactly as much as any other
    player.

    Args:
        example: The source image and its mask.
        names: Class id to ingredient name, from
            :func:`~food_shapley.download.read_id2label`.
        image_size: Side length of the model's square input.
        max_players: Upper bound on the number of players, which bounds the
            ``2**n`` exact computation.
        min_area_fraction: Ingredients covering less than this share of the
            model's input are candidates for merging. ``0.0`` disables it.
        background_player: Whether the non-ingredient pixels become a
            ``scene`` player. See the module docstring. A scene player is
            **not** added when nothing is left over -- an empty player has
            Shapley value exactly zero by construction and would only dilute
            the statistics, the same argument that drops cropped-away
            ingredients.

    Returns:
        The players, the input they index into, and what was lost on the way.

    Raises:
        ValueError: If ``max_players`` is below 1 (below 2 with
            ``background_player``), ``min_area_fraction`` is outside
            ``[0, 1]``, or the image has no ingredient pixels at all.
    """
    floor = 2 if background_player else 1
    if max_players < floor:
        because = " when the scene is a player" if background_player else ""
        msg = f"max_players must be at least {floor}{because}, got {max_players}"
        raise ValueError(msg)
    if not 0.0 <= min_area_fraction <= 1.0:
        msg = f"min_area_fraction must be in [0, 1], got {min_area_fraction}"
        raise ValueError(msg)

    image = resize_image(example.image, image_size)
    mask = resize_mask(example.mask, image_size)

    present = [int(label) for label in np.unique(mask) if label != BACKGROUND_LABEL]
    dropped = tuple(label for label in example.ingredient_labels if label not in set(present))
    if not present:
        msg = (
            f"image {example.index}: no ingredient pixels survive the {image_size}px centre crop; "
            f"the original mask holds {example.ingredient_labels}"
        )
        raise ValueError(msg)

    masks = {label: mask == label for label in present}
    areas = {label: float(masks[label].mean()) for label in present}
    # Descending area, then ascending id, so the ranking is total and does not
    # depend on how numpy happened to order equal-sized segments.
    by_size = sorted(present, key=lambda label: (-areas[label], label))

    keep = [label for label in by_size if areas[label] >= min_area_fraction]
    if not keep:
        # Everything is tiny; the largest still beats being nameless.
        keep = by_size[:1]

    # The cap is on the final player count, which includes `rest`. Comparing
    # it against the named players alone lets `min_area_fraction` push the
    # total one over the bound -- and `2**n` is where that bound earns its
    # keep.
    def total(named: list[int]) -> int:
        return len(named) + (1 if len(named) < len(by_size) else 0)

    # The scene, if there is one, takes its slot before the ingredients do.
    budget = max_players - 1 if background_player else max_players

    capped = total(keep) > budget
    if capped:
        # One slot goes to `rest`. At budget == 1 that leaves no named player
        # at all: a one-player game is degenerate either way, and covering the
        # whole food region is the more honest degenerate case.
        keep = keep[: budget - 1]

    kept = set(keep)
    merged = tuple(label for label in by_size if label not in kept)

    if len(merged) == 1:
        # Renaming one ingredient to "rest" loses its name and gains nothing.
        # The player count is unchanged, so this cannot breach the cap.
        kept.add(merged[0])
        merged = ()

    players = [Player(label=label, name=names[label], mask=masks[label]) for label in sorted(kept)]
    if merged:
        rest = np.zeros_like(mask, dtype=np.bool_)
        for label in merged:
            rest |= masks[label]
        players.append(
            Player(
                label=None,
                name=REST_PLAYER_NAME,
                mask=rest,
                merged_labels=tuple(sorted(merged)),
            )
        )

    covered = np.zeros_like(mask, dtype=np.bool_)
    for player in players:
        covered |= player.mask
    leftover = ~covered

    if background_player and leftover.any():
        players.append(Player(label=BACKGROUND_LABEL, name=SCENE_PLAYER_NAME, mask=leftover))
        # Every pixel now has an owner, which is precisely what makes the grand
        # coalition reproduce the photograph.
        leftover = np.zeros_like(leftover)

    return PlayerSet(
        index=example.index,
        image=image,
        players=tuple(players),
        background=leftover,
        dropped_labels=dropped,
        capped=capped,
    )
