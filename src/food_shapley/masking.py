"""Removal operators: rendering a coalition as an image.

`PLAN.md` Milestone 1, step 3, first half: "hide segments by blurring". A
removal operator answers the question the value function asks of it -- *what
does the image look like when only the players in ``S`` are present?* -- and
every Shapley value is a difference between two of its outputs. It is
therefore part of the claim, not an implementation detail: "ingredient X
contributes 0.4" is always shorthand for "contributes 0.4 **under this
operator**".

Week 2 adds a second operator (token dropping inside the ViT) precisely
because that qualifier matters; `PLAN.md` treats agreement between the two as
the robustness check on Milestone 1's result.

How the blur operator works
---------------------------
The whole image is blurred once, and the pixels of the visible players are
pasted back sharp. Blurring each hidden region in isolation instead would pull
black in from outside its bounding box and leave halos that have nothing to do
with the ingredient.

Two consequences, both limitations rather than bugs:

* **Silhouettes survive.** Pasting sharp pixels into a blurred image leaves a
  crisp boundary exactly on the segment outline, so the *shape* of a visible
  ingredient is preserved perfectly, and the shape of a hidden one is still
  hinted at by the blur discontinuity. `PLAN.md` lists "blur does not actually
  remove information" as a known risk; this is the mechanism.
* **Colour bleeds.** A blurred pixel near a visible carrot carries some carrot
  colour, so hidden regions are not perfectly uninformative.

The background
--------------
This operator only decides what happens to pixels that belong to **no player**.
Whether the plate and table are such pixels is
:mod:`food_shapley.segments`'s decision, driven by
:data:`~food_shapley.config.BACKGROUND_MODES`:

``background_mode = "player"`` (the Week-1 setting)
    The scene is a player, so there are no unowned pixels at all and
    ``keep_background`` has nothing to act on. The grand coalition reproduces
    the photograph exactly and the empty coalition is a fully blurred frame.
``background_mode = "visible"``
    ``keep_background=True``: unowned pixels stay sharp in every coalition, so
    they cancel out of every Shapley *difference* while still propping up
    ``v(S)`` itself.
``background_mode = "hidden"``
    ``keep_background=False``: unowned pixels are blurred everywhere, so even
    ``v(N)`` is not the photograph.

Keeping the operator ignorant of which mode is in force is deliberate: it
answers one question -- "what do the non-players look like?" -- and the choice
of who counts as a player is made, and recorded, elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from food_shapley.segments import PlayerSet

__all__ = ["BlurRemoval"]


@dataclass(frozen=True, slots=True)
class BlurRemoval:
    """Hide players by blurring everything they do not cover.

    Attributes:
        radius: Gaussian blur radius in pixels, as PIL defines it. Named
            ``radius`` rather than ``sigma`` because PIL's filter is a
            box-approximated Gaussian; calling it sigma would claim a
            precision the implementation does not have.
        keep_background: Whether pixels owned by no player stay sharp. A no-op
            when the scene is itself a player, because then there are none.
            See the module docstring.
    """

    radius: float
    keep_background: bool

    def __post_init__(self) -> None:
        """Validate the operator.

        Raises:
            ValueError: If the radius is not positive. A radius of zero makes
                every coalition identical to the original image, so every
                Shapley value would be zero and the efficiency check would
                still pass.
        """
        if self.radius <= 0:
            msg = f"blur radius must be positive, got {self.radius}"
            raise ValueError(msg)

    def render(
        self,
        players: PlayerSet,
        coalition: Iterable[int],
        *,
        blurred: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """Render the image as the model should see it for one coalition.

        Args:
            players: The player set, which carries the aligned image.
            coalition: Player indices that stay visible.
            blurred: The output of :meth:`blurred` for *this* player set, to
                avoid re-blurring. A caller rendering the whole powerset would
                otherwise recompute one identical Gaussian blur up to 2,048
                times. Optional purely for speed: the rendering is
                byte-identical either way, which is asserted in the tests.

        Returns:
            RGB array with the same shape as ``players.image``.

        Raises:
            ValueError: If ``blurred`` does not match the image's shape, which
                would mean it came from a different image and would silently
                composite one photograph's sharp pixels onto another's blur.
        """
        keep = players.coalition_mask(coalition)
        if self.keep_background:
            keep = keep | players.background
        if blurred is None:
            blurred = self.blurred(players)
        elif blurred.shape != players.image.shape:
            msg = (
                f"blurred frame has shape {blurred.shape}, expected "
                f"{players.image.shape}; it belongs to a different image"
            )
            raise ValueError(msg)
        return np.where(keep[..., None], players.image, blurred)

    def blurred(self, players: PlayerSet) -> NDArray[np.uint8]:
        """Return the fully blurred image, i.e. the operator's baseline.

        Exposed so that a caller rendering many coalitions of one image can
        reuse it; blurring is the expensive part of :meth:`render`.
        """
        filtered = Image.fromarray(players.image).filter(ImageFilter.GaussianBlur(self.radius))
        return np.asarray(filtered, dtype=np.uint8)
