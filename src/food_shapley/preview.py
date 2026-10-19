"""Render coalitions as contact sheets, for the visual check.

`PLAN.md` Milestone 1 demands two checks before the full run, and this module
exists for the second of them. `docs/week-01.md` states it as: "Render a handful
of masked coalitions and confirm that 'only bun visible' really shows only the
bun. This catches mask/image misalignment, which the efficiency check cannot."

Why a human has to look
-----------------------
The efficiency check -- ``sum(SV) == v(N) - v(empty)`` -- is a statement about
the *game*, and a game built on mis-registered masks is perfectly
self-consistent. Shift every mask ten pixels to the right and efficiency still
holds to machine precision while every ingredient attribution is wrong. The
unit tests close part of the gap (``tests/unit/test_segments.py`` asserts
byte-equality with torchvision's transforms, ``tests/unit/test_masking.py``
asserts a sharp region keeps its own stripe's colour), but both are synthetic:
they prove the geometry is self-consistent, not that label 31 on a real
photograph is actually the bun. Only looking does that.

What a sheet contains
---------------------
One row of panels per image: the model's input, the empty coalition, each
singleton, and the grand coalition. Singletons are the informative ones --
"only X visible" is exactly the claim being checked -- while the two extremes
are the endpoints every Shapley value is measured between.

This writes a diagnostic, not a run. It loads no model and computes no value,
so it produces nothing that could be mistaken for a result, and it is therefore
allowed to overwrite its own output directory. Run directories are still never
overwritten; see :mod:`food_shapley.paths`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from food_shapley.config import config_fingerprint, load_config
from food_shapley.dataset import load_split
from food_shapley.masking import BlurRemoval
from food_shapley.paths import artifact_root
from food_shapley.seeding import derive_seed
from food_shapley.segments import build_players

if TYPE_CHECKING:
    from collections.abc import Sequence

    from food_shapley.config import ExperimentConfig
    from food_shapley.dataset import FoodSegSplit
    from food_shapley.segments import PlayerSet

__all__ = ["contact_sheet", "main", "select_indices"]

#: Height of the caption strip under each panel, in pixels.
CAPTION_HEIGHT = 34
#: Gap between panels, in pixels.
PADDING = 6
#: Sheet background; mid-grey so that both blurred food and captions read.
SHEET_COLOUR = (32, 32, 32)
CAPTION_COLOUR = (235, 235, 235)


def select_indices(
    config: ExperimentConfig, split_size: int, count: int, explicit: Sequence[int]
) -> list[int]:
    """Choose which images to render.

    Explicit indices win, so a suspicious image found during the check can be
    revisited exactly. Otherwise the sample is drawn from a seed derived from
    the configuration, so "the images I looked at" is reproducible from the
    config alone rather than being whatever the run happened to pick.

    Args:
        config: The run configuration, for its seed.
        split_size: Number of images in the split.
        count: How many to sample when no explicit indices are given.
        explicit: Indices requested on the command line.

    Returns:
        Image indices, ascending.

    Raises:
        ValueError: If an explicit index is out of range, or ``count`` is not
            positive.
    """
    if explicit:
        out_of_range = sorted(i for i in explicit if not 0 <= i < split_size)
        if out_of_range:
            msg = f"image indices out of range for a split of {split_size}: {out_of_range}"
            raise ValueError(msg)
        return sorted(set(explicit))
    if count < 1:
        msg = f"number of images must be at least 1, got {count}"
        raise ValueError(msg)
    rng = np.random.default_rng(derive_seed(config.seed, "preview"))
    drawn = rng.choice(split_size, size=min(count, split_size), replace=False)
    return sorted(int(index) for index in drawn)


def _caption(panel: Image.Image, text: str) -> Image.Image:
    """Return ``panel`` with a caption strip underneath."""
    width, height = panel.size
    out = Image.new("RGB", (width, height + CAPTION_HEIGHT), SHEET_COLOUR)
    out.paste(panel, (0, 0))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    # Two short lines beat one clipped line: ingredient names such as
    # "other ingredients" do not fit across a 224 px panel.
    head, _, tail = text.partition("\n")
    draw.text((4, height + 4), head[:46], fill=CAPTION_COLOUR, font=font)
    if tail:
        draw.text((4, height + 18), tail[:46], fill=CAPTION_COLOUR, font=font)
    return out


def _panels(players: PlayerSet, operator: BlurRemoval) -> list[tuple[str, Image.Image]]:
    """Build the captioned panels for one image, in reading order."""
    full = tuple(range(len(players)))
    coalitions: list[tuple[str, tuple[int, ...]]] = [("none (v of empty set)", ())]
    coalitions += [(f"only {players.players[i].name}", (i,)) for i in full]
    if players.scene_index is not None:
        # The comparison that motivated making the scene a player: how much of
        # the dish survives when only the scene is removed, and how much when
        # only the scene remains. Both are singletons of the complement, and
        # both are easy to misjudge from the per-player panels alone.
        coalitions.append(("every ingredient, no scene", players.ingredient_indices))
    if len(players) > 1:
        coalitions.append(("all (v of N)", full))

    panels = [("original input", Image.fromarray(players.image))]
    panels += [
        (label, Image.fromarray(operator.render(players, coalition)))
        for label, coalition in coalitions
    ]
    return panels


def contact_sheet(
    players: PlayerSet, operator: BlurRemoval, *, columns: int = 6, scale: int = 1
) -> Image.Image:
    """Render one image's coalitions as a single labelled sheet.

    Args:
        players: The player set to render.
        operator: The removal operator under test.
        columns: Panels per row.
        scale: Integer magnification of each panel. The model's input is only
            224 px, which is too small to judge a segment boundary on screen.
            Magnification is nearest-neighbour and applied *after* rendering,
            so it enlarges the pixels the model actually sees rather than
            inventing smoother ones -- the check would be worthless if the
            viewer interpolated away a one-pixel misalignment.

    Returns:
        An RGB sheet.

    Raises:
        ValueError: If ``columns`` or ``scale`` is not positive.
    """
    if columns < 1:
        msg = f"columns must be at least 1, got {columns}"
        raise ValueError(msg)
    if scale < 1:
        msg = f"scale must be at least 1, got {scale}"
        raise ValueError(msg)

    captioned = [
        _caption(
            panel.resize((panel.width * scale, panel.height * scale), Image.Resampling.NEAREST),
            label,
        )
        for label, panel in _panels(players, operator)
    ]
    cell_width, cell_height = captioned[0].size
    rows = -(-len(captioned) // columns)

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * PADDING,
            rows * cell_height + (rows + 1) * PADDING,
        ),
        SHEET_COLOUR,
    )
    for position, panel in enumerate(captioned):
        row, column = divmod(position, columns)
        sheet.paste(
            panel,
            (PADDING + column * (cell_width + PADDING), PADDING + row * (cell_height + PADDING)),
        )
    return sheet


def render_split(
    config: ExperimentConfig,
    split: FoodSegSplit,
    indices: Sequence[int],
    destination: Path,
    *,
    columns: int = 6,
    scale: int = 1,
) -> list[tuple[Path, PlayerSet]]:
    """Write one contact sheet per requested image.

    Args:
        config: Settings; the geometry and the operator both come from it, so
            the sheets show what the run would actually feed the model.
        split: The loaded split.
        indices: Image indices to render.
        destination: Output directory; created if missing.
        columns: Panels per row.
        scale: Integer magnification of each panel.

    Returns:
        One ``(path, players)`` pair per rendered image.
    """
    operator = BlurRemoval(radius=config.blur_radius, keep_background=config.background_stays_sharp)
    destination.mkdir(parents=True, exist_ok=True)

    written: list[tuple[Path, PlayerSet]] = []
    for index in indices:
        players = build_players(
            split[index],
            split.names,
            image_size=config.image_size,
            max_players=config.max_players,
            min_area_fraction=config.min_area_fraction,
            background_player=config.background_is_player,
        )
        path = destination / f"image-{index:05d}.png"
        contact_sheet(players, operator, columns=columns, scale=scale).save(path)
        written.append((path, players))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Returns:
        Process exit code: 0 on success, 1 if the sheets could not be written.
    """
    parser = argparse.ArgumentParser(
        prog="python -m food_shapley.preview",
        description="Render masked coalitions for the Week-1 visual check.",
    )
    parser.add_argument("--config", type=Path, required=True, help="experiment TOML")
    parser.add_argument(
        "--index",
        type=int,
        action="append",
        default=[],
        help="image index to render; repeatable. Overrides --num-images.",
    )
    parser.add_argument(
        "--num-images", type=int, default=3, help="images to sample when no --index is given"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory; defaults to <artifact root>/visual-check/<fingerprint>",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="nearest-neighbour magnification of each panel (default: 2). "
        "224 px is too small to judge a segment boundary on screen.",
    )
    parser.add_argument("--columns", type=int, default=4, help="panels per row (default: 4)")
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_viewer",
        help="open each sheet in the system image viewer when it is written",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    fingerprint = config_fingerprint(config)
    destination = args.out or artifact_root() / "visual-check" / fingerprint

    try:
        split = load_split(config.split)
    except (FileNotFoundError, ValueError) as error:
        print(f"cannot render: {error}", file=sys.stderr)
        return 1

    indices = select_indices(config, len(split), args.num_images, args.index)
    written = render_split(
        config, split, indices, destination, columns=args.columns, scale=args.scale
    )

    print(f"config {args.config} (fingerprint {fingerprint})")
    print(f"operator blur radius={config.blur_radius} background={config.background_mode}")
    for path, players in written:
        names = ", ".join(players.names)
        print(f"{path}  {len(players)} players: {names}")
        if players.dropped_labels:
            dropped = ", ".join(split.ingredient_name(i) for i in players.dropped_labels)
            print(f"    dropped by the centre crop: {dropped}")
        if args.open_viewer:
            Image.open(path).show(title=path.name)
    print(f"\n{len(written)} sheet(s) written. Now look at them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
