"""FoodSeg103 loading.

`PLAN.md` Milestone 1, step 1, first half: read the downloaded split and hand
out one typed record per image. Turning those records into coalition players is
:mod:`food_shapley.segments`; this module does nothing but load, and makes no
decision that affects a Shapley value.

What a record contains
----------------------
An RGB photograph and a same-sized single-channel mask whose pixel values are
ingredient label ids in ``0..103``. **Id 0 is background, not an ingredient**
(:data:`~food_shapley.download.BACKGROUND_LABEL`); `PLAN.md` flags treating it
as a player as an easy mistake, so :attr:`FoodSegExample.ingredient_labels`
excludes it and nothing here offers an alternative that does not.

The mirror also ships a ``classes_on_image`` column. It is not used: it was
checked against ``numpy.unique(mask)`` on all 2,135 test images and agreed
everywhere, so it adds no information, and the mask is the thing the pipeline
actually masks with. Deriving from the mask means a record cannot claim an
ingredient it does not show.

Verification
------------
Loading checks that a manifest exists and was written at the revision the code
currently pins. It deliberately does **not** re-hash the ~110 MB of content on
every load: that check lives in ``download --verify-only`` and in
:mod:`food_shapley.envcheck`, which ``slurm/prepare.sh`` and every job script
run once up front.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from food_shapley.download import (
    BACKGROUND_LABEL,
    HF_REVISION,
    NUM_LABELS,
    manifest_path,
    read_id2label,
    read_manifest,
    split_dir,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image

__all__ = ["FoodSegExample", "FoodSegSplit", "load_split"]


@dataclass(frozen=True, slots=True)
class FoodSegExample:
    """One FoodSeg103 image and its ingredient segmentation.

    Attributes:
        index: Row index within the split. Stable for a pinned revision, and
            used as the array-task index and output filename stem.
        image: The photograph, RGB, at its original resolution.
        mask: Label ids, shape ``(height, width)``, dtype ``uint8``, aligned
            pixel-for-pixel with ``image``.
    """

    index: int
    image: Image.Image
    mask: NDArray[np.uint8]

    @property
    def size(self) -> tuple[int, int]:
        """Image size as ``(width, height)``, matching PIL's convention."""
        return self.image.size

    @property
    def ingredient_labels(self) -> tuple[int, ...]:
        """Ingredient ids present, ascending, **excluding background**.

        This is the candidate player set before merging and capping; see
        :mod:`food_shapley.segments`.
        """
        present = np.unique(self.mask)
        return tuple(int(label) for label in present if label != BACKGROUND_LABEL)


class FoodSegSplit:
    """A downloaded FoodSeg103 split, with its ingredient names.

    Indexing is by row, so ``split[7]`` is the eighth image and is the same
    image on every machine holding the pinned revision. Decoding is lazy: the
    JPEG and PNG for a row are decoded on access, not at construction.
    """

    def __init__(self, name: str, table: Any, names: Mapping[int, str]) -> None:  # noqa: ANN401
        # `table` is a datasets.Dataset. It is typed as Any because `datasets`
        # ships no py.typed marker, and letting its types leak into this
        # module's public signatures would spread that gap through the package.
        self._name = name
        self._table = table
        self._names = dict(names)

    @property
    def name(self) -> str:
        """Split name in this project's vocabulary (``train`` / ``test``)."""
        return self._name

    @property
    def names(self) -> Mapping[int, str]:
        """Label id to ingredient name, all :data:`NUM_LABELS` of them.

        Includes id 0 (``background``), which is not an ingredient.
        """
        return self._names

    def ingredient_name(self, label: int) -> str:
        """Return the ingredient name for a label id.

        Raises:
            KeyError: If the id is outside ``0..NUM_LABELS-1``.
            ValueError: If the id is background, which has no ingredient name.
        """
        if label == BACKGROUND_LABEL:
            msg = f"label {BACKGROUND_LABEL} is background, not an ingredient"
            raise ValueError(msg)
        return self._names[label]

    def __len__(self) -> int:
        """Number of images in the split."""
        return int(self._table.num_rows)

    def __getitem__(self, index: int) -> FoodSegExample:
        """Return one example.

        Raises:
            IndexError: If ``index`` is out of range.
            ValueError: If the mask holds a label id outside ``0..103``.
        """
        if not -len(self) <= index < len(self):
            msg = f"index {index} out of range for split {self._name!r} of {len(self)} images"
            raise IndexError(msg)
        position = index % len(self)
        row = self._table[position]
        return FoodSegExample(
            index=position,
            image=row["image"].convert("RGB"),
            mask=_as_mask(row["label"], position=position),
        )

    def __iter__(self) -> Iterator[FoodSegExample]:
        """Iterate over the split in row order."""
        for index in range(len(self)):
            yield self[index]


def _as_mask(label_image: Image.Image, *, position: int) -> NDArray[np.uint8]:
    """Convert a decoded label PNG to a validated label array.

    Raises:
        ValueError: If the mask is not single-channel or holds an unknown id.
            An out-of-range id would otherwise become a player with no name.
    """
    array = np.asarray(label_image)
    if array.ndim != 2:
        msg = f"image {position}: mask has shape {array.shape}, expected a single channel"
        raise ValueError(msg)
    maximum = int(array.max(initial=0))
    if maximum >= NUM_LABELS:
        msg = f"image {position}: mask holds label id {maximum}, but ids run 0..{NUM_LABELS - 1}"
        raise ValueError(msg)
    return array.astype(np.uint8, copy=False)


def load_split(split: str, *, base: Path | None = None) -> FoodSegSplit:
    """Load a downloaded FoodSeg103 split.

    Args:
        split: ``"train"`` or ``"test"``.
        base: Override for the data root; defaults to
            :func:`~food_shapley.paths.data_root`.

    Returns:
        The split, with its ingredient names attached.

    Raises:
        FileNotFoundError: If the split has not been downloaded.
        ValueError: If the manifest is malformed, or was written at a different
            mirror revision than the code now pins. A silent revision mismatch
            would make two runs incomparable while both looked valid.
    """
    directory = split_dir(split, base=base)
    manifest = read_manifest(split, base=base)
    if manifest is None:
        msg = (
            f"no manifest at {manifest_path(split, base=base)}; "
            f"run `python -m food_shapley.download --split {split}` first"
        )
        raise FileNotFoundError(msg)
    if manifest.revision != HF_REVISION:
        msg = (
            f"{split}: on disk from revision {manifest.revision}, but the code pins "
            f"{HF_REVISION}; re-download rather than mixing revisions"
        )
        raise ValueError(msg)

    # Lazy, as in `download`: `datasets` has a large import cost, and the rest
    # of the package must stay importable without paying it.
    import datasets

    table = datasets.load_from_disk(str(directory))
    return FoodSegSplit(split, table, read_id2label(split, base=base))
