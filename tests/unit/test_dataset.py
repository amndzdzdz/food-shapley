from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from food_shapley.dataset import FoodSegExample, FoodSegSplit, load_split
from food_shapley.download import HF_REVISION, NUM_LABELS, manifest_path, split_dir

# The real split is never read: these tests are about the guards around the
# load, and a stand-in table exercises them without 110 MB of parquet.

NAMES = {0: "background", **{i: f"ingredient-{i}" for i in range(1, NUM_LABELS)}}


class _FakeTable:
    """What `datasets.load_from_disk` is expected to return."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.num_rows = len(rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]


def row(mask: np.ndarray) -> dict[str, Any]:
    height, width = mask.shape
    return {
        "image": Image.new("RGB", (width, height), (1, 2, 3)),
        "label": Image.fromarray(mask.astype(np.uint8)),
    }


def make_split(masks: list[np.ndarray]) -> FoodSegSplit:
    return FoodSegSplit("test", _FakeTable([row(m) for m in masks]), NAMES)


@pytest.fixture
def two_images() -> FoodSegSplit:
    first = np.zeros((8, 10), dtype=np.uint8)
    first[:, 2:5] = 7
    first[:, 5:] = 20
    second = np.zeros((6, 6), dtype=np.uint8)
    second[1:4, 1:4] = 3
    return make_split([first, second])


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


def test_length_follows_the_table(two_images: FoodSegSplit) -> None:
    assert len(two_images) == 2


def test_example_carries_its_row_index(two_images: FoodSegSplit) -> None:
    """The index is the array-task id and the output filename stem."""
    assert two_images[1].index == 1


def test_mask_is_aligned_with_the_image(two_images: FoodSegSplit) -> None:
    example = two_images[0]
    width, height = example.size
    assert example.mask.shape == (height, width)


def test_mask_is_uint8_label_ids(two_images: FoodSegSplit) -> None:
    assert two_images[0].mask.dtype == np.uint8


def test_ingredient_labels_exclude_background(two_images: FoodSegSplit) -> None:
    """`PLAN.md`: label 0 is background, not a player."""
    assert two_images[0].ingredient_labels == (7, 20)


def test_ingredient_labels_are_ascending() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0] = 9
    mask[1] = 2
    mask[2] = 5
    assert make_split([mask])[0].ingredient_labels == (2, 5, 9)


def test_an_all_background_image_has_no_players() -> None:
    assert make_split([np.zeros((4, 4), dtype=np.uint8)])[0].ingredient_labels == ()


def test_negative_indexing_reports_the_resolved_index(two_images: FoodSegSplit) -> None:
    """A record must never carry an index that does not address it."""
    assert two_images[-1].index == 1


def test_out_of_range_index_is_rejected(two_images: FoodSegSplit) -> None:
    with pytest.raises(IndexError, match="out of range"):
        _ = two_images[2]


def test_iteration_is_in_row_order(two_images: FoodSegSplit) -> None:
    assert [example.index for example in two_images] == [0, 1]


def test_a_label_id_outside_the_mapping_is_rejected() -> None:
    """An unknown id would become a player with no ingredient name."""
    mask = np.full((4, 4), NUM_LABELS, dtype=np.uint8)
    with pytest.raises(ValueError, match="mask holds label id 104"):
        _ = make_split([mask])[0]


def test_a_multichannel_mask_is_rejected() -> None:
    table = _FakeTable(
        [{"image": Image.new("RGB", (4, 4)), "label": Image.new("RGB", (4, 4), (1, 1, 1))}]
    )
    with pytest.raises(ValueError, match="expected a single channel"):
        _ = FoodSegSplit("test", table, NAMES)[0]


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def test_names_cover_every_label_id(two_images: FoodSegSplit) -> None:
    assert len(two_images.names) == NUM_LABELS


def test_ingredient_name_looks_up_the_pinned_mapping(two_images: FoodSegSplit) -> None:
    assert two_images.ingredient_name(20) == "ingredient-20"


def test_background_has_no_ingredient_name(two_images: FoodSegSplit) -> None:
    with pytest.raises(ValueError, match="background, not an ingredient"):
        two_images.ingredient_name(0)


def test_unknown_label_id_raises(two_images: FoodSegSplit) -> None:
    with pytest.raises(KeyError):
        two_images.ingredient_name(999)


# --------------------------------------------------------------------------- #
# Loading guards
#
# These run without `datasets`: the guards must fire before the import, so a
# machine with no dataset gets a usable message rather than a parquet error.
# --------------------------------------------------------------------------- #


def test_load_split_without_a_download_points_at_the_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"download --split test"):
        load_split("test", base=tmp_path)


def test_load_split_refuses_a_stale_revision(tmp_path: Path) -> None:
    """Mixing revisions would make two runs incomparable while both looked valid."""
    directory = split_dir("test", base=tmp_path)
    directory.mkdir(parents=True)
    manifest_path("test", base=tmp_path).write_text(
        json.dumps(_manifest_payload(revision="0" * 40))
    )
    with pytest.raises(ValueError, match="re-download rather than mixing revisions"):
        load_split("test", base=tmp_path)


def test_load_split_rejects_an_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown split"):
        load_split("validation", base=tmp_path)


def _manifest_payload(*, revision: str = HF_REVISION) -> dict[str, Any]:
    return {
        "source": "huggingface",
        "repo_id": "EduardoPacheco/FoodSeg103",
        "revision": revision,
        "split": "test",
        "upstream_split": "validation",
        "num_examples": 2135,
        "expected_examples": 2135,
        "license": "Apache-2.0",
        "column_names": ["image", "label", "classes_on_image", "id"],
        "content_sha256": "0" * 64,
        "num_labels": NUM_LABELS,
        "id2label_sha256": "0" * 64,
        "datasets_version": "0.0.0-fake",
        "downloaded_utc": "2026-01-01T00:00:00+00:00",
    }


def test_example_is_immutable() -> None:
    """Records flow through the pipeline; one that can be edited in place is a trap."""
    example = make_split([np.zeros((4, 4), dtype=np.uint8)])[0]
    with pytest.raises(AttributeError):
        example.index = 5  # type: ignore[misc]


def test_example_is_constructible_directly() -> None:
    """Tests and fixtures build examples without a table; keep that cheap."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    example = FoodSegExample(index=0, image=Image.new("RGB", (4, 4)), mask=mask)
    assert example.size == (4, 4)
