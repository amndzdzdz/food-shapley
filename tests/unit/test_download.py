from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from food_shapley.download import (
    BACKGROUND_LABEL,
    EXPECTED_SPLIT_SIZES,
    HF_REPO_ID,
    HF_REVISION,
    ID2LABEL_FILENAME,
    NUM_LABELS,
    DatasetManifest,
    _hash_directory,
    _parse_id2label,
    download_split,
    id2label_path,
    main,
    manifest_path,
    read_id2label,
    read_manifest,
    split_dir,
    verify_split,
)
from food_shapley.paths import DATA_ROOT_ENV

if TYPE_CHECKING:
    from collections.abc import Callable

# --------------------------------------------------------------------------- #
# A stand-in for `datasets`.
#
# The real library is never exercised here: the test suite must not touch the
# network, and `download_split` imports `datasets` lazily inside the function,
# so swapping the module in `sys.modules` is enough. What is under test is this
# module's own logic -- split aliasing, the size guard, the manifest, and the
# integrity check -- not Hugging Face's loader.
# --------------------------------------------------------------------------- #

FAKE_DATASETS_VERSION = "0.0.0-fake"


class _FakeSplit:
    """The object `datasets.load_dataset` is expected to return."""

    def __init__(self, num_rows: int, payload: dict[str, bytes]) -> None:
        self.num_rows = num_rows
        self.column_names = ["image", "label", "classes_on_image", "id"]
        self._payload = payload

    def save_to_disk(self, path: str) -> None:
        directory = Path(path)
        for name, content in self._payload.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


@pytest.fixture
def fake_datasets(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[dict[str, object]]]:
    """Install a fake `datasets` module and return its recorded call log.

    The returned object is the call log itself; the factory shape lets a test
    override the row count before the module is installed.
    """
    calls: list[dict[str, object]] = []

    def install(num_rows: int = EXPECTED_SPLIT_SIZES["test"]) -> list[dict[str, object]]:
        module = types.ModuleType("datasets")

        def load_dataset(repo_id: str, *, split: str, revision: str) -> _FakeSplit:
            calls.append({"repo_id": repo_id, "split": split, "revision": revision})
            return _FakeSplit(
                num_rows,
                {
                    "data-00000-of-00001.arrow": b"arrow payload",
                    "state.json": b'{"_fingerprint": "deadbeef"}',
                },
            )

        # A ModuleType has no declared attributes, so mypy needs the stores to
        # be spelled out rather than assigned through dot-access.
        module.__dict__["load_dataset"] = load_dataset
        module.__dict__["__version__"] = FAKE_DATASETS_VERSION
        monkeypatch.setitem(sys.modules, "datasets", module)
        return calls

    return install


# --------------------------------------------------------------------------- #
# A stand-in for the Hub file download.
#
# `id2label.json` is not part of the parquet shards, so `download_split` fetches
# it with `hf_hub_download`. That is a network call, so it is replaced here for
# every test in the module -- autouse, so that a new test cannot accidentally
# reach the network by forgetting the fixture.
# --------------------------------------------------------------------------- #


def fake_id2label_bytes(num_labels: int = NUM_LABELS) -> bytes:
    """A structurally valid id -> name mapping.

    The names are synthetic: what `_parse_id2label` guards is the shape of the
    mapping, and pinning the real 104 names here would duplicate the dataset,
    not test the code.
    """
    names = {str(BACKGROUND_LABEL): "background"}
    names.update({str(i): f"ingredient-{i}" for i in range(1, num_labels)})
    return json.dumps(names).encode()


@pytest.fixture(autouse=True)
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve `id2label.json` from disk instead of the Hub."""
    import huggingface_hub

    cached = tmp_path / "hub-cache" / ID2LABEL_FILENAME
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(fake_id2label_bytes())

    def hf_hub_download(*, repo_id: str, filename: str, revision: str, repo_type: str) -> str:
        assert (repo_id, filename, revision, repo_type) == (
            HF_REPO_ID,
            ID2LABEL_FILENAME,
            HF_REVISION,
            "dataset",
        )
        return str(cached)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the data root into tmp_path, for the CLI tests that cannot pass `base`."""
    root = tmp_path / "data"
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    return root


# --------------------------------------------------------------------------- #
# Paths and split naming
# --------------------------------------------------------------------------- #


def test_split_dir_rejects_an_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown split 'val'"):
        split_dir("val", base=tmp_path)


def test_split_dir_error_lists_the_known_splits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test, train"):
        split_dir("val", base=tmp_path)


def test_split_dir_uses_the_project_vocabulary(tmp_path: Path) -> None:
    """Directories are named `test`, not the mirror's `validation`."""
    assert split_dir("test", base=tmp_path) == tmp_path / "raw" / "foodseg103" / "test"


def test_split_dir_honours_the_data_root_env(data_root: Path) -> None:
    assert split_dir("test") == data_root / "raw" / "foodseg103" / "test"


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def test_hash_is_independent_of_write_order(tmp_path: Path) -> None:
    """Two directories with the same content must hash alike regardless of creation order."""
    first, second = tmp_path / "a", tmp_path / "b"
    for directory, names in ((first, ("x", "y", "z")), (second, ("z", "y", "x"))):
        directory.mkdir()
        for name in names:
            (directory / name).write_bytes(name.encode())
    assert _hash_directory(first) == _hash_directory(second)


def test_hash_changes_when_content_changes(tmp_path: Path) -> None:
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "x").write_bytes(b"one")
    before = _hash_directory(directory)
    (directory / "x").write_bytes(b"two")
    assert _hash_directory(directory) != before


def test_hash_changes_when_a_file_is_renamed(tmp_path: Path) -> None:
    """Hashing names as well as bytes catches a shard written under the wrong name."""
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "x").write_bytes(b"same")
    before = _hash_directory(directory)
    (directory / "x").rename(directory / "y")
    assert _hash_directory(directory) != before


def test_hash_ignores_the_manifest(tmp_path: Path) -> None:
    """The manifest holds the hash, so it cannot be part of what is hashed."""
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "x").write_bytes(b"payload")
    before = _hash_directory(directory)
    (directory / "manifest.json").write_text("{}")
    assert _hash_directory(directory) == before


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #


def test_download_translates_test_to_the_mirrors_validation(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """The mirror calls the 2,135-image split `validation`; we must ask for that."""
    calls = fake_datasets()
    download_split("test", base=tmp_path)
    assert calls == [{"repo_id": HF_REPO_ID, "split": "validation", "revision": HF_REVISION}]


def test_download_does_not_rename_the_train_split(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    calls = fake_datasets(num_rows=EXPECTED_SPLIT_SIZES["train"])
    download_split("train", base=tmp_path)
    assert calls[0]["split"] == "train"


def test_download_writes_the_payload_and_a_manifest(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    fake_datasets()
    manifest = download_split("test", base=tmp_path)

    directory = split_dir("test", base=tmp_path)
    assert (directory / "data-00000-of-00001.arrow").is_file()
    assert manifest_path("test", base=tmp_path).is_file()
    assert manifest.num_examples == EXPECTED_SPLIT_SIZES["test"]
    assert manifest.upstream_split == "validation"
    assert manifest.revision == HF_REVISION
    assert manifest.datasets_version == FAKE_DATASETS_VERSION


# --------------------------------------------------------------------------- #
# Ingredient names
# --------------------------------------------------------------------------- #


def test_download_stores_the_ingredient_names(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """`load_dataset` never sees id2label.json, so it must be fetched separately."""
    fake_datasets()
    manifest = download_split("test", base=tmp_path)
    assert id2label_path("test", base=tmp_path).is_file()
    assert manifest.num_labels == NUM_LABELS


def test_stored_names_are_byte_identical_to_the_mirror(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """Storing the raw bytes is what makes id2label_sha256 comparable upstream."""
    fake_datasets()
    download_split("test", base=tmp_path)
    assert id2label_path("test", base=tmp_path).read_bytes() == fake_id2label_bytes()


def test_names_are_covered_by_the_content_hash(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """Fetched after save_to_disk, which clears the directory, and before hashing."""
    fake_datasets()
    manifest = download_split("test", base=tmp_path)
    assert manifest.content_sha256 == _hash_directory(split_dir("test", base=tmp_path))


def test_read_id2label_returns_integer_keys(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """Masks hold integers; a mapping keyed by strings would silently never match."""
    fake_datasets()
    download_split("test", base=tmp_path)
    mapping = read_id2label("test", base=tmp_path)
    assert set(mapping) == set(range(NUM_LABELS))


def test_id2label_includes_background_as_a_non_ingredient(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """Label 0 is in the mapping but is never a coalition player (PLAN.md)."""
    fake_datasets()
    download_split("test", base=tmp_path)
    assert BACKGROUND_LABEL in read_id2label("test", base=tmp_path)


def test_read_id2label_points_at_the_download_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"food_shapley\.download --split test"):
        read_id2label("test", base=tmp_path)


def test_parse_rejects_a_truncated_mapping() -> None:
    """A mapping short of 104 entries would leave ingredients unnamed."""
    with pytest.raises(ValueError, match="missing"):
        _parse_id2label(fake_id2label_bytes(num_labels=NUM_LABELS - 1))


def test_parse_rejects_a_shifted_mapping() -> None:
    """The failure no downstream check could catch: every ingredient renamed."""
    shifted = {str(i + 1): f"ingredient-{i}" for i in range(NUM_LABELS)}
    with pytest.raises(ValueError, match=r"missing \[0\], unexpected \[104\]"):
        _parse_id2label(json.dumps(shifted).encode())


def test_parse_rejects_non_integer_keys() -> None:
    with pytest.raises(ValueError, match="integer-valued strings"):
        _parse_id2label(b'{"background": "background"}')


def test_parse_rejects_an_empty_name() -> None:
    """An unnamed ingredient would appear in a figure as a blank label."""
    names = json.loads(fake_id2label_bytes())
    names["7"] = "  "
    with pytest.raises(ValueError, match=r"ids \[7\]"):
        _parse_id2label(json.dumps(names).encode())


def test_parse_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_id2label(b"{")


def test_verify_detects_changed_ingredient_names(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """Reported as a name change, not as a generic corrupt directory."""
    fake_datasets()
    download_split("test", base=tmp_path)
    names = json.loads(fake_id2label_bytes())
    names["1"] = "renamed"
    id2label_path("test", base=tmp_path).write_bytes(json.dumps(names).encode())

    problems = verify_split("test", base=tmp_path)
    assert any("ingredient names changed" in problem for problem in problems)


def test_verify_detects_deleted_ingredient_names(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    fake_datasets()
    download_split("test", base=tmp_path)
    id2label_path("test", base=tmp_path).unlink()
    assert any("not found" in problem for problem in verify_split("test", base=tmp_path))


def test_verify_reports_an_old_manifest_as_re_downloadable(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """A manifest from before the names were pinned must not raise inside a job."""
    fake_datasets()
    download_split("test", base=tmp_path)
    payload = json.loads(manifest_path("test", base=tmp_path).read_text())
    del payload["num_labels"]
    manifest_path("test", base=tmp_path).write_text(json.dumps(payload))

    problems = verify_split("test", base=tmp_path)
    assert len(problems) == 1
    assert "re-download" in problems[0]


def test_manifest_hash_matches_what_was_written(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    fake_datasets()
    manifest = download_split("test", base=tmp_path)
    assert manifest.content_sha256 == _hash_directory(split_dir("test", base=tmp_path))


def test_manifest_round_trips_through_disk(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    fake_datasets()
    written = download_split("test", base=tmp_path)
    assert read_manifest("test", base=tmp_path) == written


def test_download_refuses_a_mirror_with_the_wrong_size(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """A changed mirror must not be silently recorded as FoodSeg103."""
    fake_datasets(num_rows=EXPECTED_SPLIT_SIZES["test"] - 1)
    with pytest.raises(RuntimeError, match="but the FoodSeg103 paper"):
        download_split("test", base=tmp_path)


def test_a_rejected_download_leaves_no_manifest(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """Half a dataset with no manifest is recoverable; one with a manifest is a trap."""
    fake_datasets(num_rows=1)
    with pytest.raises(RuntimeError):
        download_split("test", base=tmp_path)
    assert read_manifest("test", base=tmp_path) is None


def test_download_is_idempotent(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """Re-running after a failed cluster job must not re-fetch gigabytes."""
    calls = fake_datasets()
    download_split("test", base=tmp_path)
    download_split("test", base=tmp_path)
    assert len(calls) == 1


def test_force_re_downloads(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    calls = fake_datasets()
    download_split("test", base=tmp_path)
    download_split("test", base=tmp_path, force=True)
    assert len(calls) == 2


def test_download_repairs_a_corrupt_copy(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """A copy that fails verification is replaced, not reused."""
    calls = fake_datasets()
    download_split("test", base=tmp_path)
    (split_dir("test", base=tmp_path) / "data-00000-of-00001.arrow").write_bytes(b"truncated")

    download_split("test", base=tmp_path)
    assert len(calls) == 2
    assert verify_split("test", base=tmp_path) == []


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def test_verify_reports_a_missing_download(tmp_path: Path) -> None:
    problems = verify_split("test", base=tmp_path)
    assert len(problems) == 1
    assert "not downloaded" in problems[0]


def test_verify_passes_for_a_sound_copy(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    fake_datasets()
    download_split("test", base=tmp_path)
    assert verify_split("test", base=tmp_path) == []


def test_verify_detects_modified_content(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """The failure mode that otherwise surfaces much later as a model error."""
    fake_datasets()
    download_split("test", base=tmp_path)
    (split_dir("test", base=tmp_path) / "data-00000-of-00001.arrow").write_bytes(b"tampered")

    problems = verify_split("test", base=tmp_path)
    assert len(problems) == 1
    assert "does not match" in problems[0]


def test_verify_detects_a_deleted_file(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    fake_datasets()
    download_split("test", base=tmp_path)
    (split_dir("test", base=tmp_path) / "state.json").unlink()
    assert verify_split("test", base=tmp_path) != []


def test_verify_detects_a_stale_revision(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """Bumping the pinned revision in code must invalidate the copy on disk."""
    fake_datasets()
    manifest = download_split("test", base=tmp_path)
    stale = replace(manifest, revision="0" * 40)
    _rewrite_manifest(tmp_path, stale)

    problems = verify_split("test", base=tmp_path)
    assert len(problems) == 1
    assert "re-download" in problems[0]


def test_verify_detects_a_wrong_example_count(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    fake_datasets()
    manifest = download_split("test", base=tmp_path)
    wrong = replace(manifest, num_examples=17)
    _rewrite_manifest(tmp_path, wrong)

    problems = verify_split("test", base=tmp_path)
    assert len(problems) == 1
    assert "expected 2135" in problems[0]


def test_verify_reports_every_problem_at_once(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """One run of `--verify-only` should show the whole picture, not the first fault."""
    fake_datasets()
    manifest = download_split("test", base=tmp_path)
    _rewrite_manifest(tmp_path, replace(manifest, revision="0" * 40, num_examples=17))
    (split_dir("test", base=tmp_path) / "state.json").write_bytes(b"tampered")

    assert len(verify_split("test", base=tmp_path)) == 3


def _rewrite_manifest(base: Path, manifest: DatasetManifest) -> None:
    manifest_path(manifest.split, base=base).write_text(json.dumps(manifest.to_dict()))


def test_read_manifest_rejects_unparseable_json(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    fake_datasets()
    download_split("test", base=tmp_path)
    manifest_path("test", base=tmp_path).write_text("not json")
    with pytest.raises(ValueError, match="unreadable or has an unexpected shape"):
        read_manifest("test", base=tmp_path)


def test_read_manifest_rejects_an_unexpected_shape(
    fake_datasets: Callable[..., list[dict[str, object]]], tmp_path: Path
) -> None:
    """An older manifest missing a field must fail loudly, not load half-populated."""
    fake_datasets()
    download_split("test", base=tmp_path)
    manifest_path("test", base=tmp_path).write_text('{"split": "test"}')
    with pytest.raises(ValueError, match="unreadable or has an unexpected shape"):
        read_manifest("test", base=tmp_path)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_cli_downloads_then_verifies(
    fake_datasets: Callable[..., list[dict[str, object]]],
    data_root: Path,  # noqa: ARG001
) -> None:
    fake_datasets()
    assert main(["--split", "test"]) == 0
    assert main(["--split", "test", "--verify-only"]) == 0


def test_cli_verify_only_fetches_nothing(
    fake_datasets: Callable[..., list[dict[str, object]]],
    data_root: Path,  # noqa: ARG001
) -> None:
    calls = fake_datasets()
    assert main(["--split", "test", "--verify-only"]) == 1
    assert calls == []


def test_cli_verify_only_fails_on_a_corrupt_copy(
    fake_datasets: Callable[..., list[dict[str, object]]],
    data_root: Path,  # noqa: ARG001
) -> None:
    fake_datasets()
    main(["--split", "test"])
    (split_dir("test") / "state.json").write_bytes(b"tampered")
    assert main(["--split", "test", "--verify-only"]) == 1


def test_cli_rejects_an_unknown_split(
    data_root: Path,  # noqa: ARG001
) -> None:
    with pytest.raises(SystemExit):
        main(["--split", "validation"])
