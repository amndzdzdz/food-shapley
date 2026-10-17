"""FoodSeg103 acquisition.

`PLAN.md` Milestone 1, step 0: get FoodSeg103 onto the cluster. This module
materialises the dataset into ``data/raw/`` and records exactly what was
obtained, so that a later run can be traced to a specific copy of the data
rather than to "FoodSeg103" in the abstract.

Source, and why
---------------
The authoritative distribution is the authors' password-protected zip (see
``docs/datasets.md``). At the time of writing that URL returns HTTP 502, so
this module uses the Hugging Face mirror instead. The mirror's split sizes
match the paper exactly (4,983 train / 2,135 test = 7,118), and it carries the
same Apache-2.0 license, but it is a third-party re-upload and has not been
byte-compared against the original. The manifest records which source was
used, so the comparison can be made later.

The mirror is pinned to a specific commit: a mirror can be updated or deleted
under us, and "we used FoodSeg103" is not a reproducible statement without a
revision.

Naming
------
The mirror calls the 2,135-image split ``validation``; the paper and the
official release call it ``test``. This module speaks in the paper's terms and
translates at the boundary, so that configurations say ``split = "test"``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from food_shapley.paths import data_root

__all__ = [
    "DATASET_DIRNAME",
    "EXPECTED_SPLIT_SIZES",
    "HF_REPO_ID",
    "HF_REVISION",
    "DatasetManifest",
    "download_split",
    "main",
    "split_dir",
    "verify_split",
]

#: Hugging Face mirror of FoodSeg103.
HF_REPO_ID: Final = "EduardoPacheco/FoodSeg103"

#: Pinned mirror revision (lastModified 2024-04-29). Bumping this is a change
#: to the input data and must be treated as such: new fingerprint, new run.
HF_REVISION: Final = "34e1208e14bc3595d544fc8c3f3c6673253fd9ef"

#: License of both the original dataset and the mirror.
LICENSE: Final = "Apache-2.0"

#: Directory under ``data/raw/`` holding the dataset.
DATASET_DIRNAME: Final = "foodseg103"

#: Split names in this project's vocabulary mapped to the mirror's names.
_SPLIT_ALIASES: Final[dict[str, str]] = {"train": "train", "test": "validation"}

#: Example counts from the FoodSeg103 paper; used as a download integrity check.
EXPECTED_SPLIT_SIZES: Final[dict[str, int]] = {"train": 4983, "test": 2135}


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Record of one downloaded split.

    Attributes:
        source: Where the data came from, e.g. ``"huggingface"``.
        repo_id: Mirror repository id.
        revision: Pinned commit of the mirror.
        split: Split name in this project's vocabulary (``train`` / ``test``).
        upstream_split: Split name used by the mirror.
        num_examples: Number of examples actually written.
        expected_examples: Number of examples the paper reports.
        license: License string.
        column_names: Columns present in the saved split.
        content_sha256: Hash over the saved files, for integrity checking.
        datasets_version: Version of ``datasets`` that wrote the files.
        downloaded_utc: ISO-8601 timestamp of the download.
    """

    source: str
    repo_id: str
    revision: str
    split: str
    upstream_split: str
    num_examples: int
    expected_examples: int
    license: str
    column_names: list[str]
    content_sha256: str
    datasets_version: str
    downloaded_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-serialisable dictionary."""
        return asdict(self)


def split_dir(split: str, *, base: Path | None = None) -> Path:
    """Return the directory holding a downloaded split.

    Args:
        split: ``"train"`` or ``"test"``.
        base: Override for the data root; defaults to :func:`data_root`.

    Returns:
        Path to the split directory. It may not exist yet.

    Raises:
        ValueError: If ``split`` is not a known split name.
    """
    if split not in _SPLIT_ALIASES:
        known = ", ".join(sorted(_SPLIT_ALIASES))
        msg = f"unknown split {split!r}; expected one of: {known}"
        raise ValueError(msg)
    root = data_root() if base is None else base
    return root / "raw" / DATASET_DIRNAME / split


def manifest_path(split: str, *, base: Path | None = None) -> Path:
    """Return the path of a split's manifest file."""
    return split_dir(split, base=base) / "manifest.json"


def _hash_directory(directory: Path) -> str:
    """Hash a directory's contents, independent of traversal order.

    Only the dataset payload is hashed; the manifest itself is skipped, since
    it contains the hash.
    """
    digest = hashlib.sha256()
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json")
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_manifest(split: str, *, base: Path | None = None) -> DatasetManifest | None:
    """Read a split's manifest, or return ``None`` if it has not been downloaded.

    Raises:
        ValueError: If the manifest exists but cannot be parsed.
    """
    path = manifest_path(split, base=base)
    if not path.is_file():
        return None
    try:
        payload: dict[str, Any] = json.loads(path.read_text())
        return DatasetManifest(**payload)
    except (TypeError, ValueError) as error:
        msg = f"{path}: manifest is unreadable or has an unexpected shape: {error}"
        raise ValueError(msg) from error


def verify_split(split: str, *, base: Path | None = None) -> list[str]:
    """Check a downloaded split against its manifest.

    Catches a truncated or partially written download, which otherwise shows up
    much later as a confusing model error.

    Args:
        split: ``"train"`` or ``"test"``.
        base: Override for the data root.

    Returns:
        A list of problems; empty means the split is sound.
    """
    directory = split_dir(split, base=base)
    manifest = read_manifest(split, base=base)
    if manifest is None:
        return [f"{split}: not downloaded (no manifest at {manifest_path(split, base=base)})"]

    problems: list[str] = []
    if manifest.revision != HF_REVISION:
        problems.append(
            f"{split}: downloaded from revision {manifest.revision}, "
            f"but the code now pins {HF_REVISION}; re-download"
        )
    if manifest.num_examples != manifest.expected_examples:
        got, want = manifest.num_examples, manifest.expected_examples
        problems.append(f"{split}: has {got} examples, expected {want}")
    actual_hash = _hash_directory(directory)
    if actual_hash != manifest.content_sha256:
        problems.append(
            f"{split}: content hash {actual_hash[:12]} does not match "
            f"manifest {manifest.content_sha256[:12]}; the copy is corrupt or was modified"
        )
    return problems


def download_split(split: str, *, base: Path | None = None, force: bool = False) -> DatasetManifest:
    """Download one split and write its manifest.

    Idempotent: an existing, verified split is left alone, so re-running after
    a failed cluster job does not re-download gigabytes.

    Args:
        split: ``"train"`` or ``"test"``.
        base: Override for the data root.
        force: Re-download even if a valid copy exists.

    Returns:
        The manifest of the split on disk.

    Raises:
        RuntimeError: If the download produced an unexpected number of examples.
    """
    # Imported lazily: `datasets` pulls in a large dependency tree, and the
    # rest of the package must stay importable (and fast) without it.
    import datasets

    directory = split_dir(split, base=base)
    if not force:
        existing = read_manifest(split, base=base)
        if existing is not None and not verify_split(split, base=base):
            return existing

    upstream_split = _SPLIT_ALIASES[split]
    expected = EXPECTED_SPLIT_SIZES[split]

    loaded = datasets.load_dataset(
        HF_REPO_ID,
        split=upstream_split,
        revision=HF_REVISION,
    )
    num_examples = int(loaded.num_rows)
    if num_examples != expected:
        msg = (
            f"{split}: mirror returned {num_examples} examples but the FoodSeg103 paper "
            f"reports {expected}. Refusing to record this as FoodSeg103 -- the mirror may "
            f"have changed. Check {HF_REPO_ID} at revision {HF_REVISION}."
        )
        raise RuntimeError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    loaded.save_to_disk(str(directory))

    manifest = DatasetManifest(
        source="huggingface",
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        split=split,
        upstream_split=upstream_split,
        num_examples=num_examples,
        expected_examples=expected,
        license=LICENSE,
        column_names=list(loaded.column_names),
        content_sha256=_hash_directory(directory),
        datasets_version=datasets.__version__,
        downloaded_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    manifest_path(split, base=base).write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m food_shapley.download",
        description="Download FoodSeg103 into data/raw/ and record its provenance.",
    )
    parser.add_argument(
        "--split",
        choices=sorted(_SPLIT_ALIASES),
        default="test",
        help="Split to download (default: test, the 2,135-image evaluation split).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check an existing download without fetching anything.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero if the dataset is missing or corrupt.
    """
    args = _build_parser().parse_args(argv)

    if args.verify_only:
        problems = verify_split(args.split)
        if problems:
            print(f"FoodSeg103 '{args.split}' FAILED verification:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"FoodSeg103 '{args.split}' verified at {split_dir(args.split)}")
        return 0

    manifest = download_split(args.split, force=args.force)
    print(
        f"FoodSeg103 '{manifest.split}' ready at {split_dir(manifest.split)}: "
        f"{manifest.num_examples} examples, {manifest.repo_id}@{manifest.revision[:12]}, "
        f"{manifest.license}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
