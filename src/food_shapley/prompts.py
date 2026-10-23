"""The dish prompt set.

`PLAN.md` Milestone 1, step 2, first half: "pick dish prompts (e.g. Food-101's
101 class names)". The prompt set is not a label list that sits beside the
experiment -- it **is** the model's output space. CLIP produces a score per
prompt and the value function is a log-softmax over exactly these strings, so
the set defines the denominator of every value the pipeline will ever compute.
Adding, removing or rewording a prompt changes every Shapley value.

Why FoodSeg103 needs a prompt set at all
----------------------------------------
FoodSeg103 ships ingredient masks and nothing else: its columns are ``image``,
``label``, ``classes_on_image``, ``id``. **There is no dish annotation**, so
"log-probability of the target dish" has no target to read off the data. The
project's question is "why does the model say burger?", not "is the model
right?", so the target is CLIP's own top-1 over this prompt set on the
unmasked photograph; see :mod:`food_shapley.value`. That makes the prompt set
the only thing standing between a pixel and a dish name.

Food-101 is a borrowed vocabulary, not a matched one
----------------------------------------------------
Food-101's 101 classes were not chosen for FoodSeg103, and plenty of
FoodSeg103 photographs depict no Food-101 dish. For those, the top-1 is the
least-bad of 101 options and the value function measures evidence for a label
that does not describe the picture. This is a real limitation of the design,
not a bug in this module: it is bounded by reporting the top-1 confidence
distribution over unmasked images rather than assumed away. See
``insights/week-01.md``.

Provenance
----------
Only the **class-name list** is used -- none of Food-101's 101,000 images are
downloaded, which matters because their license is restrictive (see
:data:`LICENSE`). The names are read from the dataset's feature metadata at a
pinned revision and stored under ``data/raw/food101/`` with a manifest, so a
run traces back to a specific list rather than to "Food-101" in the abstract.

Unlike FoodSeg103's ``id2label.json`` there is no single upstream file to hash,
so :attr:`PromptManifest.names_sha256` is taken over **this module's own
serialisation** of the list. It therefore detects local corruption and any
change in what upstream reports, but it is not comparable against a file
published by ETH Zurich.
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
    "DEFAULT_PROMPT_TEMPLATE",
    "DISH_NAMES_FILENAME",
    "FOOD101_NUM_CLASSES",
    "FOOD101_REPO_ID",
    "FOOD101_REVISION",
    "LICENSE",
    "PROMPT_SET_DIRNAME",
    "PromptManifest",
    "build_prompts",
    "dish_display_name",
    "download_dish_names",
    "main",
    "read_dish_names",
    "verify_dish_names",
]

#: Hugging Face copy of Food-101. Only its feature metadata is read.
FOOD101_REPO_ID: Final = "ethz/food101"

#: Pinned revision (lastModified 2026-03-09). Bumping this changes the model's
#: output space, so it is a change of experiment: new fingerprint, new run.
FOOD101_REVISION: Final = "83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9"

#: Number of dish classes Food-101 defines.
FOOD101_NUM_CLASSES: Final = 101

#: License of the Food-101 *images*, recorded even though none are downloaded.
#: ETH Zurich distributes images sourced from Foodspotting and does not own
#: them; use beyond scientific fair use must be negotiated with the picture
#: owners. The Hugging Face copy is tagged ``license:unknown``. This project
#: uses only the 101 English dish names, which are not images.
LICENSE: Final = "unknown (images: ETHZ/Foodspotting, scientific fair use; names only are used)"

#: Directory under ``data/raw/`` holding the prompt set.
PROMPT_SET_DIRNAME: Final = "food101"

#: Where the dish-name list is stored on disk.
DISH_NAMES_FILENAME: Final = "dish_names.json"

#: Prompt template. This is the Food-101 template from the CLIP paper's
#: prompt-engineering appendix, used rather than invented because a
#: hand-tuned template would be an unreported degree of freedom in every
#: number the project reports. It is still a configuration field
#: (``prompt_template``), so a run that changes it is a different run.
DEFAULT_PROMPT_TEMPLATE: Final = "a photo of {}, a type of food."


@dataclass(frozen=True, slots=True)
class PromptManifest:
    """Record of the downloaded dish-name list.

    Attributes:
        source: Where the names came from, e.g. ``"huggingface"``.
        repo_id: Repository the names were read from.
        revision: Pinned commit of that repository.
        num_classes: Number of dish names stored.
        expected_classes: Number Food-101 is documented to have.
        license: License string; see :data:`LICENSE`.
        names_sha256: Hash of this module's serialisation of the list.
        datasets_version: Version of ``datasets`` that read the metadata.
        downloaded_utc: ISO-8601 timestamp of the download.
    """

    source: str
    repo_id: str
    revision: str
    num_classes: int
    expected_classes: int
    license: str
    names_sha256: str
    datasets_version: str
    downloaded_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-serialisable dictionary."""
        return asdict(self)


def prompt_set_dir(*, base: Path | None = None) -> Path:
    """Return the directory holding the prompt set. It may not exist yet."""
    root = data_root() if base is None else base
    return root / "raw" / PROMPT_SET_DIRNAME


def names_path(*, base: Path | None = None) -> Path:
    """Return the path of the stored dish-name list."""
    return prompt_set_dir(base=base) / DISH_NAMES_FILENAME


def prompt_manifest_path(*, base: Path | None = None) -> Path:
    """Return the path of the prompt-set manifest."""
    return prompt_set_dir(base=base) / "manifest.json"


def _serialise(names: Sequence[str]) -> bytes:
    """Serialise the name list canonically, so its hash is stable."""
    return json.dumps(list(names), indent=2, ensure_ascii=False).encode() + b"\n"


def _parse_dish_names(raw: bytes) -> tuple[str, ...]:
    """Parse and validate the dish-name list.

    Validated rather than trusted: a truncated or reordered list silently
    changes the softmax denominator, and every value in the run with it.

    Raises:
        ValueError: If the payload is not exactly
            :data:`FOOD101_NUM_CLASSES` unique, non-empty strings.
    """
    try:
        payload: object = json.loads(raw)
    except ValueError as error:
        msg = f"{DISH_NAMES_FILENAME} is not valid JSON: {error}"
        raise ValueError(msg) from error

    if not isinstance(payload, list) or not all(isinstance(name, str) for name in payload):
        msg = f"{DISH_NAMES_FILENAME}: expected a JSON list of strings"
        raise ValueError(msg)

    names: list[str] = list(payload)
    if len(names) != FOOD101_NUM_CLASSES:
        msg = f"{DISH_NAMES_FILENAME}: expected {FOOD101_NUM_CLASSES} names, got {len(names)}"
        raise ValueError(msg)
    if any(not name.strip() for name in names):
        msg = f"{DISH_NAMES_FILENAME}: contains an empty name"
        raise ValueError(msg)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        # Two identical prompts split the probability mass of one dish between
        # them, so the top-1 can be a dish the model ranked second.
        msg = f"{DISH_NAMES_FILENAME}: duplicate names {duplicates}"
        raise ValueError(msg)
    return tuple(names)


def read_dish_names(*, base: Path | None = None) -> tuple[str, ...]:
    """Read the stored dish-name list, in Food-101's own order.

    Returns:
        The raw names, e.g. ``"apple_pie"``. Use :func:`dish_display_name`
        before putting one in a prompt.

    Raises:
        FileNotFoundError: If the prompt set has not been downloaded.
        ValueError: If the stored list is malformed.
    """
    path = names_path(base=base)
    if not path.is_file():
        msg = f"{path} not found; run `python -m food_shapley.prompts` first"
        raise FileNotFoundError(msg)
    return _parse_dish_names(path.read_bytes())


def read_prompt_manifest(*, base: Path | None = None) -> PromptManifest | None:
    """Read the prompt-set manifest, or ``None`` if it was never downloaded.

    Raises:
        ValueError: If the manifest exists but cannot be parsed.
    """
    path = prompt_manifest_path(base=base)
    if not path.is_file():
        return None
    try:
        payload: dict[str, Any] = json.loads(path.read_text())
        return PromptManifest(**payload)
    except (TypeError, ValueError) as error:
        msg = f"{path}: manifest is unreadable or has an unexpected shape: {error}"
        raise ValueError(msg) from error


def verify_dish_names(*, base: Path | None = None) -> list[str]:
    """Check the stored prompt set against its manifest.

    Returns:
        A list of problems; empty means the prompt set is sound.
    """
    try:
        manifest = read_prompt_manifest(base=base)
    except ValueError as error:
        return [f"food101: {error}; re-download"]
    if manifest is None:
        return [f"food101: not downloaded (no manifest at {prompt_manifest_path(base=base)})"]

    problems: list[str] = []
    if manifest.revision != FOOD101_REVISION:
        problems.append(
            f"food101: downloaded from revision {manifest.revision}, "
            f"but the code now pins {FOOD101_REVISION}; re-download"
        )
    if manifest.num_classes != manifest.expected_classes:
        got, want = manifest.num_classes, manifest.expected_classes
        problems.append(f"food101: has {got} names, expected {want}")
    try:
        names = read_dish_names(base=base)
    except (FileNotFoundError, ValueError) as error:
        problems.append(f"food101: {error}")
    else:
        actual = hashlib.sha256(_serialise(names)).hexdigest()
        if actual != manifest.names_sha256:
            problems.append(
                f"food101: names hash {actual[:12]} does not match manifest "
                f"{manifest.names_sha256[:12]}; the prompt set changed"
            )
    return problems


def download_dish_names(*, base: Path | None = None, force: bool = False) -> PromptManifest:
    """Fetch the Food-101 class names and write them with a manifest.

    Reads the dataset's *feature metadata* only: none of Food-101's 101,000
    images are downloaded. Idempotent, so a resubmitted cluster job does not
    refetch.

    Args:
        base: Override for the data root.
        force: Re-fetch even if a valid copy exists.

    Returns:
        The manifest of the prompt set on disk.

    Raises:
        RuntimeError: If upstream reports a different number of classes than
            Food-101 documents, which would mean the pinned copy is not the
            dataset this project thinks it is.
    """
    # Lazy, as in `download`: `datasets` has a large import cost and the rest
    # of the package must stay importable without paying it.
    import datasets

    if not force:
        existing = read_prompt_manifest(base=base)
        if existing is not None and not verify_dish_names(base=base):
            return existing

    builder = datasets.load_dataset_builder(FOOD101_REPO_ID, revision=FOOD101_REVISION)
    features = builder.info.features
    if features is None or "label" not in features:
        msg = (
            f"{FOOD101_REPO_ID}@{FOOD101_REVISION[:12]} exposes no 'label' feature, "
            "so the class names cannot be read from its metadata."
        )
        raise RuntimeError(msg)
    names = tuple(features["label"].names)
    if len(names) != FOOD101_NUM_CLASSES:
        msg = (
            f"{FOOD101_REPO_ID}@{FOOD101_REVISION[:12]} reports {len(names)} classes, "
            f"but Food-101 has {FOOD101_NUM_CLASSES}. Refusing to record this as Food-101."
        )
        raise RuntimeError(msg)
    # Validates uniqueness and non-emptiness before anything is written.
    payload = _serialise(names)
    _parse_dish_names(payload)

    directory = prompt_set_dir(base=base)
    directory.mkdir(parents=True, exist_ok=True)
    names_path(base=base).write_bytes(payload)

    manifest = PromptManifest(
        source="huggingface",
        repo_id=FOOD101_REPO_ID,
        revision=FOOD101_REVISION,
        num_classes=len(names),
        expected_classes=FOOD101_NUM_CLASSES,
        license=LICENSE,
        names_sha256=hashlib.sha256(payload).hexdigest(),
        datasets_version=datasets.__version__,
        downloaded_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    prompt_manifest_path(base=base).write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
    return manifest


def dish_display_name(name: str) -> str:
    """Turn a Food-101 class id into something CLIP was trained to read.

    Food-101 names are snake_case identifiers (``baby_back_ribs``). CLIP's text
    encoder was trained on web captions, where ``baby back ribs`` appears and
    ``baby_back_ribs`` does not, so the underscores are replaced rather than
    passed through.

    Args:
        name: A Food-101 class name.

    Returns:
        The same dish in prose form.
    """
    return name.replace("_", " ")


def build_prompts(names: Sequence[str], template: str) -> tuple[str, ...]:
    """Render the prompt strings CLIP will actually score.

    Args:
        names: Dish names in Food-101's order; the order is preserved, because
            a prompt's position is its class index everywhere downstream.
        template: A format string with a single ``{}`` placeholder, e.g.
            :data:`DEFAULT_PROMPT_TEMPLATE`.

    Returns:
        One prompt per name, in the same order.

    Raises:
        ValueError: If ``template`` has no ``{}`` placeholder, or ``names`` is
            empty. A template without a placeholder would give every dish the
            same prompt and a uniform, meaningless value function.
    """
    if not names:
        msg = "cannot build prompts from an empty name list"
        raise ValueError(msg)
    if "{}" not in template:
        msg = f"prompt template must contain a '{{}}' placeholder, got {template!r}"
        raise ValueError(msg)
    return tuple(template.format(dish_display_name(name)) for name in names)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m food_shapley.prompts",
        description="Fetch the Food-101 dish-name list into data/raw/ and record its provenance.",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch even if present.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check an existing prompt set without fetching anything.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        metavar="N",
        help="Print the first N rendered prompts, to see what CLIP will be given.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero if the prompt set is missing or corrupt.
    """
    args = _build_parser().parse_args(argv)

    if args.verify_only:
        problems = verify_dish_names()
        if problems:
            print("Food-101 prompt set FAILED verification:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"Food-101 prompt set verified at {prompt_set_dir()}")
    else:
        manifest = download_dish_names(force=args.force)
        print(
            f"Food-101 prompt set ready at {prompt_set_dir()}: "
            f"{manifest.num_classes} dish names, "
            f"{manifest.repo_id}@{manifest.revision[:12]}, license {manifest.license}"
        )

    if args.show:
        prompts = build_prompts(read_dish_names(), DEFAULT_PROMPT_TEMPLATE)
        for prompt in prompts[: args.show]:
            print(f"  {prompt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
