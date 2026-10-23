from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING

import pytest

from food_shapley.paths import DATA_ROOT_ENV
from food_shapley.prompts import (
    DEFAULT_PROMPT_TEMPLATE,
    DISH_NAMES_FILENAME,
    FOOD101_NUM_CLASSES,
    PromptManifest,
    _parse_dish_names,
    _serialise,
    build_prompts,
    dish_display_name,
    download_dish_names,
    main,
    names_path,
    prompt_manifest_path,
    prompt_set_dir,
    read_dish_names,
    read_prompt_manifest,
    verify_dish_names,
)

if TYPE_CHECKING:
    from pathlib import Path

# --------------------------------------------------------------------------- #
# A stand-in for `datasets`.
#
# `download_dish_names` reads Food-101's class names out of the dataset
# *metadata* -- no images -- but that is still a network call, which the suite
# must not make. `datasets` is imported lazily inside the function, so
# replacing the module in `sys.modules` is enough. What is under test is this
# module's own logic: the class-count guard, the manifest, serialisation and
# the integrity check.
# --------------------------------------------------------------------------- #

FAKE_DATASETS_VERSION = "0.0.0-fake"

#: 101 plausible snake_case names. Only the count and the shape matter here;
#: the real names are pinned by revision, not by this file.
FAKE_NAMES = tuple(f"dish_{index:03d}_name" for index in range(FOOD101_NUM_CLASSES))


class _FakeLabel:
    """The `ClassLabel` feature whose `names` carry the vocabulary."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = list(names)


class _FakeBuilder:
    """What `datasets.load_dataset_builder` is expected to return."""

    def __init__(self, features: dict[str, _FakeLabel] | None) -> None:
        self.info = types.SimpleNamespace(features=features)


#: Distinguishes "use the default label feature" from "report no features at
#: all", which is a branch of its own in `download_dish_names`.
_DEFAULT_FEATURES = object()


def _install_fake_datasets(
    monkeypatch: pytest.MonkeyPatch,
    *,
    names: tuple[str, ...] = FAKE_NAMES,
    features: object = _DEFAULT_FEATURES,
    calls: list[tuple[str, str]] | None = None,
) -> None:
    """Put a fake `datasets` module in `sys.modules`."""
    resolved = {"label": _FakeLabel(names)} if features is _DEFAULT_FEATURES else features

    def load_dataset_builder(repo_id: str, *, revision: str) -> _FakeBuilder:
        if calls is not None:
            calls.append((repo_id, revision))
        return _FakeBuilder(resolved)  # type: ignore[arg-type]

    module = types.ModuleType("datasets")
    module.load_dataset_builder = load_dataset_builder  # type: ignore[attr-defined]
    module.__version__ = FAKE_DATASETS_VERSION  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the data root into tmp_path, so the real copy is untouched.

    Autouse: a test that forgets it would read and, worse, rewrite the prompt
    set this project's results depend on.
    """
    root = tmp_path / "data"
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    return root


# --------------------------------------------------------------------------- #
# Fetching and the manifest
# --------------------------------------------------------------------------- #


def test_download_writes_names_and_manifest(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_datasets(monkeypatch)
    manifest = download_dish_names()

    assert manifest.num_classes == FOOD101_NUM_CLASSES
    assert read_dish_names() == FAKE_NAMES
    assert read_prompt_manifest() == manifest
    assert names_path().is_relative_to(data_root)


def test_download_pins_the_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """The names must come from one commit, not from whatever is current."""
    calls: list[tuple[str, str]] = []
    _install_fake_datasets(monkeypatch, calls=calls)
    manifest = download_dish_names()

    assert len(calls) == 1
    repo_id, revision = calls[0]
    assert (repo_id, revision) == (manifest.repo_id, manifest.revision)
    assert len(manifest.revision) == 40


def test_download_rejects_a_different_class_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vocabulary of the wrong size is not Food-101, whatever it is called."""
    _install_fake_datasets(monkeypatch, names=FAKE_NAMES[:50])
    with pytest.raises(RuntimeError, match="reports 50 classes"):
        download_dish_names()
    assert not names_path().exists()


def test_download_rejects_missing_label_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_datasets(monkeypatch, features={})
    with pytest.raises(RuntimeError, match="no 'label' feature"):
        download_dish_names()


def test_download_rejects_features_of_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """`features` is Optional upstream, so None is a real case, not a typo."""
    _install_fake_datasets(monkeypatch, features=None)
    with pytest.raises(RuntimeError, match="no 'label' feature"):
        download_dish_names()


def test_download_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resubmitted cluster job must not refetch."""
    calls: list[tuple[str, str]] = []
    _install_fake_datasets(monkeypatch, calls=calls)
    first = download_dish_names()
    second = download_dish_names()

    assert len(calls) == 1
    assert first == second


def test_force_refetches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    _install_fake_datasets(monkeypatch, calls=calls)
    download_dish_names()
    download_dish_names(force=True)
    assert len(calls) == 2


def test_base_override_beats_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_datasets(monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    download_dish_names(base=elsewhere)

    assert read_dish_names(base=elsewhere) == FAKE_NAMES
    assert not names_path().exists()
    assert prompt_set_dir(base=elsewhere).is_relative_to(elsewhere)


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #


def test_verify_is_clean_after_download(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_datasets(monkeypatch)
    download_dish_names()
    assert verify_dish_names() == []


def test_verify_reports_a_missing_prompt_set() -> None:
    problems = verify_dish_names()
    assert problems
    assert any("manifest" in problem for problem in problems)


def test_verify_detects_edited_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hash exists to catch exactly this: a hand-edited prompt set."""
    _install_fake_datasets(monkeypatch)
    download_dish_names()
    tampered = ("sushi", *FAKE_NAMES[1:])
    names_path().write_bytes(_serialise(tampered))

    problems = verify_dish_names()
    assert problems
    assert any("hash" in problem for problem in problems)


def test_verify_detects_a_truncated_prompt_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_datasets(monkeypatch)
    download_dish_names()
    names_path().write_bytes(_serialise(FAKE_NAMES[:100]))
    assert verify_dish_names()


def test_read_manifest_is_none_when_absent() -> None:
    assert read_prompt_manifest() is None


def test_read_names_requires_the_file() -> None:
    with pytest.raises(FileNotFoundError, match=DISH_NAMES_FILENAME):
        read_dish_names()


def test_manifest_roundtrips_through_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_datasets(monkeypatch)
    manifest = download_dish_names()
    raw = json.loads(prompt_manifest_path().read_text())
    assert PromptManifest(**raw) == manifest


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parse_accepts_a_well_formed_list() -> None:
    assert _parse_dish_names(_serialise(FAKE_NAMES)) == FAKE_NAMES


def test_parse_rejects_duplicates() -> None:
    """Two identical prompts would split one dish's probability mass in two."""
    names = (FAKE_NAMES[0], *FAKE_NAMES[2:], FAKE_NAMES[0])
    with pytest.raises(ValueError, match="duplicate"):
        _parse_dish_names(_serialise(names))


def test_parse_rejects_empty_names() -> None:
    names = ("   ", *FAKE_NAMES[1:])
    with pytest.raises(ValueError, match="empty"):
        _parse_dish_names(_serialise(names))


def test_parse_rejects_the_wrong_count() -> None:
    """A short list silently shrinks the softmax denominator."""
    with pytest.raises(ValueError, match="expected 101 names, got 100"):
        _parse_dish_names(_serialise(FAKE_NAMES[:100]))


def test_parse_rejects_a_json_object() -> None:
    with pytest.raises(ValueError, match="list"):
        _parse_dish_names(b'{"0": "pizza"}')


def test_parse_preserves_order() -> None:
    """A name's position is its class index everywhere downstream."""
    shuffled = tuple(reversed(FAKE_NAMES))
    assert _parse_dish_names(_serialise(shuffled)) == shuffled


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


def test_display_name_drops_underscores() -> None:
    """CLIP's captions contain 'baby back ribs', never 'baby_back_ribs'."""
    assert dish_display_name("baby_back_ribs") == "baby back ribs"


def test_display_name_leaves_plain_names_alone() -> None:
    assert dish_display_name("pizza") == "pizza"


def test_build_prompts_uses_the_template() -> None:
    prompts = build_prompts(("apple_pie",), DEFAULT_PROMPT_TEMPLATE)
    assert prompts == ("a photo of apple pie, a type of food.",)


def test_build_prompts_preserves_order() -> None:
    names = ("pizza", "apple_pie", "churros")
    prompts = build_prompts(names, "{}")
    assert prompts == ("pizza", "apple pie", "churros")


def test_build_prompts_rejects_a_template_without_a_placeholder() -> None:
    """Without '{}' every dish gets one prompt and every Shapley value is 0."""
    with pytest.raises(ValueError, match="placeholder"):
        build_prompts(("pizza",), "a photo of food.")


def test_build_prompts_rejects_an_empty_name_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_prompts((), DEFAULT_PROMPT_TEMPLATE)


def test_default_template_is_the_clip_paper_one() -> None:
    """Pinned so that a reword is a visible diff, not a silent drift."""
    assert DEFAULT_PROMPT_TEMPLATE == "a photo of {}, a type of food."


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_main_downloads_and_reports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_datasets(monkeypatch)
    assert main([]) == 0
    assert str(FOOD101_NUM_CLASSES) in capsys.readouterr().out


def test_main_verify_only_fails_without_a_prompt_set(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--verify-only` must never reach the network, so it must fail here."""
    assert main(["--verify-only"]) == 1
    assert capsys.readouterr().err


def test_main_verify_only_passes_after_download(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_datasets(monkeypatch)
    main([])
    assert main(["--verify-only"]) == 0


def test_main_show_prints_prompts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_datasets(monkeypatch)
    assert main(["--show", "3"]) == 0
    out = capsys.readouterr().out
    assert out.count("a photo of") >= 3
