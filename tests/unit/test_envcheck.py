from __future__ import annotations

from pathlib import Path

import pytest

from food_shapley.envcheck import REQUIRED_IMPORTS, CheckResult, check_environment, main
from food_shapley.paths import DATA_ROOT_ENV

# These tests run against the real environment: the point of the module is to
# report what is actually installed, and stubbing the imports out would only
# test the stub. The GPU path cannot be asserted absolutely, since it depends
# on the machine -- see test_require_gpu_agrees_with_torch, which asserts the
# verdict follows torch rather than asserting a particular verdict.


@pytest.fixture
def empty_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data root with no dataset in it."""
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "data"))
    return tmp_path


def test_required_imports_are_all_available() -> None:
    """Every library the pipeline needs must be in the locked environment."""
    result = check_environment()
    assert result.problems == []


def test_every_required_import_is_reported() -> None:
    result = check_environment()
    report = "\n".join(result.facts)
    for name in REQUIRED_IMPORTS:
        assert name in report


def test_facts_include_the_interpreter() -> None:
    """A job log must identify which Python actually ran, not just that one did."""
    result = check_environment()
    assert any("python" in fact for fact in result.facts)


def test_ok_tracks_problems() -> None:
    result = CheckResult()
    assert result.ok
    result.problems.append("something")
    assert not result.ok


def test_require_gpu_agrees_with_torch() -> None:
    """Whether a GPU is present is environment-dependent; the verdict must follow it."""
    import torch

    result = check_environment(require_gpu=True)
    assert result.ok == torch.cuda.is_available()


def test_gpu_absence_is_not_an_error_by_default() -> None:
    assert check_environment(require_gpu=False).ok


def test_missing_dataset_is_only_a_note_by_default(
    empty_data_root: Path,  # noqa: ARG001
) -> None:
    """The check is also run on machines that deliberately hold no data."""
    result = check_environment(dataset_split="test")
    assert result.ok
    assert any("not downloaded" in fact for fact in result.facts)


def test_missing_dataset_fails_when_required(
    empty_data_root: Path,  # noqa: ARG001
) -> None:
    result = check_environment(dataset_split="test", require_dataset=True)
    assert not result.ok
    assert any("not downloaded" in problem for problem in result.problems)


def test_dataset_check_is_skipped_when_not_asked_for(
    empty_data_root: Path,  # noqa: ARG001
) -> None:
    result = check_environment(dataset_split=None, require_dataset=True)
    assert result.ok


def test_unknown_split_is_rejected(
    empty_data_root: Path,  # noqa: ARG001
) -> None:
    with pytest.raises(ValueError, match="unknown split"):
        check_environment(dataset_split="validation")


def test_missing_prompt_set_is_only_a_note_by_default(
    empty_data_root: Path,  # noqa: ARG001
) -> None:
    result = check_environment()
    assert result.ok
    assert any("food101" in fact for fact in result.facts)


def test_missing_prompt_set_fails_when_required(
    empty_data_root: Path,  # noqa: ARG001
) -> None:
    """The 101 prompts are the model's output space.

    Without them a job has no value function to compute, so a compute node must
    stop here rather than discover it per image.
    """
    result = check_environment(require_prompts=True)
    assert not result.ok
    assert any("food101" in problem for problem in result.problems)


def test_prompt_set_is_reported_even_when_not_required() -> None:
    """Reported unconditionally, required or not.

    A job log that omits which prompt set was on the node cannot be used to
    explain the numbers it produced.
    """
    result = check_environment()
    assert any("food101" in fact for fact in result.facts)


def test_cli_passes_on_this_machine() -> None:
    assert main([]) == 0


def test_cli_reports_dataset_failure(
    empty_data_root: Path,  # noqa: ARG001
) -> None:
    assert main(["--dataset-split", "test", "--require-dataset"]) == 1


def test_cli_reports_prompt_set_failure(
    empty_data_root: Path,  # noqa: ARG001
) -> None:
    assert main(["--require-prompts"]) == 1
