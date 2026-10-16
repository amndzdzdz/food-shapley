from __future__ import annotations

from pathlib import Path

import pytest

from food_shapley.paths import ARTIFACT_ROOT_ENV, DATA_ROOT_ENV, RunLayout, artifact_root, data_root


def test_artifact_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ARTIFACT_ROOT_ENV, raising=False)
    assert artifact_root() == Path("runs")


def test_artifact_root_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, "/scratch/runs")
    assert artifact_root() == Path("/scratch/runs")


def test_data_root_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, "/scratch/data")
    assert data_root() == Path("/scratch/data")


def test_for_run_encodes_name_and_fingerprint(tmp_path: Path) -> None:
    layout = RunLayout.for_run("smoke", "abc123", base=tmp_path)
    assert layout.root == tmp_path / "smoke-abc123"


def test_create_builds_subdirectories(tmp_path: Path) -> None:
    layout = RunLayout.for_run("smoke", "abc123", base=tmp_path)
    layout.create()
    assert layout.values_dir.is_dir()
    assert layout.figures_dir.is_dir()
    assert layout.logs_dir.is_dir()


def test_create_refuses_existing_run_by_default(tmp_path: Path) -> None:
    layout = RunLayout.for_run("smoke", "abc123", base=tmp_path)
    layout.create()
    with pytest.raises(FileExistsError):
        layout.create()
