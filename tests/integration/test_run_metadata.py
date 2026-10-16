"""End-to-end check of the reproducibility contract: config in, traceable run dir out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from food_shapley.config import ExperimentConfig, config_fingerprint
from food_shapley.paths import RunLayout
from food_shapley.provenance import write_run_metadata

pytestmark = pytest.mark.integration


def _layout(tmp_path: Path, config: ExperimentConfig) -> RunLayout:
    return RunLayout.for_run(config.name, config_fingerprint(config), base=tmp_path)


def test_run_metadata_is_written(tmp_path: Path, config: ExperimentConfig) -> None:
    layout = write_run_metadata(config, layout=_layout(tmp_path, config))

    assert layout.config_file.is_file()
    assert layout.metadata_file.is_file()

    metadata: dict[str, Any] = json.loads(layout.metadata_file.read_text())
    assert metadata["config"] == config.to_dict()
    assert metadata["config_fingerprint"] == config_fingerprint(config)
    assert metadata["timestamp_utc"].endswith("+00:00")
    assert metadata["packages"]["food-shapley"] is not None


def test_run_directory_is_traceable_to_config(tmp_path: Path, config: ExperimentConfig) -> None:
    layout = write_run_metadata(config, layout=_layout(tmp_path, config))
    assert layout.root.name == f"{config.name}-{config_fingerprint(config)}"


def test_existing_run_is_not_overwritten(tmp_path: Path, config: ExperimentConfig) -> None:
    layout = write_run_metadata(config, layout=_layout(tmp_path, config))
    original = layout.metadata_file.read_text()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_run_metadata(config, layout=_layout(tmp_path, config))

    assert layout.metadata_file.read_text() == original


def test_git_provenance_is_captured_outside_a_repo(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    non_repo = tmp_path / "elsewhere"
    non_repo.mkdir()
    layout = write_run_metadata(config, layout=_layout(tmp_path, config), repo=non_repo)

    metadata: dict[str, Any] = json.loads(layout.metadata_file.read_text())
    assert metadata["git_commit"] is None
    assert metadata["git_dirty"] is False
