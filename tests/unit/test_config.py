from __future__ import annotations

from pathlib import Path

import pytest

from food_shapley.config import ExperimentConfig, config_fingerprint, load_config

VALID_TOML = """
name = "smoke"
seed = 0
dataset = "foodseg103"
split = "test"
model = "openclip/ViT-B-32"
removal = "blur"
max_players = 7
"""


def test_load_config_roundtrip(tmp_path: Path, config: ExperimentConfig) -> None:
    path = tmp_path / "run.toml"
    path.write_text(VALID_TOML)
    assert load_config(path) == config


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(VALID_TOML + "\nmax_playerz = 3\n")
    with pytest.raises(ValueError, match="unknown configuration keys: max_playerz"):
        load_config(path)


def test_load_config_reports_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text('name = "smoke"\nseed = 0\n')
    with pytest.raises(ValueError, match="missing configuration keys"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("name", "  "), ("seed", -1), ("max_players", 0)],
)
def test_invalid_fields_are_rejected(config: ExperimentConfig, field: str, value: object) -> None:
    kwargs = config.to_dict() | {field: value}
    with pytest.raises(ValueError, match=field):
        ExperimentConfig(**kwargs)


def test_fingerprint_is_stable_across_instances(config: ExperimentConfig) -> None:
    assert config_fingerprint(config) == config_fingerprint(ExperimentConfig(**config.to_dict()))


def test_fingerprint_ignores_notes(config: ExperimentConfig) -> None:
    annotated = ExperimentConfig(**(config.to_dict() | {"notes": "rerun after cluster outage"}))
    assert config_fingerprint(annotated) == config_fingerprint(config)


def test_fingerprint_tracks_semantic_change(config: ExperimentConfig) -> None:
    other = ExperimentConfig(**(config.to_dict() | {"removal": "token-drop"}))
    assert config_fingerprint(other) != config_fingerprint(config)


def test_fingerprint_length(config: ExperimentConfig) -> None:
    assert len(config_fingerprint(config, length=8)) == 8
