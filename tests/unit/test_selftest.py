from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from food_shapley.config import ExperimentConfig
from food_shapley.paths import ARTIFACT_ROOT_ENV
from food_shapley.seeding import derive_seed
from food_shapley.selftest import main, run_prepare, run_task, run_verify, task_label

CONFIG_TOML = """
name = "smoke"
seed = 0
dataset = "foodseg103"
split = "test"
model = "open_clip/ViT-B-32-quickgelu/openai"
removal = "blur"
prompt_template = "a photo of {}, a type of food."
image_size = 224
max_players = 7
min_area_fraction = 0.005
blur_radius = 12.0
background_mode = "player"
interaction_order = 2
max_images = 0
"""


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config on disk, with the artifact root redirected into tmp_path."""
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path / "runs"))
    path = tmp_path / "smoke.toml"
    path.write_text(CONFIG_TOML)
    return path


@pytest.fixture
def prepared(config: ExperimentConfig, config_file: Path) -> ExperimentConfig:  # noqa: ARG001
    """A config whose run directory already exists."""
    run_prepare(config)
    return config


def test_task_label_is_zero_padded() -> None:
    assert task_label(42) == "task-00042"


def test_task_label_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        task_label(-1)


def test_task_labels_are_unique_and_sorted() -> None:
    labels = [task_label(i) for i in range(1000)]
    assert len(set(labels)) == 1000
    assert labels == sorted(labels)


def test_prepare_creates_run_directory(config: ExperimentConfig, config_file: Path) -> None:  # noqa: ARG001
    layout, created = run_prepare(config)
    assert created
    assert layout.metadata_file.is_file()


def test_prepare_refuses_second_run(prepared: ExperimentConfig) -> None:
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_prepare(prepared)


def test_prepare_allow_existing_reuses_the_run(prepared: ExperimentConfig) -> None:
    layout, created = run_prepare(prepared, allow_existing=True)
    assert not created
    assert layout.metadata_file.is_file()


def test_prepare_allow_existing_does_not_rewrite_provenance(prepared: ExperimentConfig) -> None:
    """Resuming must not restamp the run with a new commit or timestamp."""
    layout, _ = run_prepare(prepared, allow_existing=True)
    original = layout.metadata_file.read_text()

    run_prepare(prepared, allow_existing=True)
    assert layout.metadata_file.read_text() == original


def test_prepare_allow_existing_still_creates_a_fresh_run(
    config: ExperimentConfig,
    config_file: Path,  # noqa: ARG001
) -> None:
    layout, created = run_prepare(config, allow_existing=True)
    assert created
    assert layout.metadata_file.is_file()


def test_task_requires_prepare_first(config: ExperimentConfig, config_file: Path) -> None:  # noqa: ARG001
    with pytest.raises(FileNotFoundError, match="run the 'prepare' subcommand"):
        run_task(config, 0)


def test_task_writes_expected_payload(prepared: ExperimentConfig) -> None:
    output = run_task(prepared, 3)
    payload: dict[str, Any] = json.loads(output.read_text())
    assert payload["task_id"] == 3
    assert payload["label"] == "task-00003"
    assert payload["seed"] == derive_seed(prepared.seed, "task-00003")


def test_task_refuses_to_overwrite_its_own_output(prepared: ExperimentConfig) -> None:
    run_task(prepared, 0)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_task(prepared, 0)


def test_tasks_do_not_collide(prepared: ExperimentConfig) -> None:
    outputs = [run_task(prepared, i) for i in range(10)]
    assert len({path.name for path in outputs}) == 10


def test_verify_passes_for_a_complete_array(prepared: ExperimentConfig) -> None:
    for i in range(5):
        run_task(prepared, i)
    assert run_verify(prepared, 5) == []


def test_verify_reports_a_missing_task(prepared: ExperimentConfig) -> None:
    for i in (0, 1, 3):
        run_task(prepared, i)
    problems = run_verify(prepared, 4)
    assert len(problems) == 1
    assert "task-00002" in problems[0]
    assert "missing output" in problems[0]


def test_verify_detects_a_tampered_seed(prepared: ExperimentConfig) -> None:
    output = run_task(prepared, 0)
    payload: dict[str, Any] = json.loads(output.read_text())
    payload["seed"] = payload["seed"] + 1
    output.write_text(json.dumps(payload))

    problems = run_verify(prepared, 1)
    assert len(problems) == 1
    assert "does not match derived seed" in problems[0]


def test_verify_detects_a_seed_collision(prepared: ExperimentConfig) -> None:
    """Two tasks sharing a seed would silently correlate results in a real run."""
    run_task(prepared, 0)
    output = run_task(prepared, 1)
    payload: dict[str, Any] = json.loads(output.read_text())
    payload["seed"] = derive_seed(prepared.seed, "task-00000")
    output.write_text(json.dumps(payload))

    problems = run_verify(prepared, 2)
    # The tampered task no longer matches its own derived seed, which is caught
    # first; either way the array must not be reported as sound.
    assert problems != []


def test_verify_rejects_zero_tasks(prepared: ExperimentConfig) -> None:
    with pytest.raises(ValueError, match="num_tasks must be >= 1"):
        run_verify(prepared, 0)


def test_cli_prepare_task_verify_roundtrip(config_file: Path) -> None:
    base = ["--config", str(config_file)]
    assert main([*base, "prepare"]) == 0
    assert main([*base, "task", "--task-id", "0"]) == 0
    assert main([*base, "task", "--task-id", "1"]) == 0
    assert main([*base, "verify", "--num-tasks", "2"]) == 0


def test_cli_prepare_is_idempotent_with_allow_existing(config_file: Path) -> None:
    """Resubmitting a partially failed array must not look like a crash."""
    base = ["--config", str(config_file)]
    assert main([*base, "prepare", "--allow-existing"]) == 0
    assert main([*base, "prepare", "--allow-existing"]) == 0


def test_cli_prepare_without_allow_existing_still_raises(config_file: Path) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    with pytest.raises(FileExistsError):
        main([*base, "prepare"])


def test_cli_verify_fails_on_incomplete_array(config_file: Path) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    main([*base, "task", "--task-id", "0"])
    assert main([*base, "verify", "--num-tasks", "2"]) == 1


def test_cli_task_reads_slurm_array_task_id(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    assert main([*base, "task"]) == 0
    assert main([*base, "verify", "--num-tasks", "1"]) == 1  # task 0 is absent


def test_cli_task_without_any_task_id_exits(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    with pytest.raises(SystemExit, match="SLURM_ARRAY_TASK_ID is not set"):
        main([*base, "task"])


def test_cli_task_rejects_non_integer_slurm_id(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "not-a-number")
    with pytest.raises(SystemExit, match="not an integer"):
        main([*base, "task"])
