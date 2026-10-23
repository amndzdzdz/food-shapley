"""Unit tests for the full-split job array.

Nothing here loads CLIP or FoodSeg103. :class:`FakeDecomposer` stands in for
:class:`~food_shapley.run.ImageDecomposer`, but it returns *real*
:class:`~food_shapley.indices.ImageIndices` objects -- one genuine
decomposition of the synthetic game from ``synthetic_game.py``, relabelled per
image. So the files these tests write are structurally the files a cluster run
writes, and ``save_indices``/``load_indices`` are exercised for real; only the
model is absent.

What is under test is the bookkeeping, which is where an array job actually
goes wrong: work divided without overlap or omission, a resumption that
resumes rather than quietly starting a different experiment, a half-written
file never left behind, one bad image not taking 2,000 good ones with it, and
an incomplete run that refuses to look complete.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from synthetic_game import build_game

from food_shapley.config import ExperimentConfig, config_fingerprint
from food_shapley.indices import ImageIndices, compute_indices, load_indices, save_indices
from food_shapley.paths import ARTIFACT_ROOT_ENV, RunLayout
from food_shapley.run import (
    ImageDecomposer,
    TaskReport,
    image_label,
    main,
    run_prepare,
    run_task,
    run_verify,
    selected_images,
    shard,
)
from food_shapley.seeding import derive_seed

if TYPE_CHECKING:
    from collections.abc import Collection

#: Images in the stand-in split. Deliberately not a multiple of the task
#: counts used below, so an off-by-one in the sharding leaves a remainder.
SPLIT_SIZE = 12

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


class FakeDecomposer:
    """A decomposer that produces real results without a model or a dataset.

    Attributes:
        seen: Images actually decomposed, in order. A resumed run must leave
            this empty for the images it skipped -- "skipped" has to mean *not
            recomputed*, not "recomputed and thrown away".
        draws: ``random.random()`` drawn inside each :meth:`decompose` call.
            The run seeds the process from the image's own label before
            calling, so these values prove the seed follows the image and not
            the array task.
    """

    def __init__(
        self,
        template: ImageIndices,
        *,
        split_size: int = SPLIT_SIZE,
        fail_on: Collection[int] = (),
        claim_image: int | None = None,
    ) -> None:
        """Store the template and the misbehaviours this instance should show.

        Args:
            template: A genuine decomposition, relabelled for each image.
            split_size: Images the stand-in split holds.
            fail_on: Images for which :meth:`decompose` raises.
            claim_image: If set, every result claims to be this image,
                whichever was asked for.
        """
        self._template = template
        self._split_size = split_size
        self._fail_on = frozenset(fail_on)
        self._claim_image = claim_image
        self.seen: list[int] = []
        self.draws: dict[int, float] = {}

    @property
    def split_size(self) -> int:
        """Number of images in the stand-in split."""
        return self._split_size

    def decompose(self, index: int) -> ImageIndices:
        """Return the template, relabelled as image ``index``.

        Raises:
            RuntimeError: If this image is in ``fail_on``.
        """
        self.seen.append(index)
        self.draws[index] = random.random()  # noqa: S311 - not cryptographic.
        if index in self._fail_on:
            msg = f"image {index} is unreadable"
            raise RuntimeError(msg)
        claimed = index if self._claim_image is None else self._claim_image
        return replace(self._template, image_index=claimed)


def as_decomposer(stub: FakeDecomposer) -> ImageDecomposer:
    """Hand the stub to `run_task` under its real parameter type.

    `run_task` touches exactly two members -- ``split_size`` and
    ``decompose`` -- so the stub is a faithful stand-in. Cast rather than
    widening the production signature to a `Protocol`, for the reason given in
    ``synthetic_game.as_value_function``.
    """
    return cast("ImageDecomposer", stub)


@pytest.fixture(scope="session")
def template() -> ImageIndices:
    """One real decomposition of the synthetic 4-player game."""
    game, _, _ = build_game()
    return compute_indices(game, order=2)


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


def layout_of(config: ExperimentConfig) -> RunLayout:
    """The run directory the production code will use for this config."""
    return RunLayout.for_run(config.name, config_fingerprint(config))


def stored_images(config: ExperimentConfig) -> list[int]:
    """Image indices that have an output file, in filename order."""
    values = layout_of(config).values_dir
    return [load_indices(path).image_index for path in sorted(values.glob("*.npz"))]


def execute_array(
    config: ExperimentConfig, decomposer: FakeDecomposer, num_tasks: int
) -> list[TaskReport]:
    """Run a whole array in-process, one task after another."""
    return [
        run_task(config, task_id, num_tasks, decomposer=as_decomposer(decomposer))
        for task_id in range(num_tasks)
    ]


def with_field(config: ExperimentConfig, **changes: object) -> ExperimentConfig:
    """A copy of ``config`` with some fields replaced."""
    return ExperimentConfig(**(config.to_dict() | changes))


# --------------------------------------------------------------------------
# image_label
# --------------------------------------------------------------------------


def test_image_label_is_zero_padded() -> None:
    assert image_label(42) == "image-00042"


def test_image_label_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        image_label(-1)


def test_image_labels_are_unique_and_sort_like_the_indices() -> None:
    """Filename order must equal image order, or `verify` reports nonsense."""
    labels = [image_label(index) for index in range(10000)]
    assert len(set(labels)) == 10000
    assert labels == sorted(labels)


# --------------------------------------------------------------------------
# selected_images
# --------------------------------------------------------------------------


def test_zero_max_images_selects_the_whole_split(config: ExperimentConfig) -> None:
    assert list(selected_images(config, 2135)) == list(range(2135))


def test_max_images_takes_a_prefix(config: ExperimentConfig) -> None:
    """A prefix, not a sample -- a rehearsal device, never a reported statistic."""
    assert list(selected_images(with_field(config, max_images=5), 2135)) == [0, 1, 2, 3, 4]


def test_max_images_larger_than_the_split_is_clamped(config: ExperimentConfig) -> None:
    assert list(selected_images(with_field(config, max_images=99), 3)) == [0, 1, 2]


def test_selected_images_rejects_an_empty_split(config: ExperimentConfig) -> None:
    with pytest.raises(ValueError, match="split_size must be >= 1"):
        selected_images(config, 0)


# --------------------------------------------------------------------------
# shard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_tasks", [1, 2, 3, 5, 7, 12, 13, 50])
def test_shards_partition_the_images_exactly(num_tasks: int) -> None:
    """Every image is owned by exactly one task: no gaps, no duplicates."""
    images = list(range(SPLIT_SIZE))
    shards = [shard(images, task_id, num_tasks) for task_id in range(num_tasks)]
    flat = [index for part in shards for index in part]
    assert sorted(flat) == images
    assert len(flat) == len(set(flat))


@pytest.mark.parametrize("num_tasks", [2, 3, 5])
def test_each_shard_is_ascending(num_tasks: int) -> None:
    images = list(range(SPLIT_SIZE))
    for task_id in range(num_tasks):
        part = shard(images, task_id, num_tasks)
        assert part == sorted(part)


def test_shards_are_strided_not_contiguous() -> None:
    """Per-image cost spans three orders of magnitude; blocks would pile it up."""
    assert shard(list(range(12)), 0, 4) == [0, 4, 8]
    assert shard(list(range(12)), 3, 4) == [3, 7, 11]


def test_a_shard_may_be_empty_when_tasks_outnumber_images() -> None:
    assert shard([0, 1], 5, 8) == []


def test_shard_rejects_a_non_positive_array_size() -> None:
    with pytest.raises(ValueError, match="num_tasks must be >= 1"):
        shard([0, 1], 0, 0)


@pytest.mark.parametrize("task_id", [-1, 4, 99])
def test_shard_rejects_a_task_id_outside_the_array(task_id: int) -> None:
    """A SLURM array indexed from 1 must fail loudly, not silently drop work."""
    with pytest.raises(ValueError, match=r"task_id must be in \[0, 4\)"):
        shard(list(range(12)), task_id, 4)


# --------------------------------------------------------------------------
# run_task
# --------------------------------------------------------------------------


def test_task_requires_prepare_first(
    config: ExperimentConfig,
    config_file: Path,  # noqa: ARG001
    template: ImageIndices,
) -> None:
    """Tasks must not create the run directory: N of them would race."""
    with pytest.raises(FileNotFoundError, match="run the 'prepare' subcommand"):
        run_task(config, 0, 1, decomposer=as_decomposer(FakeDecomposer(template)))


def test_one_task_writes_one_loadable_file_per_image(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    report = run_task(prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template)))

    assert report.ok
    assert report.written == tuple(range(SPLIT_SIZE))
    assert stored_images(prepared) == list(range(SPLIT_SIZE))


def test_stored_file_is_named_after_the_image_it_holds(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    """The one mislabelling no downstream number could ever reveal."""
    run_task(prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template)))
    for path in layout_of(prepared).values_dir.glob("*.npz"):
        assert image_label(load_indices(path).image_index) == path.stem


def test_a_whole_array_covers_the_split(prepared: ExperimentConfig, template: ImageIndices) -> None:
    fake = FakeDecomposer(template)
    reports = execute_array(prepared, fake, num_tasks=5)

    assert all(report.ok for report in reports)
    assert sorted(fake.seen) == list(range(SPLIT_SIZE))
    assert stored_images(prepared) == list(range(SPLIT_SIZE))


def test_resuming_skips_finished_images_without_recomputing(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    run_task(prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template)))

    resumed = FakeDecomposer(template)
    report = run_task(prepared, 0, 1, decomposer=as_decomposer(resumed))

    assert resumed.seen == []
    assert report.written == ()
    assert report.skipped == tuple(range(SPLIT_SIZE))
    assert report.ok


def test_resuming_after_a_partial_array_completes_the_run(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    """A node failure must cost only its own task's unfinished images."""
    run_task(prepared, 1, 3, decomposer=as_decomposer(FakeDecomposer(template)))

    resumed = FakeDecomposer(template)
    execute_array(prepared, resumed, num_tasks=3)

    assert sorted(resumed.seen) == [0, 2, 3, 5, 6, 8, 9, 11]
    assert stored_images(prepared) == list(range(SPLIT_SIZE))


def test_an_existing_file_is_never_overwritten(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    run_task(prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template)))
    path = layout_of(prepared).values_dir / "image-00000.npz"
    before = path.read_bytes()

    run_task(prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template)))

    assert path.read_bytes() == before


def test_a_failing_image_does_not_take_the_others_with_it(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    report = run_task(
        prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template, fail_on={3, 7}))
    )

    assert not report.ok
    assert set(report.failed) == {3, 7}
    assert report.failed[3] == "RuntimeError: image 3 is unreadable"
    assert stored_images(prepared) == [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]


def test_a_failing_image_leaves_no_file_behind(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    run_task(prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template, fail_on={3})))
    assert not (layout_of(prepared).values_dir / "image-00003.npz").exists()


def test_a_failed_image_is_retried_on_resubmission(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    """A failure must not be remembered as "done", or the run can never heal."""
    run_task(prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template, fail_on={3})))

    retried = FakeDecomposer(template)
    report = run_task(prepared, 0, 1, decomposer=as_decomposer(retried))

    assert retried.seen == [3]
    assert report.written == (3,)
    assert report.ok


def test_a_result_claiming_the_wrong_image_is_rejected(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    """Storing image 9's values as image 0 would be undetectable afterwards."""
    report = run_task(
        prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template, claim_image=9))
    )

    assert not report.ok
    assert set(report.failed) == set(range(SPLIT_SIZE)) - {9}
    assert "decomposed image 9 while working on 0" in report.failed[0]
    # Image 9 is the one case where the claim is true, and it is the one file
    # written -- the guard rejects the mismatch, not the decomposer.
    assert stored_images(prepared) == [9]


def test_an_interrupted_write_leaves_nothing_at_all(
    prepared: ExperimentConfig, template: ImageIndices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Walltime mid-write must not leave a truncated file a resume would skip."""

    def die(path: Path, result: ImageIndices) -> None:  # noqa: ARG001
        path.write_bytes(b"PK\x03\x04 truncated")
        msg = "killed mid-write"
        raise OSError(msg)

    monkeypatch.setattr("food_shapley.run.save_indices", die)
    report = run_task(prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template)))

    assert not report.ok
    assert list(layout_of(prepared).values_dir.iterdir()) == []


def test_the_task_writes_its_report(prepared: ExperimentConfig, template: ImageIndices) -> None:
    run_task(prepared, 2, 5, decomposer=as_decomposer(FakeDecomposer(template, fail_on={2})))

    path = layout_of(prepared).logs_dir / "task-00002.json"
    payload: dict[str, Any] = json.loads(path.read_text())
    assert payload["task_id"] == 2
    assert payload["num_tasks"] == 5
    assert payload["assigned"] == [2, 7]
    assert payload["written"] == [7]
    assert payload["failed"] == {"2": "RuntimeError: image 2 is unreadable"}
    assert "hostname" in payload


def test_an_empty_task_is_not_a_failure(prepared: ExperimentConfig, template: ImageIndices) -> None:
    """An array larger than the split is wasteful, not wrong."""
    report = run_task(prepared, 40, 50, decomposer=as_decomposer(FakeDecomposer(template)))

    assert report.assigned == ()
    assert report.ok


def test_max_images_limits_what_the_array_does(
    config: ExperimentConfig,
    config_file: Path,  # noqa: ARG001
    template: ImageIndices,
) -> None:
    limited = with_field(config, max_images=4)
    run_prepare(limited)
    fake = FakeDecomposer(template)

    execute_array(limited, fake, num_tasks=3)

    assert sorted(fake.seen) == [0, 1, 2, 3]
    assert stored_images(limited) == [0, 1, 2, 3]


def test_max_images_gets_its_own_run_directory(config: ExperimentConfig) -> None:
    """A 4-image rehearsal must not be mistaken for the full run."""
    assert layout_of(with_field(config, max_images=4)).root != layout_of(config).root


# --------------------------------------------------------------------------
# Independence from how the work was divided
# --------------------------------------------------------------------------


def test_each_image_is_seeded_from_its_own_label(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    fake = FakeDecomposer(template)
    run_task(prepared, 0, 1, decomposer=as_decomposer(fake))

    for index, draw in fake.draws.items():
        random.seed(derive_seed(prepared.seed, image_label(index)))
        assert draw == random.random()  # noqa: S311 - not cryptographic.


def test_the_run_is_identical_at_any_array_size(
    config: ExperimentConfig,
    config_file: Path,  # noqa: ARG001
    tmp_path: Path,
    template: ImageIndices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resubmitting with a different --array resumes; it does not re-experiment.

    The seeds drawn inside each decomposition and the arrays written to disk
    must both be functions of the image alone. If either depended on the task,
    a re-run at a different array size would be a second, subtly different
    experiment wearing the first one's directory name.
    """

    def execute(label: str, num_tasks: int) -> tuple[dict[int, float], dict[str, bytes]]:
        monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path / label))
        run_prepare(config)
        fake = FakeDecomposer(template)
        execute_array(config, fake, num_tasks)
        values = layout_of(config).values_dir
        stored = {
            path.name: load_indices(path).values.tobytes() for path in sorted(values.glob("*.npz"))
        }
        return fake.draws, stored

    one_draws, one_stored = execute("single", 1)
    many_draws, many_stored = execute("many", 7)

    assert one_draws == many_draws
    assert one_stored == many_stored
    assert len(one_stored) == SPLIT_SIZE


# --------------------------------------------------------------------------
# run_verify
# --------------------------------------------------------------------------


@pytest.fixture
def complete(prepared: ExperimentConfig, template: ImageIndices) -> ExperimentConfig:
    """A run whose every image is present and consistent."""
    run_task(prepared, 0, 1, decomposer=as_decomposer(FakeDecomposer(template)))
    return prepared


def test_verify_passes_for_a_complete_run(complete: ExperimentConfig) -> None:
    summary = run_verify(complete, SPLIT_SIZE)

    assert summary.problems == ()
    assert summary.missing == ()
    assert summary.present == SPLIT_SIZE
    assert summary.coverage == 1.0


def test_verify_reports_the_run_it_read(complete: ExperimentConfig) -> None:
    summary = run_verify(complete, SPLIT_SIZE)

    assert summary.player_counts == {4: SPLIT_SIZE}
    assert summary.targets == {"pizza": SPLIT_SIZE}
    assert summary.total_coalitions == SPLIT_SIZE * 2**4
    assert summary.max_abs_residual < 1e-9


def test_verify_names_every_missing_image(
    prepared: ExperimentConfig, template: ImageIndices
) -> None:
    """3 % silently dropped is the failure this command exists to prevent."""
    run_task(prepared, 0, 3, decomposer=as_decomposer(FakeDecomposer(template)))
    summary = run_verify(prepared, SPLIT_SIZE)

    assert summary.missing == (1, 2, 4, 5, 7, 8, 10, 11)
    assert len(summary.problems) == 8
    assert "image-00001: missing output" in summary.problems[0]
    assert summary.coverage == pytest.approx(4 / 12)


def test_verify_rejects_an_unreadable_file(complete: ExperimentConfig) -> None:
    (layout_of(complete).values_dir / "image-00005.npz").write_bytes(b"not an npz")
    summary = run_verify(complete, SPLIT_SIZE)

    assert summary.present == SPLIT_SIZE - 1
    assert len(summary.problems) == 1
    assert "image-00005: unreadable" in summary.problems[0]


def test_verify_rejects_a_file_holding_another_image(
    complete: ExperimentConfig, template: ImageIndices
) -> None:
    path = layout_of(complete).values_dir / "image-00005.npz"
    save_indices(path, replace(template, image_index=6))
    summary = run_verify(complete, SPLIT_SIZE)

    assert len(summary.problems) == 1
    assert "image-00005: file holds image 6" in summary.problems[0]


def test_verify_rejects_a_stored_efficiency_violation(
    complete: ExperimentConfig, template: ImageIndices
) -> None:
    """The residual on disk is the one a reader trusts, so re-check it."""
    path = layout_of(complete).values_dir / "image-00005.npz"
    save_indices(path, replace(template, image_index=5, efficiency_residual=1e-3))
    summary = run_verify(complete, SPLIT_SIZE)

    assert summary.present == SPLIT_SIZE - 1
    assert len(summary.problems) == 1
    assert "efficiency residual 1.000e-03 exceeds 1.0e-09" in summary.problems[0]


def test_verify_honours_an_explicit_tolerance(
    complete: ExperimentConfig, template: ImageIndices
) -> None:
    path = layout_of(complete).values_dir / "image-00005.npz"
    save_indices(path, replace(template, image_index=5, efficiency_residual=1e-3))

    assert run_verify(complete, SPLIT_SIZE, tolerance=1e-2).problems == ()


def test_verify_checks_only_the_images_the_config_covers(
    config: ExperimentConfig,
    config_file: Path,  # noqa: ARG001
    template: ImageIndices,
) -> None:
    limited = with_field(config, max_images=4)
    run_prepare(limited)
    run_task(limited, 0, 1, decomposer=as_decomposer(FakeDecomposer(template)))

    summary = run_verify(limited, SPLIT_SIZE)
    assert summary.expected == 4
    assert summary.problems == ()


def test_verify_reports_an_empty_run_as_entirely_missing(prepared: ExperimentConfig) -> None:
    summary = run_verify(prepared, SPLIT_SIZE)

    assert summary.present == 0
    assert summary.coverage == 0.0
    assert len(summary.problems) == SPLIT_SIZE


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


@pytest.fixture
def fake_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `load_split` return something of the right length, and no more."""
    monkeypatch.setattr("food_shapley.run.load_split", lambda _split: [None] * SPLIT_SIZE)


def test_cli_prepare_is_idempotent_with_allow_existing(config_file: Path) -> None:
    base = ["--config", str(config_file)]
    assert main([*base, "prepare", "--allow-existing"]) == 0
    assert main([*base, "prepare", "--allow-existing"]) == 0


def test_cli_prepare_without_allow_existing_still_raises(config_file: Path) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    with pytest.raises(FileExistsError):
        main([*base, "prepare"])


def test_cli_task_reads_the_slurm_array_variables(
    config_file: Path,
    template: ImageIndices,
    fake_split: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    fake = FakeDecomposer(template)
    monkeypatch.setattr("food_shapley.run.ImageDecomposer", lambda *_a, **_k: fake)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "4")

    assert main([*base, "task"]) == 0
    assert fake.seen == [1, 5, 9]


def test_cli_task_exits_non_zero_when_an_image_fails(
    config_file: Path,
    template: ImageIndices,
    fake_split: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SLURM must mark the task failed, or nobody will look at the log."""
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    fake = FakeDecomposer(template, fail_on={1})
    monkeypatch.setattr("food_shapley.run.ImageDecomposer", lambda *_a, **_k: fake)

    assert main([*base, "task", "--task-id", "1", "--num-tasks", "4"]) == 1


def test_cli_task_without_any_array_size_exits(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    monkeypatch.delenv("SLURM_ARRAY_TASK_COUNT", raising=False)
    with pytest.raises(SystemExit, match="SLURM_ARRAY_TASK_COUNT is not set"):
        main([*base, "task", "--task-id", "0"])


def test_cli_task_rejects_a_non_integer_array_variable(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0-31")
    with pytest.raises(SystemExit, match="not an integer"):
        main([*base, "task", "--num-tasks", "4"])


def test_cli_verify_succeeds_on_a_complete_run(
    config_file: Path,
    template: ImageIndices,
    fake_split: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    fake = FakeDecomposer(template)
    monkeypatch.setattr("food_shapley.run.ImageDecomposer", lambda *_a, **_k: fake)
    main([*base, "task", "--task-id", "0", "--num-tasks", "1"])

    assert main([*base, "verify"]) == 0


def test_cli_verify_fails_on_an_incomplete_run(
    config_file: Path,
    template: ImageIndices,
    fake_split: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    fake = FakeDecomposer(template)
    monkeypatch.setattr("food_shapley.run.ImageDecomposer", lambda *_a, **_k: fake)
    main([*base, "task", "--task-id", "0", "--num-tasks", "3"])

    assert main([*base, "verify"]) == 1


def test_cli_verify_writes_the_summary(
    config: ExperimentConfig,
    config_file: Path,
    template: ImageIndices,
    fake_split: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = ["--config", str(config_file)]
    main([*base, "prepare"])
    fake = FakeDecomposer(template)
    monkeypatch.setattr("food_shapley.run.ImageDecomposer", lambda *_a, **_k: fake)
    main([*base, "task", "--task-id", "0", "--num-tasks", "1"])
    main([*base, "verify"])

    payload: dict[str, Any] = json.loads((layout_of(config).root / "summary.json").read_text())
    assert payload["expected"] == SPLIT_SIZE
    assert payload["present"] == SPLIT_SIZE
    assert payload["coverage"] == 1.0
    assert payload["problems"] == []
    assert payload["player_counts"] == {"4": SPLIT_SIZE}
    assert np.isfinite(payload["max_abs_residual"])
