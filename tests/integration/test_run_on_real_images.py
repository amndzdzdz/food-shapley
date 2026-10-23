"""One real shard, end to end: prepare, task, task, verify.

Marked ``gpu`` and ``slow`` like the other integration files: this loads the
~600 MB CLIP checkpoint and the FoodSeg103 split, which `CLAUDE.md` keeps out
of the ordinary suite.

`tests/unit/test_run.py` proves the bookkeeping with a stand-in decomposer,
and `test_indices_on_real_images.py` proves the decomposition itself on real
images. Neither can show that the two fit together. Three things are only
observable when they do:

* the array path and the single-image path produce the *same* numbers, so the
  decompositions already validated one image at a time are the ones a cluster
  run will store;
* a real ``.npz`` written through the atomic-rename path reloads as what was
  computed;
* ``verify`` reads real player counts, real dish targets and real efficiency
  residuals off disk, rather than the synthetic game's uniform four players.

Three images, two tasks -- enough to exercise an array whose task count does
not divide its image count, cheap enough to run on a CPU.

Run it deliberately:

    uv run pytest tests/integration/test_run_on_real_images.py -m gpu
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from food_shapley.config import ExperimentConfig, config_fingerprint, load_config
from food_shapley.download import verify_split
from food_shapley.indices import DEFAULT_EFFICIENCY_TOLERANCE, load_indices
from food_shapley.paths import ARTIFACT_ROOT_ENV, RunLayout
from food_shapley.prompts import verify_dish_names
from food_shapley.run import (
    ImageDecomposer,
    TaskReport,
    image_label,
    run_prepare,
    run_task,
    run_verify,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]

CONFIG = Path("configs/week01-clip-blur.toml")

#: Images decomposed here. A prefix, because that is what `max_images` takes.
N_IMAGES = 3
#: Deliberately does not divide `N_IMAGES`: task 0 gets two images, task 1 one.
NUM_TASKS = 2


@pytest.fixture(scope="module")
def settings() -> ExperimentConfig:
    """The real Week-1 configuration, truncated to a handful of images."""
    base = load_config(CONFIG)
    return ExperimentConfig(
        **(base.to_dict() | {"name": "integration-run", "max_images": N_IMAGES})
    )


@pytest.fixture(scope="module")
def artifact_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Redirect run output into a temporary directory for the whole module."""
    root = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(ARTIFACT_ROOT_ENV, str(root))
        yield root


@pytest.fixture(scope="module")
def decomposer(settings: ExperimentConfig) -> ImageDecomposer:
    """The real split and the real checkpoint, loaded once."""
    if verify_split(settings.split):
        pytest.skip("FoodSeg103 split not downloaded; run food_shapley.download")
    if verify_dish_names():
        pytest.skip("Food-101 prompt set not downloaded; run food_shapley.prompts")
    return ImageDecomposer(settings, device="cpu")


@pytest.fixture(scope="module")
def executed(
    settings: ExperimentConfig,
    decomposer: ImageDecomposer,
    artifact_root: Path,  # noqa: ARG001 - redirects where the run is written.
) -> list[TaskReport]:
    """Run the whole array once; every test below reads its output."""
    run_prepare(settings)
    return [
        run_task(settings, task_id, NUM_TASKS, decomposer=decomposer)
        for task_id in range(NUM_TASKS)
    ]


def layout_of(settings: ExperimentConfig) -> RunLayout:
    """The run directory the production code will use for this config."""
    return RunLayout.for_run(settings.name, config_fingerprint(settings))


def test_the_array_covers_every_image_exactly_once(executed: list[TaskReport]) -> None:
    written = [index for report in executed for index in report.written]

    assert all(report.ok for report in executed)
    assert sorted(written) == list(range(N_IMAGES))
    assert [len(report.assigned) for report in executed] == [2, 1]


def test_every_image_produced_a_loadable_file(
    executed: list[TaskReport],  # noqa: ARG001 - ordering dependency.
    settings: ExperimentConfig,
) -> None:
    values = layout_of(settings).values_dir

    for index in range(N_IMAGES):
        stored = load_indices(values / f"{image_label(index)}.npz")
        assert stored.image_index == index
        assert len(stored.values) == 2**stored.n_players
        assert np.isfinite(stored.values).all()


def test_the_array_path_computes_what_the_single_image_path_does(
    executed: list[TaskReport],  # noqa: ARG001 - ordering dependency.
    settings: ExperimentConfig,
    decomposer: ImageDecomposer,
) -> None:
    """The stored numbers must be the ones already validated one image at a time.

    Asserted bit for bit, not approximately. `ImageDecomposer` fixes the
    chunk size, so both evaluations batch the coalitions identically and
    CLIP's batch non-invariance (W01-11) cannot enter. An approximate
    assertion here would hide exactly the kind of drift that makes a run
    irreproducible.
    """
    direct = decomposer.decompose(0)
    stored = load_indices(layout_of(settings).values_dir / f"{image_label(0)}.npz")

    assert stored.player_names == direct.player_names
    assert stored.ingredient_indices == direct.ingredient_indices
    assert stored.target == direct.target
    assert stored.baseline == direct.baseline
    assert stored.efficiency_residual == direct.efficiency_residual
    np.testing.assert_array_equal(stored.values, direct.values)
    np.testing.assert_array_equal(stored.coalitions, direct.coalitions)
    np.testing.assert_array_equal(stored.shapley.values, direct.shapley.values)
    np.testing.assert_array_equal(stored.interactions.values, direct.interactions.values)
    np.testing.assert_array_equal(stored.moebius.values, direct.moebius.values)


def test_verify_passes_and_reports_the_real_run(
    executed: list[TaskReport],  # noqa: ARG001 - ordering dependency.
    settings: ExperimentConfig,
    decomposer: ImageDecomposer,
) -> None:
    summary = run_verify(settings, decomposer.split_size)

    assert summary.problems == ()
    assert summary.expected == N_IMAGES
    assert summary.present == N_IMAGES
    assert summary.coverage == 1.0
    assert summary.max_abs_residual <= DEFAULT_EFFICIENCY_TOLERANCE
    # Real images, so these are the split's own structure rather than a
    # fixture's: a handful of distinct player counts, each at least 2.
    assert sum(summary.player_counts.values()) == N_IMAGES
    assert min(summary.player_counts) >= 2
    assert summary.total_coalitions == sum(
        2**players * count for players, count in summary.player_counts.items()
    )


def test_verify_notices_a_truncated_file(
    executed: list[TaskReport],  # noqa: ARG001 - ordering dependency.
    settings: ExperimentConfig,
    decomposer: ImageDecomposer,
) -> None:
    """The failure the atomic write exists to prevent, confirmed detectable.

    A file cut in half is not merely unreadable in principle -- `np.load` has
    to actually raise on it, or `verify` would pass a run that a later
    analysis reads short. Restored afterwards so the resume test below still
    sees a complete run.
    """
    path = layout_of(settings).values_dir / f"{image_label(1)}.npz"
    intact = path.read_bytes()
    try:
        path.write_bytes(intact[: len(intact) // 2])
        problems = run_verify(settings, decomposer.split_size).problems
    finally:
        path.write_bytes(intact)

    assert len(problems) == 1
    assert "image-00001: unreadable" in problems[0]
    assert run_verify(settings, decomposer.split_size).problems == ()


def test_resubmitting_the_array_recomputes_nothing(
    executed: list[TaskReport],  # noqa: ARG001 - ordering dependency.
    settings: ExperimentConfig,
    decomposer: ImageDecomposer,
) -> None:
    """A resubmission after a partial failure must resume, not redo."""
    values = layout_of(settings).values_dir
    before = {path.name: path.read_bytes() for path in sorted(values.glob("*.npz"))}

    reports = [
        run_task(settings, task_id, NUM_TASKS, decomposer=decomposer)
        for task_id in range(NUM_TASKS)
    ]

    assert all(report.written == () for report in reports)
    assert sorted(index for report in reports for index in report.skipped) == list(range(N_IMAGES))
    assert {path.name: path.read_bytes() for path in sorted(values.glob("*.npz"))} == before


def test_a_different_array_size_resumes_the_same_run(
    executed: list[TaskReport],  # noqa: ARG001 - ordering dependency.
    settings: ExperimentConfig,
    decomposer: ImageDecomposer,
) -> None:
    """Editing --array and resubmitting must not start a second experiment."""
    report = run_task(settings, 0, 1, decomposer=decomposer)

    assert report.assigned == tuple(range(N_IMAGES))
    assert report.skipped == tuple(range(N_IMAGES))
    assert report.written == ()
