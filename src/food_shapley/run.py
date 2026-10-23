"""The full-split run: a SLURM job array over every image of a split.

`PLAN.md` Milestone 1, step 5 -- "scale to the full run as a SLURM job array
across the ~2,000--7,118 test images; save each image's game values as
``.npz``" -- with its own note that this is *an orchestration day, not a
compute-bound day*. The CLIP calls are fast and embarrassingly parallel; what
actually goes wrong is bookkeeping. This module is therefore about bookkeeping,
and the research computation it performs is one line:
:func:`~food_shapley.indices.compute_indices`, per image.

Three subcommands, mirroring :mod:`food_shapley.selftest`, which rehearsed
exactly this shape without a dataset or a model::

    python -m food_shapley.run --config <cfg> prepare
    python -m food_shapley.run --config <cfg> task --num-tasks 64
    python -m food_shapley.run --config <cfg> verify

What the array guarantees
-------------------------
**One output file per image, never one per task.** ``values/image-00042.npz``
is written by whichever task owns image 42, and by no other. Tasks therefore
cannot collide, and a partial failure is re-runnable at image granularity
rather than all-or-nothing.

**A stored file does not depend on how the work was divided.** The image set
is a function of the configuration alone (:func:`selected_images`), the seed
of an image is derived from its own label and not from its task
(:func:`image_label`), and the model is deterministic at fixed batching
(W01-10, W01-11). Re-running the same config at ``--num-tasks 8`` and at
``--num-tasks 400`` produces the same files. That is what makes a re-run after
a node failure a *resumption* rather than a second, subtly different
experiment.

**Writes are atomic.** A task killed mid-write -- walltime, preemption, an
OOM -- would otherwise leave a truncated ``.npz`` that a resumed run skips as
"already done" and that every later analysis silently reads short. Each file
is written to a temporary name in the same directory and renamed into place,
so a file at its final path is always complete.

**Nothing is ever overwritten.** An image whose output already exists is
skipped, never recomputed and never replaced. To redo a run, change the
config -- the fingerprint puts it in a different directory -- or delete the
old one deliberately.

**A failing image fails loudly but alone.** One image that cannot be
decomposed -- an efficiency violation, an unreadable row -- is recorded in the
task's report and the remaining images still run, because throwing away 2,000
good decompositions to punish one bad one helps nobody. The task then exits
non-zero, so SLURM marks it failed and
:func:`run_verify` can enumerate what is missing. The failure is never
swallowed.

Why ``verify`` is not optional
------------------------------
`CLAUDE.md`'s rule for array jobs: *an array that silently drops 3 % of its
tasks looks like one that succeeded*. :func:`run_verify` re-opens every
expected file, checks it holds the image it claims to, and re-checks the
efficiency residual that :func:`~food_shapley.indices.compute_indices` already
enforced -- because the residual stored on disk is the one a reader will
trust, and the check costs milliseconds. It writes ``summary.json`` and exits
non-zero on any problem at all. The "≥ 99 % coverage" success criterion in
`docs/week-01.md` is the bar for *reporting* a run; it is not a licence for
this command to pass with images missing.

Cost
----
Measured in W01-11: 91,828 coalitions over the 2,135-image test split, 2--11
players per image. One CLIP forward pass each, in chunks of
:data:`~food_shapley.game.DEFAULT_CHUNK_SIZE`. The per-image cost varies by
three orders of magnitude with the player count, which is why
:func:`shard` deals images out in strides rather than in contiguous blocks:
contiguous blocks would hand one unlucky task a run of 11-player images and
leave the array waiting on it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from food_shapley.config import ExperimentConfig, config_fingerprint, load_config
from food_shapley.dataset import load_split
from food_shapley.game import DEFAULT_CHUNK_SIZE, IngredientGame
from food_shapley.indices import (
    DEFAULT_EFFICIENCY_TOLERANCE,
    ImageIndices,
    compute_indices,
    load_indices,
    save_indices,
)
from food_shapley.paths import RunLayout
from food_shapley.provenance import write_run_metadata
from food_shapley.seeding import derive_seed, seed_everything
from food_shapley.segments import build_players
from food_shapley.value import DishValueFunction

__all__ = [
    "ImageDecomposer",
    "RunSummary",
    "TaskReport",
    "image_label",
    "main",
    "run_prepare",
    "run_task",
    "run_verify",
    "selected_images",
    "shard",
]

#: SLURM variables recorded in each task's report, so that a task which
#: misbehaved can be traced to a node without consulting the job log.
_SLURM_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "CUDA_VISIBLE_DEVICES",
)


def image_label(index: int) -> str:
    """Return the stable label identifying one image's work.

    Used both as the output filename stem and as the :func:`derive_seed`
    label. Deriving the seed from the *image* rather than from the array task
    is deliberate: it makes a stored file independent of how the split was
    divided, so re-running with a different array size resumes instead of
    producing a second, slightly different experiment.

    Args:
        index: Position of the image in the dataset split.

    Returns:
        A zero-padded label such as ``"image-00042"``.

    Raises:
        ValueError: If ``index`` is negative.
    """
    if index < 0:
        msg = f"image index must be non-negative, got {index}"
        raise ValueError(msg)
    return f"image-{index:05d}"


def selected_images(config: ExperimentConfig, split_size: int) -> range:
    """Return the images this run covers, as a function of the config alone.

    Args:
        config: The run configuration; ``max_images`` decides the extent.
        split_size: Number of images in the loaded split.

    Returns:
        The image positions to decompose. ``max_images = 0`` means all of
        them; anything else takes that many from the start -- a prefix, not a
        sample, and so a rehearsal device rather than a basis for any reported
        statistic.

    Raises:
        ValueError: If ``split_size`` is not positive.
    """
    if split_size < 1:
        msg = f"split_size must be >= 1, got {split_size}"
        raise ValueError(msg)
    limit = split_size if config.max_images == 0 else min(config.max_images, split_size)
    return range(limit)


def shard(images: Sequence[int], task_id: int, num_tasks: int) -> list[int]:
    """Deal ``images`` out to array task ``task_id`` of ``num_tasks``.

    Strided rather than contiguous. The per-image cost is ``2^n_players`` CLIP
    forward passes and the split runs from 2 to 11 players (W01-11), a span of
    three orders of magnitude; contiguous blocks would give one task a run of
    expensive images while the rest of the array idled. Striding does not
    balance the load exactly -- nothing short of measuring every image first
    would -- but it removes the correlation between cost and position.

    Args:
        images: The run's image positions, in order.
        task_id: Zero-based array task index.
        num_tasks: Total number of array tasks.

    Returns:
        This task's images, in ascending order. May be empty if the array has
        more tasks than the run has images.

    Raises:
        ValueError: If ``num_tasks`` is not positive, or ``task_id`` is
            outside ``[0, num_tasks)``. An out-of-range task would silently
            process nothing, which looks exactly like a task whose images were
            all already done.
    """
    if num_tasks < 1:
        msg = f"num_tasks must be >= 1, got {num_tasks}"
        raise ValueError(msg)
    if not 0 <= task_id < num_tasks:
        msg = f"task_id must be in [0, {num_tasks}), got {task_id}"
        raise ValueError(msg)
    return list(images[task_id::num_tasks])


class ImageDecomposer:
    """Loads the split and the model once, then decomposes images on demand.

    One per process. The checkpoint is ~600 MB and the split is thousands of
    rows, so loading either per image would dominate the run.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        device: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Load the split and the value function named by ``config``.

        Args:
            config: The run configuration.
            device: Torch device; ``None`` selects automatically.
            chunk_size: Coalitions per forward pass. A throughput knob, not an
                experiment setting; see
                :data:`~food_shapley.game.DEFAULT_CHUNK_SIZE`.
        """
        self._config = config
        self._chunk_size = chunk_size
        self._split = load_split(config.split)
        self._value_fn = DishValueFunction.from_config(config, device=device)

    @property
    def split_size(self) -> int:
        """Number of images in the loaded split."""
        return len(self._split)

    def decompose(self, index: int) -> ImageIndices:
        """Build one image's game and compute its exact indices.

        Args:
            index: Position of the image in the split.

        Returns:
            The image's decomposition, with its efficiency residual recorded.
        """
        players = build_players(
            self._split[index],
            self._split.names,
            image_size=self._config.image_size,
            max_players=self._config.max_players,
            min_area_fraction=self._config.min_area_fraction,
            background_player=self._config.background_is_player,
        )
        game = IngredientGame.for_image(
            players, self._value_fn, self._config, chunk_size=self._chunk_size
        )
        return compute_indices(game, order=self._config.interaction_order)


@dataclass(frozen=True, slots=True)
class TaskReport:
    """What one array task did, written to ``logs/task-<id>.json``.

    Attributes:
        task_id: Zero-based array task index.
        num_tasks: Size of the array it belonged to.
        assigned: Images this task owned.
        written: Images it decomposed and stored.
        skipped: Images whose output already existed, from an earlier attempt.
        failed: Images that raised, mapped to the error message. Recorded
            rather than re-raised so that one bad image does not discard the
            task's other results -- but see :attr:`ok`, which makes the task
            exit non-zero anyway.
        seconds: Wall-clock time spent decomposing, excluding model loading.
    """

    task_id: int
    num_tasks: int
    assigned: tuple[int, ...]
    written: tuple[int, ...]
    skipped: tuple[int, ...]
    failed: Mapping[int, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether every assigned image now has an output file."""
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return {
            "task_id": self.task_id,
            "num_tasks": self.num_tasks,
            "assigned": list(self.assigned),
            "written": list(self.written),
            "skipped": list(self.skipped),
            "failed": {str(index): message for index, message in self.failed.items()},
            "seconds": round(self.seconds, 3),
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "hostname": platform.node(),
            "slurm": {key: os.environ.get(key) for key in _SLURM_ENV_KEYS},
        }


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The state of a whole run, as read back off disk.

    Attributes:
        expected: Images the configuration says the run covers.
        present: Images with a readable output file holding the right image.
        missing: Images with no output file.
        problems: Human-readable description of everything wrong, one per
            line. Empty means the run is complete and internally consistent.
        max_abs_residual: Largest absolute efficiency residual over the files
            that loaded. Reported rather than merely checked, so that the
            run's numerical health is a number instead of a yes/no.
        total_coalitions: Sum of ``2^n_players`` over the present images --
            the size of the computation that was actually performed.
        player_counts: How many images had each player count.
        targets: How many images the model assigned to each dish. The dish is
            the model's own top-1, never a ground-truth label (W01-10).
    """

    expected: int
    present: int
    missing: tuple[int, ...]
    problems: tuple[str, ...]
    max_abs_residual: float
    total_coalitions: int
    player_counts: Mapping[int, int]
    targets: Mapping[str, int]

    @property
    def coverage(self) -> float:
        """Fraction of expected images that produced a usable file."""
        return self.present / self.expected if self.expected else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a JSON-serialisable dictionary."""
        return {
            "expected": self.expected,
            "present": self.present,
            "coverage": round(self.coverage, 6),
            "missing": list(self.missing),
            "problems": list(self.problems),
            "max_abs_residual": self.max_abs_residual,
            "total_coalitions": self.total_coalitions,
            "player_counts": {str(k): v for k, v in sorted(self.player_counts.items())},
            "targets": dict(sorted(self.targets.items(), key=lambda item: (-item[1], item[0]))),
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }


def _layout(config: ExperimentConfig) -> RunLayout:
    """Resolve the run layout; identical in every subcommand and every job."""
    return RunLayout.for_run(config.name, config_fingerprint(config))


def _atomic_save(path: Path, result: ImageIndices) -> None:
    """Write ``result`` to ``path`` so that the path is never half-written.

    A task killed by walltime or preemption part-way through a write would
    otherwise leave a truncated ``.npz``. The resumed run would see the file,
    skip the image, and every later analysis would read it short -- the exact
    silent-failure shape this module exists to prevent.

    The temporary name ends in ``.npz`` because :func:`numpy.savez_compressed`
    appends that suffix when it is absent, which would leave the real
    temporary file somewhere the rename never looks.
    """
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".npz")
    os.close(handle)
    partial = Path(name)
    try:
        save_indices(partial, result)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def run_prepare(
    config: ExperimentConfig, *, allow_existing: bool = False
) -> tuple[RunLayout, bool]:
    """Create the run directory and write its provenance record.

    Args:
        config: The configuration for this run.
        allow_existing: Treat an already-prepared run as success. Resubmitting
            a partially failed array is normal and should not look like a
            crash; the existing provenance record is left untouched.

    Returns:
        The layout, and whether this call created it.

    Raises:
        FileExistsError: If the run exists and ``allow_existing`` is False.
    """
    layout = _layout(config)
    if allow_existing and layout.metadata_file.exists():
        return layout, False
    return write_run_metadata(config), True


def run_task(
    config: ExperimentConfig,
    task_id: int,
    num_tasks: int,
    *,
    decomposer: ImageDecomposer,
    on_image: Callable[[int], None] | None = None,
) -> TaskReport:
    """Decompose this task's share of the run's images.

    Args:
        config: The configuration for this run.
        task_id: Zero-based array task index.
        num_tasks: Total number of array tasks.
        decomposer: The loaded split and model.
        on_image: Invoked with each image index just before it is decomposed,
            so that a long-running task reports progress into its SLURM log
            instead of looking hung.

    Returns:
        What the task did, also written to ``logs/task-<id>.json``.

    Raises:
        FileNotFoundError: If ``prepare`` has not been run for this config.
            Letting tasks create the run directory would race N ways and could
            write N provenance records.
    """
    layout = _layout(config)
    if not layout.metadata_file.exists():
        msg = (
            f"no run metadata at {layout.metadata_file}; "
            "run the 'prepare' subcommand on the login node before submitting the array"
        )
        raise FileNotFoundError(msg)

    assigned = shard(selected_images(config, decomposer.split_size), task_id, num_tasks)
    layout.values_dir.mkdir(parents=True, exist_ok=True)
    layout.logs_dir.mkdir(parents=True, exist_ok=True)

    written: list[int] = []
    skipped: list[int] = []
    failed: dict[int, str] = {}
    started = time.monotonic()

    for index in assigned:
        output = layout.values_dir / f"{image_label(index)}.npz"
        if output.exists():
            skipped.append(index)
            continue
        if on_image is not None:
            on_image(index)
        seed_everything(derive_seed(config.seed, image_label(index)))
        try:
            result = decomposer.decompose(index)
            if result.image_index != index:
                # The file is named after `index`, so a mismatch would store
                # one image's values under another's name -- undetectable
                # afterwards, because every stored array would be internally
                # consistent.
                msg = f"decomposed image {result.image_index} while working on {index}"
                raise ValueError(msg)
            _atomic_save(output, result)
        # Broad on purpose: one bad image must not discard the task's others.
        # The task still exits non-zero, and `verify` enumerates what is left.
        except Exception as error:
            failed[index] = f"{type(error).__name__}: {error}"
            continue
        written.append(index)

    report = TaskReport(
        task_id=task_id,
        num_tasks=num_tasks,
        assigned=tuple(assigned),
        written=tuple(written),
        skipped=tuple(skipped),
        failed=failed,
        seconds=time.monotonic() - started,
    )
    path = layout.logs_dir / f"task-{task_id:05d}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return report


def run_verify(
    config: ExperimentConfig,
    split_size: int,
    *,
    tolerance: float = DEFAULT_EFFICIENCY_TOLERANCE,
) -> RunSummary:
    """Re-open every expected output and check the run is complete and sound.

    Four things are checked per image, each of them a failure that no
    downstream analysis would notice on its own: the file exists, it loads,
    it holds the image its name claims, and its stored efficiency residual is
    still inside ``tolerance``. The last is not redundant with
    :func:`~food_shapley.indices.compute_indices`: the number on disk is the
    one a reader will trust, and re-checking it costs milliseconds.

    Args:
        config: The configuration for this run.
        split_size: Number of images in the split, used to resolve
            ``max_images = 0``.
        tolerance: Largest acceptable stored residual.

    Returns:
        The run's state, with every problem enumerated.
    """
    layout = _layout(config)
    expected = selected_images(config, split_size)

    missing: list[int] = []
    problems: list[str] = []
    player_counts: dict[int, int] = {}
    targets: dict[str, int] = {}
    max_abs_residual = 0.0
    total_coalitions = 0
    present = 0

    for index in expected:
        label = image_label(index)
        path = layout.values_dir / f"{label}.npz"
        if not path.is_file():
            missing.append(index)
            problems.append(f"{label}: missing output at {path}")
            continue
        try:
            stored = load_indices(path)
        # Broad on purpose: a corrupt file is a finding to report, not a
        # reason to abandon the other 2,000.
        except Exception as error:
            problems.append(f"{label}: unreadable ({type(error).__name__}: {error})")
            continue
        if stored.image_index != index:
            problems.append(f"{label}: file holds image {stored.image_index}")
            continue
        residual = abs(stored.efficiency_residual)
        if residual > tolerance:
            problems.append(f"{label}: efficiency residual {residual:.3e} exceeds {tolerance:.1e}")
            continue

        present += 1
        max_abs_residual = max(max_abs_residual, residual)
        total_coalitions += len(stored.values)
        player_counts[stored.n_players] = player_counts.get(stored.n_players, 0) + 1
        targets[stored.target.name] = targets.get(stored.target.name, 0) + 1

    return RunSummary(
        expected=len(expected),
        present=present,
        missing=tuple(missing),
        problems=tuple(problems),
        max_abs_residual=max_abs_residual,
        total_coalitions=total_coalitions,
        player_counts=player_counts,
        targets=targets,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m food_shapley.run",
        description="Decompose a whole split as a SLURM job array.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Run configuration (TOML).")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Create the run directory (login node, once).")
    prepare.add_argument(
        "--allow-existing",
        action="store_true",
        help="Succeed if the run directory already exists (for resubmissions).",
    )

    task = sub.add_parser("task", help="Decompose this task's images (compute node).")
    task.add_argument(
        "--task-id", type=int, default=None, help="Array task index; default $SLURM_ARRAY_TASK_ID."
    )
    task.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Array size; default $SLURM_ARRAY_TASK_COUNT. Must match --array.",
    )
    task.add_argument("--device", default=None, help="Torch device; default auto.")
    task.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Coalitions per forward pass; a throughput knob, not an experiment setting.",
    )

    sub.add_parser("verify", help="Check every expected image produced a usable file.")
    return parser


def _from_env(explicit: int | None, variable: str) -> int:
    """Resolve an array parameter from the flag or the SLURM environment.

    Raises:
        SystemExit: If neither source provides a usable integer.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get(variable)
    if raw is None:
        msg = f"no command-line value given and {variable} is not set"
        raise SystemExit(msg)
    try:
        return int(raw)
    except ValueError:
        msg = f"{variable} is not an integer: {raw!r}"
        raise SystemExit(msg) from None


def _command_task(config: ExperimentConfig, args: argparse.Namespace) -> int:
    """Run one array task and report what it did."""
    task_id = _from_env(args.task_id, "SLURM_ARRAY_TASK_ID")
    num_tasks = _from_env(args.num_tasks, "SLURM_ARRAY_TASK_COUNT")
    decomposer = ImageDecomposer(config, device=args.device, chunk_size=args.chunk_size)

    def announce(index: int) -> None:
        print(f"  image {index}", flush=True)

    report = run_task(config, task_id, num_tasks, decomposer=decomposer, on_image=announce)
    print(
        f"task {task_id}/{num_tasks}: {len(report.written)} written, "
        f"{len(report.skipped)} already done, {len(report.failed)} failed, "
        f"{report.seconds:.1f}s"
    )
    for index, message in report.failed.items():
        print(f"  image {index}: {message}", file=sys.stderr)
    return 0 if report.ok else 1


def _command_verify(config: ExperimentConfig) -> int:
    """Check the whole run and write ``summary.json``."""
    summary = run_verify(config, len(load_split(config.split)))
    layout = _layout(config)
    layout.root.mkdir(parents=True, exist_ok=True)
    (layout.root / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2) + "\n")

    print(f"images         {summary.present}/{summary.expected} ({summary.coverage:.2%})")
    print(f"coalitions     {summary.total_coalitions}")
    print(f"max |residual| {summary.max_abs_residual:.3e}")
    print(f"players        {dict(sorted(summary.player_counts.items()))}")
    if summary.problems:
        print(f"\nrun INCOMPLETE: {len(summary.problems)} problem(s):", file=sys.stderr)
        for problem in summary.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"\nrun complete: every one of {summary.expected} images is present and consistent")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero on any failure, so SLURM marks the job
        failed rather than leaving a silently incomplete run behind.
    """
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "prepare":
        layout, created = run_prepare(config, allow_existing=args.allow_existing)
        verb = "prepared" if created else "reusing existing"
        print(f"{verb} run directory: {layout.root}")
        return 0
    if args.command == "task":
        return _command_task(config, args)
    return _command_verify(config)


if __name__ == "__main__":
    raise SystemExit(main())
