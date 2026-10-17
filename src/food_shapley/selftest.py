"""Infrastructure self-test for the SLURM job-array mechanics.

`PLAN.md` Milestone 1, step 0: "submit one toy SLURM job end-to-end to confirm
the pipeline runs on a node ... get the SLURM submit/array mechanics working on
a trivial job now, before any real experiments depend on it".

This module is that trivial job. It exercises exactly the parts of the
infrastructure that a real run depends on — config loading, run-directory
creation, derived per-task seeds, one output file per array task — without
touching a dataset or a model. It is deliberately not research code.

Three subcommands, matching how a real array job is submitted:

``prepare``
    Run once, on the login node. Creates the run directory and writes the
    provenance record. Fails if the run already exists, so a resubmission
    cannot clobber earlier results.
``task``
    Run once per array task, on a compute node. Writes
    ``values/task-<id>.json`` and nothing else, so tasks never collide.
``verify``
    Run once after the array finishes. Checks every task produced output and
    that the derived seeds are distinct and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from food_shapley.config import ExperimentConfig, config_fingerprint, load_config
from food_shapley.paths import RunLayout
from food_shapley.provenance import write_run_metadata
from food_shapley.seeding import derive_seed, seed_everything

__all__ = ["main", "task_label"]

#: SLURM environment variables recorded in each task's output, for debugging
#: job-array problems after the fact.
_SLURM_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "CUDA_VISIBLE_DEVICES",
)


def task_label(task_id: int) -> str:
    """Return the stable label identifying one array task.

    Used both as the output filename stem and as the :func:`derive_seed` label,
    so a task's seed is a pure function of the config and its index.

    Args:
        task_id: Zero-based array task index.

    Returns:
        A zero-padded label such as ``"task-00042"``.

    Raises:
        ValueError: If ``task_id`` is negative.
    """
    if task_id < 0:
        msg = f"task_id must be non-negative, got {task_id}"
        raise ValueError(msg)
    return f"task-{task_id:05d}"


def _layout(config: ExperimentConfig) -> RunLayout:
    """Resolve the run layout; identical in every subcommand and every job."""
    return RunLayout.for_run(config.name, config_fingerprint(config))


def run_prepare(
    config: ExperimentConfig, *, allow_existing: bool = False
) -> tuple[RunLayout, bool]:
    """Create the run directory and write its provenance record.

    Args:
        config: The configuration for this run.
        allow_existing: Treat an already-prepared run as success instead of an
            error. Resubmitting a partially failed array is a normal operation,
            and it should not look like a crash in the job log. The existing
            provenance record is left untouched either way.

    Returns:
        The layout, and whether this call created it.

    Raises:
        FileExistsError: If the run already exists and ``allow_existing`` is False.
    """
    if allow_existing and _layout(config).metadata_file.exists():
        return _layout(config), False
    return write_run_metadata(config), True


def run_task(config: ExperimentConfig, task_id: int) -> Path:
    """Execute one array task and write its output file.

    Args:
        config: The configuration for this run.
        task_id: Zero-based array task index.

    Returns:
        Path to the file written.

    Raises:
        FileNotFoundError: If ``prepare`` has not been run for this config.
        FileExistsError: If this task's output already exists.
    """
    layout = _layout(config)
    if not layout.metadata_file.exists():
        msg = (
            f"no run metadata at {layout.metadata_file}; "
            "run the 'prepare' subcommand on the login node before submitting the array"
        )
        raise FileNotFoundError(msg)

    label = task_label(task_id)
    seed = seed_everything(derive_seed(config.seed, label))
    output = layout.values_dir / f"{label}.json"
    if output.exists():
        msg = f"task output already exists at {output}; refusing to overwrite"
        raise FileExistsError(msg)

    payload: dict[str, Any] = {
        "task_id": task_id,
        "label": label,
        "seed": seed,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "python_version": sys.version.split()[0],
        "slurm": {key: os.environ.get(key) for key in _SLURM_ENV_KEYS},
    }
    layout.values_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


def run_verify(config: ExperimentConfig, num_tasks: int) -> list[str]:
    """Check that an array run completed correctly.

    Verifies that every task wrote output, that each recorded seed matches what
    :func:`derive_seed` produces for that task, and that no two tasks shared a
    seed. Seed collisions across array tasks would silently correlate results
    in a real run, so this is the check worth having.

    Args:
        config: The configuration for this run.
        num_tasks: Number of array tasks that should have run.

    Returns:
        A list of human-readable problems; empty means the run is sound.

    Raises:
        ValueError: If ``num_tasks`` is not positive.
    """
    if num_tasks < 1:
        msg = f"num_tasks must be >= 1, got {num_tasks}"
        raise ValueError(msg)

    layout = _layout(config)
    problems: list[str] = []
    seen: dict[int, str] = {}

    for task_id in range(num_tasks):
        label = task_label(task_id)
        output = layout.values_dir / f"{label}.json"
        if not output.is_file():
            problems.append(f"{label}: missing output at {output}")
            continue

        payload: dict[str, Any] = json.loads(output.read_text())
        expected = derive_seed(config.seed, label)
        actual = payload.get("seed")
        if actual != expected:
            problems.append(f"{label}: seed {actual!r} does not match derived seed {expected}")
            continue
        if actual in seen:
            problems.append(f"{label}: seed {actual} collides with {seen[actual]}")
            continue
        seen[actual] = label

    return problems


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m food_shapley.selftest",
        description="Infrastructure self-test for the SLURM job-array mechanics.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment configuration TOML.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Create the run directory (login node, once).")
    prepare.add_argument(
        "--allow-existing",
        action="store_true",
        help="Succeed if the run directory already exists (for resubmissions).",
    )

    task = sub.add_parser("task", help="Run one array task (compute node).")
    task.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Array task index; defaults to $SLURM_ARRAY_TASK_ID.",
    )

    verify = sub.add_parser("verify", help="Check that every array task produced output.")
    verify.add_argument("--num-tasks", type=int, required=True, help="Expected number of tasks.")

    return parser


def _resolve_task_id(explicit: int | None) -> int:
    """Resolve the array task index from the flag or the SLURM environment.

    Raises:
        SystemExit: If neither source provides a usable index.
    """
    if explicit is not None:
        return explicit
    from_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if from_env is None:
        msg = "no --task-id given and SLURM_ARRAY_TASK_ID is not set"
        raise SystemExit(msg)
    try:
        return int(from_env)
    except ValueError:
        msg = f"SLURM_ARRAY_TASK_ID is not an integer: {from_env!r}"
        raise SystemExit(msg) from None


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero on any failure, so SLURM marks the job failed.
    """
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "prepare":
        layout, created = run_prepare(config, allow_existing=args.allow_existing)
        verb = "prepared" if created else "reusing existing"
        print(f"{verb} run directory: {layout.root}")
        return 0

    if args.command == "task":
        output = run_task(config, _resolve_task_id(args.task_id))
        print(f"wrote {output}")
        return 0

    problems = run_verify(config, args.num_tasks)
    if problems:
        print(f"self-test FAILED with {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"self-test passed: {args.num_tasks} task(s), all seeds derived and distinct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
