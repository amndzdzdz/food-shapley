"""Compute-node environment check.

`PLAN.md` Milestone 1, step 0 requires the research stack to be installed *on
the SLURM nodes*, not merely on the login node. The two differ in the ways that
matter: GPU visibility, driver/CUDA pairing, and whether the shared filesystem
holding ``.venv`` is readable from the node at all.

This module answers one question -- "would a real run work here?" -- cheaply
enough to put at the top of every job script, and writes the answer into the
job log where it can be read after the fact.

What it deliberately does not do
--------------------------------
It imports libraries and reads versions; it never downloads a checkpoint and
never loads a model. A failing environment check must cost seconds, and a job
script must not be able to trigger a multi-gigabyte download from a compute
node with no network. ``open_clip`` is therefore only imported, not asked for
weights.

GPU absence is reported but is not an error by default: the check is also
useful on a login node. Pass ``--require-gpu`` in a job script, where a missing
GPU is a real failure.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from food_shapley.download import split_dir, verify_split

__all__ = ["CheckResult", "check_environment", "main"]

#: Import name -> why a run needs it. Every one of these is a hard requirement;
#: a missing import means the node cannot run the pipeline at all.
REQUIRED_IMPORTS: dict[str, str] = {
    "shapiq": "exact Shapley values, k-SII and Mobius transforms",
    "torch": "model inference backend",
    "open_clip": "CLIP implementation",
    "numpy": "coalition encoding and stored value arrays",
    "PIL": "image and mask handling",
    "datasets": "FoodSeg103 access",
}


@dataclass(slots=True)
class CheckResult:
    """Outcome of an environment check.

    Attributes:
        facts: Human-readable observations, in report order. Recorded whether
            or not the check passed -- a failing log is only useful if it says
            what the environment actually was.
        problems: Reasons the environment is unusable. Empty means usable.
    """

    facts: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the environment is usable."""
        return not self.problems


def _check_imports(result: CheckResult) -> None:
    """Import every required library and record its version."""
    for name, reason in REQUIRED_IMPORTS.items():
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            result.problems.append(f"cannot import {name} ({reason}): {error}")
            continue
        version = getattr(module, "__version__", "<no __version__>")
        result.facts.append(f"{name:<10} {version}")


def _check_gpu(result: CheckResult, *, require: bool) -> None:
    """Record GPU visibility, and fail if a GPU was required but is absent.

    Reports driver-level details because the usual cluster failure is not "no
    GPU" but "torch was built against a CUDA the node's driver does not
    provide", which looks identical until the versions are printed.
    """
    try:
        import torch
    except ImportError:
        # Already recorded as a problem by _check_imports.
        return

    available = torch.cuda.is_available()
    result.facts.append(f"torch built for CUDA {torch.version.cuda or '<cpu build>'}")
    result.facts.append(f"cuda available: {available}")

    if not available:
        message = "no CUDA device visible to torch"
        if require:
            result.problems.append(message)
        else:
            result.facts.append(f"note: {message} (not required here)")
        return

    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        total_gib = torch.cuda.get_device_properties(index).total_memory / 1024**3
        result.facts.append(f"gpu {index}: {name} ({total_gib:.1f} GiB)")


def _check_dataset(result: CheckResult, split: str, *, require: bool) -> None:
    """Check that a FoodSeg103 split is present and intact."""
    directory = split_dir(split)
    problems = verify_split(split)
    if problems:
        target = result.problems if require else result.facts
        prefix = "" if require else "note: "
        target.extend(f"{prefix}{problem}" for problem in problems)
        return
    result.facts.append(f"foodseg103 '{split}' verified at {directory}")


def check_environment(
    *,
    require_gpu: bool = False,
    dataset_split: str | None = None,
    require_dataset: bool = False,
) -> CheckResult:
    """Inspect the current environment.

    Args:
        require_gpu: Treat a missing CUDA device as a failure.
        dataset_split: Split to check for, or ``None`` to skip the check.
        require_dataset: Treat a missing or corrupt split as a failure.

    Returns:
        The facts observed and the problems found.
    """
    result = CheckResult()
    result.facts.append(f"host       {platform.node()}")
    result.facts.append(f"python     {sys.version.split()[0]} ({Path(sys.executable)})")
    result.facts.append(f"platform   {platform.platform()}")

    _check_imports(result)
    _check_gpu(result, require=require_gpu)
    if dataset_split is not None:
        _check_dataset(result, dataset_split, require=require_dataset)
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m food_shapley.envcheck",
        description="Report whether this machine can run the pipeline. Downloads nothing.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if no CUDA device is visible (use this in job scripts).",
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="Also verify this FoodSeg103 split, e.g. 'test'.",
    )
    parser.add_argument(
        "--require-dataset",
        action="store_true",
        help="Fail if --dataset-split is missing or corrupt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero if the environment is unusable.
    """
    args = _build_parser().parse_args(argv)
    result = check_environment(
        require_gpu=args.require_gpu,
        dataset_split=args.dataset_split,
        require_dataset=args.require_dataset,
    )

    for fact in result.facts:
        print(fact)
    if result.problems:
        print("environment check FAILED:", file=sys.stderr)
        for problem in result.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
