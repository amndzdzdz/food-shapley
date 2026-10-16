"""Run provenance.

A number in the paper must be traceable to the configuration, the code version
and the environment that produced it. Every run writes a provenance record
next to its results before any expensive computation starts.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from food_shapley.config import ExperimentConfig, config_fingerprint
from food_shapley.paths import RunLayout

__all__ = ["RunProvenance", "capture_provenance", "write_run_metadata"]

#: Packages whose versions are recorded. Extended as research deps are added.
TRACKED_PACKAGES: tuple[str, ...] = ("food-shapley",)


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Everything needed to reconstruct the context of a run.

    Attributes:
        timestamp_utc: ISO-8601 timestamp of when the run started.
        git_commit: Current commit hash, or ``None`` outside a git checkout.
        git_dirty: Whether the working tree had uncommitted changes. A dirty
            tree means the result is not reproducible from the commit alone.
        python_version: Interpreter version string.
        platform: Host platform description.
        packages: Versions of :data:`TRACKED_PACKAGES`, ``None`` if not installed.
        config_fingerprint: Fingerprint of the configuration used.
        config: The configuration itself.
    """

    timestamp_utc: str
    git_commit: str | None
    git_dirty: bool
    python_version: str
    platform: str
    packages: dict[str, str | None]
    config_fingerprint: str
    config: dict[str, Any]


def _git(args: Iterable[str], *, repo: Path) -> str | None:
    """Run a read-only git command, returning ``None`` if git is unavailable."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],  # noqa: S607 - resolved via PATH on purpose
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Look up installed versions, mapping missing packages to ``None``."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def capture_provenance(config: ExperimentConfig, *, repo: Path | None = None) -> RunProvenance:
    """Capture the provenance of a run about to start.

    Args:
        config: The configuration the run will use.
        repo: Repository to read git information from; defaults to the
            current working directory.

    Returns:
        The provenance record.
    """
    repo_path = Path.cwd() if repo is None else repo
    commit = _git(["rev-parse", "HEAD"], repo=repo_path)
    status = _git(["status", "--porcelain"], repo=repo_path)

    return RunProvenance(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=bool(status),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        packages=_package_versions(TRACKED_PACKAGES),
        config_fingerprint=config_fingerprint(config),
        config=config.to_dict(),
    )


def write_run_metadata(
    config: ExperimentConfig,
    *,
    layout: RunLayout | None = None,
    repo: Path | None = None,
) -> RunLayout:
    """Create the run directory and write its configuration and provenance.

    Refuses to touch a run directory that already contains metadata, so that a
    re-submitted SLURM job cannot silently overwrite earlier results.

    Args:
        config: The configuration the run will use.
        layout: Target layout; derived from the configuration by default.
        repo: Repository to read git information from.

    Returns:
        The layout that was written to.

    Raises:
        FileExistsError: If the run directory already holds a metadata file.
    """
    run_layout = (
        RunLayout.for_run(config.name, config_fingerprint(config)) if layout is None else layout
    )
    if run_layout.metadata_file.exists():
        msg = (
            f"run metadata already exists at {run_layout.metadata_file}; "
            "refusing to overwrite an existing run"
        )
        raise FileExistsError(msg)

    run_layout.create(exist_ok=True)
    provenance = capture_provenance(config, repo=repo)
    run_layout.config_file.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    run_layout.metadata_file.write_text(json.dumps(asdict(provenance), indent=2) + "\n")
    return run_layout
