"""Artifact layout.

Source code, raw data and generated results are kept strictly apart:

``data/raw/``
    Read-only inputs (FoodSeg103, Recipe1M, ...). Never written by the code.
``data/interim/``
    Derived but reusable inputs, e.g. merged segment masks.
``runs/<name>-<fingerprint>/``
    Everything a single run produces, including the resolved configuration and
    the provenance record. One directory per (config, code) pair.

Nothing under ``data/`` or ``runs/`` is committed; see ``.gitignore``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["RunLayout", "artifact_root", "data_root"]

#: Environment variable overriding where runs are written (e.g. cluster scratch).
ARTIFACT_ROOT_ENV = "FOOD_SHAPLEY_RUNS"
#: Environment variable overriding where input data is read from.
DATA_ROOT_ENV = "FOOD_SHAPLEY_DATA"


def artifact_root() -> Path:
    """Return the directory under which run directories are created.

    Defaults to ``runs/`` in the current working directory and can be pointed
    at cluster scratch space via the ``FOOD_SHAPLEY_RUNS`` environment variable.
    """
    return Path(os.environ.get(ARTIFACT_ROOT_ENV, "runs"))


def data_root() -> Path:
    """Return the directory under which input datasets are looked up.

    Defaults to ``data/`` and can be overridden via ``FOOD_SHAPLEY_DATA``.
    """
    return Path(os.environ.get(DATA_ROOT_ENV, "data"))


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Paths belonging to one experiment run.

    Attributes:
        root: The run directory itself.
    """

    root: Path

    @classmethod
    def for_run(cls, name: str, fingerprint: str, *, base: Path | None = None) -> RunLayout:
        """Build the layout for a run identified by name and config fingerprint.

        Args:
            name: The experiment name, from the configuration.
            fingerprint: The configuration fingerprint, see
                :func:`food_shapley.config.config_fingerprint`.
            base: Override for the artifact root; defaults to :func:`artifact_root`.

        Returns:
            The layout; no directory is created yet.
        """
        parent = artifact_root() if base is None else base
        return cls(root=parent / f"{name}-{fingerprint}")

    @property
    def config_file(self) -> Path:
        """Resolved configuration as it was used for this run."""
        return self.root / "config.json"

    @property
    def metadata_file(self) -> Path:
        """Provenance record: code version, environment, seed, timestamp."""
        return self.root / "run_metadata.json"

    @property
    def values_dir(self) -> Path:
        """Per-image coalition values and Shapley/interaction indices (``.npz``)."""
        return self.root / "values"

    @property
    def figures_dir(self) -> Path:
        """Figures generated from this run."""
        return self.root / "figures"

    @property
    def logs_dir(self) -> Path:
        """Job logs, including SLURM array task output."""
        return self.root / "logs"

    def create(self, *, exist_ok: bool = False) -> None:
        """Create the run directory tree.

        Args:
            exist_ok: Allow the run directory to already exist. Defaults to
                ``False`` so that an existing run is never silently extended or
                overwritten.

        Raises:
            FileExistsError: If the run directory exists and ``exist_ok`` is False.
        """
        self.root.mkdir(parents=True, exist_ok=exist_ok)
        for directory in (self.values_dir, self.figures_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
