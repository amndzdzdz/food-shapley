"""Experiment configuration.

Configuration is kept out of the implementation: every run is described by a
TOML file under ``configs/`` and is reduced to a stable fingerprint so that an
artifact on disk can be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any

__all__ = ["ExperimentConfig", "config_fingerprint", "load_config"]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Settings that fully determine a single experiment run.

    Every field is explicit on purpose: a run must never depend on a hidden
    default that is invisible in the stored configuration. Fields are added as
    the pipeline grows, and adding one changes the fingerprint of the run.

    Attributes:
        name: Human-readable identifier, also used as the artifact directory name.
        seed: Base random seed; per-unit seeds are derived from it.
        dataset: Dataset identifier, e.g. ``"foodseg103"``.
        split: Dataset split, e.g. ``"test"``.
        model: Model identifier of the scored model, e.g. an open_clip checkpoint.
        removal: Segment-removal operator, e.g. ``"blur"`` or ``"token-drop"``.
        max_players: Maximum number of coalition players per image after merging
            small segments and adding the "rest" player.
        notes: Free-text provenance note; excluded from the fingerprint.
    """

    name: str
    seed: int
    dataset: str
    split: str
    model: str
    removal: str
    max_players: int
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If a field is empty or outside its allowed range.
        """
        if not self.name.strip():
            msg = "config field 'name' must not be empty"
            raise ValueError(msg)
        if self.seed < 0:
            msg = f"config field 'seed' must be non-negative, got {self.seed}"
            raise ValueError(msg)
        if self.max_players < 1:
            msg = f"config field 'max_players' must be >= 1, got {self.max_players}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain, JSON-serialisable dictionary."""
        return asdict(self)


#: Fields that describe bookkeeping rather than experiment semantics.
_NON_SEMANTIC_FIELDS = frozenset({"notes"})


def load_config(path: Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a TOML file.

    Unknown keys are rejected rather than ignored, so that a typo in a
    configuration file fails loudly instead of silently falling back to a
    default.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If the file contains unknown keys or misses required ones.
    """
    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    known = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        msg = f"{path}: unknown configuration keys: {', '.join(unknown)}"
        raise ValueError(msg)

    required = {field.name for field in fields(ExperimentConfig) if field.default is MISSING}
    missing = sorted(required - set(raw))
    if missing:
        msg = f"{path}: missing configuration keys: {', '.join(missing)}"
        raise ValueError(msg)

    return ExperimentConfig(**raw)


def config_fingerprint(config: ExperimentConfig, *, length: int = 12) -> str:
    """Return a short, stable hash of the semantic content of ``config``.

    The fingerprint ignores purely descriptive fields (currently ``notes``) so
    that re-running the same experiment with a different comment does not look
    like a different experiment.

    Args:
        config: The configuration to fingerprint.
        length: Number of hex characters to keep.

    Returns:
        A lowercase hex digest prefix of the given length.
    """
    semantic = {k: v for k, v in config.to_dict().items() if k not in _NON_SEMANTIC_FIELDS}
    payload = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:length]
