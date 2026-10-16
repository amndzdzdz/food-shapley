"""Deterministic seeding.

The project computes per-image games over thousands of SLURM array tasks. A
single global seed is therefore not enough: each unit of work needs its own
seed that is reproducible regardless of execution order. :func:`derive_seed`
provides that, :func:`seed_everything` pins the process-wide generators.
"""

from __future__ import annotations

import hashlib
import os
import random

__all__ = ["derive_seed", "seed_everything"]

#: Upper bound for derived seeds; safe for Python, NumPy and PyTorch.
_SEED_MODULUS = 2**32


def derive_seed(base_seed: int, label: str) -> int:
    """Derive a stable sub-seed for a named unit of work.

    The result depends only on ``base_seed`` and ``label``, never on call
    order, so array task 417 gets the same seed whether it runs first or last.

    Args:
        base_seed: The experiment-level seed, typically ``ExperimentConfig.seed``.
        label: Identifier of the unit of work, e.g. an image id or split name.

    Returns:
        A seed in ``[0, 2**32)``.

    Raises:
        ValueError: If ``base_seed`` is negative or ``label`` is empty.
    """
    if base_seed < 0:
        msg = f"base_seed must be non-negative, got {base_seed}"
        raise ValueError(msg)
    if not label:
        msg = "label must not be empty"
        raise ValueError(msg)

    digest = hashlib.sha256(f"{base_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def seed_everything(seed: int) -> int:
    """Seed the process-wide random generators.

    Currently seeds the standard library generator and pins ``PYTHONHASHSEED``
    for child processes. NumPy and PyTorch seeding is added here once those
    dependencies enter the project (Week 1); keeping the single entry point
    means call sites do not change.

    Args:
        seed: The seed to apply.

    Returns:
        The applied seed, for convenient logging into run metadata.

    Raises:
        ValueError: If ``seed`` is outside ``[0, 2**32)``.
    """
    if not 0 <= seed < _SEED_MODULUS:
        msg = f"seed must be in [0, {_SEED_MODULUS}), got {seed}"
        raise ValueError(msg)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    return seed
