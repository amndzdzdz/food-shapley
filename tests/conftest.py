"""Shared fixtures."""

from __future__ import annotations

import pytest

from food_shapley.config import ExperimentConfig


@pytest.fixture
def config() -> ExperimentConfig:
    """A minimal, valid experiment configuration."""
    return ExperimentConfig(
        name="smoke",
        seed=0,
        dataset="foodseg103",
        split="test",
        model="openclip/ViT-B-32",
        removal="blur",
        max_players=7,
    )
