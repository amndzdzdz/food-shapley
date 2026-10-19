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
        image_size=224,
        max_players=7,
        min_area_fraction=0.005,
        blur_radius=12.0,
        background_mode="player",
    )
