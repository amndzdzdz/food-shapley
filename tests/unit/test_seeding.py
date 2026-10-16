from __future__ import annotations

import random

import pytest

from food_shapley.seeding import derive_seed, seed_everything


def test_derive_seed_is_deterministic() -> None:
    assert derive_seed(0, "img_00042") == derive_seed(0, "img_00042")


def test_derive_seed_separates_labels() -> None:
    assert derive_seed(0, "img_00042") != derive_seed(0, "img_00043")


def test_derive_seed_separates_base_seeds() -> None:
    assert derive_seed(0, "img_00042") != derive_seed(1, "img_00042")


def test_derive_seed_is_in_range() -> None:
    seeds = [derive_seed(7, f"img_{i:05d}") for i in range(256)]
    assert all(0 <= seed < 2**32 for seed in seeds)


def test_derive_seed_collisions_are_rare() -> None:
    seeds = {derive_seed(7, f"img_{i:05d}") for i in range(1000)}
    assert len(seeds) == 1000


@pytest.mark.parametrize(("base", "label"), [(-1, "a"), (0, "")])
def test_derive_seed_rejects_invalid_input(base: int, label: str) -> None:
    with pytest.raises(ValueError, match="must"):
        derive_seed(base, label)


def test_seed_everything_makes_stdlib_random_reproducible() -> None:
    seed_everything(123)
    first = [random.random() for _ in range(5)]  # noqa: S311 - not cryptographic
    seed_everything(123)
    second = [random.random() for _ in range(5)]  # noqa: S311 - not cryptographic
    assert first == second


def test_seed_everything_returns_seed() -> None:
    assert seed_everything(5) == 5


def test_seed_everything_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="seed must be in"):
        seed_everything(2**32)
