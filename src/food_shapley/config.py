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

__all__ = ["BACKGROUND_MODES", "ExperimentConfig", "config_fingerprint", "load_config"]

#: How the non-ingredient part of the frame -- plate, table, cutlery, a mug --
#: is treated. One field rather than two booleans, because the combinations are
#: mutually exclusive and a pair of flags would let a run request a
#: contradiction.
#:
#: ``"player"``
#:     The scene is a coalition player in its own right. ``v(N)`` is then the
#:     unmodified photograph and ``v(empty)`` a fully blurred frame, and the
#:     scene's contribution is *measured* instead of assumed. Costs one player,
#:     so twice the coalitions.
#: ``"visible"``
#:     The scene stays sharp in every coalition and is never attributed.
#:     Shapley values are contributions *given the scene*; ``v(N)`` is the
#:     photograph, but a telltale mug silently inflates ``v(empty)``.
#: ``"hidden"``
#:     The scene is blurred in every coalition. ``v(empty)`` is a fully blurred
#:     frame, but ``v(N)`` is **not** the photograph, so the decomposition
#:     explains a score the model never assigned to a real image.
BACKGROUND_MODES: frozenset[str] = frozenset({"player", "visible", "hidden"})


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
        model: Identifier of the scored model, as
            ``"<backend>/<architecture>/<pretrained>"`` -- e.g.
            ``"open_clip/ViT-B-32/openai"``. The pretrained tag is **not**
            optional: open_clip serves ``ViT-B-32`` weights trained on OpenAI's
            corpus, LAION-400M, LAION-2B and DataComp, and they are different
            models that give different numbers. See
            :class:`~food_shapley.scoring.CheckpointSpec`.
        removal: Segment-removal operator, e.g. ``"blur"`` or ``"token-drop"``.
        prompt_template: Format string with one ``{}`` placeholder, applied to
            every dish name to build the text CLIP scores against. Part of the
            claim: the prompt set is the model's output space, so rewording the
            template changes every value in the run.
        blur_radius: Gaussian blur radius in pixels for the ``"blur"`` operator.
            Part of the claim, not a cosmetic setting: a Shapley value is always
            "the contribution under *this* removal operator", and the radius is
            what makes a hidden ingredient more or less recoverable.
        background_mode: How the scene is treated; one of
            :data:`BACKGROUND_MODES`. Measured over the test split, the scene
            is 39 % of the model's input on average and the majority of pixels
            in 30 % of images, so this is a substantive choice about what the
            game is, not a rendering detail.
        image_size: Side length of the model's square input, in pixels. Images
            and masks are resized and centre-cropped to it together; see
            :mod:`food_shapley.segments`. It must match the checkpoint named in
            ``model``, so it is recorded rather than read off the checkpoint.
        max_players: Maximum number of coalition players per image after merging
            small segments and adding the "rest" player.
        min_area_fraction: Ingredients covering less than this share of the
            model's input are merged into the "rest" player. ``0.0`` keeps every
            segment, however small.
        max_images: How many images of the split the run covers, counted from
            the start, or ``0`` for all of them. A semantic field, not a
            throughput knob: a run over 20 images and a run over 2,135 answer
            different questions, so they must not share a run directory. It
            takes a **prefix**, not a random sample, which makes it a
            rehearsal device for checking the array mechanics cheaply and
            **not** a basis for any reported statistic.
        interaction_order: The ``k`` of the k-SII interaction index -- the
            largest group of players whose joint effect is reported. ``1``
            reduces k-SII to the Shapley value, ``2`` reports pairs. It is the
            headline choice of Milestone 2's compositionality index
            (``PLAN.md`` step 13), so it belongs in the fingerprint: two runs
            differing in it answer different questions. The Möbius transform is
            computed at full order regardless, so lowering this does not lose
            the raw interaction structure.
        notes: Free-text provenance note; excluded from the fingerprint.
    """

    name: str
    seed: int
    dataset: str
    split: str
    model: str
    removal: str
    prompt_template: str
    image_size: int
    max_players: int
    min_area_fraction: float
    blur_radius: float
    background_mode: str
    interaction_order: int
    max_images: int
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
        if self.image_size < 1:
            msg = f"config field 'image_size' must be >= 1, got {self.image_size}"
            raise ValueError(msg)
        if "{}" not in self.prompt_template:
            # Without the placeholder every dish gets the same prompt, the
            # softmax is uniform, and every Shapley value is zero -- a silent
            # way to produce a run that passes every check and means nothing.
            msg = (
                "config field 'prompt_template' must contain a '{}' placeholder, "
                f"got {self.prompt_template!r}"
            )
            raise ValueError(msg)
        if self.max_players < 1:
            msg = f"config field 'max_players' must be >= 1, got {self.max_players}"
            raise ValueError(msg)
        if not 0.0 <= self.min_area_fraction <= 1.0:
            got = self.min_area_fraction
            msg = f"config field 'min_area_fraction' must be in [0, 1], got {got}"
            raise ValueError(msg)
        if self.blur_radius <= 0:
            # Zero would make every coalition identical to the original image,
            # so every Shapley value would be zero and efficiency would still
            # hold -- a silent way to produce a meaningless run.
            msg = f"config field 'blur_radius' must be positive, got {self.blur_radius}"
            raise ValueError(msg)
        if self.max_images < 0:
            msg = f"config field 'max_images' must be >= 0 (0 means all), got {self.max_images}"
            raise ValueError(msg)
        if self.interaction_order < 1:
            msg = f"config field 'interaction_order' must be >= 1, got {self.interaction_order}"
            raise ValueError(msg)
        if self.interaction_order > self.max_players:
            # Unreachable by construction: no image can have more players than
            # the cap, so the order would be clamped on every single image and
            # the configured value would describe nothing that was computed.
            msg = (
                f"config field 'interaction_order' is {self.interaction_order} but "
                f"'max_players' is {self.max_players}, so no image can ever reach it"
            )
            raise ValueError(msg)
        if self.background_mode not in BACKGROUND_MODES:
            allowed = ", ".join(sorted(BACKGROUND_MODES))
            mode = self.background_mode
            msg = f"config field 'background_mode' must be one of {allowed}, got {mode!r}"
            raise ValueError(msg)
        if self.background_is_player and self.max_players < 2:
            # One slot is the scene, so a cap of 1 would leave no ingredient.
            msg = (
                "config field 'max_players' must be >= 2 when background_mode is "
                f"'player', because the scene occupies one slot; got {self.max_players}"
            )
            raise ValueError(msg)

    @property
    def background_is_player(self) -> bool:
        """Whether the scene gets a coalition slot of its own."""
        return self.background_mode == "player"

    @property
    def background_stays_sharp(self) -> bool:
        """Whether non-player pixels are left unblurred by the removal operator.

        False in ``"player"`` mode: there the scene *is* a player, so it is
        blurred or not according to the coalition like everything else.
        """
        return self.background_mode == "visible"

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
