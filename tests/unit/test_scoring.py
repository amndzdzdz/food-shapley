from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from food_shapley.scoring import (
    DEFAULT_BATCH_SIZE,
    CheckpointSpec,
    ClipScorer,
    PromptEmbedding,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# --------------------------------------------------------------------------- #
# A stand-in for an open_clip model.
#
# `CLAUDE.md` forbids downloading a checkpoint inside the test suite, so the
# real ViT-B-32 is never loaded here; `ClipScorer.load` is covered by the
# gpu-marked integration tests instead. What is tested here is this module's
# own arithmetic -- normalisation, the temperature, batching, the shape
# contract -- against a model whose output is a closed-form function of its
# input, so the expected logits can be written down rather than recorded.
#
# The stub embeds an image as the mean of each colour channel and a prompt as
# the one-hot of its length modulo the embedding dimension. Both are
# deterministic and neither needs weights.
# --------------------------------------------------------------------------- #

EMBED_DIM = 3
IMAGE_SIZE = 8
#: exp(LOGIT_SCALE) == 100, the value OpenAI's CLIP checkpoints converge to.
LOGIT_SCALE = float(np.log(100.0))


class _StubVisual:
    """Carries the `image_size` attribute `ClipScorer.load` checks."""

    def __init__(self, size: int) -> None:
        self.image_size = (size, size)


class _StubModel:
    """A CLIP-shaped model with no weights and no randomness."""

    def __init__(self, *, size: int = IMAGE_SIZE, scale: float = LOGIT_SCALE) -> None:
        self.visual = _StubVisual(size)
        self.logit_scale = torch.tensor(scale)
        self.text_calls = 0
        self.image_calls = 0

    def encode_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mean of each channel: (batch, 3, h, w) -> (batch, 3)."""
        self.image_calls += 1
        return tensor.mean(dim=(2, 3))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """One-hot of each prompt's token count, modulo the embedding size."""
        self.text_calls += 1
        lengths = tokens.sum(dim=1) % EMBED_DIM
        return torch.nn.functional.one_hot(lengths, num_classes=EMBED_DIM).float()

    def eval(self) -> _StubModel:
        return self

    def parameters(self) -> list[torch.Tensor]:
        return []

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"logit_scale": self.logit_scale}


def _stub_preprocess(image: object) -> torch.Tensor:
    """Stand in for open_clip's transform: to CHW float in [0, 1], no resize."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _stub_tokenizer(prompts: Sequence[str]) -> torch.Tensor:
    """One row per prompt, summing to that prompt's length."""
    width = max(len(prompt) for prompt in prompts)
    tokens = torch.zeros((len(prompts), width), dtype=torch.int64)
    for row, prompt in enumerate(prompts):
        tokens[row, : len(prompt)] = 1
    return tokens


@pytest.fixture
def stub_model() -> _StubModel:
    return _StubModel()


@pytest.fixture
def scorer(stub_model: _StubModel) -> ClipScorer:
    """A scorer over the stub, in the geometry the stub declares."""
    return ClipScorer(
        model=stub_model,
        preprocess=_stub_preprocess,
        tokenizer=_stub_tokenizer,
        spec=CheckpointSpec.parse("open_clip/Stub-B-32/stub"),
        device="cpu",
        image_size=IMAGE_SIZE,
    )


def flat(*channels: int) -> NDArray[np.uint8]:
    """A uniform image of the given RGB value, in the stub's geometry."""
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[:, :] = channels
    return image


# --------------------------------------------------------------------------- #
# CheckpointSpec
# --------------------------------------------------------------------------- #


def test_spec_parses_three_parts() -> None:
    spec = CheckpointSpec.parse("open_clip/ViT-B-32/openai")
    assert (spec.backend, spec.architecture, spec.pretrained) == (
        "open_clip",
        "ViT-B-32",
        "openai",
    )


def test_spec_roundtrips_through_str() -> None:
    identifier = "open_clip/ViT-B-32/laion2b_s34b_b79k"
    assert str(CheckpointSpec.parse(identifier)) == identifier


@pytest.mark.parametrize(
    "identifier",
    [
        # The pretrained tag is what the old config was missing: ViT-B-32 alone
        # names an architecture served with at least six different sets of
        # weights, which score the same photograph differently.
        "open_clip/ViT-B-32",
        "ViT-B-32",
        "open_clip/ViT-B-32/openai/extra",
        "open_clip//openai",
        "open_clip/ViT-B-32/  ",
        "",
    ],
)
def test_spec_rejects_an_underspecified_identifier(identifier: str) -> None:
    with pytest.raises(ValueError, match="<backend>/<architecture>/<pretrained>"):
        CheckpointSpec.parse(identifier)


def test_spec_rejects_an_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported model backend 'transformers'"):
        CheckpointSpec.parse("transformers/ViT-B-32/openai")


# --------------------------------------------------------------------------- #
# Prompt embedding
# --------------------------------------------------------------------------- #


def test_prompt_embedding_is_l2_normalised(scorer: ClipScorer) -> None:
    """Cosine, not dot product: unnormalised rows let a prompt win on length."""
    embedding = scorer.embed_prompts(["a", "bb", "ccc", "dddd"])
    norms = embedding.matrix.norm(dim=-1).numpy()
    np.testing.assert_allclose(norms, np.ones(4), atol=1e-6)


def test_prompt_embedding_keeps_its_prompts(scorer: ClipScorer) -> None:
    prompts = ["a photo of pizza.", "a photo of churros."]
    embedding = scorer.embed_prompts(prompts)
    assert embedding.prompts == tuple(prompts)
    assert len(embedding) == 2


def test_prompt_embedding_rejects_an_empty_set(scorer: ClipScorer) -> None:
    with pytest.raises(ValueError, match="empty prompt set"):
        scorer.embed_prompts([])


def test_prompts_are_embedded_once_per_call(scorer: ClipScorer, stub_model: _StubModel) -> None:
    """The reuse this module exists for: 2,000 coalitions, one text pass."""
    embedding = scorer.embed_prompts(["a", "bb"])
    scorer.logits([flat(10, 20, 30)] * 5, embedding)
    assert stub_model.text_calls == 1


# --------------------------------------------------------------------------- #
# Logits
# --------------------------------------------------------------------------- #


def test_logits_have_one_row_per_image_and_one_column_per_prompt(
    scorer: ClipScorer,
) -> None:
    embedding = scorer.embed_prompts(["a", "bb", "ccc"])
    scores = scorer.logits([flat(1, 2, 3), flat(4, 5, 6)], embedding)
    assert scores.shape == (2, 3)
    assert scores.dtype == np.float32


def test_logits_match_the_closed_form(scorer: ClipScorer) -> None:
    """Scaled cosine similarity, computed by hand against the stub."""
    embedding = scorer.embed_prompts(["a", "bb", "ccc"])
    image = flat(255, 0, 0)
    scores = scorer.logits([image], embedding)

    # The stub embeds this image as (1, 0, 0); normalised, that is (1, 0, 0).
    # Prompt lengths 1, 2, 3 one-hot to e1, e2, e0, so the cosines are 0, 0, 1.
    expected = 100.0 * np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(scores, expected, atol=1e-4)


def test_logits_apply_the_checkpoints_temperature() -> None:
    """`logit_scale` is trained, not ours; doubling it must double the logits."""
    prompts = ["a", "bb", "ccc"]
    image = flat(255, 0, 0)

    def score(scale: float) -> NDArray[np.float32]:
        model = _StubModel(scale=float(np.log(scale)))
        hot = ClipScorer(
            model=model,
            preprocess=_stub_preprocess,
            tokenizer=_stub_tokenizer,
            spec=CheckpointSpec.parse("open_clip/Stub-B-32/stub"),
            device="cpu",
            image_size=IMAGE_SIZE,
        )
        return hot.logits([image], hot.embed_prompts(prompts))

    np.testing.assert_allclose(score(200.0), 2.0 * score(100.0), atol=1e-3)


def test_logits_are_invariant_to_batch_size(scorer: ClipScorer) -> None:
    """Batching is a throughput knob; it must not change the science."""
    embedding = scorer.embed_prompts(["a", "bb", "ccc"])
    images = [flat(value, 255 - value, value // 2) for value in range(0, 256, 17)]
    one = scorer.logits(images, embedding, batch_size=1)
    many = scorer.logits(images, embedding, batch_size=len(images))
    np.testing.assert_allclose(one, many, atol=1e-5)


def test_logits_preserve_input_order(scorer: ClipScorer) -> None:
    """Task 1.4 maps a value back to its coalition by position."""
    embedding = scorer.embed_prompts(["a", "bb", "ccc"])
    images = [flat(255, 0, 0), flat(0, 255, 0), flat(0, 0, 255)]
    together = scorer.logits(images, embedding, batch_size=2)
    for position, image in enumerate(images):
        np.testing.assert_allclose(
            together[position], scorer.logits([image], embedding)[0], atol=1e-5
        )


def test_logits_split_into_the_expected_number_of_batches(
    scorer: ClipScorer, stub_model: _StubModel
) -> None:
    embedding = scorer.embed_prompts(["a", "bb"])
    scorer.logits([flat(1, 2, 3)] * 7, embedding, batch_size=3)
    assert stub_model.image_calls == 3


def test_logits_of_no_images_is_an_empty_matrix(scorer: ClipScorer) -> None:
    """An empty coalition list is not an error; it is zero work."""
    embedding = scorer.embed_prompts(["a", "bb", "ccc"])
    scores = scorer.logits([], embedding)
    assert scores.shape == (0, 3)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_logits_reject_a_nonpositive_batch_size(scorer: ClipScorer, batch_size: int) -> None:
    embedding = scorer.embed_prompts(["a"])
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        scorer.logits([flat(1, 2, 3)], embedding, batch_size=batch_size)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8),
        np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8),
        np.zeros((IMAGE_SIZE + 1, IMAGE_SIZE, 3), dtype=np.uint8),
        np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32),
    ],
)
def test_logits_reject_images_outside_the_geometry_contract(
    scorer: ClipScorer, image: NDArray[np.generic]
) -> None:
    """A wrongly shaped image is a masking bug upstream, not a resize job."""
    embedding = scorer.embed_prompts(["a"])
    with pytest.raises(ValueError, match="in the checkpoint's geometry"):
        scorer.logits([image], embedding)  # type: ignore[list-item]


def test_logits_name_the_offending_image(scorer: ClipScorer) -> None:
    embedding = scorer.embed_prompts(["a"])
    bad = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="image 2"):
        scorer.logits([flat(1, 1, 1), flat(2, 2, 2), bad], embedding)


# --------------------------------------------------------------------------- #
# The architecture/weights mismatch guard
#
# `load` itself is not exercised here -- it downloads a checkpoint, which the
# suite must not do -- but the predicate it delegates to is, because that
# predicate is what stands between the project and silently scoring with the
# wrong activation function. It caught exactly that; see W01-10.
# --------------------------------------------------------------------------- #


def _warning(text: str) -> warnings.WarningMessage:
    return warnings.WarningMessage(
        message=UserWarning(text), category=UserWarning, filename="x", lineno=1
    )


def test_mismatch_warning_becomes_an_error() -> None:
    spec = CheckpointSpec.parse("open_clip/ViT-B-32/openai")
    caught = [
        _warning(
            "QuickGELU mismatch between final model config (quick_gelu=False) "
            "and pretrained tag 'openai' (quick_gelu=True)."
        )
    ]
    with pytest.raises(RuntimeError, match="quickgelu"):
        ClipScorer._check_warnings(spec, caught)


def test_mismatch_error_names_the_checkpoint() -> None:
    spec = CheckpointSpec.parse("open_clip/ViT-B-32/openai")
    with pytest.raises(RuntimeError, match="open_clip/ViT-B-32/openai"):
        ClipScorer._check_warnings(spec, [_warning("some mismatch happened")])


def test_unrelated_warnings_are_tolerated() -> None:
    """Rate-limit and deprecation notices must not block a run."""
    spec = CheckpointSpec.parse("open_clip/ViT-B-32-quickgelu/openai")
    ClipScorer._check_warnings(
        spec,
        [
            _warning("You are sending unauthenticated requests to the HF Hub."),
            _warning("torch.load is deprecated"),
        ],
    )


def test_no_warnings_is_fine() -> None:
    spec = CheckpointSpec.parse("open_clip/ViT-B-32-quickgelu/openai")
    ClipScorer._check_warnings(spec, [])


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_provenance_records_the_full_checkpoint_identity(scorer: ClipScorer) -> None:
    provenance = scorer.provenance()
    assert provenance["architecture"] == "Stub-B-32"
    assert provenance["pretrained"] == "stub"
    assert provenance["device"] == "cpu"
    assert provenance["image_size"] == str(IMAGE_SIZE)
    assert len(provenance["parameter_sha256"]) == 64


def test_provenance_hash_tracks_the_weights(scorer: ClipScorer) -> None:
    """The point of hashing: two runs claiming one model that did not use it."""
    other = ClipScorer(
        model=_StubModel(scale=LOGIT_SCALE + 1.0),
        preprocess=_stub_preprocess,
        tokenizer=_stub_tokenizer,
        spec=CheckpointSpec.parse("open_clip/Stub-B-32/stub"),
        device="cpu",
        image_size=IMAGE_SIZE,
    )
    assert other.provenance()["parameter_sha256"] != scorer.provenance()["parameter_sha256"]


def test_scorer_exposes_its_configuration(scorer: ClipScorer) -> None:
    assert scorer.image_size == IMAGE_SIZE
    assert scorer.device == "cpu"
    assert scorer.spec.pretrained == "stub"


def test_default_batch_size_is_positive() -> None:
    assert DEFAULT_BATCH_SIZE >= 1


def test_prompt_embedding_is_constructible_directly() -> None:
    """Used by the value function's tests to build a known output space."""
    matrix = torch.eye(EMBED_DIM)
    embedding = PromptEmbedding(prompts=("a", "b", "c"), matrix=matrix)
    assert len(embedding) == EMBED_DIM
