"""CLIP as a scoring function over dish prompts.

`PLAN.md` Milestone 1, step 2, second half: "set up CLIP". This module turns
rendered coalition images into a matrix of logits over the dish prompt set and
nothing more -- which logit becomes the value of a coalition is
:mod:`food_shapley.value`, and how coalitions are enumerated is task 1.4.

Preprocessing is *not* repeated here
------------------------------------
:func:`food_shapley.segments.resize_image` already reproduces open_clip's
shortest-side resize and centre crop, byte-for-byte against torchvision, and
applies it to the image and the mask **together**. That shared geometry is
what makes "only the bun is visible" true. Running open_clip's transform again
on the rendered 224 px coalition would be a second resize of an already-resized
image.

It is nonetheless open_clip's own transform that runs here, because on an input
already at the checkpoint's size the resize and crop are no-ops (asserted in
:mod:`tests.unit.test_segments`) and only the normalisation remains -- and the
normalisation constants belong to the checkpoint, not to this file. Hardcoding
them would let a future checkpoint with different statistics drift silently.
:meth:`ClipScorer.load` refuses to run when the checkpoint's expected input
size differs from the configured ``image_size``, which is the case where the
no-op assumption would break.

Text embeddings are computed once
---------------------------------
The 101 prompts are identical for every coalition of every image, so they are
embedded once into a :class:`PromptEmbedding` and reused. At ~2,000 coalitions
per image over 2,135 images, re-encoding them per call would dominate the
entire run. The embedding is an explicit object rather than a hidden cache so
that a value function cannot silently be scoring a different prompt set than
the one recorded in its run metadata.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from food_shapley.config import load_config

if TYPE_CHECKING:
    import warnings
    from collections.abc import Iterable, Sequence

    import torch
    from numpy.typing import NDArray

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "CheckpointSpec",
    "ClipScorer",
    "PromptEmbedding",
    "main",
]

#: Images per forward pass. A throughput knob, not an experiment setting, so it
#: is a plain argument rather than a configuration field. It can perturb logits
#: in the last bits or two through floating-point non-associativity; nothing in
#: this project depends on differences at that scale.
DEFAULT_BATCH_SIZE: Final = 64

#: Backend identifier this module implements.
_OPEN_CLIP: Final = "open_clip"


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """A fully qualified model checkpoint.

    Attributes:
        backend: Library that serves the weights; currently ``"open_clip"``.
        architecture: Architecture name, e.g. ``"ViT-B-32"``.
        pretrained: open_clip pretrained tag, e.g. ``"openai"`` or
            ``"laion2b_s34b_b79k"``.

    The pretrained tag is mandatory. ``ViT-B-32`` alone names an architecture,
    not a model: open_clip serves at least ``openai``, ``laion400m_e31``,
    ``laion400m_e32``, ``laion2b_e16``, ``laion2b_s34b_b79k`` and
    ``datacomp_xl_s13b_b90k`` weights for it. They score the same photograph
    differently, so a configuration that omits the tag does not identify the
    experiment.
    """

    backend: str
    architecture: str
    pretrained: str

    @classmethod
    def parse(cls, identifier: str) -> CheckpointSpec:
        """Parse a ``"<backend>/<architecture>/<pretrained>"`` identifier.

        Args:
            identifier: The configuration's ``model`` field.

        Returns:
            The parsed specification.

        Raises:
            ValueError: If the identifier does not have exactly three
                non-empty, slash-separated parts, or names an unsupported
                backend.
        """
        parts = identifier.split("/")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            msg = (
                f"model identifier must be '<backend>/<architecture>/<pretrained>', "
                f"e.g. 'open_clip/ViT-B-32/openai'; got {identifier!r}"
            )
            raise ValueError(msg)
        backend, architecture, pretrained = (part.strip() for part in parts)
        if backend != _OPEN_CLIP:
            msg = f"unsupported model backend {backend!r}; only {_OPEN_CLIP!r} is implemented"
            raise ValueError(msg)
        return cls(backend=backend, architecture=architecture, pretrained=pretrained)

    def __str__(self) -> str:
        """Render back to the configuration's identifier form."""
        return f"{self.backend}/{self.architecture}/{self.pretrained}"


@dataclass(frozen=True, slots=True)
class PromptEmbedding:
    """L2-normalised text embeddings for one prompt set.

    Attributes:
        prompts: The prompt strings, in class order. Carried alongside the
            matrix so that a score can always be traced back to the exact text
            that produced it.
        matrix: Tensor of shape ``(num_prompts, embedding_dim)``, L2-normalised
            along the last axis and living on the scorer's device.
    """

    prompts: tuple[str, ...]
    matrix: torch.Tensor

    def __len__(self) -> int:
        """Number of prompts, i.e. the size of the model's output space."""
        return len(self.prompts)


class ClipScorer:
    """An open_clip checkpoint, wrapped as image+prompts to logits.

    Construct with :meth:`load` in production. The initialiser takes its
    collaborators directly so that tests can inject a stub model and never
    touch the network; `CLAUDE.md` forbids downloading a checkpoint inside the
    test suite.
    """

    def __init__(
        self,
        *,
        model: Any,  # noqa: ANN401 - open_clip ships no type information.
        preprocess: Any,  # noqa: ANN401
        tokenizer: Any,  # noqa: ANN401
        spec: CheckpointSpec,
        device: str,
        image_size: int,
    ) -> None:
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._spec = spec
        self._device = device
        self._image_size = image_size

    @property
    def spec(self) -> CheckpointSpec:
        """The checkpoint this scorer serves."""
        return self._spec

    @property
    def device(self) -> str:
        """Torch device the weights live on."""
        return self._device

    @property
    def image_size(self) -> int:
        """Side length of the square input the checkpoint expects."""
        return self._image_size

    @classmethod
    def load(
        cls,
        spec: CheckpointSpec,
        *,
        image_size: int,
        device: str | None = None,
    ) -> ClipScorer:
        """Load a checkpoint.

        Args:
            spec: The checkpoint to load.
            image_size: The configured input size. Checked against the
                checkpoint rather than trusted.
            device: Torch device; defaults to CUDA when available, else CPU.

        Returns:
            A scorer in eval mode with gradients disabled.

        Raises:
            ValueError: If the checkpoint expects a different input size than
                ``image_size``. Continuing would resize the already-rendered
                coalition a second time, silently breaking the alignment
                between the mask and the pixels the model sees.
            RuntimeError: If open_clip reports that the architecture and the
                pretrained tag do not agree -- see :meth:`_check_warnings`.
        """
        # Lazy: torch and open_clip cost seconds to import, and the dataset and
        # preview paths must stay usable without paying that.
        import warnings

        import open_clip
        import torch

        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, _, preprocess = open_clip.create_model_and_transforms(
                spec.architecture, pretrained=spec.pretrained, device=resolved
            )
        cls._check_warnings(spec, caught)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        expected = int(model.visual.image_size[0])
        if expected != image_size:
            msg = (
                f"checkpoint {spec} expects {expected}px input but the configuration "
                f"says image_size = {image_size}. Coalitions are rendered at the "
                f"configured size, so this mismatch would resize them a second time "
                f"and break the mask/pixel alignment."
            )
            raise ValueError(msg)

        return cls(
            model=model,
            preprocess=preprocess,
            tokenizer=open_clip.get_tokenizer(spec.architecture),
            spec=spec,
            device=resolved,
            image_size=image_size,
        )

    @staticmethod
    def _check_warnings(spec: CheckpointSpec, caught: Iterable[warnings.WarningMessage]) -> None:
        """Promote open_clip's architecture/weights warnings to errors.

        open_clip will happily load one checkpoint's weights into a differently
        configured architecture and merely *warn*. That is not a cosmetic
        complaint -- it changes what the model computes. The case this guard
        was written for: since open_clip 2.24 the ``ViT-B-32`` config uses
        standard GELU, while OpenAI's ``openai`` weights were trained with
        QuickGELU, so ``ViT-B-32/openai`` runs the right weights through the
        wrong non-linearity. The correct architecture name is
        ``ViT-B-32-quickgelu``. Measured on 40 FoodSeg103 test images, the two
        pick a different top-1 dish for 15 of them -- and the top-1 *is* the
        target, so the warning was worth 37 % of the run. See W01-10.

        A pytest ``filterwarnings = ["error"]`` would catch this in the suite,
        but the suite never loads a real checkpoint, so the check has to live
        where the loading happens.

        Raises:
            RuntimeError: If any warning mentions a mismatch between the
                architecture and the pretrained tag.
        """
        for warning in caught:
            text = str(warning.message)
            if "mismatch" not in text.lower():
                continue
            msg = (
                f"open_clip reports a mismatch loading {spec}: {text}\n"
                "The weights and the architecture configuration disagree, so the "
                "model would not compute what the checkpoint was trained to. For "
                "OpenAI's weights use the '-quickgelu' architecture, e.g. "
                "'open_clip/ViT-B-32-quickgelu/openai'."
            )
            raise RuntimeError(msg)

    def embed_prompts(self, prompts: Sequence[str]) -> PromptEmbedding:
        """Embed the prompt set once, for reuse across every coalition.

        Args:
            prompts: Prompt strings in class order.

        Returns:
            The L2-normalised embedding matrix with its prompts attached.

        Raises:
            ValueError: If ``prompts`` is empty. A softmax over nothing has no
                value to return.
        """
        import torch

        if not prompts:
            msg = "cannot embed an empty prompt set"
            raise ValueError(msg)
        tokens = self._tokenizer(list(prompts)).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens).float()
        features = features / features.norm(dim=-1, keepdim=True)
        return PromptEmbedding(prompts=tuple(prompts), matrix=features)

    def logits(
        self,
        images: Sequence[NDArray[np.uint8]],
        embedding: PromptEmbedding,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> NDArray[np.float32]:
        """Score rendered images against an embedded prompt set.

        Args:
            images: Rendered coalitions as RGB ``uint8`` arrays of shape
                ``(image_size, image_size, 3)`` -- the output of
                :meth:`~food_shapley.masking.BlurRemoval.render`.
            embedding: Prompt embedding from :meth:`embed_prompts`.
            batch_size: Images per forward pass. See :data:`DEFAULT_BATCH_SIZE`.

        Returns:
            Array of shape ``(len(images), len(embedding))``: CLIP's scaled
            cosine similarities, i.e. the logits a softmax would consume.

        Raises:
            ValueError: If ``batch_size`` is not positive or an image has the
                wrong shape or dtype.
        """
        import torch

        if batch_size < 1:
            msg = f"batch_size must be at least 1, got {batch_size}"
            raise ValueError(msg)
        if not images:
            return np.zeros((0, len(embedding)), dtype=np.float32)
        self._check_images(images)

        scale = float(self._model.logit_scale.exp().item())
        chunks: list[NDArray[np.float32]] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensor = torch.stack([self._to_tensor(image) for image in batch]).to(self._device)
            with torch.no_grad():
                features = self._model.encode_image(tensor).float()
            features = features / features.norm(dim=-1, keepdim=True)
            scores = scale * features @ embedding.matrix.T
            chunks.append(scores.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def _check_images(self, images: Sequence[NDArray[np.uint8]]) -> None:
        """Reject inputs the geometry contract does not allow.

        Raises:
            ValueError: If any image is not ``(size, size, 3)`` ``uint8``.
        """
        wanted = (self._image_size, self._image_size, 3)
        for position, image in enumerate(images):
            if image.shape != wanted or image.dtype != np.uint8:
                msg = (
                    f"image {position}: expected {wanted} uint8 in the checkpoint's "
                    f"geometry, got {image.shape} {image.dtype}. Render coalitions "
                    f"through food_shapley.segments so image and mask share one resize."
                )
                raise ValueError(msg)

    def _to_tensor(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """Apply the checkpoint's own preprocessing to one rendered image.

        The resize and crop inside that transform are no-ops here: the input is
        already square at the checkpoint's size, which :meth:`load` enforces.
        What remains is the normalisation, which is the checkpoint's own.
        """
        from PIL import Image

        tensor: torch.Tensor = self._preprocess(Image.fromarray(image))
        return tensor

    def provenance(self) -> dict[str, str]:
        """Return what must be recorded about this model in a run's metadata.

        The configuration pins the checkpoint by name; this pins what was
        actually loaded, so a run can be checked rather than trusted.
        """
        import open_clip
        import torch

        return {
            "backend": self._spec.backend,
            "architecture": self._spec.architecture,
            "pretrained": self._spec.pretrained,
            "open_clip_version": open_clip.__version__,
            "torch_version": torch.__version__,
            "device": self._device,
            "image_size": str(self._image_size),
            "parameter_sha256": self._parameter_hash(),
        }

    def _parameter_hash(self) -> str:
        """Hash the loaded weights.

        The identity of a checkpoint is its numbers. A pretrained tag can be
        re-uploaded; this detects that, and detects two runs that claim the
        same model and did not use it.
        """
        import hashlib

        import torch

        digest = hashlib.sha256()
        with torch.no_grad():
            for name, parameter in sorted(self._model.state_dict().items()):
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m food_shapley.scoring",
        description=(
            "Load the checkpoint a configuration names, and print its provenance. "
            "Run this on a login node: it is what populates the checkpoint cache "
            "so that offline compute nodes can load the weights locally."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke.toml"),
        help="Experiment configuration naming the checkpoint and input size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to load onto. CPU is enough to populate the cache.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: fetch, verify and describe a checkpoint.

    Loading is the verification. It is the step that detects an architecture
    that disagrees with its pretrained tag -- the failure that would otherwise
    only show up as quietly wrong numbers; see
    :meth:`ClipScorer._check_warnings` and W01-10.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero if the checkpoint cannot be loaded as
        configured.
    """
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    spec = CheckpointSpec.parse(config.model)
    try:
        scorer = ClipScorer.load(spec, image_size=config.image_size, device=args.device)
    except (RuntimeError, ValueError) as error:
        print(f"checkpoint {spec} FAILED to load: {error}", file=sys.stderr)
        return 1
    for key, value in scorer.provenance().items():
        print(f"{key:<18} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
