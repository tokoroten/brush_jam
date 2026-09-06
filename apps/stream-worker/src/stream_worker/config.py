"""The worker's own configuration.

Everything about the *model* - the checkpoint, the LoRA, the VAE, the sizes,
the offloading, the caches - belongs to the pipeline, and the pipeline is
`brushjam.ai.pipeline`, which this worker now imports rather than duplicates.
`PipelineSettings` reads `INPROC_*` and falls back to `STREAM_*`, so every
variable documented in docs/STREAM_WORKER.md keeps working exactly as it did.

What is left here is what the pipeline has no opinion about: which address to
listen on, whether to serve the contract with no model at all, and the
worker-only defaults (`STREAM_STEPS`, `STREAM_QUALITY_SUFFIX`).

The worker runs one profile - the few-step one - so where the pipeline offers a
choice, this asks for `fast` and reports the fast numbers on /healthz.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from brushjam.ai.pipeline import (
    LORA_REPOS,
    VAE_SOURCES,
    PipelineSettings,
)

__all__ = ["Settings", "LORA_REPOS", "VAE_SOURCES", "PipelineSettings"]

#: The profile the worker runs. It holds one few-step LoRA and has never
#: offered anything else; the server's `stream` backend advertises `fast` only.
PROFILE = "fast"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def pipeline_settings() -> PipelineSettings:
    """The shared pipeline settings, with the two knobs that are the worker's.

    `STREAM_STEPS` is the worker's default step count, which the pipeline calls
    `fast_steps`; `STREAM_QUALITY_SUFFIX` is appended to every prompt. Every
    other `STREAM_*` variable is read by `PipelineSettings` itself.
    """
    settings = PipelineSettings()
    return replace(
        settings,
        fast_steps=_env_int("STREAM_STEPS", settings.fast_steps),
        quality_suffix=os.environ.get("STREAM_QUALITY_SUFFIX", settings.quality_suffix),
    )


@dataclass(frozen=True)
class Settings:
    host: str = field(default_factory=lambda: os.environ.get("STREAM_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("STREAM_PORT", 8790))
    #: Debug/CI escape hatch: serve the contract without touching the GPU.
    dry_run: bool = field(default_factory=lambda: _env_bool("STREAM_DRY_RUN", False))
    #: Everything about the model. Shared with the room server, one definition.
    pipeline: PipelineSettings = field(default_factory=pipeline_settings)

    # -- what the HTTP surface asks about the model ------------------------
    # Delegates rather than copies: two dataclasses holding the same numbers is
    # how the worker's pipeline drifted a release behind the server's in the
    # first place.

    @property
    def checkpoint(self) -> Path:
        return self.pipeline.checkpoint

    @property
    def lora(self) -> str:
        return self.pipeline.lora

    @property
    def lora_dir(self) -> Path:
        return self.pipeline.lora_dir

    @property
    def vae(self) -> str:
        return self.pipeline.vae

    @property
    def max_size(self) -> int:
        return self.pipeline.max_size

    @property
    def max_denoise(self) -> float:
        return self.pipeline.max_denoise

    @property
    def warmup_size(self) -> int:
        return self.pipeline.warmup_size

    @property
    def default_steps(self) -> int:
        return self.pipeline.steps_for(PROFILE)

    @property
    def guidance(self) -> float:
        """Guidance for the profile this worker runs.

        `STREAM_GUIDANCE` names it; unset, the LoRA decides (DMD2 is
        guidance-distilled and wants 1.0, LCM wants a little).
        """
        return self.pipeline.guidance_for(PROFILE)

    def negative_prompt_active(self) -> bool:
        """Whether negative_prompt has any effect at the current guidance.

        Classifier-free guidance is what makes the negative prompt do anything:
        at guidance 1.0 the pipeline runs the conditional branch only, so the
        negative embeddings are computed and then never used.
        """
        return self.pipeline.negative_prompt_active()[PROFILE]

    def lora_spec(self) -> tuple[str, str, str]:
        return self.pipeline.lora_spec()

    def lora_path(self) -> Path:
        return self.pipeline.lora_path()

    def vae_spec(self) -> tuple[str | None, str | None]:
        return self.pipeline.vae_spec()
