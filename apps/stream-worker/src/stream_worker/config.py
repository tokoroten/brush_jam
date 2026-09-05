"""Environment-driven configuration for the stream worker."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: No default: the checkpoint is a 6-7 GB file that cannot live in the repo,
#: so STREAM_CHECKPOINT has to name it. Any SDXL .safetensors works.
DEFAULT_CHECKPOINT = ""
#: Small enough to download on demand, so this one defaults into the checkout.
DEFAULT_LORA_DIR = str(Path(__file__).resolve().parents[4] / "models" / "loras")

# Both are 4-step SDXL distillation LoRAs. LCM is the default; DMD2 is the
# fallback documented in docs/STREAM_WORKER.md.
# Alternative VAEs. The checkpoint's own SDXL VAE sets force_upcast=True, so
# diffusers casts it to fp32 on every call - see docs/STREAM_WORKER.md 4.4.
VAE_SOURCES = {
    # Same architecture, weights rescaled so fp16 does not overflow: a drop-in
    # replacement with no quality loss and no upcast.
    "fp16fix": ("madebyollin/sdxl-vae-fp16-fix", "AutoencoderKL"),
    # Distilled ~1M-parameter VAE: far faster, slightly softer output.
    "taesd": ("madebyollin/taesdxl", "AutoencoderTiny"),
    # Whatever is baked into the checkpoint (the upcasting one).
    "checkpoint": (None, None),
}

LORA_REPOS = {
    "lcm": ("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors", "lcm-lora-sdxl.safetensors"),
    "dmd2": ("tianweiy/DMD2", "dmd2_sdxl_4step_lora_fp16.safetensors", "dmd2_sdxl_4step_lora_fp16.safetensors"),
}


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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    host: str = field(default_factory=lambda: os.environ.get("STREAM_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("STREAM_PORT", 8790))
    checkpoint: Path = field(default_factory=lambda: Path(os.environ.get("STREAM_CHECKPOINT", DEFAULT_CHECKPOINT)))
    lora_dir: Path = field(default_factory=lambda: Path(os.environ.get("STREAM_LORA_DIR", DEFAULT_LORA_DIR)))
    # dmd2 over lcm: 1384 ms vs 1798 ms at 768, reinterprets a denoise step
    # earlier, and being guidance-distilled it costs nothing to run at CFG 1.0.
    # docs/experiments/2026-09-05-stream/REPORT.md section 8.
    lora: str = field(default_factory=lambda: os.environ.get("STREAM_LORA", "dmd2").lower())
    # Only used when the LoRA file is missing from lora_dir and must be fetched.
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN") or None)

    warmup_size: int = field(default_factory=lambda: _env_int("STREAM_WARMUP_SIZE", 768))
    max_size: int = field(default_factory=lambda: _env_int("STREAM_MAX_SIZE", 1024))
    default_steps: int = field(default_factory=lambda: _env_int("STREAM_STEPS", 4))
    # 1.0 = CFG off, which halves the UNet work (~22% end to end) and, with the
    # DMD2 LoRA, costs no measurable quality. The price is that negative_prompt
    # becomes inert; /healthz publishes negative_prompt_active so the UI can say
    # so rather than offering a field that does nothing.
    guidance: float = field(default_factory=lambda: _env_float("STREAM_GUIDANCE", 1.0))
    # Highest denoise the worker will honour. At 4 LCM steps the top of the
    # range is where the model stops reinterpreting the drawing and starts
    # replacing it; the server reads this from /healthz to cap the room slider
    # rather than hard-coding a number that only makes sense for this backend.
    max_denoise: float = field(default_factory=lambda: _env_float("STREAM_MAX_DENOISE", 0.9))
    quality_suffix: str = field(
        default_factory=lambda: os.environ.get("STREAM_QUALITY_SUFFIX", ", masterpiece, best quality")
    )
    # Text encoders live on the CPU between requests: they are ~1.8 GB in fp16
    # and every embedding we need is cached, so on an 8 GB card that VRAM is
    # better spent on the UNet. Set 0 if you have headroom.
    offload_text_encoders: bool = field(default_factory=lambda: _env_bool("STREAM_OFFLOAD_TEXT_ENCODERS", True))
    # Hand fragmented blocks back after every generation; on 8 GB the allocator
    # otherwise reserves ~2.5 GB more than it uses and the next run spills.
    empty_cache_each_run: bool = field(default_factory=lambda: _env_bool("STREAM_EMPTY_CACHE", True))
    # Which VAE to run. fp16fix removes the fp32 upcast that dominates latency.
    vae: str = field(default_factory=lambda: os.environ.get("STREAM_VAE", "fp16fix").lower())
    vae_tiling: bool = field(default_factory=lambda: _env_bool("STREAM_VAE_TILING", True))
    embed_cache_size: int = field(default_factory=lambda: _env_int("STREAM_EMBED_CACHE", 16))
    # Debug/CI escape hatch: serve the contract without touching the GPU.
    dry_run: bool = field(default_factory=lambda: _env_bool("STREAM_DRY_RUN", False))

    def lora_spec(self) -> tuple[str, str, str]:
        if self.lora not in LORA_REPOS:
            raise ValueError(f"STREAM_LORA must be one of {sorted(LORA_REPOS)}, got {self.lora!r}")
        return LORA_REPOS[self.lora]

    def lora_path(self) -> Path:
        return self.lora_dir / self.lora_spec()[2]

    def negative_prompt_active(self) -> bool:
        """Whether negative_prompt has any effect at the current guidance.

        Classifier-free guidance is what makes the negative prompt do anything:
        at guidance 1.0 the pipeline runs the conditional branch only, so the
        negative embeddings are computed and then never used.
        """
        return self.guidance > 1.0

    def vae_spec(self) -> tuple[str | None, str | None]:
        if self.vae not in VAE_SOURCES:
            raise ValueError(f"STREAM_VAE must be one of {sorted(VAE_SOURCES)}, got {self.vae!r}")
        return VAE_SOURCES[self.vae]
