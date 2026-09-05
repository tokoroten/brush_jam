"""Model-resident few-step SDXL img2img pipeline, in this process.

Moved from apps/stream-worker/src/stream_worker/pipeline.py, with one design
change: the LoRA is **attached and detached per request** instead of fused at
load time, so a single resident model serves both room profiles -

    fast     DMD2 (or LCM) LoRA on, LCMScheduler, 4 steps, cfg 1.0
    quality  LoRA off, EulerAncestralDiscreteScheduler, 14 steps, cfg 5.5

Fusing was right for a fast-only worker (no per-step PEFT overhead, no second
copy of the deltas) but it cannot be undone cheaply, so it made `quality`
impossible without a reload. Attaching costs a little per step and a little
VRAM; it buys both profiles from one 8 GB-resident model.

Everything else is kept as measured: the fp16-fix VAE (the checkpoint's own VAE
forces an fp32 upcast that dominated latency), text encoders parked in system
RAM with a prompt-embedding cache, cooperative cancellation at step boundaries,
and `torch.cuda.empty_cache()` in a `finally` on every exit path.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger("brushjam.ai.pipeline")

#: Multiple of 8 is a VAE requirement; anything above max_size is refused rather
#: than silently producing mush (and OOMing at 8 GB).
SIZE_MULTIPLE = 8
MIN_SIZE = 256

DEFAULT_CHECKPOINT = r"E:\ComfyUI\models\checkpoints\waiNSFWIllustrious_v150.safetensors"
DEFAULT_LORA_DIR = r"E:\ComfyUI\models\loras"

#: The checkpoint's own SDXL VAE sets force_upcast=True, so diffusers casts it
#: to fp32 on every call - see docs/STREAM_WORKER.md 4.4.
VAE_SOURCES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    # Same architecture, weights rescaled so fp16 does not overflow.
    "fp16fix": ("madebyollin/sdxl-vae-fp16-fix", "AutoencoderKL"),
    # Distilled ~1M-parameter VAE: far faster, slightly softer output.
    "taesd": ("madebyollin/taesdxl", "AutoencoderTiny"),
    # Whatever is baked into the checkpoint (the upcasting one).
    "checkpoint": (None, None),
}

LORA_REPOS: Dict[str, Tuple[str, str, str]] = {
    "lcm": (
        "latent-consistency/lcm-lora-sdxl",
        "pytorch_lora_weights.safetensors",
        "lcm-lora-sdxl.safetensors",
    ),
    "dmd2": (
        "tianweiy/DMD2",
        "dmd2_sdxl_4step_lora_fp16.safetensors",
        "dmd2_sdxl_4step_lora_fp16.safetensors",
    ),
}

#: DMD2 is guidance-distilled and wants no guidance at all; LCM wants a little.
FAST_GUIDANCE = {"dmd2": 1.0, "lcm": 1.5}

LCM_TRAIN_TIMESTEPS = 1000
LCM_ORIGINAL_INFERENCE_STEPS = 50


def _env(name: str, default: str) -> str:
    """`INPROC_X`, falling back to the stream worker's `STREAM_X` so an existing
    .env keeps working after the pipeline moved in here."""
    for key in (f"INPROC_{name}", f"STREAM_{name}"):
        value = os.environ.get(key)
        if value is not None and value != "":
            return value
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "")
    if raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name, "")
    return default if raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name, "")
    return default if raw == "" else float(raw)


@dataclass
class PipelineSettings:
    checkpoint: Path = field(default_factory=lambda: Path(_env("CHECKPOINT", DEFAULT_CHECKPOINT)))
    lora_dir: Path = field(default_factory=lambda: Path(_env("LORA_DIR", DEFAULT_LORA_DIR)))
    #: dmd2 over lcm: faster, and it reinterprets a drawing a denoise step
    #: earlier (docs/experiments/2026-09-05-stream/REPORT.md section 8).
    lora: str = field(default_factory=lambda: _env("LORA", "dmd2").lower())
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN") or None)
    vae: str = field(default_factory=lambda: _env("VAE", "fp16fix").lower())
    vae_tiling: bool = field(default_factory=lambda: _env_bool("VAE_TILING", True))
    max_size: int = field(default_factory=lambda: _env_int("MAX_SIZE", 1024))
    max_denoise: float = field(default_factory=lambda: _env_float("MAX_DENOISE", 0.9))
    warmup_size: int = field(default_factory=lambda: _env_int("WARMUP_SIZE", 768))
    fast_steps: int = 4
    quality_steps: int = 14
    #: Guidance for the quality profile; the fast one is chosen by its LoRA.
    quality_guidance: float = 5.5
    quality_suffix: str = ", masterpiece, best quality"
    #: Text encoders are ~1.8 GB in fp16 and every embedding is cached, so on an
    #: 8 GB card that VRAM is better spent on the UNet.
    offload_text_encoders: bool = field(
        default_factory=lambda: _env_bool("OFFLOAD_TEXT_ENCODERS", True)
    )
    #: Hand fragmented blocks back after every generation; on 8 GB the allocator
    #: otherwise reserves ~2.5 GB more than it uses and the next run spills.
    empty_cache_each_run: bool = field(default_factory=lambda: _env_bool("EMPTY_CACHE", True))
    embed_cache_size: int = field(default_factory=lambda: _env_int("EMBED_CACHE", 16))

    def lora_spec(self) -> Tuple[str, str, str]:
        if self.lora not in LORA_REPOS:
            raise ValueError(f"LORA must be one of {sorted(LORA_REPOS)}, got {self.lora!r}")
        return LORA_REPOS[self.lora]

    def lora_path(self) -> Path:
        return self.lora_dir / self.lora_spec()[2]

    def vae_spec(self) -> Tuple[Optional[str], Optional[str]]:
        if self.vae not in VAE_SOURCES:
            raise ValueError(f"VAE must be one of {sorted(VAE_SOURCES)}, got {self.vae!r}")
        return VAE_SOURCES[self.vae]

    def fast_guidance(self) -> float:
        return FAST_GUIDANCE.get(self.lora, 1.5)

    def guidance_for(self, profile: str) -> float:
        return self.fast_guidance() if profile == "fast" else self.quality_guidance

    def steps_for(self, profile: str) -> int:
        return self.fast_steps if profile == "fast" else self.quality_steps

    def negative_prompt_active(self) -> Dict[str, bool]:
        """Classifier-free guidance is what makes the negative prompt do
        anything: at guidance 1.0 only the conditional branch runs."""
        return {
            "fast": self.fast_guidance() > 1.0,
            "quality": self.quality_guidance > 1.0,
        }


def round_size(value: int, max_size: int) -> int:
    v = int(value)
    if v < MIN_SIZE or v > max_size:
        raise ValueError(f"size {v} out of range [{MIN_SIZE}, {max_size}]")
    return v - (v % SIZE_MULTIPLE)


def composite_through_mask(
    base: Image.Image, generated: Image.Image, mask: Optional[Image.Image]
) -> Image.Image:
    """out = base*(1-m) + generated*m, with a soft (8-bit) mask."""
    base_rgb = base.convert("RGB")
    gen_rgb = generated.convert("RGB")
    if gen_rgb.size != base_rgb.size:
        gen_rgb = gen_rgb.resize(base_rgb.size, Image.LANCZOS)
    if mask is None:
        return gen_rgb
    m = mask.convert("L")
    if m.size != base_rgb.size:
        m = m.resize(base_rgb.size, Image.BILINEAR)
    a = np.asarray(m, dtype=np.float32)[..., None] / 255.0
    out = np.asarray(base_rgb, dtype=np.float32) * (1.0 - a) + np.asarray(gen_rgb, dtype=np.float32) * a
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), mode="RGB")


def steps_for_strength(steps: int, strength: float) -> int:
    """Scheduler steps needed so that ``steps`` are actually run.

    diffusers' img2img keeps only the last ``int(num_inference_steps *
    strength)`` timesteps, so asking for 14 steps at 0.7 really runs 9. Used for
    the *quality* profile only, where the resolution loss from the double
    rounding is ~1/14 and the sampler is an ordinary many-step one. The fast
    profile uses an explicit LCM schedule instead - see
    :func:`lcm_timesteps_for_strength` for why that had to change.
    """
    strength = max(0.05, min(1.0, strength))
    return max(steps, int(math.ceil(steps / strength)))


def lcm_timesteps_for_strength(
    steps: int,
    strength: float,
    *,
    num_train_timesteps: int = LCM_TRAIN_TIMESTEPS,
    original_inference_steps: int = LCM_ORIGINAL_INFERENCE_STEPS,
) -> List[int]:
    """Descending LCM timestep schedule of exactly ``steps`` entries.

    Rather than asking diffusers to derive a start index from a rounded step
    count, pick the start point on the LCM distillation schedule directly from
    ``strength``: at 4 steps the two-rounding path made 0.8 and 0.9 produce
    byte-identical images (docs/experiments/2026-09-05-stream/REPORT.md 4).

    The returned list is passed to the pipeline as ``timesteps=``; the pipeline
    must then be called with ``strength=1.0`` so it does not slice it again.
    """
    steps = max(1, int(steps))
    strength = max(0.02, min(1.0, float(strength)))
    n_train = max(1, int(original_inference_steps))
    k = max(1, num_train_timesteps // n_train)
    schedule = [k * i - 1 for i in range(1, n_train + 1)]  # ascending

    start_index = int(round(strength * n_train)) - 1
    start_index = max(0, min(len(schedule) - 1, start_index))

    # Cannot run more distinct timesteps than exist below the start point.
    count = min(steps, start_index + 1)
    if count == 1:
        return [schedule[start_index]]
    indices = [int(round(start_index * (1.0 - j / (count - 1)))) for j in range(count)]
    return [schedule[i] for i in indices]


@contextmanager
def _quiet_lcm_custom_timestep_warning() -> Iterator[None]:
    """Silence one unavoidable, wrong warning from LCMScheduler.set_timesteps.

    It warns when ``timesteps[0]`` is not 999 and ``strength`` is 1.0. We always
    pass strength=1.0 - our schedule already encodes the strength - so it fires
    on every request below full strength and says nothing true.
    """
    lcm_log = logging.getLogger("diffusers.schedulers.scheduling_lcm")
    previous = lcm_log.level
    lcm_log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        lcm_log.setLevel(previous)


class GenerationCancelled(Exception):
    """Raised inside the diffusion loop when a request is cancelled.

    Dropping the caller cannot stop GPU work: the pipeline call is already
    running on a worker thread and would finish regardless, holding the GPU
    while the next request queues behind it. Cancellation is therefore
    cooperative - the caller sets a flag and the step callback notices.
    """

    def __init__(self, request_id: Optional[str] = None) -> None:
        super().__init__(f"generation cancelled ({request_id or 'unknown'})")
        self.request_id = request_id


@dataclass
class GenerateResult:
    image: Image.Image
    timings: Dict[str, float]


class InprocPipeline:
    """One resident SDXL img2img model serving both room profiles."""

    backend = "diffusers-sdxl-inproc"

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        self.settings = settings or PipelineSettings()
        self.pipe: Any = None
        self.warm = False
        self.device = "cuda"
        self._embed_cache: "OrderedDict[Tuple[str, str, bool], Tuple[Any, ...]]" = OrderedDict()
        self._torch: Any = None
        self._vae_timings: Dict[str, float] = {}
        self._schedulers: Dict[str, Any] = {}
        #: Which profile the pipeline is currently configured for, so an
        #: unchanged profile costs nothing.
        self._profile: Optional[str] = None
        self._has_lora = False
        #: Whether the fast adapter is currently merged into the UNet weights.
        self._fused = False
        #: Set once if this diffusers build cannot fuse, so the fallback is
        #: reported rather than silently costing a per-step tax forever.
        self._fuse_unavailable = False
        #: Cost of the last profile switch, surfaced in the run's timings.
        self._last_switch_ms = 0.0

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        import torch
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            LCMScheduler,
            StableDiffusionXLImg2ImgPipeline,
        )

        self._torch = torch
        s = self.settings
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; the inproc backend needs a GPU")
        if not s.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {s.checkpoint}")

        t0 = time.perf_counter()
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            str(s.checkpoint),
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
        )
        pipe.set_progress_bar_config(disable=True)
        # Both schedulers are built from the same config once, and swapped per
        # request. LCM is meaningless without the few-step LoRA and euler_a is
        # wrong with it, so the scheduler travels with the profile.
        base_config = pipe.scheduler.config
        self._schedulers = {
            "fast": LCMScheduler.from_config(base_config),
            "quality": EulerAncestralDiscreteScheduler.from_config(base_config),
        }

        lora = ensure_lora(s)
        # Attached, NOT fused: fusing cannot be undone cheaply, which is what
        # made a quality profile impossible in the worker.
        pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="fast")
        self._has_lora = True

        self._install_vae(pipe)
        pipe.to(self.device)
        if s.vae_tiling:
            # AutoencoderTiny gained tiling later than AutoencoderKL.
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
        self._instrument_vae(pipe)
        self.pipe = pipe
        self._select_profile("fast")
        self._park_text_encoders()
        log.info("loaded %s + %s in %.1fs", s.checkpoint.name, lora.name, time.perf_counter() - t0)

        if s.warmup_size:
            self.warmup(s.warmup_size)
        self.warm = True

    def _install_vae(self, pipe: Any) -> None:
        """Swap in a VAE that does not need the fp32 upcast."""
        torch = self._torch
        repo, cls_name = self.settings.vae_spec()
        if repo is None:
            log.info("VAE: checkpoint's own (force_upcast=%s)", getattr(pipe.vae.config, "force_upcast", None))
            return
        import diffusers

        vae_cls = getattr(diffusers, cls_name)
        t0 = time.perf_counter()
        vae = vae_cls.from_pretrained(repo, torch_dtype=torch.float16)
        # The replacement is fp16-safe by construction, but the flag is what
        # diffusers actually checks before upcasting.
        if hasattr(vae, "config") and hasattr(vae.config, "force_upcast"):
            vae.config.force_upcast = False
        pipe.vae = vae
        log.info("VAE: %s (%s) loaded in %.1fs", repo, cls_name, time.perf_counter() - t0)

    def _instrument_vae(self, pipe: Any) -> None:
        """Time VAE encode/decode separately from the UNet."""
        torch = self._torch
        vae = pipe.vae
        if getattr(vae, "_brushjam_timed", False):
            return

        def wrap(name: str, fn: Any) -> Any:
            def timed(*args: Any, **kwargs: Any) -> Any:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    torch.cuda.synchronize()
                    self._vae_timings[name] = self._vae_timings.get(name, 0.0) + (
                        time.perf_counter() - t0
                    ) * 1000.0

            return timed

        vae.encode = wrap("vae_encode_ms", vae.encode)
        vae.decode = wrap("vae_decode_ms", vae.decode)
        vae._brushjam_timed = True

    # ---------------------------------------------------------------- profile
    def _select_profile(self, profile: str) -> float:
        """Point the resident model at one profile's scheduler and LoRA state.

        Returns the guidance scale for it. Idempotent: an unchanged profile
        costs nothing, which is what makes a per-request choice reasonable.

        The LoRA is **fused into the UNet weights** for `fast` and unfused for
        `quality`, rather than left attached and toggled. An attached PEFT
        adapter runs its own matmuls on every Linear on every step; the worker
        this was ported from fused once at load and paid none of that
        (`apps/stream-worker/src/stream_worker/pipeline.py`). Fusing is a
        weight operation on weights that are already resident, so the adapter
        stays loaded and switching never touches the disk.
        """
        if profile not in ("fast", "quality"):
            profile = "quality"
        if self._profile == profile:
            self._last_switch_ms = 0.0
            return self.settings.guidance_for(profile)

        t0 = time.perf_counter()
        scheduler = self._schedulers.get(profile)
        if scheduler is not None:
            self.pipe.scheduler = scheduler
        if self._has_lora:
            self._set_lora(profile == "fast")
        self._last_switch_ms = (time.perf_counter() - t0) * 1000.0
        self._profile = profile
        log.info(
            "profile -> %s (lora %s) in %.0f ms",
            profile,
            self._lora_state(),
            self._last_switch_ms,
        )
        return self.settings.guidance_for(profile)

    def _lora_state(self) -> str:
        if not self._has_lora:
            return "none"
        if self._fused:
            return "fused"
        return "attached" if self._profile == "fast" else "off"

    def _set_lora(self, on: bool) -> None:
        """Merge the adapter into the base weights, or take it back out.

        The two must stay consistent: PEFT's own forward *unmerges* a merged
        layer as soon as adapters are disabled, so `disable_lora()` while fused
        would quietly undo the fuse on the first step. Enable-then-fuse, and
        unfuse-then-disable.
        """
        if not self._fuse_unavailable:
            try:
                if on:
                    self.pipe.enable_lora()
                    self.pipe.set_adapters(["fast"], adapter_weights=[1.0])
                    if not self._fused:
                        # Only the UNet: this LoRA has no text-encoder keys, and
                        # the text encoders are parked in system RAM anyway.
                        self.pipe.fuse_lora(
                            components=["unet"], adapter_names=["fast"], lora_scale=1.0
                        )
                        self._fused = True
                else:
                    if self._fused:
                        self.pipe.unfuse_lora(components=["unet"])
                        self._fused = False
                    self.pipe.disable_lora()
                return
            except Exception:  # noqa: BLE001
                # A build without fuse support, or a fuse that refused. Fall
                # back to the attached adapter, which is correct but pays the
                # per-step cost.
                log.warning("fuse_lora unavailable, falling back to an attached adapter", exc_info=True)
                self._fuse_unavailable = True
                self._fused = False

        if on:
            self.pipe.enable_lora()
            self.pipe.set_adapters(["fast"], adapter_weights=[1.0])
        else:
            self.pipe.disable_lora()

    def warmup(self, size: int) -> None:
        """One throwaway generation so CUDA kernels and autotuning are paid for."""
        size = round_size(size, self.settings.max_size)
        blank = Image.new("RGB", (size, size), "white")
        t0 = time.perf_counter()
        self.generate(
            image=blank,
            mask=None,
            prompt="warmup",
            negative_prompt="",
            strength=0.5,
            steps=self.settings.fast_steps,
            seed=0,
            width=size,
            height=size,
            profile="fast",
        )
        log.info("warmup at %d took %.2fs", size, time.perf_counter() - t0)

    # ------------------------------------------------------------- text cache
    def _park_text_encoders(self) -> None:
        if not self.settings.offload_text_encoders or self.pipe is None:
            return
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        self._torch.cuda.empty_cache()

    def _embeds(self, prompt: str, negative: str, cfg: bool) -> Tuple[Any, ...]:
        key = (prompt, negative, cfg)
        hit = self._embed_cache.get(key)
        if hit is not None:
            self._embed_cache.move_to_end(key)
            return hit

        if self.settings.offload_text_encoders:
            self.pipe.text_encoder.to(self.device)
            self.pipe.text_encoder_2.to(self.device)
        try:
            with self._torch.no_grad():
                embeds = self.pipe.encode_prompt(
                    prompt=prompt,
                    prompt_2=None,
                    device=self._torch.device(self.device),
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=cfg,
                    negative_prompt=negative if cfg else None,
                )
        finally:
            self._park_text_encoders()

        self._embed_cache[key] = embeds
        while len(self._embed_cache) > max(1, self.settings.embed_cache_size):
            self._embed_cache.popitem(last=False)
        return embeds

    # ------------------------------------------------------------- generation
    def generate(
        self,
        *,
        image: Image.Image,
        mask: Optional[Image.Image],
        prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
        profile: str = "fast",
        should_cancel: Optional[Callable[[], bool]] = None,
        request_id: Optional[str] = None,
    ) -> GenerateResult:
        torch = self._torch
        s = self.settings
        timings: Dict[str, float] = {}
        self._vae_timings = {}

        def check_cancel() -> None:
            if should_cancel is not None and should_cancel():
                raise GenerationCancelled(request_id)

        t_start = time.perf_counter()
        try:
            base = image.convert("RGB")
            if base.size != (width, height):
                base = base.resize((width, height), Image.LANCZOS)

            guidance = self._select_profile(profile)
            timings["profile_fast"] = 1.0 if profile == "fast" else 0.0
            timings["guidance"] = guidance
            timings["profile_switch_ms"] = round(self._last_switch_ms, 2)
            timings["lora_fused"] = 1.0 if self._fused else 0.0

            t = time.perf_counter()
            cfg_on = guidance > 1.0
            prompt_text = (prompt or "").strip() + s.quality_suffix
            cached = (prompt_text, negative_prompt or "", cfg_on) in self._embed_cache
            embeds = self._embeds(prompt_text, negative_prompt or "", cfg_on)
            timings["prompt_ms"] = (time.perf_counter() - t) * 1000.0
            timings["prompt_cached"] = 1.0 if cached else 0.0

            prompt_embeds, negative_embeds, pooled, negative_pooled = embeds
            generator = torch.Generator(device=self.device).manual_seed(int(seed) & 0x7FFFFFFF)
            check_cancel()

            steps_done = [0]

            def on_step_end(_pipe: Any, step: int, _timestep: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
                # The only place inside pipe() where we get control back.
                steps_done[0] = step + 1
                check_cancel()
                return kwargs

            call: Dict[str, Any] = {}
            if profile == "fast":
                # Strength as an explicit timestep schedule, so nearby
                # strengths stay distinct at 4 steps.
                schedule = lcm_timesteps_for_strength(steps, strength)
                call = {"strength": 1.0, "timesteps": schedule, "num_inference_steps": len(schedule)}
                timings["t_start"] = float(schedule[0])
                timings["steps_effective"] = float(len(schedule))
            else:
                # Ordinary many-step sampler: ask for enough scheduler steps
                # that `steps` survive the strength slice, which is the
                # ComfyUI semantic ("steps is the real sampler-step count").
                scheduled = steps_for_strength(steps, strength)
                call = {"strength": strength, "num_inference_steps": scheduled}
                timings["t_start"] = float(scheduled)
                timings["steps_effective"] = float(max(1, int(scheduled * strength)))
            timings["steps_requested"] = float(steps)

            torch.cuda.synchronize()
            t = time.perf_counter()
            with _quiet_lcm_custom_timestep_warning():
                out = self.pipe(
                    callback_on_step_end=on_step_end,
                    image=base,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    pooled_prompt_embeds=pooled,
                    negative_pooled_prompt_embeds=negative_pooled,
                    guidance_scale=guidance,
                    generator=generator,
                    output_type="pil",
                    **call,
                )
            torch.cuda.synchronize()
            timings["diffusion_ms"] = (time.perf_counter() - t) * 1000.0
            timings["steps_run"] = float(steps_done[0])

            timings["vae_encode_ms"] = round(self._vae_timings.get("vae_encode_ms", 0.0), 2)
            timings["vae_decode_ms"] = round(self._vae_timings.get("vae_decode_ms", 0.0), 2)
            # The remainder, so it carries scheduler and conditioning overhead
            # too: a bound, not a pure UNet number.
            timings["unet_ms"] = round(
                max(0.0, timings["diffusion_ms"] - timings["vae_encode_ms"] - timings["vae_decode_ms"]), 2
            )

            check_cancel()
            t = time.perf_counter()
            composed = composite_through_mask(base, out.images[0], mask)
            if composed.size != (width, height):
                raise RuntimeError(f"internal error: produced {composed.size}, expected {(width, height)}")
            timings["composite_ms"] = (time.perf_counter() - t) * 1000.0
            timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "gen %s %s %dpx: prompt %.0f (cached %s) unet %.0f vae_enc %.0f "
                    "vae_dec %.0f composite %.0f total %.0f ms",
                    request_id or "-",
                    profile,
                    width,
                    timings["prompt_ms"],
                    bool(timings["prompt_cached"]),
                    timings["unet_ms"],
                    timings["vae_encode_ms"],
                    timings["vae_decode_ms"],
                    timings["composite_ms"],
                    timings["total_ms"],
                )
            return GenerateResult(image=composed, timings=timings)
        finally:
            # Must run on every exit path: a cancelled or failed run allocated
            # the same activations, and cancellation is exactly when the next
            # request is imminent (docs/STREAM_WORKER.md 4.3).
            self._release_allocator_cache(timings)

    def _release_allocator_cache(self, timings: Dict[str, float]) -> None:
        """Return the run's transient blocks to the driver. Never raises: it
        runs in a `finally`, and masking the real exception with a cleanup
        failure would be worse than a full cache."""
        if not self.settings.empty_cache_each_run or self._torch is None:
            return
        try:
            t = time.perf_counter()
            self._torch.cuda.empty_cache()
            timings["empty_cache_ms"] = (time.perf_counter() - t) * 1000.0
        except Exception:  # pragma: no cover - driver-level failure
            log.warning("empty_cache failed", exc_info=True)

    def unload(self) -> None:
        """Release the model and its VRAM so another process can have the GPU."""
        self.pipe = None
        self.warm = False
        self._profile = None
        self._fused = False
        self._embed_cache.clear()
        if self._torch is not None:
            import gc

            gc.collect()
            self._torch.cuda.empty_cache()
            self._torch.cuda.ipc_collect()
        log.info("unloaded: GPU released")

    def memory(self) -> Dict[str, float]:
        if self._torch is None or not self._torch.cuda.is_available():
            return {}
        free, total = self._torch.cuda.mem_get_info()
        gb = 1024.0**3
        return {
            "allocated_gb": round(self._torch.cuda.memory_allocated() / gb, 2),
            "reserved_gb": round(self._torch.cuda.memory_reserved() / gb, 2),
            "max_allocated_gb": round(self._torch.cuda.max_memory_allocated() / gb, 2),
            "device_free_gb": round(free / gb, 2),
            "device_total_gb": round(total / gb, 2),
        }

    def model_name(self) -> str:
        return f"{self.settings.checkpoint.name}+{self.settings.lora_spec()[2]}"


class DryRunPipeline:
    """The contract without a GPU: echoes its input, honours cancellation.

    Used by `INPROC_DRY_RUN=1` and by every test, so the backend, the scheduler
    and the room all run their real code paths with no model anywhere.
    """

    backend = "dry-run"

    def __init__(self, settings: Optional[PipelineSettings] = None, latency_ms: float = 0) -> None:
        self.settings = settings or PipelineSettings()
        self.warm = True
        self.latency_ms = latency_ms
        self.calls: List[Dict[str, Any]] = []
        self._profile: Optional[str] = None

    def load(self) -> None:
        self.warm = True

    def unload(self) -> None:
        self.warm = False

    def warmup(self, size: int) -> None:
        return None

    def memory(self) -> Dict[str, float]:
        return {}

    def model_name(self) -> str:
        return "dry-run"

    def generate(
        self,
        *,
        image: Image.Image,
        mask: Optional[Image.Image],
        prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
        profile: str = "fast",
        should_cancel: Optional[Callable[[], bool]] = None,
        request_id: Optional[str] = None,
    ) -> GenerateResult:
        self._profile = profile
        guidance = self.settings.guidance_for(profile)
        self.calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "steps": steps,
                "seed": seed,
                "size": (width, height),
                "profile": profile,
                "guidance": guidance,
                "has_mask": mask is not None,
            }
        )
        deadline = time.monotonic() + self.latency_ms / 1000
        while time.monotonic() < deadline:
            if should_cancel is not None and should_cancel():
                raise GenerationCancelled(request_id)
            time.sleep(0.005)
        if should_cancel is not None and should_cancel():
            raise GenerationCancelled(request_id)
        base = image.convert("RGB")
        if base.size != (width, height):
            base = base.resize((width, height), Image.LANCZOS)
        schedule = (
            lcm_timesteps_for_strength(steps, strength)
            if profile == "fast"
            else list(range(steps_for_strength(steps, strength)))
        )
        return GenerateResult(
            image=composite_through_mask(base, base, mask),
            timings={
                "dry_run": 1.0,
                "total_ms": 0.0,
                "steps_requested": float(steps),
                "steps_effective": float(len(schedule)),
                "guidance": guidance,
                "profile_fast": 1.0 if profile == "fast" else 0.0,
            },
        )


def ensure_lora(settings: PipelineSettings) -> Path:
    """Return the local LoRA path, downloading it into the ComfyUI loras dir.

    The directory is shared with ComfyUI deliberately, so one file backs both
    this pipeline and a 4-step ComfyUI workflow.
    """
    repo, filename, local_name = settings.lora_spec()
    target = settings.lora_dir / local_name
    if target.exists():
        return target

    from huggingface_hub import hf_hub_download

    settings.lora_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s/%s -> %s", repo, filename, target)
    got = hf_hub_download(repo_id=repo, filename=filename, token=settings.hf_token)
    target.write_bytes(Path(got).read_bytes())
    return target
