"""Model-resident few-step SDXL img2img pipeline.

Design notes
------------
* The checkpoint is an Illustrious/SDXL single-file ``.safetensors`` loaded once
  and kept on the GPU for the process lifetime. Nothing is re-loaded per request.
* A 4-step distillation LoRA (LCM by default, DMD2 as fallback) is *fused* into
  the UNet at load time, so there is zero per-step PEFT overhead.
* Prompt embeddings are cached by ``(prompt, negative, cfg)``. On an 8 GB card
  the two SDXL text encoders (~1.8 GB fp16) are parked in system RAM between
  requests and only paged in on a cache miss.
* The request mask is applied by compositing the model output over the input
  through the (already feathered) mask, which is what keeps the ComfyUI backend
  contract: white = regenerate, black = keep the human drawing untouched.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .config import Settings

log = logging.getLogger("stream_worker.pipeline")

# Multiple of 8 is a VAE requirement; SDXL is trained at ~1024 so anything above
# max_size is refused rather than silently producing mush (and OOMing at 8 GB).
SIZE_MULTIPLE = 8
MIN_SIZE = 256


def strip_data_url(data: str) -> str:
    if data.startswith("data:"):
        _, _, tail = data.partition(",")
        return tail
    return data


def decode_png_b64(data: str) -> Image.Image:
    raw = base64.b64decode(strip_data_url(data), validate=False)
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


def encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def round_size(value: int, max_size: int) -> int:
    v = int(value)
    if v < MIN_SIZE or v > max_size:
        raise ValueError(f"size {v} out of range [{MIN_SIZE}, {max_size}]")
    return v - (v % SIZE_MULTIPLE)


def composite_through_mask(base: Image.Image, generated: Image.Image, mask: Image.Image | None) -> Image.Image:
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

    diffusers' img2img keeps only the last ``num_inference_steps * strength``
    timesteps, so asking for 4 steps at strength 0.55 would really run 2. In
    this contract "steps" means denoising steps actually performed.
    """
    strength = max(0.05, min(1.0, strength))
    return max(steps, int(math.ceil(steps / strength)))


class CancelledError(Exception):
    """Raised inside the diffusion loop when a request is cancelled.

    HTTP abort alone cannot stop GPU work: the pipeline call is already running
    in a worker thread and will finish regardless, so an abandoned request keeps
    the GPU busy and the next one queues behind it. Cancellation therefore has
    to be cooperative - the caller sets a flag, and the step callback notices.
    """

    def __init__(self, request_id: str | None = None) -> None:
        super().__init__(f"generation cancelled ({request_id or 'unknown'})")
        self.request_id = request_id


@dataclass
class GenerateResult:
    image: Image.Image
    timings: dict[str, float]


class StreamPipeline:
    """Wraps ``StableDiffusionXLImg2ImgPipeline`` with warm-state extras."""

    backend = "diffusers-sdxl-lcm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipe: Any = None
        self.warm = False
        self.device = "cuda"
        self._embed_cache: OrderedDict[tuple[str, str, bool], tuple[Any, ...]] = OrderedDict()
        self._torch: Any = None
        self._vae_timings: dict[str, float] = {}

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        import torch
        from diffusers import LCMScheduler, StableDiffusionXLImg2ImgPipeline

        self._torch = torch
        s = self.settings
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; the stream worker needs a GPU")
        if not s.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {s.checkpoint}")

        t0 = time.perf_counter()
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            str(s.checkpoint),
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)

        lora = ensure_lora(s)
        pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="fast")
        # Fusing bakes the LoRA into the UNet weights: no per-step adapter math,
        # and no extra VRAM for a second copy of the deltas.
        pipe.fuse_lora()
        pipe.unload_lora_weights()

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
        self._park_text_encoders()
        log.info("loaded %s + %s in %.1fs", s.checkpoint.name, lora.name, time.perf_counter() - t0)

        if s.warmup_size:
            self.warmup(s.warmup_size)
        self.warm = True

    def _install_vae(self, pipe: Any) -> None:
        """Swap in a VAE that does not need the fp32 upcast.

        The checkpoint's SDXL VAE sets ``force_upcast=True``, so diffusers casts
        it to fp32 for every encode/decode and back again. That was measured as
        the dominant fixed cost per request (docs/STREAM_WORKER.md 4.4), so the
        default is the fp16-safe rescale of the same weights.
        """
        torch = self._torch
        repo, cls_name = self.settings.vae_spec()
        if repo is None:
            log.info("VAE: checkpoint's own (force_upcast=%s)", getattr(pipe.vae.config, "force_upcast", None))
            return

        import diffusers

        vae_cls = getattr(diffusers, cls_name)
        t0 = time.perf_counter()
        vae = vae_cls.from_pretrained(repo, torch_dtype=torch.float16)
        # Belt and braces: the replacement is fp16-safe by construction, but the
        # flag is what diffusers actually checks before upcasting.
        if hasattr(vae, "config") and hasattr(vae.config, "force_upcast"):
            vae.config.force_upcast = False
        pipe.vae = vae
        log.info("VAE: %s (%s) loaded in %.1fs", repo, cls_name, time.perf_counter() - t0)

    def _instrument_vae(self, pipe: Any) -> None:
        """Time VAE encode/decode separately from the UNet.

        Wraps the bound methods once at load time; ``generate`` resets the
        accumulator per request. The synchronize calls make the split honest at
        the cost of a couple of stalls per request.
        """
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
                    self._vae_timings[name] = self._vae_timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

            return timed

        vae.encode = wrap("vae_encode_ms", vae.encode)
        vae.decode = wrap("vae_decode_ms", vae.decode)
        vae._brushjam_timed = True

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
            steps=self.settings.default_steps,
            seed=0,
            width=size,
            height=size,
        )
        log.info("warmup at %d took %.2fs", size, time.perf_counter() - t0)

    # ------------------------------------------------------------- text cache
    def _park_text_encoders(self) -> None:
        if not self.settings.offload_text_encoders or self.pipe is None:
            return
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        self._torch.cuda.empty_cache()

    def _embeds(self, prompt: str, negative: str, cfg: bool) -> tuple[Any, ...]:
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
        mask: Image.Image | None,
        prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
        should_cancel: Callable[[], bool] | None = None,
        request_id: str | None = None,
    ) -> GenerateResult:
        torch = self._torch
        s = self.settings
        timings: dict[str, float] = {}
        self._vae_timings = {}

        def check_cancel() -> None:
            if should_cancel is not None and should_cancel():
                raise CancelledError(request_id)

        t_start = time.perf_counter()
        base = image.convert("RGB")
        if base.size != (width, height):
            base = base.resize((width, height), Image.LANCZOS)

        t = time.perf_counter()
        cfg_on = s.guidance > 1.0
        prompt_text = (prompt or "").strip() + s.quality_suffix
        cached = (prompt_text, negative_prompt or "", cfg_on) in self._embed_cache
        embeds = self._embeds(prompt_text, negative_prompt or "", cfg_on)
        timings["prompt_ms"] = (time.perf_counter() - t) * 1000.0
        timings["prompt_cached"] = 1.0 if cached else 0.0

        prompt_embeds, negative_embeds, pooled, negative_pooled = embeds
        generator = torch.Generator(device=self.device).manual_seed(int(seed) & 0x7FFFFFFF)
        check_cancel()

        # The only place inside pipe() where we get control back. Raising here
        # unwinds the pipeline call, so a cancelled job stops at the next step
        # boundary instead of running to completion on a GPU nobody is waiting
        # for. Steps are 0.1-1 s, which bounds the cancellation latency.
        steps_done = [0]

        def on_step_end(_pipe: Any, step: int, _timestep: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            steps_done[0] = step + 1
            check_cancel()
            return kwargs

        torch.cuda.synchronize()
        t = time.perf_counter()
        out = self.pipe(
            callback_on_step_end=on_step_end,
            image=base,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            pooled_prompt_embeds=pooled,
            negative_pooled_prompt_embeds=negative_pooled,
            strength=float(strength),
            num_inference_steps=steps_for_strength(steps, strength),
            guidance_scale=s.guidance,
            generator=generator,
            output_type="pil",
        )
        torch.cuda.synchronize()
        timings["diffusion_ms"] = (time.perf_counter() - t) * 1000.0
        timings["steps_run"] = float(steps_done[0])

        # Split the diffusion span. vae_* are measured directly by the wrappers
        # installed at load; unet_ms is the remainder, so it also carries
        # scheduler and conditioning overhead - it is a bound, not a pure UNet
        # number.
        timings["vae_encode_ms"] = round(self._vae_timings.get("vae_encode_ms", 0.0), 2)
        timings["vae_decode_ms"] = round(self._vae_timings.get("vae_decode_ms", 0.0), 2)
        timings["unet_ms"] = round(
            max(0.0, timings["diffusion_ms"] - timings["vae_encode_ms"] - timings["vae_decode_ms"]), 2
        )

        check_cancel()
        t = time.perf_counter()
        composed = composite_through_mask(base, out.images[0], mask)
        # The contract promises exactly width x height; composite_through_mask
        # resizes the model output to the base, so this can only fail if `base`
        # itself was wrong. Assert rather than return a differently-sized PNG
        # that the server would silently composite at the wrong scale.
        if composed.size != (width, height):
            raise RuntimeError(f"internal error: produced {composed.size}, expected {(width, height)}")
        timings["composite_ms"] = (time.perf_counter() - t) * 1000.0

        # Return the transient blocks to the driver. Without this the caching
        # allocator ends up reserving ~7.5 GB against 5.05 GB of live tensors on
        # an 8 GB card, the device reports 0 bytes free, and the next request's
        # activations spill into shared system memory (see docs/STREAM_WORKER.md
        # 4.2.1). Costs a few ms; buys the headroom back.
        if s.empty_cache_each_run:
            t = time.perf_counter()
            torch.cuda.empty_cache()
            timings["empty_cache_ms"] = (time.perf_counter() - t) * 1000.0

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        return GenerateResult(image=composed, timings=timings)

    def unload(self) -> None:
        """Release the model and its VRAM so another process can have the GPU.

        On 8 GB this worker and ComfyUI cannot both be resident, so the two hand
        the card over rather than share it. ``load()`` puts it back.
        """
        self.pipe = None
        self.warm = False
        self._embed_cache.clear()
        if self._torch is not None:
            import gc

            gc.collect()
            self._torch.cuda.empty_cache()
            self._torch.cuda.ipc_collect()
        log.info("unloaded: GPU released")

    def memory(self) -> dict[str, float]:
        """VRAM accounting, in GB. `allocated` is live tensors, `reserved` is
        what the caching allocator holds, `free`/`total` come from the driver.
        The gap between allocated and reserved is fragmentation the process is
        sitting on and could give back."""
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


def ensure_lora(settings: Settings) -> Path:
    """Return the local LoRA path, downloading it into the ComfyUI loras dir.

    The worker deliberately shares the ComfyUI ``loras`` directory so the same
    file backs both this worker and a 4-step ComfyUI workflow.
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
