"""ComfyUI backend. Port of the retired Node server's src/ai/backends/comfyui.ts.

The workflow JSON is built here, with fixed node ids so the tests can assert on
it, and the RunPod adapter posts the identical graph to a serverless worker -
moving to the cloud is a transport change, not a pipeline change.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ...constants import MAX_AI_RESOLUTION, MAX_DENOISE, QUALITY_SUFFIX
from .base import BackendCapabilities, BackendHttpError, GenerateRequest

log = logging.getLogger("brushjam.ai.comfyui")

#: LCM wants cfg 1.0-2.0; the normal 5.5 destroys a 4-step result.
FAST_CFG = 1.5
#: DMD2 rather than LCM: measured faster and it reinterprets a drawing better at
#: the same denoise (docs/experiments/2026-09-05-stream/REPORT.md section 8).
DEFAULT_FAST_LORA = "dmd2_sdxl_4step_lora_fp16.safetensors"


@dataclass(frozen=True)
class FastProfile:
    """Few-step LoRAs are not interchangeable: DMD2 is distilled and wants cfg
    1.0 (no guidance at all), while LCM wants a little."""

    name: str
    cfg: float
    sampler: str
    scheduler: str


LCM_PROFILE = FastProfile("lcm", FAST_CFG, "lcm", "sgm_uniform")
DMD2_PROFILE = FastProfile("dmd2", 1.0, "lcm", "sgm_uniform")


def fast_profile(lora: Optional[str]) -> Optional[FastProfile]:
    if not lora:
        return None
    return DMD2_PROFILE if re.search(r"dmd2", lora, re.I) else LCM_PROFILE


def negative_prompt_active(profile: Optional[FastProfile], cfg: float) -> bool:
    """At CFG 1.0 the sampler never evaluates the negative branch, so the
    negative prompt is inert and the UI should say so."""
    return (profile.cfg if profile else cfg) > 1.0


def _decode_node(tile: Optional[int]) -> Dict[str, Any]:
    """VAEDecodeTiled on ComfyUI 0.28 requires all four size inputs."""
    samples = ["9", 0]
    vae = ["1", 2]
    if not tile or tile <= 0:
        return {"class_type": "VAEDecode", "inputs": {"samples": samples, "vae": vae}}
    return {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": samples,
            "vae": vae,
            "tile_size": tile,
            "overlap": 64,
            "temporal_size": 64,
            "temporal_overlap": 8,
        },
    }


def build_workflow(
    *,
    checkpoint: str,
    prompt: str,
    negative_prompt: str,
    image_name: str,
    mask_name: str,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    filename_prefix: str,
    fast_lora: Optional[str] = None,
    vae_tile: Optional[int] = None,
) -> Dict[str, Any]:
    """ComfyUI API-format workflow. SetLatentNoiseMask (not VAEEncodeForInpaint)
    keeps the human drawing as the img2img base."""
    profile = fast_profile(fast_lora)
    fast = profile is not None
    # With the LoRA loaded, MODEL and CLIP come from node 12 instead of the
    # checkpoint. VAE still comes from the checkpoint: LoraLoader has no VAE out.
    model = ["12", 0] if fast else ["1", 0]
    clip = ["12", 1] if fast else ["1", 1]

    workflow: Dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt + QUALITY_SUFFIX, "clip": clip}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": clip}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": mask_name, "upload": "image"}},
        "6": {"class_type": "ImageToMask", "inputs": {"image": ["5", 0], "channel": "red"}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["7", 0], "mask": ["6", 0]}},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": model,
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                # `steps` is the real sampler-step count at any denoise: ComfyUI
                # builds the longer schedule then keeps the last steps+1 sigmas.
                "steps": steps,
                "cfg": profile.cfg if profile else cfg,
                "sampler_name": profile.sampler if profile else "euler_ancestral",
                "scheduler": profile.scheduler if profile else "normal",
                "denoise": denoise,
            },
        },
        "10": _decode_node(vae_tile),
        "11": {
            "class_type": "SaveImage",
            "inputs": {"images": ["10", 0], "filename_prefix": filename_prefix},
        },
    }
    if fast:
        workflow["12"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": fast_lora,
                "strength_model": 1,
                "strength_clip": 1,
            },
        }
    return workflow


_TRANSIENT = re.compile(r"timed out|fetch failed|ECONNRESET|socket hang up|network|connect", re.I)


def _is_transient(err: Exception) -> bool:
    """A timed-out or dropped poll is worth retrying; a protocol error is not."""
    if isinstance(err, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return bool(_TRANSIENT.search(str(err)))


class ComfyUIBackend:
    name = "comfyui"

    def __init__(
        self,
        url: str,
        checkpoint: str,
        cfg: float = 5.5,
        fast_lora: Optional[str] = None,
        vae_tile: int = 512,
        poll_interval_ms: float = 250,
        timeout_ms: float = 180_000,
        request_timeout_ms: float = 30_000,
        transport: Any = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.checkpoint = checkpoint
        self.cfg = cfg
        self.fast_lora = fast_lora or None
        self.vae_tile = vae_tile
        self.poll_interval_ms = poll_interval_ms
        self.timeout_ms = timeout_ms
        self.request_timeout_ms = request_timeout_ms
        self.client_id = uuid.uuid4().hex
        #: Test seam: an httpx transport to use instead of the network.
        self._transport = transport
        #: None until the first request; then the LoRA the graph last used.
        self._last_lora: Any = _UNSET

    def _client(self, timeout_ms: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.url, timeout=timeout_ms / 1000, transport=self._transport
        )

    def identity(self, profile: str) -> Dict[str, str]:
        """The graph's own checkpoint, and the LoRA only the fast profile loads."""
        out: Dict[str, str] = {"model": self.checkpoint}
        if profile != "quality" and self.fast_lora:
            out["lora"] = self.fast_lora
        return out

    async def capabilities(self) -> BackendCapabilities:
        # Both profiles, unless there is no LoRA to build the fast graph from.
        return BackendCapabilities(
            profiles=["fast", "quality"] if self.fast_lora else ["quality"],
            max_resolution=MAX_AI_RESOLUTION,
            max_denoise=MAX_DENOISE,
            negative_prompt_active={
                "fast": negative_prompt_active(fast_profile(self.fast_lora), self.cfg),
                "quality": negative_prompt_active(None, self.cfg),
            },
        )

    def _lora_for(self, req: GenerateRequest) -> Optional[str]:
        """ComfyUI keeps one model in VRAM, so alternating profiles makes it
        load or unload the LoRA - worth saying out loud rather than looking like
        a random slow generation."""
        wanted = self.fast_lora if req.profile == "fast" else None
        if self._last_lora is not _UNSET and self._last_lora != wanted:
            to = f"fast ({wanted})" if wanted else "quality"
            log.info("switching profile to %s; the first generation after a switch reloads the model", to)
        self._last_lora = wanted
        return wanted

    async def generate(self, req: GenerateRequest) -> bytes:
        stamp = f"{req.tag}_{int(time.time() * 1000)}"
        async with self._client(self.request_timeout_ms) as client:
            image = await self._upload(client, req.image_png, f"brushjam_{stamp}_img.png")
            mask = await self._upload(client, req.mask_png, f"brushjam_{stamp}_mask.png")

            workflow = build_workflow(
                checkpoint=self.checkpoint,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                image_name=_joined(image),
                mask_name=_joined(mask),
                seed=req.seed,
                steps=req.steps,
                cfg=self.cfg,
                vae_tile=self.vae_tile,
                fast_lora=self._lora_for(req),
                denoise=req.denoise,
                filename_prefix=f"brushjam/{req.tag}",
            )
            queued = await client.post(
                "/prompt", json={"prompt": workflow, "client_id": self.client_id}
            )
            if queued.status_code >= 400:
                raise BackendHttpError(
                    f"ComfyUI /prompt failed: {queued.status_code} {_safe_text(queued)}",
                    queued.status_code,
                )
            prompt_id = queued.json().get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise RuntimeError("ComfyUI returned no prompt_id")

            # From here the job exists on the ComfyUI side: any failure must
            # cancel it, or the next retry queues a duplicate behind an orphan.
            try:
                out = await self._wait_for_output(client, prompt_id)
                view = await client.get(
                    "/view",
                    params={
                        "filename": out.get("filename", ""),
                        "subfolder": out.get("subfolder", "") or "",
                        "type": out.get("type", "output") or "output",
                    },
                )
                if view.status_code >= 400:
                    raise BackendHttpError(f"ComfyUI /view failed: {view.status_code}", view.status_code)
                return view.content
            except BaseException:
                await self._cancel_prompt(prompt_id)
                raise

    async def _upload(self, client: httpx.AsyncClient, data: bytes, filename: str) -> Dict[str, Any]:
        res = await client.post(
            "/upload/image",
            files={"image": (filename, data, "image/png")},
            data={"overwrite": "true", "type": "input"},
        )
        if res.status_code >= 400:
            raise BackendHttpError(
                f"ComfyUI upload failed: {res.status_code} {_safe_text(res)}", res.status_code
            )
        return res.json()

    async def _wait_for_output(self, client: httpx.AsyncClient, prompt_id: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self.timeout_ms / 1000
        last_poll_error: Optional[str] = None
        while True:
            # A single stalled or failed /history poll is transient: the job is
            # still running on the other side.
            try:
                res = await client.get(f"/history/{prompt_id}")
                if res.status_code < 400:
                    entry = res.json().get(prompt_id)
                    if entry:
                        if (entry.get("status") or {}).get("status_str") == "error":
                            raise RuntimeError("ComfyUI reported an execution error")
                        for node in (entry.get("outputs") or {}).values():
                            images = node.get("images") or []
                            if images:
                                return images[0]
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001
                if not _is_transient(err):
                    raise
                last_poll_error = str(err)
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"ComfyUI generation timed out (last poll: {last_poll_error})"
                    if last_poll_error
                    else "ComfyUI generation timed out"
                )
            await asyncio.sleep(self.poll_interval_ms / 1000)

    async def _cancel_prompt(self, prompt_id: str) -> None:
        """Interrupt it if it is the running job, delete it from the queue if it
        is only waiting. Never a blind POST /interrupt - that would kill another
        room's (or another app's) generation."""
        try:
            async with self._client(5000) as client:
                res = await client.get("/queue")
                if res.status_code >= 400:
                    return
                queue = res.json()

                def mentions(entries: Any) -> bool:
                    for entry in entries or []:
                        if isinstance(entry, list) and any(v == prompt_id for v in entry):
                            return True
                    return False

                if mentions(queue.get("queue_running")):
                    await client.post("/interrupt")
                elif mentions(queue.get("queue_pending")):
                    await client.post("/queue", json={"delete": [prompt_id]})
        except Exception:  # noqa: BLE001 - best effort; the result is dropped either way
            pass


class _Unset:
    pass


_UNSET = _Unset()


def _joined(upload: Dict[str, Any]) -> str:
    name = upload.get("name", "")
    subfolder = upload.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name


def _safe_text(res: httpx.Response) -> str:
    try:
        text = res.text[:1000]
    except Exception:  # noqa: BLE001
        return ""
    try:
        parsed = res.json()
        detail = parsed.get("detail") if isinstance(parsed, dict) else None
        if isinstance(detail, str):
            return detail[:300]
        if detail is not None:
            return str(detail)[:300]
    except Exception:  # noqa: BLE001
        pass
    return text[:300]


async def comfy_reachable(url: str, timeout_ms: float = 1500) -> bool:
    """Cheap reachability probe used to pick the default backend at boot."""
    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
            res = await client.get(f"{url.rstrip('/')}/system_stats")
            return res.status_code < 400
    except Exception:  # noqa: BLE001
        return False
