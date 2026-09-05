"""RunPod serverless backend. Port of the retired Node server's src/ai/backends/runpod.ts.

The same workflow JSON as the local ComfyUI backend, posted to RunPod's
`worker-comfyui` image, so moving to the cloud is a transport change only.

`/runsync` is not synchronous for long jobs: RunPod gives it about 90 s and
then answers `{id, status: "IN_PROGRESS"}`, leaving the caller to poll
`/status/{id}`. A cold start here is 80-120 s, so that path is the normal one
for the first generation, not an edge case.
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from typing import Any, Dict, Optional

import httpx

from ...constants import MAX_AI_RESOLUTION, MAX_DENOISE
from .base import BackendCapabilities, BackendHttpError, GenerateRequest
from .comfyui import build_workflow, fast_profile, negative_prompt_active

#: Terminal job states. Anything else means "still going".
DONE = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


class RunpodBackend:
    name = "runpod"

    def __init__(
        self,
        endpoint_id: str,
        api_key: str,
        checkpoint: str,
        cfg: float = 5.5,
        vae_tile: int = 512,
        fast_lora: Optional[str] = None,
        base_url: str = "https://api.runpod.ai/v2",
        timeout_ms: float = 300_000,
        poll_interval_ms: float = 1000,
        transport: Any = None,
    ) -> None:
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.checkpoint = checkpoint
        self.cfg = cfg
        self.vae_tile = vae_tile
        self.fast_lora = fast_lora or None
        self.base_url = base_url.rstrip("/")
        self.timeout_ms = timeout_ms
        self.poll_interval_ms = poll_interval_ms
        #: Test seam: an httpx transport to use instead of the network.
        self._transport = transport

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.endpoint_id}"

    @property
    def headers(self) -> Dict[str, str]:
        return {"content-type": "application/json", "authorization": f"Bearer {self.api_key}"}

    async def capabilities(self) -> BackendCapabilities:
        # Same workflow builder as ComfyUI, so the same two profiles.
        return BackendCapabilities(
            profiles=["fast", "quality"] if self.fast_lora else ["quality"],
            max_resolution=MAX_AI_RESOLUTION,
            max_denoise=MAX_DENOISE,
            negative_prompt_active={
                "fast": negative_prompt_active(fast_profile(self.fast_lora), self.cfg),
                "quality": negative_prompt_active(None, self.cfg),
            },
        )

    async def generate(self, req: GenerateRequest) -> bytes:
        stamp = f"{req.tag}_{int(time.time() * 1000)}"
        image_name = f"brushjam_{stamp}_img.png"
        mask_name = f"brushjam_{stamp}_mask.png"
        workflow = build_workflow(
            checkpoint=self.checkpoint,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image_name=image_name,
            mask_name=mask_name,
            seed=req.seed,
            steps=req.steps,
            cfg=self.cfg,
            vae_tile=self.vae_tile,
            fast_lora=self.fast_lora if req.profile == "fast" else None,
            denoise=req.denoise,
            filename_prefix=f"brushjam/{req.tag}",
        )
        deadline = time.monotonic() + self.timeout_ms / 1000
        async with httpx.AsyncClient(
            base_url=self.url, headers=self.headers, transport=self._transport
        ) as client:
            res = await self._call(
                client,
                "POST",
                "/runsync",
                deadline,
                json={
                    "input": {
                        "workflow": workflow,
                        "images": [
                            {"name": image_name, "image": base64.b64encode(req.image_png).decode("ascii")},
                            {"name": mask_name, "image": base64.b64encode(req.mask_png).decode("ascii")},
                        ],
                    }
                },
            )
            if res.status_code >= 400:
                raise BackendHttpError(
                    f"RunPod /runsync failed: {res.status_code} {res.text[:300]}", res.status_code
                )
            job = res.json()
            if job.get("status") not in DONE:
                job_id = job.get("id")
                if not job_id:
                    raise RuntimeError(
                        f"RunPod returned status {job.get('status') or 'unknown'} with no job id"
                    )
                try:
                    job = await self._poll(client, job_id, deadline)
                except BaseException:
                    await self._cancel(job_id)
                    raise
        return decode_image(job)

    async def _call(
        self, client: httpx.AsyncClient, method: str, path: str, deadline: float, **kwargs: Any
    ) -> httpx.Response:
        left = max(1.0, deadline - time.monotonic())
        try:
            return await client.request(method, path, timeout=left, **kwargs)
        except httpx.TimeoutException as err:
            raise RuntimeError(f"RunPod request timed out: {path}") from err

    async def _poll(self, client: httpx.AsyncClient, job_id: str, deadline: float) -> Dict[str, Any]:
        last_poll_error: Optional[str] = None
        while True:
            try:
                res = await self._call(client, "GET", f"/status/{job_id}", deadline)
                if res.status_code < 400:
                    job = res.json()
                    if job.get("status") in DONE:
                        return job
                    last_poll_error = None
                elif res.status_code >= 500 or res.status_code == 429:
                    # The job outlives a blip on the control plane; keep polling.
                    last_poll_error = f"status {res.status_code}"
                else:
                    raise BackendHttpError(f"RunPod /status failed: {res.status_code}", res.status_code)
            except (asyncio.CancelledError, BackendHttpError):
                raise
            except Exception as err:  # noqa: BLE001
                last_poll_error = str(err)
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"RunPod generation timed out (last poll: {last_poll_error})"
                    if last_poll_error
                    else "RunPod generation timed out"
                )
            await asyncio.sleep(self.poll_interval_ms / 1000)

    async def _cancel(self, job_id: str) -> None:
        """Best effort: a job left running would block the endpoint's only worker."""
        try:
            async with httpx.AsyncClient(
                base_url=self.url, headers=self.headers, timeout=5, transport=self._transport
            ) as client:
                await client.post(f"/cancel/{job_id}")
        except Exception:  # noqa: BLE001
            pass


_DATA_URL = re.compile(r"^data:image/\w+;base64,")


def decode_image(job: Dict[str, Any]) -> bytes:
    """Pull the PNG out of a finished job, with the exact error messages the
    server would print."""
    status = job.get("status")
    if status and status != "COMPLETED":
        detail = job.get("error") or "; ".join((job.get("output") or {}).get("errors") or [])
        raise RuntimeError(f"RunPod job {status}{': ' + detail if detail else ''}")
    if job.get("error"):
        raise RuntimeError(f"RunPod error: {job['error']}")
    images = (job.get("output") or {}).get("images") or []
    if not images:
        detail = "; ".join((job.get("output") or {}).get("errors") or [])
        raise RuntimeError(f"RunPod returned no image: {detail}" if detail else "RunPod returned no image")
    first = images[0]
    if first.get("type") == "s3_url":
        raise RuntimeError(
            "RunPod worker is configured for S3 upload; this backend expects base64 images"
        )
    if not first.get("data"):
        raise RuntimeError("RunPod returned an image entry with no data")
    return base64.b64decode(_DATA_URL.sub("", first["data"]))
