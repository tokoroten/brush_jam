"""HTTP client for the model-resident worker in apps/stream-worker.
Port of the retired Node server's src/ai/backends/stream.ts.

Kept after the pipeline moved in-process, because the worker is still the way
to run the model in a separate process (a different machine, or a GPU handover
where this server must not hold VRAM).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import httpx

from ...constants import QUALITY_SUFFIX
from .base import BackendCapabilities, BackendHttpError, GenerateRequest

log = logging.getLogger("brushjam.ai.stream")

#: Used when the worker does not report its own limits.
DEFAULT_STREAM_MAX_RESOLUTION = 1024
DEFAULT_STREAM_MAX_DENOISE = 0.9

_BASE64 = re.compile(r"^[A-Za-z0-9+/\s]+={0,2}$")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_size(data: bytes) -> Optional[Tuple[int, int]]:
    """Width/height straight out of the IHDR chunk; no decode needed."""
    if len(data) < 24 or data[:8] != _PNG_MAGIC or data[12:16] != b"IHDR":
        return None
    return struct.unpack_from(">I", data, 16)[0], struct.unpack_from(">I", data, 20)[0]


@dataclass
class StreamHealth:
    ok: bool = False
    #: The model is loaded and has run at least once.
    warm: bool = False
    #: Largest square the worker will accept, 0 when it does not say.
    max_size: int = 0
    busy: bool = False
    backend: str = ""
    current_request_id: Optional[str] = None
    max_denoise: Optional[float] = None
    negative_prompt_active: Optional[bool] = None
    #: Whatever it reports about how it samples: steps, guidance, vae, model.
    sampling: Dict[str, Any] = field(default_factory=dict)
    #: Why the worker is not usable, for the log line.
    reason: Optional[str] = None


def _strip_quality_suffix(prompt: str) -> str:
    """The worker adds QUALITY_SUFFIX itself; do not send it twice."""
    return prompt[: -len(QUALITY_SUFFIX)] if prompt.endswith(QUALITY_SUFFIX) else prompt


def _strip_data_url(value: str) -> str:
    if value.startswith("data:"):
        comma = value.find(",")
        if comma >= 0:
            return value[comma + 1 :]
    return value


class StreamBackend:
    name = "stream"

    def __init__(
        self,
        url: str,
        timeout_ms: float = 120_000,
        probe_timeout_ms: float = 1500,
        queue: bool = True,
        settle_timeout_ms: float = 15_000,
        transport: Any = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout_ms = timeout_ms
        self.probe_timeout_ms = probe_timeout_ms
        #: Wait for the GPU when the worker is busy instead of a 409: the
        #: scheduler already allows one in-flight request per room.
        self.queue = queue
        self.settle_timeout_ms = settle_timeout_ms
        #: Test seam: an httpx transport to use instead of the network.
        self._transport = transport
        #: Resolves once a cancelled generation has really stopped. Cancelling
        #: only asks the worker to stop at its next diffusion step, so a retry
        #: issued immediately would queue behind the job we just abandoned.
        self._settling: Optional[asyncio.Task] = None

    def _client(self, timeout_ms: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.url, timeout=timeout_ms / 1000, transport=self._transport
        )

    async def generate(self, req: GenerateRequest) -> bytes:
        if self._settling is not None:
            await asyncio.shield(self._settling)
        # Dropping the HTTP request does NOT stop the GPU: the worker is already
        # inside a diffusion loop on a background thread. The id lets us tell it
        # to stop at its next step.
        request_id = uuid.uuid4().hex
        body = {
            "image_b64": base64.b64encode(req.image_png).decode("ascii"),
            "mask_b64": base64.b64encode(req.mask_png).decode("ascii"),
            "prompt": _strip_quality_suffix(req.prompt),
            "negative_prompt": req.negative_prompt,
            "denoise": req.denoise,
            "steps": req.steps,
            "seed": req.seed,
            "width": req.size,
            "height": req.size,
            "request_id": request_id,
            "queue": self.queue,
        }
        try:
            async with self._client(self.timeout_ms) as client:
                res = await client.post("/generate", json=body)
        except (asyncio.CancelledError, httpx.HTTPError) as err:
            # Abandoned or timed out: free the GPU rather than leaving it
            # working on a result nobody will read.
            self._start_settling(request_id)
            if isinstance(err, asyncio.CancelledError):
                raise
            if isinstance(err, httpx.TimeoutException):
                raise RuntimeError("stream worker request timed out") from err
            raise RuntimeError(f"stream worker request failed: {err}") from err

        # 499 is the worker acknowledging our own cancellation.
        if res.status_code == 499:
            raise asyncio.CancelledError()
        if res.status_code == 409:
            raise BackendHttpError(f"stream worker is busy: {_safe_text(res)}", res.status_code)
        if res.status_code >= 400:
            raise BackendHttpError(
                f"stream worker /generate failed: {res.status_code} {_safe_text(res)}", res.status_code
            )
        try:
            payload = res.json()
        except Exception as err:  # noqa: BLE001
            raise RuntimeError(f"stream worker returned invalid JSON: {err}") from err
        return _decode_image(payload, req.size)

    def _start_settling(self, request_id: str) -> None:
        self._settling = asyncio.ensure_future(self._cancel_and_settle(request_id))

    async def _cancel_and_settle(self, request_id: str) -> None:
        """Cancel, then wait for `busy` to clear. Bounded: if the worker never
        reports itself idle we give up and let the next request queue."""
        try:
            await self._cancel(request_id)
            deadline = time.monotonic() + self.settle_timeout_ms / 1000
            while time.monotonic() < deadline:
                health = await self.health()
                # Not reachable, idle, or already on someone else's job.
                if not health.ok or not health.busy:
                    return
                if health.current_request_id and health.current_request_id != request_id:
                    return
                await asyncio.sleep(0.25)
        finally:
            self._settling = None

    async def _cancel(self, request_id: str) -> None:
        """Best effort "stop working on this": the caller has already given up."""
        try:
            async with self._client(5000) as client:
                await client.post("/cancel", json={"request_id": request_id})
        except Exception:  # noqa: BLE001
            pass

    async def health(self) -> StreamHealth:
        """Structured probe. Reachability alone is not enough: an unloaded or
        dry-run worker answers `ok` and then echoes the input, and one capped
        below the configured window answers `ok` and then 400s every request."""

        def dead(reason: str) -> StreamHealth:
            return StreamHealth(reason=reason)

        try:
            async with self._client(self.probe_timeout_ms) as client:
                res = await client.get("/healthz")
            if res.status_code >= 400:
                return dead(f"/healthz answered {res.status_code}")
            body = res.json()
        except Exception as err:  # noqa: BLE001
            return dead(str(err))
        if body.get("ok") is not True:
            return dead("/healthz reported not ok")

        def num(v: Any) -> Optional[float]:
            return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        def text(v: Any) -> Optional[str]:
            return v if isinstance(v, str) and v else None

        active = body.get("negative_prompt_active")
        return StreamHealth(
            ok=True,
            warm=body.get("warm") is True,
            max_size=int(num(body.get("max_size")) or 0),
            max_denoise=num(body.get("max_denoise")),
            negative_prompt_active=active if isinstance(active, bool) else None,
            busy=body.get("busy") is True,
            backend=text(body.get("backend")) or "stream",
            current_request_id=text(body.get("current_request_id")),
            sampling={
                "steps": num(body.get("steps")),
                "guidance": num(body.get("guidance")),
                "vae": text(body.get("vae")),
                "model": text(body.get("model")),
                "lora": text(body.get("lora")),
            },
        )

    async def capabilities(self) -> BackendCapabilities:
        """One fused few-step LoRA, so there is no quality profile to offer:
        asking for 14 steps would silently run 4."""
        health = await self.health()
        negative = True if health.negative_prompt_active is None else health.negative_prompt_active
        return BackendCapabilities(
            profiles=["fast"],
            max_resolution=health.max_size if health.max_size > 0 else DEFAULT_STREAM_MAX_RESOLUTION,
            # Above ~0.9 an LCM worker tends to ignore the drawing entirely.
            max_denoise=(
                health.max_denoise if health.max_denoise is not None else DEFAULT_STREAM_MAX_DENOISE
            ),
            negative_prompt_active={"fast": negative, "quality": negative},
        )

    async def healthy(self) -> bool:
        return (await self.health()).ok


def _decode_image(body: Dict[str, Any], size: int) -> bytes:
    """The worker's own output, validated: base64 silently accepts nearly
    anything, so a truncated or wrong-sized image would surface much later."""
    raw = body.get("image_b64")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("stream worker returned no image")
    payload = _strip_data_url(raw).strip()
    if not payload or not _BASE64.match(payload):
        raise RuntimeError("stream worker returned a malformed base64 image")
    try:
        data = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as err:
        raise RuntimeError("stream worker returned a malformed base64 image") from err
    if not data:
        raise RuntimeError("stream worker returned an empty image")
    dims = png_size(data)
    if dims is None:
        raise RuntimeError("stream worker returned something that is not a PNG")
    if dims != (size, size):
        raise RuntimeError(
            f"stream worker returned {dims[0]}x{dims[1]}, expected {size}x{size}"
        )
    return data


def _safe_text(res: httpx.Response) -> str:
    """FastAPI errors are a JSON `detail` field; show that, not the envelope."""
    try:
        parsed = res.json()
        detail = parsed.get("detail") if isinstance(parsed, dict) else None
        if isinstance(detail, str):
            return detail[:300]
        if detail is not None:
            return str(detail)[:300]
    except Exception:  # noqa: BLE001
        pass
    try:
        return res.text[:300]
    except Exception:  # noqa: BLE001
        return ""


async def stream_health(url: str, timeout_ms: float = 1500) -> StreamHealth:
    """Full capability probe used by auto-selection and the startup warning."""
    return await StreamBackend(url, probe_timeout_ms=timeout_ms).health()


async def stream_reachable(url: str, timeout_ms: float = 1500) -> bool:
    return await StreamBackend(url, probe_timeout_ms=timeout_ms).healthy()
