"""Full-canvas AI scheduler. Port of apps/server/src/ai/scheduler.ts.

Debounces drawing activity into at most one in-flight generation per room and
applies only results that are still useful. Human drawing never waits on any of
this: the render and the backend call both leave the event loop.

Patch mode (dirty regions, crop selection, feathered masks) is deliberately not
ported - see docs/PYTHON_SERVER_PLAN.md section 0. The Node implementation
stays in git history.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

from .constants import CANVAS_SIZE, DEFAULT_NEGATIVE_PROMPT
from .ai.backends import AIBackend, BackendHttpError, GenerateRequest
from .geometry import Rect
from .protocol import Message

log = logging.getLogger("brushjam.ai")

_TRANSIENT = re.compile(
    r"timed out|timeout|abort|econnrefused|econnreset|enotfound|socket hang up|"
    r"fetch failed|not answering|no answer|not warm",
    re.I,
)
_STATUS_5XX = re.compile(r"(?:^|\s)(?:status|failed:?|code)\s*5\d\d\b", re.I)
_STATUS_4XX = re.compile(r"(?:^|\s)(?:status|failed:?|code)\s*4\d\d\b", re.I)
_REFUSAL = re.compile(
    r"out of range|too large|too small|max_size|not supported|unsupported|malformed|invalid", re.I
)


def is_permanent_error(err: Any) -> bool:
    """A refusal, not a failure: the backend understood the request and said no,
    so repeating it unchanged cannot work."""
    if isinstance(err, BackendHttpError):
        return 400 <= err.status < 500
    message = str(err)
    if _TRANSIENT.search(message):
        return False
    if _STATUS_5XX.search(message):
        return False
    if _STATUS_4XX.search(message):
        return True
    return bool(_REFUSAL.search(message))


@dataclass
class RenderJob:
    """An immutable render job captured synchronously from the room."""

    revision: int
    prompt: str
    denoise: Optional[float]
    negative_prompt: Optional[str]
    resolution: Optional[int]
    profile: Optional[str]
    #: Blocking; the scheduler runs it off the event loop.
    render: Callable[[Rect, int], bytes]


class SchedulerHost(Protocol):
    def begin_job(self) -> RenderJob:
        """MUST capture room state synchronously - no awaits before the copy."""

    def get_revision(self) -> int:
        ...

    def build_full_mask(self, size: int) -> Any:
        ...

    async def apply_result(
        self, patch: bytes, crop: Rect, apply: Rect, mask: Any, for_revision: int
    ) -> Dict[str, Any]:
        ...

    def emit(self, msg: Message) -> None:
        ...

    def on_error(self, message: str, repeated: int) -> None:
        ...


@dataclass
class SchedulerOptions:
    window: int
    apply: int
    steps: int
    denoise: float
    debounce_ms: float
    canvas_size: int = CANVAS_SIZE
    fast_steps: Optional[int] = None
    #: Identical consecutive refusals after which it stops retrying by itself.
    max_repeated_errors: int = 2
    error_backoff_ms: float = 2000
    #: Hard cap on a single generation before it is abandoned.
    watchdog_ms: float = 180_000
    tag: str = "room"
    seed: Optional[Callable[[], int]] = None


def _now_ms() -> float:
    # `perf_counter`, not `monotonic`: both are monotonic, but on Windows
    # `monotonic` ticks at ~15.6 ms, which is the same order as the stages this
    # measures and quantises every reported `latencyMs`.
    return time.perf_counter() * 1000


class AIScheduler:
    def __init__(self, host: SchedulerHost, backend: AIBackend, opts: SchedulerOptions) -> None:
        self.host = host
        self.backend = backend
        self.opts = opts
        self._state: str = "idle"
        self._in_flight = False
        self._pending = False
        self._last_accepted = 0
        self._stopped = False
        #: Full-canvas mode: anything at all changed since the last generation.
        self._changed = False
        #: Bumped on every prompt change so an in-flight run notices it is stale.
        self._prompt_epoch = 0
        #: Absolute debounce deadline; scheduling can move it later, never earlier.
        self._not_before = 0.0
        self._backoff_until = 0.0
        self._timer: Optional[asyncio.Task] = None
        self._run_task: Optional[asyncio.Task] = None
        self._stuck_on: Optional[str] = None
        self._last_error: Optional[str] = None
        self._repeated_error = 0

    @property
    def state(self) -> str:
        return self._state

    # -- triggers ---------------------------------------------------------

    def mark_dirty(self, rects: List[Rect]) -> None:
        if self._stopped or not rects:
            return
        # A new edit is a different request; give it a chance.
        self._clear_stuck()
        # No regions, no crop selection: the whole canvas is the unit of work.
        self._changed = True
        self._set_state("queued")
        self._schedule(self.opts.debounce_ms)

    def nudge(self) -> None:
        """A prompt or settings change re-runs the AI."""
        if self._stopped:
            return
        self._clear_stuck()
        # Bumped even while a run is in flight: that run captured the old prompt.
        self._prompt_epoch += 1
        self._changed = True
        self._set_state("queued")
        self._schedule(self.opts.debounce_ms)

    def retry_now(self) -> None:
        """The backend became usable again: forget the error state and run what
        is still owed."""
        self._clear_stuck()
        # Both floors have to go: the failed run pushed the debounce out too.
        self._backoff_until = 0.0
        self._not_before = 0.0
        if self._changed or self._pending:
            self._schedule(0)

    def stop(self) -> None:
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._run_task is not None:
            self._run_task.cancel()

    # -- internals --------------------------------------------------------

    def _set_state(self, state: str, message: Optional[str] = None, latency_ms: Optional[float] = None) -> None:
        self._state = state
        msg: Message = {"t": "ai_status", "state": state, "forRevision": self.host.get_revision()}
        if message is not None:
            msg["message"] = message
        if latency_ms is not None:
            msg["latencyMs"] = latency_ms
        self.host.emit(msg)

    def _schedule(self, ms: float) -> None:
        """Scheduling never *shortens* a wait: the debounce deadline and the
        error backoff are both absolute."""
        if self._stopped:
            return
        now = _now_ms()
        target = max(now + ms, self._not_before, self._backoff_until)
        if ms > 0:
            self._not_before = max(self._not_before, now + ms)
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.ensure_future(self._wait_then_tick(max(0.0, target - now)))

    async def _wait_then_tick(self, ms: float) -> None:
        try:
            await asyncio.sleep(ms / 1000)
        except asyncio.CancelledError:
            return
        self._timer = None
        await self._tick()

    async def _tick(self) -> None:
        if self._stopped:
            return
        if self._in_flight:
            self._pending = True
            return
        self._run_task = asyncio.ensure_future(self._run_full())
        try:
            await self._run_task
        except asyncio.CancelledError:
            self._in_flight = False

    async def _run_full(self) -> None:
        """Render everything, regenerate everything, replace everything. One
        request in flight, latest revision wins, and a change that arrives
        mid-flight simply queues another whole-canvas run."""
        if not self._changed:
            self._set_state("idle")
            return
        size = self.opts.canvas_size
        rect: Rect = {"x": 0, "y": 0, "width": size, "height": size}
        job = self.host.begin_job()
        for_revision = job.revision
        # The whole canvas is rendered, then resampled to the generation size;
        # the result is scaled back to the canvas when it is composited.
        resolution = job.resolution or self.opts.window
        t_mask = _now_ms()
        mask = self.host.build_full_mask(resolution)
        mask_ms = _now_ms() - t_mask

        self._changed = False
        self._in_flight = True
        self._pending = False
        prompt_epoch = self._prompt_epoch
        self._set_state("generating")
        started_at = _now_ms()
        timed_out = False

        try:
            negative = job.negative_prompt or ""
            request_seed = (self.opts.seed or _default_seed)()
            t_render = _now_ms()
            image_png = await asyncio.to_thread(job.render, rect, resolution)
            render_ms = _now_ms() - t_render
            req = GenerateRequest(
                profile=job.profile or "quality",
                prompt=job.prompt,
                # An empty room setting means "keep the built-in list".
                negative_prompt=negative if negative.strip() else DEFAULT_NEGATIVE_PROMPT,
                image_png=image_png,
                mask_png=mask.png,
                size=resolution,
                denoise=job.denoise if job.denoise is not None else self.opts.denoise,
                steps=self._steps_for(job.profile),
                seed=request_seed,
                tag=f"{self.opts.tag}_r{for_revision}",
            )
            t_backend = _now_ms()
            try:
                patch = await asyncio.wait_for(
                    self.backend.generate(req), self.opts.watchdog_ms / 1000
                )
            except asyncio.TimeoutError:
                timed_out = True
                raise

            if for_revision < self._last_accepted:
                self._set_state("idle")
            else:
                backend_ms = _now_ms() - t_backend
                t_apply = _now_ms()
                applied = await self.host.apply_result(
                    patch, rect, rect, mask.alpha, for_revision
                )
                apply_ms = _now_ms() - t_apply
                self._last_accepted = for_revision
                latency_ms = round(_now_ms() - started_at)
                if log.isEnabledFor(logging.DEBUG):
                    # Where a single edit's wait actually went. `backend` is the
                    # whole call including any queueing; `apply` is the decode,
                    # upsample, composite and the PNG the client downloads.
                    log.debug(
                        "run r%d %dpx: mask %.0f render %.0f backend %.0f apply %.0f "
                        "= %d ms (%d KiB in, %d KiB out)",
                        for_revision,
                        resolution,
                        mask_ms,
                        render_ms,
                        backend_ms,
                        apply_ms,
                        latency_ms,
                        len(image_png) // 1024,
                        len(patch) // 1024,
                    )
                self.host.emit(
                    {
                        "t": "ai_result",
                        "rect": applied["rect"],
                        "url": applied["url"],
                        "aiRevision": for_revision,
                        "aiGeneration": applied["aiGeneration"],
                        "crop": rect,
                        "apply": rect,
                        "latencyMs": latency_ms,
                        # The profile this run used, not the room's current one.
                        "profile": job.profile or "quality",
                    }
                )
                if prompt_epoch != self._prompt_epoch:
                    # The prompt changed while this was generating.
                    self._changed = True
                self._set_state("idle", None, latency_ms)
            self._after_run(0)
        except asyncio.CancelledError:
            self._in_flight = False
            raise
        except Exception as err:  # noqa: BLE001 - every failure is reportable
            if self._stopped:
                self._in_flight = False
                return
            # the work was not done, so it is still owed
            self._changed = True
            message = "generation timed out" if timed_out else str(err) or err.__class__.__name__
            self._note_error(message, None if timed_out else err)
            if self._stuck_on is not None:
                # A request the backend refuses outright will be refused again.
                self._set_state("error", f"{message} - not retrying until something changes")
                self._in_flight = False
                return
            self._set_state("error", message)
            self._backoff_until = _now_ms() + self.opts.error_backoff_ms
            self._after_run(self.opts.error_backoff_ms)

    def _steps_for(self, profile: Optional[str]) -> int:
        """The fast profile is only fast because it runs fewer steps."""
        if profile == "fast":
            return self.opts.fast_steps if self.opts.fast_steps is not None else self.opts.steps
        return self.opts.steps

    def _note_error(self, message: str, err: Any) -> None:
        self._repeated_error = self._repeated_error + 1 if self._last_error == message else 1
        self._last_error = message
        # Only a request the backend REFUSED is worth giving up on: a worker
        # that is down answers with the same connection error every time.
        if self._repeated_error >= self.opts.max_repeated_errors and is_permanent_error(
            err if err is not None else message
        ):
            self._stuck_on = message
        self.host.on_error(message, self._repeated_error)

    def _clear_stuck(self) -> None:
        self._stuck_on = None
        self._repeated_error = 0
        self._last_error = None

    def _after_run(self, delay_ms: float) -> None:
        self._in_flight = False
        if self._stopped:
            return
        if self._pending or self._changed:
            self._schedule(delay_ms)


def _default_seed() -> int:
    return random.randrange(2**31)
