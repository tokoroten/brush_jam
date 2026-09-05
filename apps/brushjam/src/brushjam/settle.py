"""Waiting out work that cannot be cancelled.

Dropping the caller does not stop a thread, a PNG encoder or a GPU. Anywhere
the scheduler's admission slot, or a model's loaded flag, depends on the
physical work having *stopped* rather than on the await having returned, the
await has to survive its own cancellation.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable


async def settle_future(future: "asyncio.Future[Any]") -> Any:
    """Wait for `future`, and keep waiting through repeated cancellation.

    A single `shield` is not enough: whoever cancelled once - a watchdog, then
    a shutdown - can cancel again, and the second one would abandon the wait
    and let the caller return while the work is still running. This loops
    until the future is genuinely done, then re-raises the cancellation it
    absorbed so the caller still learns it was cancelled.

    Only for work with a bounded end. The watchdog that cancelled has already
    given up on the result; this is about not lying to whoever comes next.
    """
    cancelled: BaseException | None = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as err:
            cancelled = err
        except BaseException:
            # The work failed; that is the future's business, not ours.
            break
    if cancelled is not None:
        raise cancelled
    return future.result()


async def settled(awaitable: Awaitable[Any]) -> Any:
    """`settle_future` for something that is not a future yet."""
    return await settle_future(asyncio.ensure_future(awaitable))
