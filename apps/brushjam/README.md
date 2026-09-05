# brushjam

One Python process: the built web client, the room protocol (HTTP + WebSocket)
and inference. A port of `apps/server` (Node) plus `apps/stream-worker`, keeping
the wire protocol in `packages/shared/src/protocol.ts` byte-for-byte identical.

```bash
uv sync --extra dev
uv run pytest
uv run brushjam            # http://127.0.0.1:8787
```

See `docs/PYTHON_SERVER_PLAN.md`.
