# `@brushjam/tools`

Scripts that measure or exercise a **running** Brush Jam server over its public
HTTP and WebSocket surface. Nothing here imports a server's internals, so they
work against the Python server, and would work against anything else that
speaks the same protocol.

```bash
pnpm latency -- --url http://127.0.0.1:8787 --n 5 --profile fast --resolution 768
pnpm playtest-sim -- --url http://127.0.0.1:8787 --users 3 --minutes 1
pnpm smoke -- --url http://127.0.0.1:8787
```

| script | what it answers |
| --- | --- |
| `latency` | what one edit costs, end to end: `stroke_end` to pixels, the `ai_result` message, and the server's own pipeline time |
| `playtest-sim` | does a real session hold together - N users drawing for M minutes, then convergence, presence and error checks |
| `smoke` | is this server alive and does one stroke come back as a picture |

`--json <path>` on `latency` writes the samples and the settings they were
measured under.
