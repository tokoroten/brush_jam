# Retiring the Node server

The Python server in `apps/brushjam` has replaced `apps/server`. Six rounds of
adversarial review ended in GO (see
[`PYTHON_SERVER.md`](PYTHON_SERVER.md#review)), the protocol is byte-identical,
and the tooling that had to survive has already been moved.

**Nothing here has been done.** This is the list for one reviewable deletion
commit, for the user to approve first.

## What has already been prepared

- `tools/` (`@brushjam/tools`) holds `latency`, `playtest-sim` and `smoke`.
  They speak only the public HTTP and WebSocket surface, import no server
  internals, and are reachable as `pnpm latency`, `pnpm playtest-sim`,
  `pnpm smoke`.
- Root scripts `py:serve`, `py:test`, `dev:py`, `dev:py:lan`, `build:py` exist
  beside the Node ones.
- The README's quick start and playtest sections describe the Python path; the
  Node instructions are under **Legacy: the Node server and the stream worker**.
- The Vite dev proxy already targets `:8787`, so it needs no change.

## Delete

| path | why |
| --- | --- |
| `apps/server/src/**` | the room server the Python one replaces |
| `apps/server/test/**` | 530 tests of that implementation |
| `apps/server/build.mjs`, `tsconfig.json`, `package.json` | its build |
| `apps/server/scripts/export-fixtures.ts` | see **The fixtures** below |
| `apps/server/scripts/quality-grid.ts` | drives backends directly, not over HTTP - see **What is lost** |
| `apps/server/scripts/runpod-smoke.ts` | same: constructs `RunpodBackend` itself |

## Change

| file | change |
| --- | --- |
| `package.json` | drop `dev`, `dev:stream`, `dev:lan`, `start`, `export-fixtures`; rename `dev:py` -> `dev`, `dev:py:lan` -> `dev:lan`, `build:py` -> `build`, `py:serve` -> `start`. Keep `test` and `typecheck` (they become shared + web + tools). |
| `pnpm-workspace.yaml` | nothing - `apps/*` stops matching a directory that is gone |
| `README.md` | delete the **Legacy: the Node server and the stream worker** subsection; delete the **Denoise / quality grid** section (or reword it as removed); drop `apps/server/` from the Layout tree |
| `docs/PYTHON_SERVER.md` | replace **Regenerating the parity fixtures** with a note that they are frozen |
| `docs/MVP_PLAN.md` | §§ referring to `apps/server/src/...` become historical - add a header note rather than rewriting a plan document |
| `docs/RUNPOD.md` | `pnpm --filter @brushjam/server dev` -> `AI_BACKEND=runpod pnpm start`; the `runpod-smoke` and `quality-grid` lines go |
| `docs/experiments/README.md` | the `quality-grid` commands go; the existing reports stay as records |
| `docs/STREAM_WORKER.md` | `apps/server/src/ai/backends/stream.ts` -> `apps/brushjam/src/brushjam/ai/backends/stream.py` |
| `.env.example` | `AI_BACKEND` values become `auto \| inproc \| stream \| comfyui \| runpod \| mock`; drop the `pnpm dev:stream` sentence |
| `apps/stream-worker/README.md` | the stream.ts reference |

No CI workflows and no Dockerfiles exist in this repository, so there is
nothing to change there.

## Environment variables

Every variable the Node server read is also read by the Python server, with two
exceptions and one addition:

- **`AI_MODE=patch` is gone.** The Python server implements full-canvas mode
  only and refuses to boot on `patch` with a message. Anyone with
  `AI_MODE=patch` in a `.env` must remove it.
- **`AI_CFG` and `AI_VAE_TILE` are parsed and validated but only reach the
  ComfyUI backend**, as before. They are not dead, but they do nothing for
  `inproc`.
- New: `INPROC_*` (checkpoint, LoRA, VAE, warmup, dry run), and the capacity
  limits `ROOM_CREATE_PER_MIN`, `UNJOINED_ROOM_TTL_MS`, `MAX_ROOM_SOCKETS`,
  `MAX_TOTAL_SOCKETS`, `MAX_ROOM_POINTS`, `MAX_ROOM_SNAPSHOT_BYTES`. Each
  falls back to the `STREAM_*` name where the worker had one, so an existing
  `.env` keeps working.

## The fixtures

`apps/server/scripts/export-fixtures.ts` is the one script that **cannot** be
moved, because its dependency is the reference implementation itself: it runs
the real Node `validateClientMessage`, `applyClientMessage`, `snapshot`,
`fnv1a`/`noiseRGB` and `renderStrokes`.

Porting the Node reducer into `packages/shared` to keep it regenerable would
resurrect the thing being retired, as a library, forever. So:

- The generated JSON in `apps/brushjam/fixtures/` is committed and stays.
  `tests/test_fixtures.py` replays all of it on every run - 45 protocol
  samples, 27 noise vectors, a 27-step reducer trace with the full snapshot
  after every step, and 256 noise-placement cases at fractional offsets.
- After the deletion they stop being a *parity* oracle and become a
  *regression* record: they pin the Python server to the behaviour it was
  built to match. That is the useful half, and it is the half that survives.
- `docs/PYTHON_SERVER.md` should say this where it currently documents the
  regeneration command.

## What is lost

- **`quality-grid`** - the denoise sweep that produced
  `docs/experiments/*/grid.png`. It calls `generate()` directly with a fixed
  seed and a full-white mask, so it cannot be a client of a running server
  without a new endpoint. The existing reports stay. If it is wanted again,
  the honest form is a small Python script in `apps/brushjam/scripts/` that
  imports the backend the same way - roughly a day, not a blocker.
- **`runpod-smoke`** - one live generation straight through `RunpodBackend`.
  `pnpm smoke -- --url ...` against a server running `AI_BACKEND=runpod`
  covers the same ground through the room protocol.
- **530 TypeScript tests.** They tested an implementation that no longer runs.
  Their content is not lost: the Python suite (278 tests) covers the same
  reducer, validator, scheduler, raster and backend behaviour, and the
  cross-language fixtures are what proves the two agreed.

## `apps/stream-worker`: keep

**Recommendation: keep it, and do not let it drift.**

It is not redundant. The Python server's `stream` backend talks to it over
HTTP, which is how a model on *another machine* is reached - a second box, a
rented GPU, a laptop that cannot hold SDXL. The in-process backend cannot do
that by construction.

But its pipeline is now duplicated. `apps/stream-worker/src/stream_worker/pipeline.py`
(553 lines) and `apps/brushjam/src/brushjam/ai/pipeline.py` (823 lines) are the
same design, and the second is strictly ahead: it fuses and unfuses the LoRA per
profile instead of fusing once at load, so it serves `quality` as well as
`fast`; it carries the stage timings; it settles cancelled work before letting
go of the device. The worker has none of that.

Two ways to stop the drift, in order of preference:

1. **Make the worker import the pipeline from `apps/brushjam`.** One
   implementation, one place to fix a VRAM bug. It needs `apps/brushjam` to be
   installable as a dependency of `apps/stream-worker` (a path dependency in
   its `pyproject.toml`), which is a small change, and the worker keeps its own
   HTTP app, config and queue. This also gives the remote worker a `quality`
   profile for free, which it has never had.
2. Leave it, and accept that the worker is the older model host. Acceptable
   only while nobody is deploying it.

Doing nothing is the option that ages badly: the next VRAM fix will land in one
file and not the other, and the difference will surface as a machine-specific
bug months later.

## Order

1. Get the user's approval for this list.
2. Delete and change everything above in **one** commit,
   `retire the Node server`.
3. `pnpm install` (the lockfile loses `@brushjam/server`), then
   `pnpm -r typecheck`, `pnpm -r test`, `pnpm py:test`, `pnpm build`.
4. Start the server and run `pnpm smoke`, `pnpm latency -- --n 5` and
   `pnpm playtest-sim -- --users 3 --minutes 1` against it. Those three are the
   acceptance test for the deletion: if they pass, nothing anybody used is
   gone.
