# Brush Jam on a RunPod GPU Pod

Runs the single-process Python server (`apps/brushjam`, `AI_BACKEND=inproc`) on
one RTX 4090 so friends can join from anywhere. Everything lives in
`deploy/runpod/`.

A **Pod**, not Serverless: Serverless has no usable WebSocket, and Brush Jam is
nothing but a WebSocket. The Pod HTTP proxy does carry WS
(`wss://{podId}-8787.proxy.runpod.net/ws`, verified from Tokyo, ~230 ms to
open).

## How the code gets in

No git remote, no registry, no SSH. Instead:

1. The pod runs the stock image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
2. Its `dockerStartCmd` is `deploy/runpod/start.sh` with `receiver.py` inlined
   (a stdlib HTTP server on **:8788**), followed by a boot loop.
3. `deploy.py` tars `apps/brushjam` + `bootstrap.sh` (~1 MB) and `PUT`s it to
   `https://{podId}-8788.proxy.runpod.net/upload?token=…`.
4. The boot loop extracts it and runs `bootstrap.sh`, which installs `uv`,
   `uv sync --extra inproc`, downloads the checkpoint, and execs `uv run brushjam`
   on **:8787**.

Uploading again replaces the code and restarts the server: the receiver SIGTERMs
the running server (by the pid `bootstrap.sh` recorded) and the loop re-extracts.

That signal is a shortcut, not the mechanism. The boot loop runs the server as
a **child** and watches two things at once - the child exiting, and
`/workspace/app.tgz` appearing - so an upload is installed even when the signal
reaches nobody: it lands in the window before `bootstrap.sh` has published its
pid, or the pid on disk is a process that no longer exists. When the loop sees
a pending tarball it terminates the child, waits for it (`BOOT_STOP_SECONDS`,
30 by default) and only then extracts, so the replacement never comes up beside
a server still holding the GPU and port 8787. Before this, such an upload was
acknowledged by the receiver and then waited for forever while the old code
kept running.

## What an upload costs

**About three minutes**, nearly all of it the model loading back onto the GPU.
An upload replaces `/workspace/app` wholesale, so nothing that lives inside it
survives - which is why the Python environment does not live there:
`bootstrap.sh` exports `UV_PROJECT_ENVIRONMENT=/workspace/venv`. With uv's
default (`.venv` inside the project, and so inside the swap) every upload paid
for a full torch reinstall, several minutes, for a change that should only
restart the server. `uv sync` still runs and is a no-op unless `uv.lock` moved.

A **new pod** is 10-15 minutes and re-downloads everything unless the volume is
reused. Two changes need one even so, because they are written into the pod by
`dockerStartCmd` and not by the tarball: `receiver.py` and `start.sh`.

## Prerequisites

- Repo-root `.env` with `RUNPOD_API_KEY`, `HF_TOKEN`, `CIVITAI_TOKEN`. They are
  read with python-dotenv, passed to the pod as env vars, and never printed or
  logged.
- The web client built into the package: `pnpm build`. The pod has no Node,
  so an unbuilt client is a 404 in a friend's browser. `deploy.py` refuses to
  build a tarball without `src/brushjam/static/index.html`.

## Commands

```bash
uv run --project apps/brushjam python deploy/runpod/deploy.py deploy      # create + upload
uv run --project apps/brushjam python deploy/runpod/deploy.py status      # pod, boot phase, /healthz
uv run --project apps/brushjam python deploy/runpod/deploy.py log -f      # follow /workspace/boot.log
uv run --project apps/brushjam python deploy/runpod/deploy.py upload      # ship new code, restart
uv run --project apps/brushjam python deploy/runpod/deploy.py watch       # stop it once nobody is using it
uv run --project apps/brushjam python deploy/runpod/deploy.py stop        # keep the volume, stop paying for the GPU
uv run --project apps/brushjam python deploy/runpod/deploy.py start       # bring it back (models still there)
uv run --project apps/brushjam python deploy/runpod/deploy.py terminate --yes   # destroy pod AND volume
```

`deploy` polls `/status` after the upload and prints each boot phase until the
server answers `/healthz`, so `log -f` is only needed for detail. It writes
`deploy/runpod/.pod` (pod id + upload token, gitignored); every
other command reads it. Local checks, no pod needed:
`python deploy/runpod/test_deploy.py`.

## Leaving it running by accident

A 4090 bills by the hour whether or not anybody is drawing, and the way this
costs money is not a session that runs long - it is the pod nobody remembered
to stop after everyone went to bed.

```bash
uv run --project apps/brushjam python deploy/runpod/deploy.py watch --idle-minutes 30
```

It polls `/healthz` every 60 seconds and stops the pod - `stop`, not
`terminate`, so the volume and the models survive - once nothing has happened
for that long, printing why. "Something happening" is either of the two numbers
the server reports:

| field | means |
|---|---|
| `active_sockets` | people in rooms. A room left open with nobody drawing still counts: somebody is sitting there. |
| `last_generation_at` | when the model last produced anything, so a room driven by `tools/` over HTTP counts too. |

The deadline is the later of those two plus the window: a generation five
minutes ago means twenty-five minutes left, not "busy for thirty minutes and
then thirty more" - which is what the first version did, stopping an idle pod
at sixty minutes on a `--idle-minutes 30` watch. The same rule means a watch
started beside a pod that has been idle for hours stops it at the first poll.

A pod that stops answering `/healthz` for the whole window is stopped as well:
that is the GPU burning money with nothing to show for it. A server too old to
report either field is `unknown` and is never stopped - a watcher that cannot
see the players must not turn the lights off.

It prints only when the state changes, `--terminate` destroys the pod instead
of stopping it, and **Ctrl+C leaves the pod exactly as it found it**. Run it in
a second terminal beside `log -f`; it needs nothing but the `.pod` file and the
API key.

## The pod

| | |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 4090, SECURE cloud |
| Volume | 40 GB at `/workspace` (survives stop/start) |
| Container disk | 20 GB |
| Ports | `8787/http` (the game), `8788/http` (the receiver) |
| Env | `AI_BACKEND=inproc`, `HOST=0.0.0.0`, `PORT=8787`, `HF_HOME=/workspace/hf`, `INPROC_CHECKPOINT=/workspace/models/checkpoints/sdxl-checkpoint.safetensors`, `INPROC_LORA_DIR=/workspace/models/loras`, `ROOM_CREATE_PER_MIN=60`, plus the three secrets |

**`ROOM_CREATE_PER_MIN=60` is deliberate.** The RunPod proxy terminates TLS, so
every player arrives from one client IP and the per-IP room-creation limit
applies to the whole group rather than to each person. 10/min (the default)
would lock everyone out after the first few rooms.

## First boot

~10–15 minutes, almost all of it downloads, and all of it visible in `log -f`:

| phase | what it is |
|---|---|
| `waiting-for-upload` | receiver up, no tarball yet |
| `extract` | tarball landed |
| `uv` | installing uv |
| `deps` | `uv sync --extra inproc` (torch cu124, ~3 GB) |
| `checkpoint` | Civitai version `CIVITAI_VERSION` (default 2167369, an Illustrious-class SDXL) → `sdxl-checkpoint.safetensors` (~7 GB, retried once, rejected under 6 GB — a Civitai auth failure is a small HTML page with HTTP 200) |
| `server` | `uv run brushjam`; the DMD2 LoRA and the fp16-fix VAE are fetched by the pipeline itself into `HF_HOME`, then the model loads (~30 s) |

`status` reports `server_healthy: true` when `/healthz` answers. The URL to send
to friends is `https://{podId}-8787.proxy.runpod.net/`.

Everything except the tarball is on the persistent volume - models, the HF
cache and `/workspace/venv` - so a `stop` / `start` cycle skips straight to the
model load.

## Notes and limits

- **Cost**: a 4090 SECURE pod bills while RUNNING; `stop` keeps only the volume.
  `status` prints `$/hr`.
- **One GPU, one generation at a time.** The server's own admission control
  handles the queue; expect ~1.5–2 s per edit at fast/768 and ~9 s at
  quality/1024 on a 3070, faster on a 4090.
- **The proxy blocks the default urllib user agent.** A request sent as
  `Python-urllib/3.x` gets a flat 403 from `*.proxy.runpod.net` - the identical
  request with curl's user agent goes through. `deploy.py` names itself
  (`brushjam-deploy/1.0`) on every proxied call, and anything else talking to
  the pod has to as well. The failure is indistinguishable from a rejected
  upload token, so check this first.
- **A pod created before the checkpoint was renamed** has the file under its
  old name on the volume. Set `INPROC_CHECKPOINT` in that pod's env to the old
  path rather than letting it download seven gigabytes again.
- **The receiver only accepts the real app.** A body under 10 KB is refused
  before a byte is written, and one that is not a gzip archive listing
  `bootstrap.sh` is refused before anything is replaced or signalled. An empty
  PUT with a valid token used to be accepted: it overwrote `app.tgz` and
  SIGTERMed the server, which killed a bootstrap midway through `uv sync` and
  left the pod with nothing to run.
- **Uploads must carry a Content-Length.** A chunked PUT cannot be stored, so
  the receiver answers 411 (no length) or 400 (unparseable, or a body cut
  short) rather than the 403 it uses for a bad token. `deploy.py` sends the
  tarball as bytes for this reason - urllib sends a file object chunked.
- **The receiver is written by `dockerStartCmd`, not by the tarball.** Changing
  `receiver.py` needs a new pod; `upload` only replaces the server code.
- **The receiver is the attack surface.** It is token-gated (a 24-byte urlsafe
  token, new per deploy) and only accepts `PUT /upload`, `GET /log`,
  `GET /status`. `boot.log` and `/status` never contain a secret. Anyone with
  the pod URL can still reach the game on 8787 — RunPod proxy URLs are
  unguessable but public, so treat the link as the only access control.
- **No SSH by design.** If something is wrong that `log` cannot show, use the
  RunPod web console.
