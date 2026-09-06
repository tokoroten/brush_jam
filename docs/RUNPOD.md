# RunPod serverless backend

> **Status (2026-09-05 11:xx JST): torn down at the user's request.** The endpoint `srmomg5bp1e2hm`, template `mv8u4g9k6w` and network volume `pqx6rs7f48` were deleted via the REST API (all returned 204); `RUNPOD_ENDPOINT_ID` was removed from `.env`. Re-deploying means repeating the steps below (about 30 minutes, ~$0.05 plus the volume rent). The measured numbers remain valid as a reference.


`AI_BACKEND=runpod` runs the **same ComfyUI workflow** the local backend builds,
on a RunPod serverless endpoint instead of a local ComfyUI. It is a transport
change, not a pipeline change: the same `buildWorkflow()` output, the same
checkpoint, the same DMD2 LoRA, the same `VAEDecodeTiled`.

Deployed and measured on 2026-09-05. Everything below is real: ids, prices and
timings come from the live account, not from documentation.

---

## 1. What was created

| Resource | Id | Where | Price |
|---|---|---|---|
| Network volume `brushjam-models` | `pqx6rs7f48` | EU-RO-1 | 30 GB x $0.07/GB/month = **$2.10/month** ($0.0029/hr) |
| Serverless template `brushjam-comfyui` | `mv8u4g9k6w` | - | free (15 GB container disk, ~$0.10/GB/month **only while a worker runs**) |
| Serverless endpoint `brushjam-comfyui` | `srmomg5bp1e2hm` | EU-RO-1 | **$0** at rest; $1.10/hr per running worker (see §4) |
| CPU pod `brushjam-model-loader` | `r1sdoc67bdua1v` | EU-RO-1 | $0.16/hr - **terminated**, lived ~3 min, ~$0.01 |

The endpoint is the only thing still standing, and it costs nothing when idle:
`workersMin: 0`, `workersMax: 1`, `idleTimeout: 5 s`. The account's
`currentSpendPerHr` with no traffic is $0.003/hr - that is the network volume
alone, which confirms the endpoint really does bill nothing at zero workers.

### Why EU-RO-1

Datacenters were filtered on two things at once: network-volume support and 4090
availability. Of the 50 datacenters the GraphQL API lists, EU-RO-1 was the only
one with `storageSupport: true` **and** RTX 4090 stock reported `High`
(EUR-IS-1, EUR-NO-1 and US-IL-1 had storage plus `Low` 4090 stock; EU-CZ-1 and
EUR-IS-2 had good stock but no network volumes at all). A network volume cannot
be moved between datacenters, so this choice pins the endpoint's region.

### Endpoint configuration

```
image        runpod/worker-comfyui:5.10.0-base   (base = no bundled models; ours come from the volume)
GPUs         ADA_24 (RTX 4090) > AMPERE_24 (A5000 / L4), RTX 3090 excluded
volume       pqx6rs7f48, mounted read-write at /runpod-volume
workersMin   0        workersMax  1
idleTimeout  5 s      flashboot   on
executionTimeoutMs 600000
scaler       QUEUE_DELAY 4
```

`-base` rather than `-sdxl`: the `-sdxl` tag only bundles the stock SD XL
checkpoint, which this project does not use. ComfyUI itself handles SDXL either
way, and a smaller image is a faster cold start.

### Volume contents

Serverless workers see the volume at `/runpod-volume`, and `worker-comfyui`'s
`extra_model_paths.yaml` expects models under `/runpod-volume/models/...`.
(Pods mount the same volume at `/workspace` - that is why the loader pod wrote
to `/workspace/models`.)

```
/runpod-volume/models/checkpoints/waiNSFWIllustrious_v150.safetensors   6 938 040 682 B
/runpod-volume/models/loras/dmd2_sdxl_4step_lora_fp16.safetensors         393 854 592 B
```

Both were verified by SHA-256 against the local copies in `E:\ComfyUI\models\`:

```
befc694a296f75e996488ebf9f9db8a1493bd059b6e704b975829e87d5aeb4fa  waiNSFWIllustrious_v150.safetensors
b3d9173815a4b595991c3a7a0e0e63ad821080f314a0b2a3cc31ecd7fcf2cbb8  dmd2_sdxl_4step_lora_fp16.safetensors
```

> **Superseded 2026-09-06.** The repo default is now Civitai version **2940478**
> (Nova Anime XL IL v19.0, model 376130, `novaAnimeXL_ilV190.safetensors`,
> SHA-256 `fa486caafc330f133605d3c18b418d183812f14946631c6544bfb28730db6d6f`),
> because WAI's Civitai permissions do not allow use on a generation service.
> The volume described below still holds the WAI file; a pod keeps using it
> until the file at `sdxl-checkpoint.safetensors` is replaced, since the
> bootstrap only downloads when nothing large is there yet.

The checkpoint was Civitai model **827184** ("WAI-illustrious-SDXL") version
**2167369** ("v15.0"). Civitai serves it as `waiIllustriousSDXL_v150.safetensors`
- it is renamed to `waiNSFWIllustrious_v150.safetensors` on the volume so the
repo's default `COMFYUI_CHECKPOINT` keeps working unchanged. Its published
SHA-256 matches the bytes above, so the rename is the only difference. The LoRA
is Hugging Face `tianweiy/DMD2`, file `dmd2_sdxl_4step_lora_fp16.safetensors`.

Both files were fetched by a throwaway CPU pod (`python:3.12-slim`, 4 vCPU,
`cpu3g`) whose `dockerStartCmd` did the downloads and published its log on the
pod's `:8888` proxy URL, so no SSH was needed. The tokens went in as pod env
vars and never appeared in any log. Total download time: **60 seconds** for
7.3 GB. The pod was terminated immediately afterwards.

---

## 2. How to run it

`RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` are both in the repo-root `.env`
(gitignored). The server loads that file at startup, so for normal use:

```
AI_BACKEND=runpod pnpm start
```

Scripts do **not** read `.env` - the server's `src/index.ts` does. Pass the two
variables explicitly when driving a script:

```bash
# one live generation, straight through the backend, no server or WebSocket
RUNPOD_ENDPOINT_ID=srmomg5bp1e2hm RUNPOD_API_KEY=... \

# denoise sweep
AI_BACKEND=runpod RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=... \
    --denoise 0.65,0.8 --profile fast
```

Everything else behaves as it does with local ComfyUI: `AI_PROFILE`,
`AI_WINDOW`, `AI_DENOISE`, `AI_STEPS`, `AI_FAST_STEPS`, `AI_CFG`, `AI_VAE_TILE`,
`COMFYUI_CHECKPOINT` and `COMFYUI_FAST_LORA` all still apply, because the
workflow is built by the same code. `COMFYUI_URL` is ignored.

`RUNPOD_TIMEOUT_MS` (default 300 000) is the backend's own deadline for one
generation. It has to stay well above the cold start; note that the scheduler's
`AI_WATCHDOG_MS` (default 180 000) is the tighter of the two in practice.

---

## 3. Measured numbers

Endpoint in EU-RO-1, client in Japan. `worker-comfyui:5.10.0-base`, checkpoint
and LoRA off the network volume, `AI_VAE_TILE=512`, denoise 0.7, `CANVAS_SIZE=1024`,
`AI_MODE=full`.

### Cold start

| | |
|---|---|
| First request after the endpoint has scaled to zero | **78-81 s** |
| of which: queued, waiting for a machine (`delayTime`) | ~26 s |
| of which: container boot + 7 GB checkpoint load from the volume + sampling (`executionTime`) | ~53 s |
| Warm worker, `delayTime` | ~0.1 s |

Two independent measurements agreed: 81 s (a bare `/run` probe) and 78.2 s
(first edit of a latency run). A worker that has only just gone idle comes back
much faster - 5-11 s - because FlashBoot restores it rather than rebooting.

This is why `RunpodBackend` polls: RunPod's `/runsync` gives up waiting after
about 90 s and answers `{id, status: "IN_PROGRESS"}`, leaving the caller to poll
`/status/{id}`. On a cold endpoint that is the normal path, not an edge case.

### Warm per-edit latency (`scripts/latency.ts`, n=6, first edit excluded)

`stroke_end` -> pixels on screen, as a client feels it: debounce + input render
+ RunPod round trip + composite + PNG fetch.

| profile / size | end-to-end median | server pipeline median | GPU `executionTime` |
|---|---|---|---|
| fast / 768 | **3 010 ms** | 2 578 ms | ~1 030 ms |
| fast / 1024 | **4 376 ms** | 3 913 ms | ~1 500 ms |
| quality / 1024 | **4 641 ms** | 4 198 ms | ~2 890 ms |
| quality / 768 (not run end-to-end) | - | ~2 940 ms observed backend-only | ~1 660 ms |

For comparison, the same three configs on the local RTX 3070 (8 GB) measured
3 710 / 5 665 / 10 294 ms (`docs/experiments/2026-09-05-comfyui/REPORT.md`). So
the 4090 is roughly **3.5x faster at 14 steps** and the cloud round trip eats
most of the win at 4 steps.

**About 1.3-1.4 s of every request is transport, not GPU.** `executionTime` minus
wall clock is consistent across all four configs, and it is base64: a 1024x1024
PNG image plus its mask go up as base64 JSON and the result comes back the same
way, from Japan to Romania. Nothing in the workflow will reduce that; only
region or an S3 output would.

### Quality

`docs/experiments/2026-09-05-runpod/fast-768/` and `quality-768/` hold a
2 drawings x 2 denoise grid for each profile (drawings a and c, denoise 0.65 and
0.8). Per-cell backend time: fast median 3 659 ms, quality median 4 779 ms. The
images behave exactly as the local run did at the same settings: at 0.65 the
house gains a door and windows, at 0.8 it becomes an architectural sketch, and
the noise band in drawing (c) resolves into structure (at 768 into character art
and pattern rather than the row of buildings the local 1024 run produced - the
same "the noise pen wants 1024" finding, not a RunPod difference). Same
checkpoint, same LoRA, same workflow, same seed; the backend is the only thing
that changed, and it does not show.

### Cost per generation

Serverless flex rate for the 24 GB "4090 PRO" tier is **$1.10/hr**
($0.000306/s); if the endpoint falls back to A5000/L4 (`AMPERE_24`) it is
**$0.69/hr** ($0.000192/s). Billing is per second of *worker running time*,
which includes start time, execution and the idle timeout.

| | 4090 @ $1.10/hr |
|---|---|
| fast / 768, back to back (1.03 s) | **$0.00031** |
| fast / 1024 (1.5 s) | $0.00046 |
| quality / 1024 (2.9 s) | **$0.00089** |
| a single isolated edit (+5 s idle timeout) | +$0.0015 |
| one cold start (~53 s of worker time) | **$0.016** |

So a 30-minute playtest session in fast/768 at one edit every ~6 s - roughly 300
edits - costs about **$0.09 of GPU plus one $0.016 cold start**, call it
**$0.11**, against $2.10/month of storage that is charged whether anyone draws
or not. The dominant cost of light use is the volume, not the GPU.

---

## 4. Tearing it down

In cost order. Each is a single REST call with
`Authorization: Bearer $RUNPOD_API_KEY`.

The **network volume is the only thing that bills while nothing is happening**
($2.10/month). Delete it and the endpoint stops working - the models go with it.

```bash
# 1. the endpoint (stops any accidental GPU spend; free at rest anyway)
curl -X DELETE https://rest.runpod.io/v1/endpoints/srmomg5bp1e2hm \
  -H "Authorization: Bearer $RUNPOD_API_KEY"

# 2. the template
curl -X DELETE https://rest.runpod.io/v1/templates/mv8u4g9k6w \
  -H "Authorization: Bearer $RUNPOD_API_KEY"

# 3. the network volume - THIS DELETES THE 7 GB CHECKPOINT AND THE LORA
curl -X DELETE https://rest.runpod.io/v1/networkvolumes/pqx6rs7f48 \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

Check nothing is left, and that nothing is running:

```bash
curl -s https://rest.runpod.io/v1/endpoints       -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -s https://rest.runpod.io/v1/networkvolumes  -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -s https://rest.runpod.io/v1/pods            -H "Authorization: Bearer $RUNPOD_API_KEY"
```

A pod (the loader kind) is deleted the same way: `DELETE /v1/pods/{podId}`.
`DELETE` is terminate, not stop - a stopped pod still bills for its disk.

To rebuild the volume from scratch, the loader pod's recipe is: `python:3.12-slim`,
`computeType: CPU`, `cpuFlavorIds: ["cpu3g"]`, the volume at `/workspace`, and a
`dockerStartCmd` that `curl`s the Civitai version `CIVITAI_VERSION` download URL (with
`Authorization: Bearer $CIVITAI_TOKEN`) into `/workspace/models/checkpoints` and
the Hugging Face `tianweiy/DMD2` file into `/workspace/models/loras`.

---

## 5. Open issues

- **The endpoint is not authenticated beyond the RunPod API key.** Anyone with
  the key can run arbitrary ComfyUI workflows on it. The key lives only in
  `.env` (gitignored) and is never logged, but the room server holds it in
  memory and passes it on every request.
- **Region is fixed by the volume.** EU-RO-1 was chosen for 4090 stock, not for
  latency to any particular player. From Japan the fixed transport cost is
  ~1.3 s per edit. A playtest in Europe would feel meaningfully faster; moving
  region means recreating the volume and re-uploading 7.3 GB.
- **Cold start is 80 s and there is no warm-up hook.** The first person to draw
  after an idle period waits well over a minute, which the UI presents as an
  ordinary slow generation. `AI_WATCHDOG_MS` defaults to 180 s so it does not
  fire, but the margin is not large. Nothing pre-warms the endpoint; a cheap fix
  would be to fire a tiny throwaway workflow when a room's first client
  connects.
- **`workersMax: 1`.** Two rooms generating at once serialise behind one worker.
  That is the authorised limit, not a recommendation.
- **ComfyUI caches by prompt hash.** Re-sending a byte-identical workflow returns
  the previous image in ~200 ms without touching the GPU. This silently
  invalidated a first round of "warm" measurements here; `runpod-smoke.ts` (since deleted with the Node server)
  varies the seed per run for that reason. Anything that benchmarks this backend
  must vary its input.
- **Not measured:** behaviour under a real cancel (the abort path calls
  `/cancel/{id}`, which is covered by unit tests but was never exercised against
  the live endpoint), the `AMPERE_24` fallback GPUs (every request in these
  measurements landed on a 4090), and multi-room contention.
- **Not measured:** whether `flashboot` costs anything. RunPod bills worker
  running time and FlashBoot resumes were 5-11 s here, but no billing record had
  been aggregated by the time of writing, so the cost figures above are computed
  from the published per-second rate rather than read off an invoice.
