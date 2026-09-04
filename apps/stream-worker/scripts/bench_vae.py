"""Measure every VAE option at every size, in one command.

    uv run python scripts/bench_vae.py                 # all 3 VAEs x 512/768/1024
    uv run python scripts/bench_vae.py --vaes fp16fix --sizes 512

The worker must NOT already be running: this script starts and stops one worker
process per VAE, because the VAE is chosen at load time. Free ComfyUI first
(POST /free), since two residents on an 8 GB card make every number meaningless
- see docs/STREAM_WORKER.md 3.1.

Prints a markdown table of wall-clock medians plus the phase breakdown, and
writes samples/bench_vae.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import NEGATIVE, PROMPT, b64, post  # noqa: E402
from make_sample import make_drawing, make_mask  # noqa: E402

ALL_VAES = ["checkpoint", "fp16fix", "taesd"]


def wait_until_warm(url: str, proc: subprocess.Popen, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"worker exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=5) as res:
                health = json.loads(res.read().decode("utf-8"))
            if health.get("warm"):
                return health
            if health.get("error"):
                raise RuntimeError(f"worker failed to load: {health['error']}")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"worker did not become warm within {timeout}s")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def measure(url: str, size: int, runs: int, warmups: int, steps: int, denoise: float, use_mask: bool) -> dict:
    drawing = make_drawing(size)
    payload = {
        "image_b64": b64(drawing),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "denoise": denoise,
        "steps": steps,
        "seed": 12345,
        "width": size,
        "height": size,
        "queue": True,
    }
    if use_mask:
        payload["mask_b64"] = b64(make_mask(size))

    for _ in range(warmups):
        post(f"{url}/generate", payload)

    walls: list[float] = []
    phases: dict[str, list[float]] = {}
    for i in range(runs):
        payload["seed"] = 12345 + i
        t0 = time.perf_counter()
        body = post(f"{url}/generate", payload)
        walls.append((time.perf_counter() - t0) * 1000.0)
        for key in ("unet_ms", "vae_encode_ms", "vae_decode_ms", "png_encode_ms", "empty_cache_ms"):
            if key in body["timings"]:
                phases.setdefault(key, []).append(body["timings"][key])

    row = {
        "size": size,
        "runs": runs,
        "wall_median_ms": round(statistics.median(walls)),
        "wall_min_ms": round(min(walls)),
        "wall_max_ms": round(max(walls)),
    }
    for key, values in phases.items():
        row[key.replace("_ms", "_median_ms")] = round(statistics.median(values))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8790")
    ap.add_argument("--vaes", nargs="+", default=ALL_VAES, choices=ALL_VAES)
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 768, 1024])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--denoise", type=float, default=0.8)
    ap.add_argument("--mask", action="store_true", default=True)
    ap.add_argument("--load-timeout", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=Path("samples") / "bench_vae.json")
    args = ap.parse_args()

    try:
        with urllib.request.urlopen(f"{args.url}/healthz", timeout=3):
            sys.exit(f"a worker is already listening on {args.url}; stop it first (this script starts its own)")
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        pass

    port = args.url.rsplit(":", 1)[-1]
    results: dict[str, list[dict]] = {}
    memory: dict[str, dict] = {}
    load_seconds: dict[str, float] = {}

    for vae in args.vaes:
        env = {**os.environ, "STREAM_VAE": vae, "STREAM_PORT": port, "STREAM_WARMUP_SIZE": str(min(args.sizes))}
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print(f"\n=== STREAM_VAE={vae} ===", flush=True)
        started = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "stream_worker"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            health = wait_until_warm(args.url, proc, args.load_timeout)
            load_seconds[vae] = round(time.perf_counter() - started, 1)
            memory[vae] = health.get("memory", {})
            print(f"  warm in {load_seconds[vae]}s, memory={memory[vae]}", flush=True)
            rows = []
            for size in args.sizes:
                row = measure(args.url, size, args.runs, args.warmups, args.steps, args.denoise, args.mask)
                print(f"  {size}: {row['wall_median_ms']} ms median  {row}", flush=True)
                rows.append(row)
            results[vae] = rows
        except Exception as err:  # keep going: one bad VAE should not lose the rest
            print(f"  FAILED: {type(err).__name__}: {err}", flush=True)
            results[vae] = []
        finally:
            stop(proc)
            time.sleep(3)  # let the driver reclaim the VRAM before the next load

    print("\n\n### Wall-clock median, ms\n")
    header = "| VAE | cold start | " + " | ".join(f"{s}²" for s in args.sizes) + " |"
    print(header)
    print("| --- |" + " --- |" * (len(args.sizes) + 1))
    for vae in args.vaes:
        cells = []
        for size in args.sizes:
            row = next((r for r in results.get(vae, []) if r["size"] == size), None)
            cells.append(f"{row['wall_median_ms']} ms" if row else "—")
        cold = f"{load_seconds[vae]} s" if vae in load_seconds else "—"
        print(f"| `{vae}` | {cold} | " + " | ".join(cells) + " |")

    print("\n### Phase breakdown (median ms)\n")
    print("| VAE | size | unet | vae encode | vae decode | png encode |")
    print("| --- | --- | --- | --- | --- | --- |")
    for vae in args.vaes:
        for row in results.get(vae, []):
            print(
                f"| `{vae}` | {row['size']}² | {row.get('unet_median_ms', '—')} | "
                f"{row.get('vae_encode_median_ms', '—')} | {row.get('vae_decode_median_ms', '—')} | "
                f"{row.get('png_encode_median_ms', '—')} |"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"results": results, "memory": memory, "load_seconds": load_seconds}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
