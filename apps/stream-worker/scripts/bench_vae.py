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
import atexit
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Every worker this script starts. No path out of here - exception, Ctrl+C,
# SIGTERM - may leave an orphan holding 5 GB of VRAM.
_CHILDREN: list[subprocess.Popen] = []

# This console is cp932; a stray non-ASCII character in the summary would raise
# UnicodeEncodeError *after* all the GPU work is done and throw the results
# away. Force UTF-8 and degrade rather than crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
            # dry-run never loads a model, so it is ready as soon as it answers.
            if health.get("warm") or health.get("backend") == "dry-run":
                return health
            if health.get("error"):
                raise RuntimeError(f"worker failed to load: {health['error']}")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"worker did not become warm within {timeout}s")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    if proc in _CHILDREN:
        _CHILDREN.remove(proc)


def stop_all() -> None:
    for proc in list(_CHILDREN):
        stop(proc)


def gpu_used_mib() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


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
    ap.add_argument(
        "--budget-seconds",
        type=float,
        default=840.0,
        help="overall deadline; remaining configurations are skipped rather than overrunning a granted GPU window",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="rehearse the harness with STREAM_DRY_RUN=1: no model, no GPU, meaningless timings",
    )
    args = ap.parse_args()

    atexit.register(stop_all)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: sys.exit(130))
        except (ValueError, OSError):
            pass  # not the main thread, or unsupported here

    started_all = time.perf_counter()

    def remaining() -> float:
        return args.budget_seconds - (time.perf_counter() - started_all)

    try:
        with urllib.request.urlopen(f"{args.url}/healthz", timeout=3):
            sys.exit(f"a worker is already listening on {args.url}; stop it first (this script starts its own)")
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        pass

    port = args.url.rsplit(":", 1)[-1]
    results: dict[str, list[dict]] = {}
    memory: dict[str, dict] = {}
    load_seconds: dict[str, float] = {}

    skipped: list[str] = []
    for vae in args.vaes:
        if remaining() <= 0:
            skipped.append(vae)
            print(f"\n=== STREAM_VAE={vae} SKIPPED (out of time budget) ===", flush=True)
            continue
        env = {**os.environ, "STREAM_VAE": vae, "STREAM_PORT": port, "STREAM_WARMUP_SIZE": str(min(args.sizes))}
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if args.dry_run:
            env["STREAM_DRY_RUN"] = "1"
        print(f"\n=== STREAM_VAE={vae} ({remaining():.0f}s of budget left) ===", flush=True)
        started = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "stream_worker"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _CHILDREN.append(proc)
        try:
            health = wait_until_warm(args.url, proc, args.load_timeout)
            load_seconds[vae] = round(time.perf_counter() - started, 1)
            memory[vae] = health.get("memory", {})
            print(f"  warm in {load_seconds[vae]}s, memory={memory[vae]}", flush=True)
            rows = []
            for size in args.sizes:
                if remaining() <= 0:
                    print(f"  {size}: skipped (out of time budget)", flush=True)
                    continue
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
    header = "| VAE | cold start | " + " | ".join(f"{s}px" for s in args.sizes) + " |"
    print(header)
    print("| --- |" + " --- |" * (len(args.sizes) + 1))
    for vae in args.vaes:
        cells = []
        for size in args.sizes:
            row = next((r for r in results.get(vae, []) if r["size"] == size), None)
            cells.append(f"{row['wall_median_ms']} ms" if row else "n/a")
        cold = f"{load_seconds[vae]} s" if vae in load_seconds else "n/a"
        print(f"| `{vae}` | {cold} | " + " | ".join(cells) + " |")

    print("\n### Phase breakdown (median ms)\n")
    print("| VAE | size | unet | vae encode | vae decode | png encode |")
    print("| --- | --- | --- | --- | --- | --- |")
    for vae in args.vaes:
        for row in results.get(vae, []):
            print(
                f"| `{vae}` | {row['size']}px | {row.get('unet_median_ms', 'n/a')} | "
                f"{row.get('vae_encode_median_ms', 'n/a')} | {row.get('vae_decode_median_ms', 'n/a')} | "
                f"{row.get('png_encode_median_ms', 'n/a')} |"
            )

    print("\n### VRAM at idle, per VAE\n")
    print("| VAE | allocated | reserved | peak | device free |")
    print("| --- | --- | --- | --- | --- |")
    for vae in args.vaes:
        m = memory.get(vae) or {}
        if not m:
            continue
        print(
            f"| `{vae}` | {m.get('allocated_gb', 'n/a')} GB | {m.get('reserved_gb', 'n/a')} GB | "
            f"{m.get('max_allocated_gb', 'n/a')} GB | {m.get('device_free_gb', 'n/a')} GB |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"results": results, "memory": memory, "load_seconds": load_seconds, "skipped": skipped},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    if skipped:
        print(f"SKIPPED for time: {', '.join(skipped)}")

    stop_all()
    time.sleep(3)
    print(
        f"\nall workers stopped; GPU now at {gpu_used_mib()} "
        f"(elapsed {time.perf_counter() - started_all:.0f}s)"
    )


if __name__ == "__main__":
    main()
