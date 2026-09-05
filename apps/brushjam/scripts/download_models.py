"""Fetch an SDXL checkpoint into ./models/checkpoints and print the .env line.

    uv run --project apps/brushjam python apps/brushjam/scripts/download_models.py

The checkpoint is the one thing this project cannot ship or guess: 6-7 GB, and
which one you want is a taste decision. Any SDXL `.safetensors` works. The
default here is the Civitai model version the server was developed against; set
`CIVITAI_VERSION` to fetch a different one, and `CIVITAI_TOKEN` if the model
needs an account (most do).

The 4-step DMD2 LoRA and the fp16-fix VAE are small and are downloaded by the
pipeline itself on first use, into `INPROC_LORA_DIR` and the Hugging Face
cache. There is nothing to do about those.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO_ROOT / "models" / "checkpoints"
#: Illustrious-based SDXL; what the measurements in docs/ were taken with.
DEFAULT_VERSION = "2167369"
DEFAULT_NAME = "sdxl-illustrious.safetensors"
#: A Civitai auth failure is a small HTML page with HTTP 200, so size is the
#: only honest check that what arrived is a model.
MIN_BYTES = 1_000_000_000


def load_env() -> None:
    """Read the repo-root .env, so CIVITAI_TOKEN need not be exported."""
    path = REPO_ROOT / ".env"
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
        return
    except ImportError:
        pass
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def download(url: str, dest: Path, token: str) -> None:
    request = urllib.request.Request(url)
    request.add_header("user-agent", "brushjam-download/1.0")
    if token:
        request.add_header("authorization", f"Bearer {token}")

    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("content-length") or 0)
            done = 0
            with tmp.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
            print()
    except urllib.error.HTTPError as err:
        tmp.unlink(missing_ok=True)
        hint = " (set CIVITAI_TOKEN in .env)" if err.code in (401, 403) else ""
        sys.exit(f"download failed: HTTP {err.code}{hint}")

    size = tmp.stat().st_size
    if size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        sys.exit(
            f"the download is only {size / 1e6:.1f} MB, which is not a checkpoint. "
            "Civitai answers an auth failure with a small HTML page and HTTP 200, "
            "so this usually means CIVITAI_TOKEN is missing or wrong."
        )
    tmp.replace(dest)


def main(argv: list | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--version",
        default=os.environ.get("CIVITAI_VERSION") or DEFAULT_VERSION,
        help="Civitai model *version* id (the number in the download URL)",
    )
    parser.add_argument("--name", default=DEFAULT_NAME, help="file name to save it as")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="where to put it")
    parser.add_argument("--url", help="download from here instead of Civitai")
    args = parser.parse_args(argv)

    dest = Path(args.dir) / args.name
    if dest.exists() and dest.stat().st_size >= MIN_BYTES:
        print(f"already there: {dest} ({dest.stat().st_size / 1e9:.1f} GB)")
    else:
        url = args.url or (
            f"https://civitai.com/api/download/models/{args.version}?type=Model&format=SafeTensor"
        )
        print(f"downloading to {dest}")
        download(url, dest, os.environ.get("CIVITAI_TOKEN", ""))
        print(f"done: {dest.stat().st_size / 1e9:.1f} GB")

    print("\nPut this in your .env:\n")
    print(f"INPROC_CHECKPOINT={dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
