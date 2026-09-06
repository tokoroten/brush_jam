"""Render every style/mood preset over one fixed drawing, as a contact sheet.

The presets are prompts for a particular checkpoint, so the only way to know
whether one works is to look at what it produces. This drives a *running*
server as an ordinary client - HTTP for the room and the images, the WebSocket
for the room protocol - and never loads a model itself, so it can be pointed
at whatever server happens to be warm.

    uv run --no-sync --project apps/brushjam python \
        apps/brushjam/scripts/preset_sheet.py --variant after

One room per variant rather than one per preset: the room's whole input is the
reference drawing, which does not change, so the settings are all that differ
between two generations, and a run of twenty rooms would eat the room budget
of a server someone else is using. The AI result is never fed back as input,
so a generation cannot inherit the one before it.

Prompts are read out of the TypeScript source, which is where they live; the
`--presets` flag takes an older copy of that file so a "before" sheet can be
rendered with the prompts as they were.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import websockets
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
PRESETS_TS = ROOT / "packages" / "shared" / "src" / "presets.ts"
DEFAULT_INPUT = (
    ROOT / "docs" / "experiments" / "2026-09-05-stream" / "dmd2-768" / "2026-09-05" / "a_input.png"
)
LOOK_GROUPS = ("画風 / style", "雰囲気 / mood")


# --------------------------------------------------------------------- presets


def parse_presets(source: Path) -> List[Dict[str, Any]]:
    """The preset table, read out of the TypeScript.

    Deliberately a regex and not a parser: the table is a literal array of
    object literals with five known keys, and importing it would mean a node
    build step in the middle of an experiment script.
    """
    text = source.read_text(encoding="utf-8")
    found = re.search(r"DEFAULT_NEGATIVE_PROMPT\s*=\s*'([^']*)'", text)
    negative_default = found.group(1) if found else ""

    presets: List[Dict[str, Any]] = []
    for pid, body in re.findall(r"\{\s*id: '([^']+)',(.*?)\n  \},", text, re.S):

        def field(name: str, body: str = "") -> Optional[str]:
            match = re.search(rf"{name}:\s*\n?\s*(['\"`])(.*?)\1,", body, re.S)
            return match.group(2) if match else None

        prompt = field("prompt", body)
        if prompt is None:
            continue
        negative = field("negative", body) or ""
        group = field("group", body) or ""
        label = field("label", body) or pid
        denoise = re.search(r"denoise:\s*([0-9.]+)", body)
        presets.append(
            {
                "id": pid,
                "label": label,
                "group": group,
                # A template literal's one substitution, spelled out.
                "prompt": prompt.replace("${DEFAULT_NEGATIVE_PROMPT}", negative_default),
                "negative": negative.replace("${DEFAULT_NEGATIVE_PROMPT}", negative_default),
                "denoise": float(denoise.group(1)) if denoise else None,
            }
        )
    return presets


# ----------------------------------------------------------------- the client


class Room:
    """One room, driven over the WebSocket."""

    def __init__(self, base: str, room_id: str, socket: Any) -> None:
        self.base = base
        self.room_id = room_id
        self.socket = socket
        self.snapshot: Dict[str, Any] = {}

    async def send(self, msg: Dict[str, Any]) -> None:
        await self.socket.send(json.dumps(msg))

    async def wait_for(self, kind: str, timeout: float = 300.0) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no {kind} within {timeout}s")
            raw = await asyncio.wait_for(self.socket.recv(), timeout=left)
            msg = json.loads(raw)
            if msg.get("t") == "error":
                raise RuntimeError(f"server refused: {msg.get('message')}")
            if msg.get("t") == kind:
                return msg


async def open_room(client: httpx.AsyncClient, base: str) -> str:
    response = await client.post(f"{base}/api/rooms")
    response.raise_for_status()
    return response.json()["roomId"]


async def place_drawing(client: httpx.AsyncClient, room: Room, drawing: Path) -> None:
    """Upload the fixed input as a reference layer that the AI reads."""
    upload = await client.post(
        f"{room.base}/rooms/{room.room_id}/images",
        content=drawing.read_bytes(),
        headers={"content-type": "image/png"},
    )
    upload.raise_for_status()
    stored = upload.json()
    canvas = room.snapshot["canvasSize"]
    scale = canvas / max(stored["width"], stored["height"])
    await room.send(
        {
            "t": "layer_create",
            "layer": {
                "kind": "reference",
                "imageId": stored["imageId"],
                "x": 0,
                "y": 0,
                "scale": scale,
            },
        }
    )
    created = await room.wait_for("layer_created")
    # Reference layers are decoration until they are turned on for the AI.
    await room.send(
        {"t": "layer_update", "id": created["layer"]["id"], "patch": {"includeInAI": True}}
    )
    await room.wait_for("layer_updated")


async def generate(room: Room, preset: Dict[str, Any], seed: int, denoise: float) -> Dict[str, Any]:
    await room.send({"t": "set_prompt", "prompt": preset["prompt"]})
    await room.send(
        {
            "t": "set_ai_settings",
            "denoise": denoise,
            "negativePrompt": preset["negative"],
            "seed": seed,
            "aiProfile": "fast",
        }
    )
    return await room.wait_for("ai_result")


async def run_variant(
    base: str,
    presets: List[Dict[str, Any]],
    drawing: Path,
    out: Path,
    seed: int,
    variant: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        room_id = await open_room(client, base)
        url = f"{base.replace('http', 'ws', 1)}/ws/rooms/{room_id}?name=sheet"
        async with websockets.connect(url, max_size=8 << 20) as socket:
            room = Room(base, room_id, socket)
            room.snapshot = (await room.wait_for("snapshot"))["snapshot"]
            await place_drawing(client, room, drawing)
            # The upload alone starts a generation; let it finish so the next
            # ai_result is unambiguously the first preset's.
            await room.wait_for("ai_result")

            for preset in presets:
                denoise = preset["denoise"] or room.snapshot["denoise"]
                started = time.monotonic()
                result = await generate(room, preset, seed, denoise)
                image = await client.get(f"{base}{result['url']}")
                image.raise_for_status()
                path = out / f"{variant}_{preset['id']}.png"
                Image.open(io.BytesIO(image.content)).convert("RGB").save(path)
                records.append(
                    {
                        "variant": variant,
                        "id": preset["id"],
                        "label": preset["label"],
                        "group": preset["group"],
                        "prompt": preset["prompt"],
                        "negative": preset["negative"],
                        "denoise": denoise,
                        "seed": seed,
                        "latencyMs": result.get("latencyMs"),
                        "file": path.name,
                    }
                )
                print(
                    f"  {variant:6} {preset['id']:16} d={denoise:.2f} "
                    f"{time.monotonic() - started:5.1f}s",
                    flush=True,
                )
    return records


# ------------------------------------------------------------------ the sheet


def contact_sheet(records: List[Dict[str, Any]], out: Path, drawing: Path, name: str) -> Path:
    """One row per preset: the input, then each variant, captioned."""
    by_id: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        by_id.setdefault(record["id"], {})[record["variant"]] = record
    variants = sorted({r["variant"] for r in records})

    cell = 320
    caption = 40
    left = 200
    width = left + cell * (1 + len(variants))
    height = len(by_id) * (cell + caption)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    source = Image.open(drawing).convert("RGB").resize((cell, cell))
    for row, (pid, variant_map) in enumerate(by_id.items()):
        y = row * (cell + caption)
        any_record = next(iter(variant_map.values()))
        draw.text((8, y + 8), pid, fill="black", font=font)
        draw.text((8, y + 26), f"d={any_record['denoise']}", fill="#666666", font=font)
        sheet.paste(source, (left, y))
        draw.text((left + 4, y + cell + 4), "input", fill="#666666", font=font)
        for column, variant in enumerate(variants):
            record = variant_map.get(variant)
            x = left + cell * (column + 1)
            if record is None:
                continue
            image = Image.open(out / record["file"]).convert("RGB").resize((cell, cell))
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + cell + 4), variant, fill="#666666", font=font)
            draw.text((x + 4, y + cell + 20), record["prompt"][:88], fill="#666666", font=font)
    path = out / name
    sheet.save(path)
    return path


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--variant", default="after", help="label for this run")
    parser.add_argument("--presets", type=Path, default=PRESETS_TS)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs" / "experiments" / "2026-09-07-presets"
    )
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--only", default="", help="comma-separated preset ids")
    parser.add_argument(
        "--sheet-only", action="store_true", help="rebuild the sheet from results.json"
    )
    parser.add_argument("--sheet", default="sheet.png")
    parser.add_argument(
        "--denoise", type=float, default=0.0, help="override every preset's denoise"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.json"
    existing: List[Dict[str, Any]] = (
        json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    )

    if not args.sheet_only:
        parsed = parse_presets(args.presets)
        if args.only:
            # An explicit list may name a preset from any group.
            wanted = {s.strip() for s in args.only.split(",") if s.strip()}
            presets = [p for p in parsed if p["id"] in wanted]
        else:
            presets = [p for p in parsed if p["group"] in LOOK_GROUPS]
        if not presets:
            print("no presets matched", file=sys.stderr)
            return 2
        print(f"{len(presets)} presets, variant {args.variant}", flush=True)
        if args.denoise:
            for preset in presets:
                preset["denoise"] = args.denoise
        fresh = await run_variant(
            args.server, presets, args.input, args.out, args.seed, args.variant
        )
        done = {(r["variant"], r["id"]) for r in fresh}
        existing = [r for r in existing if (r["variant"], r["id"]) not in done] + fresh
        results_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    sheet = contact_sheet(existing, args.out, args.input, args.sheet)
    print(f"sheet: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
