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
from typing import Any, Callable, Dict, List, Optional

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
        profile = re.search(r"profile:\s*'(\w+)'", body)
        presets.append(
            {
                "id": pid,
                "label": label,
                "group": group,
                # A template literal's one substitution, spelled out.
                "prompt": prompt.replace("${DEFAULT_NEGATIVE_PROMPT}", negative_default),
                "negative": negative.replace("${DEFAULT_NEGATIVE_PROMPT}", negative_default),
                "denoise": float(denoise.group(1)) if denoise else None,
                "profile": profile.group(1) if profile else "fast",
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


async def generate(
    room: Room, preset: Dict[str, Any], seed: int, denoise: float, profile: str
) -> Dict[str, Any]:
    await room.send({"t": "set_prompt", "prompt": preset["prompt"]})
    await room.send(
        {
            "t": "set_ai_settings",
            "denoise": denoise,
            "negativePrompt": preset["negative"],
            "seed": seed,
            "aiProfile": profile,
        }
    )
    return await room.wait_for("ai_result")


async def wait_for_quiet(client: httpx.AsyncClient, base: str, mine: int) -> int:
    """Hold off while somebody else is using the server.

    A sweep is a few hundred generations on a machine a person may be drawing
    on, and one 8GB card runs one generation at a time. `generations` in
    /healthz counts every room's, so a count that moves when this script is not
    generating is someone else at work: back off until it has been still, and
    the server idle, for thirty seconds.

    `mine` is the count as of this script's last generation; the value returned
    takes their generations into account, so the next call starts from what the
    server now says rather than accusing them of it twice.
    """
    quiet_since: Optional[float] = None
    announced = False
    while True:
        health = (await client.get(f"{base}/healthz")).json()
        count = health.get("generations", 0)
        if count != mine or health.get("busy"):
            mine = count
            quiet_since = None
            if not announced:
                print("  ...someone else is generating; waiting", flush=True)
                announced = True
        else:
            if quiet_since is None:
                quiet_since = time.monotonic()
            if not announced or time.monotonic() - quiet_since >= 30.0:
                return count
        await asyncio.sleep(3.0)


async def run_jobs(
    base: str,
    jobs: List[Dict[str, Any]],
    drawing: Path,
    input_label: str,
    out: Path,
    seed: int,
    variant: str,
    polite: bool,
    save: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> List[Dict[str, Any]]:
    """Every job in one room: the input never changes, so only settings do."""
    records: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        room_id = await open_room(client, base)
        url = f"{base.replace('http', 'ws', 1)}/ws/rooms/{room_id}?name=sheet"
        async with websockets.connect(url, max_size=8 << 20) as socket:
            room = Room(base, room_id, socket)
            room.snapshot = (await room.wait_for("snapshot"))["snapshot"]
            await place_drawing(client, room, drawing)
            # The upload alone starts a generation; let it finish so the next
            # ai_result is unambiguously the first job's.
            await room.wait_for("ai_result")
            seen = (await client.get(f"{base}/healthz")).json().get("generations", 0)

            for job in jobs:
                preset = job["preset"]
                denoise = job["denoise"] or room.snapshot["denoise"]
                profile = job["profile"]
                if polite:
                    seen = await wait_for_quiet(client, base, seen)
                started = time.monotonic()
                result = await generate(room, preset, seed, denoise, profile)
                seen += 1
                image = await client.get(f"{base}{result['url']}")
                image.raise_for_status()
                name = f"{variant}_{input_label}_{profile}_d{denoise:.2f}_{preset['id']}.png"
                Image.open(io.BytesIO(image.content)).convert("RGB").save(out / name)
                records.append(
                    {
                        "variant": variant,
                        "input": input_label,
                        "id": preset["id"],
                        "label": preset["label"],
                        "group": preset["group"],
                        "prompt": preset["prompt"],
                        "negative": preset["negative"],
                        "denoise": denoise,
                        "profile": result.get("profile", profile),
                        "seed": seed,
                        "latencyMs": result.get("latencyMs"),
                        "file": name,
                    }
                )
                # Saved as they arrive: a sweep is long enough that losing it
                # to an interrupted run means spending the GPU time twice.
                if save is not None:
                    save(records)
                print(
                    f"  {input_label} {profile:7} d={denoise:.2f} {preset['id']:16}"
                    f"{time.monotonic() - started:5.1f}s",
                    flush=True,
                )
    return records


# ------------------------------------------------------------------ the sheet


def sweep_sheet(records: List[Dict[str, Any]], out: Path, pid: str, name: str) -> Path:
    """One preset: denoise down the rows, input x profile across the columns."""
    mine = [r for r in records if r["id"] == pid]
    denoises = sorted({r["denoise"] for r in mine})
    columns = sorted({(r["input"], r["profile"]) for r in mine})
    cell = 300
    caption = 20
    left = 90
    top = 24
    sheet = Image.new(
        "RGB",
        (left + cell * len(columns), top + len(denoises) * (cell + caption)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), pid, fill="black", font=font)
    for column, key in enumerate(columns):
        draw.text((left + column * cell + 4, 6), f"{key[0]} / {key[1]}", fill="black", font=font)
    for row, denoise in enumerate(denoises):
        y = top + row * (cell + caption)
        draw.text((8, y + cell // 2), f"d={denoise:g}", fill="black", font=font)
        for column, key in enumerate(columns):
            found = [
                r
                for r in mine
                if r["denoise"] == denoise and (r["input"], r["profile"]) == key
            ]
            if not found:
                continue
            image = Image.open(out / found[0]["file"]).convert("RGB").resize((cell, cell))
            sheet.paste(image, (left + column * cell, y))
    path = out / name
    sheet.save(path)
    return path


def parse_inputs(spec: str) -> List[tuple]:
    """`label=path,label=path`; a bare path is labelled by its stem."""
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        label, _, path = item.partition("=")
        if not path:
            path, label = label, Path(label).stem
        pairs.append((label, Path(path)))
    return pairs


DEFAULT_INPUTS = (
    f"a={DEFAULT_INPUT},"
    f"c={DEFAULT_INPUT.with_name('c_input.png')}"
)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--variant", default="after", help="label for this run")
    parser.add_argument("--presets", type=Path, default=PRESETS_TS)
    parser.add_argument("--inputs", default=DEFAULT_INPUTS, help="label=path,label=path")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs" / "experiments" / "2026-09-07-presets"
    )
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--only", default="", help="comma-separated preset ids")
    parser.add_argument("--denoises", default="", help="sweep these instead of each preset's own")
    parser.add_argument(
        "--profiles", default="fast", help="fast, quality, or `preset` for each preset's own"
    )
    parser.add_argument("--sheets", action="store_true", help="one sheet per preset")
    parser.add_argument("--sheet-prefix", default="sweep", help="filename prefix for the sheets")
    parser.add_argument("--sheet-only", action="store_true", help="draw sheets, generate nothing")
    parser.add_argument(
        "--rude", action="store_true", help="do not wait for other people's generations"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / args.results
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
        denoises = [float(d) for d in args.denoises.split(",") if d.strip()]
        profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
        # Resolved here rather than at generation time: a job with an unknown
        # denoise cannot be compared against what has already been rendered,
        # and the resume check ran first.
        jobs = [
            {
                "preset": preset,
                "denoise": denoise if denoise is not None else preset["denoise"],
                "profile": preset["profile"] if profile == "preset" else profile,
            }
            for profile in profiles
            for denoise in (denoises or [None])
            for preset in presets
        ]
        def _grid(value: Optional[float]) -> Optional[float]:
            return None if value is None else round(value, 3)

        def key(record: Dict[str, Any]) -> tuple:
            return (
                record["variant"],
                record["input"],
                record["id"],
                _grid(record["denoise"]),
                record["profile"],
            )

        for label, drawing in parse_inputs(args.inputs):
            have = {key(r) for r in existing}
            todo = [
                job
                for job in jobs
                if (
                    args.variant,
                    label,
                    job["preset"]["id"],
                    _grid(job["denoise"]),
                    job["profile"],
                )
                not in have
            ]
            if not todo:
                print(f"input {label}: nothing to do", flush=True)
                continue
            print(f"{len(todo)} generations on input {label}", flush=True)
            kept = list(existing)

            def save(records: List[Dict[str, Any]], kept: List[Dict[str, Any]] = kept) -> None:
                results_path.write_text(
                    json.dumps(kept + records, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            fresh = await run_jobs(
                args.server,
                todo,
                drawing,
                label,
                args.out,
                args.seed,
                args.variant,
                not args.rude,
                save,
            )
            existing = kept + fresh

    if args.sheets:
        for pid in sorted({r["id"] for r in existing}):
            name = f"{args.sheet_prefix}_{pid}.png"
            print(f"sheet: {sweep_sheet(existing, args.out, pid, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
