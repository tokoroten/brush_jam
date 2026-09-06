#!/usr/bin/env python3
"""How long the AI input takes to render, from scratch and incrementally.

    uv run --project apps/brushjam python apps/brushjam/scripts/bench_render.py
    ... --strokes 4000 --canvas 1024 --batch 20

The AI input used to be rendered from the whole stroke log every time. That is
free at a few hundred strokes and most of a second at a few thousand, which is
what a 30-minute soak found: throughput fell from 41 generations a minute to 11
with a backend that returned instantly, because every generation redrew every
stroke anybody had ever made.

This measures both halves of that:

  from scratch   one render of the whole log, the way it used to work
  incremental    the same log, rendered `batch` strokes at a time, with the
                 layer raster kept between renders (raster.py, LayerCacheStore)

`batch` is the interesting knob: it is how many strokes land between two
generations. Twenty is a busy room; one is somebody drawing slowly.

No GPU, no model, no server - this is Pillow and numpy.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brushjam.raster import LAYER_CACHE, render_crop_input  # noqa: E402
from brushjam.room import (  # noqa: E402
    RoomLimits,
    apply_client_message,
    capture_render_snapshot,
    create_room,
    join_member,
    sorted_layers,
)
from brushjam.validate import validate_client_message  # noqa: E402

ROOM_ID = "benchxx"


def build_room(canvas: int):
    state = create_room(
        ROOM_ID, 0.55, canvas, canvas, True, "fast", RoomLimits(), seed=7
    )
    user = join_member(state, "Bench", "tok-bench-0001")["userId"]
    return state, user


def send(state, user: str, msg: Dict[str, Any]) -> None:
    validated = validate_client_message(msg)
    if not validated.ok:  # pragma: no cover - a bug in this script, not the server
        raise SystemExit(f"the bench built an invalid message: {validated.error}")
    apply_client_message(state, user, validated.msg)


def commit_stroke(state, user: str, rng: random.Random, layer_id: str, n: int, canvas: int) -> None:
    """One stroke of the shape a person actually draws: a short run of points."""
    x, y = rng.uniform(0, canvas), rng.uniform(0, canvas)
    points: List[Dict[str, float]] = []
    for _ in range(rng.randint(4, 10)):
        x = min(canvas - 1, max(0.0, x + rng.uniform(-40, 40)))
        y = min(canvas - 1, max(0.0, y + rng.uniform(-40, 40)))
        points.append({"x": x, "y": y, "p": rng.uniform(0.4, 1.0)})
    stroke = {
        "id": f"b{n:05d}",
        "layerId": layer_id,
        "tool": rng.choices(["pen", "noise", "eraser"], weights=[8, 1, 1])[0],
        "color": rng.choice(["#1b1b1b", "#c0392b", "#2980b9", "#27ae60"]),
        "width": rng.randint(2, 24),
        "alpha": rng.choice([1.0, 1.0, 0.5]),
        "points": points,
    }
    send(state, user, {"t": "stroke_start", "stroke": stroke})
    send(state, user, {"t": "stroke_end", "strokeId": stroke["id"], "points": []})


def render(state, canvas: int, size: int, room_id) -> float:
    crop = {"x": 0, "y": 0, "width": canvas, "height": canvas}
    started = time.perf_counter()
    render_crop_input(capture_render_snapshot(state), crop, size, room_id=room_id)
    return (time.perf_counter() - started) * 1000.0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strokes", type=int, default=4000)
    parser.add_argument("--canvas", type=int, default=1024)
    parser.add_argument("--size", type=int, default=768, help="generation size")
    parser.add_argument("--batch", type=int, default=20, help="strokes between renders")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    state, user = build_room(args.canvas)
    layer_id = sorted_layers(state)[0]["id"]

    print(f"building {args.strokes} strokes on a {args.canvas} canvas...", flush=True)
    built = time.perf_counter()
    for n in range(args.strokes):
        commit_stroke(state, user, rng, layer_id, n, args.canvas)
    print(f"  {(time.perf_counter() - built):.1f}s in the reducer\n")

    LAYER_CACHE.clear()
    scratch = [render(state, args.canvas, args.size, None) for _ in range(3)]
    print(f"from scratch   {statistics.median(scratch):8.0f} ms   "
          f"(every one of {args.strokes} strokes, {len(scratch)} runs)")

    # Incremental, replayed the way a room actually fills up: render, add a
    # batch, render again. The first render is a full one, so it is reported
    # separately from the steady state.
    LAYER_CACHE.clear()
    replay, replay_user = build_room(args.canvas)
    replay_layer = sorted_layers(replay)[0]["id"]
    rng = random.Random(args.seed)
    first = render(replay, args.canvas, args.size, ROOM_ID)
    steady: List[float] = []
    for n in range(args.strokes):
        commit_stroke(replay, replay_user, rng, replay_layer, n, args.canvas)
        if (n + 1) % args.batch == 0:
            steady.append(render(replay, args.canvas, args.size, ROOM_ID))
    if not steady:  # pragma: no cover - only with --batch > --strokes
        steady = [render(replay, args.canvas, args.size, ROOM_ID)]

    print(f"incremental    {statistics.median(steady):8.0f} ms   "
          f"(median of {len(steady)} renders, {args.batch} new strokes each)")
    print(f"  first render {first:8.0f} ms   (nothing cached yet)")
    print(f"  last render  {steady[-1]:8.0f} ms   (at {args.strokes} strokes)")
    print(f"  cache        {LAYER_CACHE.stats()['bytes'] / 1e6:8.1f} MB")

    ratio = statistics.median(scratch) / max(0.001, statistics.median(steady))
    print(f"\n{ratio:.1f}x faster at {args.strokes} strokes, "
          f"and it stops growing with the log rather than tracking it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
