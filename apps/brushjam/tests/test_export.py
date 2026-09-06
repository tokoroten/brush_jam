"""Taking a room home: the zip, the video, and the AVI writer underneath them.

The video half is checked twice - once by parsing the RIFF back in pure Python,
which runs everywhere, and once with ffprobe/ffmpeg where they are installed,
which is the only proof that a real decoder agrees with our reading of the
spec. The second is skipped rather than required: ffmpeg is a verification
tool here, never a runtime dependency.
"""

from __future__ import annotations

import io
import json
import shutil
import struct
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest
from PIL import Image
from starlette.testclient import TestClient

from brushjam.ai.backends.mock import MockBackend
from brushjam.app import create_app
from brushjam.avi import AviTooLarge, MjpegAviWriter
from brushjam.config import load_config
from brushjam.export import ExportBusy, ExportGuard, build_avi, build_zip
from brushjam.history import HistoryStore

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe are not on PATH"
)


def jpeg(size=(64, 64), colour=(200, 40, 40)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, format="JPEG", quality=90)
    return out.getvalue()


def entry(**kw) -> Dict[str, Any]:
    base = {
        "aiRevision": 3,
        "aiGeneration": 1,
        "time": 1_700_000_000_000,
        "prompt": "a hill",
        "negativePrompt": "blurry",
        "denoise": 0.8,
        "seed": 42,
        "profile": "fast",
        "aiResolution": 768,
        "latencyMs": 1800,
        "model": "novaAnimeXL_ilV190.safetensors",
        "lora": "dmd2_sdxl_4step_lora_fp16.safetensors",
    }
    base.update(kw)
    return base


def populate(
    tmp_path: Path, count: int = 3, *, inputs: bool = True, size=(64, 64)
) -> HistoryStore:
    store = HistoryStore(tmp_path / "history")
    for i in range(count):
        store.record(
            "abcd",
            jpeg(size, (20 * i, 40, 200)),
            entry(prompt=f"p{i}", seed=i),
            jpeg((32, 32), (10, 200, 10)) if inputs else None,
        )
    return store


# --------------------------------------------------------- the AVI writer


def parse_avi(data: bytes) -> Dict[str, Any]:
    """Read a RIFF AVI back the way a player would, and check it adds up."""
    assert data[:4] == b"RIFF" and data[8:12] == b"AVI "
    riff_size = struct.unpack_from("<I", data, 4)[0]
    assert riff_size == len(data) - 8, "the RIFF size must cover the whole file"

    out: Dict[str, Any] = {"frames": [], "index": []}
    at = 12
    while at + 8 <= len(data):
        fourcc = data[at : at + 4]
        size = struct.unpack_from("<I", data, at + 4)[0]
        body = at + 8
        if fourcc == b"LIST":
            kind = data[body : body + 4]
            if kind == b"movi":
                movi_at = body
                cursor = body + 4
                end = body + size
                while cursor + 8 <= end:
                    chunk = data[cursor : cursor + 4]
                    length = struct.unpack_from("<I", data, cursor + 4)[0]
                    assert chunk == b"00dc", chunk
                    out["frames"].append(
                        (cursor - movi_at, data[cursor + 8 : cursor + 8 + length])
                    )
                    cursor += 8 + length + (length & 1)
                assert cursor == end, "the movi list must end exactly on a chunk boundary"
            else:
                # hdrl: walk it the same way to reach avih and strh/strf.
                cursor = body + 4
                end = body + size
                while cursor + 8 <= end:
                    sub = data[cursor : cursor + 4]
                    length = struct.unpack_from("<I", data, cursor + 4)[0]
                    payload = data[cursor + 8 : cursor + 8 + length]
                    if sub == b"avih":
                        fields = struct.unpack_from("<10I", payload)
                        out["avih"] = {
                            "micros_per_frame": fields[0],
                            "max_bytes_per_sec": fields[1],
                            "flags": fields[3],
                            "total_frames": fields[4],
                            "streams": fields[6],
                            "suggested_buffer": fields[7],
                            "width": fields[8],
                            "height": fields[9],
                        }
                    elif sub == b"LIST" and payload[:4] == b"strl":
                        inner = cursor + 12
                        stop = cursor + 8 + length
                        while inner + 8 <= stop:
                            tag = data[inner : inner + 4]
                            n = struct.unpack_from("<I", data, inner + 4)[0]
                            chunk = data[inner + 8 : inner + 8 + n]
                            if tag == b"strh":
                                out["strh"] = {
                                    "type": chunk[0:4],
                                    "handler": chunk[4:8],
                                    "scale": struct.unpack_from("<I", chunk, 20)[0],
                                    "rate": struct.unpack_from("<I", chunk, 24)[0],
                                    "length": struct.unpack_from("<I", chunk, 32)[0],
                                    "suggested_buffer": struct.unpack_from("<I", chunk, 36)[0],
                                }
                            elif tag == b"strf":
                                out["strf"] = {
                                    "width": struct.unpack_from("<i", chunk, 4)[0],
                                    "height": struct.unpack_from("<i", chunk, 8)[0],
                                    "bit_count": struct.unpack_from("<H", chunk, 14)[0],
                                    "compression": chunk[16:20],
                                }
                            inner += 8 + n + (n & 1)
                    cursor += 8 + length + (length & 1)
        elif fourcc == b"idx1":
            for at_entry in range(body, body + size, 16):
                chunk_id, flags, offset, length = struct.unpack_from("<4sIII", data, at_entry)
                out["index"].append((chunk_id, flags, offset, length))
        at = body + size + (size & 1)
    return out


def test_the_avi_writer_produces_a_riff_that_reads_back(tmp_path: Path) -> None:
    frames = [jpeg((16, 16), (i * 20, 0, 0)) for i in range(5)]
    dest = tmp_path / "out.avi"
    with open(dest, "wb") as handle:
        with MjpegAviWriter(handle, 32, 16, 6) as writer:
            for frame in frames:
                writer.add_frame(frame)
    parsed = parse_avi(dest.read_bytes())

    assert parsed["avih"]["total_frames"] == 5
    assert parsed["avih"]["streams"] == 1
    assert parsed["avih"]["width"] == 32 and parsed["avih"]["height"] == 16
    assert parsed["avih"]["flags"] & 0x10, "AVIF_HASINDEX must be set: there is an idx1"
    assert parsed["avih"]["micros_per_frame"] == round(1_000_000 / 6)
    assert parsed["avih"]["max_bytes_per_sec"] > 0
    assert parsed["strh"]["type"] == b"vids" and parsed["strh"]["handler"] == b"MJPG"
    assert parsed["strh"]["rate"] == 6 and parsed["strh"]["scale"] == 1
    assert parsed["strh"]["length"] == 5
    assert parsed["strf"]["compression"] == b"MJPG"
    assert parsed["strf"]["bit_count"] == 24
    assert parsed["strf"]["width"] == 32 and parsed["strf"]["height"] == 16

    # The payloads come back byte for byte, in order.
    assert [payload for _offset, payload in parsed["frames"]] == frames
    # And the index points at them: every entry is a keyframe, and its offset
    # is where the chunk actually is.
    assert len(parsed["index"]) == 5
    for (chunk_id, flags, offset, length), (real_offset, payload) in zip(
        parsed["index"], parsed["frames"]
    ):
        assert chunk_id == b"00dc" and flags & 0x10
        assert offset == real_offset and length == len(payload)
    # A buffer big enough for the largest chunk, which is what a desktop player
    # allocates from.
    biggest = max(len(payload) for _offset, payload in parsed["frames"])
    assert parsed["avih"]["suggested_buffer"] >= biggest
    assert parsed["strh"]["suggested_buffer"] == parsed["avih"]["suggested_buffer"]


def test_an_odd_length_frame_is_padded_to_an_even_offset(tmp_path: Path) -> None:
    odd = b"\xff\xd8\xff" + b"a" * 12  # 15 bytes, deliberately not a real JPEG
    dest = tmp_path / "odd.avi"
    with open(dest, "wb") as handle:
        with MjpegAviWriter(handle, 8, 8, 4) as writer:
            writer.add_frame(odd)
            writer.add_frame(odd)
    parsed = parse_avi(dest.read_bytes())
    assert [payload for _offset, payload in parsed["frames"]] == [odd, odd]
    assert all(offset % 2 == 0 for offset, _payload in parsed["frames"])


def test_the_writer_refuses_to_grow_past_its_cap(tmp_path: Path) -> None:
    dest = tmp_path / "big.avi"
    with open(dest, "wb") as handle:
        writer = MjpegAviWriter(handle, 8, 8, 4, max_bytes=2000)
        writer.add_frame(b"\xff\xd8" + b"x" * 800)
        with pytest.raises(AviTooLarge):
            writer.add_frame(b"\xff\xd8" + b"x" * 2000)


# ------------------------------------------------------------------ the zip


def test_the_zip_holds_every_frame_and_a_manifest(tmp_path: Path) -> None:
    store = populate(tmp_path, 3)
    dest = tmp_path / "out.zip"
    assert build_zip(store, "abcd", dest, canvas_size=1024) == 3
    with zipfile.ZipFile(dest) as archive:
        names = archive.namelist()
        assert names == [
            "draw_00000.jpg",
            "gen_00000.jpg",
            "draw_00001.jpg",
            "gen_00001.jpg",
            "draw_00002.jpg",
            "gen_00002.jpg",
            "manifest.json",
        ]
        # Stored, not deflated: these are JPEGs.
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
        manifest = json.loads(archive.read("manifest.json"))
        # The bytes are the stored ones, not a re-encode.
        assert archive.read("gen_00001.jpg") == store.read("abcd", 1)
        assert archive.read("draw_00001.jpg") == store.read("abcd", 1, "in")
    assert manifest["roomId"] == "abcd" and manifest["canvasSize"] == 1024
    assert manifest["exportedAt"] > 0
    assert [f["n"] for f in manifest["frames"]] == [0, 1, 2]
    first = manifest["frames"][0]
    assert first["draw"] == "draw_00000.jpg" and first["gen"] == "gen_00000.jpg"
    assert first["prompt"] == "p0" and first["seed"] == 0
    assert first["negativePrompt"] == "blurry" and first["profile"] == "fast"
    assert first["denoise"] == 0.8 and first["aiResolution"] == 768
    assert first["latencyMs"] == 1800 and first["aiRevision"] == 3
    assert first["aiGeneration"] == 1 and first["time"] == 1_700_000_000_000
    assert first["model"] == "novaAnimeXL_ilV190.safetensors"
    assert first["lora"] == "dmd2_sdxl_4step_lora_fp16.safetensors"


def test_an_entry_from_before_the_model_was_recorded_is_null(tmp_path: Path) -> None:
    """A room that was in use before this shipped. Every frame has the same
    shape; the fields that were never written are null rather than missing."""
    store = HistoryStore(tmp_path / "history")
    old = entry()
    del old["model"], old["lora"]
    store.record("abcd", jpeg(), old)
    dest = tmp_path / "out.zip"
    build_zip(store, "abcd", dest, canvas_size=512)
    with zipfile.ZipFile(dest) as archive:
        frame = json.loads(archive.read("manifest.json"))["frames"][0]
    assert frame["model"] is None and frame["lora"] is None and frame["draw"] is None
    assert frame["gen"] == "gen_00000.jpg" and frame["prompt"] == "a hill"


def test_an_entry_with_no_input_has_a_null_draw(tmp_path: Path) -> None:
    store = populate(tmp_path, 2, inputs=False)
    dest = tmp_path / "out.zip"
    build_zip(store, "abcd", dest, canvas_size=512)
    with zipfile.ZipFile(dest) as archive:
        assert "draw_00000.jpg" not in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
    assert [f["draw"] for f in manifest["frames"]] == [None, None]


# ------------------------------------------------------------------ the avi


def test_the_video_is_input_left_and_result_right(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    store.record("abcd", jpeg((64, 64), (255, 0, 0)), entry(), jpeg((32, 32), (0, 0, 255)))
    store.record("abcd", jpeg((64, 64), (255, 0, 0)), entry())  # no input
    dest = tmp_path / "out.avi"
    frames, width, height = build_avi(store, "abcd", dest, fps=4)
    assert (frames, width, height) == (2, 128, 64)
    parsed = parse_avi(dest.read_bytes())
    assert parsed["avih"]["width"] == 128 and parsed["avih"]["height"] == 64

    with Image.open(io.BytesIO(parsed["frames"][0][1])) as first:
        assert first.format == "JPEG" and first.size == (128, 64)
        rgb = first.convert("RGB")
        assert rgb.getpixel((16, 32))[2] > 200  # the input, blue, on the left
        assert rgb.getpixel((96, 32))[0] > 200  # the result, red, on the right
    with Image.open(io.BytesIO(parsed["frames"][1][1])) as second:
        rgb = second.convert("RGB")
        # No input: white, the background the AI input itself is drawn on.
        assert min(rgb.getpixel((16, 32))) > 240
        assert rgb.getpixel((96, 32))[0] > 200


def test_the_width_parameter_scales_both_halves(tmp_path: Path) -> None:
    store = populate(tmp_path, 1, size=(64, 64))
    dest = tmp_path / "out.avi"
    frames, width, height = build_avi(store, "abcd", dest, fps=4, width=400)
    assert (frames, width, height) == (1, 400, 200)


def test_too_many_entries_is_refused_rather_than_made(tmp_path: Path) -> None:
    from brushjam.export import ExportTooLarge

    store = populate(tmp_path, 3)
    with pytest.raises(ExportTooLarge):
        build_avi(store, "abcd", tmp_path / "out.avi", max_frames=2)


# ---------------------------------------------------------------- the guard


def test_one_export_per_room_and_a_global_limit() -> None:
    guard = ExportGuard(limit=2)
    with guard.claim("aaaa"):
        with pytest.raises(ExportBusy):
            with guard.claim("aaaa"):
                pass
        with guard.claim("bbbb"):
            with pytest.raises(ExportBusy):
                with guard.claim("cccc"):
                    pass
    assert guard.running == 0


def test_a_failed_export_releases_its_claim() -> None:
    guard = ExportGuard()
    with pytest.raises(ValueError):
        with guard.claim("aaaa"):
            raise ValueError("boom")
    assert guard.running == 0
    with guard.claim("aaaa"):
        pass


# --------------------------------------------------------------- the routes


def app_client(tmp_path: Path, **env) -> TestClient:
    settings = {
        "AI_BACKEND": "mock",
        "CANVAS_SIZE": "512",
        "AI_WINDOW": "512",
        "AI_DEBOUNCE_MS": "10",
        "HISTORY_DIR": str(tmp_path / "history"),
    }
    settings.update(env)
    return TestClient(create_app(load_config(settings), MockBackend(latency_ms=0)))


def test_the_export_routes_serve_a_room_that_has_history(tmp_path: Path) -> None:
    populate(tmp_path, 3)
    with app_client(tmp_path) as client:
        zipped = client.get("/rooms/abcd/history.zip")
        assert zipped.status_code == 200
        assert zipped.headers["content-type"] == "application/zip"
        assert zipped.headers["cache-control"] == "no-store"
        assert zipped.headers["content-disposition"] == (
            'attachment; filename="brushjam-abcd-history.zip"'
        )
        with zipfile.ZipFile(io.BytesIO(zipped.content)) as archive:
            assert "manifest.json" in archive.namelist()
            assert json.loads(archive.read("manifest.json"))["canvasSize"] == 512

        video = client.get("/rooms/abcd/history.avi?fps=8&width=300")
        assert video.status_code == 200
        assert video.headers["content-type"] == "video/x-msvideo"
        assert video.headers["cache-control"] == "no-store"
        assert video.headers["content-disposition"] == (
            'attachment; filename="brushjam-abcd-history.avi"'
        )
        parsed = parse_avi(video.content)
        assert parsed["strh"]["rate"] == 8 and parsed["avih"]["total_frames"] == 3
        assert parsed["avih"]["width"] == 300

    # Nothing is left behind in the history directory.
    leftovers = list((tmp_path / "history" / ".exports").glob("*"))
    assert leftovers == [], leftovers


def test_silly_parameters_are_clamped_rather_than_refused(tmp_path: Path) -> None:
    populate(tmp_path, 1)
    with app_client(tmp_path) as client:
        parsed = parse_avi(client.get("/rooms/abcd/history.avi?fps=9999").content)
        assert parsed["strh"]["rate"] == 30
        parsed = parse_avi(client.get("/rooms/abcd/history.avi?fps=0&width=1").content)
        assert parsed["strh"]["rate"] == 1 and parsed["avih"]["width"] == 256


def test_a_room_with_no_history_is_a_404(tmp_path: Path) -> None:
    with app_client(tmp_path) as client:
        assert client.get("/rooms/abcd/history.zip").status_code == 404
        assert client.get("/rooms/abcd/history.avi").status_code == 404
        # And an id that is not an id never reaches the store.
        assert client.get("/rooms/AB/history.zip").status_code == 404


def test_exports_are_off_when_the_history_is(tmp_path: Path) -> None:
    populate(tmp_path, 1)
    with app_client(tmp_path, HISTORY_ENABLED="0") as client:
        assert client.get("/rooms/abcd/history.zip").status_code == 404
        assert client.get("/rooms/abcd/history.avi").status_code == 404


def test_a_second_export_of_the_same_room_is_429(tmp_path: Path) -> None:
    populate(tmp_path, 2)
    with app_client(tmp_path) as client:
        # Hold the claim from outside, which is exactly what a build in flight
        # does, and ask for another one.
        guard: ExportGuard = client.app.state.exports
        with guard.claim("abcd"):
            busy = client.get("/rooms/abcd/history.zip")
        assert busy.status_code == 429
        assert int(busy.headers["retry-after"]) > 0
        # ... and once it is released the export works again.
        assert client.get("/rooms/abcd/history.zip").status_code == 200


def test_too_many_entries_for_a_video_is_413(tmp_path: Path) -> None:
    populate(tmp_path, 3)
    with app_client(tmp_path, HISTORY_EXPORT_MAX_FRAMES="2") as client:
        refused = client.get("/rooms/abcd/history.avi")
        assert refused.status_code == 413
        assert "zip" in refused.json()["error"]
        # The zip has no such limit.
        assert client.get("/rooms/abcd/history.zip").status_code == 200


def test_a_generation_end_to_end_can_be_exported(tmp_path: Path) -> None:
    with app_client(tmp_path) as client:
        room_id = client.post("/api/rooms").json()["roomId"]
        with client.websocket_connect(f"/ws/rooms/{room_id}?name=a") as socket:
            deadline = time.time() + 5
            while time.time() < deadline:
                if json.loads(socket.receive_text())["t"] == "snapshot":
                    break
            socket.send_text(json.dumps({"t": "set_prompt", "prompt": "a hill"}))
            deadline = time.time() + 5
            while time.time() < deadline:
                if json.loads(socket.receive_text())["t"] == "ai_result":
                    break
        video = client.get(f"/rooms/{room_id}/history.avi")
        assert video.status_code == 200
        parsed = parse_avi(video.content)
        # A 512 canvas and a 512 generation: two 512 halves side by side.
        assert parsed["avih"]["width"] == 1024 and parsed["avih"]["height"] == 512
        assert parsed["avih"]["total_frames"] == 1


# ------------------------------------------------------- what a decoder says


@needs_ffmpeg
def test_ffprobe_agrees_it_is_an_mjpeg_stream(tmp_path: Path) -> None:
    store = populate(tmp_path, 7, size=(96, 96))
    dest = tmp_path / "out.avi"
    build_avi(store, "abcd", dest, fps=6)

    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate,avg_frame_rate",
            "-show_entries",
            "format=format_name",
            "-of",
            "json",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(probe.stdout)
    stream = info["streams"][0]
    assert stream["codec_name"] == "mjpeg"
    assert (stream["width"], stream["height"]) == (192, 96)
    assert int(stream["nb_frames"]) == 7
    assert stream["r_frame_rate"] == "6/1" and stream["avg_frame_rate"] == "6/1"
    assert "avi" in info["format"]["format_name"]

    decoded = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(dest), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    assert decoded.returncode == 0, decoded.stderr
    assert decoded.stderr.strip() == "", decoded.stderr
