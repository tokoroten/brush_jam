"""Header probe and structural validation for uploads. Port of
the retired Node server's src/imageInfo.ts.

The structural walk exists because a decoder handed a file with a valid header
and no image data can do something much worse than raise (the Node server saw
`@napi-rs/canvas` segfault on exactly that). Pillow is better behaved, but the
same preflight keeps decompression bombs away from the decoder and keeps the
error messages identical to the Node server's.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from .constants import CANVAS_SIZE

#: Uploads may not exceed the world size on either side, nor this pixel count.
MAX_IMAGE_SIDE = CANVAS_SIZE
MAX_IMAGE_PIXELS = 4096 * 4096


@dataclass(frozen=True)
class ImageInfo:
    mime: str
    width: int
    height: int


def _u32be(b: bytes, off: int) -> int:
    return struct.unpack_from(">I", b, off)[0]


def _u16be(b: bytes, off: int) -> int:
    return struct.unpack_from(">H", b, off)[0]


def probe_image(data: bytes) -> Optional[ImageInfo]:
    return _probe_png(data) or _probe_jpeg(data) or _probe_webp(data)


def _probe_png(b: bytes) -> Optional[ImageInfo]:
    if len(b) < 24:
        return None
    if _u32be(b, 0) != 0x89504E47 or _u32be(b, 4) != 0x0D0A1A0A:
        return None
    if b[12:16] != b"IHDR":
        return None
    return ImageInfo("image/png", _u32be(b, 16), _u32be(b, 20))


def _probe_jpeg(b: bytes) -> Optional[ImageInfo]:
    if len(b) < 4 or b[0] != 0xFF or b[1] != 0xD8:
        return None
    offset = 2
    while offset + 9 < len(b):
        if b[offset] != 0xFF:
            offset += 1
            continue
        marker = b[offset + 1]
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        length = _u16be(b, offset + 2)
        # SOF0..SOF15, excluding the DHT/JPG/DAC markers that share the range
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return ImageInfo("image/jpeg", _u16be(b, offset + 7), _u16be(b, offset + 5))
        if length < 2:
            return None
        offset += 2 + length
    return None


def _probe_webp(b: bytes) -> Optional[ImageInfo]:
    if len(b) < 30:
        return None
    if b[0:4] != b"RIFF" or b[8:12] != b"WEBP":
        return None
    fmt = b[12:16]
    if fmt == b"VP8 ":
        return ImageInfo(
            "image/webp",
            struct.unpack_from("<H", b, 26)[0] & 0x3FFF,
            struct.unpack_from("<H", b, 28)[0] & 0x3FFF,
        )
    if fmt == b"VP8L":
        bits = struct.unpack_from("<I", b, 21)[0]
        return ImageInfo("image/webp", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if fmt == b"VP8X":
        w = 1 + (b[24] | (b[25] << 8) | (b[26] << 16))
        h = 1 + (b[27] | (b[28] << 8) | (b[29] << 16))
        return ImageInfo("image/webp", w, h)
    return None


def validate_structure(b: bytes, mime: str) -> Optional[str]:
    if mime == "image/png":
        return _validate_png(b)
    if mime == "image/jpeg":
        return _validate_jpeg(b)
    return _validate_webp(b)


def _validate_png(b: bytes) -> Optional[str]:
    offset = 8  # past the signature
    saw_ihdr = False
    saw_idat = False
    while offset + 12 <= len(b):
        length = _u32be(b, offset)
        chunk_type = b[offset + 4 : offset + 8]
        if length > len(b):
            return "truncated PNG chunk"
        if offset == 8 and chunk_type != b"IHDR":
            return "PNG does not start with IHDR"
        if chunk_type == b"IHDR":
            saw_ihdr = True
        if chunk_type == b"IDAT":
            saw_idat = True
        if chunk_type == b"IEND":
            if not saw_ihdr or not saw_idat:
                return "PNG has no image data"
            return None
        offset += 12 + length  # length + type + data + crc
    return "incomplete PNG: no IEND chunk"


def _validate_jpeg(b: bytes) -> Optional[str]:
    if len(b) < 4:
        return "incomplete JPEG"
    if b[-2] != 0xFF or b[-1] != 0xD9:
        return "incomplete JPEG: no end-of-image marker"
    for i in range(2, len(b) - 1):
        if b[i] == 0xFF and b[i + 1] == 0xDA:
            return None  # start of scan
    return "JPEG has no image data"


def _validate_webp(b: bytes) -> Optional[str]:
    if len(b) < 12:
        return "incomplete WebP"
    declared = struct.unpack_from("<I", b, 4)[0]
    if declared + 8 > len(b):
        return "incomplete WebP: truncated RIFF payload"
    if declared < 12:
        return "WebP has no image data"
    return None


@dataclass(frozen=True)
class ImageCheck:
    ok: bool
    info: Optional[ImageInfo] = None
    error: Optional[str] = None


def check_image(data: bytes, declared_mime: str) -> ImageCheck:
    info = probe_image(data)
    if info is None:
        return ImageCheck(False, error="unrecognised image: expected PNG, JPEG or WebP")
    if info.mime != declared_mime:
        return ImageCheck(
            False,
            error=f"content-type {declared_mime} does not match the actual {info.mime}",
        )
    if info.width < 1 or info.height < 1:
        return ImageCheck(False, error="image has no pixels")
    if info.width > MAX_IMAGE_SIDE or info.height > MAX_IMAGE_SIDE:
        return ImageCheck(
            False,
            error=(
                f"image is larger than {MAX_IMAGE_SIDE}px on a side "
                f"({info.width}x{info.height})"
            ),
        )
    if info.width * info.height > MAX_IMAGE_PIXELS:
        return ImageCheck(False, error="image has too many pixels")
    structural = validate_structure(data, info.mime)
    if structural:
        return ImageCheck(False, error=structural)
    return ImageCheck(True, info=info)
