"""Upload preflight (apps/server/test/imageInfo.test.ts)."""

from __future__ import annotations

import io
import struct

from PIL import Image

from brushjam.imageinfo import MAX_IMAGE_SIDE, check_image, probe_image, validate_structure


def _png(width: int = 8, height: int = 4) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (7, 8, 9)).save(out, format="PNG")
    return out.getvalue()


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (12, 6), (7, 8, 9)).save(out, format="JPEG")
    return out.getvalue()


def _webp() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (10, 5), (7, 8, 9)).save(out, format="WEBP")
    return out.getvalue()


def test_probe_reads_the_header_dimensions() -> None:
    assert probe_image(_png(8, 4)) == probe_image(_png(8, 4))
    png = probe_image(_png(8, 4))
    assert (png.mime, png.width, png.height) == ("image/png", 8, 4)
    jpeg = probe_image(_jpeg())
    assert (jpeg.mime, jpeg.width, jpeg.height) == ("image/jpeg", 12, 6)
    webp = probe_image(_webp())
    assert (webp.mime, webp.width, webp.height) == ("image/webp", 10, 5)
    assert probe_image(b"not an image at all, really not") is None


def test_a_header_only_png_is_rejected_before_any_decode() -> None:
    """The reason this exists: a file with a valid signature and IHDR and no
    image data made the Node decoder segfault."""
    header = _png()[:33]
    assert validate_structure(header, "image/png") is not None
    result = check_image(header, "image/png")
    assert not result.ok and "PNG" in result.error


def test_a_lying_content_type_is_rejected() -> None:
    result = check_image(_png(), "image/jpeg")
    assert not result.ok
    assert result.error == "content-type image/jpeg does not match the actual image/png"


def test_a_giant_declared_size_is_rejected_without_decoding() -> None:
    data = bytearray(_png())
    # Rewrite IHDR width/height to 50000 x 50000: a decompression bomb claim.
    struct.pack_into(">I", data, 16, 50_000)
    struct.pack_into(">I", data, 20, 50_000)
    result = check_image(bytes(data), "image/png")
    assert not result.ok
    assert f"larger than {MAX_IMAGE_SIDE}px" in result.error


def test_a_good_file_passes() -> None:
    result = check_image(_png(64, 32), "image/png")
    assert result.ok and (result.info.width, result.info.height) == (64, 32)
    assert check_image(_jpeg(), "image/jpeg").ok
    assert check_image(_webp(), "image/webp").ok


def test_a_truncated_jpeg_is_rejected() -> None:
    result = check_image(_jpeg()[:-2], "image/jpeg")
    assert not result.ok and "JPEG" in result.error
