"""A Motion JPEG AVI writer, in about a page of struct calls.

A room's history is already a pile of JPEGs, so the cheapest possible video of
it is a container that stores JPEGs verbatim: Motion JPEG. AVI is the format
that does that and that every player on a Windows desktop opens without being
told anything - Media Player, VLC, the Photos app - which is the point, because
the thing being exported is a keepsake, not a delivery master.

Writing it by hand rather than shelling out to ffmpeg is deliberate. ffmpeg is
on this machine, but it must not be a *runtime* dependency of the server: an
export that fails on a box without it would be a feature that works here and
nowhere else. The whole format used here is four chunks:

    RIFF....AVI
      LIST....hdrl
        avih....        the main header: frame period, size, frame count
        LIST....strl
          strh....      the one video stream: 'vids' / 'MJPG', rate and length
          strf....      a BITMAPINFOHEADER whose biCompression is 'MJPG'
      LIST....movi
        00dc....        one JPEG per frame, padded to an even length
      idx1....          16 bytes per frame: id, flags, offset, length

Everything that has to be counted first - the frame count, the RIFF and LIST
sizes, the largest frame - is written as a placeholder and patched by `close`,
so frames stream to the file as they are encoded rather than being held in
memory. That means the handle must be seekable: this writes to a temp file, and
that is the reason.
"""

from __future__ import annotations

import struct
from typing import BinaryIO, List

#: The whole file is little-endian, and every chunk is `fourcc, size, payload`
#: with the payload padded to an even length.
_U32 = struct.Struct("<I")

AVIF_HASINDEX = 0x00000010
AVIF_ISINTERLEAVED = 0x00000100
AVIIF_KEYFRAME = 0x00000010

#: Every MJPEG frame stands alone, so every frame is a keyframe and the stream
#: id is the only one there is.
STREAM_CHUNK_ID = b"00dc"

#: Plain AVI addresses the movi list with 32-bit offsets, so a file that grows
#: past 2 GiB needs OpenDML's second index - which is exactly the kind of thing
#: that plays here and not in the player somebody actually has. The writer
#: refuses instead, well short of the limit.
MAX_FILE_BYTES = 1_500_000_000


class AviTooLarge(RuntimeError):
    """The file would have needed OpenDML. Raised before it is written."""


class MjpegAviWriter:
    """One video stream, one JPEG per frame. Not thread safe by design: it is
    driven by the single worker thread that is building one export."""

    def __init__(
        self,
        handle: BinaryIO,
        width: int,
        height: int,
        fps: int,
        *,
        max_bytes: int = MAX_FILE_BYTES,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("an AVI frame needs a positive size")
        if fps <= 0:
            raise ValueError("an AVI needs a positive frame rate")
        self.handle = handle
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.max_bytes = int(max_bytes)
        self.frames = 0
        self._largest = 0
        #: (offset from the `movi` fourcc, payload length) per frame, which is
        #: all `idx1` is.
        self._index: List[tuple] = []
        self._closed = False
        self._write_header()

    # -- writing ----------------------------------------------------------

    def _write_header(self) -> None:
        self.handle.write(b"RIFF")
        self._riff_size_at = self.handle.tell()
        self.handle.write(_U32.pack(0))  # patched by close
        self.handle.write(b"AVI ")

        hdrl = self._avih() + self._strl()
        self.handle.write(b"LIST" + _U32.pack(len(hdrl) + 4) + b"hdrl" + hdrl)

        self.handle.write(b"LIST")
        self._movi_size_at = self.handle.tell()
        self.handle.write(_U32.pack(0))  # patched by close
        #: idx1 offsets are measured from here, the `movi` fourcc itself, which
        #: is what every player and every muxer means by them.
        self._movi_at = self.handle.tell()
        self.handle.write(b"movi")

    def _avih(self) -> bytes:
        body = struct.pack(
            "<IIIIIIIIII16x",
            round(1_000_000 / self.fps),  # dwMicroSecPerFrame
            0,  # dwMaxBytesPerSec, patched
            0,  # dwPaddingGranularity
            AVIF_HASINDEX | AVIF_ISINTERLEAVED,
            0,  # dwTotalFrames, patched
            0,  # dwInitialFrames
            1,  # dwStreams
            0,  # dwSuggestedBufferSize, patched
            self.width,
            self.height,
        )
        # `LIST....hdrl` is 12 bytes, then this chunk's own 8-byte header.
        self._avih_body_at = self.handle.tell() + 12 + 8
        return b"avih" + _U32.pack(len(body)) + body

    def _strl(self) -> bytes:
        strh_body = struct.pack(
            "<4s4sIHHIIIIIIIIhhhh",
            b"vids",
            b"MJPG",
            0,  # dwFlags
            0,  # wPriority
            0,  # wLanguage
            0,  # dwInitialFrames
            1,  # dwScale
            self.fps,  # dwRate: scale/rate = one frame period
            0,  # dwStart
            0,  # dwLength, patched
            0,  # dwSuggestedBufferSize, patched
            0xFFFFFFFF,  # dwQuality: "use the default"
            0,  # dwSampleSize: 0 = one sample per chunk
            0,
            0,
            self.width,
            self.height,
        )
        strf_body = struct.pack(
            "<IiiHH4sIiiII",
            40,  # biSize
            self.width,
            self.height,
            1,  # biPlanes
            24,  # biBitCount
            b"MJPG",  # biCompression
            self.width * self.height * 3,  # biSizeImage
            0,
            0,
            0,
            0,
        )
        strh = b"strh" + _U32.pack(len(strh_body)) + strh_body
        strf = b"strf" + _U32.pack(len(strf_body)) + strf_body
        # Offsets of the two fields `close` patches, relative to the file: the
        # RIFF header (12) + the hdrl LIST header (12) + avih (8 + 56) + the
        # strl LIST header (12) + this chunk's header (8).
        self._strh_body_at = 12 + 12 + 8 + 56 + 12 + 8
        return b"LIST" + _U32.pack(len(strh) + len(strf) + 4) + b"strl" + strh + strf

    def add_frame(self, jpeg: bytes) -> None:
        """Append one already-encoded JPEG."""
        if self._closed:  # pragma: no cover - defensive
            raise RuntimeError("this AVI is closed")
        if not jpeg:
            raise ValueError("an empty frame is not a frame")
        offset = self.handle.tell() - self._movi_at
        padded = len(jpeg) + (len(jpeg) & 1)
        # The index and the trailing idx1 chunk are 16 bytes a frame on top of
        # the payload; counted here so the refusal happens before the write
        # rather than after the file is already too big to fix.
        projected = self.handle.tell() + 8 + padded + (self.frames + 1) * 16 + 8
        if projected > self.max_bytes:
            raise AviTooLarge(
                f"the video would be about {projected // (1024 * 1024)} MB, "
                f"past the {self.max_bytes // (1024 * 1024)} MB this writer allows"
            )
        self.handle.write(STREAM_CHUNK_ID + _U32.pack(len(jpeg)) + jpeg)
        if len(jpeg) & 1:
            # RIFF chunks start on even offsets. The pad byte is not part of
            # the chunk's declared length and is not part of the JPEG.
            self.handle.write(b"\x00")
        self._index.append((offset, len(jpeg)))
        self._largest = max(self._largest, padded + 8)
        self.frames += 1

    def close(self) -> None:
        """Write `idx1` and patch every count the header could not know."""
        if self._closed:
            return
        self._closed = True
        if self.frames == 0:
            raise ValueError("an AVI with no frames is not a video")
        movi_end = self.handle.tell()
        index = b"".join(
            struct.pack("<4sIII", STREAM_CHUNK_ID, AVIIF_KEYFRAME, offset, length)
            for offset, length in self._index
        )
        self.handle.write(b"idx1" + _U32.pack(len(index)) + index)
        end = self.handle.tell()

        # `movi` covers its own fourcc plus every chunk in it.
        self._patch(self._movi_size_at, movi_end - self._movi_at)
        # RIFF covers everything after its own 8-byte header.
        self._patch(self._riff_size_at, end - self._riff_size_at - 4)

        duration = self.frames / float(self.fps)
        total = sum(length for _offset, length in self._index)
        # Conservative on purpose: players size their read buffer from
        # dwSuggestedBufferSize, and a value that is too small is the classic
        # way a hand-written AVI plays in ffmpeg and stutters in a desktop
        # player. The largest chunk, rounded up, is always enough.
        suggested = self._largest + 16
        self._patch(self._avih_body_at + 4, int(total / duration) + 1)  # dwMaxBytesPerSec
        self._patch(self._avih_body_at + 16, self.frames)  # dwTotalFrames
        self._patch(self._avih_body_at + 28, suggested)  # dwSuggestedBufferSize
        self._patch(self._strh_body_at + 32, self.frames)  # dwLength
        self._patch(self._strh_body_at + 36, suggested)  # dwSuggestedBufferSize
        self.handle.seek(end)

    def _patch(self, at: int, value: int) -> None:
        self.handle.seek(at)
        self.handle.write(_U32.pack(value & 0xFFFFFFFF))

    def __enter__(self) -> "MjpegAviWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
