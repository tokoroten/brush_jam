"""Deterministic pixel noise for the noise pen.

Bit-exact port of packages/shared/src/noise.ts: the value of a pixel depends
only on the stroke id and the pixel's *world* coordinates, so a server-side
crop produces exactly the same pixels as the client's full-size layer.
"""

from __future__ import annotations

from typing import Tuple

MASK32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """Math.imul: 32-bit signed multiply, returned unsigned."""
    return (a * b) & MASK32


def fnv1a(text: str) -> int:
    """32-bit FNV-1a of a string, over the low byte of each UTF-16 code unit."""
    h = 0x811C9DC5
    for ch in text:
        # charCodeAt(i) & 0xff, i.e. UTF-16 code units, not code points.
        code = ord(ch)
        if code > 0xFFFF:  # astral: JS would see a surrogate pair
            code -= 0x10000
            for unit in (0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF)):
                h ^= unit & 0xFF
                h = _imul(h, 0x01000193)
            continue
        h ^= code & 0xFF
        h = _imul(h, 0x01000193)
    return h & MASK32


def _to_int32(x: int) -> int:
    """JS `x | 0`."""
    x &= MASK32
    return x - 0x1_0000_0000 if x >= 0x8000_0000 else x


def noise_hash(seed: int, x: int, y: int) -> int:
    """One 32-bit hash of (seed, x, y); the three low bytes become RGB."""
    h = (seed ^ _imul(_to_int32(x), 0x9E3779B1) ^ _imul(_to_int32(y), 0x85EBCA77)) & MASK32
    h = _imul(h ^ (h >> 15), 0x2C1B3C6D) & MASK32
    h = _imul(h ^ (h >> 12), 0x297A2D39) & MASK32
    return (h ^ (h >> 15)) & MASK32


def noise_rgb(seed: int, x: int, y: int) -> Tuple[int, int, int]:
    """Uniform RGB noise for one world pixel."""
    h = noise_hash(seed, x, y)
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)
