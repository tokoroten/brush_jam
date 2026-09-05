"""Mirror of the retired Node server's src/ids.ts."""

from __future__ import annotations

import secrets

ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"

MEMBER_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
]


def short_id(length: int = 8) -> str:
    raw = secrets.token_bytes(length)
    return "".join(ALPHABET[b % len(ALPHABET)] for b in raw)


def member_color(index: int) -> str:
    return MEMBER_COLORS[index % len(MEMBER_COLORS)]
