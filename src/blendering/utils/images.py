"""Helpers for handling viewport screenshots."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def decode_b64_png(b64: str) -> bytes:
    """Decode a base64 PNG (with or without data-url prefix) into raw bytes."""
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def encode_b64_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    """Encode raw image bytes as an OpenAI-style data URL."""
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def save_screenshot(image_bytes: bytes, directory: Path) -> Path:
    """Save bytes to a uniquely-numbered PNG; return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    n = len(list(directory.glob("step-*.png")))
    out = directory / f"step-{n:03d}.png"
    out.write_bytes(image_bytes)
    return out


def thumbnail_bytes(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """Downscale a screenshot so we don't blow vision-model token budgets."""
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
