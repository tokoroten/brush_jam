"""PNG in and out of the JSON body.

The HTTP surface is the worker's own business, so these live here rather than
in the shared pipeline: the pipeline deals in PIL images and knows nothing
about how a caller chose to put one in a request.
"""

from __future__ import annotations

import base64
import io

from PIL import Image


def strip_data_url(data: str) -> str:
    """`data:image/png;base64,AAA...` -> `AAA...`, and anything else unchanged.

    Browsers hand out data URLs and it costs nothing to accept one.
    """
    if data.startswith("data:"):
        _, _, tail = data.partition(",")
        return tail
    return data


def decode_png_b64(data: str) -> Image.Image:
    raw = base64.b64decode(strip_data_url(data), validate=False)
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


def encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
