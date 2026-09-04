"""Sweep denoise at a fixed size and save one PNG per value."""
import sys, json, base64, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import b64, post, PROMPT, NEGATIVE
from make_sample import make_drawing
from PIL import Image

size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
out = Path("samples"); out.mkdir(exist_ok=True)
drawing = make_drawing(size)
for d in [0.55, 0.7, 0.85, 1.0]:
    t = time.perf_counter()
    r = post("http://127.0.0.1:8790/generate", {
        "image_b64": b64(drawing), "prompt": PROMPT, "negative_prompt": NEGATIVE,
        "denoise": d, "steps": 4, "seed": 42, "width": size, "height": size})
    Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).save(out / f"probe_{size}_d{d}.png")
    print(f"denoise {d}: {(time.perf_counter()-t)*1000:.0f} ms")
