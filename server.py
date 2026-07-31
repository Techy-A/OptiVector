"""
SVG AI Vectorize — web server.

Serves the pitch UI (pitch/index.html) and a real conversion endpoint that runs
the VTracer engine in vectorize.py: image upload -> scalable SVG + stats.

Run:
    python3 -m pip install fastapi uvicorn python-multipart vtracer resvg-py
    python3 server.py            # -> http://localhost:8000
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse

import xml.etree.ElementTree as ET

from PIL import Image

import vectorize as vz
import vectorize_layered as vl

# Decompression-bomb guard: a 25 MB PNG/WebP can decode to tens of millions of
# pixels. PIL only warns at its default threshold; raise hard well above our cap.
Image.MAX_IMAGE_PIXELS = 40_000_000
MAX_PIXELS = 25_000_000   # ~25 MP (e.g. 5000×5000) — reject larger before any work

app = FastAPI(title="SVG AI Vectorize")


def _normalize_to_png(raw_path, dst_path):
    """Any upload (jpg/png/webp/RGBA/odd-extension) → clean RGB PNG.
    VTracer picks its decoder by extension and chokes on missing/odd ones;
    PIL reads anything. Transparent pixels are flattened onto white so the
    background is consistent (and removable by the fg/bg layer split)."""
    im = Image.open(raw_path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    im.save(dst_path)
HERE = os.path.dirname(os.path.abspath(__file__))
PITCH = os.path.join(HERE, "pitch")


@app.get("/")
def index():
    return FileResponse(os.path.join(PITCH, "index.html"))


MAX_UPLOAD = 25 * 1024 * 1024  # 25 MB — guards against giant-upload OOM/DoS
_QUALITIES = ("auto", "max", "balanced", "small", "logo")


@app.post("/api/vectorize")
def api_vectorize(file: UploadFile = File(...), quality: str = Form("auto"),
                  layered: str = Form("false")):
    # SYNC endpoint on purpose: vtracer/PIL/resvg are blocking, so FastAPI runs this
    # in a worker thread and concurrent uploads run in parallel. An `async def` here
    # would run the blocking work on the event loop and serialise every request.
    if quality not in _QUALITIES:
        quality = "auto"
    raw_bytes = file.file.read()
    if len(raw_bytes) > MAX_UPLOAD:
        return JSONResponse({"error": "image too large (max 25 MB)"}, status_code=413)

    tmp = tempfile.mkdtemp()
    try:
        raw = os.path.join(tmp, "upload.bin")
        with open(raw, "wb") as f:
            f.write(raw_bytes)
        # Reject oversized resolutions before decoding fully (decompression-bomb DoS).
        try:
            with Image.open(raw) as probe:
                w, h = probe.size
        except Exception:
            return JSONResponse({"error": "unreadable image — upload a valid PNG, JPG, or WEBP"},
                                status_code=400)
        if w * h > MAX_PIXELS:
            return JSONResponse({"error": "image resolution too large (max ~25 megapixels)"},
                                status_code=413)
        inp = os.path.join(tmp, "input.png")
        try:
            _normalize_to_png(raw, inp)          # robust to any format/extension/alpha
        except Exception:
            return JSONResponse({"error": "unreadable image — upload a valid PNG, JPG, or WEBP"},
                                status_code=400)
        out = os.path.join(tmp, "out.svg")

        # Phase 3: analyze the image once → adapt preset + layering per PNG.
        prof = vz.analyze_image(inp)
        preset = prof["preset"] if quality == "auto" else quality

        try:
            if layered == "true":
                lq = preset if preset in ("max", "balanced", "small", "logo") else "balanced"
                # Pass the RAW upload (alpha intact), not the white-flattened `inp`:
                # vectorize_layered needs the original transparency for the magenta
                # sentinel that distinguishes a white SUBJECT from a white background.
                # (PIL sniffs format by content, so the .bin extension is fine.)
                vl.vectorize_layered(raw, out, lq, prompts=None, layer_mode=prof["layer_mode"])
            else:
                vz.vectorize(inp, out, quality, report=False)
        except BaseException:  # vtracer is a Rust ext → raises pyo3 PanicException (BaseException)
            return JSONResponse({"error": "vectorize failed — try another image or quality"},
                                status_code=500)

        with open(out, encoding="utf-8") as fh:
            svg = fh.read()
        kb = round(os.path.getsize(out) / 1024)
        paths = svg.count("<path")

        layers = []
        if layered == "true":
            ns = "{http://www.w3.org/2000/svg}"
            root = ET.fromstring(svg)
            for g in root.findall(f"{ns}g"):
                layers.append({"id": g.get("id"), "count": len(g.findall(f"{ns}path"))})

        l1 = vz.measure_fidelity(out, inp)
        fidelity = f"{round((1 - l1) * 100, 1)}%" if l1 is not None else "—"

        return JSONResponse({
            "svg": svg, "sizeKb": kb, "paths": paths,
            "preset": preset, "fidelity": fidelity, "layers": layers,
            "kind": prof["kind"], "n_colors": prof["n_colors"],
        })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # never leak the per-request temp dir


if __name__ == "__main__":
    import uvicorn
    # Bind to localhost by default (internal pitch tool). Set HOST=0.0.0.0 to expose
    # it on the LAN (e.g. to demo from a phone) and PORT to change the port.
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))
