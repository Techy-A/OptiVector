# SVG AI Vectorize

Turn any raster image (PNG / JPG / WEBP) into a clean, **infinitely-scalable SVG**
that stays razor-sharp at any print size — with optional **editable, named layers**
and automatic **background removal**.

Best for logos, icons, flat/line art, and clean illustrations.

> Engine: [VTracer](https://github.com/visioncortex/vtracer) (a contour tracer).
> This is faithful vectorization, not AI generation — it reproduces your image, it
> doesn't invent detail. For the full design, read [`HOW_IT_WORKS.txt`](HOW_IT_WORKS.txt).
>
> *(The earlier differentiable Gaussian-splat engine, "CurveSplat", lives under
> `src/` + `tests/` but is retired for faithful-logo work — see `SESSION_SUMMARY.md`.)*

---

## Quick start

```bash
# 1. Install deps
python3 -m pip install fastapi uvicorn python-multipart vtracer resvg-py pillow numpy scipy

# 2. Run the web app
python3 server.py            # -> http://localhost:8000
```

Open http://localhost:8000, drop an image, hit **Vectorize**, download the SVG.

> Binds to `127.0.0.1` by default. To demo from another device on your network:
> `HOST=0.0.0.0 python3 server.py` (optionally `PORT=…`).

### Command line (no server)

```bash
# Plain vectorize (auto-detects logo vs artwork, picks settings)
python3 vectorize.py logo.png                      # -> logo.svg
python3 vectorize.py photo.jpg -o out.svg -q max --report

# With editable, named layers + background removal
python3 vectorize_layered.py logo.png              # -> logo.layered.svg
```

Quality presets (`-q`): `auto` (default), `max`, `balanced`, `small`, `logo`.

---

## What you get

- **True vector** — zoom forever, no pixelation; prints crisp at any size.
- **Tuned for size** — flat logos land around tens of KB, not hundreds.
- **Editable layers** (when the art supports it) — paths grouped into named `<g>`
  layers tagged for Figma / Illustrator / Inkscape; toggle or recolor each.
- **Background removed** — output ships on transparency.
- **Measured fidelity** — every result reports how close it is to the original
  (rendered back with resvg and compared on the foreground).

---

## How layering decides (important)

Two layer shapes exist; the tool **measures** which is safe per image — it never
ships a layered-but-broken file:

| Image kind | Layers you get | Why |
|---|---|---|
| Flat logos, line art, simple multi-color (swirl, knot, eagle-on-star) | **Per-color layers** | Colors don't overlap → grouping by color is lossless |
| Dense / painterly / overlapping art (colorful sticker, lion, phoenix) | **Foreground + background** | Color-grouping would reorder strokes and damage the picture |

Mechanism: render the color-grouped version and the faithful original-order
version, diff the two; keep color layers only if the picture is unchanged. (Details
in `HOW_IT_WORKS.txt` → "THE GATE".)

---

## Project layout

| File | Role |
|---|---|
| [`vectorize.py`](vectorize.py) | Engine: presets, adaptive pre-upscale, tracing, fidelity measurement, image analysis |
| [`vectorize_layered.py`](vectorize_layered.py) | Editable layers: color-vs-fg/bg decision, background removal, transparency handling |
| [`server.py`](server.py) | FastAPI app: serves the UI + `POST /api/vectorize` |
| [`pitch/index.html`](pitch/index.html) | Single-file web UI (dark/lime, dropzone, layers panel, zoom demo) |
| [`HOW_IT_WORKS.txt`](HOW_IT_WORKS.txt) | Full technical explanation of the whole pipeline |
| `samples/` | Test images |

---

## API

`POST /api/vectorize` — multipart form:

| field | values | default |
|---|---|---|
| `file` | the image (PNG/JPG/WEBP, ≤ 25 MB and ≤ ~25 megapixels) | required |
| `quality` | `auto` / `max` / `balanced` / `small` / `logo` | `auto` |
| `layered` | `true` / `false` | `false` |

Returns JSON:

```json
{
  "svg": "<svg>…</svg>",
  "sizeKb": 153, "paths": 1130, "preset": "balanced",
  "fidelity": "97.5%", "kind": "artwork", "n_colors": 9,
  "layers": [{"id": "background", "count": 107}, {"id": "white", "count": 864}]
}
```

Notes: the endpoint runs in a worker thread (concurrent uploads don't block each
other); for true multi-core throughput run `uvicorn server:app --workers 4`.

---

## Limits (honest)

- **Gradients** band into flat color steps (contour-tracer limitation) — a metallic
  gradient logo looks chunky. Flat/solid art is the sweet spot.
- **Dense/painterly art** gets faithful fg/bg only, not per-color layers.
- **Layer names** are nearest-color guesses (a dark antialiased tone may read
  "brown") — rename in your editor; the pixels are correct.
- **Photos** produce large SVGs with many paths — inherent to faithful tracing.

---

*Internal build · © 2026 Techy-A*
