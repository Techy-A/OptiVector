# Vectorizer — ship notes

## What this is
`vectorize.py` — image → clean, infinitely-scalable SVG, crisp at any print size.
Engine: **VTracer** (contour tracer) with fidelity-tuned presets. This is the right
tool for logos/faithful conversion. (CurveSplat — the differentiable splat engine in
`src/` — is **retired for this goal**: it makes painterly approximations, never
pixel-exact crisp output. Kept for reference only.)

## Use
```bash
python3 -m pip install vtracer resvg-py        # resvg-py only needed for --report
python3 vectorize.py logo.png                  # auto preset -> logo.svg
python3 vectorize.py photo.jpg -o out.svg -q small --report
```
Presets: `auto` (detects flat-logo vs gradient), `max` (smoothest splines, biggest),
`balanced`, `small` (tiny), `logo` (flat solid art). `--report` prints foreground
L1 fidelity vs the original (needs resvg-py).

## What it does well / honest limits (measured)
| Input | Result | Verdict |
|---|---|---|
| **Solid logo/icon** | 45 KB, L1 0.005, crisp at any zoom | ✅ ship — exactly the goal |
| **Line-art / text** | use `-q max` (default over-simplifies thin lines) | ✅ with the flag |
| **Photo / gradient / watercolor** | faithful (L1 0.03) but BIG: ~500 KB–2.5 MB, thousands of paths | ⚠ physics — any tracer does this; sell as "faithful, heavy", not "tiny" |

Why gradients are big: a tracer emits one filled path per quantized color region;
smooth gradients → many bands → many paths. No vectorizer escapes this. `-q small`
trades fidelity for a web-friendly file.

Fidelity is measured on FOREGROUND pixels at native aspect ratio (whole-frame L1 is
background-flattered on subject-on-white images — don't trust that number).

## Ship-to-customers next step (optional)
The repo already has a FastAPI `server.py` + React `frontend/` (built for CurveSplat).
Swap the engine: point the endpoint at `vectorize.vectorize(...)` instead of the
CurveSplat subprocess → a deployable browser tool (upload image → download SVG).
~half a day of wiring.

## Don't
- Don't pitch this as beating a tracer — it IS a tracer (VTracer). The value vs raw
  freeconvert/VTracer = tuned presets, auto-detect, fidelity report, and **15× smaller
  files** (phoenix 185 KB vs freeconvert's 2.8 MB at similar fidelity).
- Don't feed it gradient art expecting tiny files. Use flat art for the small-file win.
