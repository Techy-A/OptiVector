"""
vectorize.py — production image→SVG vectorizer (VTracer engine).

Turns a raster image into a clean, infinitely-scalable SVG that looks like the
original and stays razor-sharp at any print size. The right tool for logos and
faithful conversion (a contour tracer), tuned via measured fidelity vs file-size.

Quality presets (measured L1 vs original on a gradient test image; lower = closer;
all are true vector → crisp at any zoom):

    max       ~0.0115 L1, biggest file   — maximum fidelity
    balanced  ~0.0135 L1, mid file       — default
    small     ~0.017  L1, smallest file  — web/embed friendly

`auto` detects flat logos (few solid colors) vs gradient/photo art and picks
suitable settings + reports the measured fidelity (needs resvg-py).

Usage:
    python3 vectorize.py logo.png                       # -> logo.svg (auto)
    python3 vectorize.py photo.jpg -o out.svg -q max --report
"""

import argparse
import io
import os
import tempfile

import numpy as np
import vtracer
from PIL import Image

# Tuned via fidelity sweep (resvg render vs original). polygon ≈ same L1 as spline
# at ~3× smaller file, so small/balanced use polygon; max uses spline for the
# smoothest curves at extreme zoom.
PRESETS = {
    "max":      dict(color_precision=6, filter_speckle=2, layer_difference=16, mode="spline",  path_precision=3),
    "balanced": dict(color_precision=6, filter_speckle=6, layer_difference=16, mode="polygon", path_precision=2),
    "small":    dict(color_precision=6, filter_speckle=8, layer_difference=48, mode="polygon", path_precision=2),
    # Flat logos: polygon stays tiny + faithful (measured 34KB/L1 0.005 on a logo);
    # the earlier spline/ld8 guess ballooned to 500KB for no fidelity gain.
    "logo":     dict(color_precision=6, filter_speckle=4, layer_difference=24, mode="polygon", path_precision=2),
}


def prepare_for_trace(path, target_px=1400, max_factor=3, bg_fill=(255, 255, 255)):
    """
    Pre-process before tracing. The tracer hugs the antialiased pixel boundary,
    so small/low-res images trace with wobbly, staircased edges. Upscaling
    (LANCZOS) first gives a smoother boundary → cleaner curves and HIGHER fidelity
    (measured: L1 0.0045 → 0.0031 on a 736px logo). Flattens alpha onto bg_fill
    (white for the flat path; a magenta sentinel for the layered path so the
    transparent region traces to one droppable colour without eating white subject).

    Returns (working_png_path, upscale_factor). Upscale is adaptive: tiny images
    get up to max_factor×, images already ≥ target_px are left as-is.
    """
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, bg_fill)
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    W, H = im.size
    m = max(W, H)
    factor = 1
    if m < target_px:
        factor = min(max_factor, max(1, round(target_px / m)))
        if factor > 1:
            im = im.resize((W * factor, H * factor), Image.LANCZOS)
    fd, work = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    im.save(work)
    return work, factor


def analyze_image(path):
    """
    Classify an image so the pipeline can adapt per-PNG (Phase 3). Returns a
    profile: kind + recommended preset + layering strategy. Cheap (160px thumb).

      flat_logo      — few flat colors (icons/logos)         → 'logo' preset, color layers
      line_art       — few colors + high edge density (line/tribal art, sketches)
                                                              → 'max' preset, color layers
      gradient_photo — many colors (photos/watercolor)        → 'balanced', fg/bg layers
                       (per-color layering on a photo is noise, so it's disabled)
    """
    im = Image.open(path).convert("RGB")
    im.thumbnail((160, 160))
    a = np.asarray(im, np.float32)
    # Coarse (2-bit) color count collapses JPEG/AA noise that inflated naive counts.
    n_colors = len(np.unique((a // 64).astype(int).reshape(-1, 3), axis=0))
    g = a.mean(2)
    # Guard np.diff on a 1px-wide/tall thumbnail (empty diff → NaN + RuntimeWarning).
    ex = np.abs(np.diff(g, axis=1)) if g.shape[1] > 1 else np.zeros(1)
    ey = np.abs(np.diff(g, axis=0)) if g.shape[0] > 1 else np.zeros(1)
    edge = (ex > 30).mean() / 2 + (ey > 30).mean() / 2

    # Preset adapts only on what's reliably measurable:
    #   high edge density → line art (thin strokes) → 'max' (low speckle keeps them)
    #   very few colors   → clean flat logo         → 'logo' (tiny files)
    #   else              → 'balanced'
    # line art = high edge density AND few colors (e.g. black/red tribal art).
    # The color condition stops detailed COLOURFUL art (high edge but many colors,
    # e.g. a sticker) from being mislabeled line_art → shattered by the max preset.
    if edge > 0.085 and n_colors <= 12:
        kind, preset = "line_art", "max"
    elif n_colors <= 8:
        kind, preset = "flat_logo", "logo"
    else:
        kind, preset = "artwork", "balanced"
    # layer_mode='color' asks vectorize_layered for editable per-colour layers; it
    # SELF-CORRECTS to fg/bg when colour grouping would hurt fidelity (dense
    # overlapping art), so a per-kind threshold here would be redundant + too blunt
    # (e.g. it'd deny a simple black+star logo its colour layers).
    return {"kind": kind, "n_colors": int(n_colors), "edge": round(float(edge), 3),
            "preset": preset, "layer_mode": "color"}


def is_flat_art(path, max_colors=48):
    """Heuristic: few distinct colors → flat logo/icon. Quantizes to 3-bit/channel
    so JPEG compression noise doesn't inflate the count and misroute genuine logos."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((128, 128))
    q = (np.asarray(im) // 32).reshape(-1, 3)          # 3-bit/channel: JPEG-noise robust
    n = len(np.unique(q, axis=0))
    return n <= max_colors, n


def measure_fidelity(svg_path, original_path):
    """
    Render the SVG back (resvg) and report L1 vs original — measured on the
    FOREGROUND only (non-near-white pixels) at the image's NATIVE aspect ratio.
    Whole-frame L1 is dominated by background on subject-on-white images and
    would flatter the result; native aspect avoids distorting non-square inputs.
    Returns None if resvg-py is unavailable, or whole-frame L1 if no foreground.
    """
    try:
        import resvg_py
        orig = Image.open(original_path).convert("RGB")
        W, H = orig.size
        raw = resvg_py.svg_to_bytes(svg_path=svg_path, width=W, height=H, background="#ffffff")
        a = np.asarray(Image.open(io.BytesIO(bytes(raw))).convert("RGB"), np.float32) / 255
        b = np.asarray(orig, np.float32) / 255
    except Exception:                          # missing resvg, bad SVG, size mismatch
        return None
    diff = np.abs(a - b)
    fg = b.mean(axis=2) < 0.9                            # subject = not near-white
    return float(diff[fg].mean()) if fg.any() else float(diff.mean())


def vectorize(image_path, out_path=None, quality="auto", report=False):
    out_path = out_path or os.path.splitext(image_path)[0] + ".svg"

    if quality == "auto":
        prof = analyze_image(image_path)
        preset_name = prof["preset"]
        print(f"✓ auto: {prof['kind']} ({prof['n_colors']} colors) → '{preset_name}' preset")
    else:
        preset_name = quality
    params = PRESETS[preset_name]

    work, factor = prepare_for_trace(image_path)
    try:
        vtracer.convert_image_to_svg_py(work, out_path, colormode="color", **params)
    finally:
        if work != image_path and os.path.exists(work):
            os.remove(work)
    if factor > 1:
        print(f"  · upscaled {factor}× before tracing (smoother edges)")

    kb = os.path.getsize(out_path) / 1024
    with open(out_path) as fh:
        npaths = fh.read().count("<path")
    line = f"✓ {out_path}  [{preset_name}]  {kb:.0f} KB, {npaths} paths"
    if report:
        l1 = measure_fidelity(out_path, image_path)
        line += f", L1(fg)={l1:.4f}" if l1 is not None else "  (install resvg-py for fidelity report)"
    print(line)
    # Honest heads-up: gradient/photo input explodes path count regardless of tool.
    if npaths > 2000 or kb > 1000:
        print(f"  ⚠ large output ({npaths} paths, {kb:.0f} KB) — gradient/photo input. "
              f"Try -q small for a web-friendly file (lower fidelity), or expect big SVGs for faithful gradients.")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Image → scalable SVG (VTracer).")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-q", "--quality", default="auto",
                    choices=["auto", "max", "balanced", "small", "logo"])
    ap.add_argument("--report", action="store_true", help="measure fidelity (L1 vs original)")
    args = ap.parse_args()
    vectorize(args.image, args.out, args.quality, args.report)


if __name__ == "__main__":
    main()
