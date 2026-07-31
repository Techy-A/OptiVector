"""
vectorize_layered.py — semantic, EDITABLE SVG layers (Bet 2).

Tracers (VTracer/freeconvert) dump hundreds of flat <path>s with no structure —
spaghetti a designer can't edit. This traces the image, then GROUPS the paths into
named <g> layers so the output opens in Figma/Illustrator with editable, toggleable
layers (e.g. hide the background, recolor the subject).

Default: foreground vs background (no ML — background = paths matching the corner
color). Optional: --prompts "subject,background,text" uses CLIPSeg (segmentation.py)
to assign paths to N named semantic layers by where each path sits.

Usage:
    python3 vectorize_layered.py logo.png
    python3 vectorize_layered.py photo.jpg --prompts "person,background"
"""

import argparse
import copy
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import vtracer
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))

SVG_NS = "http://www.w3.org/2000/svg"
INK_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INK_NS)
_NUM = re.compile(r"-?\d+\.?\d*")


def _fill_rgb(path_el):
    """Extract an (r,g,b) 0-255 tuple from a path's fill attr or style."""
    fill = path_el.get("fill") or ""
    if not fill:
        m = re.search(r"fill:\s*([^;]+)", path_el.get("style", ""))
        fill = m.group(1).strip() if m else ""
    if fill.startswith("#"):
        h = fill[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    m = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", fill)
    if m:
        return tuple(int(x) for x in m.groups())
    return None


_NAMED = {
    "black": (25, 25, 25), "white": (242, 242, 242), "gray": (130, 130, 130),
    "red": (200, 35, 35), "orange": (232, 120, 25), "yellow": (240, 205, 30),
    "gold": (190, 150, 60), "brown": (120, 80, 40), "green": (45, 155, 60),
    "teal": (30, 150, 150), "blue": (45, 95, 200), "purple": (135, 60, 165),
    "pink": (232, 130, 165),
}


def _color_name(rgb):
    """Nearest human color name for an (r,g,b) — used to auto-name layers."""
    return min(_NAMED, key=lambda n: sum((a - b) ** 2 for a, b in zip(_NAMED[n], rgb)))


def _kmeans_colors(cols, uniq, k, iters=8):
    """Tiny k-means over fill colors. Seeded with the k most spread-out distinct
    colors (farthest-point) so the real palette is captured, not 13 fixed names."""
    centers = uniq[:1].astype(np.float32)
    for _ in range(1, k):
        d = ((uniq[:, None, :] - centers[None, :, :]) ** 2).sum(2).min(1)
        centers = np.vstack([centers, uniq[int(d.argmax())]])
    for _ in range(iters):
        assign = np.argmin(((cols[:, None, :] - centers[None, :, :]) ** 2).sum(2), axis=1)
        for c in range(k):
            m = cols[assign == c]
            if len(m):
                centers[c] = m.mean(0)
    return centers


_OCR_READER = None


def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR_READER


def _ocr_text_layer(image_path, factor, min_conf=0.45):
    """Detect printed text → editable <text> elements (real font) + their boxes
    (upscaled coords) so the blobby traced version can be removed. Returns ([], [])
    on failure — OCR is best-effort and never blocks a conversion."""
    try:
        res = _get_ocr_reader().readtext(image_path)
        arr = np.asarray(Image.open(image_path).convert("RGB"))
    except Exception:
        return [], []
    els, boxes = [], []
    for bbox, txt, conf in res:
        t = txt.strip()
        if conf < min_conf or len(t) < 2 or "pngtree" in t.lower() or ".com" in t.lower():
            continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        # Pad the REMOVAL box so the whole traced glyph blob is cleared (letters
        # overrun easyocr's tight box → otherwise remnants double up under the <text>).
        px, py = (x1 - x0) * 0.06, (y1 - y0) * 0.35
        boxes.append(((x0 - px) * factor, (y0 - py) * factor, (x1 + px) * factor, (y1 + py) * factor))
        # text colour = darkest pixel in the box (the ink, not the gaps)
        reg = arr[int(y0):max(int(y1), int(y0) + 1), int(x0):max(int(x1), int(x0) + 1)].reshape(-1, 3)
        col = reg[reg.mean(1).argmin()] if len(reg) else np.array([0, 0, 0])
        el = ET.Element(f"{{{SVG_NS}}}text", {
            "x": str(round(x0 * factor, 1)), "y": str(round(y1 * factor, 1)),
            "font-size": str(round((y1 - y0) * 0.95 * factor, 1)),
            "font-family": "Arial, Helvetica, sans-serif", "font-weight": "700",
            "fill": f"rgb({int(col[0])},{int(col[1])},{int(col[2])})"})
        el.text = t
        els.append(el)
    return els, boxes


def _translate(el):
    """vtracer emits each path with transform='translate(tx,ty)'. Must be applied
    or positions are wrong (made the knot weave drop + text boxes misalign)."""
    m = re.search(r"translate\(\s*([-\d.]+)[ ,]+([-\d.]+)", el.get("transform") or "")
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def _xy(el):
    nums = [float(x) for x in _NUM.findall(el.get("d", ""))]
    tx, ty = _translate(el)
    xs = [x + tx for x in nums[0::2]]
    ys = [y + ty for y in nums[1::2]]
    return xs, ys


def _centroid(el):
    xs, ys = _xy(el)
    n = min(len(xs), len(ys))
    return (sum(xs[:n]) / n, sum(ys[:n]) / n) if n else (0.0, 0.0)


def _bbox(el):
    xs, ys = _xy(el)
    return (min(xs), min(ys), max(xs), max(ys)) if xs and ys else (0.0, 0.0, 0.0, 0.0)


def _emit_tree(layers, Wt, Ht, idxmap, copy_els=True):
    """Build a layered <svg>: one <g id=layer> per layer, tagged as a real editor
    LAYER (inkscape:groupmode/label → Figma/AI/Inkscape show + name it). Layers are
    emitted in ORIGINAL paint order (mean source index) so overlapping designs stack
    right; text always last; background hidden by default (display:none). Order is
    computed from the original element refs (in idxmap), THEN elements are deep-copied
    on append so the same source paths can seed several candidate trees."""
    out = ET.Element(f"{{{SVG_NS}}}svg",
                     {"viewBox": f"0 0 {Wt} {Ht}", "width": str(Wt), "height": str(Ht)})

    def _order(kv):
        name, els = kv
        if name == "text":
            return float("inf")
        idxs = [idxmap[id(e)] for e in els if id(e) in idxmap]
        return sum(idxs) / len(idxs) if idxs else float("inf")

    for name, els in sorted(layers.items(), key=_order):
        if not els:
            continue
        attrs = {"id": name,
                 f"{{{INK_NS}}}groupmode": "layer",
                 f"{{{INK_NS}}}label": name}
        if name == "background":
            attrs["style"] = "display:none"
        grp = ET.SubElement(out, f"{{{SVG_NS}}}g", attrs)
        for el in els:
            grp.append(copy.deepcopy(el) if copy_els else el)
    return out


def _render_arr(tree, Wt, Ht, size=520):
    """Render a layer tree (background hidden) to a small RGB array on white. Used to
    compare the colour-regrouped picture against the faithful one. None on failure."""
    try:
        import io as _io
        import resvg_py
    except Exception:
        return None
    fd, p = tempfile.mkstemp(suffix=".svg"); os.close(fd)
    try:
        ET.ElementTree(tree).write(p, encoding="unicode", xml_declaration=False)
        sc = size / (max(Wt, Ht) or 1)
        w, h = max(1, int(Wt * sc)), max(1, int(Ht * sc))
        raw = resvg_py.svg_to_bytes(svg_path=p, width=w, height=h, background="#ffffff")
        return np.asarray(Image.open(_io.BytesIO(bytes(raw))).convert("RGB"), np.float32) / 255
    except Exception:
        return None
    finally:
        os.remove(p)


def vectorize_layered(image_path, out_path=None, quality="balanced", prompts=None,
                      layer_mode="color", ocr=False):
    out_path = out_path or os.path.splitext(image_path)[0] + ".layered.svg"
    SENTINEL = (250, 0, 250)   # magenta — flags the transparent region for clean removal
    src = Image.open(image_path)
    use_sentinel = False
    if src.mode in ("RGBA", "LA", "P"):
        rgba = src.convert("RGBA")
        transp = float((np.asarray(rgba)[..., 3] < 200).mean())
        img = Image.new("RGB", rgba.size, (255, 255, 255))   # white-flatten for colour array
        img.paste(rgba, mask=rgba.split()[-1])
    else:
        rgba, transp = None, 0.0
        img = src.convert("RGB")
    arr0 = np.asarray(img)
    corner = np.array([arr0[0, 0], arr0[0, -1], arr0[-1, 0], arr0[-1, -1]]).mean(0)

    if transp >= 0.02:
        # Genuine transparency. If the opaque subject is near-WHITE (e.g. an eagle's
        # white head touching the border), white-flatten can't tell subject from white
        # background → trace the transparent region to a magenta sentinel and drop
        # exactly that. Otherwise white-flatten (transparent gaps render white-on-top,
        # e.g. the knot weave — not a dropped hole baring a black silhouette).
        a = np.asarray(rgba); op = a[a[..., 3] > 200][:, :3]
        near_white = float((op.mean(1) > 200).mean()) if len(op) else 0.0
        use_sentinel = near_white > 0.03
        if use_sentinel:
            bg_rgb, trace_bg = np.array([float(c) for c in SENTINEL]), SENTINEL
        else:
            bg_rgb, trace_bg = corner, (255, 255, 255)
    else:
        # Opaque — incl. PNGs flagged alpha but fully opaque (white-bg logos like the
        # eagle-on-star). Background = corner colour AND touches the border.
        bg_rgb, trace_bg = corner, (255, 255, 255)
    W, H = img.size
    arr = arr0

    # 1. Trace (reuse the tuned engine presets + adaptive pre-upscale for clean edges).
    import vectorize as vz
    tdir = tempfile.mkdtemp()
    tmp = os.path.join(tdir, "t.svg")
    params = vz.PRESETS.get(quality, vz.PRESETS["balanced"])
    work, factor = vz.prepare_for_trace(image_path, bg_fill=trace_bg)
    try:
        vtracer.convert_image_to_svg_py(work, tmp, colormode="color", **params)
    finally:
        if work != image_path and os.path.exists(work):
            os.remove(work)
    root = ET.parse(tmp).getroot()
    shutil.rmtree(tdir, ignore_errors=True)   # trace temp parsed into memory — don't leak it
    paths = [el for el in root.iter(f"{{{SVG_NS}}}path")]
    idxmap = {id(el): i for i, el in enumerate(paths)}   # original paint order
    # Traced coords live in upscaled space — the output viewBox must match.
    Wt, Ht = W * factor, H * factor

    # Optional OCR: detect text → editable <text>, and the boxes to clear of
    # blobby traced glyphs. Best-effort; off by default (slow model load).
    text_els, text_boxes = _ocr_text_layer(image_path, factor) if ocr else ([], [])

    def in_text_box(el):
        if not text_boxes:
            return False
        cx, cy = _centroid(el)
        return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in text_boxes)

    margin = 0.02 * max(Wt, Ht)

    def is_background(el):
        """Transparent (alpha) image → background is exactly the magenta-sentinel
        fill: drop it, never touching white subject. Flat-bg image (JPG) → background
        is the bg-colour AND bbox touches the border (keeps internal bg-coloured
        shapes like knot weave / letter counters; only the outer frame removed)."""
        rgb = _fill_rgb(el)
        if rgb is None:
            return False
        if use_sentinel:
            return np.abs(np.array(rgb) - bg_rgb).mean() < 40   # ≈ magenta sentinel
        if np.abs(np.array(rgb) - bg_rgb).mean() >= 28:
            return False
        x0, y0, x1, y1 = _bbox(el)
        return x0 <= margin or y0 <= margin or x1 >= Wt - margin or y1 >= Ht - margin

    # 2. Split foreground vs background (shared by every mode).
    bg_list, fg_all = [], []
    for el in paths:
        if in_text_box(el):
            continue
        (bg_list if is_background(el) else fg_all).append(el)
    # Full-bleed / solid image: the subject IS the corner colour and touches every
    # border, so the heuristic would flag everything as background → a blank SVG.
    # Keep it all as foreground instead (nothing safe to remove).
    if not fg_all and bg_list:
        fg_all, bg_list = bg_list, []

    def color_layers(fg_els):
        """Regroup foreground into named per-colour layers (k-means on actual fills,
        capped at 8, then MERGE perceptually-close clusters so antialiasing-noise
        shades don't become frayed junk layers). This REORDERS paths into colour
        groups, so the caller only keeps it when a render check shows the picture
        didn't change (safe on flat art; buries detail on overlapping art)."""
        fg = [(el, _fill_rgb(el)) for el in fg_els]
        fg = [(el, c) for el, c in fg if c is not None]
        result = {}
        if not fg:
            return result
        cols = np.array([c for _, c in fg], dtype=np.float32)
        uniq = np.unique(cols, axis=0)
        k = int(min(8, len(uniq)))
        centers = _kmeans_colors(cols, uniq, k)
        assign0 = np.argmin(((cols[:, None, :] - centers[None, :, :]) ** 2).sum(2), axis=1)
        MERGE_T = 60.0
        parent = list(range(k))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(k):
            for j in range(i + 1, k):
                if np.linalg.norm(centers[i] - centers[j]) < MERGE_T:
                    parent[find(j)] = find(i)
        roots = sorted({find(i) for i in range(k)})
        root_idx = {r: n for n, r in enumerate(roots)}
        cid = np.array([root_idx[find(a)] for a in assign0])
        used = {}
        for ci in sorted(range(len(roots)), key=lambda c: int((cid == c).sum()), reverse=True):
            members = [fg[i][0] for i in range(len(fg)) if cid[i] == ci]
            if not members:
                continue
            base = _color_name(cols[cid == ci].mean(0))
            name = base if base not in used else f"{base}-{used[base] + 1}"
            used[base] = used.get(base, 0) + 1
            result[name] = members
        return result

    if prompts:
        from curvesplat.segmentation import generate_semantic_masks
        names = [p.strip() for p in prompts]
        masks = generate_semantic_masks(arr, names)        # list of HxW arrays
        masks = [np.asarray(Image.fromarray((m > 0).astype(np.uint8) * 255).resize((W, H))) > 127
                 for m in masks]
        layers = {n: [] for n in names}
        for el in fg_all:
            cx, cy = _centroid(el)
            ix = int(np.clip(cx / factor, 0, W - 1))   # centroid is in upscaled coords
            iy = int(np.clip(cy / factor, 0, H - 1))
            scores = [int(m[iy, ix]) for m in masks]
            layers[names[int(np.argmax(scores))]].append(el)
        layers["background"] = bg_list
    else:
        faithful = {"background": bg_list, "foreground": fg_all}
        if layer_mode == "fgbg":
            layers = faithful
        else:
            # Per-colour layers REORDER paths into colour groups. That's lossless on flat
            # art (colours don't overlap → identical picture) but buries detail on
            # overlapping art (a light eye painted under dark feathers). So render BOTH
            # the colour regroup and the faithful original-order version and DIFF the two
            # renders: if regrouping barely moved the picture, keep the editable colour
            # layers; if it flipped pixels (eye, frayed edges), keep faithful fg/bg.
            # Mean diff alone misses a small eye → also gate on the fraction of badly-
            # changed pixels.
            colored = {"background": bg_list}
            colored.update(color_layers(fg_all))
            n_color = len([k for k in colored if k != "background"])
            ca = _render_arr(_emit_tree(colored, Wt, Ht, idxmap), Wt, Ht)
            fa = _render_arr(_emit_tree(faithful, Wt, Ht, idxmap), Wt, Ht)
            if ca is not None and fa is not None and n_color >= 2:
                dif = np.abs(ca - fa).max(2)
                changed, badfrac = float(dif.mean()), float((dif > 0.4).mean())
                safe = changed < 0.008 and badfrac < 0.008
            else:
                changed = badfrac = -1.0
                safe = False
            layers = colored if safe else faithful
            print(f"  · regroup Δ={changed:.4f} bad={badfrac:.4f} ({n_color} colours)"
                  f"  → {'colour layers' if safe else 'fg/bg (faithful)'}")

    # Editable text on top (real <text>, replacing the removed blobby glyphs).
    if text_els:
        layers["text"] = text_els

    # 3. Emit the chosen layering. No canvas rect → transparent background; the
    # "background" group ships hidden (display:none).
    tree = _emit_tree(layers, Wt, Ht, idxmap, copy_els=False)
    ET.ElementTree(tree).write(out_path, encoding="unicode", xml_declaration=False)
    summary = ", ".join(f"{n}={len(e)}" for n, e in layers.items() if e)
    kb = os.path.getsize(out_path) / 1024
    print(f"✓ {out_path}  [{quality}]  {kb:.0f} KB  ·  layers: {summary}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Image → SVG with editable semantic layers.")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-q", "--quality", default="balanced",
                    choices=["max", "balanced", "small", "logo"])
    ap.add_argument("--prompts", default=None,
                    help="comma-separated semantic layers, e.g. 'person,background'")
    ap.add_argument("--ocr", action="store_true",
                    help="detect printed text → editable <text> (best-effort; clean text only)")
    args = ap.parse_args()
    prompts = args.prompts.split(",") if args.prompts else None
    vectorize_layered(args.image, args.out, args.quality, prompts, ocr=args.ocr)


if __name__ == "__main__":
    main()
