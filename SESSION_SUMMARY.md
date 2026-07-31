# CurveSplat — Session Summary

## Goal
Make CurveSplat produce **crisp, tracer-quality SVG** on complex images (a watercolour
phoenix) instead of blurry painterly output — while keeping the painterly engine — and
make it **fast** on the remote RTX A6000.

## Root cause of the blur (diagnosed)
CurveSplat rendered *everything* as soft stroked Gaussian splats (`fill="none"`, 3σ
soft falloff, min width 0.5px). Large flat regions were built from overlapping soft
strokes → blur. Tracers (VTracer/Adobe) emit **filled flat shapes with hard edges**.
Architectural gap, not a tuning problem.

---

## What we BUILT (crispness — done & verified)

| Piece | File | Status |
|---|---|---|
| Differentiable **filled-shape renderer** (closed Bezier loops, sharp ~1px edge via polyline-SDF + winding sign) | `src/python/curvesplat/fill_renderer.py` | ✅ float64 gradcheck passes |
| **FilledShape** scene primitive (holds `(S,4,2)` loop segments) | `src/python/curvesplat/scene.py` | ✅ |
| Tracer-style **filled-region scaffold** (k-means → marching squares → closed Bezier loops, z-ordered by area) | `src/python/curvesplat/sdf_init.py` (`initialize_filled_regions`) | ✅ |
| **Edge-sharpness loss** (Sobel-gradient L1) | `src/python/curvesplat/losses.py` (`compute_edge_loss`) | ✅ |
| **SVG export** for filled `fill="rgb()" … Z` paths + strokes | `examples/vectorize_cuda.py` (`export_to_svg`) | ✅ |
| **`--mode hybrid`** (crisp filled base + painterly strokes) + Stage-6 polish gated by MS-SSIM | `examples/vectorize_cuda.py` | ✅ |

**Quality proof (init only, zero optimization):** filled base cut L1 ~60% vs painterly
init on flat art (logo 0.128→0.051, text 0.126→0.055). Crispness goal achieved.

**Tests:** 68 passing (was 54). New: `tests/test_fill_renderer.py`,
`tests/test_filled_init.py`, `tests/test_batched_renderer.py`.

---

## What we ATTEMPTED (speed — hit the pure-PyTorch wall)

Per-step at full 512px / 193 curves was **~9s/step** (GPU util ~0–27%). Three fixes tried:

| Attempt | Result | Why |
|---|---|---|
| **Batched renderer** (vectorize all curves, drop Python loop) | no net gain | pixel-bound; eager can't fuse |
| **`torch.compile`** the batched fn | no gain (+92s warmup) | graph-broke on `autocast(False)`/`.float()` |
| **Tiled / bbox renderer** (each curve only its local pixels) | no gain on GPU | `.item()` bbox calls → ~760 GPU↔CPU syncs/step |

All three are **numerically correct** (gradchecked / equivalence-tested) — they just
don't beat ~9s/step. `render_curves_batched` is the active path; `render_curves_tiled`
exists but is unused (sync-bound on GPU).

**Conclusion:** ~9s/step is roughly the **pure-PyTorch ceiling** for this splat method.
Real speed needs a **custom CUDA kernel** — exactly what `SPEC.md §4` always planned
(`splat_forward.cu` / `splat_backward.cu`). That's a separate multi-day project.

---

## Remote environment notes (aidev2, RTX A6000)
- `torch.compile` + AMP crashes (`expected Half but found Float`) → use `--no_amp` or it's now out of hot path.
- `torch` install URL in `HOW_TO_RUN_CUDA.md` is broken (`/ml/cu121` → should be `/whl/cu121`).
- GPU0 often hogged by another user (~39GB) → use `CUDA_VISIBLE_DEVICES=1`.
- `rsync` runs on the **Mac**, everything else in SSH.

---

## How to run (lean config — finishes in minutes, real output today)
```bash
CUDA_VISIBLE_DEVICES=1 python3 -u examples/vectorize_cuda.py \
  --image phoenix.png --output_dir examples/output \
  --mode hybrid --n_regions 20 --n_fill 70 --n_edge 70 \
  --num_samples 8 --max_dim 384 --steps 250 --no_adaptive \
  --lambda_edge 0.6 --polish_steps 60 --log_every 10 > run.log 2>&1 &
tail -f run.log
```
New flags added this session: `--mode`, `--n_regions`, `--lambda_edge`,
`--polish_steps`, `--ssim_gate`, `--log_every`.

Pull results to Mac:
```bash
scp aryan@172.16.15.240:~/SVG_AI/examples/output/optimized_artwork.svg ~/Downloads/
scp aryan@172.16.15.240:~/SVG_AI/examples/output/cuda_vectorization_comparison.png ~/Downloads/
```

---

## Status
- ✅ **Crispness goal: done** — hybrid filled+painterly, correct, tested.
- ⚠️ **Speed goal: blocked at pure-PyTorch ceiling** (~9s/step at 512). Workaround = lean
  config (`--max_dim 384`, fewer curves, `--num_samples 8`) ≈ 1–2s/step.
- 🔜 **Real speed fix = custom CUDA splat kernel** (SPEC §4) — not started, big project.

Notes on a watercolour like the phoenix: it's the *worst case* for crisp fills (soft
gradients, almost no flat regions → only 3 filled regions extracted). Most of the image
is carried by painterly strokes. Filled base helps the solid body/wing blocks.
```
