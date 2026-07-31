# CurveSplat — Implementation Specification

## 1. Project summary

CurveSplat is a differentiable image-to-SVG vectorizer. It initializes vector
curves from an SDF-based contour extraction, optimizes them by rendering with
a differentiable Gaussian-splatting rasterizer (curves approximated as
elongated 2D Gaussians along Bezier splines), and optionally polishes the
result with an exact (DiffVG-style) rasterizer before exporting to SVG.

This spec consolidates the two architecture documents provided
(`CurveSplat system overview` and `full stage-by-stage pipeline`) into a
single source of truth for implementation, and ties it to the existing
project conventions defined in:

- `Agents.md` — agent behavior / coding conventions for this repo
- `Frontend.md` — frontend architecture and conventions for the demo UI
- `brandguidelines.md` — visual styling for any UI surfaces (dark theme,
  neon lime accent)

Any UI, dashboard, or demo page built for this project must follow
`brandguidelines.md` and `Frontend.md`. Any agent working autonomously in
this repo must follow `Agents.md`.

## 2. Pipeline overview

```
Raster image (PNG/JPG/HEIC)
  → Stage 0: Preprocess (decode, resize, denoise, GPU transfer)
  → Stage 1: SDF init (edges → SDF via JFA → marching squares → Douglas-Peucker
             → corner detection → Bezier fit) → initial curve set (50-300 curves)
  → Stage 2: Gaussian-splatting rasterizer (differentiable forward render)
  → Stage 3: Loss (L1 + LPIPS + MS-SSIM + smoothness/width/color/count reg.)
  → Stage 4: Backward pass (autograd + custom CUDA gradient kernels)
  → Stage 5: Optimization (AdamW, LR schedule, adaptive split/merge/prune/densify,
             coarse-to-fine resolution, convergence check)
  → Stage 6: Exact polish (optional DiffVG-style boundary rasterization)
  → Stage 7: SVG export (path construction, color quantization, layer ordering,
             simplification)
```

## 3. Core data structures

```python
Curve = {
    control_points: float[4][2]   # P0..P3, cubic Bezier
    color:          float[3]      # RGB, sigmoid-activated
    width:          float         # softplus-activated stroke width
    opacity:        float         # sigmoid-activated, [0,1]
    is_closed:      bool          # filled shape vs stroke
    z_index:        int           # layer ordering
}

Scene = {
    curves:     list[Curve]
    background: float[3]          # learnable background color
    canvas_size: (H, W)
}
```

Parameter count: 13 per curve (8 control point coords + 3 color + 1 width +
1 opacity). A 500-curve scene is ~6,500 parameters.

## 4. Repository structure

```
curvesplat/
├── src/
│   ├── cuda/
│   │   ├── splat_forward.cu
│   │   ├── splat_backward.cu
│   │   ├── sdf_jfa.cu
│   │   ├── marching_squares.cu
│   │   └── bezier_eval.cuh
│   ├── cpp/
│   │   ├── binding.cpp
│   │   └── scene.cpp
│   └── python/curvesplat/
│       ├── __init__.py
│       ├── scene.py          # Curve/Scene dataclasses
│       ├── renderer.py        # nn.Module wrapper (forward render)
│       ├── losses.py          # LPIPS, MS-SSIM, regularizers
│       ├── optimizer.py       # AdamW + adaptive curve management
│       ├── svg_export.py      # SVG path generation
│       └── trainer.py         # Full training loop
├── tests/
├── examples/
│   ├── vectorize_photo.py
│   └── vectorize_logo.py
├── frontend/                  # see Frontend.md
└── CMakeLists.txt
```

## 5. Dependencies

| Library | Purpose |
|---|---|
| PyTorch 2.3+ | autograd, optimizers, tensors |
| CUDA 12.1+ | custom kernel compilation |
| pybind11 | C++/Python bindings |
| OpenCV (GPU) | preprocessing, SDF fallback |
| lpips | perceptual loss |
| pytorch-msssim | structural loss |
| svgpathtools | SVG export validation |

## 6. Phased roadmap (MVP)

Each phase is a milestone with explicit acceptance criteria. Agents should
not start phase N+1 until phase N's acceptance criteria pass.

### Phase 1 — Pure PyTorch splatting (no CUDA)
- Implement `Curve`, `Scene` dataclasses (`scene.py`).
- Implement a pure-PyTorch differentiable renderer for a *single* curve
  (`renderer.py`), using the Gaussian-splat math in §7.
- Target: a 256x256 image, single curve, simple target (filled circle).
- **Acceptance**: gradients flow end-to-end (verify with `torch.autograd.gradcheck`
  or numerical finite-difference check); the curve's parameters converge toward
  matching the circle within 500 Adam steps.

### Phase 2 — SDF initialization
- Implement edge detection (Sobel multi-scale + Canny), Otsu threshold,
  JFA-based SDF (pure PyTorch/NumPy ok for now), marching squares contour
  extraction, Douglas-Peucker simplification, corner detection, and
  least-squares cubic Bezier fitting (`scene.py` init helpers).
- **Acceptance**: given a binary star-shape mask, produce ~10 curves whose
  rendered silhouette visually matches the star (MS-SSIM > 0.9 before any
  optimization).

### Phase 3 — Multi-curve scene + training loop
- Extend renderer to N curves (batched Gaussian splatting, still pure PyTorch).
- Implement `losses.py`: L1 + LPIPS (frozen VGG-16) to start; MS-SSIM and
  regularizers stubbed.
- Implement `trainer.py` basic loop with AdamW and LR schedule (§8).
- **Acceptance**: 50 curves optimize to match a simple flat-color icon
  (L1 < 0.02 after convergence).

### Phase 4 — CUDA forward kernel
- Port the tile-based Gaussian splatting forward pass to a CUDA kernel
  (`splat_forward.cu`), matching the pure-PyTorch reference numerically
  (max abs diff < 1e-4).
- **Acceptance**: ≥10x speedup over the PyTorch reference at 512x512,
  200 curves.

### Phase 5 — CUDA backward kernel
- Implement `splat_backward.cu` per the gradient equations in §7.4.
- Wire into a custom `torch.autograd.Function`.
- **Acceptance**: gradcheck against the PyTorch-autograd reference passes
  within tolerance; full end-to-end GPU optimization runs.

### Phase 6 — Adaptive curve management + full losses
- Implement split/merge/prune/densify (§9) running every 500 steps.
- Add MS-SSIM and all regularization terms to `losses.py`.
- **Acceptance**: photo optimization (1024x1024) converges with curve count
  stabilizing (not monotonically growing) and MS-SSIM > 0.9.

### Phase 7 — SVG export
- Implement `svg_export.py`: path construction, k-means color quantization
  (≤256 colors), layer ordering by z_index, collinear-segment merging,
  coordinate quantization to 2 decimals, metadata embedding.
- **Acceptance**: output validates with `svgpathtools`; renders visually
  equivalent to `I_render` in a browser.

### Phase 8 — Exact polish, benchmark, demo UI
- Implement optional DiffVG-style exact-boundary polish (Stage 6), gated by
  MS-SSIM < 0.95 check.
- Build the demo frontend per `Frontend.md`, styled per `brandguidelines.md`.
- Benchmark against VTracer on a standard image set; produce a comparison
  report.

## 7. Gaussian-splatting math (forward)

```
Bezier:   P(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
Tangent:  P'(t) = 3(1-t)^2 (P1-P0) + 6(1-t)t (P2-P1) + 3t^2 (P3-P2)

theta = atan2(P'_y, P'_x)
sigma_long  = curve.width
sigma_short = curve.width / 4
Sigma = R(theta) . diag(sigma_long^2, sigma_short^2) . R(theta)^T

For pixel x:
  d = x - mu
  mahalanobis^2 = d^T . Sigma^-1 . d     (skip if > 9, i.e. 3-sigma cutoff)
  weight = exp(-0.5 * mahalanobis^2)
  alpha  = weight * curve.opacity
  C_out += alpha * curve.color * (1 - alpha_acc)
  alpha_acc += alpha
```

### 7.4 Backward gradients

```
dW/dmu    = -weight * Sigma^-1 * (x - mu)
dW/dSigma = 0.5 * weight * (Sigma^-1 (x-mu)(x-mu)^T Sigma^-1 - Sigma^-1)
dW/dalpha = weight
dmu/dPi   = Bi(t) * I_2x2          (Bezier basis function chain rule)
dSigma/dw = 2w * R . diag(1, 1/16) . R^T
```

## 8. Optimizer / schedule

```
AdamW(lr=0.01, betas=(0.9,0.999), weight_decay=0.01)
Warmup: linear 0 -> 0.01 over first 100 steps
Phase 1 (0-500):    lr=0.01    res=256
Phase 2 (500-2000): lr=0.005   res=512, adaptive mgmt begins
Phase 3 (2000-4000): lr=0.001  res=1024 (optional exact polish at end)
Phase 4 (4000+):    lr=0.0001  res=1024, fine convergence
Convergence: stop if loss improvement < 0.01% for 500 consecutive steps,
or max_steps=10000.
```

## 9. Adaptive curve management (every 500 steps)

- **SPLIT**: if a curve's max per-sample error exceeds threshold and curve
  length > 20px, split at the max-error parameter t into two Beziers
  (de Casteljau or refit).
- **MERGE**: if two adjacent curves have color distance < 0.1 and overlapping
  bounding boxes, try fitting a single curve through combined samples; accept
  if reconstruction error increases < 5%.
- **PRUNE**: remove curves whose energy contribution Ei / total_energy < 0.001.
- **DENSIFY**: insert samples where curvature kappa(t) exceeds threshold and
  sample spacing > 5px.

## 10. Total loss

```
L_total = 1.0*L_pixel(L1) + 0.5*L_percep(LPIPS) + 0.3*L_ssim(MS-SSIM)
        + 0.01*L_smooth + 0.001*L_width + 0.05*L_color + 0.1*L_count
```

## 11. Risk mitigation

| Risk | Mitigation |
|---|---|
| CUDA kernel crashes | Pure PyTorch prototype first; port incrementally |
| Gradient explosion | Clip grad norm to 1.0; fp32 for critical paths |
| Local minima (stuck curves) | Coarse-to-fine resolution; restart curve count |
| Memory overflow | Tile-based rendering; gradient checkpointing |
| Slow convergence | Curriculum: simple shapes -> complex images |

## 12. Definition of done — Phase 1

- [ ] `scene.py` defines `Curve` and `Scene` dataclasses matching §3
- [ ] `renderer.py` implements single-curve differentiable rendering per §7
- [ ] Unit test: gradcheck passes on a small canvas (e.g. 32x32)
- [ ] Example script: single curve converges to a filled circle target within
      500 Adam steps, with a saved before/after PNG comparison
- [ ] All code follows conventions in `Agents.md`
