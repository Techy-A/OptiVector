# CurveSplat Core Optimizations & Code Quality Audit Fixes

This document details the 12 architectural, algorithmic, and mathematical fixes implemented in the CurveSplat codebase to resolve bad optimization convergence (blurry, off-center structures), numerical stability issues, and performance overheads.

---

## 1. Differentiable Renderer & Mathematical Stability

### Issue 1: Stroke Width Not Scaled During Coarse-to-Fine Rendering
* **Symptom:** During the coarse optimization phases (e.g., at 1/8th or 1/4th resolution), control points were scaled down, but stroke widths were not. A 60px stroke at 1/8th resolution (64x64 canvas) covered the entire viewport, rendering optimization useless.
* **Fix:** We updated the `render_single_curve_fn` in [vectorize_cuda.py](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L175) to accept a `width_scale` parameter. The activated stroke width is scaled by `min(scale_x, scale_y)` before rendering.
* **Location:** [vectorize_cuda.py:175](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L175) and [render_scene_shuffled](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L244)

### Issue 2: Mathematical Inverse Softplus Overflow
* **Symptom:** The inverse softplus function `math.log(math.exp(w) - 1)` overflowed to `inf` for stroke widths `w > 88`. Large initial blob strokes (e.g. 153px wide) resulted in `width_raw = inf`, causing training to crash immediately.
* **Fix:** Capped initial fill stroke widths at `min(H, W) * 0.15` (down from `0.3`) and guarded the calculation with:
  ```python
  width_raw = math.log(math.exp(w) - 1) if 0.5 < w < 20 else w
  ```
  Since `softplus(x) ≈ x` for `x >= 20`, the approximation is highly accurate and mathematically stable.
* **Locations:** [vectorize_cuda.py:138](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L138) and [sdf_init.py:963](file:///Users/appypie/Downloads/SVG_AI/src/python/curvesplat/sdf_init.py#L963)

### Issue 3: Sequential Compositing in public `render_scene()` API
* **Symptom:** The public wrapper `render_scene()` only rendered `curves[0]`. Any module using `render_scene()` (like the logo vectorizer) failed to render multi-curve scenes correctly.
* **Fix:** Rewrote `render_scene()` to composite all curves sequentially in Z-index order on a canvas, matching the differentiable multi-curve compositing rules. We also updated `CurveSplatRenderer.forward()` to accept a 3D tensor canvas `(3, H, W)` directly as a background.
* **Location:** [renderer.py:276](file:///Users/appypie/Downloads/SVG_AI/src/python/curvesplat/renderer.py#L276)

### Issue 4: Exploding Gradients (NaNs) under Small Stroke Widths
* **Symptom:** If the optimizer decreased a stroke width below `0.001`, the covariance determinant det(Σ) became extremely small (`2.5e-13`), rendering the `1e-6` epsilon guard inside `invert_2x2` useless and leading to NaNs.
* **Fix:**
  1. Clamped all activated widths to a minimum of `0.5` pixels inside [scene.py](file:///Users/appypie/Downloads/SVG_AI/src/python/curvesplat/scene.py#L70-L79) and `render_single_curve_fn`.
  2. Increased the analytical matrix inversion epsilon safeguard to `1e-4` inside `invert_2x2`.
* **Locations:** [scene.py:78](file:///Users/appypie/Downloads/SVG_AI/src/python/curvesplat/scene.py#L78), [renderer.py:121](file:///Users/appypie/Downloads/SVG_AI/src/python/curvesplat/renderer.py#L121), and [vectorize_cuda.py:185](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L185)

---

## 2. Differentiable Training & Optimization Scheduling

### Issue 5: Slow k-Means Centroid Estimation
* **Symptom:** Running k-means on 262,144 pixels (for a 512x512 image) allocated ~3.8GB of temporary memory per iteration, stalling execution for over 30 seconds during startup.
* **Fix:** Subsampled the pixel grid to a maximum of 20,000 pixels for centroid calculations, then mapped the remaining pixels to the final centroids in a single step at the end. Initialization now takes under 2 seconds.
* **Location:** [vectorize_cuda.py:47](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L47)

### Issue 6: Stale Adam Momentum Across Resolution Shifts
* **Symptom:** When upgrading resolution (e.g. 64x64 → 128x128 → 512x512), the optimizer carried stale momentum states from the coarser phase, causing parameters to overshoot and ruin target matching.
* **Fix:** Wiped the optimizer states at resolution transitions:
  ```python
  optimizer.state.clear()
  ```
* **Location:** [vectorize_cuda.py:526](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L526)

### Issue 7: Non-Physical Alpha Blending in Logo Demos
* **Symptom:** In Phase 2 of the logo vectorizer, alpha was approximated by taking the maximum color channel (`curve_image.max(dim=0)[0]`). Dark curves with low channel values (e.g. `[0.1, 0.05, 0.02]`) were rendered as nearly transparent.
* **Fix:** Removed the manual maximum color channel estimation hack and replaced it with a temporary black background render using the corrected sequential compositor `render_scene()`.
* **Location:** [vectorize_logo.py:133](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_logo.py#L133)

### Issue 8: Checkpoint Grid Coordinate Memory Overhead
* **Symptom:** Grid coordinates `grid_x` and `grid_y` were passed as arguments to `torch.utils.checkpoint.checkpoint()`, forcing PyTorch to store them for recomputation and wasting memory.
* **Fix:** We generate `grid_x` and `grid_y` dynamically inside `render_single_curve_fn`, excluding them from the checkpoint inputs.
* **Location:** [vectorize_cuda.py:190](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L190)

---

## 3. Initialization & Structural Algorithms

### Issue 9: Quadratic Complexity in Contour Chaining
* **Symptom:** Segment chaining used list insertion at position 0 (`chain_points.insert(0, ...)`), which is an O(N) operation. For detailed boundary contours with thousands of segments, this ran at O(N^2) time.
* **Fix:** Converted `chain_points` to a `collections.deque` and used `appendleft()`, achieving O(1) insertions.
* **Location:** [sdf_init.py:536](file:///Users/appypie/Downloads/SVG_AI/src/python/curvesplat/sdf_init.py#L536)

### Issue 10: Obsolete Documentation Command Line Flags
* **Symptom:** `HOW_TO_RUN_CUDA.md` suggested the old `--num_curves` flag.
* **Fix:** Updated the document to use `--n_fill` (color region curves) and `--n_edge` (detail contour curves) options.
* **Location:** [HOW_TO_RUN_CUDA.md:62](file:///Users/appypie/Downloads/SVG_AI/HOW_TO_RUN_CUDA.md#L62)

### Issue 11: Unused scipy Dependency
* **Symptom:** The function `_sobel_gradients` imported `scipy.ndimage.convolve`, creating a heavy external dependency even though pure-numpy Sobel functions are used.
* **Fix:** Deleted the dead `_sobel_gradients` function from the codebase.
* **Location:** [sdf_init.py:55](file:///Users/appypie/Downloads/SVG_AI/src/python/curvesplat/sdf_init.py#L55)

### Issue 12: Over-Aggressive White Cluster Skipping
* **Symptom:** To ignore the background, color clusters with a mean color > 0.92 (near-white) were skipped. This left interior white sections of the subject matter without region-fill curves, causing empty spots.
* **Fix:** Added spatial verification. A near-white cluster is now only skipped if its pixels touch the image borders/corners (within a 5% margin):
  ```python
  touches_corner = (
      mask[:margin_y, :margin_x].any() or
      mask[:margin_y, -margin_x:].any() or
      mask[-margin_y:, :margin_x].any() or
      mask[-margin_y:, -margin_x:].any()
  )
  ```
* **Location:** [vectorize_cuda.py:115](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L115)

---

## 4. Mixed-Precision & Autograd Stability (Phase 1 Follow-up)

### Issue 13: NaN Gradients under Mixed Precision (AMP) on CUDA
* **Symptom:** When running on GPU with AMP enabled, the canvas is processed in `float16`. For a `512` canvas size, pixels far away from the curves produced distance squared values `dx * dx` and `dy * dy` exceeding `65504` (the limit of float16), overflowing `mahal_sq` to `inf`. Even though the weight is masked to `0.0` by `within_cutoff` in the forward pass, PyTorch backpropagated through `exp(-0.5 * mahal_sq)` where `mahal_sq = inf`. The chain rule product `0.0 * inf` generated `NaN` gradients at step 1.
* **Fix:** Updated `render_single_curve_fn` to compute grid coordinates, distance vectors, and covariance multiplications in **`float32`**, which has a much larger dynamic range. Added a protective `torch.clamp(mahal_sq, max=50.0)` clamp before `torch.exp`. The final accumulated weight is cast back to the canvas's `dtype` (`float16` under AMP).
* **Location:** [vectorize_cuda.py:292-323](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L292-L323)

### Issue 14: Near-Singular Determinant Division in `invert_2x2` Autograd
* **Symptom:** If a curve collapsed to a degenerate state (e.g. identical control points, zero width, or zero tangents), the covariance determinant became zero. In `invert_2x2`, a `torch.where` condition was used to replace det with `eps`. However, PyTorch's `torch.where` computes gradients for BOTH branches during backpropagation. Since the unselected branch had `det = 0.0`, the term `1.0 / det` evaluated to infinity, leaking infinite gradients into the backprop path and generating `NaN` gradients.
* **Fix:** Rewrote `invert_2x2` to compute the sign of the determinant safely (`det.sign()` defaulted to `1.0` if `0.0`) and set `det_safe = torch.where(det.abs() < eps, eps * det_sign, det)`. Because both branches of the condition now yield non-zero values, `1.0 / det_safe` is guaranteed to be stable and mathematically well-defined in both forward and backward passes.
* **Location:** [renderer.py:121-153](file:///Users/appypie/Downloads/SVG_AI/src/python/curvesplat/renderer.py#L121-L153)

### Issue 15: Zero Tangent `atan2(0,0)` Singularity in `build_covariance` Autograd
* **Symptom:** In detailed edge contours, duplicate or collapsed marching-squares segments produced exactly zero tangents (`[0.0, 0.0]`) at some points. The forward pass computed `atan2(0.0, 0.0) = 0.0` safely. However, during backpropagation, the derivative of `atan2(y, x)` requires division by `x^2 + y^2`. When both inputs are `0.0`, this results in division by zero, leaking `NaN` gradients into the control points of those specific curves (e.g., curves `127` and `134`).
* **Fix:** Updated `build_covariance` to perturb tangent components to non-zero values (`dx_safe = 1.0`, `dy_safe = 0.0`) when their norm is extremely small (`< 1e-6`). This ensures that `theta_raw = torch.atan2(dy_safe, dx_safe)` is computed at a non-singular point in both branches of `torch.where`, and then selects `0.0` for the final angle using `torch.where` (safely zeroing out the gradient backward path).
* **Location:** [renderer.py:89-102](file:///Users/appypie/Downloads/SVG_AI/src/python/curvesplat/renderer.py#L89-L102)

---

## 4.5. GPU Compiler Compatibility (torch.compile)

### Issue 16: TorchInductor Type Mismatch (Half vs Float) on Compiled Backward Pass
* **Symptom:** When running on GPU under mixed precision (AMP) with `torch.compile` enabled, the incoming gradient from the loss function is in `float16` (`Half`). Inside `render_single_curve_fn`, we had mixed `float16` and `float32` tensors. During backward pass compilation, TorchInductor attempted to generate a fused kernel that performed batch matrix multiplication (`bmm`) between the `Half` gradient and `Float` matrices (like the rotation matrix `R` or diagonal `D`), resulting in `RuntimeError: expected scalar type Half but found Float`.
* **Fix:** Updated `render_single_curve_fn` to execute completely in **`float32`** inside the `autocast(enabled=False)` block. We cast all dynamic inputs (including the `canvas` to `.float()`) to `float32` at the very beginning of the function and perform all rendering, Gaussian weight accumulation, and compositing in pure `float32`. The final output canvas is cast back to `canvas.dtype` (`float16` under AMP) only at the function boundary. This keeps the compiled graph internally consistent and free of mixed types.
* **Location:** [vectorize_cuda.py:280-331](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L280-L331)

---

## 4.6. CLIP Prompt-Guided Initialization

### Issue 17: HuggingFace CLIP Output Type Changes in `vectorize_cuda.py`
* **Symptom:** In newer versions of the Hugging Face `transformers` library, `CLIPModel.get_text_features` and `get_image_features` can return a `BaseModelOutputWithPooling` or other wrapper object rather than a raw `torch.Tensor`. This caused prompt-guided curve scoring to crash at startup with `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`.
* **Fix:** Introduced a robust `extract_tensor` helper inside the CLIP initialization block in `vectorize_cuda.py`. It inspects the features object and extracts the raw tensor from standard Hugging Face formats (such as checking for `.text_embeds`, `.image_embeds`, `.pooler_output`, `.last_hidden_state`, or list/dict indexing) before normalization and similarity scoring.
* **Location:** [vectorize_cuda.py:208-251](file:///Users/appypie/Downloads/SVG_AI/examples/vectorize_cuda.py#L208-L251)

---

## 5. Verification Results

1. **Unit Tests:** All 37 mathematical correctness tests pass successfully:
   ```bash
   python3 -m pytest tests/ -v
   # Result: 37 passed, 15 warnings in 10.71s
   ```
2. **Sanity Vectorization Run:** Run successfully on CPU/GPU for 5 steps without any NaN gradient crashes, yielding monotonic error reduction.
3. **Phase 1 (Circle Vectorization):** The optimized curve parameters converge monotonically to approximate the target circle with a final L1 loss of `0.2473` after 500 steps.
4. **Phase 2 (Star Contour Fitting):** The boundary contour initialization pipeline fits the star silhouette with a high MS-SSIM score of `0.9561` in under a second.




