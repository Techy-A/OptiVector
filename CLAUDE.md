# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes, plus project-specific
facts for CurveSplat. Read both sections before working.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility"/"configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

## 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- Remove orphans YOUR changes created; mention pre-existing dead code, don't delete it.

## 4. Goal-Driven Execution
Define success criteria. Loop until verified.

- "Add validation" → "Write tests for invalid inputs, then make them pass."
- "Fix the bug" → "Write a test that reproduces it, then make it pass."
- "Refactor X" → "Ensure tests pass before and after."
- Don't trust a metric without looking: on sparse images (subject on white),
  L1/MS-SSIM are dominated by background — **look at the rendered image**, not just numbers.

---

## Project: CurveSplat — differentiable image→SVG vectorizer

Hybrid renderer: crisp **filled shapes** (tracer-quality flat regions) + painterly
**Gaussian-splat strokes**, optimized by gradient descent, exported to SVG.

### Layout
- `src/python/curvesplat/` — `renderer.py` (splat math + `render_curves_batched` reference),
  `fill_renderer.py` (filled closed-loop SDF), `cuda_splat.py` (CUDA fast path + JIT loader),
  `scene.py` (`Curve`, `FilledShape`, `Scene`), `sdf_init.py` (init + `initialize_filled_regions`),
  `losses.py`, `segmentation.py`, `style.py`, `proxy.py`.
- `src/cuda/splat_cuda.cu` — CUDA alpha kernel (forward + analytic backward).
- `examples/vectorize_cuda.py` — main GPU pipeline (`render_scene_shuffled`, `export_to_svg`, `main`).
- `tests/` — run `python3 -m pytest tests/ -q` (CUDA tests skip without a GPU).

### Renderer paths (all numerically equivalent — gradchecked)
1. `render_cuda` (CUDA kernel) — fast path, used when `HAS_CUDA_SPLAT` + CUDA.
2. `render_curves_batched` — pure-PyTorch **reference + fallback**. Keep it; tests gate against it.
3. `render_curves_tiled` — exists but unused (per-curve `.item()` syncs make it slow on GPU).
Composite (alpha-over) is always done in PyTorch via reverse-cumprod.

### Performance facts (learned the hard way — don't re-derive)
- Pure-PyTorch dense render = ~9s/step at 512px (eager can't fuse; `torch.compile` graph-breaks).
- CUDA kernel → **~0.36s/step (~25×)**. This is the only thing that broke the wall.
- The fill renderer's autograd memory scales with `#segments · samples · H · W`. Uncapped
  contours OOM at full res → segments capped (`MAX_SEG=48` in `sdf_init._fit_closed_loop`),
  `scene.fill_samples=12`. Don't remove these caps without bounding memory another way.
- Adaptive-density ablation re-renders the scene once per curve → memory-heavy on a shared GPU.

### Remote GPU run (RTX A6000, aidev2) — known-good
```bash
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH   # /usr/bin/nvcc is 11.5 — too old, breaks the build
pip install ninja                                                          # JIT needs it
CUDA_VISIBLE_DEVICES=1 python3 -u examples/vectorize_cuda.py \             # GPU0 hogged by another user
  --image phoenix.png --output_dir examples/output --mode hybrid \
  --n_fill 100 --n_edge 100 --steps 400 --max_dim 384 --no_adaptive \
  --lambda_edge 0.6 --polish_steps 100 --log_every 10
```
- `rsync` runs on the Mac (`/Users/appypie/Downloads/SVG_AI/`), everything else over SSH.
- First import compiles `src/cuda/splat_cuda.cu` (~30-60s, cached).

### Output-quality reality
- Watercolor/photo with soft gradients = **worst case** for crisp fills (few flat regions).
  Expect painterly-impressionist, not tracer-crisp. For clean flat art, more fills help.
- Quality levers: more curves (`--n_fill/--n_edge`), more `--steps`, full `--max_dim 512`,
  thinner `--max_width_edge`, higher `--lambda_edge`, re-enable adaptive (drop `--no_adaptive`).

### Stale docs (flagged, not cleaned — don't trust blindly)
- `Agents.md`, `Frontend.md`, `brandguidelines.md` are generic/pasted skill files, not
  CurveSplat-specific. `HOW_TO_RUN_CUDA.md` has a broken torch URL (`/ml/cu121` → `/whl/cu121`).
  `SESSION_SUMMARY.md` has the accurate current state.
