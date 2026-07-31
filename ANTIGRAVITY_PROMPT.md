# Antigravity command: kick off CurveSplat — Phase 1

Paste this as your first message to the Antigravity agent in this workspace.

---

## Context

This workspace implements **CurveSplat**, a differentiable image-to-SVG
vectorizer (Gaussian-splatting along Bezier curves, SDF initialization,
optional DiffVG-style polish).

Before doing anything else, **read these four files in full** and treat them
as binding instructions for this session:

1. `SPEC.md` — the complete technical specification, phased roadmap, math,
   and definition-of-done for each phase.
2. `Agents.md` — how you (the agent) should behave in this repo: coding
   style, commit/PR conventions, testing requirements, and any
   workflow rules.
3. `Frontend.md` — conventions for the demo frontend (structure, framework,
   state management) — relevant from Phase 8 onward, but read now so later
   work is consistent.
4. `brandguidelines.md` — visual identity (dark theme `#0A0A0A` / `#1A1A1A`
   backgrounds, neon lime `#D7FE3D` accent, sans-serif typography). Apply
   this to **any** UI, dashboard, plot legend, or demo page you create —
   never use default/unstyled components for anything user-facing.

## Task for this session

Implement **Phase 1 only**, as defined in `SPEC.md` §6 ("Pure PyTorch
splatting, no CUDA") and §12 (Definition of Done). Do not start Phase 2
work, do not write CUDA, do not scaffold the frontend yet.

Concretely:

1. Set up the repo structure from `SPEC.md` §4 (only the directories needed
   for Phase 1: `src/python/curvesplat/`, `tests/`, `examples/`).
2. Implement `scene.py`: `Curve` and `Scene` dataclasses exactly matching
   `SPEC.md` §3 (field names, types, activations — sigmoid for color/opacity,
   softplus for width).
3. Implement `renderer.py`: a pure-PyTorch `nn.Module` that differentiably
   renders a **single curve** onto a canvas using the Gaussian-splatting math
   in `SPEC.md` §7 (Bezier evaluation, tangent, rotation/covariance, per-pixel
   Mahalanobis cutoff at 3-sigma, alpha blending). No CUDA, no tiling — a
   straightforward per-pixel loop or vectorized tensor ops over the full
   canvas is fine at 32x32–256x256.
4. Write a unit test (`tests/test_splat_kernel.py` or similar) that runs
   `torch.autograd.gradcheck` on the renderer at a small canvas size (e.g.
   32x32) and asserts it passes.
5. Write `examples/vectorize_logo.py` (or a notebook-equivalent script) that:
   - creates a 256x256 target image containing a filled circle,
   - initializes a single `Curve` with random/sensible starting parameters,
   - runs AdamW for up to 500 steps minimizing L1 loss between the rendered
     output and the target,
   - saves a before/after PNG comparison to `examples/output/`.
6. Add a short `README.md` section (or top-of-file docstring) explaining how
   to run the example and what success looks like.

## Constraints

- Follow every coding convention in `Agents.md` (formatting, typing, test
  framework, commit message style — whatever it specifies).
- Keep everything in pure PyTorch — no custom CUDA kernels, no pybind11,
  even though those exist in the eventual repo structure.
- Numerical correctness over performance: this phase exists to validate the
  math and gradient flow, not to be fast.
- If any instruction in `SPEC.md` conflicts with `Agents.md`, follow
  `Agents.md` for *how* to write code and `SPEC.md` for *what* the code must
  compute, and flag the conflict explicitly in your summary at the end.

## Definition of done (must all be true before you stop)

- [ ] `scene.py`, `renderer.py` exist and match `SPEC.md` §3 and §7
- [ ] `gradcheck` test passes
- [ ] Example script runs end-to-end and produces a before/after image where
      the rendered curve visibly approximates the circle
- [ ] A short summary of what was built, how to run it, and any deviations
      from the spec (with justification)

Do not proceed to Phase 2 (`SPEC.md` §6, SDF initialization) without an
explicit go-ahead.
