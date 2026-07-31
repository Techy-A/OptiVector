
# CURVESPLAT DEVELOPMENT ROADMAP
## Comprehensive Action Plan for Differentiation & Optimization

================================================================================
PHASE 0: BASELINE ASSESSMENT (Week 0)
================================================================================

### Immediate Tasks (Before Any Upgrades)
1. Run CurveSplat on 5 diverse test images:
   - Simple flat-color illustration (e.g., logo)
   - Photograph with gradients (e.g., portrait)
   - Complex organic artwork (e.g., your phoenix)
   - Line art / sketch
   - Text-heavy image

2. Record baseline metrics for each:
   - Training time (per phase: coarse/mid/fine)
   - Final L1 loss, MS-SSIM score
   - Curve count (initial, final, pruned)
   - SVG file size
   - Visual inspection notes (artifacts, ghost curves, self-intersections)

3. Profile GPU utilization during training:
   - Is rasterization the bottleneck?
   - Is gradient computation the bottleneck?
   - Is memory bandwidth saturated?

4. Establish evaluation protocol:
   - LPIPS score (perceptual similarity)
   - FID score (if you have a reference dataset)
   - User study: show 3 outputs (yours, DiffVG, Bézier Splatting) to 10 people, ask which looks best

================================================================================
PHASE 1: QUICK WINS (Weeks 1-2)
================================================================================

---
TASK 1.1: Implement LPIPS Perceptual Loss
---

WHY: MS-SSIM captures structure but misses texture and semantic content. LPIPS
     aligns with human perception. Bézier Splatting evaluates with LPIPS but
     doesn't optimize with it — you can be first.

IMPLEMENTATION:
  pip install lpips

  In your loss function:
    loss = λ_l1 * L1(rendered, target) 
         + λ_msssim * (1 - MS_SSIM(rendered, target))
         + λ_lpips * LPIPS(rendered, target)

  Recommended weights:
    Phase Coarse: λ_l1=1.0, λ_msssim=0.0, λ_lpips=0.0
    Phase Mid:    λ_l1=0.5, λ_msssim=0.5, λ_lpips=0.0
    Phase Fine:   λ_l1=0.3, λ_msssim=0.4, λ_lpips=0.3

VALIDATION:
  - Compare LPIPS scores before/after on your 5 test images
  - Target: 10-20% LPIPS improvement
  - Check training time impact: LPIPS adds ~5-10% overhead (acceptable)

---
TASK 1.2: Opacity-Aware Gradient Masking
---

WHY: Curves with opacity < 0.05 still consume gradient bandwidth and create
     "ghost curves" that clutter the SVG without contributing visually.

IMPLEMENTATION:
  At each training step:
    1. Track moving average opacity per curve (EMA with α=0.9)
    2. If EMA_opacity < 0.05 for >50 consecutive steps:
       - Zero out gradients for that curve's parameters
       - Mark as "dormant"
    3. If dormant curve's opacity rises above 0.1:
       - Reactivate gradients
    4. During pruning (every 100 steps):
       - Prioritize dormant curves for deletion

VALIDATION:
  - Measure: curves marked dormant vs. final curve count
  - Target: 15-30% reduction in final curve count
  - Visual: fewer "fuzzy" artifacts in output

---
TASK 1.3: Error-Map Connected Component Spawning
---

WHY: Your 8×8 grid spawns curves blindly. Connected components target actual
     shape boundaries and texture regions.

IMPLEMENTATION:
  Replace grid spawn with:
    1. Compute error map: |rendered - target| (per-pixel absolute difference)
    2. Apply Gaussian blur (σ=2px) to smooth noise
    3. Threshold at mean + 0.5*std to get binary mask
    4. Find connected components (scipy.ndimage.label)
    5. For each component:
       - Compute centroid (spawn center)
       - Compute major axis orientation (curve direction)
       - Compute area (number of curves to spawn: area / threshold)
    6. Spawn curves with:
       - Position = component centroid
       - Orientation = major axis angle
       - Color = mean color of target in component region
       - Width = proportional to component bounding box

VALIDATION:
  - Compare: grid spawn vs. component spawn on same image
  - Measure: reconstruction loss after 100 steps post-spawn
  - Target: 20-30% faster error reduction

================================================================================
PHASE 2: CORE ARCHITECTURE IMPROVEMENTS (Weeks 3-5)
================================================================================

---
TASK 2.1: Learned Resolution Scheduler
---

WHY: Fixed 64→128→full is rigid. Simple images converge in 2 phases; complex
     ones need 4+. A learned scheduler adapts per-image.

IMPLEMENTATION (Heuristic Version - No Training Needed):

  State at each step:
    - current_resolution
    - loss_history (last 50 steps)
    - step_count_in_phase

  Upsample trigger:
    if step_count_in_phase > min_steps:
      loss_slope = linear_regression_slope(loss_history)
      if loss_slope > -threshold:  # loss is plateauing
        upsample()
        reset_lr(lr * 0.7)

  Parameters to tune:
    - min_steps per phase: [100, 150, 200]
    - threshold: [-0.001, -0.0005, -0.0001]
    - lr_decay: [0.5, 0.7, 0.8]

VALIDATION:
  - Test on 20 images with different complexity
  - Measure: total training time vs. fixed scheduler
  - Measure: final loss vs. fixed scheduler
  - Target: 20-40% time reduction with equal or better quality

---
TASK 2.2: Topology Regularization (Anti Self-Intersection)
---

WHY: Aggressive gradients push control points past each other, creating
     self-intersecting curves that look broken in SVG output.

IMPLEMENTATION (Winding Number Loss):

  For each cubic Bezier curve with control points P0, P1, P2, P3:
    1. Sample N points along the curve (N=20)
    2. Compute polygon area using shoelace formula
    3. If area sign changes between segments → potential self-intersection
    4. Add loss term: λ_topo * max(0, -dot(cross(P1-P0, P2-P1), cross(P2-P1, P3-P2)))

  Alternative (Simpler - from LIVE paper):
    Xing loss = sum over all pairs of curve segments of:
              exp(-distance_between_segments² / σ²) * intersection_penalty

VALIDATION:
  - Count self-intersections before/after (use shapely.geometry for detection)
  - Target: >90% reduction in self-intersections
  - Check: does regularization hurt convergence speed? (should be <5% slower)

---
TASK 2.3: SVG Quantization & Merge Pass
---

WHY: Differentiable vectorizers output thousands of micro-curves. Real SVGs
     need to be small, load fast, and render efficiently.

IMPLEMENTATION (Post-Optimization):

  Step 1: Snap Control Points
    - Define grid resolution (e.g., 0.5% of image dimension)
    - Round each control point to nearest grid point
    - Remove duplicate points

  Step 2: Merge Similar Curves
    - Group curves by color (ΔRGB < 5)
    - Within color group, merge curves where:
      - Endpoints are within 2px
      - Tangent directions align (angle < 15°)
    - Replace pair with single blended curve

  Step 3: Simplify Paths
    - Rasterize each merged path to pixel coordinates
    - Apply Ramer-Douglas-Peucker (ε=1px)
    - Re-fit Bezier to simplified points

  Step 4: Layer Sorting
    - Sort curves by opacity (descending) for correct compositing
    - Group into <g> elements by color similarity

VALIDATION:
  - Measure: SVG file size before/after
  - Measure: browser render time (Chrome DevTools Performance tab)
  - Target: 50-70% file size reduction, <2% visual quality loss

================================================================================
PHASE 3: DIFFERENTIATORS (Weeks 6-10)
================================================================================

---
TASK 3.1: Semantic Layer Decomposition (YOUR MOAT)
---

WHY: Nobody in differentiable vectorization uses semantic segmentation to
     pre-decompose images. This is your defensible differentiator.

IMPLEMENTATION:

  Step 1: Segmentation (Pre-Processing)
    - Use SAM (Segment Anything Model) or MobileSAM for speed
    - Or use CLIP-based segmentation (e.g., CLIPSeg) for prompt-guided regions
    - Input: target image + optional text prompt
    - Output: N binary masks (e.g., "bird body", "fire wings", "background")

  Step 2: Layer-Specific Optimization
    - For each mask:
      - Crop target to mask bounding box
      - Run CurveSplat with layer-specific hyperparameters:
        - Background: 50 curves, wide strokes, low opacity
        - Body: 200 curves, medium strokes, high opacity
        - Details (eyes, claws): 300 curves, thin strokes, very high opacity
    - Each layer optimized independently (parallelizable across GPUs)

  Step 3: Layer-Aware Compositing
    - Composite all layers with z-order from segmentation depth
    - Apply layer-specific blur/opacity adjustments
    - Final global refinement (optional): 50 steps at full resolution

HYPERPARAMETER TABLE (Per Layer):
  | Layer Type    | Curve Budget | Stroke Width | Opacity | LR Multiplier |
  |---------------|-------------|--------------|---------|---------------|
  | Background    | 50-100      | 5-10px       | 0.3-0.5 | 0.5x          |
  | Mid-ground    | 150-250     | 2-5px        | 0.6-0.8 | 1.0x          |
  | Foreground    | 200-400     | 1-3px        | 0.8-1.0 | 1.2x          |
  | Fine Details  | 100-200     | 0.5-1px      | 0.9-1.0 | 1.5x          |

VALIDATION:
  - Compare: single-layer vs. multi-layer on same image
  - User study: which looks more "intentional" and "artistic"?
  - Target: Multi-layer rated significantly better for complex images

---
TASK 3.2: Style-Aware Vectorization
---

WHY: "Vectorize this photo in that style" is an unsolved problem. Current tools
     either trace mechanically or generate from scratch.

IMPLEMENTATION:

  Step 1: Style Encoder
    - Use pre-trained VGG features (layers conv1_1, conv2_1, conv3_1, conv4_1)
    - Or use a small StyleGAN2 layer for style embedding
    - Input: reference style image (e.g., "ink wash painting")
    - Output: 512-dim style vector

  Step 2: Style Loss Terms
    Add to optimization objective:
      - Gram matrix loss: ||Gram(VGG(rendered)) - Gram(VGG(style_ref))||²
      - Stroke statistics loss: match stroke width distribution
      - Color palette loss: match dominant colors in style reference

  Step 3: Style-Specific Initialization
    - Analyze style reference:
      - Count dominant colors → set region-fill palette
      - Measure stroke density → set initial curve count
      - Measure edge sharpness → set Canny threshold

VALIDATION:
  - Test styles: "flat vector", "ink wash", "sketch", "watercolor"
  - User study: rate style faithfulness 1-10
  - Target: Average rating >7/10 for recognizable styles

---
TASK 3.3: Real-Time Preview via Neural Proxy
---

WHY: 1-3 minute wait kills UX. A lightweight CNN can predict the final result
     in <100ms for interactive preview.

IMPLEMENTATION:

  Step 1: Data Collection
    - Run CurveSplat on 1000+ images, saving checkpoints every 50 steps
    - For each: save (target_image, curve_state_at_step_k, final_rendered_output)
    - Curve state = rasterized curve mask + parameter tensor

  Step 2: Proxy Model
    - Architecture: U-Net or lightweight EfficientNet
    - Input: (target_image, current_curve_render) concatenated
    - Output: predicted final render
    - Loss: L1 + LPIPS against actual final render

  Step 3: Integration
    - During training: show proxy prediction every 10 steps
    - After training: use proxy for "quick preview" mode
    - Full optimization runs in background, swaps to real result when done

VALIDATION:
  - Measure: proxy prediction time (<100ms target)
  - Measure: proxy accuracy (LPIPS to final result < 0.15)
  - User feedback: does preview feel "responsive"?

================================================================================
PHASE 4: BENCHMARKING & PUBLICATION (Weeks 11-14)
================================================================================

---
TASK 4.1: Build Evaluation Dataset
---

Create "CurveBench" — a standardized benchmark for differentiable vectorization:
  - 100 images across categories: logos, portraits, landscapes, illustrations, photos
  - Each with: target PNG, human-annotated "ideal" SVG (for reference), complexity score
  - Metrics: LPIPS, FID, curve count, SVG size, render time, user preference

---
TASK 4.2: Comparative Benchmarking
---

Run your final CurveSplat against:
  - DiffVG (baseline differentiable rasterizer)
  - Bézier Splatting (2025, fastest)
  - LIVE (CVPR 2022, layer-wise)
  - Potrace (traditional tracer)
  - Adobe Image Trace (commercial baseline)

Document:
  - Speed vs. quality tradeoff curves
  - Failure cases (where does each method break?)
  - User preference rankings

---
TASK 4.3: Paper / Blog Post
---

Write up your differentiators:
  - CLIP-guided semantic initialization
  - Shuffled compositing for gradient flow
  - Semantic layer decomposition
  - Style-aware vectorization

Target venues:
  - SIGGRAPH / CVPR (if results are strong)
  - arXiv + Hacker News / Reddit (for visibility)
  - Technical blog with interactive demos

================================================================================
CRITICAL PATH SUMMARY
================================================================================

WEEK 1-2:  [QUICK WINS] LPIPS + Opacity Masking + Better Spawning
WEEK 3-5:  [ARCHITECTURE] Learned Scheduler + Topology + SVG Cleanup
WEEK 6-8:  [MOAT #1] Semantic Layer Decomposition
WEEK 9-10: [MOAT #2] Style-Aware Vectorization
WEEK 11-12:[UX] Neural Proxy Preview
WEEK 13-14:[SHIP] Benchmarking + Publication

================================================================================
DECISION GATES
================================================================================

After Week 2:
  → Did LPIPS improve perceptual quality? If not, debug weights/architecture.
  → Did opacity masking reduce curve count? If not, adjust thresholds.

After Week 5:
  → Is learned scheduler actually faster? If not, fall back to fixed schedule.
  → Is SVG cleanup preserving quality? If not, relax merge thresholds.

After Week 8:
  → Does semantic decomposition beat single-layer? If not, simplify segmentation.
  → Is the overhead worth it? (SAM adds ~2s per image — acceptable?)

After Week 10:
  → Do users care about style-aware? If not, deprioritize, focus on semantic layers.

================================================================================
RISK MITIGATION
================================================================================

RISK: SAM/CLIP integration is too slow for real-time use
  MITIGATION: Use MobileSAM (10× faster) or cache segmentation results

RISK: LPIPS loss destabilizes training
  MITIGATION: Start with λ_lpips=0.1, gradually increase. Use gradient clipping.

RISK: Semantic layers create visible seams at boundaries
  MITIGATION: Add 10px overlap between layers, blend with soft masks

RISK: Style loss makes output look "generic"
  MITIGATION: Weight style loss low (λ=0.1) and increase content loss weight

RISK: Neural proxy is inaccurate
  MITIGATION: Only show proxy for first 50% of training; switch to actual render after

================================================================================
METRICS DASHBOARD
================================================================================

Track these weekly:
  ┌─────────────────────────┬──────────┬──────────┬──────────┐
  │ Metric                  │ Baseline │ Target   │ Current  │
  ├─────────────────────────┼──────────┼──────────┼──────────┤
  │ Training time (phoenix) │ 3 min    │ 1 min    │ ___      │
  │ Final LPIPS             │ ___      │ -20%     │ ___      │
  │ Curve count (phoenix)   │ ___      │ -25%     │ ___      │
  │ SVG file size           │ ___      │ -50%     │ ___      │
  │ Self-intersections      │ ___      │ -90%     │ ___      │
  │ User preference (1-10)  │ ___      │ 8+       │ ___      │
  └─────────────────────────┴──────────┴──────────┴──────────┘

================================================================================
