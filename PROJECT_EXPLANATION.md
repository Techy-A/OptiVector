# CurveSplat: How It Works & How It Compares

CurveSplat is a GPU-accelerated differentiable vectorization system that optimizes Bezier curves directly to reconstruct target images using gradient descent. Below is a detailed technical summary of the system architecture and how it differs from traditional or generative approaches.

---

## 1. Core Architecture: Differentiable Rasterization

Traditional vector graphics rasterizers take curve mathematical definitions and output pixels. This is a one-way process. 

**CurveSplat** implements a **differentiable rasterizer**:
- **Parametric Curves:** Every Bezier curve is defined by parameters: control points $P_0, P_1, P_2, P_3$, stroke width, RGB color, and opacity.
- **Differentiable Splatting:** The renderer draws these curves by evaluating them as 2D Gaussian-like splats. Because every rendering operation (coordinate projection, thickness, alpha blending) is represented by differentiable mathematical functions, PyTorch can track operations on the computational graph.
- **Gradient Backpropagation:** We calculate the difference (loss) between the rendered image and the target PNG. PyTorch computes the gradient of this loss with respect to each curve parameter and uses an optimizer (like AdamW) to iteratively nudge the curves closer to the target.

---

## 2. System Pipeline

### Phase 1: Hybrid Initialization
To give the optimization a strong starting point, CurveSplat uses a dual-initialization strategy:
1. **Region-Fill Curves (Wide):** Runs a localized $k$-means color clustering on the target image. It places wide, high-opacity curves to cover large color blocks.
2. **CLIP Prompt-Guiding:** If a prompt is provided (e.g., *"phoenix bird with fire wings"*), the system generates candidate region curves, crops the target image around them, and ranks them using OpenAI's CLIP model. It retains only the top 50% highest-scoring curves, focusing them on semantically important regions of the image.
3. **Contour-Edge Curves (Thin):** Runs Canny-like edge detection and a Signed Distance Field (SDF) solver to fit thin, sharp curves directly onto the outlines of structures.

### Phase 2: Coarse-to-Fine Scheduling
Optimization is split across three phases to lock in large structures before addressing details:
- **Coarse (e.g., 64x64 pixels):** High learning rate. Curves establish broad boundaries and shapes.
- **Mid (e.g., 128x128 pixels):** Standard learning rate. Scales up detail.
- **Fine (Full resolution):** Lower learning rate. Activates **L1 Loss + MS-SSIM Loss** to sharpen edges and align high-frequency visual details.

### Phase 3: Shuffled Compositing
To prevent upper layers from blocking the gradient flow to lower layers (the vanishing gradient problem of serial compositing), CurveSplat shuffles the rendering order of the curves dynamically at every training step.

### Phase 4: Adaptive Density (Pruning & Spawning)
Every 100 steps:
- **Pruning:** Deletes the 10 lowest-contributing curves (the curves whose deletion changes the output image the least).
- **Spawning:** Calculates the reconstruction error map, divides it into an $8 \times 8$ grid, and spawns 10 new curves in the highest-error cells.

---

## 3. How It Differs From Other Systems

| Feature | CurveSplat (Differentiable Rendering) | Traditional Tracers (Potrace, Adobe Image Trace) | Generative AI Vector Tools (Text-to-SVG) |
| :--- | :--- | :--- | :--- |
| **Core Method** | Optimization via gradient descent. | Pure mathematical heuristic contour tracing. | Generative diffusion / GAN neural networks. |
| **Gradient & Shading** | **Excellent.** Natively fits gradients, complex shading, and transparency layers. | **Poor.** Can only output flat, solid color shapes. Gradients are vectorized as jagged color bands. | **Good**, but generated from scratch rather than fitting a target image. |
| **Fidelity to Target** | Highly faithful; directly minimizes visual loss against the target image. | High fidelity for simple shapes, poor for complex/shaded photographs. | Cannot easily reconstruct a specific input image. |
| **Aesthetics** | Painterly, stylized, overlapping organic layers. | Mechanical, exact, vector-node-heavy outlines. | Unpredictable, varies wildly based on model training data. |
| **Speed** | 1–3 minutes on a GPU (needs iterative training). | Instantaneous (sub-second). | 5–15 seconds (single-forward pass of a neural net). |
