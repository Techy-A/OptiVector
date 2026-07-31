# CurveSplat — How to Run Guide

This guide describes how to install dependencies, run tests, run the demos, and test CurveSplat on your own custom images.

---

## 1. Installation & Prerequisites

To run CurveSplat, you need Python 3, PyTorch, and a few image processing and testing libraries.

### Install Dependencies:
Open your terminal and run the following command to install the required Python packages:

```bash
pip install torch matplotlib pytest pytorch-msssim
```

---

## 2. Running the Demos

All demos are run from the workspace root directory using the `examples/vectorize_logo.py` runner script.

### Phase 1 Demo: Differentiable Circle Optimization
Runs 500 steps of gradient descent using the differentiable Gaussian-splatting renderer to optimize a single Bezier curve to approximate a filled circle.

```bash
python3 examples/vectorize_logo.py --phase 1
```
*Outputs saved to:* `examples/output/before.png`, `after.png`, and `comparison.png`.

---

### Phase 2 Demo: SDF Boundary Contour Initialization
Creates a 5-pointed star mask target, extracts boundary edges, computes the signed distance field via JFA, simplifies contours, segments corners, and fits 10 cubic Bezier curves to the outline.

```bash
python3 examples/vectorize_logo.py --phase 2
```
*Outputs saved to:* `examples/output/phase2_star_demo.png` showing the target star vs the filled silhouette of the initialized curves. Reports the final MS-SSIM score (typically `~0.9561`).

---

## 3. How to Test Your Own Image

You can use the Phase 2 initialization pipeline to automatically extract and fit Bezier curves to any image of yourself or a logo!

### Step 1: Copy your image to the workspace
Copy your image (e.g., `my_photo.png` or `my_photo.jpg`) into the workspace directory `/Users/appypie/Downloads/SVG_AI/`.

### Step 2: Run the vectorization script
Run the example runner using `--phase 2` and specify your image with the `--image` parameter:

```bash
python3 examples/vectorize_logo.py --phase 2 --image my_photo.png
```

### Step 3: View the output
The script will perform edge detection, compute the SDF, fit curves, and render them as colored strokes matching the original image. 

Open the visual output comparison at:
`examples/output/custom_vectorized.png`

---

## 4. Running the Tests

To verify the mathematical correctness of the autograd gradient flow and initialization algorithms, run:

```bash
# Run all unit and acceptance tests (37 tests)
python3 -m pytest tests/ -v
```
