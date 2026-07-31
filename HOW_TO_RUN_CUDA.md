# Running CurveSplat on a Remote RTX A6000 GPU (via SSH)

This guide details how to move the CurveSplat codebase to a remote SSH environment, install dependencies, run the GPU-accelerated vectorizer, and download the resulting SVG artwork back to your local machine.

---

## 1. Preparing and Uploading the Codebase

From your **local terminal** (on your local computer, not inside the SSH server), upload the `SVG_AI` project directory to the remote server using `scp` or `rsync`:

```bash
# Option A: Using rsync (highly recommended, copies and updates files)
rsync -avz --exclude '.pytest_cache' --exclude 'examples/output' /Users/appypie/Downloads/SVG_AI/ username@remote_host:/path/to/destination/SVG_AI/

# Option B: Using scp (traditional copy)
scp -r /Users/appypie/Downloads/SVG_AI username@remote_host:/path/to/destination/
```

*Replace `username`, `remote_host`, and `/path/to/destination/` with your actual remote SSH server details.*

---

## 2. Uploading Your Target Image to Test

To test the script with your own picture (e.g., `my_photo.png`), upload it to the `SVG_AI` directory on the remote server:

```bash
scp /Users/appypie/Downloads/my_photo.png username@remote_host:/path/to/destination/SVG_AI/
```

---

## 3. Setting Up the Remote CUDA Environment

Log into your remote server via SSH:
```bash
ssh username@remote_host
```

Once inside the SSH shell, change directory into the uploaded folder:
```bash
cd /path/to/destination/SVG_AI/
```

### Install Python Dependencies
The remote machine must have PyTorch configured with CUDA support. Run:

```bash
# Install PyTorch with CUDA (matches most standard A6000 setups)
pip install torch --index-url https://download.pytorch.org/ml/cu121

# Install image processing, MSSSIM, and plotting libraries
pip install matplotlib pytorch-msssim
```

---

## 4. Running the GPU Differentiable Optimizer

Run the GPU script `examples/vectorize_cuda.py`. It runs the edge-contour initialization and fits the Bezier curves iteratively using PyTorch Autograd on the RTX A6000.

### Mixed Precision & Performance (AMP)
By default, the script runs with **Automatic Mixed Precision (AMP)** enabled to accelerate the render forward pass using `fp16` while preserving `fp32` parameters for numerical stability. This delivers a 2-3x speedup on CUDA. To disable mixed precision, pass the `--no_amp` flag.

### Default Run (Uses 100 region fill curves and 100 edge contour curves, runs for 500 steps):
```bash
python3 examples/vectorize_cuda.py --image my_photo.png
```

### High-Fidelity Run (Uses 150 fill + 150 edge curves, runs for 1000 steps for maximum detail):
```bash
python3 examples/vectorize_cuda.py --image my_photo.png --steps 1000 --n_fill 150 --n_edge 150 --lr 0.03
```

### Prompt-Guided Initialization
You can guide the curve initialization by providing a text description prompt. This downloads the CLIP model, generates double the candidate region-fill curves, and retains the top 50% highest-similarity candidates to target relevant image regions:
```bash
python3 examples/vectorize_cuda.py --image my_photo.png --prompt "phoenix bird with fire wings"
```

### Running Benchmarks
To evaluate performance directly (reports warm-up and timed step average FPS), run:
```bash
python3 examples/vectorize_cuda.py --image my_photo.png --benchmark
```

### Running on a Specific GPU (e.g., GPU 1)
If GPU 0 is occupied by a persistent background/system service, you can target GPU 1 (which has more free memory) by prepending `CUDA_VISIBLE_DEVICES=1` to your execution command:

```bash
# Run on GPU 1
CUDA_VISIBLE_DEVICES=1 python3 examples/vectorize_cuda.py --image my_photo.png --steps 1000 --n_fill 150 --n_edge 150 --lr 0.03
```

### Output Logs:
The script will output performance updates in steps of 50. Thanks to mixed precision and sample optimization, multi-curve fitting is highly performant.

---

## 5. Downloading Outputs Back to Local Machine

When the script completes, it saves two files under `examples/output/`:
- `cuda_vectorization_comparison.png` — Side-by-side target vs. optimized render.
- `optimized_artwork.svg` — The final vectorized SVG file.

To copy these outputs from the remote server back to your **local computer**, run these commands from your **local terminal**:

```bash
# Download the comparison PNG
scp username@remote_host:/path/to/destination/SVG_AI/examples/output/cuda_vectorization_comparison.png /Users/appypie/Downloads/

# Download the final SVG
scp username@remote_host:/path/to/destination/SVG_AI/examples/output/optimized_artwork.svg /Users/appypie/Downloads/
```
Now, you can open `optimized_artwork.svg` in any browser or design tool (Illustrator, Figma) to view your vectorized artwork!
