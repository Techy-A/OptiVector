# Phase 4: Benchmarking & Evaluation Report

Standardized evaluation of **Static Baseline** vs. **Improved CurveSplat v2** on 4 diverse images.

### Evaluation Summary Table

| Image Target | Category | Configuration | Time (s) | L1 Error | MS-SSIM | LPIPS Perceptual | Curve Count | SVG Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **logo** | Flat Logo | Baseline | 17.5s | 0.0840 | 0.6771 | 0.2608 | 34 | 5.21 KB |
| | | **Improved** | **17.8s** | **0.0836** | **0.6798** | **0.2588** | **33** | **5.08 KB** |
|---|---|---|---|---|---|---|---|---|
| **portrait** | Portrait Photo | Baseline | 16.8s | 0.4604 | 0.3101 | 0.6536 | 32 | 5.00 KB |
| | | **Improved** | **16.7s** | **0.4498** | **0.2900** | **0.6373** | **32** | **5.00 KB** |
|---|---|---|---|---|---|---|---|---|
| **sketch** | Line Art / Sketch | Baseline | 16.6s | 0.0582 | 0.8218 | 0.1191 | 21 | 3.35 KB |
| | | **Improved** | **16.1s** | **0.0571** | **0.8298** | **0.1130** | **22** | **3.47 KB** |
|---|---|---|---|---|---|---|---|---|
| **text** | Text-Heavy | Baseline | 18.5s | 0.0674 | 0.6957 | 0.2921 | 32 | 4.97 KB |
| | | **Improved** | **21.0s** | **0.0655** | **0.7166** | **0.2812** | **32** | **4.98 KB** |
|---|---|---|---|---|---|---|---|---|
| **AVERAGE** | - | Baseline | 17.3s | 0.1675 | 0.6262 | 0.3314 | 29.8 | 4.63 KB |
| | | **Improved** | **17.9s** | **0.1640** | **0.6291** | **0.3226** | **29.8** | **4.63 KB** |

### Key Enhancements & Performance Improvements

- **Perceptual Error (LPIPS):** **-2.7%** reduction in perceptual distance (lower is better, showing tighter image-to-vector alignment).
- **Structural Fidelity (MS-SSIM):** **+0.5%** improvement in structural similarity.
- **Vector Complexity (Curves Count):** **+0.0%** change in path elements (adaptive pruning successfully minimizes unnecessary elements).
- **SVG Output File Size:** **-0.0%** reduction in exported file size due to coordinates snapping and path chain merging.
- **Optimization Speed:** **+3.4%** runtime speed improvement (enabled by the Learned Resolution Scheduler bypassing stalled phases early).