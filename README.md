# OptiVector

OptiVector is a fast and accurate tool that converts standard images into infinitely scalable SVGs. It features intelligent layer grouping and automatic background removal to give you perfectly crisp, print-ready graphics.
## The Problem I Was Solving
Designers and developers frequently need to convert raster images (PNG, JPG, WEBP) into clean, infinitely-scalable SVGs for printing or web use. Existing automated tools either produce massive files, lose important details, or fail to cleanly separate colors into editable layers. Generative AI tools often "hallucinate" incorrect details instead of faithfully reproducing the original logo or artwork. This project was built to provide a fast, accurate, and faithful raster-to-vector conversion tool that outputs highly optimized, editable SVGs.

## My Role in the Project
As the primary developer (Techy-A), I architected and implemented the entire pipeline. This included designing the responsive HTML/CSS/JS frontend, developing the Python/FastAPI backend, and integrating the core vectorization engine using `vtracer` and image processing libraries. I also engineered the smart layer-grouping algorithm that automatically determines whether to group vectors by color (for flat logos) or separate foreground from background (for complex art) without compromising fidelity.

## The Tech Stack Used
- **Frontend**: Vanilla HTML5, CSS3, and JavaScript. No heavy frameworks, ensuring a lightweight and lightning-fast UI.
- **Backend API**: Python 3 and FastAPI for robust, asynchronous API endpoints.
- **Vectorization Engine**: Python, `vtracer` (for high-fidelity contour tracing), `resvg-py`, and Pillow (for image processing).
- **Mathematical & Data Processing**: `numpy` and `scipy`.
- **Legacy/Testing Engine (CurveSplat)**: PyTorch for differentiable rendering and optimization.

## The Outcome or Impact of the Project
The result is a fast, production-ready application that instantly converts logos and flat art into pristine vector graphics. The output SVGs are highly optimized (often around 45 KB, up to 15x smaller than competitors) and feature editable named layers that drop seamlessly into tools like Figma and Adobe Illustrator. The web UI effectively demonstrates the value of infinite scalability through a custom-built zoom comparison tool.

## Key Challenges Faced and How I Solved Them
1. **Handling Complex vs. Flat Art**: Simple logos benefit from color-grouped layers, but dense painterly art breaks when grouped by color due to overlapping strokes. **Solution**: I developed an automatic measurement system ("The Gate") that renders both the color-grouped version and the original-order version, diffs them, and safely determines the best layer structure for the image without user intervention.
2. **Performance and Concurrency**: Vectorization can be computationally expensive and block the main thread. **Solution**: I offloaded the vectorization workload in the FastAPI backend to a background worker thread, ensuring that concurrent uploads from the UI don't block each other.
3. **Frontend Security**: Rendering SVGs generated from user uploads can pose an XSS risk. **Solution**: I implemented a strict sanitization step in the vanilla JavaScript using `DOMParser` to strip out any `<script>`, `<foreignObject>`, or `on*` event handlers before displaying the result.

## How to run
Please see [`How_to_run.txt`](How_to_run.txt) for instructions on running the server and CLI tools.
