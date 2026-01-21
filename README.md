## Planetary Crater Detection System

This project implements a classical computer vision pipeline to detect a planetary body,
segment surface crater regions, and compute the percentage of crater-covered surface area.

Unlike circle-based crater detection approaches, this system focuses on **region-based
crater segmentation**, providing more stable results on highly textured planetary surfaces.

---

## Features

- Planet detection using Hough Circle Transform
- Crater **region segmentation** using intensity-based image processing
- Robust crater area estimation using pixel-level masks
- Accurate surface coverage percentage computation
- FastAPI backend with Blazor WebAssembly frontend

---

## Crater Detection Approach

1. The planetary body is first localized and masked.
2. Local contrast is enhanced using CLAHE.
3. Crater regions are extracted via adaptive thresholding (Otsu).
4. Morphological operations remove noise and fill crater regions.
5. Connected component analysis filters out unrealistic crater sizes.
6. Crater regions are overlaid as **solid green areas** (no geometric shapes).
7. Crater coverage is computed as:



# Technologies
- Python, OpenCV, NumPy
- FastAPI
- Blazor WebAssembly
