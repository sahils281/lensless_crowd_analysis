# Lensless Crowd Analysis

## Overview

This repository contains a prototype pipeline for **privacy-preserving crowd monitoring** using lensless **Fresnel Zone Aperture (FZA)** imaging, combined with digital reconstruction and post-processing for downstream analytics (e.g., object tracking, anomaly detection).

**Workflow:**  
Scene video → frame extraction → lensless reconstruction → optional video reassembly → downstream crowd-analysis models.

The pipeline showcases two reconstruction approaches:

- **Back-Propagation (BP)**
- **Compressed Sensing (CS) using TwIST**

---

## Directory Structure

```text
.
├── frame_000000.png
├── lensless.m
├── lensless_reconstruct.m
├── lensless_reconstruct_bp_cs.m
├── center_crop.m
├── video_frames_pipeline.py
├── original_script.m
└── functions/
```

---

## Prerequisites

### MATLAB
- R2019b or later
- Image Processing Toolbox
- Add functions folder to path:
```matlab
addpath('./functions')
```

### Python
- Python 3.8+
- opencv-python
```bash
pip install opencv-python
```

---

## Workflow

### 1. Extract Frames
```bash
python video_frames_pipeline.py extract --video input.mp4 --out_dir frames_raw --num 120
```

### 2. Reconstruct Frames
Run in MATLAB:
```matlab
lensless
```

Outputs saved to:
- `frames_processed_cs`
- `frames_processed_bp`

### 3. Assemble Video
```bash
python video_frames_pipeline.py assemble --frames_dir frames_processed_cs --out output.mp4 --fps 30
```

---

## Reconstruction Details

### FZA Mask Generation
```matlab
[x, y] = meshgrid(linspace(-S/2, S/2 - S/N, N));
r2 = x.^2 + y.^2;
mask = 0.5 * (1 + cos(pi * r2 / r1^2));
```

---

## Scripts

### lensless_reconstruct.m
```matlab
[bp, cs] = lensless_reconstruct(I);
```

---

## Acknowledgments

- TwIST by **José Bioucas-Dias** and **Mário Figueiredo**
- TV denoising adapted from **Pascal Getreuer**
