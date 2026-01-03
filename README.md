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
├── frame_000000.png               # Example raw frame
├── lensless.m                     # Batch processor for lensless reconstruction
├── lensless_reconstruct.m         # Grayscale reconstruction (BP + CS)
├── lensless_reconstruct_bp_cs.m   # RGB-aware reconstruction wrapper
├── center_crop.m                  # Utility to crop reconstructed frames
├── video_frames_pipeline.py       # Python script to extract/assemble video frames
├── original_script.m              # Reference script demonstrating end-to-end steps
└── functions/                     # MATLAB helper functions
    ├── FZA.m                      # Generates Fresnel Zone Aperture mask
    ├── MyForwardOperatorPropagation.m
    ├── MyAdjointOperatorPropagation.m
    ├── TwIST.m                    # TwIST implementation
    ├── tvdenoise.m, TVnorm.m      # TV regularization helpers
    ├── pinhole.m                  # Pinhole imaging model
    ├── conv2c.m, diffh.m, diffv.m # Convolution & finite differences
    └── (additional utilities)
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
