# Camera Calibration and Lens Distortion Correction Using OpenCV

## 1. Introduction

Digital cameras often introduce optical distortions that cause straight lines to appear curved or misaligned. The two primary forms of distortion are:

- **Radial distortion** — barrel or pincushion distortion
- **Tangential distortion** — caused by lens misalignment with the image sensor

OpenCV provides several functions for correcting these distortions, including:

- `cv2.undistort()`
- `cv2.calibrateCamera()`
- `cv2.findChessboardCorners()`

To perform distortion correction, two calibration outputs are required:

| Parameter | Description |
|---|---|
| `camMatrix` | Intrinsic camera parameters (focal length, optical center, etc.) |
| `distCoeff` | Lens distortion coefficients |

These values cannot be generated automatically without calibration images. They must be estimated by photographing a known calibration pattern (typically a chessboard) using the same camera configuration used for later image capture.

---

# 2. Camera Calibration Procedure

## 2.1 Preparing the Chessboard Pattern

OpenCV commonly uses black-and-white chessboard patterns because corner intersections are easy to detect accurately.

Recommended patterns:

- `9×6` inner-corner chessboard
- `7×7` inner-corner chessboard

### Calibration board requirements

- Printed clearly at high quality
- Mounted on a rigid flat surface
- Free from bending or warping

A clipboard or foam board is recommended to keep the calibration target flat.

---

## 2.2 Capturing Calibration Images

For reliable calibration:

- Capture approximately **10–20 images**
- Use different viewing angles and positions
- Move the chessboard toward image corners and edges
- Include both near and far distances
- Ensure images are sharp and well-focused
- Keep camera resolution, zoom, and focus fixed throughout calibration

Capturing the full field of view improves distortion estimation accuracy.

---

## 2.3 Detecting Chessboard Corners

OpenCV detects chessboard intersections using:

```python
cv2.findChessboardCorners()