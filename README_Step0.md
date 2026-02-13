============================
STEP 0 — AFFINE TRANSFORMATION
============================

PURPOSE
---------------------------
Step 0 transform input images by:
1) Detecting the color card using SAM2
2) Estimating an AFFINE TRANSFORMATION from the detected card
3) Warping the image to reduce tilt and perspective distortion
4) Resizing all outputs to 4000 x 6000

Output of Step 0 is REQUIRED for Steps 1, 2, and 3.

INPUTS
---------------------------
Image directory containing raw images (.jpg, .png)

Priority for input folder:
1) Command-line argument: --image-dir PATH
2) Environment variable: FABA_IMAGES_DIR
3) Default (relative to script): ../faba_images

============================
HOW TO RUN
============================
!! Before running, copy the script to the SAM2 folder:

   cd sam2
   cp ../Step0_AffineTransformation.py .

Run the script:

   python Step0_AffineTransformation.py

Optional arguments:

   python Step0_AffineTransformation.py --image-dir /path/to/images
   python Step0_AffineTransformation.py --image-dir /path/to/images --max-images 10

ARGUMENTS
---------------------------
--image-dir       Path to raw input images
--max-images      Process only first N images (for testing)

OUTPUTS
---------------------------
1) Affine-corrected images
   - Directory: ../corrected_images_affine/
   - Filename: same as input
   - Color: BGR
   - Size: 4000 x 6000

2) Color card masks (saved by default)
   - Directory: ../corrected_images_mask_affine/
   - Filename format: mask_<original_filename>.png
   - Binary mask (255 = card)
   - Useful for debugging, QA, validation

WHAT STEP 0 DOES NOT DO
---------------------------
❌ No biological feature extraction
❌ No color calibration
❌ No measurements
❌ No assumptions about downstream models
It only normalizes geometry.

NOTES FOR DOWNSTREAM STEPS (1–3)
---------------------------
Steps 1, 2, 3 must use images from:

   corrected_images_affine/

All images will already:
- Have consistent orientation
- Have reduced perspective distortion
- Share identical resolution

KNOWN LIMITATIONS (CURRENT WIP)
--------------------------------
- Uses AFFINE (not full homography, only 3 points)
- Relies on SAM2 mask quality
- Card detection assumes the card is the dominant rectangular object

!! These limitations are being actively explored in the wip/perspective-correction branch.
   Homography using 4+ points will improve perspective correction.


============================
STEP 0 — Perspective Correction using SAM2.1
============================


Overview
--------------------------------
This script performs automatic perspective correction for faba bean imaging datasets using SAM2.1 automatic mask generation. It detects a reference paper sheet and three color calibration patches to compute a homography and produce standardized, rectified images.

The resulting images are resized and padded to a fixed resolution of 4000 × 6000 pixels, ensuring consistent downstream processing.



Method Summary
--------------------------------
1) Automatic Mask Generation
   - SAM2.1 generates segmentation masks for all objects in the image.

2) Reference Object Detection
   - Paper sheet is identified using spatial and size constraints:
      + Located in bottom-left region
      + Large mask area (>500000 pixels)
      + Width constraint (<2000 pixels)

3) Color Patch Detection
   - Three calibration squares detected in top-left region
   - Constraints:

      + Area between 80k and 300k pixels
      + Near-square aspect ratio (0.8–1.2)
      + Largest three masks selected

4) Geometry Construction
   - Paper transformed into an axis-aligned rectangle
   - Color patches normalized into square shapes
   - Homography computed using paper + patch corner points

5) Perspective Correction
   - Image warped using computed homography
   - Canvas expanded to prevent cropping

6) Normalization
   - Output resized while preserving aspect ratio
   - Zero-padded onto a 4000×6000 canvas

Project structure must allow:
--------------------------------
sam2/checkpoints/sam2.1_hiera_large.pt
configs/sam2.1/sam2.1_hiera_l.yaml


Run from repository root:
--------------------------------
faba-bean-image-classification/

Usage
--------------------------------
cd faba-bean-image-classification

python sam2/Step0_PerspectiveCorrection.py \
    --image-dir /path/to/input_images \
    --out-img-dir ../perspective_corrected_images \
    --out-mask-dir ../perspective_corrected_masks \
    --max-images 50

Argument	Required	Description
--------------------------------
--image-dir          Yes   Folder containing input images
--out-img-dir        Yes   Output folder for rectified images
--out-mask-dir       Yes   Output folder for debug masks
--max-images / -m	   No    Process only first N images

Supported formats:
--------------------------------
.jpg .jpeg .png .tif .tiff

Outputs
--------------------------------
<out-img-dir>/<image_name>

   - Perspective-corrected
   - Resized + padded to 4000×6000
   - Same filename as input

Debug Masks
----------------------------------------------------------------
<out-mask-dir>/<image_name>_selected_masks.png


Binary merge of detected paper + color patches
----------------------------------------------------------------
<out-mask-dir>/<image_name>_selected_masks_corrected.png

   - Perspective-corrected version aligned with output image

Failure Conditions
--------------------------------
The image will be skipped if:

   - Paper reference not detected
   - Fewer than three color patches found
   - Homography computation fails

Warnings are printed to console.


Assumptions & Dataset Constraints
----------------------------------------------------------------
This script assumes a controlled imaging setup:

- Paper reference positioned in bottom-left
- Color patches located in top-left
- Lighting and background allow reliable SAM segmentation
