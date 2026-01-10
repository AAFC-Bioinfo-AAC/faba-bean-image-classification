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
