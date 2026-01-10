"""
Step 0 — Affine-Based Perspective Normalization

This script performs affine transformation on raw images by:
1. Detecting the color card using SAM2
2. Extracting card corners from the SAM mask
3. Computing an affine transform to reduce tilt/perspective distortion
4. Warping and resizing images to a fixed resolution (4000 x 6000)

Outputs from this step are the REQUIRED inputs for Steps 1, 2, and 3.

IMPORTANT:
- This script intentionally uses AFFINE transformation (not full homography).

LIMITATIONS:
- Affine transformation uses only 3 points of the card; therefore:
    * Full perspective distortion cannot be corrected if the card is strongly skewed.
    * Cards partially outside the image may lead to inaccurate alignment.
    * Non-planar distortions (e.g., lens barrel/pincushion) are not corrected.
- Future improvement: a homography-based perspective normalization using 4 or more scattered points
  across the card will be implemented to better handle strong tilt or partial cards.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import pandas as pd
import matplotlib.pyplot as plt
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# =============================================================
# Utility functions
# =============================================================

def order_points(pts):
    """
    Orders four points as:
    top-left, top-right, bottom-right, bottom-left.

    Parameters
    ----------
    pts : array-like, shape (4, 2)

    Returns
    -------
    np.ndarray, shape (4, 2)
    """
    pts = np.array(pts, dtype="float32")

    # The top-left will have the smallest sum,
    # the bottom-right will have the largest sum
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    # The top-right will have the smallest diff,
    # the bottom-left will have the largest diff
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


# =============================================================
# Input resolution (CLI / ENV / default)
# =============================================================

def resolve_image_folder():
    """
    Resolve input image directory with the following priority:
    1. --image-dir command-line argument
    2. FABA_IMAGES_DIR environment variable
    3. ../faba_images relative to script location
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image-dir", "-i", default=None)
    parser.add_argument("--max-images", "-m", type=int, default=None)
    known, _ = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.abspath(os.path.join(script_dir, "..", "faba_images"))

    if known.image_dir:
        return os.path.abspath(known.image_dir), "--image-dir", known.max_images

    env_dir = os.environ.get("FABA_IMAGES_DIR")
    if env_dir:
        return os.path.abspath(env_dir), "FABA_IMAGES_DIR", known.max_images

    return default_dir, "script-relative default", known.max_images

# =============================================================
# Output directories
# =============================================================

OUTPUT_IMG_DIR = "../corrected_images_affine"
OUTPUT_MASK_DIR = "../corrected_images_mask_affine"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

print(f"Affine-corrected output directory: {OUTPUT_IMG_DIR}")
print(f"Affine mask output directory: {OUTPUT_MASK_DIR}")


# =============================================================
# Load images
# =============================================================

folder_path, source, max_images = resolve_image_folder()
print(f"Using images folder (source={source}): {folder_path}")

image_paths = sorted(glob.glob(os.path.join(folder_path, "*.*")))
print(f"Found {len(image_paths)} files")

images = []
for img_path in image_paths:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Warning: could not read {img_path}")
        continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    images.append((os.path.basename(img_path), img_rgb))

if max_images:
    images = images[:max_images]
    print(f"Limited to first {len(images)} images")

print(f"Loaded {len(images)} images")


# =============================================================
# SAM2 model initialization
# =============================================================

model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"

sam2_model = build_sam2(model_cfg, checkpoint)
sam2_model.to("cpu")

predictor = SAM2ImagePredictor(sam2_model)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    min_mask_region_area=10
)

# =============================================================
# Main processing loop
# =============================================================

masks_and_names = []

for name, img in images:

    # ---------------------------------------------------------
    # SAM2 mask prediction (box-conditioned)
    # ---------------------------------------------------------

    predictor.set_image(img)

    box = np.array([0, 0, 3100, 2100], dtype=np.float32)
    masks, scores, _ = predictor.predict(
        box=box[None, :],
        multimask_output=True
    )

    best_id = int(np.argmax(scores))
    mask = masks[best_id]
    masks_and_names.append((name, mask))

    mask_uint8 = (mask.astype(np.uint8) * 255)

    # ---------------------------------------------------------
    # Extract card corners from mask
    # ---------------------------------------------------------

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    card_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, 0.02 * peri, True)

    if len(approx) != 4:
        print(f"ERROR: {name} | contour has {len(approx)} points")
        continue

    tl, tr, br, bl = order_points(approx.reshape(4, 2))
    src_points = np.array([tl, tr, br, bl], dtype=np.float32)

    # ---------------------------------------------------------
    # Destination geometry
    # ---------------------------------------------------------

    max_x = max(tl[0], tr[0], br[0], bl[0])
    max_y = max(tl[1], tr[1], br[1], bl[1])
    min_x = min(tl[0], tr[0], br[0], bl[0])
    min_y = min(tl[1], tr[1], br[1], bl[1])

    dst_points = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ], dtype=np.float32)

    # ---------------------------------------------------------
    # Affine transform (first 3 points only)
    # ---------------------------------------------------------

    # NOTE: Using only the first 3 points limits correction to an affine approximation.
    # [Future Plan: Full perspective (homography) would require 4 points and will be implemented].

    M = cv2.getAffineTransform(src_points[:3], dst_points[:3])

    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    corners_h = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])
    warped = (M @ corners_h.T).T

    x_min, x_max = int(warped[:, 0].min()), int(warped[:, 0].max())
    y_min, y_max = int(warped[:, 1].min()), int(warped[:, 1].max())

    T = np.array([[1, 0, -x_min], [0, 1, -y_min]], dtype=np.float32)
    M_final = (T @ np.vstack([M, [0, 0, 1]]))[:2]

    dst = cv2.warpAffine(
        img,
        M_final,
        (x_max - x_min, y_max - y_min),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # ---------------------------------------------------------
    # Resize to canonical output resolution
    # ---------------------------------------------------------

    dst = cv2.resize(dst, (4000, 6000), interpolation=cv2.INTER_LINEAR)
    dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

    out_path = os.path.join(OUTPUT_IMG_DIR, name)
    cv2.imwrite(out_path, dst_bgr)
    print(f"Wrote: {out_path} | size={dst.shape[:2]}")


# =============================================================
# Save card masks (debug / QA)
# =============================================================

for name, m in masks_and_names:
    mask_uint8 = (m.astype(np.uint8) * 255)
    mask_bgr = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
    out_mask = os.path.join(OUTPUT_MASK_DIR, f"mask_{name}.png")
    cv2.imwrite(out_mask, mask_bgr)
    print(f"Wrote mask: {out_mask}")