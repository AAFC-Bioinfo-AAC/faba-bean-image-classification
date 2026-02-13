# ============================================================
# How to run (Step0_PerspectiveCorrection)
# ============================================================
# Purpose:
# - Uses SAM2.1 automatic masks to detect:
#       • Paper (bottom-left region)
#       • Three color calibration patches (top-left region)
# - Estimates a homography and performs perspective correction.
# - Outputs a rectified image resized and padded to exactly 4000 x 6000 pixels.
# - Saves debug masks showing the selected regions before and after correction.
#
# IMPORTANT:
# - Run from the project root directory:
#       faba-bean-image-classification/
#   so the checkpoint path resolves correctly:
#       sam2/checkpoints/sam2.1_hiera_large.pt
#
# ------------------------------------------------------------
# Usage:
# ------------------------------------------------------------
# cd faba-bean-image-classification
#
# Basic:
# python sam2/Step0_PerspectiveCorrection.py \
#     --image-dir /path/to/input_images \
#     --out-img-dir ../perspective_corrected_images \
#     --out-mask-dir ../perspective_corrected_masks
#
# Limit number of processed images:
# python sam2/Step0_PerspectiveCorrection.py \
#     --image-dir /path/to/input_images \
#     --out-img-dir ../perspective_corrected_images \
#     --out-mask-dir ../perspective_corrected_masks \
#     --max-images 50
#
# ------------------------------------------------------------
# Arguments:
# ------------------------------------------------------------
# --image-dir      (required) Input folder containing images
#                  Supported formats: .jpg .jpeg .png .tif .tiff
#
# --out-img-dir    (required) Output folder for perspective-corrected images
#
# --out-mask-dir   (required) Output folder for debug mask visualizations
#
# --max-images / -m (optional)
#                  Process only the first N images (sorted by filename)
#
# ------------------------------------------------------------
# Outputs:
# ------------------------------------------------------------
# <out-img-dir>/
#     <image_name>.<ext>
#         Perspective-corrected image padded to 4000x6000
#
# <out-mask-dir>/
#     <image_name>_selected_masks.png
#         Binary merged mask of detected paper + color patches
#
#     <image_name>_selected_masks_corrected.png
#         Perspective-corrected merged mask aligned with output image
#
# NOTE:
# - One mask pair is saved per input image (no overwriting).
# - Output directories are created automatically if missing.
#
# ------------------------------------------------------------
# Next pipeline step (example):
# ------------------------------------------------------------
# python sam2/Step1_SAM2.1.py \
#     ../perspective_corrected_images \
#     ../output_SAM_Pers_60
# ============================================================


import os
import cv2
import argparse
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# -------------------------
# Utility functions
# -------------------------
def get_mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    return np.min(xs), np.min(ys), np.max(xs), np.max(ys)

def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def get_mask_corners(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)

def get_mask_center(mask):
    ys, xs = np.where(mask > 0)
    return np.array([np.mean(xs), np.mean(ys)], dtype=np.float32)

def save_mask_image(mask, path):
    """Save binary mask as image"""
    cv2.imwrite(path, (mask*255).astype(np.uint8))

def create_merged_selected_mask_binary(paper_mask, color_patch_masks):
    """
    Create a single binary debug mask containing all selected masks.
    Returns a 2D uint8 mask (0/1) that is the union of:
    - paper mask
    - color patch masks
    """
    ref_mask = paper_mask
    if ref_mask is None:
        for m in color_patch_masks or []:
            if m is not None:
                ref_mask = m
                break
    if ref_mask is None:
        return None

    merged = np.zeros(ref_mask.shape[:2], dtype=np.uint8)
    if paper_mask is not None:
        merged |= paper_mask.astype(np.uint8)
    for mask in color_patch_masks or []:
        if mask is None:
            continue
        merged |= mask.astype(np.uint8)
    return merged


def build_square_patch_from_top(corners):
    """
    corners: (4,2) ordered TL,TR,BR,BL (SOURCE patch corners)
    returns: (4,2) destination corners TL,TR,BR,BL
    """
    tl, tr, br, bl = corners

    # 1. straighten top edge
    top_y = max(tl[1], tr[1])  # "lower" of the two (image coords)

    tl_dst = np.array([tl[0], top_y], dtype=np.float32)
    tr_dst = np.array([tr[0], top_y], dtype=np.float32)

    # 2. compute square size
    width = np.linalg.norm(tr_dst - tl_dst)

    # 3. bottom corners
    bl_dst = tl_dst + np.array([0, width], dtype=np.float32)
    br_dst = tr_dst + np.array([0, width], dtype=np.float32)

    return np.array([tl_dst, tr_dst, br_dst, bl_dst], dtype=np.float32)

def build_straight_vertical_line(TR_BR):
    """
    TR_BR: (2,2) array of top-right and bottom-right corners
    returns: (2,2) array of straightened vertical line points
    """
    tr, br = TR_BR
    # Compute the straight vertical line by aligning x-coordinates
    x = tr[0]  # Use x-coordinate of top-right corner
    y_min = min(tr[1], br[1])
    y_max = max(tr[1], br[1])
    return np.array([[x, y_min], [x, y_max]], dtype=np.float32)

def build_straight_paper_dst(corners):
    """
    corners: (4,2) ordered TL,TR,BR,BL (SOURCE patch corners)
    returns: (4,2) destination corners TL,TR,BR,BL
    """
    tl, tr, br, bl = corners

    max_x = max(tl[0], bl[0], tr[0], br[0])
    min_x = min(tl[0], bl[0], tr[0], br[0])
    max_y = max(tl[1], bl[1], tr[1], br[1])
    min_y = min(tl[1], bl[1], tr[1], br[1])

    tl_dst = np.array([min_x, min_y], dtype=np.float32)
    tr_dst = np.array([max_x, min_y], dtype=np.float32)
    bl_dst = np.array([min_x, max_y], dtype=np.float32)
    br_dst = np.array([max_x, max_y], dtype=np.float32)

    return np.array([tl_dst, tr_dst, br_dst, bl_dst], dtype=np.float32)

# -------------------------
# Main function
# -------------------------
def rectify_with_sam_masks(image, mask_generator: SAM2AutomaticMaskGenerator, mask_save_folder=None, debug=False, image_stem=None):
    # H_img, W_img = image.shape[:2]
    W_img = 4000
    H_img = 6000
    
    # 1. Generate masks
    masks = mask_generator.generate(image)
    
    # 2. Detect paper: bottom-left region
    paper_mask_dict = None
    for m in masks:
        x, y, w, h = m['bbox']
        if x <= 1900 and y >= 4500 and w < 2000 and m['area'] > 500000:  # bottom-left region
            paper_mask_dict = m
    if paper_mask_dict is None:
        print("⚠️ Paper not detected")
        return image

    paper_mask = paper_mask_dict['segmentation']
    paper_corners = order_points(get_mask_corners(paper_mask))
    # TL, TR, BR, BL

    paper_TR_BR = paper_corners[[1,2]]  # TR and BR corners

    # -------------------------
    # 3. Detect 3 color patches (largest masks in top-left)
    # -------------------------
    color_masks = []
    for m in masks:
        x, y, w, h = m['bbox']
        if x <= 2800 and y <= 1950 and m['area'] > 80000 and m['area'] < 300000 and w/h > 0.8 and w/h < 1.2:  # color patches location and area constraints
            color_masks.append(m)
    if len(color_masks) < 3:
        print("⚠️ Not enough color patches detected")
        return image

    # Pick 3 largest masks
    color_masks = sorted(color_masks, key=lambda m: m['area'], reverse=True)[:3]
    
    color_patch_corners = []
    color_patch_masks = []
    for idx, cm in enumerate(color_masks):
        mask = cm['segmentation']
        color_patch_masks.append(mask)
        corners = order_points(get_mask_corners(mask))
        color_patch_corners.append(corners)

    # Save one merged debug mask image (paper + patches)
    merged_mask = None
    if mask_save_folder:
        os.makedirs(mask_save_folder, exist_ok=True)
        merged_name = "selected_masks.png" if not image_stem else f"{image_stem}_selected_masks.png"
        merged_path = os.path.join(mask_save_folder, merged_name)
        merged_mask = create_merged_selected_mask_binary(paper_mask, color_patch_masks)
        if merged_mask is not None:
            save_mask_image(merged_mask, merged_path)

    color_patch_corners = np.vstack(color_patch_corners)  # shape: (12,2)

    src_paper = paper_corners  # (4,2)
    src_paper_TR_BR = paper_TR_BR  # (2,2)
    # -------------------------
    # 4. Build source points (paper + 3 patches)
    # -------------------------
    src_pts = np.vstack([src_paper, color_patch_corners]).astype(np.float32) # 4 + 12 = 16
    src_pts_new = np.vstack([src_paper_TR_BR, color_patch_corners]).astype(np.float32) # 2 + 12 = 14

    # 5. Build destination points (controlled rectangle)
    dst_paper = build_straight_paper_dst(paper_corners)
    dst_paper_TR_BR = build_straight_vertical_line(src_paper_TR_BR)

    dst_color_corners = []
    for cm in color_masks:
        src_corners = order_points(get_mask_corners(cm['segmentation']))
        dst_patch = build_square_patch_from_top(src_corners)
        dst_color_corners.append(dst_patch)
    dst_color_corners = np.vstack(dst_color_corners)  # (12,2)

    dst_pts = np.vstack([dst_paper, dst_color_corners])  # (16,2)
    dst_pts_new = np.vstack([dst_paper_TR_BR, dst_color_corners])  # (14,2)

    # 6. Compute homography

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    # H, _ = cv2.findHomography(
    #     src_paper,    # src
    #     dst_paper,        # dst
    #     method=0          # no RANSAC needed with 4 points
    # )

    if H is None:
        print("⚠️ Homography failed")
        return image

    # 7. Expand canvas to avoid cropping
    corners = np.array([[0,0],[W_img,0],[W_img,H_img],[0,H_img]], dtype=np.float32).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    xs, ys = warped_corners[:,0,0], warped_corners[:,0,1]
    tx, ty = -xs.min() if xs.min()<0 else 0, -ys.min() if ys.min()<0 else 0
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
    H_final = T @ H
    out_w, out_h = int(np.ceil(xs.max() + tx)), int(np.ceil(ys.max() + ty))

    # Warp the image
    warped = cv2.warpPerspective(
        image, 
        H_final, 
        (out_w, out_h), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0)
    )

    if debug:
        overlay = warped.copy()
        for pt in dst_pts.astype(int):
            cv2.circle(overlay, tuple(pt), 10, (0,0,255), -1)
        return warped, overlay

    # -------------------------
    # Resize with aspect ratio and pad to 4000x6000
    # -------------------------
    target_w, target_h = 4000, 6000
    h, w = warped.shape[:2]

    # Compute scale to fit inside target
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize keeping aspect ratio
    warped_resized = cv2.resize(warped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create black canvas and paste resized image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = warped_resized

    # Final output
    img_final = canvas

    # -------------------------
    # Save perspective-corrected merged selected mask (aligned to img_final)
    # -------------------------
    if mask_save_folder and merged_mask is not None:
        corrected_name = (
            "selected_masks_corrected.png"
            if not image_stem
            else f"{image_stem}_selected_masks_corrected.png"
        )
        corrected_path = os.path.join(mask_save_folder, corrected_name)

        warped_mask = cv2.warpPerspective(
            (merged_mask * 255).astype(np.uint8),
            H_final,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask_resized = cv2.resize(
            warped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

        mask_canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        mask_canvas[start_y:start_y+new_h, start_x:start_x+new_w] = warped_mask_resized

        corrected_mask = (mask_canvas > 0).astype(np.uint8)
        save_mask_image(corrected_mask, corrected_path)

    return img_final

# Static model configuration and checkpoint paths
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"

sam2_model = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(sam2_model)

# Device setup
device = "cpu"
sam2_model.to(device)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    # This parameter filters out masks smaller than this value during generation
    min_mask_region_area=500
)

parser = argparse.ArgumentParser()
parser.add_argument("--image-dir", required=True)
parser.add_argument("--out-img-dir", required=True)
parser.add_argument("--out-mask-dir", required=True)
parser.add_argument("--max-images", "-m", type=int, default=None)
args = parser.parse_args()

input_folder = args.image_dir
output_folder = args.out_img_dir
mask_save_folder = args.out_mask_dir

os.makedirs(output_folder, exist_ok=True)
os.makedirs(mask_save_folder, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(image_extensions)
])

if args.max_images:
    files = files[:args.max_images]

for fname in files:
    print(f"Processing: {fname}")
    path = os.path.join(input_folder, fname)

    img = cv2.imread(path)
    if img is None:
        print(f"❌ Could not read {fname}")
        continue
  
    image_stem = os.path.splitext(fname)[0]
    result = rectify_with_sam_masks(
        img,
        mask_generator,
        mask_save_folder=mask_save_folder,
        image_stem=image_stem,
    )

    if result is None:
        print(f"⚠️ Skipped {fname}")
        continue

    out_path = os.path.join(output_folder, fname)
    cv2.imwrite(out_path, result)
    print(f"✅ Saved corrected image to {out_path}")
