# __authors__ = "Harpreet Kaur Bargota"
# __project__ = "Faba Bean Feature Extraction Pipeline (Step 3)"
# __credits__ = "Agriculture and Agri-Food Canada"

"""
Step 3: Color Calibration & TGW Prediction
1. Calibrates colors using a standard 24-patch color card (Macbeth ColorChecker).
2. Extracts the dominant color of each seed.
3. Predicts Thousand Grain Weight (TGW) using morphological features.
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LinearRegression
from skimage.color import rgb2lab, deltaE_cie76
from scipy.spatial.distance import cdist
from matplotlib.colors import CSS4_COLORS

# --- Configuration ---

# Crop coordinates for Color Card (Scaled for 50% resized image)
# y0, y1, x0, x1
COLOR_CARD_CROP = (0, 900, 0, 1400)

# Reference RGB values for a standard Macbeth ColorChecker (24 patches)
REFERENCE_RGBS = np.array([
    [113, 81, 68], [200, 148, 131], [88, 122, 159], [88, 108, 67],
    [128, 129, 178], [87, 192, 175], [227, 125, 51], [66, 90, 172],
    [198, 82, 99], [91, 60, 108], [158, 191, 68], [231, 163, 48],
    [44, 62, 147], [62, 149, 77], [180, 48, 57], [240, 201, 46],
    [194, 85, 155], [0, 137, 173], [236, 235, 236], [203, 206, 208],
    [161, 164, 168], [119, 121, 124], [82, 83, 89], [50, 50, 51]
], dtype=np.float32)

# Precompute CSS4 color names for fast lookup
_css4_names = list(CSS4_COLORS.keys())
_css4_rgb = np.array([tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0,2,4)) 
                      for h in CSS4_COLORS.values()]) / 255.0
_css4_lab = rgb2lab(_css4_rgb.reshape(-1,1,3)).reshape(-1,3)


# --- Helper Functions ---

def load_and_resize_image(path):
    """Loads an image and resizes it to 50% to match Step 1/Step 2 coordinates."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    # Resize 50%
    width = int(img.shape[1] * 0.50)
    height = int(img.shape[0] * 0.50)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def get_color_card_patches(image, rows=4, cols=6):
    """Crops the color card and extracts average RGB for 24 patches."""
    # 1. Crop to the known location of the card
    y0, y1, x0, x1 = COLOR_CARD_CROP
    card_img = image[y0:y1, x0:x1]
    
    # 2. Split into grid
    h, w, _ = card_img.shape
    patch_h, patch_w = h // rows, w // cols
    
    measured_rgb = []
    for r in range(rows):
        for c in range(cols):
            py0, py1 = r * patch_h, (r + 1) * patch_h
            px0, px1 = c * patch_w, (c + 1) * patch_w
            patch = card_img[py0:py1, px0:px1]
            avg_rgb = np.mean(patch.reshape(-1, 3), axis=0)
            measured_rgb.append(avg_rgb)
            
    return np.array(measured_rgb, dtype=np.float32)

def compute_ccm(measured, reference):
    """Calculates the Color Correction Matrix using Linear Regression."""
    reg = LinearRegression(fit_intercept=False)
    reg.fit(measured, reference)
    return reg.coef_

def apply_ccm(image, ccm):
    """Applies the correction matrix to an image."""
    h, w, _ = image.shape
    flat = image.reshape(-1, 3)
    corrected = np.dot(flat, ccm.T)
    corrected = np.clip(corrected, 0, 255)
    return corrected.reshape(h, w, 3).astype(np.uint8)

def rgb_to_name(rgb):
    """Finds the closest CSS4 color name using Delta-E in LAB space."""
    rgb_norm = np.array(rgb, dtype=np.float64) / 255.0
    lab = rgb2lab(np.array([[rgb_norm]]))[0][0]
    dists = cdist([lab], _css4_lab)[0]
    return _css4_names[int(np.argmin(dists))]

def get_dominant_color(img, x, y, w, h):
    """Extracts dominant color from a bounding box, ignoring blue backgrounds."""
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(img.shape[1], int(x+w)), min(img.shape[0], int(y+h))
    
    roi = img[y0:y1, x0:x1]
    if roi.size == 0: return (0, 0, 0)

    # Flatten and count colors
    counts = Counter(map(tuple, roi.reshape(-1, 3)))
    
    # Return most common color that isn't a blue background shade
    for color, _ in counts.most_common():
        name = rgb_to_name(color).lower()
        if "blue" not in name and name not in ["dodgerblue", "cornflowerblue"]:
            return color
            
    return counts.most_common(1)[0][0]


# --- Main Pipeline ---

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/step3_color.py <input_image_dir> <output_dir>")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # 1. Locate Data
    # Find the feature CSV from Step 2
    csv_path = output_dir / "Faba_bean_Features_extraction.csv"
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}. Did Step 2 run?")
        sys.exit(1)
    
    # Find any image to use for calibration (Assuming lighting is constant)
    img_files = list(image_dir.glob("*.[jJpP]*")) # jpg, png, etc
    if not img_files:
        print("Error: No images found in input directory.")
        sys.exit(1)
    
    print("[Step 3/3] Starting Color Calibration...")

    # 2. Train Color Correction Matrix (CCM)
    # We use the first available image to calculate the correction
    calib_img_bgr = load_and_resize_image(img_files[0])
    calib_img_rgb = cv2.cvtColor(calib_img_bgr, cv2.COLOR_BGR2RGB)
    
    measured_rgb = get_color_card_patches(calib_img_rgb)
    ccm = compute_ccm(measured_rgb, REFERENCE_RGBS)
    print("CCM Calculated.")

    # 3. Process All Images (Correct & Save)
    corrected_dir = output_dir / "corrected_images"
    corrected_dir.mkdir(exist_ok=True)
    
    # We cache loaded images in memory to speed up seed lookup later
    # (Optional: If dataset is massive, load them on demand instead)
    corrected_images_cache = {}

    print(f"Applying correction to {len(img_files)} images...")
    for p in img_files:
        # Load & Resize
        img_bgr = load_and_resize_image(p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Correct
        corrected_rgb = apply_ccm(img_rgb, ccm)
        
        # Save to disk (converted back to BGR for OpenCV)
        save_path = corrected_dir / p.name
        cv2.imwrite(str(save_path), cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR))
        
        # Cache for next step (Key is filename without extension, matching 'Class' in CSV)
        corrected_images_cache[p.stem] = corrected_rgb

    # 4. Extract Seed Colors
    print("Extracting seed colors...")
    df = pd.read_csv(csv_path)
    
    rgb_results = []
    color_names = []

    for index, row in df.iterrows():
        try:
            image_name = str(row['Class']) # e.g. "Faba-123"
            
            if image_name not in corrected_images_cache:
                # Fallback if cache missed (e.g. extension mismatch)
                rgb_results.append((0,0,0))
                color_names.append("Image Not Found")
                continue

            img = corrected_images_cache[image_name]
            
            # Extract dominant color from bounding box
            dom_rgb = get_dominant_color(
                img, 
                x=row['bbox-1'], y=row['bbox-0'], 
                w=(row['bbox-3'] - row['bbox-1']), 
                h=(row['bbox-2'] - row['bbox-0'])
            )
            
            rgb_results.append(list(dom_rgb))
            color_names.append(rgb_to_name(dom_rgb))

        except Exception as e:
            print(f"Warning on row {index}: {e}")
            rgb_results.append((0,0,0))
            color_names.append("Error")

    df['RGB_Seed'] = rgb_results
    df['Color_Name'] = color_names

    # 5. Predict Thousand Grain Weight (TGW)
    # Using the linear regression formula provided
    print("Predicting Thousand Grain Weight (TGW)...")
    
    # Formula coefficients
    intercept = 296.9785
    coef_area = 5.5020
    coef_width = 18.4537
    coef_length = -17.3898
    coef_circ = -607.6333
    coef_aspect = 344.1165

    if 'Area-SAM_taubin(mm2)' in df.columns:
        df['TGW(g)'] = (
            intercept +
            (coef_area * df['Area-SAM_taubin(mm2)']) +
            (coef_width * df['Width-SAM_taubin(mm)']) +
            (coef_length * df['Length-SAM_taubin(mm)']) +
            (coef_circ * df['Circularity-SAM']) +
            (coef_aspect * df['Aspect Ratio'])
        )
    else:
        print("  Warning: Missing columns for TGW calculation. Skipping.")

    # 6. Save Final Output
    final_csv = output_dir / "FE_Color.csv"
    df.to_csv(final_csv, index=False)
    print(f"\n Pipeline Complete! Final data saved to: {final_csv}")

if __name__ == "__main__":
    main()