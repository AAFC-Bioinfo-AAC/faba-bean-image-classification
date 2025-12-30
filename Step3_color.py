#hkb
#__author__="harpreet kaur bargota"
#__email__="harpreet.bargota@agr.gc.ca"
#__Project__="Faba bean Feature extraction pipeline (Step3)"
#References:
#https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2lab
#https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors
#https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

"""
Step 3: Dominant Seed Color Extraction with Color Calibration

This script processes images from a folder to extract dominant seed colors based on bounding box coordinates 
(`bbox-0,1,2,3`) from a CSV (typically generated in Step 2).

Key steps:
1. Color Calibration:
   - Detects a calibration image with a 24-patch color card.
   - Measures average RGB values of patches and compares them to reference RGBs.
   - Computes a Color Correction Matrix (CCM) using linear regression.
   - Applies CCM to all images and saves corrected versions.
   - Records calibration accuracy (mean and max ΔE) in 'calibration_clusters.csv'.

2. Dominant Color Extraction:
   - For each bounding box, extracts the region of interest (ROI) from the corrected image.
   - Finds the most frequent RGB color, excluding blue shades.
   - Converts the RGB to the closest CSS4 color name using the CIE Lab color space.

3. 1. Color Calibration:

Detects a calibration image with a 24-patch color card.

Measures average RGB values of patches and compares them to reference RGBs.

Computes a Color Correction Matrix (CCM) using linear regression.

Applies CCM to all images and saves corrected versions.

Records calibration accuracy (mean and max ΔE).

2. Dominant Color Extraction:

For each bounding box, extracts the region of interest (ROI) from the corrected image.

Finds the most frequent RGB color, excluding blue shades.

Converts the RGB to the closest CSS4 color name using the CIE Lab color space.

3. Thousand Grain Weight (TGW) Prediction:

Predicts Thousand Grain Weight (TGW, g) using a predefined linear regression model.

4. Output:
   - Adds two new columns to the bounding box CSV:
       - 'RGB value of Seed': dominant RGB value
       - 'color_seeds': corresponding CSS4 color name
       - 'TGW(g)': predicted Thousand Grain Weight (g)
   - Saves the updated CSV as 'FE_Color.csv' in the output folder.

Usage:
    python3 Step3_color.py /path/to/image_folder /path/to/output_folder
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LinearRegression
from skimage.color import rgb2lab, deltaE_cie76
from collections import Counter
from scipy.spatial.distance import cdist
from matplotlib.colors import CSS4_COLORS

# ---------------------- CONFIG ----------------------

COLOR_CARD_CROP = (0, 1800, 0, 2800)

REFERENCE_RGBS = np.array([
    [113, 81, 68], [200, 148, 131], [88, 122, 159], [88, 108, 67],
    [128, 129, 178], [87, 192, 175], [227, 125, 51], [66, 90, 172],
    [198, 82, 99], [91, 60, 108], [158, 191, 68], [231, 163, 48],
    [44, 62, 147], [62, 149, 77], [180, 48, 57], [240, 201, 46],
    [194, 85, 155], [0, 137, 173], [236, 235, 236], [203, 206, 208],
    [161, 164, 168], [119, 121, 124], [82, 83, 89], [50, 50, 51]
], dtype=np.float32)

# Precompute CSS4 color names
_css4_names = list(CSS4_COLORS.keys())
_css4_rgb = np.array([tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0,2,4)) 
                      for h in CSS4_COLORS.values()]) / 255.0
_css4_lab = rgb2lab(_css4_rgb.reshape(-1,1,3)).reshape(-1,3)

# ---------------------- HELPERS ----------------------

def find_calibration_image(folder: Path):
    patterns = ["calib", "calibration", "color", "colorcard", "color_card"]
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    candidates = [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]
    for p in candidates:
        if any(pat in p.name.lower() for pat in patterns):
            return p
    return candidates[0] if candidates else None

def crop_color_card(image):
    y0, y1, x0, x1 = COLOR_CARD_CROP
    H, W = image.shape[:2]
    y0 = max(0, min(y0,H-1)); y1 = max(0, min(y1,H))
    x0 = max(0, min(x0,W-1)); x1 = max(0, min(x1,W))
    return image[y0:y1, x0:x1]

def measure_color_card_patches(image, rows=4, cols=6):
    h, w, _ = image.shape
    patch_h, patch_w = h//rows, w//cols
    measured = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r*patch_h, (r+1)*patch_h
            x0, x1 = c*patch_w, (c+1)*patch_w
            patch = image[y0:y1, x0:x1]
            avg_rgb = np.mean(patch.reshape(-1,3), axis=0)
            measured.append(avg_rgb)
    return np.array(measured, dtype=np.float32)

def compute_ccm(measured, reference):
    reg = LinearRegression(fit_intercept=False)
    reg.fit(measured, reference)
    return reg.coef_

def apply_ccm(image, ccm):
    h, w, _ = image.shape
    flat = image.reshape(-1,3)
    corrected = np.dot(flat, ccm.T)
    corrected = np.clip(corrected,0,255)
    return corrected.reshape(h,w,3).astype(np.uint8)

def compute_deltaE(measured, reference):
    measured_lab = rgb2lab(measured[np.newaxis,:,:]/255.0)
    reference_lab = rgb2lab(reference[np.newaxis,:,:]/255.0)
    deltaE = deltaE_cie76(measured_lab, reference_lab)
    return np.mean(deltaE), np.max(deltaE)

def find_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]

def rgb_to_css4_color_name(rgb):
    rgb_norm = np.array(rgb, dtype=np.float64)/255.0
    lab = rgb2lab(np.array([[rgb_norm]]))[0][0]
    d = cdist([lab], _css4_lab)[0]
    idx = int(np.argmin(d))
    return _css4_names[idx]

def get_dominant_color_excluding_blue(img, x, y, w, h):
    H, W, _ = img.shape
    x0, y0 = max(0,int(round(x))), max(0,int(round(y)))
    x1, y1 = min(W,int(round(x+w))), min(H,int(round(y+h)))
    roi = img[y0:y1,x0:x1]
    flat = roi.reshape(-1,3)
    counts = Counter(map(tuple, flat))
    for color,_ in counts.most_common():
        name = rgb_to_css4_color_name(color)
        if "blue" not in name.lower() and name.lower() not in ["dodgerblue","cornflowerblue"]:
            return color
    return counts.most_common(1)[0][0]

def find_first_csv(folder: Path):
    csvs = [p for p in folder.iterdir() if p.suffix.lower()==".csv" and p.is_file()]
    return csvs[0] if csvs else None

# ---------------------- MAIN ----------------------

def main():
    if len(sys.argv)!=3:
        print("Usage: python3 calibrate_then_extract_pipeline.py /image_folder /output_folder")
        sys.exit(1)
    
    image_folder = Path(sys.argv[1]).expanduser().resolve()
    output_folder = Path(sys.argv[2]).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Calibration image
    calib_path = find_calibration_image(image_folder)
    if calib_path is None: raise FileNotFoundError("No images found in folder")
    print("Using calibration image:", calib_path)
    
    img_bgr = cv2.imread(str(calib_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    card = crop_color_card(img_rgb)
    measured_rgb = measure_color_card_patches(card)
    
    if measured_rgb.shape[0]!=24:
        raise ValueError("Expected 24 patches. Check crop or card visibility.")
    
    # Compute CCM
    ccm = compute_ccm(measured_rgb, REFERENCE_RGBS)
    
    # Prepare corrected folder
    corrected_dir = output_folder / "corrected_images"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    
    images = find_images(image_folder)
    calibration_records = []
    for p in images:
        img_bgr = cv2.imread(str(p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        corrected_rgb = apply_ccm(img_rgb, ccm)
        out_path = corrected_dir / p.name
        cv2.imwrite(str(out_path), cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR))
        card_corr = crop_color_card(corrected_rgb)
        measured_corr = measure_color_card_patches(card_corr)
        meanE, maxE = compute_deltaE(measured_corr, REFERENCE_RGBS)
        calibration_records.append({"Image":p.name,"Mean_DeltaE":meanE,"Max_DeltaE":maxE})
    
    # Extract dominant seed colors
    bbox_csv = find_first_csv(output_folder)
    if bbox_csv is None:
        print("No bbox CSV found in output folder. Skipping color extraction.")
        return
    
    df_bbox = pd.read_csv(bbox_csv)
    rgb_values, color_names = [], []
    for idx,row in df_bbox.iterrows():
        try:
            x = row["bbox-1"]
            y = row["bbox-0"]
            w = row["bbox-3"] - row["bbox-1"]
            h = row["bbox-2"] - row["bbox-0"]
            cls = str(row["Class"])
            img_candidates = [corrected_dir / f"{cls}{ext}" for ext in [".jpg",".JPG",".png",".PNG",".tif",".tiff"]]
            img_path = next((p for p in img_candidates if p.exists()), None)
            if img_path is None: raise FileNotFoundError(f"No image for Class {cls}")
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            dom = get_dominant_color_excluding_blue(img, x,y,w,h)
            rgb_values.append([int(c) for c in dom])
            color_names.append(rgb_to_css4_color_name(dom))
        except Exception as e:
            print(f"Warning row {idx}: {e}")
            rgb_values.append([0,0,0])
            color_names.append("unknown")
    
    df_bbox["RGB value of Seed"] = rgb_values
    df_bbox["color_seeds"] = color_names

    # -------------------------
    # Calculate Thousand Grain Weight (TGW)
    # -------------------------
    required_cols = ['Area-SAM_taubin(mm2)','Width-SAM_taubin(mm)','Length-SAM_taubin(mm)',
                     'Circularity-SAM','Aspect Ratio']
    if all(col in df_bbox.columns for col in required_cols):
        df_bbox['TGW(g)'] = (
            296.9785397519971
            + (5.5020 * df_bbox['Area-SAM_taubin(mm2)'])
            + (18.4537 * df_bbox['Width-SAM_taubin(mm)'])
            + (-17.3898 * df_bbox['Length-SAM_taubin(mm)'])
            + (-607.6333 * df_bbox['Circularity-SAM'])
            + (344.1165 * df_bbox['Aspect Ratio'])
        )
        print("Added 'TGW(g)' column to DataFrame")
    else:
        missing = [col for col in required_cols if col not in df_bbox.columns]
        print(f"⚠️ Cannot compute TGW(g). Missing columns: {missing}")

    # Save final CSV
    fe_csv = output_folder / "FE_Color.csv"
    df_bbox.to_csv(fe_csv,index=False)
    print("Saved FE_Color.csv with TGW(g):", fe_csv)
    print("Pipeline complete.")

# ---------------------- RUN ----------------------

if __name__=="__main__":
    main()
