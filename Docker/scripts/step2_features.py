# __authors__ = "Mathew Richards, Harpreet Kaur Bargota, Hao Nan Tobey Wang"
# __project__ = "Faba Bean Feature Extraction Pipeline (Step 2)"
# __credits__ = "Agriculture and Agri-Food Canada"

import os
import glob
import argparse
import warnings
import cv2
import numpy as np
import pandas as pd
from skimage import measure
from skimage.measure import regionprops_table
from circle_fit import taubinSVD

# Suppress pandas warnings
warnings.filterwarnings(action='ignore')

# --- Configuration ---
COIN_DIAMETER_MM = 23.88  # Diameter of the reference coin (Loonie)
CIRCULARITY_THRESHOLD = 0.7

def classify_shape(row):
    """Classifies bean shape based on calculated shape factors."""
    s1 = 'Elongated' if row['Shapefactor1'] <= 0.5 else 'Compact'
    s2 = 'Oval' if row['Shapefactor2'] <= 0.5 else 'Circular'
    s3 = 'Circular' if row['Shapefactor3'] >= 0.9 else 'Elongated'
    s4 = 'Ellipse' if row['Shapefactor4'] >= 0.9 else 'Irregular'
    return f"{s1},{s2},{s3},{s4}"

def process_sam_masks(sam_masks_dir, output_folder):
    """
    Extracts morphological features from SAM masks, calibrates using a coin,
    and saves the results.
    """
    # 1. Setup Directories
    if not os.path.exists(sam_masks_dir):
        print(f"Error: Input directory '{sam_masks_dir}' not found.")
        return
    os.makedirs(output_folder, exist_ok=True)

    all_data = []

    # 2. Process Each Image Subfolder
    subfolders = [f for f in os.listdir(sam_masks_dir) if os.path.isdir(os.path.join(sam_masks_dir, f))]
    
    for subfolder_name in subfolders:
        subfolder_path = os.path.join(sam_masks_dir, subfolder_name)
        print(f"Processing: {subfolder_name}")

        # Load Metadata
        csv_files = glob.glob(os.path.join(subfolder_path, '*.csv'))
        if not csv_files:
            print(f"No metadata CSV found in {subfolder_name}. Skipping.")
            continue
        
        df_metadata = pd.read_csv(csv_files[0])

        # --- A. Detect Reference Coin ---
        # Thresholds are adjusted for the 50% resize done in Step 1.
        # X >= 1500 (was 3000), Area >= 50,000 (was 200,000)
        coin_candidates = df_metadata[
            (df_metadata['bbox_x0'] >= 1500) & 
            (df_metadata['area'] >= 50000)
        ]

        if coin_candidates.empty:
            print(f"Warning: No coin found in {subfolder_name}. Skipping...")
            continue
        
        # Take the first valid coin candidate
        coin_index = coin_candidates.index[0]
        
        # --- B. Filter Out Noise/Labels ---
        # All thresholds scaled by 0.5 (coordinates) or 0.25 (area)
        conditions = [
            (df_metadata['bbox_x0'] <= 1400) & (df_metadata['bbox_y0'] <= 975),  # Colorcard
            (df_metadata['bbox_x0'] <= 950) & (df_metadata['bbox_y0'] >= 2325),  # Label
            (df_metadata['bbox_x0'] <= 2000) & (df_metadata['bbox_y0'] >= 2600), # Scale
            (df_metadata['bbox_x0'] >= 1500) & (df_metadata['area'] >= 50000),   # Coin (Main check)
            (df_metadata['bbox_x0'] >= 1500) & (df_metadata['bbox_y0'] >= 2200), # Coin (Alt check)
            (df_metadata['area'] <= 1250),   # Tiny noise
            (df_metadata['bbox_h'] >= 950),  # Duplicate masks (Height)
            (df_metadata['bbox_w'] >= 350)   # Duplicate masks (Width)
        ]
        
        for cond in conditions:
            df_metadata = df_metadata.drop(df_metadata[cond].index)

        valid_bean_indices = df_metadata.index.tolist()

        # --- C. Coin Calibration ---
        # Load coin mask
        coin_path = os.path.join(subfolder_path, f'{coin_index}.png')
        mask_coin = cv2.imread(coin_path, cv2.IMREAD_GRAYSCALE)
        
        # 1. Standard RegionProps Calibration
        label_coin = measure.label(mask_coin)
        props_coin = regionprops_table(label_coin, properties=('area', 'perimeter', 'axis_major_length', 'axis_minor_length'))
        
        # Calculate pixels-to-mm factors (SAM)
        cal_area = (np.pi * (COIN_DIAMETER_MM / 2)**2) / props_coin['area'][0]
        cal_len = COIN_DIAMETER_MM / props_coin['axis_major_length'][0]
        cal_wid = COIN_DIAMETER_MM / props_coin['axis_minor_length'][0]
        cal_peri = (np.pi * COIN_DIAMETER_MM) / props_coin['perimeter'][0]

        # 2. Taubin SVD Calibration
        contours_coin, _ = cv2.findContours(mask_coin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = np.vstack(contours_coin[0]).squeeze()
        xc, yc, radius_taubin, sigma = taubinSVD(pts)
        
        cal_area_taubin = (np.pi * (COIN_DIAMETER_MM / 2)**2) / (np.pi * radius_taubin**2)
        cal_linear_taubin = COIN_DIAMETER_MM / (2 * radius_taubin)

        # 3. MinEnclosingCircle Calibration
        _, radius_min = cv2.minEnclosingCircle(contours_coin[0])
        
        cal_area_min = (np.pi * (COIN_DIAMETER_MM / 2)**2) / (np.pi * radius_min**2)
        cal_linear_min = COIN_DIAMETER_MM / (2 * radius_min)

        # --- D. Process Bean Masks ---
        combined_mask = None

        for idx in valid_bean_indices:
            mask_path = os.path.join(subfolder_path, f'{idx}.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue

            # Circularity Check
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0: continue
                
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity > CIRCULARITY_THRESHOLD:
                    if combined_mask is None:
                        combined_mask = mask.copy()
                    else:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                    break

        if combined_mask is None:
            continue

        # Save Combined Mask Image
        contour_img = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 15)
        cv2.imwrite(os.path.join(output_folder, f"{subfolder_name}_combined_mask.png"), contour_img)

        # --- E. Extract Features ---
        label_image = measure.label(combined_mask)
        props = regionprops_table(label_image, properties=(
            'centroid', 'bbox', 'area', 'eccentricity', 'equivalent_diameter_area',
            'perimeter', 'solidity', 'area_convex', 'extent', 'axis_major_length', 'axis_minor_length'
        ))
        df_feats = pd.DataFrame(props)
        
        # Calculate Shape Factors
        df_feats['Aspect_Ratio'] = df_feats['axis_major_length'] / df_feats['axis_minor_length']
        df_feats['Roundness'] = (4 * np.pi * df_feats['area']) / (df_feats['perimeter'] ** 2)
        df_feats['Compactness'] = df_feats['equivalent_diameter_area'] / df_feats['axis_major_length']
        df_feats['Circularity-SAM'] = 1 / df_feats['Roundness']
        df_feats['Shapefactor1'] = df_feats['axis_major_length'] / df_feats['area']
        df_feats['Shapefactor2'] = df_feats['axis_minor_length'] / df_feats['area']
        df_feats['Shapefactor3'] = df_feats['area'] / ((df_feats['axis_major_length'] / 2) ** 2 * np.pi)
        df_feats['Shapefactor4'] = df_feats['area'] / ((df_feats['axis_major_length'] / 2) * (df_feats['axis_minor_length'] / 2) * np.pi)
        df_feats['class'] = subfolder_name.split('.JPG')[0] # Clean class name

        # Apply Calibrations (SAM)
        df_feats["Area_mm2_SAM"] = df_feats["area"] * cal_area
        df_feats["Length_mm_SAM"] = df_feats["axis_major_length"] * cal_len
        df_feats["Width_mm_SAM"] = df_feats["axis_minor_length"] * cal_wid
        df_feats["Perimeter_mm_SAM"] = df_feats["perimeter"] * cal_peri

        # Apply Calibrations (Taubin)
        df_feats["Area_mm2_SAM_taubin"] = df_feats["area"] * cal_area_taubin
        df_feats["Length_mm_SAM_taubin"] = df_feats["axis_major_length"] * cal_linear_taubin
        df_feats["Width_mm_SAM_taubin"] = df_feats["axis_minor_length"] * cal_linear_taubin
        df_feats["Perimeter_mm_SAM_taubin"] = df_feats["perimeter"] * cal_linear_taubin

        # Apply Calibrations (MinEnc)
        df_feats["Area_mm2_SAM_minEnc"] = df_feats["area"] * cal_area_min
        df_feats["Length_mm_SAM_minEnc"] = df_feats["axis_major_length"] * cal_linear_min
        df_feats["Width_mm_SAM_minEnc"] = df_feats["axis_minor_length"] * cal_linear_min
        df_feats["Perimeter_mm_SAM_minEnc"] = df_feats["perimeter"] * cal_linear_min

        all_data.append(df_feats)

    # 3. Aggregate and Save Results
    if not all_data:
        print("No valid data extracted.")
        return

    df_final = pd.concat(all_data, ignore_index=True)
    df_final['Shape'] = df_final.apply(classify_shape, axis=1)

    # Rename Columns
    column_map = {
        'class': 'Class',
        'Area_mm2_SAM': 'Area-SAM(mm2)',
        'Length_mm_SAM': 'Length-SAM(mm)',
        'Width_mm_SAM': 'Width-SAM(mm)',
        'Perimeter_mm_SAM': 'Perimeter-SAM(mm)',
        'Area_mm2_SAM_taubin': 'Area-SAM_taubin(mm2)',
        'Length_mm_SAM_taubin': 'Length-SAM_taubin(mm)',
        'Width_mm_SAM_taubin': 'Width-SAM_taubin(mm)',
        'Perimeter_mm_SAM_taubin': 'Perimeter-SAM_taubin(mm)',
        'Area_mm2_SAM_minEnc': 'Area-SAM_minEnc(mm2)',
        'Length_mm_SAM_minEnc': 'Length-SAM_minEnc(mm)',
        'Width_mm_SAM_minEnc': 'Width-SAM_minEnc(mm)',
        'Perimeter_mm_SAM_minEnc': 'Perimeter-SAM_minEnc(mm)',
        'area': 'Area-SAM(pix)',
        'eccentricity': 'Eccentricity',
        'equivalent_diameter_area': 'Equivalent diameter area',
        'perimeter': 'Perimeter(pix)',
        'axis_major_length': 'Axis Major Length-SAM(pix)',
        'axis_minor_length': 'Axis Minor Length-SAM(pix)',
        'Aspect_Ratio': 'Aspect Ratio'
    }
    df_final.rename(columns=column_map, inplace=True)
    
    # Save Feature Extraction CSV
    csv_path = os.path.join(output_folder, "Faba_bean_Features_extraction.csv")
    df_final.to_csv(csv_path, index_label="Seed No.")
    print(f"Saved Features: {csv_path}")

    # Save Seed Count Excel
    count_df = df_final['Class'].value_counts().reset_index()
    count_df.columns = ['Class_ID', 'Seed Count']
    excel_path = os.path.join(output_folder, "Seed Count.xlsx")
    count_df.to_excel(excel_path, index=False)
    print(f"Saved Seed Counts: {excel_path}")

    print("\n[Step 2/3] Feature Extraction Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Feature Extraction")
    parser.add_argument("sam_masks_dir", help="Input directory containing SAM masks")
    parser.add_argument("output_folder", help="Output directory for results")
    args = parser.parse_args()

    process_sam_masks(args.sam_masks_dir, args.output_folder)