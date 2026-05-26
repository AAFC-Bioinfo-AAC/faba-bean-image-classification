# ============================================================
# FABA BEAN SAM PIPELINE
# ============================================================
# Full end-to-end pipeline:
#
# STEP 0 : Perspective correction
# STEP 1 : Run SAM2 segmentation
# STEP 2 : Extract bean measurements
# STEP 3 : Merge with Excel ground truth
# STEP 4 : Statistics + plots + reports
#
# Author: Customized for Faba Bean Morphometric Analysis
# ============================================================

import os
import cv2
import glob
import torch
import shutil
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import pearsonr, ttest_rel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from skimage import measure
from skimage.measure import regionprops_table

from circle_fit import taubinSVD

warnings.filterwarnings("ignore")

# ============================================================
# ARGUMENT PARSER
# ============================================================

parser = argparse.ArgumentParser(
    description="Perspective Correction Pipeline"
)

parser.add_argument(
    "--project_dir",
    type=Path,
    default=Path.cwd() / "perspective_correction_pipeline",
    help="Project root directory"
)

parser.add_argument(
    "--image_dir",
    type=Path,
    default=None,
    help="Optional custom image directory"
)

parser.add_argument(
    "--n_images",
    type=int,
    default=None,
    help="Number of images to process (default: all)"
)

args = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = args.project_dir
N_IMAGES = args.n_images

# Default image location
RAW_IMAGES = PROJECT_DIR / "data/images"

# Override if user provides custom image directory
if args.image_dir is not None:
    RAW_IMAGES = args.image_dir
    
GROUNDTRUTH_XLSX = PROJECT_DIR / "data/Faba_Seed_Analyzer_Data_August_2024.xlsx"

OUTPUT_DIR = PROJECT_DIR / "outputs"

ORIGINAL_DIR = OUTPUT_DIR / "original"
CORRECTED_DIR = OUTPUT_DIR / "perspective_corrected"

ORIGINAL_MASKS = ORIGINAL_DIR / "sam_masks"
CORRECTED_MASKS = CORRECTED_DIR / "sam_masks"

ORIGINAL_FEATURES = ORIGINAL_DIR / "features"
CORRECTED_FEATURES = CORRECTED_DIR / "features"

FINAL_DIR = OUTPUT_DIR / "final_comparison"

DEBUG_DIR = OUTPUT_DIR / "debug"
DEBUG_POINTS_DIR = DEBUG_DIR / "source_points"

DEVICE = "cuda"

COIN_DIAMETER_MM = 23.88

# ============================================================
# CREATE FOLDERS
# ============================================================

dirs = [
    ORIGINAL_MASKS,
    CORRECTED_MASKS,
    ORIGINAL_FEATURES,
    CORRECTED_FEATURES,
    FINAL_DIR,
    DEBUG_POINTS_DIR,
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 0 — PERSPECTIVE CORRECTION
# ============================================================

DEST_POINTS = np.float32([
    [2850, 1974],  # bottom-right color card
    [1687, 4815],  # top-right label
    [1687, 5343],  # bottom-right label
    [3583, 4871]   # coin center
])

def get_source_points(df_metadata):

    """
    Source points:
    1. bottom-right corner of color card
    2. top-right corner of label
    3. bottom-right corner of label
    4. center of coin
    """

    # --------------------------------------------------------
    # COLOR CARD
    # --------------------------------------------------------

    colorcard = df_metadata[
        (df_metadata['bbox_x0'] <= 2800) &
        (df_metadata['bbox_y0'] <= 1950) &
        ((df_metadata['bbox_y0'] + df_metadata['bbox_h']) <= 2000)
    ]

    if len(colorcard) == 0:
        raise ValueError("Color card not detected.")

    colorcard = colorcard.iloc[0]

    colorcard_bottom_right = [
        colorcard["bbox_x0"] + colorcard["bbox_w"],
        colorcard["bbox_y0"] + colorcard["bbox_h"]
    ]

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    label = df_metadata[
        (df_metadata['bbox_x0'] <= 1900) &
        (df_metadata['bbox_y0'] >= 4650)
    ]

    if len(label) == 0:
        raise ValueError("Label not detected.")

    label = label.iloc[0]

    label_top_right = [
        label["bbox_x0"] + label["bbox_w"],
        label["bbox_y0"]
    ]

    label_bottom_right = [
        label["bbox_x0"] + label["bbox_w"],
        label["bbox_y0"] + label["bbox_h"]
    ]

    # --------------------------------------------------------
    # COIN
    # --------------------------------------------------------

    coin = df_metadata[
        (df_metadata['bbox_x0'] >= 3000) &
        (df_metadata['area'] >= 200000)
    ]

    if len(coin) == 0:
        raise ValueError("Coin not detected.")

    coin = coin.iloc[0]

    coin_center = [
        coin["bbox_x0"] + (coin["bbox_w"] / 2),
        coin["bbox_y0"] + (coin["bbox_h"] / 2)
    ]

    # --------------------------------------------------------
    # SOURCE POINTS
    # --------------------------------------------------------

    src_points = np.float32([
        colorcard_bottom_right,
        label_top_right,
        label_bottom_right,
        coin_center
    ])

    return src_points

def save_debug_source_points(
        image,
        src_points,
        output_path
):

    """
    Save source points overlay for debugging.
    """

    debug_img = image.copy()

    point_names = [
        "ColorCard_BR",
        "Label_TR",
        "Label_BR",
        "Coin_Center"
    ]

    colors = [
        (0,0,255),     # red
        (0,255,0),     # green
        (255,0,0),     # blue
        (0,255,255)    # yellow
    ]

    # --------------------------------------------------------
    # DRAW POINTS
    # --------------------------------------------------------

    for i, pt in enumerate(src_points):

        x = int(pt[0])
        y = int(pt[1])

        cv2.circle(
            debug_img,
            (x, y),
            25,
            colors[i],
            -1
        )

        cv2.putText(
            debug_img,
            point_names[i],
            (x + 30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            colors[i],
            3
        )

    # --------------------------------------------------------
    # DRAW POLYGON
    # --------------------------------------------------------

    polygon = src_points.astype(np.int32)

    cv2.polylines(
        debug_img,
        [polygon],
        isClosed=True,
        color=(255,255,255),
        thickness=8
    )

    cv2.imwrite(str(output_path), debug_img)

def perspective_correct_image(
        image_path,
        metadata_csv,
        output_path
):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return

    df_metadata = pd.read_csv(metadata_csv)

    try:

        src_points = get_source_points(df_metadata)

        # ========================================================
        # SAVE DEBUG OVERLAY
        # ========================================================

        debug_output_path = (
            DEBUG_POINTS_DIR /
            f"{Path(image_path).stem}_debug_points.JPG"
        )

        save_debug_source_points(
            image=image,
            src_points=src_points,
            output_path=debug_output_path
        )

        # ========================================================
        # HOMOGRAPHY
        # ========================================================

        H = cv2.getPerspectiveTransform(
            src_points,
            DEST_POINTS
        )

        corrected = cv2.warpPerspective(
            image,
            H,
            (4000, 6000),
            flags=cv2.INTER_CUBIC
        )

        cv2.imwrite(str(output_path), corrected)

        print(f"Perspective corrected: {image_path}")

    except Exception as e:

        print(f"Perspective correction failed:")
        print(image_path)
        print(e)

def run_perspective_correction(
        raw_image_dir,
        sam_masks_dir,
        output_dir
):

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        glob.glob(str(raw_image_dir / "*.JPG"))
    )

    # Limit number of images if requested
    if N_IMAGES is not None:
        image_files = image_files[:N_IMAGES]

    for image_path in image_files:

        image_name = Path(image_path).stem

        metadata_csv = (
            sam_masks_dir /
            image_name /
            "metadata.csv"
        )

        if not metadata_csv.exists():

            print(f"Metadata missing for {image_name}")
            continue

        output_path = (
            output_dir /
            f"{image_name}.JPG"
        )

        perspective_correct_image(
            image_path=image_path,
            metadata_csv=metadata_csv,
            output_path=output_path
        )

# ============================================================
# STEP 1 — SAM2 SEGMENTATION
# ============================================================

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_large.pt"

def build_sam_model():

    sam_model = build_sam2(MODEL_CFG, CHECKPOINT)
    sam_model.to(DEVICE)

    mask_generator = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=64,
        pred_iou_thresh=0.7,
        min_mask_region_area=500
    )

    return mask_generator

def run_sam_on_directory(input_dir, output_dir):

    mask_generator = build_sam_model()

    image_files = sorted(glob.glob(str(input_dir / "*.JPG")))
    
    # Limit number of images if requested
    if N_IMAGES is not None:
        image_files = image_files[:N_IMAGES]

    for image_path in image_files:

        image_name = Path(image_path).stem

        print(f"Processing {image_name}")

        image = cv2.imread(image_path)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            masks = mask_generator.generate(image_rgb)

        image_output_dir = output_dir / image_name
        image_output_dir.mkdir(exist_ok=True)

        metadata = []

        for i, mask in enumerate(masks):

            mask_img = (mask["segmentation"] * 255).astype(np.uint8)

            mask_path = image_output_dir / f"{i}.png"

            cv2.imwrite(str(mask_path), mask_img)

            metadata.append({
                "id": i,
                "area": mask["area"],
                "bbox_x0": mask["bbox"][0],
                "bbox_y0": mask["bbox"][1],
                "bbox_w": mask["bbox"][2],
                "bbox_h": mask["bbox"][3],
                "predicted_iou": mask["predicted_iou"],
                "stability_score": mask["stability_score"]
            })

        df_meta = pd.DataFrame(metadata)

        df_meta.to_csv(
            image_output_dir / "metadata.csv",
            index=False
        )

# ============================================================
# STEP 2 — FEATURE EXTRACTION
# ============================================================

def evaluate_coin_mask(mask):

    mask_u8 = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8
    )

    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) == 0:
        return 0.0, 0

    largest_area = areas.max()
    total_area = areas.sum()

    dominant_ratio = largest_area / total_area
    num_components = len(areas)

    return dominant_ratio, num_components

def extract_features(SAM_masks_dir, output_csv, mode):

    sam_suffix = "SAM" if mode == "original" else "PersSAM"

    all_rows = []

    subfolders = sorted(os.listdir(SAM_masks_dir))

    for subfolder in subfolders:

        subfolder_path = os.path.join(SAM_masks_dir, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        print(f"Extracting: {subfolder}")

        metadata_csv = glob.glob(
            os.path.join(subfolder_path, "*.csv")
        )[0]

        df_metadata = pd.read_csv(metadata_csv)

        # ----------------------------------------------------
        # COIN DETECTION
        # ----------------------------------------------------

        df_coin = df_metadata[
            (df_metadata["bbox_x0"] >= 3000) &
            (df_metadata["area"] >= 200000)
        ]

        if len(df_coin) == 0:
            continue

        coin_idx = df_coin.index[0]

        coin_mask_path = os.path.join(
            subfolder_path,
            f"{coin_idx}.png"
        )

        coin_mask = cv2.imread(
            coin_mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        contours_coin, _ = cv2.findContours(
            coin_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        cnt = max(contours_coin, key=cv2.contourArea)

        pts = np.vstack(cnt).squeeze()

        xc, yc, radius_pixels, sigma = taubinSVD(pts)

        calibration_factor_length = (
            COIN_DIAMETER_MM / (2 * radius_pixels)
        )

        calibration_factor_area = (
            np.pi * (COIN_DIAMETER_MM/2)**2
        ) / (np.pi * radius_pixels**2)

        # ----------------------------------------------------
        # FILTER NON-BEANS
        # ----------------------------------------------------

        conditions = [
            (df_metadata['bbox_x0'] <= 2800) & (df_metadata['bbox_y0'] <= 1950),
            (df_metadata['bbox_x0'] <= 1900) & (df_metadata['bbox_y0'] >= 4650),
            (df_metadata['bbox_x0'] <= 4000) & (df_metadata['bbox_y0'] >= 5200),
            (df_metadata['bbox_x0'] >= 3000) & (df_metadata['area'] >= 200000),
            (df_metadata['area'] <= 5000),
            (df_metadata['bbox_h'] >= 1900),
            (df_metadata['bbox_w'] >= 700)
        ]

        for cond in conditions:
            df_metadata = df_metadata.drop(df_metadata[cond].index)

        # ----------------------------------------------------
        # PROCESS EACH BEAN INDIVIDUALLY
        # ----------------------------------------------------

        bean_counter = 0

        for idx in df_metadata.index:

            mask_path = os.path.join(
                subfolder_path,
                f"{idx}.png"
            )

            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            if mask is None:
                continue

            label_image = measure.label(mask)

            props = regionprops_table(
                label_image,
                properties=(
                    'area',
                    'perimeter',
                    'axis_major_length',
                    'axis_minor_length',
                    'centroid'
                )
            )

            if len(props["area"]) == 0:
                continue

            bean_counter += 1

            area_pix = props["area"][0]
            perimeter_pix = props["perimeter"][0]
            major_pix = props["axis_major_length"][0]
            minor_pix = props["axis_minor_length"][0]

            area_mm2 = area_pix * calibration_factor_area
            length_mm = major_pix * calibration_factor_length
            width_mm = minor_pix * calibration_factor_length

            all_rows.append({
                "Image_ID": subfolder,
                "Bean_ID": bean_counter,
                "Mode": mode,

                f"Area(pix)_{sam_suffix}": area_pix,
                f"Perimeter(pix)_{sam_suffix}": perimeter_pix,

                f"Length(mm)_{sam_suffix}": length_mm,
                f"Width(mm)_{sam_suffix}": width_mm,
                f"Area(mm²)_{sam_suffix}": area_mm2
            })

    df_final = pd.DataFrame(all_rows)

    df_final.to_csv(output_csv, index=False)

    print(df_final.head())

# ============================================================
# STEP 3 — MERGE WITH GROUNDTRUTH
# ============================================================

def merge_groundtruth(
        groundtruth_excel,
        original_csv,
        corrected_csv,
        output_excel
):

    gt = pd.read_excel(groundtruth_excel)

    original = pd.read_csv(original_csv)
    corrected = pd.read_csv(corrected_csv)

    print("COLUMNS IN original")
    print(original.columns.tolist())
    print("COLUMNS IN corrected:")
    print(corrected.columns.tolist())

    del original['Mode']
    del corrected['Mode']

    merge = original.merge(
        corrected,
        on=["Image_ID", "Bean_ID"],
        how="left"
    )

    print("COLUMNS IN MERGE")
    print(merge.columns.tolist())

    merged = merge.merge(
        gt,
        on=["Image_ID", "Bean_ID"],
        how="left"
    )

    merged.to_excel(output_excel, index=False)

    print("COLUMNS IN MERGED with GT:")
    print(merged.columns.tolist())

    return merged

# ============================================================
# STEP 4 — STATISTICS + PLOTS
# ============================================================

def compute_statistics(df, mode, output_dir):

    sam_suffix = "SAM" if mode == "original" else "PersSAM"
    traits = [
        (f"Length(mm)_{sam_suffix}", "Length(mm)_MM", "Length"),
        (f"Width(mm)_{sam_suffix}", "Width(mm)_MM", "Width"),
        (f"Area(pix)_{sam_suffix}", "Area(pix)_MM", "Area")
    ]

    stats_rows = []

    for sam_col, mm_col, label in traits:

        sub = df[[sam_col, mm_col]].dropna()

        y_true = sub[mm_col]
        y_pred = sub[sam_col]

        mae = mean_absolute_error(y_true, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        r2 = r2_score(y_true, y_pred)

        r, p = pearsonr(y_true, y_pred)

        bias = np.mean(y_pred - y_true)

        stats_rows.append({
            "Trait": label,
            "Mode": mode,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Pearson_r": r,
            "Bias": bias
        })

        # ----------------------------------------------------
        # SCATTER PLOT
        # ----------------------------------------------------

        plt.figure(figsize=(6,6))

        plt.scatter(y_true, y_pred)

        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())
        ]

        plt.plot(lims, lims, '--')

        plt.xlabel(f"MM {label}")
        plt.ylabel(f"SAM {label}")

        plt.title(f"{label} : {mode}")

        plt.savefig(
            os.path.join(
                output_dir,
                f"{mode}_{label}_scatter.png"
            ),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

    stats_df = pd.DataFrame(stats_rows)

    stats_df.to_csv(
        os.path.join(
            output_dir,
            f"{mode}_statistics.csv"
        ),
        index=False
    )

    return stats_df

# ============================================================
# STEP 5 — COMPARE ORIGINAL VS CORRECTED
# ============================================================

def compare_modes(df, output_dir):

    traits = [
        "Length(mm)",
        "Width(mm)",
        "Area(pix)"
    ]

    comparison_rows = []

    for trait in traits:

        orig_error = np.abs(
            df[f"{trait}_SAM"].values -
            df[f"{trait}_MM"].values
        )

        corr_error = np.abs(
            df[f"{trait}_PersSAM"].values -
            df[f"{trait}_MM"].values
        )

        min_len = min(len(orig_error), len(corr_error))

        orig_error = orig_error[:min_len]
        corr_error = corr_error[:min_len]

        t_stat, p_val = ttest_rel(
            orig_error,
            corr_error
        )

        comparison_rows.append({
            "Trait": trait,
            "Original_MAE": np.mean(orig_error),
            "Corrected_MAE": np.mean(corr_error),
            "P_value": p_val
        })

    df_comp = pd.DataFrame(comparison_rows)

    df_comp.to_csv(
        os.path.join(
            output_dir,
            "perspective_comparison.csv"
        ),
        index=False
    )

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1A — SAM ON ORIGINAL IMAGES
    # --------------------------------------------------------

    print("========================================")
    print("STEP 1A — SAM ORIGINAL")
    print("========================================")

    run_sam_on_directory(
        RAW_IMAGES,
        ORIGINAL_MASKS
    )

    # --------------------------------------------------------
    # STEP 0 — PERSPECTIVE CORRECTION
    # --------------------------------------------------------

    print("========================================")
    print("STEP 0 — PERSPECTIVE CORRECTION")
    print("========================================")

    corrected_images_dir = (
        CORRECTED_DIR / "corrected_images"
    )

    corrected_images_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    run_perspective_correction(
        raw_image_dir=RAW_IMAGES,
        sam_masks_dir=ORIGINAL_MASKS,
        output_dir=corrected_images_dir
    )

    # --------------------------------------------------------
    # STEP 1B — SAM ON CORRECTED IMAGES
    # --------------------------------------------------------

    print("========================================")
    print("STEP 1B — SAM CORRECTED")
    print("========================================")

    run_sam_on_directory(
        corrected_images_dir,
        CORRECTED_MASKS
    )

    # --------------------------------------------------------
    # STEP 2 — FEATURES
    # --------------------------------------------------------

    print("========================================")
    print("STEP 2 — FEATURE EXTRACTION")
    print("========================================")

    original_csv = (
        ORIGINAL_FEATURES /
        "original_features.csv"
    )

    corrected_csv = (
        CORRECTED_FEATURES /
        "corrected_features.csv"
    )

    extract_features(
        ORIGINAL_MASKS,
        original_csv,
        "original"
    )

    extract_features(
        CORRECTED_MASKS,
        corrected_csv,
        "corrected"
    )

    # --------------------------------------------------------
    # STEP 3 — MERGE
    # --------------------------------------------------------

    print("========================================")
    print("STEP 3 — MERGE")
    print("========================================")

    merged_excel = (
        FINAL_DIR /
        "merged_results.xlsx"
    )

    merged_df = merge_groundtruth(
        GROUNDTRUTH_XLSX,
        original_csv,
        corrected_csv,
        merged_excel
    )

    # --------------------------------------------------------
    # STEP 4 — STATISTICS
    # --------------------------------------------------------

    print("========================================")
    print("STEP 4 — STATISTICS")
    print("========================================")

    compute_statistics(
        merged_df,
        "original",
        FINAL_DIR
    )

    compute_statistics(
        merged_df,
        "corrected",
        FINAL_DIR
    )

    # --------------------------------------------------------
    # STEP 5 — FINAL COMPARISON
    # --------------------------------------------------------

    print("========================================")
    print("STEP 5 — COMPARISON")
    print("========================================")

    compare_modes(
        merged_df,
        FINAL_DIR
    )

    print("========================================")
    print("PIPELINE COMPLETED")
    print("========================================")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()