# Cli to run the script: python ./Trained_Smp_UNet.py --images faba_images --bean_model smp_unet_mitb0_faba.pth --coin_model coin_unet_mitb0_coin.pth --coin_diameter_mm 23.88

# --------------------------- 
# Auto-install required packages if missing
# ---------------------------

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "segmentation-models-pytorch",
    "timm",
    "efficientnet-pytorch",
    "Pillow"
]

for pkg in required_packages:
    try:
        __import__(pkg.split("-")[0])  # crude check if installed
    except ImportError:
        print(f"⚡ Installing missing package: {pkg}")
        install(pkg)

# ---------------------------
# Imports
# ---------------------------

import os
import argparse
import torch
import numpy as np
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image, ImageOps
import segmentation_models_pytorch as smp
from skimage import measure

"""
Python script for automated seed phenotyping that uses deep learning–based U-Net
segmentation to detect seeds and a reference coin, performs pixel-to-millimeter
calibration, extracts morphological and shape features, estimates thousand grain
weight (TGW), saves binary segmentation masks, and outputs seed-level features
and seed counts as CSV files.
"""

# ---------------------------
# Model loading
# ---------------------------
def load_model(model_path, device):
    model = smp.Unet(
        encoder_name="mit_b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

# ---------------------------
# Image preprocessing
# ---------------------------
def preprocess(image_path, img_size):
    img = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

def predict_mask(model, img_tensor, device, thresh=0.5):
    with torch.no_grad():
        pred = model(img_tensor.to(device))
    return (pred.squeeze().cpu().numpy() > thresh).astype(np.uint8)

# ---------------------------
# Feature extraction
# ---------------------------
def compute_features(region, px_to_mm):
    area_px = region.area
    perimeter_px = region.perimeter
    major = region.major_axis_length
    minor = region.minor_axis_length

    length_mm = major * px_to_mm
    width_mm = minor * px_to_mm
    area_mm2 = area_px * (px_to_mm ** 2)
    perimeter_mm = perimeter_px * px_to_mm

    roundness = (4 * np.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else 0
    circularity_sam = 1 / roundness if roundness > 0 else 0

    shapefactor1 = major / area_px if area_px > 0 else 0
    shapefactor2 = minor / area_px if area_px > 0 else 0
    shapefactor3 = area_px / (((major / 2) ** 2) * np.pi) if major > 0 else 0
    shapefactor4 = area_px / ((major / 2) * (minor / 2) * np.pi) if major > 0 and minor > 0 else 0

    aspect_ratio = major / minor if minor > 0 else 0

    TGW_g = (
        296.9785397519971
        + (5.5020 * area_mm2)
        + (18.4537 * width_mm)
        + (-17.3898 * length_mm)
        + (-607.6333 * circularity_sam)
        + (344.1165 * aspect_ratio)
    )

    return {
        "Area_mm2": area_mm2,
        "Length_mm": length_mm,
        "Width_mm": width_mm,
        "Perimeter_mm": perimeter_mm,
        "Area_pix": area_px,
        "Axis_Major_Length_pix": major,
        "Axis_Minor_Length_pix": minor,
        "Perimeter_pix": perimeter_px,
        "Eccentricity": region.eccentricity,
        "Equivalent_Diameter": region.equivalent_diameter,
        "Solidity": region.solidity,
        "Convex_Area": region.convex_area,
        "Extent": region.extent,
        "Centroid_Row": region.centroid[0],
        "Centroid_Col": region.centroid[1],
        "Aspect_Ratio": aspect_ratio,
        "Roundness": roundness,
        "Circularity_SAM": circularity_sam,
        "Shapefactor1": shapefactor1,
        "Shapefactor2": shapefactor2,
        "Shapefactor3": shapefactor3,
        "Shapefactor4": shapefactor4,
        "TGW_g": TGW_g
    }

# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.mask_out, exist_ok=True)

    bean_model = load_model(args.bean_model, device)
    coin_model = load_model(args.coin_model, device)

    feature_rows = []
    count_rows = []

    for img_name in os.listdir(args.images):
        if not img_name.endswith(".JPG"):
            continue

        img_path = os.path.join(args.images, img_name)
        base = os.path.splitext(img_name)[0]

        img_tensor = preprocess(img_path, args.img_size)

        coin_mask = predict_mask(coin_model, img_tensor, device)
        coin_regions = measure.regionprops(measure.label(coin_mask))

        if len(coin_regions) == 0:
            print(f"⚠ No coin detected in {img_name}")
            continue

        coin = max(coin_regions, key=lambda r: r.area)
        px_to_mm = args.coin_diameter_mm / coin.major_axis_length

        bean_mask = predict_mask(bean_model, img_tensor, device)

        mask_path = os.path.join(args.mask_out, f"{base}_mask.png")
        cv2.imwrite(mask_path, bean_mask * 255)

        bean_regions = measure.regionprops(measure.label(bean_mask))
        count_rows.append({"class": base, "seed_count": len(bean_regions)})

        for i, region in enumerate(bean_regions):
            feats = compute_features(region, px_to_mm)
            feats["class"] = base
            feats["seed_id"] = i + 1
            feats["px_to_mm"] = px_to_mm
            feature_rows.append(feats)

    pd.DataFrame(feature_rows).to_csv(args.feature_csv, index=False)
    pd.DataFrame(count_rows).to_csv(args.count_csv, index=False)

    print("Feature extraction process completed successfully")
    print(f"Features CSV: {args.feature_csv}")
    print(f"Seed count CSV: {args.count_csv}")
    print(f"Binary masks saved in: {args.mask_out}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Seed Phenotyping (CSV + Masks)")
    parser.add_argument("--images", required=True)
    parser.add_argument("--bean_model", required=True)
    parser.add_argument("--coin_model", required=True)
    parser.add_argument("--coin_diameter_mm", type=float, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--feature_csv", default="seed_features.csv")
    parser.add_argument("--count_csv", default="seed_count.csv")
    parser.add_argument("--mask_out", default="bean_masks")
    args = parser.parse_args()
    run_pipeline(args)
