# __authors__ = "Mathew Richards, Harpreet Kaur Bargota"
# __project__ = "Faba Bean Feature Extraction Pipeline (Step 1)"
# __credits__ = "Agriculture and Agri-Food Canada"

import os
import sys
import cv2
import torch
import pandas as pd
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- Configuration ---
# Paths to the SAM 2 model configuration and weights (inside the container)
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_small.pt"

# Target dimensions (Post-Resize)
EXPECTED_WIDTH = 2000
EXPECTED_HEIGHT = 3000

def main(input_dir, output_dir):
    """
    Runs SAM 2.1 on all images in input_dir and saves masks/metadata to output_dir.
    Automatically resizes images to 50% to ensure stability on CPU.
    """
    
    # 1. Initialize SAM 2 Model
    # We explicitly force CPU mode and reduce batch size to prevent memory crashes
    print("[Step 1/3] Initializing SAM 2.1 (CPU Mode)...")
    device = "cpu"
    sam_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam_model, points_per_batch=16)

    # 2. Verify Directories
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' not found.")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Find Images
    valid_exts = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

    if not image_files:
        raise ValueError(f"No valid images found in '{input_dir}'.")

    invalid_images = []

    # 4. Process Loop
    for image_name in image_files:
        print(f"\nProcessing {image_name}...")
        image_path = os.path.join(input_dir, image_name)

        # Load Image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping unreadable image: {image_name}")
            invalid_images.append((image_name, "Unreadable"))
            continue

        # Resize Image (50%)
        # Crucial for CPU RAM management. 4000x6000 -> 2000x3000
        scale_percent = 0.50
        width_new = int(image.shape[1] * scale_percent)
        height_new = int(image.shape[0] * scale_percent)
        image = cv2.resize(image, (width_new, height_new), interpolation=cv2.INTER_AREA)

        # Validate New Dimensions
        h, w, _ = image.shape
        if w != EXPECTED_WIDTH or h != EXPECTED_HEIGHT:
            print(f"Error: Invalid dimensions for {image_name}. Found {w}x{h}, expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}.")
            invalid_images.append((image_name, f"Invalid Size ({w}x{h})"))
            continue

        # Prepare Output Folder
        image_base_name = os.path.splitext(image_name)[0]
        image_output_dir = os.path.join(output_dir, image_base_name)
        os.makedirs(image_output_dir, exist_ok=True)

        # Generate Masks (Inference)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            masks = mask_generator.generate(image_rgb)

        # Save Results
        metadata = []
        for i, mask in enumerate(masks):
            # Save binary mask as PNG
            mask_img = (mask["segmentation"] * 255).astype("uint8")
            Image.fromarray(mask_img).save(os.path.join(image_output_dir, f"{i}.png"))

            # Collect metadata
            metadata.append([
                i,
                mask["area"],
                *mask["bbox"],
                *mask["point_coords"][0],
                mask["predicted_iou"],
                mask["stability_score"],
                *mask["crop_box"]
            ])

        # Save Metadata CSV
        columns = [
            "id", "area", "bbox_x0", "bbox_y0", "bbox_w", "bbox_h",
            "point_input_x", "point_input_y", "predicted_iou", "stability_score",
            "crop_box_x0", "crop_box_y0", "crop_box_w", "crop_box_h"
        ]
        pd.DataFrame(metadata, columns=columns).to_csv(
            os.path.join(image_output_dir, "metadata.csv"), index=False
        )

        print(f"Saved {len(masks)} masks for {image_name}.")

    # 5. Final Report
    print("\n" + "="*40)
    print("Processing Complete!")
    if invalid_images:
        print("Skipped Images:")
        for name, reason in invalid_images:
            print(f"   - {name}: {reason}")
    else:
        print("All images processed successfully.")
    print("="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/step1_sam.py <input_dir> <output_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])