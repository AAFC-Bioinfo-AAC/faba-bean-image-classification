#__authors__= "Mathew Richards" and "harpreet kaur bargota"
#__email__= "mathew.richards@agr.gc.ca" and "harpreet.bargota@agr.gc.ca"
#__Project__="Faba bean Feature extraction pipeline (Step1)"

#References include:
#https://github.com/facebookresearch/sam2
#https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/
#Adapted from https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py


# import the required libraries
import os
import sys
import cv2
import torch
import pandas as pd
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Static model configuration and checkpoint paths
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"

def main(input_dir, output_dir):
    """
    Generates binary images (.png) and metadata (.csv) using the SAM 2.1 segmentation model.

    Args:
        input_dir (str): Directory containing input images (.JPG, .JPEG, .PNG)
        output_dir (str): Directory to save output masks and metadata

    Raises:
        FileNotFoundError: If input_dir doesn't exist.
        ValueError: If images are missing or unreadable.
    """

    # Expected dimensions
    EXPECTED_WIDTH = 4000
    EXPECTED_HEIGHT = 6000

    # Device setup
    device = "cpu"

    # Load SAM2 model
    sam_model = build_sam2(model_cfg, checkpoint)
    sam_model.to(device)
    mask_generator = SAM2AutomaticMaskGenerator(sam_model)

    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Gather valid image files
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        raise ValueError(
            f"The input directory '{input_dir}' does not contain any valid image files "
            f"in supported formats {valid_extensions}."
        )

    # Track invalid images
    invalid_images = []

    # Process each image
    for image_name in image_files:
        image_path = os.path.join(input_dir, image_name)
        print(f"\nProcessing {image_name}...")

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  Warning: Could not read '{image_name}'. Skipping...")
            invalid_images.append((image_name, "Unreadable image"))
            continue

        # Validate shape
        height, width, _ = image.shape
        if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
            print(
                f"❌ Error: Image '{image_name}' does not have the required dimensions.\n"
                f"   Expected → Width={EXPECTED_WIDTH}, Height={EXPECTED_HEIGHT}\n"
                f"   Found    → Width={width}, Height={height}\n"
                f"   Skipping this image..."
            )
            invalid_images.append(
                (image_name, f"Invalid dimensions (found {width}x{height})")
            )
            continue  # Continue with the next image

        # If valid dimensions, proceed with processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create subdirectory for image masks
        image_base_name = os.path.splitext(image_name)[0]
        image_output_dir = os.path.join(output_dir, image_base_name)
        os.makedirs(image_output_dir, exist_ok=True)

        # Generate masks
        with torch.inference_mode():
            masks = mask_generator.generate(image_rgb)

        # Prepare metadata
        metadata = []
        metadata_header = [
            "id", "area", "bbox_x0", "bbox_y0", "bbox_w", "bbox_h",
            "point_input_x", "point_input_y", "predicted_iou", "stability_score",
            "crop_box_x0", "crop_box_y0", "crop_box_w", "crop_box_h"
        ]

        for i, mask in enumerate(masks):
            # Save mask image
            mask_image = (mask["segmentation"] * 255).astype("uint8")
            output_path = os.path.join(image_output_dir, f"{i}.png")
            Image.fromarray(mask_image).save(output_path)

            # Save metadata
            mask_metadata = [
                i,
                mask["area"],
                *mask["bbox"],
                *mask["point_coords"][0],
                mask["predicted_iou"],
                mask["stability_score"],
                *mask["crop_box"]
            ]
            metadata.append(mask_metadata)

        # Write metadata CSV
        df_metadata = pd.DataFrame(metadata, columns=metadata_header)
        metadata_path = os.path.join(image_output_dir, "metadata.csv")
        df_metadata.to_csv(metadata_path, index=False)

        print(f"✅ Masks and metadata saved for {image_name}.")

    # Summary report
    print("\nProcessing complete!")
    if invalid_images:
        print("⚠️  The following images were skipped due to errors:")
        for name, reason in invalid_images:
            print(f"   - {name}: {reason}")
    else:
        print("✅ All images met the required conditions and were processed successfully.")


if __name__ == "__main__":
    # Ensure correct usage
    if len(sys.argv) != 3:
        print("Usage: python sam2/Step1_SAM2.1.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    main(input_dir, output_dir)
