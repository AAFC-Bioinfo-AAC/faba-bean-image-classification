#hkb

#__Project__="Faba bean Feature extraction pipeline (Step1)"
#References include:
#https://github.com/facebookresearch/sam2
#https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/
#Adapted from https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py


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
# checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
def main(input_dir, output_dir):
    """
        Generates the binary images (.png) and metadata file (.csv) using Segmentanything model SAM 2.1 for images. The checkpoints used are 'sam2.1_hiera_large.pt' 
        and model configs is sam2.1_hiera_l.yaml'.

        Arguments:
        input_dir: The input directory containing the faba bean images (.JPG) files. The images have the class labels as the name of the files.
        output_dir: The output directory where the binary masks (.png, indexed from 0 onwards- 0.png, 1.png) and metadata file (.csv) containing 
        the information about the area, bounding box cordinates, input points, IOU score, crop box for the objects in the binary mask.

         Raises:
        TypeError: If the conditions are not met.
    
        """  

    # Device setup (uncomment if CUDA is available)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # Specify the device (CPU in this case)
    device = "cpu"
    #print(f"Using device: {device}")

    # Load SAM2 model
    # if not os.path.exists(checkpoint):
    #     raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    # print("Loading SAM2 model...")
    sam_model = build_sam2(model_cfg, checkpoint)
    sam_model.to(device)
    mask_generator = SAM2AutomaticMaskGenerator(sam_model)

    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the input directory
    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_dir, image_name)
            print(f"Processing {image_name}...")

            # Read and prepare the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read {image_name}. Skipping...")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create subdirectory for image masks
            image_base_name = os.path.splitext(image_name)[0]
            image_output_dir = os.path.join(output_dir, image_base_name)
            os.makedirs(image_output_dir, exist_ok=True)

            # Generate masks
            with torch.inference_mode():
                masks = mask_generator.generate(image_rgb)

            # Metadata storage
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

            # Save metadata as CSV
            df_metadata = pd.DataFrame(metadata, columns=metadata_header)
            metadata_path = os.path.join(image_output_dir, "metadata.csv")
            df_metadata.to_csv(metadata_path, index=False)

            print(f"Masks and metadata saved for {image_name}.")

    print("All images processed!")

if __name__ == "__main__":
    # Ensure correct usage
    if len(sys.argv) != 5:
        print("Usage: python script.py input_dir <input_directory> output_dir <output_directory>")
        sys.exit(1)

    # Parse command-line arguments
    input_arg = sys.argv[1]
    input_dir = sys.argv[2]
    output_arg = sys.argv[3]
    output_dir = sys.argv[4]

    if input_arg != "input_dir" or output_arg != "output_dir":
        print("Error: Arguments must be in the form 'input_dir <input_directory> output_dir <output_directory>'")
        sys.exit(1)

    # Run main function
    main(input_dir, output_dir)
