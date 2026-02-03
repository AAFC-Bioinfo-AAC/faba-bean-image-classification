# import the required libraries
import os
import sys
import shutil
import cv2
import torch
import pandas as pd
from PIL import Image
from math import ceil
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# Static model configuration and checkpoint paths
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_large.pt"

# Expected dimensions
EXPECTED_WIDTH = 4000
EXPECTED_HEIGHT = 6000


def get_slurm_task_info():
    """
    Get SLURM task information from environment variables.
    Works with both srun -n<N> and SLURM array jobs.
    
    Returns:
        tuple: (task_id, num_tasks) - 0-indexed task ID and total number of tasks
    """
    # First check for srun -n<N> (SLURM_PROCID and SLURM_NTASKS)
    if 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        task_id = int(os.environ['SLURM_PROCID'])
        num_tasks = int(os.environ['SLURM_NTASKS'])
        print(f"[SLURM srun mode] Task {task_id + 1}/{num_tasks}", flush=True)
        return task_id, num_tasks
    
    # Check for SLURM array jobs
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        # SLURM_ARRAY_TASK_COUNT might not always be set
        num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 
                        os.environ.get('SLURM_ARRAY_TASK_MAX', task_id + 1)))
        print(f"[SLURM array mode] Task {task_id + 1}/{num_tasks}", flush=True)
        return task_id, num_tasks
    
    # No SLURM environment - single task
    print("[Sequential mode] No SLURM environment detected", flush=True)
    return 0, 1


def distribute_files(files, task_id, num_tasks):
    """
    Distribute files across tasks evenly.
    
    Args:
        files (list): List of all files
        task_id (int): Current task ID (0-indexed)
        num_tasks (int): Total number of tasks
        
    Returns:
        list: Files assigned to this task
    """
    if num_tasks <= 1:
        return files
    
    total_files = len(files)
    chunk_size = ceil(total_files / num_tasks)
    start_idx = task_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_files)
    
    assigned_files = files[start_idx:end_idx]
    
    print(f"Task {task_id}: Processing files {start_idx} to {end_idx-1} ({len(assigned_files)} files)", flush=True)
    
    return assigned_files


def _build_sam_model(device="cpu"):
    """
    Build and return SAM2 model and mask generators.
    
    Args:
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        tuple: (mask_generator, mask_generator188)
    """
    torch.cuda.empty_cache()
    
    sam_model = build_sam2(MODEL_CFG, CHECKPOINT)
    sam_model.to(device)
    
    # Default mask generator
    mask_generator = SAM2AutomaticMaskGenerator(sam_model)
    
    # Special mask generator for "188" images
    mask_generator188 = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=64,
        pred_iou_thresh=0.6,
        min_mask_region_area=500
    )
    
    return mask_generator, mask_generator188


def _process_single_image(image_path, output_dir, mask_generator, mask_generator188):
    """
    Process a single image with SAM2 model.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save output masks and metadata
        mask_generator: Default SAM2 mask generator
        mask_generator188: Special mask generator for "188" images
        
    Returns:
        tuple: (success, image_name, error_message)
    """
    image_name = os.path.basename(image_path)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️  Warning: Could not read '{image_name}'. Skipping...", flush=True)
        return (False, image_name, "Unreadable image")
    
    # Validate shape
    height, width, _ = image.shape
    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        error_msg = f"Invalid dimensions (found {width}x{height}, expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT})"
        print(f"❌ Error: Image '{image_name}' - {error_msg}. Skipping...", flush=True)
        return (False, image_name, error_msg)
    
    # Create subdirectory for image masks
    image_base_name = os.path.splitext(image_name)[0]
    image_output_dir = os.path.join(output_dir, image_base_name)
    metadata_file = os.path.join(image_output_dir, "metadata.csv")
    
    # Check if already processed by verifying metadata.csv exists (indicates successful completion)
    if os.path.exists(metadata_file):
        print(f"Skipping {image_name} (already processed - metadata.csv exists)", flush=True)
        return (True, image_name, "Already processed")
    
    # If directory exists but no metadata.csv, it's an incomplete run - remove and reprocess
    if os.path.exists(image_output_dir):
        print(f"Removing incomplete output directory for {image_name}", flush=True)
        shutil.rmtree(image_output_dir)
    
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    if "188" in image_name:
        with torch.inference_mode():
            masks = mask_generator188.generate(image_rgb)
    else:
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
    
    print(f"✅ Masks and metadata saved for {image_name} ({len(masks)} masks).", flush=True)
    
    torch.cuda.empty_cache()
    
    return (True, image_name, None)


def _sam_segment(files, output_dir, device="cpu"):
    """
    Process multiple images with SAM2 model.
    
    Args:
        files (list): List of image file paths to process
        output_dir (str): Directory to save output masks and metadata
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        list: List of (success, image_name, error_message) tuples
    """
    # Handle case where files might be empty
    if not files:
        print(f"No files to process in this chunk", flush=True)
        return []
    
    print(f"Processing {len(files)} files", flush=True)
    print(f"First file: {files[0]}", flush=True)
    if len(files) > 1:
        print(f"Last file: {files[-1]}", flush=True)
    
    # Build the model once for all images in this chunk
    print(f"Building SAM2 model on device: {device}", flush=True)
    mask_generator, mask_generator188 = _build_sam_model(device)
    print(f"SAM2 model built successfully", flush=True)
    
    results = []
    invalid_images = []
    
    for idx, image_path in enumerate(files):
        print(f"Processing image {idx+1}/{len(files)}: {os.path.basename(image_path)}", flush=True)
        result = _process_single_image(
            image_path, 
            output_dir, 
            mask_generator, 
            mask_generator188
        )
        results.append(result)
        
        if not result[0] and result[2] != "Already processed":
            invalid_images.append((result[1], result[2]))
    
    # Summary for this chunk
    print(f"Chunk processing complete! Processed {len(files)} images.", flush=True)
    if invalid_images:
        print(f"The following images were skipped due to errors:", flush=True)
        for name, reason in invalid_images:
            print(f"   - {name}: {reason}", flush=True)
    
    return results


def main(input_dir, output_dir, device="cpu"):
    """
    Main function for SAM2 segmentation with SLURM parallel processing support.
    
    Automatically detects SLURM environment (srun -n<N>) and distributes work.
    Each SLURM task processes a different subset of images.
    
    Args:
        input_dir (str): Directory containing input images (.JPG, .JPEG, .PNG)
        output_dir (str): Directory to save output masks and metadata
        device (str): Device to use ('cuda' or 'cpu'), default 'cpu'
    """
    # Get SLURM task info (automatically detects if running under SLURM)
    task_id, num_tasks = get_slurm_task_info()
    
    print(f"========================================", flush=True)
    print(f"SAM2 Segmentation - Task {task_id}/{num_tasks}", flush=True)
    print(f"========================================", flush=True)
    print(f"Input directory: {input_dir}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    
    # Ensure output directory exists (all tasks do this - it's idempotent)
    os.makedirs(output_dir, exist_ok=True)
    
    # Gather valid image files (sorted for consistent distribution across tasks)
    valid_extensions = (".jpg", ".jpeg", ".png")
    all_image_files = sorted([
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(valid_extensions)
    ])
    
    if not all_image_files:
        raise ValueError(
            f"The input directory '{input_dir}' does not contain any valid image files "
            f"in supported formats {valid_extensions}."
        )
    
    print(f"Found {len(all_image_files)} total images", flush=True)
    
    # Distribute files across SLURM tasks
    my_files = distribute_files(all_image_files, task_id, num_tasks)
    
    if not my_files:
        print(f"Task {task_id}: No files assigned, exiting.", flush=True)
        return
    
    # Filter out already processed images from this task's assigned files
    files_to_process = []
    for image_path in my_files:
        image_base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, image_base_name)
        metadata_file = os.path.join(image_output_dir, "metadata.csv")
        # Only skip if metadata.csv exists (indicates successful completion)
        if os.path.exists(metadata_file):
            print(f"Skipping {os.path.basename(image_path)} - already processed", flush=True)
        else:
            files_to_process.append(image_path)
    
    if not files_to_process:
        print(f"Task {task_id}: All assigned images already processed.", flush=True)
        return
    
    print(f"Task {task_id}: {len(files_to_process)} images to process (after filtering)", flush=True)
    
    # Process the files
    _sam_segment(files_to_process, output_dir=output_dir, device=device)
    
    print(f"Task {task_id}: SAM2 segmentation completed!", flush=True)


if __name__ == "__main__":
    # Ensure correct usage
    if len(sys.argv) < 3:
        print("Usage: python SAM2_segment.py <input_directory> <output_directory> [--device <device>]", flush=True)
        print("", flush=True)
        print("Examples:", flush=True)
        print("  Sequential:  python SAM2_segment.py ./images ./output", flush=True)
        print("  Parallel:    srun -n4 --cpus-per-task=8 --mem-per-cpu=8G python SAM2_segment.py ./images ./output", flush=True)
        print("", flush=True)
        print("The script automatically detects SLURM environment and distributes work.", flush=True)
        print("No --parallel flag needed!", flush=True)
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Parse optional arguments
    device = 'cpu'  # default
    if '--device' in sys.argv:
        idx = sys.argv.index('--device')
        if idx + 1 < len(sys.argv):
            device = sys.argv[idx + 1]
    
    main(input_dir, output_dir, device=device)
