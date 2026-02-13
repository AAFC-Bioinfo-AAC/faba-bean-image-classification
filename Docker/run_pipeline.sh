#!/bin/bash
set -e  # Exit immediately if any command fails

# Usage check
if [ "$#" -ne 2 ]; then
    echo "Usage: docker run ... <input_folder> <output_folder>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

echo "========================================================"
echo " Faba Bean Analysis Pipeline (CPU Mode)"
echo " Authors: M. Richards, H.K. Bargota, H.N.T. Wang"
echo "========================================================"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

# ---------------------------------------------------------
# STEP 1: SAM 2.1 Segmentation
# ---------------------------------------------------------
echo -e "\n[Step 1/3] Running SAM 2.1 Segmentation..."
# Note: We run from /app so the relative path "sam2/checkpoints" works
cd /app
python scripts/step1_sam.py "$INPUT_DIR" "$OUTPUT_DIR/step1_masks"

# ---------------------------------------------------------
# STEP 2: Feature Extraction
# ---------------------------------------------------------
echo -e "\n[Step 2/3] Extracting Morphological Features..."
# Input is the Output from Step 1
python scripts/step2_features.py "$OUTPUT_DIR/step1_masks" "$OUTPUT_DIR/step2_features"

# ---------------------------------------------------------
# STEP 3: Color Analysis
# ---------------------------------------------------------
echo -e "\n[Step 3/3] Analyzing Color & TGW..."
# Step 3 reads the original images AND the CSV from Step 2
# Note: Step 3 expects the CSV in the output folder.
python scripts/step3_color.py "$INPUT_DIR" "$OUTPUT_DIR/step2_features"

echo -e "\n========================================================"
echo " Pipeline Complete!"
echo " Results saved to: $OUTPUT_DIR/step2_features"
echo "========================================================"