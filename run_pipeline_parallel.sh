#!/bin/bash
#SBATCH --job-name=Parallel_Pipeline
#SBATCH --partition=slow
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

### ===============================================
### Usage: (in the faba-bean-image-classification/ directory)
### ===============================================
### sbatch run_pipeline_parallel.sh /path/to/raw_images
### or use default input directory of fabean_images:
### sbatch run_pipeline_parallel.sh
### to run step0 perspective transformation provide STEP0_METHOD, otherwise the default is affine:
### STEP0_METHOD=perspective sbatch run_pipeline_parallel.sh

# ---------------------------------------------------
# Step0 method selection
# Options: affine | perspective
# ---------------------------------------------------
STEP0_METHOD="${STEP0_METHOD:-affine}"

# ---------------------------------------------------
# CLI / ENV parsing
# ---------------------------------------------------
RAW_INPUT_DIR="${1:-${INPUT_DIR:-../../../../../../data/phenomics_images/faba_images}}"

# Unique run folder per job (placed in parent dir of submission cwd)
RUN_SUFFIX="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${SLURM_SUBMIT_DIR}/../${RUN_SUFFIX}_pipeline_output"

mkdir -p "$RUN_DIR"

# ---------------------------------------------------
# Directory layout for this run (all inside RUN_DIR)
# ---------------------------------------------------
STEP0_IMG_DIR="${RUN_DIR}/images_step0"
STEP0_MASK_DIR="${RUN_DIR}/masks_step0"
OUT_SAM="${RUN_DIR}/SAM"
OUT_FE="${RUN_DIR}/FE"

mkdir -p "$STEP0_IMG_DIR" "$STEP0_MASK_DIR" "$OUT_SAM" "$OUT_FE"

# ---------------------------------------------------
# Move SLURM log into run directory as <jobid>.out (wait briefly for file)
# ---------------------------------------------------

# SLURM writes "%x_%j.out" into the submission cwd; we wait a short time
# for it to appear and then move it into the run dir as slurm.out.

# SLURM_STDOUT_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
SLURM_STDOUT_NAME="${SLURM_JOB_NAME:-local}_${SLURM_JOB_ID:-manual}.out"

# DEST_LOG="${RUN_DIR}/${SLURM_JOB_ID}.out"
DEST_LOG="${RUN_DIR}/${SLURM_JOB_ID:-manual}.out"

# Wait up to 60s for SLURM stdout file to appear, then move it.
for i in {1..60}; do
    if [[ -f "$SLURM_STDOUT_NAME" ]]; then
        mv "$SLURM_STDOUT_NAME" "$DEST_LOG"
        break
    fi
    sleep 1
done

# If it never appeared, create an empty placeholder in the run dir
if [[ ! -f "$DEST_LOG" ]]; then
    echo "Note: SLURM stdout not found in submission dir; created placeholder." > "$DEST_LOG"
fi

# ---------------------------------------------------
# EXPORT for Python scripts
# ---------------------------------------------------
export STEP0_IMG_DIR
export STEP0_MASK_DIR
export OUT_FE
export RUN_DIR

# ---------------------------------------------------
# Optional default PRE-affine FE baseline
# ---------------------------------------------------

DEFAULT_PRE_FE="../output_FE_all"

if [[ -z "${OUT_FE_PRE:-}" && -f "${DEFAULT_PRE_FE}/FE_Color.csv" ]]; then
    OUT_FE_PRE="$DEFAULT_PRE_FE"
fi

OUT_FE_POST="$OUT_FE"
export OUT_FE_PRE
export OUT_FE_POST

echo "================ PIPELINE RUN ================"
echo "Raw input images : $RAW_INPUT_DIR"
echo "Run directory    : $RUN_DIR"
echo "Step0 method     : $STEP0_METHOD"
echo "Step0 images     : $STEP0_IMG_DIR"
echo "Step0 masks      : $STEP0_MASK_DIR"
echo "SAM output       : $OUT_SAM"
echo "FE output (POST) : $OUT_FE"
echo "SLURM log        : $DEST_LOG"
echo "=============================================="

if [[ "$STEP0_METHOD" != "affine" && "$STEP0_METHOD" != "perspective" ]]; then
    echo "ERROR: STEP0_METHOD must be affine or perspective"
    exit 1
else echo "Step0 Method: $STEP0_METHOD"
fi

echo "===== PATH CHECK ====="
pwd

for f in \
    sam2/Step0_AffineTransformation.py \
    sam2/Step0_PerspectiveCorrection.py \
    sam2/Step1_SAM2.1.py \
    Step2_SAM2.1.py \
    Step3_color.py \
    plot_histogram.py \
    plot_comparison.py
do
    if [[ -f "$f" ]]; then
        echo "OK: $f"
    else
        echo "MISSING: $f"
        exit 1
    fi
done

echo "======================"


# ---------------------------------------------------
# Step 0: Geometric correction - affine | perspective
# ---------------------------------------------------
if [[ "$STEP0_METHOD" == "affine" ]]; then

    python sam2/Step0_AffineTransformation.py \
        --image-dir "$RAW_INPUT_DIR" \
        --out-img-dir "$STEP0_IMG_DIR" \
        --out-mask-dir "$STEP0_MASK_DIR"

elif [[ "$STEP0_METHOD" == "perspective" ]]; then

    python sam2/Step0_PerspectiveCorrection.py \
        --image-dir "$RAW_INPUT_DIR" \
        --out-img-dir "$STEP0_IMG_DIR" \
        --out-mask-dir "$STEP0_MASK_DIR"

else
    echo "ERROR: Unknown STEP0_METHOD=$STEP0_METHOD"
    exit 1
fi

# ---------------------------------------------------
# Step 1: SAM segmentation (parallel)
# ---------------------------------------------------

srun --mpi=pmi2 -n30 --cpus-per-task=3 --mem-per-cpu=8G python sam2/Step1_SAM2.1.py \
    "$STEP0_IMG_DIR" \
    "$OUT_SAM" \
    --parallel

# python sam2/Step1_SAM2.1.py \
#     "$STEP0_IMG_DIR" \
#     "$OUT_SAM" \
#     --parallel

# ---------------------------------------------------
# Step 2: Feature extraction
# ---------------------------------------------------
python Step2_SAM2.1.py "$OUT_SAM" "$OUT_FE"

# ---------------------------------------------------
# Step 3: Color features
# ---------------------------------------------------
python Step3_color.py "$STEP0_IMG_DIR" "$OUT_FE"

# ---------------------------------------------------
# Plots
# ---------------------------------------------------
CSV_POST="${OUT_FE}/FE_Color.csv"

if [[ -f "$CSV_POST" ]]; then
    python plot_histogram.py --input-csv "$CSV_POST"
else
    echo "WARNING: expected FE CSV not found at $CSV_POST; skipping histogram."
fi

if [[ -n "${OUT_FE_PRE:-}" ]] && [[ -f "${OUT_FE_PRE}/FE_Color.csv" ]]; then
    python plot_comparison.py \
        --csv-pre  "${OUT_FE_PRE}/FE_Color.csv" \
        --csv-post "${OUT_FE_POST}/FE_Color.csv"
else
    echo "INFO: Pre-affine FE not provided or not found. Skipping comparison."
fi

echo "Pipeline completed successfully."
echo "All outputs under: $RUN_DIR"