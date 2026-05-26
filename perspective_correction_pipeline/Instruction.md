# Faba Bean SAM Pipeline

End-to-end pipeline for faba bean morphometric analysis using SAM2 segmentation, perspective correction, and statistical evaluation.

---

# Pipeline Overview

The pipeline performs:

1. Perspective correction
2. SAM2 segmentation
3. Bean feature extraction
4. Ground truth merging
5. Statistical analysis and comparison

---

# Project Structure

```text
perspective_correction_pipeline/
│
├── data/
│   ├── images/
│   └── Faba_Seed_Analyzer_Data_August_2024.xlsx
│
├── outputs/
│   ├── original/
│   ├── perspective_corrected/
│   ├── final_comparison/
│   └── debug/
│
└── main.py
```

---

# Requirements

## Python Packages

Install required packages:

```bash
pip install numpy pandas matplotlib scipy scikit-image scikit-learn opencv-python torch circle-fit
```

SAM2 installation is also required.

## Activate conda environment

```bash
conda activate fababean_env
```

---

# Input Data

## Images

Place raw `.JPG` images inside:

```text
perspective_correction_pipeline/data/images/
```

## Ground Truth Excel

Place the Excel file here:

```text
perspective_correction_pipeline/data/Faba_Seed_Analyzer_Data_August_2024.xlsx
```

---

# Running the Pipeline

Run with the default path (data/images):

```bash
python perspective_correction_pipeline/main.py
```

Or:

```bash
python main.py --image_dir path/to/images --n_images number_of_images
```

Example:

```bash
python main.py --image_dir "/data/phenomics_images/faba_images" --n_images 20
```

Schedule a slurm job:
```bash
sbatch perspective_correction_pipeline/run.sh
```

---

# Output Files

## Segmentation Masks

Generated in:

```text
outputs/original/sam_masks/
outputs/perspective_corrected/sam_masks/
```

## Feature CSV Files

```text
outputs/original/features/
outputs/perspective_corrected/features/
```

## Final Results

```text
outputs/final_comparison/
```

Includes:

- merged Excel results
- statistics CSV files
- scatter plots
- perspective comparison results

---

# Debug Outputs

Perspective correction source points are saved for inspection:

```text
outputs/debug/source_points/
```

These images help verify whether the selected homography points are correct.

---

# Notes

- Coin diameter is used for spatial calibration:
  
```python
COIN_DIAMETER_MM = 23.88
```

- The pipeline assumes:
  - fixed camera settings
  - consistent image resolution
  - visible coin, color card, and label in each image

- GPU is recommended for SAM2 inference.

---