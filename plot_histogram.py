# ============================================================
# How to run
# ============================================================
# This script reads a Feature-Extraction CSV (FE_Color.csv) and saves plots to:
#   <RUN_DIR>/plot_histograms/
#
# 1) Via the pipeline .sh (SLURM):
#   cd faba-bean-image-classification
#   sbatch run_pipeline_parallel.sh /path/to/raw_images
#   # plot_histogram.py is called automatically at the end of the pipeline.
#
# 2) Manual run (no scheduler, no .sh):
#   # Option A: pass CSV directly
#   export RUN_DIR=/path/to/No#_pipeline_output
#   python plot_histogram.py --input-csv "/path/to/No#_pipeline_output/FE/FE_Color.csv"
#
#   # Option B: use OUT_FE (no CLI args)
#   export OUT_FE=/path/to/No#_pipeline_output/FE
#   export RUN_DIR=/path/to/No#_pipeline_output
#   python plot_histogram.py
#
# Notes:
# - If RUN_DIR is not set, the script tries to pick the latest "*_pipeline_output" under ~/faba-bean-project.
# - Output folder is always "<RUN_DIR>/plot_histograms" (created if missing).
# ============================================================

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import sys
import glob

def get_latest_run_dir(base_path):
    # Expand the user path (e.g., /home/user/...)
    full_base_path = os.path.expanduser(base_path)
    
    # Create a pattern to match folders like '21868_pipeline_output'
    # '*' matches the variable numbers
    search_pattern = os.path.join(full_base_path, "*_pipeline_output")
    
    # Get a list of all matching directories
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        # Fallback if no directories exist yet
        return os.path.join(full_base_path, "default_pipeline_output")

    # Extract the number, convert to int for proper sorting, and find the max
    # We split by '_' and take the first part of the folder name
    latest_dir = max(matching_dirs, key=lambda x: int(os.path.basename(x).split('_')[0]))
    
    return latest_dir
    
# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Generate histogram and scatter plots from FE CSV")
parser.add_argument(
    "--input-csv",
    help="Path to FE_Color.csv. If omitted, uses $OUT_FE/FE_Color.csv",
    default=None
)
args = parser.parse_args()

# -------------------------
# Resolve input CSV path
# -------------------------
if args.input_csv:
    csv_path = args.input_csv
else:
    out_fe = os.environ.get("OUT_FE")
    if not out_fe:
        sys.exit("ERROR: --input-csv not provided and OUT_FE environment variable not set.")
    csv_path = os.path.join(out_fe, "FE_Color.csv")

if not os.path.exists(csv_path):
    sys.exit(f"ERROR: CSV file not found: {csv_path}")

# -------------------------
# Resolve output directory
# RUN_DIR/plot_histograms
# -------------------------

# default RUN_DIR is the last run output folder
BASE_PATH = "~/faba-bean-project"
DEFAULT_RUN_DIR = get_latest_run_dir(BASE_PATH)
run_dir = os.environ.get("RUN_DIR", DEFAULT_RUN_DIR)
print(f"Targeting directory: {run_dir}")

# run_dir = os.environ.get("RUN_DIR")
if not run_dir:
    sys.exit("ERROR: RUN_DIR environment variable not set.")
output_dir = os.path.join(run_dir, "plot_histograms")
os.makedirs(output_dir, exist_ok=True)

print(f"Using CSV: {csv_path}")
print(f"Saving plots to: {output_dir}")

# -------------------------
# User settings
# -------------------------
length_keywords = ["length"]
width_keywords = ["width"]
bins = 35

# -------------------------
def extract_id(id_value):
    s = str(id_value)
    if "Vf" in s:
        return s.split("Vf", 1)[1]
    return s

# -------------------------
# Load CSV
# -------------------------
df = pd.read_csv(csv_path)

# Identify ID column
id_col = next((c for c in df.columns if 'class' in c.lower()), None)

# Find width columns
width_cols = [c for c in df.columns if "width" in c.lower() and "(mm)" in c.lower()]

# Find length columns
length_cols = [
    c for c in df.columns
    if ("length" in c.lower()) and ("(mm)" in c.lower()) and ("pix" not in c.lower())
]

print("Length columns found:", length_cols)
print("Width columns found:", width_cols)

# -------------------------
# Plot helper
# -------------------------
def plot_group(columns, title, filename):
    if not columns:
        print(f"No columns found for {title}.")
        return

    plt.figure(figsize=(12, 6))

    outlier_records = []
    all_outlier_ids = set()

    for col in columns:
        cols = [col] + ([id_col] if id_col else [])
        col_data = df[cols].dropna()

        outliers = (col_data[col] > 35) | (col_data[col] < 3)

        if outliers.any():
            for _, row in col_data.loc[outliers].iterrows():
                outlier_records.append({
                    "column": col,
                    "id": row[id_col] if id_col else row.name,
                    "value": row[col],
                })

            if id_col:
                all_outlier_ids.update(col_data.loc[outliers, id_col].unique())
            else:
                all_outlier_ids.update(col_data.loc[outliers].index.tolist())

        clean_data = col_data.loc[~outliers, col]
        plt.hist(clean_data, bins=bins, alpha=0.5, label=col)

    plt.title(title)
    plt.xlabel("Value (mm)")
    plt.ylabel("Frequency")
    plt.legend()

    if all_outlier_ids:
        plt.gca().text(
            1.02, 0.5,
            "Outlier IDs:\n" + "\n".join(map(str, sorted(all_outlier_ids))),
            transform=plt.gca().transAxes,
            fontsize=9,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    if outlier_records:
        outlier_df = pd.DataFrame(outlier_records)
        outlier_df.to_csv(
            out_path.replace(".png", "_outliers.csv"),
            index=False
        )

# -------------------------
# Scatter plot logic
# -------------------------
target_lengths = ['Length-SAM(mm)', 'Length-SAM_taubin(mm)', 'Length-SAM_minEnc(mm)']
threshold = 35

# -------------------------
# Generate plots
# -------------------------
plot_group(length_cols, "All Length Distributions", "all_lengths.png")
plot_group(width_cols, "All Width Distributions", "all_widths.png")

for l_col in target_lengths:
    if l_col not in df.columns or not width_cols:
        print(f"Skipping {l_col}: column not found.")
        continue

    suffix = l_col.split('-')[-1].split('(')[0]
    w_col = next((c for c in width_cols if suffix in c), width_cols[0])

    mask_or = (
        (df[w_col] > threshold) |
        (df[l_col] > threshold) |
        (df[l_col] < 3) |
        (df[w_col] < 3)
    )

    flagged_df = df[mask_or]
    flagged_ids = flagged_df[id_col].unique() if id_col else flagged_df.index.tolist()

    print(f"\n--- {l_col} vs {w_col} ---")
    print(f"Flagged IDs: {list(flagged_ids)}")

    plt.figure(figsize=(10, 7))

    normal = df[~mask_or]
    plt.scatter(normal[w_col], normal[l_col], alpha=0.5, label="Valid")

    plt.axvline(threshold, linestyle="--", alpha=0.3)
    plt.axhline(threshold, linestyle="--", alpha=0.3)

    plt.title(f"{l_col} vs {w_col} (Threshold {threshold} mm)")
    plt.xlabel(w_col)
    plt.ylabel(l_col)
    plt.legend()
    plt.grid(alpha=0.2)

    save_name = f"separation_{suffix.lower()}.png"
    plt.savefig(os.path.join(output_dir, save_name), dpi=300)
    plt.close()

    print(f"Saved: {save_name}")
