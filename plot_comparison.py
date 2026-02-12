# ============================================================
# How to run
# ============================================================
# This script compares two FE_Color.csv files (pre vs post Step0) against
# ground truth and saves plots/outputs to:
#   <RUN_DIR>/plot_comparison/
#
# Ground truth file used (relative to your current working directory):
#   helper_scripts/Groundtruth_data/Faba_Seed_Analyzer_Data_August_2024.xlsx
#   sheet: "Individual S2 Seed Data"
#
# 1) Run via the pipeline .sh (SLURM):
#   cd faba-bean-image-classification
#   sbatch run_pipeline_parallel.sh /path/to/raw_images
#   # plot_comparison.py is called automatically if a pre-data FE exists.
#
# 2) Manual run (no scheduler, no .sh):
#   cd faba-bean-image-classification   # so the ground-truth relative path works
#
#   # Option A: pass CSV paths directly
#   export RUN_DIR=/path/to/21874_pipeline_output   # where you want outputs saved
#   python plot_comparison.py \
#     --csv-pre  /path/to/pre_run/FE/FE_Color.csv \
#     --csv-post /path/to/post_run/FE/FE_Color.csv
#
#   # Option B: use env vars (no CLI flags)
#   export OUT_FE_PRE=/path/to/pre_run/FE
#   export OUT_FE_POST=/path/to/post_run/FE
#   export RUN_DIR=/path/to/21874_pipeline_output
#   python plot_comparison.py
#
# Notes:
# - If RUN_DIR is not set, the script tries to pick the latest "*_pipeline_output" under ~/faba-bean-project.
# - If the ground-truth .xlsx is elsewhere, edit gt_path/gt_sheet in this file.
# ============================================================

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import glob

# ======================================================
# ARGUMENT PARSING
# ======================================================
parser = argparse.ArgumentParser(description="Compare Pre/Post affine FE results against ground truth")
parser.add_argument("--method", default=os.environ.get("STEP0_METHOD", "affine"), help="Step0 normalization method (affine or perspective)")
parser.add_argument("--csv-pre", help="Pre-affine FE_Color.csv")
parser.add_argument("--csv-post", help="Post-affine FE_Color.csv")
args = parser.parse_args()

method_name = args.method.lower()

# ======================================================
# RESOLVE CSV PATHS
# ======================================================
if args.csv_pre:
    csv_pre = args.csv_pre
else:
    out_fe_pre = os.environ.get("OUT_FE_PRE")
    if not out_fe_pre:
        sys.exit("ERROR: --csv-pre not provided and OUT_FE_PRE not set.")
    csv_pre = os.path.join(out_fe_pre, "FE_Color.csv")

if args.csv_post:
    csv_post = args.csv_post
else:
    out_fe_post = os.environ.get("OUT_FE_POST")
    if not out_fe_post:
        sys.exit("ERROR: --csv-post not provided and OUT_FE_POST not set.")
    csv_post = os.path.join(out_fe_post, "FE_Color.csv")

for p in (csv_pre, csv_post):
    if not os.path.exists(p):
        sys.exit(f"ERROR: CSV file not found: {p}")


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

    
# ======================================================
# OUTPUT DIRECTORY (BASE/plot_comparison)
# ======================================================

# default RUN_DIR is the last run output folder
BASE_PATH = "~/faba-bean-project"
DEFAULT_RUN_DIR = get_latest_run_dir(BASE_PATH)
run_dir = os.environ.get("RUN_DIR", DEFAULT_RUN_DIR)

print(f"Targeting directory: {run_dir}")

# run_dir = os.environ.get("RUN_DIR")
if not run_dir:
    sys.exit("ERROR: RUN_DIR environment variable not set.")
output_dir = os.path.join(run_dir, "plot_comparison")
os.makedirs(output_dir, exist_ok=True)

print(f"Pre CSV : {csv_pre}")
print(f"Post CSV: {csv_post}")
print(f"Output  : {output_dir}")

# ======================================================
# USER SETTINGS
# ======================================================
gt_path  = "helper_scripts/Groundtruth_data/Faba_Seed_Analyzer_Data_August_2024.xlsx"
gt_sheet = "Individual S2 Seed Data"

bins = 40
exclude_patterns = [] # no outlier at this stage, all images are fine!

METHODS = ["SAM", "SAM_taubin", "SAM_minEnc"]

DIMENSIONS = {
    "Length": {
        "gt_mm": "GT_MM_Length",
        "gt_dcm": "GT_DCM_Length",
        "xlabel": "Length (mm)"
    },
    "Width": {
        "gt_mm": "GT_MM_Width",
        "gt_dcm": "GT_DCM_Width",
        "xlabel": "Width (mm)"
    }
}

PRE_STAGE_LABEL  = "Pre"
POST_STAGE_LABEL = f"Post-{method_name.capitalize()}"

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def normalize_id(x):
    if pd.isna(x): return "nan"
    # Convert to string, remove leading/trailing whitespace, 
    # and remove any hidden carriage returns (\r) or tabs (\t)
    s = str(x).strip().replace('\r', '').replace('\n', '').replace('\t', '')
    # s = str(x)
    return s[s.index("Vf"):] if "Vf" in s else s

def find_id_column(df):
    for c in df.columns:
        if c.lower() in ("class", "id"):
            return c
    raise ValueError("No ID/Class column found")

def build_col(dim, method, stage):
    return f"{dim}-{method}(mm)_{stage}"

def safe_filename(s):
    """
    Convert a string into a filesystem-safe filename.
    """
    s = s.replace("—", "")
    s = s.replace("/", "_")
    s = s.replace("\\", "_")
    s = re.sub(r"[^\w\-_.() ]", "", s)
    return s.replace(" ", "_")

# ======================================================
# LOAD CSV DATA
# ======================================================
df_pre  = pd.read_csv(csv_pre)
df_post = pd.read_csv(csv_post)

for df in (df_pre, df_post):
    id_col = find_id_column(df)
    df[id_col] = df[id_col].apply(normalize_id)
    df.rename(columns={id_col: "ID"}, inplace=True)

# ======================================================
# LOAD GROUND TRUTH
# ======================================================
gt = pd.read_excel(gt_path, sheet_name=gt_sheet)
gt["ID"] = gt["ID"].astype(str).apply(normalize_id)

gt = gt.rename(columns={
    "Length(mm)":   "GT_MM_Length",
    "Width(mm)":    "GT_MM_Width",
    "Length(mm).1": "GT_DCM_Length",
    "Width(mm).1":  "GT_DCM_Width",
})

# ======================================================
# MERGE & FILTER
# ======================================================

merged = (
    df_pre.merge(df_post, on="ID", suffixes=("_Pre", "_Post"))
          .merge(
              gt[["ID", "GT_MM_Length", "GT_MM_Width",
                   "GT_DCM_Length", "GT_DCM_Width"]],
              on="ID",
              how="inner"
          )
)

if exclude_patterns:
    # Only filter if the list is not empty
    exclude_mask = merged["ID"].str.contains("|".join(exclude_patterns), na=False)
    merged = merged.loc[~exclude_mask].reset_index(drop=True)
    print(f"Applied exclusion. Rows remaining: {len(merged)}")
else:
    print("No exclusion patterns provided. Keeping all rows.")

# Ensure we have data left before continuing
if merged.empty:
    sys.exit("ERROR: Dataframe is empty after filtering. Check your patterns or merge.")

# # This replaces inf with NaN so .dropna() and .notna() work correctly
# merged = merged.replace([np.inf, -np.inf], np.nan)

print(f"Rows after adding Ground Truth: {len(merged)}")


# ======================================================
# PLOTTING FUNCTIONS
# ======================================================
def plot_pre_post_histogram(df, pre_col, post_col, title, xlabel):
    # Select only the rows where both columns are valid
    mask = df[pre_col].notna() & df[post_col].notna()
    data_subset = df[mask]
    
    if data_subset.empty:
        print(f"Skipping histogram {title}: No valid data.")
        return

    pre = data_subset[pre_col]
    post = data_subset[post_col]

    combined = pd.concat([pre, post])
    bins_ = np.histogram_bin_edges(combined, bins=bins)

    plt.figure(figsize=(7, 5))
    plt.hist(pre,  bins=bins_, alpha=0.6, label=PRE_STAGE_LABEL)
    plt.hist(post, bins=bins_, alpha=0.6, label=POST_STAGE_LABEL)

    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    
    fname = safe_filename(title)
    save_path = os.path.join(output_dir, f"{fname}_dist.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}") # Confirmation print
    plt.close()

def compare_pre_post(df, pre_col, post_col, gt_col, title):
    # CRITICAL: Find rows where all three columns have real numbers
    mask = df[pre_col].notna() & df[post_col].notna() & df[gt_col].notna()
    data_subset = df[mask]

    if data_subset.empty:
        print(f"Skipping comparison {title}: No overlapping valid data.")
        return {}, {}

    pre = data_subset[pre_col]
    post = data_subset[post_col]
    gt = data_subset[gt_col]

    def metrics(pred):
        err = pred - gt
        return {
            "mean_error": err.mean(),
            "rmse": np.sqrt((err**2).mean()),
            "mean_abs_error": err.abs().mean(),
            "mean_rel_error_%": (err.abs() / gt).mean() * 100,
            "correlation": pred.corr(gt),
        }

    m_pre, m_post = metrics(pre), metrics(post)

    # ---------- Scatter plots ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    
    # Calculate limits for the 1:1 line
    all_vals = pd.concat([gt, pre, post])
    lim_min, lim_max = all_vals.min(), all_vals.max()
    lims = [lim_min, lim_max]

    ax[0].scatter(gt, pre, alpha=0.5)
    ax[0].plot(lims, lims, "k--")
    ax[0].set_title(PRE_STAGE_LABEL)
    ax[0].set_xlabel("Ground Truth (mm)")
    ax[0].set_ylabel("Predicted (mm)")
    ax[0].grid(alpha=0.2)

    ax[1].scatter(gt, post, alpha=0.5)
    ax[1].plot(lims, lims, "k--")
    ax[1].set_title(POST_STAGE_LABEL)
    ax[1].set_xlabel("Ground Truth (mm)")
    ax[1].grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leaves space for suptitle
    fname = safe_filename(title)
    plt.savefig(os.path.join(output_dir, f"{fname}_scatter.png"), dpi=300)
    plt.close()

    # ---------- Error histogram ----------
    plt.figure(figsize=(7, 5))
    plt.hist(pre - gt,  bins=bins, alpha=0.6, label=PRE_STAGE_LABEL)
    plt.hist(post - gt, bins=bins, alpha=0.6, label=POST_STAGE_LABEL)
    plt.xlabel("Prediction − Ground Truth (mm)")
    plt.ylabel("Frequency")
    plt.title(f"{title} — Error Distribution")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(output_dir, f"{fname}_error.png"), dpi=300)
    plt.close()

    return m_pre, m_post

# ======================================================
# SUMMARY PLOTS FROM SAVED CSV
# ======================================================
def plot_summary_from_csv(csv_path, save_dir):

    if not os.path.exists(csv_path):
        print(f"Summary CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("Summary CSV is empty. Skipping plots.")
        return

    # consistent ordering
    df = df.sort_values(["method", "dimension", "ground_truth"])

    # readable label
    df["label"] = (
        df["method"] + " | " +
        df["dimension"] + " | " +
        df["ground_truth"]
    )

    def make_plot(metric, title, filename):

        pivot = df.pivot(index="label", columns="stage", values=metric)

        if PRE_STAGE_LABEL not in pivot or POST_STAGE_LABEL  not in pivot:
            print(f"Skipping {metric} plot: missing stages.")
            return

        plt.figure(figsize=(10, 6))

        plt.plot(pivot.index, pivot[PRE_STAGE_LABEL], marker='o')
        plt.plot(pivot.index, pivot[POST_STAGE_LABEL], marker='x')

        plt.xticks(rotation=90)
        plt.title(title)
        plt.tight_layout()

        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

    make_plot("rmse",
              "RMSE Pre vs Post",
              "rmse_comparison.png")

    make_plot("mean_abs_error",
              "Mean Absolute Error Pre vs Post",
              "mae_comparison.png")

    make_plot("correlation",
              "Correlation Pre vs Post",
              "correlation_comparison.png")

# ======================================================
# RUN ALL COMPARISONS
# ======================================================
summary = []

for method in METHODS:
    for dim, meta in DIMENSIONS.items():
        pre_col  = build_col(dim, method, "Pre")
        post_col = build_col(dim, method, "Post")

        if pre_col not in merged or post_col not in merged:
            continue

        plot_pre_post_histogram(
            merged,
            pre_col,
            post_col,
            f"{dim} Distribution ({method}) — Pre vs Post",
            meta["xlabel"]
        )

        for gt_name, gt_col in [("GT_MM", meta["gt_mm"]),
                                ("GT_DCM", meta["gt_dcm"])]:

            title = f"{dim} ({method}) — Pre/Post vs {gt_name}"

            m_pre, m_post = compare_pre_post(
                merged, pre_col, post_col, gt_col, title
            )

            summary.append({
                "method": method,
                "dimension": dim,
                "ground_truth": gt_name,
                "stage": PRE_STAGE_LABEL,
                **m_pre
            })
            summary.append({
                "method": method,
                "dimension": dim,
                "ground_truth": gt_name,
                "stage": POST_STAGE_LABEL,
                **m_post
            })

# ======================================================
# SAVE METRICS
# ======================================================
summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    os.path.join(output_dir, "ground_truth_comparison_all_methods.csv"),
    index=False
)

print(summary_df)


# ------------------------------------------------------
# NEW: Generate summary plots from saved CSV
# ------------------------------------------------------
summary_csv_path = os.path.join(
    output_dir,
    "ground_truth_comparison_all_methods.csv"
)

plot_summary_from_csv(
    summary_csv_path,
    output_dir
)