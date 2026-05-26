#!/bin/bash
#SBATCH --job-name=perspective_correction_pipeline
#SBATCH --partition=slow
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=perspective_correction_pipeline/outputs/%x_%j.out

# python perspective_correction_pipeline/main.py --image_dir "/data/phenomics_images/faba_images" --n_images 1

python perspective_correction_pipeline/main.py --image_dir "/data/phenomics_images/faba_images"