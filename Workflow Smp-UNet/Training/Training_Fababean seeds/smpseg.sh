#!/bin/bash
#SBATCH --job-name=smp_seg
#SBATCH --chdir=/home/AGR.GC.CA/bargotah/pipeline/
#SBATCH --output=smpseg.out
#SBATCH --error=smpseg.err
#SBATCH --partition=slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    
#SBATCH --time=48:00:00

srun python train_faba_segmentation.py --train_images faba_images --train_masks MASKS --epochs 50 --batch_size 4 --lr 1e-4
