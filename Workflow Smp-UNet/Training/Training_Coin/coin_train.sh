#!/bin/bash
#SBATCH --job-name=std_train
#SBATCH --chdir=/home/AGR.GC.CA/bargotah/pipeline
#SBATCH --output=cointrain.out
#SBATCH --error=cointrain.err
#SBATCH --partition=slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    
#SBATCH --time=48:00:00

srun python coin_training.py --train_images faba_images --train_masks coin_masks --epochs 20 --batch_size 4 --lr 1e-4

