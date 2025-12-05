#!/bin/bash
#SBATCH --job-name=fababean-pipelineSAM
#SBATCH --chdir=/home/AGR.GC.CA/bargotah/image/
#SBATCH --output=pipeline.out
#SBATCH --error=pipeline.err
#SBATCH --partition=slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00


# Run Python scripts sequentially
echo "Running Step1..."

srun python amg.py --checkpoint sam_vit_h_4b8939.pth --model-type vit_h --input faba_images --output ouput_SAM --device cpu

if [ $? -ne 0 ]; then
    echo "Error: Step1 failed. Exiting."
    exit 1
fi


echo "Running Step2..."

srun python Step2_SAM.py ouput_SAM ouput_FE

if [ $? -ne 0 ]; then
    echo "Error: Step2 failed. Exiting."
    exit 1
fi

echo "Running Step3..."

python Step3_color.py faba_images ouput_FE

if [ $? -ne 0 ]; then
    echo "Error: Step3 failed. Exiting."
    exit 1
fi

echo "All scripts executed successfully!"

