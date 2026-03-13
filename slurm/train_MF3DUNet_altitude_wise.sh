#!/bin/bash
#SBATCH --job-name="MF3D_vol"
#SBATCH --output=outputs/train.MF3D_vol.%J.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=62G

#SBATCH --account=p709-24-2

export TMPDIR=/scratch/p709-24-2/tmp
cd ..
srun /home/ppavlik/miniconda3/envs/nowcasting/bin/python \
 train_model.py \
 MF-3D-U-Net_altitude_wise -n Layers8