#!/bin/bash
#SBATCH --job-name="MF3D"
#SBATCH --output=outputs/train.MF3D.%J.out
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=124G

#SBATCH --account=XXX

export TMPDIR=/scratch/XXX/tmp
cd ..
srun /home/ppavlik/miniconda3/envs/nowcasting/bin/python \
 train_model.py \
 MF-3D-U-Net -n CMAX