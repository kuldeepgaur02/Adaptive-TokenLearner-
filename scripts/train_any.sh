#!/bin/bash
#SBATCH --partition=cse-gpu-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openmmlab

cd ~/mmdetection3d
nvidia-smi

python -u tools/train.py "$1" --resume
