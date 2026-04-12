#!/bin/bash -l
#SBATCH --job-name=tsf
#SBATCH --time=48:00:00
#SBATCH --account=plgar2025-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu

module add Python
export PYTHONPATH="${PYTHONPATH}:/net/tscratch/people/plgkpiotr6/pip"
python train_models.py