#!/bin/bash -l
#SBATCH --job-name=tsf
#SBATCH --time=48:00:00
#SBATCH --account=plgar2025-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu

module add GCCcore/14.3.0 Python/3.13.5
export PYTHONPATH="${PYTHONPATH}:/net/tscratch/people/plgkpiotr6/mgr/pip/"
python model_training/train_models.py --runs=per_target  --split-per-session --disable-mape --clip-predictions