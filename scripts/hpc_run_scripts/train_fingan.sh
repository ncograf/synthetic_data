#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=24:00:00
#SBATCH --output=run_output.out
#SBATCH --error=run_error.out

module load eth_proxy

poetry run accelerate launch --config-file accelerate.yaml ./src/fingan_takahashi/train_fingan.py
