#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=30:00
#SBATCH --output=run_output.out
#SBATCH --error=run_error.out

module load eth_proxy

poetry run python scripts/train_fourier_flows.py --learning-rate 1e-2 --gamma 0.999 --epochs 100 --batch-size 128 --hidden-dim 200 --num-layer 10 --lag 1 --seq-len 101 --dtype float32 --symbols MSFT
