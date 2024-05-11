#!/bin/bash

#SBATCH --ntasks=1
##SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=4:00:00
#SBATCH --output=run_output.out
#SBATCH --error=run_error.out

module load eth_proxy

poetry run accelerate launch --config_file accelerate.yaml scripts/train_fourier_flows.py --learning-rate 1e-3 --gamma 0.995 --epochs 700 --batch-size 512 --hidden-dim 200 --num-layer 8 --lag 1 --seq-len 501 --dtype float32 -s MSFT -s AMZN -s TSLA -s AAPL -s IBM -s GS -s BRO -s BLK -s GOOGL
