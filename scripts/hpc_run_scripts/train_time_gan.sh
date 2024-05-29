#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=5:00:00
#SBATCH --output=run_output.out
#SBATCH --error=run_error.out

module load eth_proxy

poetry run accelerate launch --config_file accelerate.yaml scripts/train_time_gan.py --learning-rate 1e-3 --gamma 0.995 --epochs 100 --batch-size 512 --hidden-dim 4 --num-layer 3 --lag 1 --seq-len 32 --dtype float32 -s MSFT -s AMZN 
