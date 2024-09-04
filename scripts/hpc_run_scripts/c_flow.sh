#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=10:00:00

module load eth_proxy

poetry run accelerate launch --config_file accelerate.yaml ./src/c_flows/train_c_flow.py --seq-len 2048 --learning-rate 1e-4
