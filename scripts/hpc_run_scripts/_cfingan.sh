#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=06:00:00

module load eth_proxy

#poetry run accelerate launch --config_file accelerate.yaml ./src/fourier_flows/train_fourier_flow.py --dist laplace -s lu -s vc -s le -s cf
poetry run accelerate launch --config_file accelerate.yaml ./src/cfingan/train_cfingan.py --dist normal --seq-len 256
