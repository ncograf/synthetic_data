#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH --output=run_output.out
#SBATCH --error=run_error.out

module load eth_proxy

#poetry run accelerate launch --config_file accelerate.yaml ./src/fourier_flows/train_fourier_flow.py --dist laplace -s lu -s vc -s le -s cf
poetry run accelerate launch --config_file accelerate.yaml ./src/fourier_flows/train_fourier_flow.py --dist normal -s lu -s vc -s le -s cf
