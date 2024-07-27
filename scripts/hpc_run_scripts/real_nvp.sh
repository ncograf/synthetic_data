#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00

module load eth_proxy

# poetry run accelerate launch --config_file accelerate.yaml ./src/real_nvps/train_real_nvp.py --dist laplace -s lu -s vc -s le -s cf
poetry run accelerate launch --config_file accelerate.yaml ./src/real_nvps/train_real_nvp.py --dist normal
