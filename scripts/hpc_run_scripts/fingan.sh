#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=48:00:00

module load eth_proxy

#poetry run accelerate launch --config-file accelerate.yaml ./src/fingan_takahashi/train_fingan.py --dist normal -m mean -m var -m skew -m skew -m kurt -s lu -s le -s vc -s cf
poetry run accelerate launch --config-file accelerate.yaml ./src/fingan_takahashi/train_fingan.py --dist stdnormal
