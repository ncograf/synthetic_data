#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=1:00:00
#SBATCH --output=run_output.out
#SBATCH --error=run_error.out

TRAIN_RUN=$(head -n 1 "last_train_run")

module load eth_proxy

poetry run accelerate launch --config_file accelerate.yaml scripts/eval_fourier_flows.py --num-samples 2000 --dtype float32 --train-run $TRAIN_RUN
