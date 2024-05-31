#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=05:00:00
#SBATCH --output=run_output.out
#SBATCH --error=run_error.out

module load eth_proxy

poetry run accelerate launch --config_file accelerate.yaml --no_python ./scripts/start_agent.sh -n 10 eip6bnc8
