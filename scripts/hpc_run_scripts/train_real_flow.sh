#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=4096M
#SBATCH --time=10:00:00
#SBATCH --output=run_output.out
#SBATCH --error=run_error.out

module load eth_proxy

poetry run accelerate launch --config_file accelerate.yaml --no_python ./scripts/start_agent.sh -n 5 fbhx2ry5
