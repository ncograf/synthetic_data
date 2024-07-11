#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

poetry run python ./src/garch/grid_search_garch.py