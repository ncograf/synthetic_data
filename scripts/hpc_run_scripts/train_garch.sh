#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=48:00:00

poetry run python ./src/garch/grid_search_garch.py