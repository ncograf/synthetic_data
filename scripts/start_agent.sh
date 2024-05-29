#!/bin/bash

# Help to display when calling wrong
help()
{
    echo "Start agent with a given sweepID"
    echo
    echo "Syntax: start_agent sweep_id [-h | -n #runs]"
    echo "Note that before using this script there has to be an existing"
    echo "sweep on wandb, which can be created with init_sweep.sh."
    echo "Parameter:"
    echo "sweep_id   Sweep id on wandb."
    echo "Options:"
    echo "h     Print this Help."
    echo "n     Number of runs the agent should do."
    echo
    echo "Example:"
    echo "start_agent -n 20 2kafwer3"
    echo
}

# initialize sweep with the first parameter given
while getopts ":hn:" option; do
    case $option in
        h|\?) # display Help
            help
            exit;;
        n)
            N=$OPTARG
    esac
done

if [[ -z $1 ]];
then
    help
    exit 1
fi

# create sweep
if [[ -z $N ]]
then
    poetry run wandb agent $1
else
    poetry run wandb agent --count $N $3
fi
    