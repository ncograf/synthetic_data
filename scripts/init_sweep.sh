# Help to display when calling wrong
help()
{
    echo "Initialize new sweep and output it's Id in the terminal."
    echo
    echo "Syntax: init_sweep conifg_file [-h]"
    echo "parameter:"
    echo "  config_file   Path to configuration file."
    echo "options:"
    echo "  h     Print this Help."
    echo
}

# initialize sweep with the first parameter given
while getopts ":h:" option; do
    case $option in
        h|\?) # display Help
            help
            exit;;
    esac
done

if [[ -z $1 ]];
then
    help
    exit 1
fi

# create sweep
poetry run wandb sweep $1