# Synthetic Data Project Code

The project uses [poetry](https://python-poetry.org/) as a python version manager.

Moreover the porject uses [Weights and Biases](https://docs.wandb.ai/) for logging.
To make sure everything works, set the `WANDB_API_KEY`, `WANDB_ENTITY` and `WANDB_PROJECT` environment
variables, before running the code.
This can be achieved by defining them in `.env` and installing the [Poetry Dotenv Plugin](https://pypi.org/project/poetry-dotenv-plugin/).

## Code organization

The main folders are `src` containing all the source code, `scripts` for training, evaluation and miscellaneous scrits and the `notebooks` folder for jupyter notebooks.
Unit tests are in `tests` and the `boosted` directory contains some bits of `c++` code to speed up certain functions.