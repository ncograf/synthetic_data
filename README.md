# Synthetic Data Project Code

This project is part of my master thesis on Synthetic Data Generation.
The thesis was supervised by Prof. Dr. Patrick Cheridito (ETH Zürich), Dr. Koushik Balasubramanian (ADIA) and Dr. Vincent Zoonekynd (ADIA).

The goal is to generate synthetic data and evaluate it. As training data we used the S\&P 500 daily stock returns.

## Setup

The project uses [poetry](https://python-poetry.org/) as a
python version manager. To setup this project, install poetry
and run `poetry install` in the projects root directory. You
will an installation of python 3.6 (preferably with managed with
[pyenv](https://github.com/pyenv/pyenv)).

To compile the C++ components of the project, go to the `boosted`
directory and call `setup.sh`.

Further for the plots it is necessary to have latex installed.
More exact requirements are specified in the
[matplotlib documentation](https://matplotlib.org/stable/users/explain/text/pgf.html)

Lastly you need to configure the some environment variables,
as the code depends on it for
storing and caching data.
This can be achieved by defining the variables in `.env`
and installing the
[Poetry Dotenv Plugin](https://pypi.org/project/poetry-dotenv-plugin/).
Necessary variables are

- WANDB_ENTITIY
- WANDB_PROJECT
- WANDB_API_KEY
- RESULT_DIR
- DATA_DIR
- THESIS_DIR

The first three environment variables:
`WANDB_API_KEY`, `WANDB_ENTITY` and `WANDB_PROJECT` are used to
set up online logging with [Weights and Biases](https://docs.wandb.ai/).
The variable `RESULT_DIR` specifies the directory to store
training results, `DATA_DIR` further specifies the directory
to store and cache data used for training. And
last but not least, `THESIS_DIR` specifies a directory
for the scripts to output figures and tables.

## Run the models

TODO

## Code organization

The main folder is `src` containing all the source code.
Moreover, `scripts` contains a had full of scripts
evaluation and training.
The `notebooks` with for jupyter notebooks, contains mostly
experimental code and is used as a kind of scratch paper
convenient to try new ideas.
Unit tests are in `tests` and the `boosted` directory
contains some bits of `c++` code to speed up certain functions.

### src

The source folder is divided into further folder
by categories. For example, each Architecture contains
it's own folder with some files defining the model and
a training scripts.

One of the more important folders is the `stylized_facts`
subfolder. It contains the code for the computations
of the stylized facts, stylized scores and stylized losses.
In particular most of the functionality used
for the evaluation of the models.

The `visualization` subfolder contains code for
plotting stylized facts, result plots and tables.
