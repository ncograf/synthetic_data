import json
import os
from pathlib import Path
from typing import List

import chained_filter
import click
import conditional_flow_index_generator
import len_filter
import numpy as np
import pandas as pd
import real_data_loader as data
import time_filter


@click.command()
@click.option(
    "-l",
    "--learning-rate",
    type=float,
    default=1e-3,
    help="Initial learning rate to train the model.",
)
@click.option(
    "-g",
    "--gamma",
    type=float,
    default=1,
    help="Gamma factor for exponential LRscheduler.",
)
@click.option("-e", "--epochs", type=int, default=400, help="Training epochs.")
@click.option("-b", "--batch-size", type=int, default=512, help="Training batch size.")
@click.option(
    "-h", "--hidden-dim", type=int, default=128, help="Hidden layer dimension."
)
@click.option(
    "-n", "--num-layer", type=int, default=10, help="Number of Layer in Neural Network."
)
@click.option(
    "--lag", type=int, default=1, help="Lag for creating the trianing seqences."
)
@click.option(
    "-sl", "--seq-len", type=int, default=512, help="Seqence Lenght to train on."
)
@click.option(
    "-d",
    "--dtype",
    type=click.Choice(["float16", "float32", "float64"]),
    default="float32",
    help="Data type to train the model.",
)
@click.option(
    "-s",
    "--symbols",
    type=str,
    multiple=True,
    default=["MSFT", "AMZN"],
    help="Symbols to be fitted. For each symbol one fit is done.",
)
@click.option("--seed", type=int, default=99, help="Seed to be used for training")
@click.option(
    "--wandb-off",
    is_flag=True,
    help="Turn off wandb logging. (Usefull for debugging or similar)",
)
def train_conditional_flow(
    learning_rate: float,
    gamma: float,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    num_layer: int,
    lag: int,
    seq_len: int,
    dtype: str,
    symbols: List[str],
    seed: int,
    wandb_off: bool,
):
    root_dir = Path(__file__).parent.parent

    if wandb_off:
        os.environ["WANDB_MODE"] = "disabled"
    else:
        api_key_file = root_dir / "wandb_keys.json"
        if not api_key_file.exists():
            raise FileExistsError(f"No API key found in {api_key_file} for wandb.")
        os.environ["WANDB_MODE"] = "online"
        with api_key_file.open() as open_file:
            os.environ["WANDB_API_KEY"] = json.load(open_file)["synthetic_data"]
            open_file.close()

    data_loader = data.RealDataLoader(cache=root_dir / "data/cache")
    price_data = data_loader.get_timeseries(
        "Adj Close", data_path=root_dir / "data/raw_yahoo_data"
    )

    index_generator = conditional_flow_index_generator.ConditionalFlowIndexGenerator(
        cache="data/cache"
    )

    # first filter training data
    price_data = price_data.loc[:, symbols]
    first_date = pd.Timestamp(year=1990, month=1, day=1)
    min_lenght = 2000
    filter_list = [
        len_filter.LenFilter(min_lenght),
    ]
    ch_filter = chained_filter.ChainedFilter(
        filter_chain=filter_list,
        time_filter=time_filter.TimeFilter(first_date=first_date),
    )
    ch_filter.fit_filter(price_data)
    ch_filter.apply_filter(price_data)

    # remove all nan rows
    mask = np.any(np.isnan(np.array(price_data)), axis=1)
    price_data.drop(index=price_data.index[mask], inplace=True)

    # fit all symbols in the index and store them on wandb
    config = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "num_layer": num_layer,
        "lag": lag,
        "seq_len": seq_len,
    }
    index_generator.fit_index(
        price_index_data=price_data, train_config=config, seed=seed, dtype=dtype
    )


if __name__ == "__main__":
    train_conditional_flow()
