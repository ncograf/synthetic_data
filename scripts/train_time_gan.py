import json
import os
from pathlib import Path
from typing import List

import click
import real_data_loader as data
import time_gan_index_generator


@click.command()
@click.option(
    "-l",
    "--learning-rate",
    type=float,
    default=1e-2,
    help="Initial learning rate to train the model.",
)
@click.option(
    "-g",
    "--gamma",
    type=float,
    default=0.999,
    help="Gamma factor for exponential LRscheduler.",
)
@click.option("-e", "--epochs", type=int, default=1000, help="Training epochs.")
@click.option("-b", "--batch-size", type=int, default=128, help="Training batch size.")
@click.option("-h", "--hidden-dim", type=int, default=4, help="Hidden layer dimension.")
@click.option(
    "-n", "--num-layer", type=int, default=3, help="Number of Layer in Neural Network."
)
@click.option(
    "--lag", type=int, default=1, help="Lag for creating the trianing seqences."
)
@click.option(
    "-sl", "--seq-len", type=int, default=24, help="Seqence Lenght to train on."
)
@click.option(
    "--embed-dim", type=int, default=2, help="Hidden embedding dimension in networks"
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
    default=["MSFT"],
    help="Symbols to be fitted. For each symbol one fit is done.",
)
@click.option(
    "--wandb-off",
    is_flag=True,
    help="Turn off wandb logging. (Usefull for debugging or similar)",
)
def train_time_gan(
    learning_rate: float,
    gamma: float,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    num_layer: int,
    lag: int,
    seq_len: int,
    embed_dim: int,
    dtype : str,
    symbols: List[str],
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

    index_generator = time_gan_index_generator.TimeGanIndexGenerator("data/cache")

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
        "embed_dim": embed_dim,
    }
    index_generator.fit_index(
        price_index_data=price_data.loc[:, symbols], dtype=dtype, train_config=config
    )


if __name__ == "__main__":
    train_time_gan()
