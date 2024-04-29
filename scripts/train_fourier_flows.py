import json
import os
from pathlib import Path
from typing import List

import click
import fourier_flow_generator
import real_data_loader as data


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
@click.option(
    "-h", "--hidden-dim", type=int, default=200, help="Hidden layer dimension."
)
@click.option(
    "-n", "--num-layer", type=int, default=10, help="Number of Layer in Neural Network."
)
@click.option(
    "--lag", type=int, default=1, help="Lag for creating the trianing seqences."
)
@click.option(
    "-s", "--seq-len", type=int, default=101, help="Seqence Lenght to train on."
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
    "--symbol",
    type=str,
    multiple=True,
    default=["MSFT"],
    help="Symbols to be fitted. For each symbol one fit is done.",
)
def train_fourier_flow(
    learning_rate: float,
    gamma: float,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    num_layer: int,
    lag: int,
    seq_len: int,
    dtype: str,
    symbol: List[str],
):
    root_dir = Path(__file__).parent.parent
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
    fourier_flow = fourier_flow_generator.FourierFlowGenerator(
        dtype=dtype, cache=root_dir / "data/cache", use_cuda=True
    )

    for sym in symbol:
        fourier_flow.symbol = sym  # set symbol mainly for logging purposes
        fourier_flow.fit_model(price_data=price_data.loc[:, sym], **config)

        # TODO remove the todo tag
        fourier_flow.generate_data(
            len=500, burn=0, model_version="latest", tags=["debug version"]
        )


if __name__ == "__main__":
    train_fourier_flow()
