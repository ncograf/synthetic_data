import json
import os
from pathlib import Path
from typing import List

import click
import fourier_flow_index_generator


@click.command()
@click.option(
    "-n",
    "--num-samples",
    type=int,
    default=2000,
    help="Number of samples to sample.",
)
@click.option(
    "-d",
    "--dtype",
    type=click.Choice(["float32", "float64"]),
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
def eval_fourier_flow(
    num_samples,
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

    index_generator = fourier_flow_index_generator.FourierFlowIndexGenerator()

    # compute the evaluations and store it on wandb
    artifacts = [f"base_fourier_flow_{sym}" for sym in symbol]
    version = ["latest" for _ in symbol]
    model_info = list(zip(artifacts, symbol, version))
    index_generator.sample_wandb_index(
        model_info=model_info, dtype=dtype, sample_len=num_samples
    )


if __name__ == "__main__":
    eval_fourier_flow()
