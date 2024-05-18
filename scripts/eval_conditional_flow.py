import json
import os
from pathlib import Path

import click
import conditional_flow_index_generator
from accelerate.utils import set_seed


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
    "--notes",
    type=str,
    default="",
    help="Notes to describe the run",
)
@click.option(
    "--train-run",
    type=str,
    required=True,
    help="Train run to be used for predicting.",
)
@click.option("--seed", type=int, default=99, help="Seed to fix randomness")
@click.option(
    "--wandb-off",
    is_flag=True,
    help="Turn off wandb logging. (Usefull for debugging or similar)",
)
def eval_conditional_flow(
    num_samples: int, dtype: str, notes: str, train_run: str, seed: int, wandb_off: bool
):
    # set python, numpy, torch, cuda seeds
    set_seed(seed)

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

    index_generator = conditional_flow_index_generator.ConditionalFlowIndexGenerator()

    # compute the evaluations and store it on wandb
    index_generator.sample_wandb_index(
        train_run=train_run,
        dtype=dtype,
        notes=notes,
        sample_len=num_samples,
        seed=seed,
    )


if __name__ == "__main__":
    eval_conditional_flow()
