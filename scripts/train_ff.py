import os
import time
from pathlib import Path

import fourier_flow_index_generator
import real_data_loader as data
import wandb_train


# @click.command()
# @click.option(
#    "--wandb-off",
#    is_flag=True,
#    help="Turn off wandb logging. (Usefull for debugging or similar)",
# )
def main(
    wandb_off: bool,
):
    root_dir = Path(__file__).parent.parent

    # get real data
    data_loader = data.RealDataLoader(cache=root_dir / "data/cache")
    price_data = data_loader.get_timeseries(
        "Adj Close", data_path=root_dir / "data/raw_yahoo_data"
    )

    model = "fourier_flow"

    train_config = {
        "train_config": {
            "use_cuda": True,
            "train_seed": 99,
            "fourier_flow_config": {
                "hidden_dim": 256,
                "seq_len": 512,
                "num_layer": 8,
            },
            "dtype": "float32",
            "batch_size": 512,
            "lag": 1,
            "epochs": 100,
            "optim_config": {
                "lr": 0.001,
            },
            "lr_config": {"gamma": 0.995},
        },
        "sample_config": {"n_sample": 2000, "n_burn": 500, "sample_seed": 99},
    }

    cache = root_dir / f"data/cache/train_{model}_{time.time()}"
    cache.mkdir(parents=True, exist_ok=True)

    price_data = price_data.loc[:, ["MSFT", "AMZN"]].iloc[-4000:, :]

    if wandb_off:
        os.environ["WANDB_MODE"] = "disabled"

    generator = fourier_flow_index_generator.FourierFlowIndexGenerator()

    wandb_train.wandb_train(
        index_generator=generator,
        price_data=price_data,
        train_config=train_config,
        cache=cache,
    )


if __name__ == "__main__":
    main(True)
