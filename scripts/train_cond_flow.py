import os
import time
from pathlib import Path

import conditional_flow_index_generator
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

    model = "cond_flow"

    train_config = {
        "train_config": {
            "use_cuda": True,
            "train_seed": 99,
            "cond_flow_config": {
                "hidden_dim": 64,
                "dim": 2,
                "conditional_dim": 64,
                "n_layer": 8,
                "num_model_layer": 3,
                "drop_out": 0,
                "activation": "sigmoid",
                "norm": "layer",
                "dtype": "float32",
            },
            "seq_len": 64,
            "batch_size": 512,
            "lag": 1,
            "epochs": 10,
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

    generator = conditional_flow_index_generator.ConditionalFlowIndexGenerator()

    wandb_train.wandb_train(
        index_generator=generator,
        price_data=price_data,
        train_config=train_config,
        cache=cache,
    )


if __name__ == "__main__":
    main(wandb_off=False)
