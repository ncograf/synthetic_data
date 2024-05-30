import os
import time
from pathlib import Path

import real_data_loader as data
import time_gan_index_generator
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

    model = "time_gan"

    train_config = {
        "train_config": {
            "use_cuda": True,
            "train_seed": 99,
            "time_gan_config": {
                "hidden_dim": 24,
                "num_layer": 3,
                "embed_dim": 6,
                "n_stocks": 1,
            },
            "seq_len": 24,
            "dtype": "float32",
            "epochs": 200,
            "batch_size": 128,
            "lag": 1,
            "gamma": 1,
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

    generator = time_gan_index_generator.TimeGanIndexGenerator()

    wandb_train.wandb_train(
        index_generator=generator,
        price_data=price_data,
        train_config=train_config,
        cache=cache,
    )


if __name__ == "__main__":
    main(False)
