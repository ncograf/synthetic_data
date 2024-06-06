import os
import time
from pathlib import Path

import fin_gan_index_generator
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

    model = "fin_gan"

    train_config = {
        "train_config": {
            "use_cuda": True,
            "train_seed": 99,
            "fingan_config": {
                "input_dim": 100,
                "arch": "MLP",
                "layers": [128, 2048],
                "drop_out": 0,
                "norm": "none",
                "activation": "tanh",
            },
            "dtype": "float32",
            "seq_len": 8192,
            "batch_size": 16,
            "epochs": 1000,
            "optim_gen_config": {
                "lr": 2e-4,
                "betas": (0.5, 0.999),
            },
            "optim_disc_config": {
                "lr": 1e-5,
                "betas": (0.2, 0.999),
            },
            "lr_config": {
                "gamma": 0.999,
            },
        },
        "sample_config": {"n_sample": 8192, "n_burn": 0, "sample_seed": 99},
    }

    cache = root_dir / f"data/cache/train_{model}_{time.time()}"
    cache.mkdir(parents=True, exist_ok=True)

    price_data = price_data.loc[:, ["MSFT", "AMZN"]].iloc[-9000:, :]

    if wandb_off:
        os.environ["WANDB_MODE"] = "disabled"

    generator = fin_gan_index_generator.FinGanIndexGenerator()

    wandb_train.wandb_train(
        index_generator=generator,
        price_data=price_data,
        train_config=train_config,
        cache=cache,
    )


if __name__ == "__main__":
    main(False)
