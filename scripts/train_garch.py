import os
import time
from pathlib import Path
from typing import Literal

import garch_copula_index_generator
import garch_index_generator
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
    model: Literal["univar_garch", "copula_garch"],
):
    root_dir = Path(__file__).parent.parent

    # get real data
    data_loader = data.RealDataLoader(cache=root_dir / "data/cache")
    price_data = data_loader.get_timeseries(
        "Adj Close", data_path=root_dir / "data/raw_yahoo_data"
    )

    train_config = {
        "train_config": {
            "garch_config": {"q": 2, "p": 3, "dist": "studentt"},
            "train_seed": 10,
        },
        "sample_config": {"n_sample": 2000, "n_burn": 500, "sample_seed": 99},
    }

    cache = root_dir / f"data/cache/train_{model}_{time.time()}"
    cache.mkdir(parents=True, exist_ok=True)

    price_data = price_data.loc[:, ["MSFT", "AMZN", "TSLA", "A"]].iloc[-4000:, :]

    if wandb_off:
        os.environ["WANDB_MODE"] = "disabled"

    if model == "univar_garch":
        generator = garch_index_generator.GarchIndexGenerator()
    elif model == "copula_garch":
        generator = garch_copula_index_generator.GarchCopulaIndexGenerator()
    else:
        raise ValueError(f"The given model {model} is not supported.")

    wandb_train.wandb_train(
        index_generator=generator,
        price_data=price_data,
        train_config=train_config,
        cache=cache,
    )


if __name__ == "__main__":
    main(True, "copula_garch")
