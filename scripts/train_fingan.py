import os
from datetime import datetime
from pathlib import Path

import fin_gan_trainer
import real_data_loader as data

import wandb


# @click.command()
# @click.option(
#    "--wandb-log",
#    is_flag=True,
#    help="Turn on wandb logging.",
# )
def main(
    wandb_log: bool,
):
    # load real data
    root_dir = Path(__file__).parent.parent
    data_loader = data.RealDataLoader(cache=root_dir / "data/cache")
    price_data = data_loader.get_timeseries(
        "Adj Close", data_path=root_dir / "data/raw_yahoo_data"
    )

    # setup cache for the train run
    TIME_FORMAT = "%Y_%m_%d-%H_%M_%S"
    time = datetime.now().strftime(TIME_FORMAT)
    cache = root_dir / f"data/cache/train_FinGanTakahashi_{time}"
    cache.mkdir(parents=True, exist_ok=True)

    # decide whether or not to log to
    if not wandb_log:
        os.environ["WANDB_MODE"] = "disabled"
    else:
        # check settings (the envvars should be available in the poetry env (if configured correctly))
        if os.getenv("WANDB_API_KEY") is None:
            raise EnvironmentError("WANDB_API_KEY is not set!")
        if os.getenv("WANDB_ENTITY") is None:
            raise EnvironmentError("WANDB_ENTITY is not set!")
        if os.getenv("WANDB_PROJECT") is None:
            raise EnvironmentError("WANDB_PROJECT is not set!")

    # define training config and train model
    train_config = {
        "seq_len": 512,
        "train_seed": 99,
        "dtype": "float32",
        "epochs": 1000,
        "batch_size": 1,
        "optim_gen_config": {
            "lr": 2e-4,
            "betas": (0.5, 0.999),
        },
        "optim_disc_config": {
            "lr": 1e-5,
            "betas": (0.1, 0.999),
        },
        "lr_config": {
            "gamma": 0.999,
        },
    }

    # if wandb is enabled this will log the run
    with wandb.init():
        trainer = fin_gan_trainer.FinGanTrainer()
        trainer.fit(price_data=price_data, config=train_config, cache=cache)


if __name__ == "__main__":
    main(True)
