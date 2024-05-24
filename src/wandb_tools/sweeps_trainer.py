import datetime
from pathlib import Path

import chained_filter
import generator_factory
import len_filter
import numpy as np
import pandas as pd
import real_data_loader as data
import time_filter
import wandb_decorator
import wandb_train

import wandb


def wandb_sweep_run():
    """Train run of index generator with wandb logging and integrated evaluation
    inside a sweep (i.e. the configuration must be given in wandb.config)

    Args:
        index_generator (base_index_generator.BaseIndexGenerator): generator to be fitted
        price_data (pd.DataFrame): price data to fit on
        train_config (Dict[str, Any]): training configuration according to the generator
            including `sample_config` key
            including `train_config` key
        cache (str | Path): local cache directory
    """

    @wandb_decorator.wandb_run
    def _train():
        config = wandb.config

        # get root directory
        root_dir = Path(__file__).parent.parent.parent

        # read outer config
        model_name = config["model"]

        train_data = config["train_data"]
        symbols = train_data["symbols"]
        first_data_dt = datetime.datetime.strptime(train_data["first_date"], "%Y-%m-%d")
        first_date = pd.Timestamp(
            year=first_data_dt.year, month=first_data_dt.month, day=first_data_dt.day
        )
        min_len = train_data["min_len"]
        nan_rows = train_data["nan_allowed"]

        generator = generator_factory.generator_factory(model_name)

        # get real data
        data_loader = data.RealDataLoader(cache=root_dir / "data/cache")
        price_data = data_loader.get_timeseries(
            "Adj Close", data_path=root_dir / "data/raw_yahoo_data"
        )
        # filter training data
        price_data = price_data.loc[:, symbols]
        filter_list = [len_filter.LenFilter(min_len)]
        ch_filter = chained_filter.ChainedFilter(
            filter_chain=filter_list,
            time_filter=time_filter.TimeFilter(first_date=first_date),
        )
        ch_filter.fit_filter(price_data)
        ch_filter.apply_filter(price_data)

        if not nan_rows:
            # remove all nan rows
            mask = np.any(np.isnan(np.array(price_data)), axis=1)
            price_data.drop(index=price_data.index[mask], inplace=True)

        cache = Path(config["cache"])
        cache.mkdir(parents=True, exist_ok=True)

        metadata = wandb_train.wandb_fit(
            generator, price_data, config["train_config"], cache
        )

        wandb_train.wandb_eval(
            generator,
            sample_config=config["sample_config"],
            metadata=metadata,
            cache=cache,
            real_data=price_data,  # note that the decorator needs this extra argument
        )

    # start training
    _train()


if __name__ == "__main__":
    wandb_sweep_run()
