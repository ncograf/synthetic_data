import random
from datetime import datetime
from pathlib import Path

import garch
import numpy as np
import numpy.typing as npt
import pandas as pd
import real_data_loader
import torch
from arch.univariate import GARCH, ConstantMean, Normal, StudentsT


def train_garch():
    """Fit Garch model with the given stock market prices

    For each price a Garch model is fitted and for the sampling
    a copula is fit on top of that.

    To have suitable data, all nan samples are dropped.

    saves models to cache
    """

    N_TICKS = 9216

    # setup cache for the train run
    root_dir = Path(__file__).parent.parent.parent

    # load real data
    data_loader = real_data_loader.RealDataLoader(cache=root_dir / "data/cache")
    price_data = data_loader.get_timeseries(
        "Adj Close", data_path=root_dir / "data/raw_yahoo_data"
    )

    # all colums in the dataframe must have at least seq_len non_nan elements
    non_nans = np.array(np.sum(~np.isnan(price_data), axis=0))
    price_data = price_data.drop(
        price_data.columns[non_nans <= N_TICKS], axis="columns"
    )

    config = {
        "dist": "normal",
        "p": 3,
        "q": 3,
    }
    dist, p, q = config["dist"], config["p"], config["q"]

    TIME_FORMAT = "%Y_%m_%d-%H_%M_%S"
    time = datetime.now().strftime(TIME_FORMAT)
    cache = root_dir / f"data/cache/Garch_{dist}_{p}_{q}_{time}"
    cache.mkdir(parents=True, exist_ok=True)

    symbols = price_data.columns

    for sym in symbols:
        sym_price = price_data.loc[:, sym]

        if dist == "normal":
            distribution = Normal()  # for univariate garch
        elif dist.lower() == "studentt":
            distribution = StudentsT()  # for univariate garch
        else:
            ValueError("The chosen distribution is not supported")

        # algorithms only work with dataframes
        if isinstance(sym_price, pd.Series):
            sym_price = sym_price.to_frame()

        # create data
        data = np.array(sym_price)
        returns = ((data[1:] / data[:-1]) - 1) * 100
        return_mask = ~np.isnan(returns).flatten()

        # fit garch models
        model = ConstantMean(returns[return_mask])
        model.volatility = GARCH(p=p, q=q)
        model.distribution = distribution
        fitted = model.fit(disp="off")

        scaled_returns = returns[return_mask]
        scaled_returns = scaled_returns / fitted.conditional_volatility

        # store fitted garch model in minimal format
        garch_config = {}
        garch_config["alpha"] = [fitted.params[f"alpha[{i}]"] for i in range(1, p + 1)]
        garch_config["beta"] = [fitted.params[f"beta[{i}]"] for i in range(1, q + 1)]
        garch_config["mu"] = fitted.params["mu"]
        garch_config["omega"] = fitted.params["omega"]

        if dist.lower() == "studentt":
            garch_config["nu"] = fitted.params["nu"]

        garch_config["dist"] = dist

        out_dict = {
            "garch": garch_config,
            "fit_score": fitted.loglikelihood,
        }

        torch.save(out_dict, cache / f"{sym}.pt")


def sample_garch(folder: str | Path, seed: int = 99) -> npt.NDArray:
    """Generate data from garch model

    Args:
        folder (str | Path): path to the garch
        seed (int, optional): manual seed. Defaults to 99.

    Returns:
        npt.NDArray: log return simulations
    """

    LENGTH = 8192
    BURN = 512

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    folder = Path(folder)

    log_returns = []
    for file in folder.iterdir():
        if file.suffix == ".pt":
            garch_dict = torch.load(file)
            model = garch.GarchModel(garch_dict=garch_dict["garch"])

            return_simulation = model.sample(length=LENGTH, burn=BURN, dtype=np.float64)
            log_returns.append(np.log(return_simulation))

    log_returns = np.stack(log_returns, axis=1)

    return log_returns


if __name__ == "__main__":
    train_garch()
