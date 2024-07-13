import json
import os
import random
from datetime import datetime
from pathlib import Path

import arch.univariate as arch_uni
import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import real_data_loader
import torch


def _train_garch(config, price_data: pd.DataFrame | None = None) -> Path:
    """Fit Garch model with the given stock market prices

    For each price a Garch model is fitted and for the sampling
    a copula is fit on top of that.

    To have suitable data, all nan samples are dropped.

    saves models to cache

    Args:
        config (dict): arch configruation as in the arch.univariance.arch_model
            constructor function
        price_data

    Returns:
        Path : path to the results directory
    """

    N_TICKS = 9216

    # load real data
    data_dir = Path(os.environ["DATA_DIR"])
    data_loader = real_data_loader.RealDataLoader(cache=data_dir / "cache")
    price_data = data_loader.get_timeseries(
        "Adj Close", data_path=data_dir / "raw_yahoo_data"
    )

    # all colums in the dataframe must have at least seq_len non_nan elements
    non_nans = np.array(np.sum(~np.isnan(price_data), axis=0))
    price_data = price_data.drop(
        price_data.columns[non_nans <= N_TICKS], axis="columns"
    )

    TIME_FORMAT = "%Y_%m_%d-%H_%M_%S"
    time = datetime.now().strftime(TIME_FORMAT)
    cache = Path(os.environ["RESULT_DIR"]) / f"{config['vol']}_{config['dist']}_{time}"
    cache.mkdir(parents=True, exist_ok=True)

    symbols = price_data.columns

    # store model info data
    model_info = {"config": config, "time": time, "symbols": list(symbols)}
    with Path(cache / "meta_data.json").open("w") as meta_file:
        meta_file.write(json.dumps(model_info, indent=4))

    models_dict = {}

    for sym in symbols:
        sym_price = price_data.loc[:, sym]

        # algorithms only work with dataframes
        if isinstance(sym_price, pd.Series):
            sym_price = sym_price.to_frame()

        # create data
        data = np.array(sym_price)
        returns = ((data[1:] / data[:-1]) - 1) * 100
        return_mask = ~np.isnan(returns).flatten()

        # fit garch models
        model = arch_uni.arch_model(returns[return_mask], **config)
        model_result = model.fit()

        out_dict = {
            "init_params": config,
            "fit_params": model_result.params,
            "fit_score": model_result.loglikelihood,
        }

        models_dict[sym] = out_dict

    torch.save(models_dict, cache / "garch_models.pt")

    return cache


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

    models_dict = torch.load(folder / "garch_models.pt")
    log_returns = []
    for sym in models_dict:
        garch_dict = models_dict[sym]
        model = arch_uni.arch_model(y=None, **garch_dict["init_params"])
        return_simulation = model.simulate(
            garch_dict["fit_params"], nobs=LENGTH, burn=BURN
        ).loc[:, "data"]
        return_simulation = np.asarray(return_simulation) / 100 + 1
        return_simulation[return_simulation <= 0] = 1e-8
        log_returns.append(np.log(return_simulation))

    log_returns = np.stack(log_returns, axis=1)

    return log_returns


@click.command()
@click.option(
    "--mean",
    default="Constant",
    type=click.STRING,
    help="Name of the mean model.  Currently supported options are: 'Constant', 'Zero'",
)
@click.option(
    "--vol",
    default="GARCH",
    type=click.STRING,
    help="Name of the volatility model.  Currently supported options are: 'GARCH', 'ARCH', 'EGARCH', 'FIGARCH', 'APARCH'",
)
@click.option(
    "--p", default=1, type=click.INT, help="Lag order of the symmetric innovation"
)
@click.option(
    "--o", default=0, type=click.INT, help="Lag order of the asymmetric innovation"
)
@click.option(
    "--q",
    default=1,
    type=click.INT,
    help="Lag order of lagged volatility or equivalent",
)
@click.option(
    "--power",
    default=2.0,
    type=click.FLOAT,
    help="Power to use with GARCH and related models",
)
@click.option(
    "--dist",
    default="normal",
    type=click.STRING,
    help=(
        "Name of the error distribution.  Currently supported options are:\n"
        "* Normal: 'normal', 'gaussian'\n"
        "* Students's t: 't', 'studentst'\n"
        "* Skewed Student's t: 'skewstudent', 'skewt'\n"
        "* Generalized Error Distribution: 'ged', 'generalized error'"
    ),
)
def train_garch(mean, vol, p, o, q, power, dist):
    """
    Initialization of common ARCH model specifications

    Parameters
    ----------
    mean : str, optional
        Name of the mean model.  Currently supported options are: 'Constant',
            'Zero'. Defautls to 'Constant'.
    vol : str, optional
        Name of the volatility model.  Currently supported options are:
        'GARCH' (default), 'ARCH', 'EGARCH', 'FIGARCH', 'APARCH'
    p : int , optional
        Lag order of the symmetric innovation
    o : int, optional
        Lag order of the asymmetric innovation
    q : int, optional
        Lag order of lagged volatility or equivalent
    power : float, optional
        Power to use with GARCH and related models
    dist : int, optional
        Name of the error distribution.  Currently supported options are:

            * Normal: 'normal', 'gaussian' (default)
            * Students's t: 't', 'studentst'
            * Skewed Student's t: 'skewstudent', 'skewt'
            * Generalized Error Distribution: 'ged', 'generalized error"

    Returns
    -------
        None : Models are stored in the RESULT_DIR (env variable) directory

    Examples
    --------

    Train a basic GARCH(1,1) with a constant mean can be constructed using only
    the return data

    >>> ~$ train_garch --mean Constant --vol GARCH --p 1 --q 1 --dist normal

    Alternative more complicated inputs might be given

    >>> ~$ train_garch --mean Zero --vol GARCH --p 2 --q 3 --dist skewt

    Notes
    -----
    Input that are not relevant for a particular specification
    are silently ignored.
    """
    _train_garch(
        {"mean": mean, "vol": vol, "p": p, "o": o, "q": q, "power": power, "dist": dist}
    )


if __name__ == "__main__":
    train_garch()
