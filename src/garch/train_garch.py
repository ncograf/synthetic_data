import json
import os
from datetime import datetime
from pathlib import Path

import arch.univariate as arch_uni
import click
import load_data
import numpy as np
import numpy.typing as npt
import torch


def _train_garch(config) -> Path:
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

    MIN_T = 9216

    # load real data
    price_data = load_data.load_prices("sp500")
    log_returns, symbols = load_data.get_log_returns(price_data, MIN_T)

    TIME_FORMAT = "%Y_%m_%d-%H_%M_%S"
    time = datetime.now().strftime(TIME_FORMAT)
    cache = (
        Path(os.environ["RESULT_DIR"])
        / "garch_runs"
        / f"{config['vol']}_{config['dist']}_{time}"
    )
    cache.mkdir(parents=True, exist_ok=True)

    # store model info data
    model_info = {"config": config, "time": time, "symbols": list(symbols)}
    with Path(cache / "meta_data.json").open("w") as meta_file:
        meta_file.write(json.dumps(model_info, indent=4))

    models_dict = {}

    for i, sym in enumerate(symbols):
        # create data
        data = log_returns[:, i]
        returns = (np.exp(data) - 1) * 100
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

    torch.save(models_dict, cache / "models.pt")

    return cache


def load_garch(file: str | Path) -> dict:
    """Load garch models from memory

    Args:
        file (str | Path): file containing all garch models

    Returns:
        dict : dict with tuples containing models and parameters
    """

    file = Path(file)
    models_dict = torch.load(file)
    symbols = list(models_dict.keys())

    garch_dict = {}
    for sym in symbols:
        model = arch_uni.arch_model(y=None, **(models_dict[sym]["init_params"]))
        params = models_dict[sym]["fit_params"]
        garch_dict[sym] = (model, params)

    return garch_dict


def sample_garch(
    garch_models: dict, n_stocks: int = -1, len: int = 8192
) -> npt.NDArray:
    """Generate data from garch model

    Args:
        garch_models (dict): Dictonary of garch models
        n_stocks (int): number of stocks
        len (int): sampled sequence length

    Returns:
        npt.NDArray: log return simulations
    """

    BURN = 512
    symbols = list(garch_models.keys())

    if n_stocks > 0:
        symbols = np.random.choice(symbols, n_stocks, replace=True)

    log_returns = []
    for sym in symbols:
        model, params = garch_models[sym]
        return_simulation = model.simulate(params, nobs=len, burn=BURN).loc[:, "data"]
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
        None : Models are stored in the RESULT_DIR / garch_runs (env variable) directory

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
