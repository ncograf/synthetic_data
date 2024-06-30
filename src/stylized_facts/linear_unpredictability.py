from typing import Any, Dict, List

import boosted_stats
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import linregress


def linear_unpredictability(log_returns: npt.ArrayLike, max_lag: int) -> npt.NDArray:
    """Linear unpredictability

    :math:`Corr(r_t, r_{t+k}) \approx 0, \quad \text{for } k \geq 1`

    where k is chosen sucht that k <= `max_lag`

    Args:
        log_returns (pd.DataFrame): price log returns
        max_lag (int): maximal lag to compute

    Returns:
        npt.NDArray: max_lag x (log_returns.shape[1]) for each stock
    """

    log_returns = np.array(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(f"Log Price has {log_returns.ndim} dimensions.")

    log_returns = np.array(log_returns)

    # compute the means / var for each stock
    mu = np.nanmean(log_returns, axis=0)
    var = np.nanvar(log_returns, axis=0)

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    centered_log_returns = np.array(log_returns - mu)
    if centered_log_returns.dtype.name == "float32":
        correlation = boosted_stats.lag_prod_mean_float(
            centered_log_returns, max_lag, False
        )
    elif centered_log_returns.dtype.name == "float64":
        correlation = boosted_stats.lag_prod_mean_double(
            centered_log_returns, max_lag, False
        )

    lin_unpred = correlation / var
    return lin_unpred


def linear_unpredictability_stats(
    log_returns: npt.ArrayLike, max_lag: int
) -> Dict[str, Any]:
    """Linear unpredictabiliity statistics

    Args:
        log_returns (npt.ArrayLike): log returns
        max_lag (int): maximal lag

    Returns:
        Dict[str, Any]: result dictonary with keys:
            corr: pearson correlation coefficient of linear fit
            p_value: p value of linear fit
            beta: slope of linear fit
            alpha: intercept of linear fit
            mse: mean squared (error) of linear fit with assumption y = 0
            mse_std: empirical standard deviation over all stocks
            data: autocorrelation data (max_lag x stocks)

    """

    lin_upred = linear_unpredictability(log_returns=log_returns, max_lag=max_lag)
    regression = linregress(
        np.linspace(1, max_lag, max_lag, endpoint=True), np.mean(lin_upred, axis=1)
    )
    mean_squared = np.mean(lin_upred**2)
    mse_stockwise = np.mean(lin_upred**2, axis=0)
    mse_std = np.std(mse_stockwise)

    stats = {
        "corr": regression.rvalue,
        "p_value": regression.pvalue,
        "beta": regression.slope,
        "alpha": regression.intercept,
        "mse": mean_squared,
        "mse_std": mse_std,
        "data": lin_upred,
    }

    return stats


def visualize_stat(
    plot: plt.Axes, log_returns: npt.NDArray, name: str, print_stats: List[str]
):
    data = linear_unpredictability_stats(log_returns=log_returns, max_lag=1000)
    ac_data = np.mean(data["data"], axis=1)
    mse, mse_std = (data["mse"], data["mse_std"])
    data_label = (
        f"$\\text{{MSE}} =${mse:.2e}, $\\sigma_{{\\text{{MSE}}}}=${mse_std:.2e}"
    )

    for key in print_stats:
        print(f"{name} linear unpred {key} {data[key]}")

    lin_upred_axes_setting = {
        "title": f"{name} linear unpredictability",
        "ylabel": r"$Corr(r_t, r_{t+k})$",
        "xlabel": "lag k",
        "xscale": "log",
        "yscale": "linear",
        "ylim": (-1, 1),
    }

    lin_unpred_plot_setting = {
        "alpha": 1,
        "marker": "o",
        "color": "cornflowerblue",
        "markersize": 2,
        "linestyle": "None",
    }

    plot.set(**lin_upred_axes_setting)
    plot.plot(ac_data, **lin_unpred_plot_setting, label=data_label)
    plot.legend()
