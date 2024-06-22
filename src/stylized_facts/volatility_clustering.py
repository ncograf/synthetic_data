from typing import Tuple

import boosted_stats
import numpy as np
import numpy.typing as npt
from scipy.stats import linregress


def volatility_clustering(log_returns: npt.ArrayLike, max_lag: int) -> npt.NDArray:
    """Compute Autocorrelation of absolute log returns

    Args:
        log_returns (array_like): log_returns
        max_lag (int): maximum lag

    Returns:
        npt.NDArray: autocorrelation (max_lag x stocks)
    """

    log_returns = np.array(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(
            f"Log Returns have {log_returns.ndim} dimensions must have 1 or 2."
        )

    # make asolute values
    abs_log_returns = np.abs(log_returns)

    # compute the means / var for each stock
    mu = np.nanmean(abs_log_returns, axis=0)
    var = np.nanvar(abs_log_returns, axis=0)

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    centered_abs_log_returns = np.array(abs_log_returns - mu)
    if centered_abs_log_returns.dtype.name == "float32":
        correlation = boosted_stats.lag_prod_mean_float(
            centered_abs_log_returns, max_lag, False
        )
    elif centered_abs_log_returns.dtype.name == "float64":
        correlation = boosted_stats.lag_prod_mean_double(
            centered_abs_log_returns, max_lag, False
        )

    vol_clustering = correlation / var
    return vol_clustering


def _fit_exp(x: npt.NDArray, y: npt.NDArray) -> Tuple[float, float, float]:
    """Fit the powerlaw on the data

    The method fits
    :math:`y = a \exp(b x) \iff \log(y) = a + b \log(x)`
    using a linear regression in the log log space

    Args:
        x (npt.NDArray): x data
        y (npt.NDArray): y data

    Returns:
        Tuple[float, float, float]: a, b and pearson_coefficient
    """

    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    log_x = np.log(x)
    log_y = np.log(y)

    fit = linregress(log_x, log_y)
    goodness = abs(fit.rvalue)
    slope = fit.slope
    intercept = fit.intercept

    return intercept, slope, goodness


def fit_powerlaw_volatility_clustering(
    x: npt.NDArray, y: npt.NDArray
) -> Tuple[float, float, float]:
    """Fit the powerlaw on a subset of the data

    The subset is chosen greedily to maximize the pearson coefficient

    The method fits
    :math:`y = a \exp(b x) \iff \log(y) = a + b \log(x)`
    using a linear regression in the log log space

    Args:
        x (npt.NDArray): x data
        y (npt.NDArray): y data

    Returns:
        Tuple[float, float, float]: a, b and pearson_coefficient
    """

    _, _, c_g = _fit_exp(x, y)
    n_g = 1
    while n_g > c_g:
        # make sure to have at least 500 points
        # note that this is heuristically optimied for 1000 bins
        if 500 > x.size:
            return _fit_exp(x, y)

        n_delta = int(x.size * 0.06)

        n_x, n_y = x[:-n_delta], y[:-n_delta]

        _, _, n_g = _fit_exp(n_x, n_y)

        x, y = n_x, n_y

    return _fit_exp(x, y)


vol_clust_axes_setting = {
    "title": "volatility clustering",
    "ylabel": r"$Corr(|r_t|, |r_{t+k}|)$",
    "xlabel": "lag k",
    "xscale": "log",
    "yscale": "log",
}
vol_clust_plot_setting = {
    "alpha": 1,
    "marker": "o",
    "color": "blue",
    "markersize": 1,
    "linestyle": "None",
}
