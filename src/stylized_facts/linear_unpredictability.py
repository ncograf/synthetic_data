from typing import Any, Dict

import boosted_stats
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import linregress


def linear_unpredictability(log_returns: pd.DataFrame, max_lag: int) -> npt.NDArray:
    """Linear unpredictability

    :math:`Corr(r_t, r_{t+k}) \approx 0, \quad \text{for } k \geq 1`

    where k is chosen sucht that k <= `max_lag`

    Args:
        log_returns (pd.DataFrame): price log returns
        max_lag (int): maximal lag to compute

    Returns:
        npt.NDArray: max_lag x (log_returns.shape[1]) for each stock
    """

    if isinstance(log_returns, pd.Series):
        log_returns = log_returns.to_frame()

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


def linear_unpredictability_statistics(
    log_returns: pd.DataFrame, max_lag: int
) -> Dict[str, Any]:
    lin_upred = linear_unpredictability(log_returns=log_returns, max_lag=max_lag)
    regression = linregress(
        np.linspace(1, max_lag, max_lag, endpoint=True), np.mean(lin_upred, axis=1)
    )
    mean_squared = np.mean(lin_upred**2)

    stats = {
        "regression_r_score": regression.rvalue,
        "regression_p_value": regression.pvalue,
        "regression_slope": regression.slope,
        "regression_intercept": regression.intercept,
        "mean_squared_value": mean_squared,
        "data": lin_upred,
    }

    return stats


lin_upred_axes_setting = {
    "title": "linear unpredictability",
    "ylabel": r"$Corr(r_t, r_{t+k})$",
    "xlabel": "lag k",
    "xscale": "log",
    "yscale": "linear",
    "ylim": (-1, 1),
}
lin_unpred_plot_setting = {
    "alpha": 1,
    "marker": "o",
    "color": "blue",
    "markersize": 1,
    "linestyle": "None",
}
