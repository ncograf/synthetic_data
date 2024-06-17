import boosted_stats
import numpy as np
import pandas as pd


def volatility_clustering(log_returns: pd.DataFrame, max_lag: int):
    if isinstance(log_returns, pd.Series):
        log_returns = log_returns.to_frame()

    log_returns = np.array(log_returns)

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
